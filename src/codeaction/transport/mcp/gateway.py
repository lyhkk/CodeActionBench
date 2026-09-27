#!/usr/bin/env python3
"""JSON-RPC gateway serializing native MCP and scratch public-tool calls onto one sim stream."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from codeaction.runtime.public_tool_socket import PublicToolSocketServer
from codeaction.contracts.identity import sha256_json
from codeaction.interface.interface_extras import BASH_EXEC_DEFINITION, validate_bash_exec_result


MAX_LINE = 20 * 1024 * 1024
BASH_TOOL = BASH_EXEC_DEFINITION
PROFILE_REFERENCE = "reference-mcp"
PROFILE_REFERENCE_CODE_FIRST = "reference-code-first"
PROFILE_DIRECT = "vendor-mcp-direct"
PROFILE_GATEWAY = "vendor-mcp-gateway"
ATTESTATION_MARK = "[gateway-attestation] "


def _encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _read_line(stream):
    raw = stream.readline(MAX_LINE + 1)
    if len(raw) > MAX_LINE:
        raise ValueError("JSON-RPC line exceeds gateway limit")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("JSON-RPC message is not newline terminated")
    return raw


def _reference_delivered_tools(raw_tools):
    """Mirror the reference container's mechanical MCP→native function-tool transform."""
    out = []
    for item in raw_tools:
        if not isinstance(item, dict) or not isinstance(item.get("inputSchema"), dict):
            raise ValueError("reference MCP tool definition is invalid")
        out.append({
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item["inputSchema"],
            },
        })
    return out


def _connect(host, port, timeout_s):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            conn = socket.create_connection((host, port), timeout=min(2, timeout_s))
            conn.settimeout(None)
            return conn
        except OSError as exc:
            last = exc
            time.sleep(0.1)
    raise OSError(f"connection to {host}:{port} failed") from last


def call_launcher(host, port, script, request_id, timeout_s=130.0):
    request = _encode({"request_id": request_id, "script": script})
    with _connect(host, port, 5) as conn:
        conn.settimeout(timeout_s)
        conn.sendall(request)
        conn.shutdown(socket.SHUT_WR)
        stream = conn.makefile("rb")
        raw = _read_line(stream)
    value = json.loads(raw.decode())
    if not isinstance(value, dict):
        raise ValueError("scratch launcher returned a non-object")
    return value


@dataclass
class Pending:
    route: str
    original_id: object = None
    method: str = ""
    event: threading.Event = field(default_factory=threading.Event)
    response: dict | None = None


class GatewaySession:
    def __init__(self, agent_conn, sim_conn, *, launcher_host, launcher_port, public_socket,
                 tool_timeout_s=610.0, interface_profile=PROFILE_DIRECT,
                 expected_delivered_sha256=None):
        self.agent = agent_conn
        self.sim = sim_conn
        self.launcher_host = launcher_host
        self.launcher_port = launcher_port
        self.tool_timeout = tool_timeout_s
        self._agent_write = threading.Lock()
        self._sim_write = threading.Lock()
        self._bash_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._closed = threading.Event()
        if interface_profile not in (PROFILE_REFERENCE, PROFILE_REFERENCE_CODE_FIRST,
                                     PROFILE_DIRECT, PROFILE_GATEWAY):
            raise ValueError(f"unsupported gateway interface profile {interface_profile!r}")
        self.interface_profile = interface_profile
        self.expected_delivered_sha256 = expected_delivered_sha256
        self.attestation = None
        self.public = (
            PublicToolSocketServer(public_socket, set(), self._public_dispatch).start()
            if interface_profile == PROFILE_GATEWAY else None)

    def _write(self, conn, lock, value):
        with lock:
            conn.sendall(_encode(value))

    def _new_id(self):
        with self._pending_lock:
            value = f"gateway-{self._next_id}"
            self._next_id += 1
            return value

    def _send_sim_request(self, method, params, pending):
        request_id = self._new_id()
        with self._pending_lock:
            self._pending[request_id] = pending
        try:
            self._write(self.sim, self._sim_write,
                        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise

    def _public_dispatch(self, name, arguments):
        pending = Pending("public", method="tools/call")
        self._send_sim_request("tools/call", {"name": name, "arguments": arguments}, pending)
        if not pending.event.wait(self.tool_timeout):
            raise TimeoutError("sim tool call timed out")
        response = pending.response or {}
        if "error" in response:
            raise ValueError("sim rejected public tool call")
        return response.get("result")

    def _sim_reader(self):
        stream = self.sim.makefile("rb")
        try:
            while True:
                raw = _read_line(stream)
                if not raw:
                    return
                message = json.loads(raw.decode())
                response_id = message.get("id") if isinstance(message, dict) else None
                with self._pending_lock:
                    pending = self._pending.pop(response_id, None)
                if pending is None:
                    self._write(self.agent, self._agent_write, message)
                    continue
                if pending.method == "tools/list" and isinstance(message.get("result"), dict):
                    tools = message["result"].get("tools")
                    if isinstance(tools, list):
                        names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
                        if self.interface_profile == PROFILE_GATEWAY:
                            self.public.set_allowed_tools(
                                name for name in names if isinstance(name, str))
                            if "bash_exec" in names:
                                raise ValueError("episode server exposed undeclared bash_exec")
                            tools.append(BASH_TOOL)
                        if self.interface_profile in (PROFILE_REFERENCE,
                                                      PROFILE_REFERENCE_CODE_FIRST):
                            observed = sha256_json(_reference_delivered_tools(tools))
                        else:
                            observed = sha256_json(tools)
                        healthy = (
                            self.expected_delivered_sha256 is None
                            or observed == self.expected_delivered_sha256)
                        self.attestation = {
                            "schema_version": "0.1",
                            "interface_profile": self.interface_profile,
                            "expected_delivered_sha256": self.expected_delivered_sha256,
                            "observed_delivered_sha256": observed,
                            "healthy": healthy,
                        }
                        print(ATTESTATION_MARK + json.dumps(
                            self.attestation, sort_keys=True, separators=(",", ":")),
                            file=sys.stderr, flush=True)
                        if not healthy:
                            raise ValueError("gateway delivered tool surface hash mismatch")
                if pending.route == "agent":
                    message["id"] = pending.original_id
                    self._write(self.agent, self._agent_write, message)
                else:
                    pending.response = message
                    pending.event.set()
        finally:
            self._closed.set()
            with self._pending_lock:
                pending_values = list(self._pending.values())
                self._pending.clear()
            for pending in pending_values:
                pending.event.set()

    def _handle_bash(self, message):
        request_id = message.get("id")
        try:
            params = message.get("params") or {}
            arguments = params.get("arguments") or {}
            if set(arguments) != {"script"} or not isinstance(arguments.get("script"), str):
                raise ValueError("bash_exec requires exactly one string field: script")
            with self._bash_lock:
                audit_id = uuid.uuid4().hex
                print("[filesystem-audit-request] " + json.dumps(
                    {"request_id": audit_id}, separators=(",", ":")), flush=True)
                payload = call_launcher(self.launcher_host, self.launcher_port,
                                        arguments["script"], audit_id)
                payload.pop("_audit", None)
                payload.pop("request_id", None)
                validate_bash_exec_result(payload)
            result = {"content": [{"type": "text", "text": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"))}],
                      "isError": payload.get("ok") is not True}
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            payload = {"ok": False, "error": f"bash_exec failed: {type(exc).__name__}"}
            validate_bash_exec_result(payload)
            response = {"jsonrpc": "2.0", "id": request_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(payload)}],
                                   "isError": True}}
        try:
            self._write(self.agent, self._agent_write, response)
        except OSError:
            pass

    def run(self):
        sim_reader = threading.Thread(target=self._sim_reader, daemon=True)
        sim_reader.start()
        stream = self.agent.makefile("rb")
        try:
            while not self._closed.is_set():
                raw = _read_line(stream)
                if not raw:
                    return
                message = json.loads(raw.decode())
                if not isinstance(message, dict):
                    raise ValueError("JSON-RPC message must be an object")
                method = message.get("method")
                request_id = message.get("id")
                if self.interface_profile == PROFILE_GATEWAY \
                        and method == "tools/call" and request_id is not None \
                        and (message.get("params") or {}).get("name") == "bash_exec":
                    threading.Thread(target=self._handle_bash, args=(message,), daemon=True).start()
                elif method is not None and request_id is not None:
                    pending = Pending("agent", original_id=request_id, method=method)
                    self._send_sim_request(method, message.get("params"), pending)
                else:
                    self._write(self.sim, self._sim_write, message)
        finally:
            if self.public is not None:
                self.public.stop()
            for conn in (self.agent, self.sim):
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--sim-host", default="sim")
    parser.add_argument("--sim-port", type=int, default=8766)
    parser.add_argument("--launcher-host", default="scratch-launcher")
    parser.add_argument("--launcher-port", type=int, default=8767)
    parser.add_argument("--public-socket", default="/run/codeaction-public/public-tools.sock")
    parser.add_argument("--connect-timeout", type=float, default=180)
    parser.add_argument("--interface-profile",
                        choices=(PROFILE_REFERENCE, PROFILE_REFERENCE_CODE_FIRST,
                                 PROFILE_DIRECT, PROFILE_GATEWAY),
                        default=os.environ.get(
                            "CODEACTION_INTERFACE_PROFILE", PROFILE_DIRECT))
    parser.add_argument("--expected-delivered-sha256",
                        default=os.environ.get("CODEACTION_EXPECTED_DELIVERED_SHA256") or None)
    args = parser.parse_args()
    with socket.create_server((args.listen_host, args.listen_port), family=socket.AF_INET,
                              backlog=1) as listener:
        print(f"[mcp-gateway] listening on {args.listen_host}:{args.listen_port}", file=sys.stderr)
        agent, _ = listener.accept()
        sim = _connect(args.sim_host, args.sim_port, args.connect_timeout)
        GatewaySession(agent, sim, launcher_host=args.launcher_host,
                       launcher_port=args.launcher_port, public_socket=args.public_socket,
                       interface_profile=args.interface_profile,
                       expected_delivered_sha256=args.expected_delivered_sha256).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
