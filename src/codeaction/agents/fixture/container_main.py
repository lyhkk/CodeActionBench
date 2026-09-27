#!/usr/bin/env python3
"""Offline vendor-agent lifecycle agent used only to test the isolated container harness."""
import json
import hashlib
import os
import socket
import sys
import time


HOST = os.environ.get("CODEACTION_SIM_HOST", "gateway")
PORT = int(os.environ.get("CODEACTION_SIM_PORT", "8765"))
CONNECT_TIMEOUT_S = 180.0
CALL_TIMEOUT_S = 610.0
SURFACE_MARK = "[raw-mcp-surface] "


def _sha256_json(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class Client:
    def __init__(self):
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.socket = socket.create_connection((HOST, PORT), timeout=2.0)
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.1)
        else:
            raise OSError(f"fixture could not connect to gateway: {last_error}")
        self.socket.settimeout(CALL_TIMEOUT_S)
        self.stream = self.socket.makefile("rwb")
        self.next_id = 1

    def call(self, method, params=None):
        request_id = self.next_id
        self.next_id += 1
        request = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        self.stream.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
        self.stream.flush()
        while True:
            raw = self.stream.readline()
            if not raw:
                raise EOFError("gateway closed before fixture received a response")
            response = json.loads(raw)
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"{method} failed")
            return response["result"]

    def notify(self, method, params=None):
        value = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            value["params"] = params
        self.stream.write((json.dumps(value, separators=(",", ":")) + "\n").encode())
        self.stream.flush()

    def tool(self, name, arguments=None):
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise RuntimeError(f"fixture tool {name} failed")
        return result.get("content", [])

    def close(self):
        try:
            self.stream.close()
        finally:
            self.socket.close()


def main():
    client = Client()
    try:
        client.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                    "clientInfo": {"name": "codeaction-fixture", "version": "1"}})
        client.notify("notifications/initialized")
        tools = client.call("tools/list").get("tools", [])
        names = [tool["name"] for tool in tools]
        required = {"get_robot_state", "done"}
        if not required.issubset(set(names)):
            raise RuntimeError("fixture tool surface is incomplete")
        if "bash_exec" in names:
            raise RuntimeError("vendor-mcp-direct unexpectedly exposed bash_exec")
        expected_names = json.loads(os.environ.get(
            "CODEACTION_EXPECTED_ORDERED_NAMES", "[]"))
        expected_hash = os.environ.get("CODEACTION_EXPECTED_DELIVERED_SHA256", "")
        observed_hash = _sha256_json(tools)
        if names != expected_names or observed_hash != expected_hash:
            raise RuntimeError("vendor-mcp-direct raw tool surface mismatch")
        print(SURFACE_MARK + json.dumps({
            "schema_version": "1.0",
            "ordered_names": names,
            "observed_delivered_sha256": observed_hash,
            "expected_delivered_sha256": expected_hash,
            "healthy": True,
        }, sort_keys=True, separators=(",", ":")), flush=True)
        client.tool("get_robot_state", {"arms": ["left", "right"]})
        client.tool("done", {"report": "offline vendor-agent direct fixture completed",
                             "success_claim": False})
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
