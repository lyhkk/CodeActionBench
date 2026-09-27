"""Bounded Unix-socket transport for public benchmark tool calls from scratch containers.

This module is deliberately unaware of RoboTwin, MCP, the verifier, and ground truth.  A gateway
supplies the exact public tool allowlist and a dispatch callback; one JSON-line request maps to one
callback invocation.
"""
from __future__ import annotations

import json
import os
import socket
import stat
import threading
from pathlib import Path


MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _receive_line(conn: socket.socket, limit: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = conn.recv(min(65536, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ValueError(f"request exceeds {limit} bytes")
        if b"\n" in chunk:
            break
    raw = b"".join(chunks)
    line, separator, trailing = raw.partition(b"\n")
    if not separator:
        raise ValueError("request must end with a newline")
    if trailing:
        raise ValueError("only one request is allowed per connection")
    return line


class PublicToolSocketServer:
    """Threaded one-request-per-connection server with a mutable, fail-closed allowlist."""

    def __init__(self, socket_path, allowed_tools, dispatch,
                 *, max_request_bytes=MAX_REQUEST_BYTES,
                 max_response_bytes=MAX_RESPONSE_BYTES, request_timeout_s=5.0):
        self.path = Path(socket_path)
        self._allowed = frozenset(allowed_tools)
        self._dispatch = dispatch
        self._max_request = int(max_request_bytes)
        self._max_response = int(max_response_bytes)
        self._request_timeout = float(request_timeout_s)
        self._listener = None
        self._thread = None
        self._stopping = threading.Event()
        self._allowed_lock = threading.Lock()

    def set_allowed_tools(self, names):
        with self._allowed_lock:
            self._allowed = frozenset(str(name) for name in names)

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            mode = self.path.lstat().st_mode
            if not stat.S_ISSOCK(mode):
                raise ValueError("public-tool socket path exists and is not a socket")
            self.path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.path))
        os.chmod(self.path, 0o666)
        listener.listen(8)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, name="public-tool-socket", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stopping.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self):
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            try:
                conn.settimeout(self._request_timeout)
                line = _receive_line(conn, self._max_request)
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict) or set(request) != {"tool", "arguments"}:
                    raise ValueError("request must contain exactly tool and arguments")
                tool, arguments = request["tool"], request["arguments"]
                if not isinstance(tool, str) or not isinstance(arguments, dict):
                    raise ValueError("tool must be a string and arguments must be an object")
                with self._allowed_lock:
                    allowed = tool in self._allowed
                if not allowed:
                    raise ValueError(f"tool is not public: {tool!r}")
                response = {"ok": True, "result": self._dispatch(tool, arguments)}
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                response = {"ok": False, "error": str(exc)}
            except socket.timeout:
                response = {"ok": False, "error": "request timed out"}
            except OSError:
                response = {"ok": False, "error": "request transport failed"}
            except Exception as exc:
                response = {"ok": False, "error": f"tool dispatch failed: {type(exc).__name__}"}
            encoded = _json_bytes(response)
            if len(encoded) > self._max_response:
                encoded = _json_bytes({"ok": False,
                                       "error": f"response exceeds {self._max_response} bytes"})
            try:
                conn.sendall(encoded)
            except OSError:
                pass


def call_public_tool(socket_path, tool, arguments, *, timeout_s=30.0,
                     max_response_bytes=MAX_RESPONSE_BYTES):
    """Call one public tool and return its result, raising ValueError on bounded protocol errors."""
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        raise ValueError("tool must be a string and arguments must be an object")
    request = _json_bytes({"tool": tool, "arguments": arguments})
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(float(timeout_s))
        conn.connect(str(socket_path))
        conn.sendall(request)
        conn.shutdown(socket.SHUT_WR)
        raw = bytearray()
        while True:
            chunk = conn.recv(min(65536, max_response_bytes + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > max_response_bytes:
                raise ValueError(f"response exceeds {max_response_bytes} bytes")
    try:
        response = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("public-tool server returned invalid JSON") from exc
    if not isinstance(response, dict) or response.get("ok") is not True:
        error = response.get("error", "unknown public-tool error") if isinstance(response, dict) \
            else "invalid public-tool response"
        raise ValueError(str(error))
    return response.get("result")
