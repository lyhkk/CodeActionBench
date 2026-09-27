#!/usr/bin/env python3
"""Map local MCP stdio to the sim service's internal-network TCP stream."""
import argparse
import os
import socket
import sys
import threading
import time

CHUNK_BYTES = 64 * 1024


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _stdin_to_socket(sock):
    try:
        while True:
            data = os.read(0, CHUNK_BYTES)
            if not data:
                break
            sock.sendall(data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _socket_to_stdout(sock):
    try:
        while True:
            data = sock.recv(CHUNK_BYTES)
            if not data:
                break
            _write_all(1, data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CODEACTION_SIM_HOST", "sim"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CODEACTION_SIM_PORT", "8766")))
    ap.add_argument("--connect-timeout", type=float, default=120.0)
    ap.add_argument("--retry-interval", type=float, default=0.1)
    args = ap.parse_args()
    if not 1 <= args.port <= 65535:
        ap.error("--port must be in 1..65535")
    if args.connect_timeout <= 0 or args.retry_interval <= 0:
        ap.error("timeouts must be positive")
    deadline = time.monotonic() + args.connect_timeout
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"mcp_tcp_client: connect failed: {last_error}", file=sys.stderr)
            return 2
        try:
            sock = socket.create_connection((args.host, args.port), timeout=min(2.0, remaining))
            break
        except OSError as exc:
            last_error = exc
            time.sleep(min(args.retry_interval, max(0.0, remaining)))
    sock.settimeout(None)
    with sock:
        inbound = threading.Thread(target=_stdin_to_socket, args=(sock,), daemon=True)
        outbound = threading.Thread(target=_socket_to_stdout, args=(sock,), daemon=True)
        inbound.start()
        outbound.start()
        outbound.join()
        # Server EOF is authoritative. Claude may keep our stdin open after a sim boot failure;
        # returning here lets the MCP client process close fd0 and the socket instead of deadlocking.
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        inbound.join(timeout=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
