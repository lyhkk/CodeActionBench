#!/usr/bin/env python3
"""Expose one stdio MCP child over one internal-network TCP connection.

This is a byte-transparent transport seam. It never parses JSON-RPC and writes diagnostics only to
stderr; fd1 belongs exclusively to a child response forwarded through the connected socket.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading

CHUNK_BYTES = 64 * 1024


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _socket_to_fd(sock, fd):
    try:
        while True:
            data = sock.recv(CHUNK_BYTES)
            if not data:
                break
            _write_all(fd, data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _fd_to_socket(fd, sock):
    try:
        while True:
            data = os.read(fd, CHUNK_BYTES)
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
        try:
            os.close(fd)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-host", default="0.0.0.0")  # noqa: S104 - internal Compose network
    ap.add_argument("--listen-port", type=int, default=8766)
    ap.add_argument("--accept-timeout", type=float, default=120.0)
    ap.add_argument("--child-exit-timeout", type=float, default=180.0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        ap.error("a child command is required after --")
    if not 1 <= args.listen_port <= 65535:
        ap.error("--listen-port must be in 1..65535")
    if args.accept_timeout <= 0 or args.child_exit_timeout <= 0:
        ap.error("timeouts must be positive")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((args.listen_host, args.listen_port))
        listener.listen(1)
        listener.settimeout(args.accept_timeout)
        try:
            conn, _ = listener.accept()
        except socket.timeout:
            print("mcp_tcp_server: accept timeout", file=sys.stderr)
            return 2
    finally:
        listener.close()

    with conn:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        child_stdin_fd = os.dup(proc.stdin.fileno())
        child_stdout_fd = os.dup(proc.stdout.fileno())
        proc.stdin.close()
        proc.stdout.close()
        inbound = threading.Thread(target=_socket_to_fd,
                                   args=(conn, child_stdin_fd), daemon=True)
        outbound = threading.Thread(target=_fd_to_socket,
                                    args=(child_stdout_fd, conn), daemon=True)
        inbound.start()
        outbound.start()
        inbound.join()
        try:
            rc = proc.wait(timeout=args.child_exit_timeout)
        except subprocess.TimeoutExpired:
            print("mcp_tcp_server: child exit timeout", file=sys.stderr)
            proc.kill()
            proc.wait()
            rc = 124
        outbound.join(timeout=5)
        return rc


if __name__ == "__main__":
    sys.exit(main())
