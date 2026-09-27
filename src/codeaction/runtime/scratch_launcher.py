#!/usr/bin/env python3
"""Fixed-profile scratch-container launcher; the request controls only the shell script text."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import uuid


MAX_REQUEST_BYTES = 72 * 1024
MAX_SCRIPT_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_AUDIT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
_SAFE_VOLUME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_IMAGE = re.compile(r"^(?:sha256:[0-9a-f]{64}|[^\s@]+@sha256:[0-9a-f]{64})$")
_SAFE_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_QUOTA_MARK = re.compile(rb"^CODEACTION_WORKSPACE_BYTES=([0-9]+)$", re.MULTILINE)
_SPECIAL_FILE_MARK = re.compile(rb"^CODEACTION_WORKSPACE_SPECIAL_FILES=([0-9]+)$", re.MULTILINE)
_SETID_MARK = re.compile(rb"^CODEACTION_WORKSPACE_SETID_FILES=([0-9]+)$", re.MULTILINE)
_QUOTED_PATH = re.compile(r'"(/[^"\\]*(?:\\.[^"\\]*)*)"')
_FD_PATH = re.compile(r"<[0-9A-Za-z_]+:(/[^>]*)>|<(/[^>]*)>")


def _required_env(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def validate_profile(image, tool_volume, workspace_volume, timeout_s):
    if not _SAFE_IMAGE.fullmatch(image):
        raise ValueError("SCRATCH_IMAGE must be a content ID or registry digest")
    if not _SAFE_VOLUME.fullmatch(tool_volume):
        raise ValueError("SCRATCH_TOOL_VOLUME has invalid format")
    if not _SAFE_VOLUME.fullmatch(workspace_volume):
        raise ValueError("SCRATCH_WORKSPACE_VOLUME has invalid format")
    if tool_volume == workspace_volume:
        raise ValueError("tool and workspace volumes must be different")
    timeout = float(timeout_s)
    if timeout <= 0 or timeout > 120:
        raise ValueError("SCRATCH_TIMEOUT_S must be in (0, 120]")
    return timeout


def _wrapped_script():
    # fd3 is the outer audit stream. It is closed in the model-authored subshell, whose stderr is
    # redirected to stdout, so the script cannot mix ordinary diagnostics into the strace stream.
    file_blocks = MAX_WORKSPACE_BYTES // 512
    return ("exec 3>&2\nset +e\n(\n  exec 3>&-\n  exec 2>&1\n"
            f"  ulimit -f {file_blocks}\n  /bin/sh -ceu \"$1\"\n"
            ")\ncodeaction_rc=$?\n"
            "codeaction_workspace_bytes=$(du -sb /workspace | cut -f1)\n"
            "codeaction_special_files=$(find /workspace -mindepth 1 "
            "\\( -type l -o -type s -o -type b -o -type c -o -type p "
            "\\) -print | wc -l)\n"
            "codeaction_setid_files=$(find /workspace -mindepth 1 -perm /6000 -print | wc -l)\n"
            "printf 'CODEACTION_WORKSPACE_BYTES=%s\\n' \"$codeaction_workspace_bytes\" >&3\n"
            "printf 'CODEACTION_WORKSPACE_SPECIAL_FILES=%s\\n' \"$codeaction_special_files\" >&3\n"
            "printf 'CODEACTION_WORKSPACE_SETID_FILES=%s\\n' \"$codeaction_setid_files\" >&3\n"
            f"if [ \"$codeaction_workspace_bytes\" -gt {MAX_WORKSPACE_BYTES} ]; then\n"
            "  find /workspace -mindepth 1 -delete\n"
            "fi\n"
            "if [ \"$codeaction_special_files\" -gt 0 ]; then\n"
            "  find /workspace -mindepth 1 "
            "\\( -type l -o -type s -o -type b -o -type c -o -type p "
            "\\) -delete\n"
            "fi\n"
            "if [ \"$codeaction_setid_files\" -gt 0 ]; then\n"
            "  find /workspace -mindepth 1 -perm /6000 -exec chmod a-s {} +\n"
            "fi\n"
            "exit \"$codeaction_rc\"\n")


def docker_command(image, tool_volume, workspace_volume, container_name, script):
    code = os.environ.get("SCRATCH_CODE_DIR")
    mounts = ["--mount", f"type=bind,src={code},dst=/opt/codeaction,readonly"] if code else []
    return [
        "docker", "run", "--rm", "--name", container_name,
        "--read-only", "--network", "none", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
        "--user", "65532:65532",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--mount", f"type=volume,src={tool_volume},dst=/run/codeaction,readonly",
        "--mount", f"type=volume,src={workspace_volume},dst=/workspace",
        "--workdir", "/workspace",
        "--env", "CODEACTION_TOOL_SOCKET=/run/codeaction/public-tools.sock",
        *mounts,
        "--entrypoint", "/usr/bin/strace", image,
        "-f", "-qq", "-yy", "-s", "256", "-e", "trace=%file,getdents64",
        "/bin/sh", "-ceu", _wrapped_script(), "codeaction-wrapper", script,
    ]


def _drain(stream, output, lock, cap, totals):
    while True:
        chunk = stream.read(8192)
        if not chunk:
            return
        with lock:
            totals[0] += len(chunk)
            remaining = cap - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])


def _decode_trace_path(value):
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except (UnicodeDecodeError, ValueError):
        return value


def audit_trace(raw, *, truncated=False):
    text = raw.decode("utf-8", errors="replace")
    violations = []
    trace_lines = 0
    workspace_bytes = None
    special_files = None
    setid_files = None
    match = _QUOTA_MARK.search(raw)
    if match:
        workspace_bytes = int(match.group(1))
    special_match = _SPECIAL_FILE_MARK.search(raw)
    if special_match:
        special_files = int(special_match.group(1))
    setid_match = _SETID_MARK.search(raw)
    if setid_match:
        setid_files = int(setid_match.group(1))
    if workspace_bytes is not None and workspace_bytes > MAX_WORKSPACE_BYTES:
        violations.append({"reason": "workspace_quota", "path": "/workspace"})
    if special_files:
        violations.append({"reason": "workspace_special_file", "path": "/workspace"})
    if setid_files:
        violations.append({"reason": "workspace_setid", "path": "/workspace"})

    sensitive = ("/Robotwin", "/root", "/home", "/run/secrets", "/var/run/docker.sock")
    allowed_enumeration = ("/workspace", "/usr/local/lib/python3.12", "/opt/codeaction")
    for line in text.splitlines():
        if line.startswith(("CODEACTION_WORKSPACE_BYTES=", "CODEACTION_WORKSPACE_SPECIAL_FILES=",
                            "CODEACTION_WORKSPACE_SETID_FILES=")):
            continue
        if "(" not in line:
            continue
        trace_lines += 1
        paths = [_decode_trace_path(value) for value in _QUOTED_PATH.findall(line)]
        for pair in _FD_PATH.findall(line):
            paths.append(pair[0] or pair[1])
        for path in paths:
            if any(path == prefix or path.startswith(prefix + "/") for prefix in sensitive):
                violations.append({"reason": "sensitive_path", "path": path[:256]})
            if re.match(r"^/proc/(?:self|[0-9]+)/(?:environ|cmdline|fd)(?:/|$)", path):
                violations.append({"reason": "process_introspection", "path": path[:256]})
        if "getdents64(" in line:
            for path in paths:
                if not any(path == prefix or path.startswith(prefix + "/")
                           for prefix in allowed_enumeration):
                    violations.append({"reason": "directory_enumeration", "path": path[:256]})

    deduped = []
    seen = set()
    for item in violations:
        key = (item["reason"], item["path"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    incomplete = (truncated or trace_lines == 0 or workspace_bytes is None
                  or special_files is None or setid_files is None or "strace:" in text)
    outcome = "incomplete" if incomplete else ("violation" if deduped else "clean")
    return {"backend": "docker-strace", "outcome": outcome,
            "trace_lines": trace_lines, "trace_truncated": bool(truncated),
            "workspace_bytes": workspace_bytes, "workspace_limit_bytes": MAX_WORKSPACE_BYTES,
            "workspace_special_files": special_files, "workspace_setid_files": setid_files,
            "violations": deduped[:32]}


def run_script(script, *, image, tool_volume, workspace_volume, timeout_s):
    if not isinstance(script, str):
        raise ValueError("script must be a string")
    if "\x00" in script or len(script.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise ValueError(f"script must be UTF-8 text no larger than {MAX_SCRIPT_BYTES} bytes")
    name = f"codeaction-scratch-{uuid.uuid4().hex[:16]}"
    command = docker_command(image, tool_volume, workspace_volume, name, script)
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={"PATH": os.environ.get("PATH", "")})
    output, trace = bytearray(), bytearray()
    output_lock, trace_lock = threading.Lock(), threading.Lock()
    output_total, trace_total = [0], [0]
    readers = [
        threading.Thread(target=_drain,
                         args=(proc.stdout, output, output_lock, MAX_OUTPUT_BYTES, output_total),
                         daemon=True),
        threading.Thread(target=_drain,
                         args=(proc.stderr, trace, trace_lock, MAX_AUDIT_BYTES, trace_total),
                         daemon=True),
    ]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(["docker", "rm", "-f", name], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                       env={"PATH": os.environ.get("PATH", "")}, timeout=15)
        try:
            exit_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=2)
    with output_lock:
        raw = bytes(output)
    with trace_lock:
        audit_raw = bytes(trace)
    audit = audit_trace(audit_raw, truncated=trace_total[0] > MAX_AUDIT_BYTES)
    if timed_out:
        audit["outcome"] = "incomplete"
        audit["timed_out"] = True
    return {"ok": exit_code == 0 and not timed_out, "exit_code": exit_code,
            "timed_out": timed_out, "truncated": output_total[0] > MAX_OUTPUT_BYTES,
            "output": raw.decode("utf-8", errors="replace"), "_audit": audit}


def _receive_request(conn):
    raw = bytearray()
    while True:
        chunk = conn.recv(min(8192, MAX_REQUEST_BYTES + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
        if b"\n" in chunk:
            break
    line, separator, trailing = bytes(raw).partition(b"\n")
    if not separator or trailing:
        raise ValueError("exactly one newline-terminated request is required")
    value = json.loads(line.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != {"request_id", "script"}:
        raise ValueError("request must contain exactly request_id and script")
    if not isinstance(value["request_id"], str) or not _SAFE_REQUEST_ID.fullmatch(
            value["request_id"]):
        raise ValueError("request_id has invalid format")
    return value["request_id"], value["script"]


def _handle(conn, profile):
    with conn:
        conn.settimeout(5)
        request_id = "invalid"
        audit = {"backend": "docker-strace", "outcome": "incomplete",
                 "violations": [], "error": "request rejected before execution"}
        try:
            request_id, script = _receive_request(conn)
            response = run_script(script, **profile)
            audit = response.pop("_audit")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            response = {"ok": False, "error": str(exc)}
        except Exception as exc:
            response = {"ok": False, "error": f"scratch launch failed: {type(exc).__name__}"}
        audit["request_id"] = request_id
        print("[filesystem-audit-result] " + json.dumps(
            audit, separators=(",", ":"), sort_keys=True), flush=True)
        try:
            conn.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
        except OSError:
            pass


def main() -> int:
    image = _required_env("SCRATCH_IMAGE")
    tool_volume = _required_env("SCRATCH_TOOL_VOLUME")
    workspace_volume = _required_env("SCRATCH_WORKSPACE_VOLUME")
    timeout_s = validate_profile(image, tool_volume, workspace_volume,
                                 os.environ.get("SCRATCH_TIMEOUT_S", "60"))
    profile = {"image": image, "tool_volume": tool_volume,
               "workspace_volume": workspace_volume, "timeout_s": timeout_s}
    host = os.environ.get("SCRATCH_LISTEN_HOST", "0.0.0.0")
    port = int(os.environ.get("SCRATCH_LISTEN_PORT", "8767"))
    with socket.create_server((host, port), family=socket.AF_INET, backlog=8) as listener:
        print(f"[scratch-launcher] listening on {host}:{port}", flush=True)
        while True:
            conn, _ = listener.accept()
            threading.Thread(target=_handle, args=(conn, profile), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
