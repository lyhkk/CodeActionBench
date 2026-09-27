"""No-subscription smoke client for mcp_episode_server.py: speaks raw JSON-RPC/MCP over local or
SSH stdio and exercises the FULL episode lifecycle against the real sim:
initialize → tools/list → get_embodiment → capture_head (asserts a real image content block)
→ done(success_claim=False) (asserts the ack is neutral: no verdict) → disconnect → then checks
on disk that result.json exists with a verifier verdict and the transcript starts with meta.
Run: "$ROBOTWIN_PYTHON" -m codeaction.transport.mcp.smoke_client
"""
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from codeaction.paths import PROJECT_ROOT, ROBOTWIN_ROOT


LOCAL_ROOT = PROJECT_ROOT
LOCAL_PYTHON = os.environ.get("ROBOTWIN_PYTHON", sys.executable)
REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "")
REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "python")
SSH_HOST = os.environ.get("ROBOTWIN_SSH_HOST", "")
SERVER_CMD = os.environ.get("SMOKE_SERVER_CMD", "")
APPEND_SERVER_ARGS = os.environ.get("SMOKE_APPEND_SERVER_ARGS", "1")
if "SMOKE_TRANSPORT" in os.environ:
    TRANSPORT = os.environ["SMOKE_TRANSPORT"]
else:
    TRANSPORT = "local"
if TRANSPORT not in {"local", "ssh", "cmd"}:
    raise ValueError("SMOKE_TRANSPORT must be local, ssh or cmd")
if TRANSPORT == "cmd" and not SERVER_CMD:
    raise ValueError("SMOKE_TRANSPORT=cmd requires SMOKE_SERVER_CMD")
if APPEND_SERVER_ARGS not in {"0", "1"}:
    raise ValueError("SMOKE_APPEND_SERVER_ARGS must be 0 or 1")

if TRANSPORT == "ssh" and not (SSH_HOST and REMOTE_ROOT):
    raise ValueError("SSH smoke requires ROBOTWIN_SSH_HOST and REMOTE_ROOT")

SERVER_MODULE = "codeaction.runtime.sim_server"
OUT_REL = os.environ.get(
    "SMOKE_OUT_REL", f"runs/vendor_smoke_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
SSH = ["ssh", "-o", "LogLevel=QUIET", SSH_HOST]
BOOT_TIMEOUT_S = 420.0
CALL_TIMEOUT_S = 120.0


class Client:
    def __init__(self):
        gpu = os.environ.get("SMOKE_GPU", "0")
        args = ["--out", OUT_REL, "--agent-label", "smoke-client"]
        if TRANSPORT == "cmd":
            # e.g. SMOKE_SERVER_CMD="docker run -i --rm --gpus all -v ... image:tag";
            # the standard server args are appended, exactly like the other transports.
            # Direct server commands need the standard args. A transport client connects to an
            # already-configured server, so transport-client smoke sets SMOKE_APPEND_SERVER_ARGS=0.
            cmd = shlex.split(SERVER_CMD) + (args if APPEND_SERVER_ARGS == "1" else [])
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
        elif TRANSPORT == "local":
            if not Path(LOCAL_PYTHON).is_file():
                raise FileNotFoundError(f"ROBOTWIN_PYTHON is missing: {LOCAL_PYTHON}")
            cmd = [LOCAL_PYTHON, "-m", SERVER_MODULE, *args]
            env = os.environ.copy()
            env.update({"ROBOTWIN_ROOT": str(ROBOTWIN_ROOT), "CUDA_VISIBLE_DEVICES": gpu,
                        "PYOPENGL_PLATFORM": "egl"})
            self.p = subprocess.Popen(cmd, cwd=LOCAL_ROOT, env=env, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
        else:
            remote = (
                f"cd {shlex.quote(REMOTE_ROOT)} && "
                f"ROBOTWIN_ROOT={shlex.quote(REMOTE_ROOT + '/backend/robotwin')} CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
                f"PYOPENGL_PLATFORM=egl {shlex.quote(REMOTE_PYTHON)} -m {SERVER_MODULE} "
                + " ".join(shlex.quote(x) for x in args)
            )
            self.p = subprocess.Popen(SSH + [remote], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
        self.q = queue.Queue()
        self._id = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.p.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except Exception:
                pass

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def call(self, method, params=None, timeout=CALL_TIMEOUT_S):
        self._id += 1
        rid = self._id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        while True:
            d = self.q.get(timeout=timeout)
            if d.get("id") == rid:
                if "error" in d:
                    raise RuntimeError(f"{method} -> {d['error']}")
                return d["result"]

    def tool(self, name, arguments=None, timeout=CALL_TIMEOUT_S):
        r = self.call("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
        return r.get("content", [])


def main():
    checks = []

    def ok(label, cond, extra=""):
        checks.append((label, bool(cond)))
        print(f"  {'PASS' if cond else 'FAIL'} {label} {extra}")

    c = Client()
    r = c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                              "clientInfo": {"name": "smoke", "version": "0"}}, timeout=60)
    ok("initialize", r.get("serverInfo", {}).get("name") == "codeaction", str(r.get("serverInfo")))
    c.notify("notifications/initialized")

    tools = [t["name"] for t in c.call("tools/list", timeout=60).get("tools", [])]
    need = {"capture_head", "capture_wrist", "capture_evidence_views",
            "move_delta", "reach_tcp", "run_code", "done"}
    ok("tools/list", need.issubset(set(tools)), f"{len(tools)} tools")
    ok("get_task retired", "get_task" not in tools)

    # first sim-touching call — queues behind the scene boot
    print(f"  ... get_embodiment (waits for scene boot, up to {BOOT_TIMEOUT_S:.0f}s)")
    emb = json.loads(c.tool("get_embodiment", timeout=BOOT_TIMEOUT_S)[0]["text"])
    ok("get_embodiment", "gripper" in json.dumps(emb).lower())

    cap = c.tool("capture_head", {})
    kinds = [x.get("type") for x in cap]
    img = next((x for x in cap if x.get("type") == "image"), None)
    ok("capture_head kinds", kinds == ["text", "image"], str(kinds))
    ok("capture_head image", img is not None and len(img.get("data", "")) > 10000
       and img.get("mimeType") == "image/png",
       f"b64len={len(img.get('data', '')) if img else 0}")
    payload = json.loads(cap[0]["text"])
    ok("capture_head payload", payload.get("obs_id")
       and "in this tool result" in payload.get("image", ""))

    ack = json.loads(c.tool("done", {"report": "smoke: no action taken",
                                     "success_claim": False})[0]["text"])
    ok("done neutral ack", ack.get("ok") is True and "verifier" not in json.dumps(ack)
       and "success" not in json.dumps(ack), json.dumps(ack))

    after = json.loads(c.tool("capture_head", {})[0]["text"])
    ok("post-done lockout", "EPISODE_OVER" in after.get("error", ""), after.get("error", "")[:60])

    c.p.stdin.close()
    c.p.wait(timeout=180)

    if TRANSPORT in ("local", "cmd"):
        out = LOCAL_ROOT / "." / OUT_REL
        res_txt = (out / "result.json").read_text(encoding="utf-8")
        meta_txt = (out / "transcript.jsonl").read_text(encoding="utf-8").splitlines()[0]
    else:
        out = f"{REMOTE_ROOT}/{OUT_REL}"
        chk = subprocess.run(
            SSH + [f"cat {shlex.quote(out + '/result.json')}; echo ---; "
                   f"head -1 {shlex.quote(out + '/transcript.jsonl')}"],
            capture_output=True, text=True, timeout=60)
        res_txt, _, meta_txt = chk.stdout.partition("---")
    try:
        res = json.loads(res_txt)
        meta = json.loads(meta_txt)
    except Exception:
        res, meta = {}, {}
    ok("result.json verifier", "verifier" in res and "success" in res.get("verifier", {}),
       json.dumps(res.get("verifier", {})))
    ok("result.json status", res.get("stats", {}).get("status") == "done",
       str(res.get("stats", {}).get("status")))
    ok("verifier says not lifted", res.get("verifier", {}).get("success") is False)
    ok("transcript meta", meta.get("event") == "meta" and "tools" in meta
       and meta.get("interface") == "mcp-agent")

    n_fail = sum(1 for _, c_ in checks if not c_)
    print(f"\nmcp episode smoke: {'ALL GREEN' if n_fail == 0 else f'{n_fail} FAILED'} "
          f"({len(checks)} checks)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
