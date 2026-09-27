"""run_code sandbox (spec §5 v4) — structural isolation for model-authored Python.

THE architectural argument: ground truth lives in the PARENT process only. The child receives (a) a
restricted exec namespace and (b) proxy functions that RPC (tool_name, kwargs) over a pipe to the
parent, which dispatches to the real ToolBox and returns JSON-safe dicts. Even a FULLY escaped child
gains only the ability to send tool requests — exactly the surface the model already has via
tool-use. On top of that, defense-in-depth: an AST allowlist (only exact numpy/math imports, no
dunder attribute access, no eval/exec/open/getattr...), a minimal builtins table, a per-run tool-call
ceiling (v0.5: runaway-loop protection only — internal calls are NOT charged to the episode budget;
run_code counts as one round), and a wall-clock timeout with a hard child kill.

Persistence (v0.3, value position #4 made real): ONE child process serves the whole episode, so
variables and defs persist across run_code calls (in-episode skill/state reuse — the model can keep
its own measurements in its own namespace). Each run returns only what IT set (`result` is cleared
before every exec). A code exception does NOT reset the namespace. A wall-clock timeout, child
death, or an atomic action returning ABORTED kills the child — that run returns
`namespace_reset: true` and the next run starts fresh. Agent-owned virtual files live in the parent
and survive those execution-namespace resets.

What model code CAN do (the four value positions, spec §5): numpy/math over load_image() pixel
copies (model-built measurement instruments — legitimate 识图), proprioceptive/tactile inner loops
over tool dicts, batch collection ("code collects, model reviews" — the runner attaches captured
obs images next turn), and def-based in-episode skill reuse. Full tracebacks are returned (debugging
is a measured ability, §12/§14). Set a variable named `result` to return a value."""
import ast
import io
import json
import math
import time
import traceback
from multiprocessing import get_context
from pathlib import PurePosixPath

import numpy as np

from codeaction.runtime.composition import (DEFAULT_READ_CHUNK_BYTES as _DEFAULT_READ_CHUNK_BYTES,
                                 MAX_PROGRAM_FILE_BYTES as _MAX_PROGRAM_FILE_BYTES,
                                 MAX_PROGRAM_FILES as _MAX_PROGRAM_FILES,
                                 MAX_PROGRAM_PATH_BYTES as _MAX_PROGRAM_PATH_BYTES,
                                 MAX_PROGRAM_PATH_PARTS as _MAX_PROGRAM_PATH_PARTS,
                                 MAX_PROGRAM_TOTAL_BYTES as _MAX_PROGRAM_TOTAL_BYTES,
                                 MAX_READ_CHUNK_BYTES as _MAX_READ_CHUNK_BYTES,
                                 MODEL_VISIBLE_RESULT_MAX_BYTES,
                                 STDOUT_MAX_CHARS as _STDOUT_CAP,
                                 declared_composition_contract)
from codeaction.contracts.failures import (PhysicalTimeBudgetExhausted, ProgramWorkspaceError,
                              is_recoverable_action_abort)

_BANNED_NAMES = {
    "eval", "exec", "open", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "exit", "quit", "memoryview",
    "type", "super", "object", "classmethod", "staticmethod", "property",
}
_SPAWN_TIMEOUT_S = 30.0
_SAFE_MODEL_IMPORTS = {"numpy", "math"}


def _collect_obs_ids(payload):
    out = []

    def walk(x):
        if isinstance(x, dict):
            oid = x.get("obs_id")
            if isinstance(oid, str):
                out.append(oid)
            obs_ids = x.get("obs_ids")
            if isinstance(obs_ids, list):
                out.extend(v for v in obs_ids if isinstance(v, str))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(payload)
    dedup = []
    seen = set()
    for oid in out:
        if oid not in seen:
            seen.add(oid)
            dedup.append(oid)
    return dedup


# The arguments a primitive was called with, bounded. Without them the trace says a reach_tcp
# happened at tick 12 but not WHICH ARM or where, so an offline re-verdict of a predicate that
# reads the robot (open_laptop measures one arm's TCP) has to parse the model's own code text to
# find out. These are the model's own inputs -- already visible to it -- so recording them leaks
# nothing; the cap keeps one draw_marks call from dominating the transcript.
_TRACE_ARGS_MAX_BYTES = 2000
# Keys the sandbox writes into internal_trace for the RECORD ONLY. They are stripped from the
# model-visible projection, so adding one leaves the agent's information surface byte-identical
# and no episode recorded before it becomes a different tested unit.
SERVER_ONLY_TRACE_KEYS = ("args", "args_omitted", "achieved_pose")


def _trace_pose(payload):
    """Where a motion primitive ended, for the record.

    Whether a TCP survives into the archive should not depend on how the agent chose to return
    it: one released episode printed every pose to stdout and left the transcript with no `tcp`
    key at all, which made an offline re-verdict of a predicate that measures the robot
    impossible for that run. The server sees the achieved pose on every primitive, so it records
    it.
    """
    if not isinstance(payload, dict):
        return {}
    pose = payload.get("resulting_pose")
    if not isinstance(pose, dict):
        return {}
    kept = {key: pose[key] for key in ("tcp", "ee") if isinstance(pose.get(key), list)}
    return {"achieved_pose": kept} if kept else {}


def _trace_args(kw):
    if not isinstance(kw, dict) or not kw:
        return {}
    try:
        encoded = json.dumps(kw, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return {"args_omitted": "not serialisable"}
    if len(encoded.encode("utf-8")) > _TRACE_ARGS_MAX_BYTES:
        return {"args_omitted": f"over {_TRACE_ARGS_MAX_BYTES} bytes"}
    return {"args": json.loads(encoded)}


def _interrupted_code_result(calls, obs_ids, internal_trace, action):
    """Stop open-loop code after any atomic action reports ABORTED."""
    abort_reason = str(action.get("abort_reason") or "action_aborted")
    failure = action.get("failure")
    if not isinstance(failure, dict):
        failure = {
            "stage": "execution",
            "code": "ACTION_ABORTED",
            "message": "an atomic action returned ABORTED",
        }
    payload = {
        "ok": False,
        "status": "ABORTED",
        "abort_reason": abort_reason,
        "failure": failure,
        "error": (
            "run_code stopped after an atomic action returned ABORTED; inspect "
            "interrupted_action and replan in the next agent turn (episode remains active). "
            "Calls submitted after this one in the same assistant turn were not executed"),
        "stdout": "",
        "value": None,
        "result_assigned": False,
        "tool_calls": calls,
        "obs_ids": obs_ids,
        "internal_trace": internal_trace,
        "filesystem_policy": "structurally_denied",
        "namespace_reset": True,
    }
    if isinstance(action, dict):
        payload["interrupted_action"] = action
    return payload


class SandboxViolation(ValueError):
    """Raised (parent-side, before any process is spawned) when code fails the AST allowlist."""


def validate_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxViolation(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            if not names.issubset(_SAFE_MODEL_IMPORTS):
                raise SandboxViolation("only `import numpy as np` and `import math` are allowed; "
                                       "both modules are already pre-bound")
        if isinstance(node, ast.ImportFrom):
            raise SandboxViolation("from-imports are not allowed; np and math are pre-bound")
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise SandboxViolation("async is not allowed")
        if isinstance(node, ast.ClassDef):
            raise SandboxViolation("class definitions are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxViolation(f"dunder attribute access is not allowed: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise SandboxViolation(f"{node.id!r} is not allowed")



def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Minimal __import__ for the exec namespace. C-level lazy imports inside already-trusted
    numpy/math (error formatting, casting machinery) resolve __import__ from the EXECUTING
    frame's builtins — without it they die with an inscrutable KeyError '__import__' (live
    incident 2026-07-12: 11/19 vendor-agent run_code calls killed mid-analysis). Model code can never
    reach this by name (the AST gate bans the identifier); the AST gate permits only exact
    numpy/math import statements, and library internals use the same restricted path. Anything
    else raises ImportError."""
    import sys as _sys
    import importlib as _il
    if level:
        raise ImportError("relative imports are not allowed in run_code")
    root = str(name).partition(".")[0]
    if root not in ("numpy", "math"):
        raise ImportError(f"import of {name!r} is not allowed in run_code")
    mod = _sys.modules.get(name)
    if mod is None:
        if root != "numpy" or name == "numpy":
            raise ImportError(f"import of {name!r} is not allowed in run_code")
        mod = _il.import_module(name)          # numpy's OWN lazy submodules only
    return mod if fromlist else _sys.modules[root]


def _safe_builtins():
    import builtins as _b
    allow = ("abs", "min", "max", "sum", "round", "sorted", "reversed", "enumerate", "zip",
             "len", "range", "map", "filter", "any", "all", "isinstance", "repr",
             "list", "dict", "tuple", "set", "str", "int", "float", "bool",
             "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
             "NameError", "AttributeError",
             "ZeroDivisionError", "RuntimeError", "StopIteration")
    out = {n: getattr(_b, n) for n in allow}
    out["__import__"] = _restricted_import     # numpy/math internal lazy imports (see above)
    return out


def _worker(conn):
    """Child: ONE persistent restricted namespace for the whole episode. Executes code payloads
    sent by the parent one at a time; every tool/image access is an RPC to the parent."""
    out_box = {"buf": io.StringIO()}

    def _print(*a, **k):
        k.pop("file", None)
        print(*a, file=out_box["buf"], **k)
        if out_box["buf"].tell() > _STDOUT_CAP:
            raise RuntimeError("stdout budget exhausted")

    def _rpc(kind, name, kw):
        conn.send((kind, name, kw))
        ok, payload = conn.recv()
        if not ok:
            raise RuntimeError(payload)
        return payload

    def _proxy(name):
        def call(**kw):
            return _rpc("tool", name, kw)
        call.__name__ = name
        return call

    def load_image(obs_id):
        """PNG bytes from the parent → HxWx3 uint8 array (a COPY — never a renderer handle)."""
        raw = _rpc("image", str(obs_id), {})
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))

    conn.send(("names", None, {}))
    _ok, tool_names = conn.recv()
    ns = {"__builtins__": _safe_builtins(), "np": np, "math": math,
          "print": _print, "load_image": load_image}
    ns.update({n: _proxy(n) for n in tool_names})

    while True:
        try:
            kind, _name, kw = conn.recv()
        except (EOFError, OSError):
            return
        if kind != "exec":
            continue
        out_box["buf"] = io.StringIO()
        ns.pop("result", None)         # a run returns only what IT set — no stale leak-through
        err = None
        try:
            exec(compile(kw["code"], "<run_code>", "exec"), ns)   # noqa: S102 — the sandbox's purpose
        except BaseException:
            err = traceback.format_exc(limit=8)
        result_assigned = "result" in ns
        val = ns.get("result")
        try:
            json.dumps(val)
        except Exception:
            val = repr(val)
        conn.send(("done", None, {"value": val,
                                  "result_assigned": result_assigned,
                                  "stdout": out_box["buf"].getvalue()[:_STDOUT_CAP],
                                  "error": err}))


class Sandbox:
    """Parent-side controller. dispatch(name, kwargs)->JSON-safe dict runs a REAL tool;
    image_loader(obs_id)->PNG bytes. max_tool_calls is the per-run RUNAWAY ceiling (v0.5: not an
    episode budget — run_code charges one round; internals are free but a stuck while-loop must
    die before the wall clock). One persistent child serves all runs (namespace persists);
    timeout/child-death kills it and the NEXT run respawns fresh."""

    def __init__(self, dispatch, image_loader, tool_names, max_tool_calls=500, timeout_s=120.0):
        self._dispatch = dispatch
        self._image_loader = image_loader
        self._tool_names = [n for n in tool_names if n not in ("done", "run_code")]
        self._max_calls = int(max_tool_calls)
        self._timeout = float(timeout_s)
        self._ctx = get_context("spawn")
        self._proc = None
        self._conn = None
        self._program_files = {}

    @staticmethod
    def _program_path(path, *, suffixes=(".py", ".md")):
        if not isinstance(path, str):
            raise ProgramWorkspaceError("invalid_virtual_file_argument",
                                        "path must be a string")
        try:
            encoded = path.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument", "path must be valid UTF-8") from exc
        parts = path.split("/")
        if (not encoded or len(encoded) > _MAX_PROGRAM_PATH_BYTES or path.startswith("/")
                or "\\" in path or "\x00" in path or any(p in ("", ".", "..") for p in parts)
                or len(parts) > _MAX_PROGRAM_PATH_PARTS):
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument",
                f"path must be a normalized relative path with at most "
                f"{_MAX_PROGRAM_PATH_PARTS} parts and {_MAX_PROGRAM_PATH_BYTES} UTF-8 bytes")
        normalized = str(PurePosixPath(path))
        if not normalized.endswith(tuple(suffixes)):
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument",
                f"path must end in one of {', '.join(suffixes)}")
        return normalized

    @property
    def program_limits(self):
        return {"max_files": _MAX_PROGRAM_FILES,
                "max_file_bytes": _MAX_PROGRAM_FILE_BYTES,
                "max_total_bytes": _MAX_PROGRAM_TOTAL_BYTES,
                "max_path_bytes": _MAX_PROGRAM_PATH_BYTES,
                "max_path_parts": _MAX_PROGRAM_PATH_PARTS,
                "suffixes": [".py", ".md"],
                "default_read_chunk_bytes": _DEFAULT_READ_CHUNK_BYTES,
                "max_read_chunk_bytes": _MAX_READ_CHUNK_BYTES}

    @property
    def composition_contract(self):
        return declared_composition_contract(
            max_internal_tool_calls=self._max_calls, timeout_s=self._timeout)

    @property
    def filesystem_audit(self):
        return {"schema_version": "0.1", "backend": "restricted-python-ast",
                "outcome": "structurally_denied", "host_filesystem_exposed": False,
                "program_workspace": "in_memory",
                "program_workspace_limits": self.program_limits,
                "enforcement": ["restricted_builtins", "import_allowlist", "dunder_rejection",
                                "no_open", "no_subprocess", "normalized_virtual_paths"]}

    def seed_file(self, path, content):
        """Harness-owned write of a reference document into the agent's virtual workspace.

        Used by the code-first surface to place the primitive-library reference where `read_file`
        and `list_files` can reach it. It carries tool semantics only — the same text the default
        surface delivers as schemas — never scene facts or a suggested procedure.
        """
        return self.write_file(path, content)

    def write_file(self, path, content):
        path = self._program_path(path)
        if not isinstance(content, str) or "\x00" in content:
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument", "content must be text without NUL bytes")
        try:
            raw = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument", "content must be valid UTF-8") from exc
        if len(raw) > _MAX_PROGRAM_FILE_BYTES:
            raise ProgramWorkspaceError(
                "virtual_workspace_limit", f"file exceeds {_MAX_PROGRAM_FILE_BYTES} bytes")
        if path not in self._program_files and len(self._program_files) >= _MAX_PROGRAM_FILES:
            raise ProgramWorkspaceError(
                "virtual_workspace_limit",
                f"workspace file count exceeds {_MAX_PROGRAM_FILES}")
        old_size = len(self._program_files.get(path, "").encode("utf-8"))
        total = sum(len(value.encode("utf-8")) for value in self._program_files.values())
        if total - old_size + len(raw) > _MAX_PROGRAM_TOTAL_BYTES:
            raise ProgramWorkspaceError(
                "virtual_workspace_limit", f"workspace exceeds {_MAX_PROGRAM_TOTAL_BYTES} bytes")
        self._program_files[path] = content
        return {"ok": True, "path": path, "bytes": len(raw),
                "filesystem_policy": "structurally_denied"}

    def read_file(self, path, offset_bytes=0, max_bytes=_DEFAULT_READ_CHUNK_BYTES):
        path = self._program_path(path)
        if path not in self._program_files:
            raise ProgramWorkspaceError(
                "virtual_file_not_found", f"program file not found: {path}")
        if (not isinstance(offset_bytes, int) or isinstance(offset_bytes, bool)
                or offset_bytes < 0):
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument", "offset_bytes must be a non-negative integer")
        if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
                or not 4 <= max_bytes <= _MAX_READ_CHUNK_BYTES):
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument",
                f"max_bytes must be an integer from 4 to {_MAX_READ_CHUNK_BYTES}")
        raw = self._program_files[path].encode("utf-8")
        total = len(raw)
        if offset_bytes > total:
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument",
                f"offset_bytes must be <= total_bytes ({total})")
        try:
            raw[:offset_bytes].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProgramWorkspaceError(
                "invalid_virtual_file_argument",
                "offset_bytes must be at a UTF-8 code-point boundary") from exc

        end = min(total, offset_bytes + max_bytes)
        while end > offset_bytes:
            try:
                content = raw[offset_bytes:end].decode("utf-8")
                break
            except UnicodeDecodeError:
                end -= 1
        else:
            content = ""

        while True:
            payload = {
                "ok": True, "path": path, "content": content,
                "start_byte": offset_bytes, "end_byte": end,
                "total_bytes": total, "returned_bytes": end - offset_bytes,
                "next_offset_bytes": None if end == total else end,
                "eof": end == total, "filesystem_policy": "structurally_denied",
            }
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")
            if len(encoded) <= MODEL_VISIBLE_RESULT_MAX_BYTES:
                return payload
            span = end - offset_bytes
            if span <= 1:
                raise RuntimeError("model-visible result ceiling cannot hold a file chunk")
            end = offset_bytes + max(1, span * 3 // 4)
            while end > offset_bytes:
                try:
                    content = raw[offset_bytes:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1

    def list_files(self):
        files = [{"path": path, "bytes": len(content.encode("utf-8"))}
                 for path, content in sorted(self._program_files.items())]
        return {"ok": True, "files": files,
                "total_bytes": sum(item["bytes"] for item in files),
                "limits": self.program_limits,
                "filesystem_policy": "structurally_denied"}

    def run_program(self, path):
        path = self._program_path(path, suffixes=(".py",))
        if path not in self._program_files:
            raise ProgramWorkspaceError(
                "virtual_file_not_found", f"program file not found: {path}")
        payload = self.run(self._program_files[path])
        payload["path"] = path
        payload["filesystem_policy"] = "structurally_denied"
        return payload

    def _spawn(self):
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_worker, args=(child,), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        if not parent.poll(_SPAWN_TIMEOUT_S):          # child announces itself with "names"
            self._kill()
            raise RuntimeError("sandbox child failed to start")
        parent.recv()
        parent.send((True, self._tool_names))

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.join(timeout=5)
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._proc, self._conn = None, None

    def close(self):
        """Episode over: reap the child (daemon=True also dies with the parent)."""
        self._kill()
        self._program_files.clear()

    @property
    def max_tool_calls(self) -> int:
        return self._max_calls

    def run(self, code: str, max_tool_calls=None) -> dict:
        """Execute one code block.

        ``max_tool_calls`` may further restrict THIS run below the configured per-run runaway
        ceiling (fixtures/tests use it); it never raises that ceiling. Since v0.5 the episode
        budget never flows in here — internal calls are uncharged.
        """
        run_max_calls = self._max_calls
        if max_tool_calls is not None:
            run_max_calls = min(run_max_calls, max(0, int(max_tool_calls)))
        try:
            validate_code(code or "")
        except SandboxViolation as e:
            return {"ok": False, "error": f"rejected: {e}", "stdout": "", "value": None,
                    "result_assigned": False, "tool_calls": 0, "obs_ids": [],
                    "internal_trace": [], "filesystem_policy": "structurally_denied",
                    "namespace_reset": False}
        if self._proc is None or not self._proc.is_alive():
            self._spawn()
        self._conn.send(("exec", None, {"code": code}))
        calls, obs_ids, internal_trace = 0, [], []
        deadline = time.time() + self._timeout
        while True:
            if not self._conn.poll(max(0.05, deadline - time.time())):
                self._kill()
                return {"ok": False,
                        "error": f"sandbox timeout after {self._timeout}s (child killed; "
                                 f"persistent namespace reset)",
                        "stdout": "", "value": None, "result_assigned": False,
                        "tool_calls": calls, "obs_ids": obs_ids,
                        "internal_trace": internal_trace,
                        "filesystem_policy": "structurally_denied",
                        "namespace_reset": True}
            try:
                kind, name, kw = self._conn.recv()
            except (EOFError, OSError):
                self._kill()
                return {"ok": False, "error": "sandbox child died (persistent namespace reset)",
                        "stdout": "", "value": None, "result_assigned": False,
                        "tool_calls": calls, "obs_ids": obs_ids,
                        "internal_trace": internal_trace,
                        "filesystem_policy": "structurally_denied",
                        "namespace_reset": True}
            if kind == "tool":
                if calls >= run_max_calls:
                    if not internal_trace or internal_trace[-1].get(
                            "error") != "tool_call_ceiling_exhausted":
                        internal_trace.append({"index": calls + 1, "tool": name, "ok": False,
                                               "error": "tool_call_ceiling_exhausted"})
                    self._conn.send((False, f"sandbox per-run tool-call ceiling exhausted "
                                            f"({run_max_calls})"))
                    continue
                calls += 1
                try:
                    payload = self._dispatch(name, kw or {})
                    call_obs_ids = _collect_obs_ids(payload)
                    obs_ids.extend(call_obs_ids)
                    trace_item = {"index": calls, "tool": name, "ok": True,
                                  **_trace_args(kw), **_trace_pose(payload)}
                    if isinstance(payload, dict):
                        if payload.get("status") is not None:
                            trace_item["status"] = payload.get("status")
                        if payload.get("action_id") is not None:
                            trace_item["action_id"] = payload.get("action_id")
                        if payload.get("tick") is not None:
                            trace_item["tick"] = payload.get("tick")
                        if payload.get("abort_reason") is not None:
                            trace_item["abort_reason"] = payload.get("abort_reason")
                        achieved = payload.get("achieved") or {}
                        if isinstance(achieved, dict):
                            for key in ("failure_category", "stop_reason", "transition"):
                                if achieved.get(key) is not None:
                                    trace_item[key] = achieved.get(key)
                    if call_obs_ids:
                        trace_item["obs_ids"] = call_obs_ids
                    internal_trace.append(trace_item)
                    if is_recoverable_action_abort(payload):
                        # Do not let the same open-loop code block ignore the abort and issue more
                        # primitives. The next external agent turn remains available and receives
                        # the complete structured ActionResult.
                        self._kill()
                        return _interrupted_code_result(
                            calls, obs_ids, internal_trace, payload)
                    self._conn.send((True, payload))
                except PhysicalTimeBudgetExhausted as exc:
                    # This is an episode terminal, not a model-code exception. Killing the child
                    # prevents another queued primitive and re-raising lets EpisodeRuntime run
                    # the normal finalizer/verifier path. Re-raising alone discarded everything
                    # the block had already done, because `run()` never returns and the trace is
                    # a local: the primitives that ran before the threshold vanished from the
                    # record. Carry them on the exception so the terminal can keep them.
                    self._kill()
                    exc.internal_trace = list(internal_trace)
                    exc.tool_calls = calls
                    exc.obs_ids = list(obs_ids)
                    raise
                except Exception as e:
                    # The arguments matter MORE on this path than on the success path: a failed
                    # reach_tcp that records only its exception type says nothing about which arm
                    # was aimed where, and that is exactly what a later reading of the failure
                    # needs. There is no achieved pose to record, because nothing was achieved.
                    internal_trace.append({"index": calls, "tool": name, "ok": False,
                                           "error": type(e).__name__, **_trace_args(kw)})
                    self._conn.send((False, f"{type(e).__name__}: {e}"))
            elif kind == "image":
                try:
                    raw = self._image_loader(name)
                    # `load_image` is not an episode tool call, but it is an explicit request to
                    # make this stored observation model-visible again. Feed its obs_id through
                    # the same result projection as newly captured images; the projection's
                    # last-occurrence dedupe prevents capture+load or repeated loads from sending
                    # duplicate bytes.
                    obs_ids.append(str(name))
                    self._conn.send((True, raw))
                except Exception as e:
                    self._conn.send((False, f"{type(e).__name__}: {e}"))
            elif kind == "done":
                res = kw
                return {"ok": res.get("error") is None, "error": res.get("error"),
                        "stdout": res.get("stdout", ""), "value": res.get("value"),
                        "result_assigned": bool(res.get("result_assigned", False)),
                        "tool_calls": calls, "obs_ids": obs_ids,
                        "internal_trace": internal_trace,
                        "filesystem_policy": "structurally_denied",
                        "namespace_reset": False}


def make_sandbox(toolbox, **kwargs):
    """Wire the restricted code sandbox to the same runtime registry and image store.

    Lives beside Sandbox because the episode server calls it directly; the harness-side
    facade re-exports it. Serialization is imported lazily to keep sandbox import-light.
    """
    from pathlib import Path
    from codeaction.interface.schemas import serialize, serialize_tool_result
    registry = toolbox.registry_map()
    return Sandbox(
        lambda name, arguments: (
            serialize_tool_result(name, registry[name](**(arguments or {})))
            if getattr(toolbox, "enforce_result_contracts", False)
            else serialize(registry[name](**(arguments or {})))
        ),
        lambda observation_id: Path(toolbox.image_path(observation_id)).read_bytes(),
        list(registry),
        **kwargs,
    )
