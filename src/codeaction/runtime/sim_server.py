"""vendor-agent episode host: serves ONE benchmark episode over MCP stdio to an external agent CLI
(spawned locally or over SSH). Scene setup + GT + the out-of-band verifier live HERE, on the
host side of the tool channel, never with the agent; the agent-facing surface is
codeaction.runtime.mcp_bridge (audited).

fd discipline (load-bearing): fd1 IS the JSON-RPC channel, and sapien/cuRobo print to stdout on
boot — so we dup fd1 for MCP FIRST and point fd1/fd2 at a log file before any sim import.

Threading: ONE worker thread owns the sim end-to-end (boot job first, then every tool dispatch,
then finalize) — FIFO ordering means a tool call issued during boot simply waits its turn; the
agent side must allow a long first-tool timeout (MCP_TOOL_TIMEOUT).

Verifier lifecycle: z0 is recorded at setup; `done` (or agent disconnect / budget / fatal) fires
finalize exactly once → task-card verifier runs in-parent, result.json + run_meta.json are
written, the HTML report is rebuilt. The verdict NEVER goes back over the tool channel.

Run (remote, spawned by the Mac driver):
  CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=egl "$ROBOTWIN_PYTHON" \
    -m codeaction.runtime.sim_server \
    --out data/vendor_stage3
"""
import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codeaction.paths import PROJECT_ROOT, REPOSITORY_ROOT, ROBOTWIN_ROOT

_RT = ROBOTWIN_ROOT
for p in (_RT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Bounded cuRobo planning for benchmark episodes (see envs/robot/planner.py _plan_config):
# pinned timeout + no auto graph escalation. Set BEFORE the worker thread boots the sim.
os.environ.setdefault("CODEACTION_CUROBO_BOUNDED_PLAN", "1")
# The benchmark owns collision consequences instead of giving cuRobo a privileged fixed table:
# the per-physics-step monitor below terminates any non-probe robot-world contact.
os.environ["CODEACTION_CUROBO_TABLE_WORLD"] = "0"

TASK = None          # loaded from the declarative task package (tasks/<name>/) in main()
CARD = None

# Wedge wall: if ONE tool call exceeds this, the sim worker is considered wedged (2026-07-08: a
# reach_tcp hung >600 s — not deterministically reproducible — and took the MCP connection, a
# zombie GPU process, and possibly the box with it). We die HONESTLY before the client's
# MCP_TOOL_TIMEOUT (600 s): write result.json (python-side counters only, NO env access from this
# thread), then os._exit — killing the process frees the GPU and leaves forensics on disk.
# Legit worst case stays far below: scene boot ~180 s + bounded plans (CODEACTION_CUROBO_BOUNDED_PLAN) <60 s.
TOOL_WEDGE_S = 480.0


def _secure_stdio(log_path):
    """Reserve the real stdout for JSON-RPC; everything else (sim prints, our logs) → log file."""
    raw_out = os.fdopen(os.dup(1), "wb", buffering=0)
    logf = open(log_path, "ab", buffering=0)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    sys.stdout = io.TextIOWrapper(logf, encoding="utf-8", line_buffering=True)
    sys.stderr = sys.stdout
    return raw_out


def _log(*a):
    print(f"[mcp_host {time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


def _write_preflight_failure(out_dir, detail):
    from codeaction.contracts.failures import FailureCode, default_failure
    failure = default_failure(
        FailureCode.HARNESS_CONTRACT_VIOLATION,
        stage="preflight",
        detail_safe=detail,
    )
    result = {
        "status": "preflight_failed",
        "stats": {"status": "preflight_failed", "budget_used": 0,
                  "total_calls": 0, "failure": failure.to_dict()},
        "verifier": {"success": False, "score": 0.0, "kind": "not_run"},
        "failure": failure.to_dict(),
    }
    (Path(out_dir) / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# Both reference profiles are served by the same host; they differ only in which
# primitives are delivered as model-facing schemas.
_REFERENCE_PROFILES = ("reference-mcp", "reference-code-first")

class Host:
    """Owns scene + GT + finalization. The bridge only ever sees the ToolBox/sandbox."""

    def __init__(self, args, out_dir):
        self.args = args
        self.out = out_dir
        self.bridge = None
        self.env = None
        self.target = None
        self.z0 = None
        self.boot_error = None
        self._persisted = False
        self.provenance = args.provenance
        self.latch_monitor = None      # poll_and_latch (G2 #6): set only when the card declares it
        self.observer = None
        self.recorder = None
        self.video_meta = None
        self.video_error = None
        self._recording_finalized = False
        # Requests admitted while one MCP tool is executing share this generation. If that tool
        # returns an aborted action, already-queued siblings are cancelled; a request
        # submitted after the response observes the next generation and may execute normally.
        self._abort_generation = 0
        self._abort_generation_lock = threading.Lock()

    def abort_generation(self):
        with self._abort_generation_lock:
            return self._abort_generation

    def _advance_abort_generation(self):
        with self._abort_generation_lock:
            self._abort_generation += 1

    def boot(self):
        try:
            import numpy as np
            from codeaction.backends.robotwin.geometry import actor_center
            from codeaction.interface.tools import ToolBox
            from codeaction.runtime.sandbox import make_sandbox
            from codeaction.runtime.mcp_bridge import EpisodeBridge
            _log("booting scene ...")
            sc = CARD["scene"]
            if sc.get("loader") == "env":       # generic RoboTwin task (env_check_success verifier)
                from codeaction.backends.robotwin.scene import load_task_scene
                from codeaction.verification.verifiers import snapshot_actor_positions, snapshot_actor_quats
                ctx = load_task_scene(sc["task_name"], seed=self.args.seed,
                                      config=sc.get("config", "demo_clean_aloha"),
                                      expected_env_source=sc.get("env_source"))
                self.env, vp = ctx["env"], ctx["vp"]
                self.target, self.z0 = None, None   # no single target; verifier calls env.check_success
                self.table_z = float(ctx.get("table_z", 0.74))
                vspec = CARD["verifier"]
                names = set(vspec.get("destructive_actors") or [])
                names.update(m["actor"] for m in vspec.get("milestones", []) if "actor" in m)
                self.initial_poses = snapshot_actor_positions(self.env, sorted(names))
                self.initial_quats = snapshot_actor_quats(
                    self.env, vspec.get("integrity_actors") or [])
            else:
                raise ValueError(
                    f"unsupported formal scene loader: {sc.get('loader')!r}; "
                    "CodeAction release tasks require loader='env'")
            tb = ToolBox(self.env, vp, str(self.out / "tools"))
            self._attach_episode_observers(CARD["verifier"], tb)
            sb = (make_sandbox(
                      tb,
                      max_tool_calls=self.args.run_code_max_internal_calls,
                      timeout_s=180.0)
                  if self.args.hybrid else None)
            self.bridge = EpisodeBridge(
                tb, TASK, str(self.out), sandbox=sb,
                max_steps=self.args.max_tool_calls,
                wall_budget_s=self.args.wall_budget,
                physical_time_budget_s=self.args.physical_time_budget_s,
                on_finalize=self.persist,
                agent_label=self.args.agent_label,
                agent_cli_version=self.args.agent_cli_version,
                sim_step_fn=lambda: self.observer.step_count,
                interface_profile=self.args.interface_profile,
                expected_tool_surface=self.args.expected_tool_surface,
                expected_instruction_surface=self.args.expected_instruction_surface,
                tool_discovery=self.args.tool_discovery)
            _log(f"scene ready (z0 recorded, hybrid={bool(sb)}); serving tools")
        except Exception as e:
            self.boot_error = f"{type(e).__name__}: {e}"
            _log("BOOT FAILED:", self.boot_error)

    def _attach_episode_observers(self, vspec, toolbox):
        """Attach the one per-physics-step seam used by latch telemetry and continuous recording.

        Also takes the end-state funnel's agent-start baseline, because attaching the observer IS
        the boundary where the model-controlled episode begins: everything before this line is
        host setup, everything after is the agent."""
        from codeaction.verification.verifiers import snapshot_milestone_baseline
        self.milestone_baseline = snapshot_milestone_baseline(
            self.env, vspec.get("milestones") or [],
            initial_poses=getattr(self, "initial_poses", None))
        from codeaction.runtime.step_observer import StepObserver
        from codeaction.runtime.episode_recorder import EpisodeRecorder
        self.observer = StepObserver(
            self.env,
            physical_time_budget_s=self.args.physical_time_budget_s,
            expert_sim_duration_s=self.args.expert_sim_duration_s,
        )
        latch_spec = dict(vspec.get("latch") or {})
        if latch_spec:
            from codeaction.verification.verifiers import LatchMonitor, select_latch_events
            # Diagnostic events are OPT-IN; required ones always run because their verifier
            # declaration fails closed without monitor state.
            events = select_latch_events(
                latch_spec, diagnostics=self.args.in_episode_diagnostics)
            if events:
                self.latch_monitor = LatchMonitor(
                    self.env, events, poll_every=int(latch_spec.get("poll_every", 5)),
                    initial_poses=getattr(self, "initial_poses", None))
                self.observer.add("latch", self.latch_monitor.poll)
                _log(f"latch monitor attached: {len(self.latch_monitor.specs)} event(s), "
                     f"poll_every={self.latch_monitor.poll_every}, "
                     f"diagnostics={self.args.in_episode_diagnostics}")
        try:
            self.recorder = EpisodeRecorder(self.env, self.out).start()
            self.observer.add("recorder", self.recorder.on_step)
            _log("episode recorder attached: head_camera every=10 timeline=sim_time")
        except Exception as e:
            self.video_error = f"{type(e).__name__}: {e}"
            _log(f"episode recorder disabled: {self.video_error}")
        # Attach even if recording failed: the transcript still gets a valid physics-step clock.
        self.observer.attach()
        # The tools read the SAME clock, so "did this call execute anything" is answered by the
        # simulator rather than by a wrapper's return.
        toolbox.attach_sim_step_source(lambda: self.observer.step_count)
        toolbox.attach_sim_state_source(self.observer.state)
        toolbox.attach_episode_terminal_check(self.observer.raise_if_exhausted)

    def _finish_episode_observers(self):
        """Detach callbacks and close ffmpeg once; safe from persist(), boot failure, or shutdown."""
        observer_state = None
        if self.observer is not None:
            self.observer.detach()
            observer_state = self.observer.state()
        if not self._recording_finalized:
            self._recording_finalized = True
            if self.recorder is not None:
                try:
                    self.video_meta = self.recorder.finalize()
                except Exception as e:
                    self.video_error = f"{type(e).__name__}: {e}"
                    _log(f"recorder finalize failed: {self.video_error}")
        return observer_state

    def dispatch(self, name, arguments, admitted_generation=None):
        """Runs on the sim worker thread (queued behind boot)."""
        if self.boot_error is not None:
            return [{"type": "text",
                     "text": json.dumps({"error": f"scene boot failed: {self.boot_error}"})}]
        if admitted_generation is not None \
                and int(admitted_generation) != self.abort_generation():
            return self.bridge.cancel_after_recoverable_abort(name, arguments)
        contents = self.bridge.dispatch(name, arguments)
        try:
            from codeaction.contracts.failures import is_recoverable_action_abort
            first = contents[0] if contents else {}
            payload = json.loads(first.get("text", "{}")) if first.get("type") == "text" else {}
        except Exception:
            payload = {}
        if is_recoverable_action_abort(payload):
            self._advance_abort_generation()
        return contents

    def reference_ready(self):
        """Transport-only readiness barrier; invocation is queued behind ``boot``."""
        if self.args.interface_profile not in _REFERENCE_PROFILES:
            raise ValueError("reference readiness control is unavailable for this profile")
        if self.boot_error is not None or self.bridge is None:
            raise RuntimeError(f"scene boot failed: {self.boot_error or 'bridge unavailable'}")
        return {"ok": True, "status": self.bridge.status}

    def reference_finalize(self, arguments):
        """Finalize a reference attempt whose model loop ended without a model-issued ``done``."""
        if self.args.interface_profile not in _REFERENCE_PROFILES or self.bridge is None:
            raise ValueError("reference finalize control is unavailable for this profile")
        from codeaction.contracts.failures import failure_from_dict
        arguments = dict(arguments or {})
        if set(arguments) != {"status", "failure", "total_calls", "budget_used"}:
            raise ValueError("reference finalize control has invalid fields")
        from codeaction.runtime.mcp_control import REFERENCE_FINALIZE_STATUSES
        status = str(arguments["status"])
        if status not in REFERENCE_FINALIZE_STATUSES:
            raise ValueError(f"reference finalize status is unsupported: {status}")
        failure_value = arguments.get("failure")
        failure = failure_from_dict(failure_value) if failure_value is not None else None
        if failure_value is not None and failure is None:
            raise ValueError("reference finalize failure record is invalid")
        total_calls = arguments.get("total_calls")
        if isinstance(total_calls, bool) or not isinstance(total_calls, int) \
                or total_calls < 0:
            raise ValueError("reference finalize total_calls is invalid")
        budget_used = arguments.get("budget_used")
        if isinstance(budget_used, bool) or not isinstance(budget_used, int) \
                or budget_used < 0:
            raise ValueError("reference finalize budget_used is invalid")
        self.bridge.finalize(status, failure=failure, total_calls=total_calls,
                             budget_used=budget_used)
        return {"ok": True, "status": self.bridge.status}

    def reference_wall_credit(self, arguments):
        """Credit provider-outage seconds back to this attempt's wall budget."""
        if self.args.interface_profile not in _REFERENCE_PROFILES or self.bridge is None:
            raise ValueError("reference wall-credit control is unavailable for this profile")
        arguments = dict(arguments or {})
        if set(arguments) != {"seconds"}:
            raise ValueError("reference wall-credit control has invalid fields")
        seconds = arguments["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
                or not seconds > 0 or seconds != seconds or seconds in (float("inf"),):
            raise ValueError("reference wall-credit seconds must be a positive finite number")
        applied = self.bridge.credit_wall_budget(float(seconds))
        return {"ok": True, "applied_s": applied}

    def reference_malformed(self, arguments):
        """Dispatch a model call whose argument string failed JSON decoding before MCP."""
        if self.args.interface_profile not in _REFERENCE_PROFILES or self.bridge is None:
            raise ValueError("reference malformed control is unavailable for this profile")
        from codeaction.runtime.mcp_control import MALFORMED_ARGUMENTS_MARKER
        arguments = dict(arguments or {})
        if set(arguments) != {"tool"} or not isinstance(arguments.get("tool"), str):
            raise ValueError("reference malformed control has invalid fields")
        tool = arguments["tool"]
        if tool not in self.bridge.tool_names:
            raise ValueError("reference malformed control named an unavailable tool")
        return self.bridge.dispatch(tool, {MALFORMED_ARGUMENTS_MARKER: True})

    def reference_cancelled(self, arguments):
        """Mirror one reference-side same-turn cancellation without executing its public tool."""
        if self.args.interface_profile not in _REFERENCE_PROFILES or self.bridge is None:
            raise ValueError("reference cancelled-call control is unavailable for this profile")
        arguments = dict(arguments or {})
        if set(arguments) != {"tool"} or not isinstance(arguments.get("tool"), str):
            raise ValueError("reference cancelled-call control has invalid fields")
        tool = arguments["tool"]
        if tool not in self.bridge.tool_names:
            raise ValueError("reference cancelled-call control named an unavailable tool")
        self.bridge.cancel_after_recoverable_abort(tool, {})
        return {"ok": True, "executed": False}

    def persist(self, status, done_report):
        """on_finalize: the ONLY place GT is read. Verdict goes to disk, never to the agent."""
        if self._persisted:
            return
        self._persisted = True
        from codeaction.contracts.failures import FailureCode, default_failure
        from codeaction.verification.verifiers import (run_task_verifier, check_destructive,
                                       check_destructive_actors, check_integrity_actors,
                                       enforce_destructive_failure)
        failure = self.bridge.failure
        observer_state = None
        try:
            # Stop polling before the verifier reads end state, then hand over what latched during
            # the episode. Detaching first keeps finalize free of monitor side effects.
            observer_state = self._finish_episode_observers()
            latch_state = self.latch_monitor.state() if self.latch_monitor is not None else None
            ver = run_task_verifier(CARD["verifier"], env=self.env, target=self.target,
                                    z0=self.z0, initial_poses=getattr(self, "initial_poses", None),
                                    latch_state=latch_state,
                                    milestone_baseline=getattr(self, "milestone_baseline", None))
            if self.target is not None:            # legacy/custom single-target scene
                ver.update(check_destructive(self.env, self.target, self.table_z))
            elif CARD["verifier"].get("destructive_actors"):   # env-loader multi-actor tasks
                dspec = CARD["verifier"].get("destructive_check") or {}
                ver.update(check_destructive_actors(
                    self.env, CARD["verifier"]["destructive_actors"], self.table_z,
                    xy_bounds=dspec.get("xy_bounds"),
                    initial_poses=getattr(self, "initial_poses", None),
                    min_z_m=dspec.get("min_z_m")))
            if CARD["verifier"].get("integrity_actors"):       # posture axis (R5, analysis-only)
                ver.update(check_integrity_actors(
                    self.env, CARD["verifier"]["integrity_actors"],
                    getattr(self, "initial_quats", None) or {}))
            ver = enforce_destructive_failure(ver)
        except Exception as exc:
            failure = default_failure(
                FailureCode.VERIFIER_ERROR,
                detail_safe=f"{type(exc).__name__} while finalizing verifier",
            )
            ver = {"success": False, "score": 0.0, "kind": "verifier_error"}
            _log(f"verifier failed: {failure.detail_safe}")
        result_interface = (
            "reference-agent"
            if self.args.interface_profile in _REFERENCE_PROFILES else "mcp-agent")
        out = {"stats": self.bridge.stats(), "verifier": ver, "task": TASK,
               "failure": failure.to_dict() if failure is not None else None,
               "task_name": CARD["task"]["name"] if CARD else None,
               "model": self.args.agent_label, "interface": result_interface,
               "tool_surface": self.args.expected_tool_surface,
               "instruction_surface": self.args.expected_instruction_surface,
               "recording": {
                   "enabled": self.recorder is not None, "camera": "head_camera",
                   "every_steps": 10, "timeline": "sim_time", "video_meta": self.video_meta,
                   **({"error": self.video_error} if self.video_error else {}),
               }}
        if self.provenance and self.provenance.get("schema_version") == "0.2":
            out["identity"] = self.provenance["expected_identity"]
            out["identity_attestation"] = {
                "controller_expected": {
                    "tool_surface": self.provenance["expected_identity"][
                        "comparison"]["tool_surface"],
                    "instruction_surface": self.provenance["expected_identity"][
                        "comparison"]["instruction_surface"],
                },
                "runtime_observed": {
                    "episode_server_tool_surface": self.args.expected_tool_surface,
                    "instruction_surface": self.args.expected_instruction_surface,
                },
                "episode_server_match": True,
                "gateway_attestation": "gateway_attestation.json",
            }
        if observer_state is not None:      # a disabled callback silently stops latching — forensics
            out["step_observer"] = observer_state
        (self.out / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        _log(f"finalized: status={status} verifier={json.dumps(ver)}")
        try:
            from codeaction.reporting.reports import finalize_run
            report_meta = dict(self.provenance or {})
            report_meta["recording"] = out["recording"]
            if "identity" in out:
                report_meta["identity"] = out["identity"]
                report_meta["identity_attestation"] = out["identity_attestation"]
            rebuilt = finalize_run(
                self.out,
                name=(f"{CARD['task']['name']} · {self.args.agent_label} · "
                      f"{result_interface}"),
                tag=(f"{'reference-scaffold' if result_interface == 'reference-agent' else 'vendor-agent'}"
                     f" · {self.out.name}"),
                verifier=ver, interface=result_interface,
                extra_meta=report_meta)
            _log("report ingested" if rebuilt else
                 "report metadata recorded; rebuild deferred to the host (read-only worktree)")
        except Exception as e:
            _log(f"report ingest skipped: {e}")

    def wedge_exit(self, tool_name):
        """A tool call blew past TOOL_WEDGE_S: the sim worker thread is wedged (cannot be safely
        interrupted), so record honest forensics WITHOUT touching env/GT from this thread and kill
        the whole process — freeing the GPU and closing the MCP channel deterministically."""
        try:
            from codeaction.contracts.failures import FailureCode, default_failure
            failure = default_failure(
                FailureCode.TOOL_WEDGE,
                detail_safe=f"tool={tool_name}; ceiling_s={TOOL_WEDGE_S:g}",
            )
            stats = self.bridge.stats() if self.bridge is not None else {"steps": None}
            stats = {**stats, "failure": failure.to_dict()}
            out = {"stats": stats, "verifier": {"success": False, "score": 0.0,
                                                "kind": "tool_wedge_unscored"},
                   "status": "tool_wedge", "wedged_tool": tool_name, "task": TASK,
                   "failure": failure.to_dict(),
                   "task_name": CARD["task"]["name"] if CARD else None,
                   "model": self.args.agent_label, "interface": "mcp-agent"}
            (self.out / "result.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            _log(f"TOOL WEDGE: {tool_name!r} exceeded {TOOL_WEDGE_S}s — exiting hard "
                 f"(result.json written; sim worker unrecoverable)")
        finally:
            os._exit(2)

    def shutdown(self):
        """Agent disconnected (or server exiting): close out an unfinished episode, then the sim."""
        if self.bridge is not None:
            self.bridge.finalize("no_done")        # idempotent; no-op after done/budget/fatal
        elif self.boot_error is None:
            _log("shutdown before boot completed")
        self._finish_episode_observers()           # persist() normally finalized already
        try:
            if self.env is not None:
                self.env.close_env()
        except Exception:
            pass


async def serve(raw_out, host, exec_pool, tool_defs):
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    server = Server("codeaction")

    @server.list_tools()
    async def _list_tools():
        return [types.Tool(name=d["name"], description=d["description"],
                           inputSchema=d["inputSchema"], _meta=d.get("_meta"))
                for d in tool_defs]

    @server.call_tool()
    async def _call_tool(name, arguments):
        import asyncio
        from codeaction.runtime.mcp_control import (
            REFERENCE_CANCELLED_TOOL,
            REFERENCE_FINALIZE_TOOL,
            REFERENCE_MALFORMED_TOOL,
            REFERENCE_READY_TOOL,
            REFERENCE_WALL_CREDIT_TOOL,
        )
        loop = asyncio.get_running_loop()
        if name in (
                REFERENCE_CANCELLED_TOOL,
                REFERENCE_READY_TOOL, REFERENCE_FINALIZE_TOOL,
                REFERENCE_MALFORMED_TOOL, REFERENCE_WALL_CREDIT_TOOL):
            if host.args.interface_profile not in _REFERENCE_PROFILES:
                raise ValueError("reference transport control is unavailable")
            if name == REFERENCE_CANCELLED_TOOL:
                callback = lambda: host.reference_cancelled(arguments)
            elif name == REFERENCE_READY_TOOL:
                callback = host.reference_ready
            elif name == REFERENCE_FINALIZE_TOOL:
                callback = lambda: host.reference_finalize(arguments)
            elif name == REFERENCE_WALL_CREDIT_TOOL:
                callback = lambda: host.reference_wall_credit(arguments)
            else:
                callback = lambda: host.reference_malformed(arguments)
            payload = await asyncio.wait_for(
                loop.run_in_executor(exec_pool, callback), timeout=TOOL_WEDGE_S)
            if name == REFERENCE_MALFORMED_TOOL:
                return [
                    types.ImageContent(
                        type="image", data=item["data"], mimeType=item["mimeType"])
                    if item["type"] == "image"
                    else types.TextContent(type="text", text=item["text"])
                    for item in payload
                ]
            return [types.TextContent(
                type="text", text=json.dumps(payload, separators=(",", ":")))]
        try:
            admitted_generation = host.abort_generation()
            contents = await asyncio.wait_for(
                loop.run_in_executor(
                    exec_pool, host.dispatch, name, arguments, admitted_generation),
                timeout=TOOL_WEDGE_S)
        except asyncio.TimeoutError:
            host.wedge_exit(name)              # writes result.json + os._exit(2); never returns
        out = []
        for c in contents:
            if c["type"] == "image":
                out.append(types.ImageContent(type="image", data=c["data"],
                                              mimeType=c["mimeType"]))
            else:
                out.append(types.TextContent(type="text", text=c["text"]))
        return out

    text_out = io.TextIOWrapper(raw_out, encoding="utf-8", line_buffering=True)
    async with stdio_server(stdout=anyio.wrap_file(text_out)) as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click_bell", help="task package name in the task pack")
    ap.add_argument("--task-pack", default=None,
                    help="task pack root; defaults to the committed in-repository pack")
    ap.add_argument("--out", default="data/vendor_stage3",
                    help="run dir, relative to .")
    ap.add_argument("--agent-label", default="claude-code")
    # The seat's tool-discovery mode selects the vendor prompt's discovery sentence, so the
    # episode server must recompute the expected instruction surface with the same value the
    # controller rendered with. The first Codex batch failed every cell at this preflight
    # because the server recomputed with the default while the controller had rendered eager.
    ap.add_argument("--tool-discovery", default="deferred_toolsearch",
                    choices=("deferred_toolsearch", "eager_all"))
    ap.add_argument("--agent-cli-version", default=None,
                    help="vendor CLI version string captured by the driver; the agent half of the "
                         "(agent, model) pair. Recorded in the transcript scaffold card.")
    ap.add_argument("--seed", type=int, default=0,
                    help="attempt/layout seed for the generic env loader (varies per vendor-agent episode)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="dev-only override of the task card's max_tool_calls")
    ap.add_argument("--wall-budget", type=float, default=None,
                    help="dev-only override of the task card's wall_budget_s")
    ap.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                    help="disable run_code (hybrid is the default: the leaderboard interface)")
    ap.add_argument("--in-episode-diagnostics", action="store_true",
                    help="also poll the card's DIAGNOSTIC latch events (default: only the "
                         "required ones). Diagnostics never affect the verdict, so this is safe "
                         "to toggle between runs.")
    ap.add_argument("--interface-profile", default="vendor-mcp-direct",
                    choices=("reference-mcp", "reference-code-first", "vendor-mcp-direct"),
                    help="episode-server profile; gateway extras are applied outside this process")
    ap.add_argument("--provenance-file", default=None,
                    help="controller JSON inside the attempt directory; validated before boot")
    args = ap.parse_args()

    global TASK, CARD
    from codeaction.benchmark.taskcard import (TASKS_ROOT, instruction_for_scene_seed, load_task,
                                  validate_task_pack)
    task_pack = Path(args.task_pack).resolve() if args.task_pack else TASKS_ROOT.resolve()
    pack_info = validate_task_pack(task_pack)
    if args.task not in pack_info["tasks"]:
        raise ValueError(f"task {args.task!r} is not registered in the selected task pack")
    # Runnability only: pin drift (a modified tool set or instruction contract) is an
    # eligibility question the controller answers; the episode server must still boot and score.
    _drift: dict = {}
    CARD = load_task(args.task, tasks_root=task_pack, strict_pins=False,
                     drift_sink=_drift)  # canary + leak audit inside
    if _drift:
        _log(f"pin drift (run continues, controller decides eligibility): {sorted(_drift)}")
    TASK = instruction_for_scene_seed(CARD, args.seed)
    b = CARD.get("budgets", {})
    if args.max_steps is None:
        args.max_steps = int(b["max_tool_calls"])
    if args.wall_budget is None:
        args.wall_budget = float(b["wall_budget_s"])
    args.physical_time_budget_s = float(b["physical_time_budget_s"])
    args.expert_sim_duration_s = float(CARD["metadata"]["expert_sim_duration_s"])
    args.max_tool_calls = int(args.max_steps)
    args.run_code_max_internal_calls = int(b["run_code_max_internal_calls"])
    from codeaction.contracts.harness_parameters import (EPISODE_TABLE_WORLD_DISABLED,
                                             declared_harness_parameters)
    from codeaction.interface.instructions import reference_instruction_surface, vendor_instruction_surface
    from codeaction.interface.tool_surface import surface_identity
    from codeaction.interface.tools import orientation_anchor_enabled
    args.expected_tool_surface = surface_identity(
        args.interface_profile, hybrid=args.hybrid)
    is_reference_profile = args.interface_profile in _REFERENCE_PROFILES
    instruction_surface_builder = (
        reference_instruction_surface if is_reference_profile else vendor_instruction_surface)
    surface_kwargs = {} if is_reference_profile else {"tool_discovery": args.tool_discovery}
    instruction_surface = instruction_surface_builder(
        task_text=TASK,
        max_tool_calls=args.max_tool_calls,
        physical_time_budget_s=args.physical_time_budget_s,
        run_code_max_internal_calls=args.run_code_max_internal_calls,
        harness_parameters=declared_harness_parameters(
            orientation_anchor=orientation_anchor_enabled(),
            table_world_disabled=EPISODE_TABLE_WORLD_DISABLED),
        **surface_kwargs,
    )
    args.expected_instruction_surface = {
        key: instruction_surface[key] for key in (
            "instruction_contract_sha256", "instruction_surface_sha256",
            "fragment_ids", "fragment_manifest")
    }

    os.chdir(str(_RT))          # RoboTwin loaders use repo-root-relative asset paths
    requested_out = Path(args.out)
    out_dir = requested_out if requested_out.is_absolute() else PROJECT_ROOT / requested_out
    out_dir.mkdir(parents=True, exist_ok=True)
    args.provenance = None
    if args.provenance_file:
        from codeaction.evidence.provenance import load_provenance
        try:
            args.provenance = load_provenance(args.provenance_file, out_dir)
        except ValueError as exc:
            print(f"[mcp_host] invalid provenance: {exc}", file=sys.stderr)
            return 2
        if (args.provenance["task_pack_version"], args.provenance["task_pack_sha256"]) != (
                pack_info["taskset_version"], pack_info["sha256"]):
            _write_preflight_failure(
                out_dir, "task pack disagrees with controller provenance")
            print("[mcp_host] task pack disagrees with controller provenance", file=sys.stderr)
            return 2
        if args.provenance.get("schema_version") != "0.2":
            _write_preflight_failure(out_dir, "container run requires provenance 0.2")
            print("[mcp_host] container run requires provenance 0.2",
                  file=sys.stderr)
            return 2
        from codeaction.contracts.identity import budget_identity, task_card_identity
        controller_identity = args.provenance["expected_identity"]
        comparison = controller_identity["comparison"]
        controller_profile = args.provenance["interface_profile"]
        expected_final_surface = surface_identity(
            controller_profile, hybrid=args.hybrid)
        checks = {
            "task": task_card_identity(CARD["dir"]),
            "task_pack": {
                "id": "robotwin-codeaction",
                "version": pack_info["taskset_version"],
                "sha256": pack_info["sha256"],
            },
            "tool_surface": expected_final_surface,
            "instruction_surface": args.expected_instruction_surface,
            "budgets": budget_identity(CARD["budgets"]),
        }
        mismatches = [
            key for key, observed in checks.items()
            if comparison.get(key) != observed
        ]
        if controller_identity["trial"].get("scene_seed") != args.seed:
            mismatches.append("trial.scene_seed")
        if comparison.get("tested_unit", {}).get("interface_profile") != controller_profile:
            mismatches.append("tested_unit.interface_profile")
        if mismatches:
            _write_preflight_failure(
                out_dir, "controller/runtime identity mismatch: " + ",".join(mismatches))
            print(f"[mcp_host] identity preflight mismatch: {mismatches}", file=sys.stderr)
            return 2
    # ONE episode per attempt dir (protocol enforcement, found 2026-07-09): after a finalize, a
    # reconnecting agent CLI would otherwise get a FRESH scene retry with retained knowledge —
    # vendor_grab_roller/s3 recorded ep1 budget_exhausted, then a CC auto-reconnect booted a
    # second scene and its ep2 success overwrote result.json. Refuse to serve a second episode.
    if (out_dir / "result.json").exists():
        print(f"[mcp_host] result.json already exists in {out_dir}; one episode per attempt dir — "
              f"refusing to serve another", file=sys.stderr)
        return 3
    raw_out = _secure_stdio(out_dir / "host.log")
    _log(f"start task={CARD['task']['name']} (schema {CARD['schema_version']}) out={out_dir} "
         f"hybrid={args.hybrid} agent={args.agent_label} "
         f"budgets={args.max_tool_calls} calls/{args.wall_budget}s")

    # static audits BEFORE serving anything (sim-free)
    from codeaction.interface.registry import D0_TOOLS
    from codeaction.evidence.leak_audit import audit_registry, audit_instruction_text
    from codeaction.interface.tool_surface import mcp_server_definitions
    ia = audit_instruction_text(TASK)
    ra = audit_registry(D0_TOOLS)
    if not (ia["clean"] and ra["clean"]):
        _log(f"AUDIT FAILED instruction={ia} registry={ra}")
        return 2
    tool_defs = mcp_server_definitions(args.interface_profile, hybrid=args.hybrid)
    _log(f"audits clean; {len(tool_defs)} tools: {[d['name'] for d in tool_defs]}")

    host = Host(args, out_dir)
    exec_pool = ThreadPoolExecutor(max_workers=1)   # ONE thread owns the sim (FIFO)
    exec_pool.submit(host.boot)                      # job #1; tool calls queue behind it

    import asyncio
    try:
        asyncio.run(serve(raw_out, host, exec_pool, tool_defs))
    except Exception as e:
        _log(f"server loop ended with error: {e}")
    finally:
        exec_pool.submit(host.shutdown).result(timeout=120)
        exec_pool.shutdown(wait=False)
        _log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
