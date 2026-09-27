"""MCP bridge for the vendor-agent (bring-your-own-agent) interface: one episode's tool surface served
to an external agent CLI (Claude Code, ...) over MCP, with the SAME single-source schemas
(TOOL_SPECS), the SAME dispatch/error taxonomy as the flat runner, and the SAME transcript format
(make_reports auto-ingests it).

Deliberately mcp-package-free: the bridge produces NEUTRAL content dicts
  {"type": "text", "text": <json>}  |  {"type": "image", "data": <b64>, "mimeType": "image/png"}
and the host (data/mcp_episode_server.py) maps them onto mcp.types objects. This keeps the bridge
locally testable (no mcp install needed) and inside the standard codeaction source leak audit.

GT wall: the bridge sees only a ToolBox registry + an optional run_code sandbox. The verifier is
NEVER called here and its verdict is NEVER returned to the agent — `on_finalize(status, report)` is
a host callback (host runs the out-of-band verifier there); the agent's `done` gets a neutral ack.

reference-scaffold/vendor-agent contract differences (declared, not hidden): frame retention and temperature are
agent-managed in the vendor agent (the CLI owns its context), so the transcript meta records them as such;
per-turn model text/latency are not observable server-side and are logged as empty."""
import base64
import json
import time
from pathlib import Path

from codeaction.benchmark import gt_probe
from codeaction.runtime.episode import EpisodeRuntime, REFERENCE_EPISODE_CONTRACT, VENDOR_EPISODE_CONTRACT
from codeaction.contracts.harness_parameters import declared_harness_parameters
from codeaction.interface.instructions import reference_instruction_surface, vendor_instruction_surface
from codeaction.runtime.mcp_control import MALFORMED_ARGUMENTS_MARKER
from codeaction.contracts.version import SCAFFOLD_NAME, SCAFFOLD_VERSION
from codeaction.interface.schemas import PROGRAM_TOOL_NAMES, TOOL_SPECS, to_json
from codeaction.contracts.tool_results import (
    MODEL_VISIBLE_RESULT_MAX_BYTES,
    RUN_CODE_RESULT_MAX_IMAGES,
    TOOL_RESULT_POLICY_ID,
    TOOL_RESULT_POLICY_VERSION,
)
from codeaction.interface.tool_surface import (
    DEFAULT_ALWAYS_LOAD_TOOLS,
    INTERFACE_REFERENCE,
    INTERFACE_VENDOR_DIRECT,
    REFERENCE_INTERFACE_PROFILES,
    assert_surface_preflight,
    resolve_interface_profile,
    surface_identity,
)
from codeaction.contracts.version import TRANSCRIPT_SCHEMA_VERSION, git_version

EPISODE_MAX_STEPS = 60          # outer calls; run_code internals are uncharged (v0.5)
EPISODE_WALL_S = 1800.0


def _projection_meta(projection):
    return {
        "policy": projection.policy,
        "truncated": projection.truncated,
        "original_bytes": projection.original_bytes,
        "model_bytes": projection.model_bytes,
        "model_image_count": len(projection.model_image_refs),
        "all_image_count": len(projection.all_image_refs),
        "unique_model_image_count": len(set(projection.model_image_refs)),
        "duplicate_observation_ids_suppressed": (
            projection.duplicate_observation_ids_suppressed),
    }

# Claude Code can defer MCP schemas. Keep the code-composition surface and the non-contact scale
# evidence visible from episode start; every other tool remains discoverable through ToolSearch.
def episode_tool_names(reg_names, hybrid, interface_profile=INTERFACE_VENDOR_DIRECT):
    """The episode tool surface, in declared order. Static: computable before the sim boots."""
    profile = resolve_interface_profile(interface_profile)
    composition = (["run_code", *profile.delivered_program_tools] if hybrid else [])
    return (composition
            + [n for n in reg_names if n in TOOL_SPECS] + ["done"])


def build_tool_defs(names, always_load=()):
    """Build MCP tool definitions from TOOL_SPECS, with optional Claude tool preloading."""
    always_load = set(always_load)
    listed = {n for n in names if n in TOOL_SPECS}
    missing = always_load - listed
    if missing:
        raise ValueError(f"always-load tools are not in this episode surface: {sorted(missing)}")
    out = []
    for n in names:
        if n not in TOOL_SPECS:
            continue
        desc, params = TOOL_SPECS[n]
        tool_def = {"name": n, "description": desc, "inputSchema": params}
        if n in always_load:
            tool_def["_meta"] = {"anthropic/alwaysLoad": True}
        out.append(tool_def)
    return out


def default_always_load_tools(names):
    """Return the default eager-load subset that actually exists in this episode surface."""
    listed = set(names)
    return [name for name in DEFAULT_ALWAYS_LOAD_TOOLS if name in listed]


def _text(payload) -> dict:
    return {"type": "text", "text": to_json(payload)}


def _image(png_path) -> dict:
    data = base64.b64encode(Path(png_path).read_bytes()).decode("ascii")
    return {"type": "image", "data": data, "mimeType": "image/png"}


class EpisodeBridge:
    """One attempt. dispatch(name, args) -> list of neutral content dicts; finalize() is idempotent
    and fires on_finalize exactly once (done | budget_exhausted | wall_budget | episode_fatal |
    unintended_collision | no_done — the host calls finalize('no_done') itself when the agent
    disconnects without done)."""

    def __init__(self, toolbox, task_text, out_dir, sandbox=None,
                 max_steps=EPISODE_MAX_STEPS, wall_budget_s=EPISODE_WALL_S,
                 physical_time_budget_s=900.0,
                 on_finalize=None, agent_label="mcp-agent", agent_cli_version=None,
                 sim_step_fn=None, interface_profile=INTERFACE_VENDOR_DIRECT,
                 expected_tool_surface=None, expected_instruction_surface=None,
                 tool_discovery="deferred_toolsearch",
                 contract_profile=None):
        self._toolbox = toolbox
        self._task = str(task_text)
        self._sandbox = sandbox
        self._max_steps = int(max_steps)
        self._wall_budget_s = float(wall_budget_s)
        self._physical_time_budget_s = float(physical_time_budget_s)
        self._on_finalize = on_finalize
        self._sim_step_fn = sim_step_fn
        self._pending_finalize_error = None
        self._active_sim_step_start = None
        run_code_limit = (getattr(self._sandbox, "max_tool_calls", 1)
                          if self._sandbox is not None else 1)
        # Both reference profiles are the reference-scaffold scaffold; they differ only in which primitives
        # are delivered as schemas. Testing equality against reference-mcp alone made the
        # code-first arm render vendor-agent instructions and its own control contract, so its
        # instruction surface disagreed with the controller's before the first turn.
        reference_profile = interface_profile in REFERENCE_INTERFACE_PROFILES
        instruction_builder = reference_instruction_surface if reference_profile else vendor_instruction_surface
        self._instruction_surface = instruction_builder(
            **({} if reference_profile else {"tool_discovery": tool_discovery}),
            task_text=self._task, max_tool_calls=self._max_steps,
            physical_time_budget_s=self._physical_time_budget_s,
            run_code_max_internal_calls=int(run_code_limit),
            harness_parameters=declared_harness_parameters(toolbox),
            composition_contract=(
                getattr(self._sandbox, "composition_contract", None)
                if self._sandbox is not None else None),
            include_program_workspace=bool(
                resolve_interface_profile(interface_profile).delivered_program_tools))
        contract_profile = contract_profile or (
            REFERENCE_EPISODE_CONTRACT if reference_profile else VENDOR_EPISODE_CONTRACT)
        self._runtime = EpisodeRuntime(
            toolbox,
            self._task,
            profile=contract_profile,
            sandbox=self._sandbox,
            max_tool_calls=self._max_steps,
            wall_budget_s=self._wall_budget_s,
            before_finalize=self._before_runtime_finalize,
            on_finalize=self._runtime_finalized,
        )
        self._surface_identity = None
        if expected_tool_surface is not None:
            self._surface_identity = surface_identity(
                interface_profile,
                hybrid=sandbox is not None,
                runtime_registry_names=self._runtime.registry,
            )
            assert_surface_preflight(expected_tool_surface, self._surface_identity)
            self.tool_names = list(self._surface_identity["ordered_names"])
        else:
            self.tool_names = episode_tool_names(
                list(self._runtime.registry), hybrid=sandbox is not None,
                interface_profile=interface_profile)
        # What we list over MCP is what the runtime will execute — nothing else in the registry.
        self._runtime.restrict_to(self.tool_names)
        if expected_instruction_surface is not None:
            observed = {
                key: self._instruction_surface[key] for key in (
                    "instruction_contract_sha256", "instruction_surface_sha256",
                    "fragment_ids", "fragment_manifest")
            }
            mismatch = [
                key for key, value in expected_instruction_surface.items()
                if observed.get(key) != value
            ]
            if mismatch:
                raise ValueError(f"instruction surface preflight mismatch: {mismatch}")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self._sandbox is not None:
            (out / "filesystem_audit.json").write_text(
                json.dumps(self._sandbox.filesystem_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        self._tpath = out / "transcript.jsonl"
        # Privileged, host-side only. Deliberately a SEPARATE file from the transcript so nothing
        # that assembles a model-facing payload can pick it up by accident.
        self._gtpath = out / "gt_snapshots.jsonl"
        # The server records only the scaffold identity it can attest. Reference model/context
        # events come from the isolated agent container and are joined by the controller.
        scaffold = (
            {
                "name": SCAFFOLD_NAME,
                "version": SCAFFOLD_VERSION,
                "config_source": "reference-agent-transcript",
                "bridge": "codeaction-mcp-bridge",
            }
            if reference_profile else
            {
                "name": "vendor-agent-cli",
                "agent_label": agent_label,
                "version": str(agent_cli_version or "unrecorded"),
                "temperature": "agent-default",
                "frame_retention": "agent-managed",
                "context_policy": "agent-managed",
                "bridge": "codeaction-mcp-bridge",
            }
        )
        self._log({"event": "meta", "schema_version": TRANSCRIPT_SCHEMA_VERSION,
                   "scaffold": scaffold,
                   "task": self._task, "tools": self.tool_names,
                   "frame_retention": (
                       "reference-agent-transcript" if reference_profile else "agent-managed"),
                   # Observed from the live ToolBox. This record is written where the ToolBox
                   # actually runs, which is why the parameter is reported here and not by the
                   # reference agent: that process is in a different container and can only
                   # report a default it cannot see.
                   "orientation_anchor": bool(
                       getattr(self._toolbox, "_orientation_anchor", True)),
                   "temperature": (
                       "reference-agent-transcript" if reference_profile else "agent-default"),
                   "model": agent_label, "interface": "mcp-agent",
                   "budget_unit": "tool_calls", "budget_visible": True,
                   "max_steps": self._max_steps,
                   "max_tool_calls": self._max_steps,
                   "budget_contract_version": contract_profile.contract_version,
                   "tool_surface": self._surface_identity,
                   "instruction_surface": {
                       key: self._instruction_surface[key] for key in (
                           "instruction_contract_sha256", "instruction_surface_sha256",
                           "fragment_ids", "fragment_manifest")
                   },
                   "run_code_max_calls": (
                       getattr(self._sandbox, "max_tool_calls", None)
                       if self._sandbox is not None else None),
                   "run_code_internal_charged": False,
                   "run_program_internal_charged": False,
                   "tool_result_policy": {
                       "id": TOOL_RESULT_POLICY_ID,
                       "version": TOOL_RESULT_POLICY_VERSION,
                       "max_bytes": MODEL_VISIBLE_RESULT_MAX_BYTES,
                   },
                   "image_policy": {
                       "run_code_result_max_images": RUN_CODE_RESULT_MAX_IMAGES,
                       "reference_context_max_image_rounds": (
                           "reference-agent-transcript"
                           if reference_profile else "agent-managed"),
                   },
                   "filesystem_policy": ("structurally_denied"
                                         if self._sandbox is not None else None),
                   "program_workspace_limits": (
                       getattr(self._sandbox, "program_limits", None)
                       if self._sandbox is not None else None),
                   "sim_step_clock": "physics_steps" if sim_step_fn is not None else None,
                   **git_version(__file__)})

    # -- bookkeeping -----------------------------------------------------------------------------
    def _log(self, rec):
        with open(self._tpath, "a", encoding="utf-8") as fh:
            fh.write(to_json(rec) + "\n")

    def _record_gt_snapshot(self, step, tool):
        """Append one privileged scene snapshot. Analysis-only; must never affect the episode.

        Enabled by default because it costs nothing the model can observe and the alternative is
        re-running paid batches to get attribution. Every failure is swallowed: a diagnostic that
        can break dispatch is worse than no diagnostic.
        """
        if self._gtpath is None:
            return
        try:
            env = getattr(self._toolbox, "_env", None)
            if env is None:
                return
            record = gt_probe.snapshot(
                env, tick=getattr(self._toolbox, "_tick", None), step=step, tool=tool)
            with open(self._gtpath, "a", encoding="utf-8") as fh:
                fh.write(to_json(record) + "\n")
        except Exception:
            pass

    def _sim_step(self):
        """Read host-only physics telemetry; a broken clock must never affect dispatch."""
        if self._sim_step_fn is None:
            return None
        try:
            return int(self._sim_step_fn())
        except Exception:
            return None

    @property
    def over(self) -> bool:
        return self._runtime.over

    @property
    def steps(self):
        return self._runtime.steps

    @property
    def status(self):
        return self._runtime.status

    @property
    def done_report(self):
        return self._runtime.done_report

    @property
    def failure(self):
        return self._runtime.failure

    def stats(self) -> dict:
        return {"status": self.status, "done_report": self.done_report,
                "steps": self._runtime.budget_used,
                "budget_used": self._runtime.budget_used,
                "total_calls": self._runtime.total_calls,
                "tool_calls_used": self._runtime.budget_used,
                "tool_call_budget": self._runtime.max_tool_calls,
                "total_tool_dispatches": self._runtime.total_calls,
                # Signed difference adopted from the reference agent at finalize. Without it the
                # counters agree by construction and the disagreement this reconciliation exists
                # to tolerate leaves no trace in the released artifact.
                "reference_only_calls": self._runtime.reference_only_calls,
                "reference_only_budget": self._runtime.reference_only_budget,
                "infra_retries": 0, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "wall_s": self._runtime.wall_s, "transcript": str(self._tpath),
                "failure": self.failure.to_dict() if self.failure is not None else None}

    def _before_runtime_finalize(self, status, context):
        call = context.get("call")
        if call is not None and call.interrupted_partial:
            # Server-side only: the model's terminal payload is the terminal, unchanged. This row
            # is what keeps the primitives an interrupted block already ran attributable.
            self._log({"event": "interrupted_block", "step": call.step, "tool": call.name,
                       "status": status, **call.interrupted_partial,
                       "sim_step_start": self._active_sim_step_start,
                       "sim_step_end": self._sim_step()})
        if status == "done" and call is not None:
            self._log({"event": "done", "step": call.step, "report": self.done_report,
                       "charged": call.charged,
                       "budget_used": self._runtime.budget_used,
                       "total_calls": self._runtime.total_calls,
                       "sim_step": self._sim_step(), "dropped": []})
        elif status == "episode_fatal" and call is not None:
            self._pending_finalize_error = call.error_detail
            self._log({"event": "episode_fatal", "step": call.step, "tool": call.name,
                       "error": call.error_detail, "failure": call.failure.to_dict(),
                       "charged": call.charged,
                       "budget_used": self._runtime.budget_used,
                       "total_calls": self._runtime.total_calls,
                       "sim_step_start": self._active_sim_step_start,
                       "sim_step_end": self._sim_step()})
        elif status == "control_call_wedge" and call is not None:
            self._log({"event": "control_call_wedge", "step": call.step,
                       "tool": call.name, "failure": call.failure.to_dict(),
                       "charged": call.charged,
                       "budget_used": self._runtime.budget_used,
                       "total_calls": self._runtime.total_calls,
                       "sim_step_start": self._active_sim_step_start,
                       "sim_step_end": self._sim_step()})
        elif status == "unintended_collision" and call is not None:
            self._pending_finalize_error = call.error_detail
            self._log({"event": "unintended_collision", "step": call.step,
                       "tool": call.name, "error": call.error_detail,
                       "args": call.args,
                       "result": call.projection.full_payload,
                       "model_result": call.payload,
                       "result_projection": _projection_meta(call.projection),
                       "failure": call.failure.to_dict(),
                       "charged": call.charged,
                       "budget_used": self._runtime.budget_used,
                       "total_calls": self._runtime.total_calls,
                       "sim_step_start": self._active_sim_step_start,
                       "sim_step_end": self._sim_step()})

    def _runtime_finalized(self, status, done_report):
        error = self._pending_finalize_error
        self._log({"event": "end", **self.stats(), **({"error": str(error)} if error else {})})
        if self._on_finalize is not None:
            try:
                self._on_finalize(status, done_report)
            except Exception as exc:                 # host failure must not crash the tool channel
                self._log({"event": "finalize_error", "error": str(exc)})

    def finalize(self, status, error=None, failure=None, total_calls=None, budget_used=None):
        """End the episode once. Fires the host callback (verifier + persistence live THERE)."""
        self._pending_finalize_error = error
        if total_calls is not None:
            self._runtime.reconcile_external_total_calls(total_calls)
        if budget_used is not None:
            self._runtime.reconcile_external_budget_used(budget_used)
        return self._runtime.finalize(status, failure=failure)

    def credit_wall_budget(self, seconds) -> float:
        """Return provider-outage seconds to the sim-side wall clock. See mcp_control."""
        return self._runtime.credit_wall_budget(seconds)

    # -- dispatch --------------------------------------------------------------------------------
    def dispatch(self, name, args) -> list:
        t_call = time.time()
        args = dict(args or {})
        malformed_arguments = args == {MALFORMED_ARGUMENTS_MARKER: True}
        runtime_args = None if malformed_arguments else args
        sim_step_start = self._sim_step()
        self._active_sim_step_start = sim_step_start
        call = self._runtime.dispatch(
            name, runtime_args, malformed_arguments=malformed_arguments)
        # Out-of-band scene truth for offline attribution. Taken AFTER dispatch so it describes the
        # state the call produced, written to its own file, and never merged into `call` -- the
        # model has no filesystem and no path to it. See codeaction.benchmark.gt_probe.
        self._record_gt_snapshot(call.step, name)

        if call.kind in ("tool", "recoverable_abort"):
            rec = {"event": "tool", "step": call.step, "tool": name,
                   "args": None if malformed_arguments else args,
                   "result": call.projection.full_payload,
                   "model_result": call.payload,
                   "result_projection": _projection_meta(call.projection),
                   "dropped": [], "usage": {}, "model_text": "",
                   "server_latency_s": round(time.time() - t_call, 2),
                   "charged": call.charged,
                   "budget_used": self._runtime.budget_used,
                   "total_calls": self._runtime.total_calls,
                   "sim_step_start": sim_step_start, "sim_step_end": self._sim_step()}
            if call.failure is not None:
                rec["failure"] = call.failure.to_dict()
            self._log(rec)

        contents = [_text(call.payload)]
        for ref in call.projection.model_image_refs:
            try:
                contents.append(_image(ref))
            except Exception:
                pass
        return contents

    def cancel_after_recoverable_abort(self, name, args) -> list:
        """Return and record a queued same-turn call without dispatching it to ToolBox."""
        call = self._runtime.cancel_after_recoverable_abort(name, dict(args or {}))
        self._log({
            "event": "tool_cancelled_after_action_abort",
            "step": call.step,
            "tool": str(name),
            "args": dict(args or {}),
            "result": call.projection.full_payload,
            "model_result": call.payload,
            "result_projection": _projection_meta(call.projection),
            "dropped": [],
            "usage": {},
            "model_text": "",
            "server_latency_s": 0.0,
            "charged": call.charged,
            "budget_used": self._runtime.budget_used,
            "total_calls": self._runtime.total_calls,
            "sim_step_start": self._sim_step(),
            "sim_step_end": self._sim_step(),
        })
        return [_text(call.payload)]
