"""Host-only forensic record of what one motion call did INTERNALLY. Never agent-visible.

Same standing as `codeaction.gt_probe` and the `observer/` frames: privileged, analysis-only, written
to a file the model has no path to (the tool surface exposes no filesystem, audited).

Why it is host-only rather than part of the ActionResult. The audit is right that the harness must
not destroy control facts it produced — a chunk fallback, a bounded correction, or the leg whose
plan was refused all really happened. But "must not be destroyed" is not "must be handed to the
model": the agent's next decision needs where the TCP is (`observed_after`), whether anything moved
(`execution.physics_steps` / `partial`) and why the planner refused (`achieved.planner_status`).
Which internal leg produced the shortfall changes none of those, and putting ~25 nested keys into
every motion result would enlarge the context to describe machinery the model cannot address.
Attribution is OUR job and happens offline, so the evidence goes where attribution happens.

Structural guarantee, not a promise: no action schema in `result_contracts` declares these keys and
every action schema is `additionalProperties: False`, so a record that ever leaked into a tool
result would fail the wire contract rather than reach a model.

Pure: builds the record, writes nothing. The caller appends it (ToolBox writes beside `observer/`,
the same way `mcp_bridge._record_gt_snapshot` persists the GT probe).
"""
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


# Ordered, closed vocabulary of the internal methods one call may use.
STAGE_KINDS = (
    "primary_plan", "chunk_leg", "correction", "waypoint_legs", "gripper_actuation",
)

# Why a leg loop stopped issuing commands.
STOP_CONDITIONS = (
    "converged", "contact_signature_changed", "leg_plan_refused", "leg_budget_exhausted",
    "stalled", "trajectory_deviation", "travel_budget_exhausted",
    "correction_refused", "correction_budget_exhausted", "actuation_failed",
    "no_tcp_read", "not_started",
)

TRACE_METADATA_KEYS = frozenset({
    "n_legs", "n_legs_attempted", "moved_m", "target_distance_remaining_m",
    "corrections_attempted", "corrections_executed",
})

TRACE_SCHEMA_VERSION = "1.2"


@dataclass(frozen=True)
class Stage:
    """One internal method the call actually used, in call order."""

    kind: str
    ok: Optional[bool]
    index: Optional[int] = None
    physics_steps: Optional[int] = None
    planner_status: Optional[Mapping[str, Any]] = None
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        if self.kind not in STAGE_KINDS:
            raise ValueError(
                f"unknown motion stage kind {self.kind!r}; declared kinds: {list(STAGE_KINDS)}")
        return {
            "kind": str(self.kind),
            "ok": (None if self.ok is None else bool(self.ok)),
            "index": (None if self.index is None else int(self.index)),
            "physics_steps": (None if self.physics_steps is None else int(self.physics_steps)),
            "planner_status": (dict(self.planner_status)
                               if self.planner_status is not None else None),
            "detail": (None if self.detail is None else str(self.detail)),
        }


@dataclass
class MotionTrace:
    """Collect one call's internal stages and per-leg measurements."""

    tool: str
    stages: List[Stage] = field(default_factory=list)
    legs: List[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    stop_condition: Optional[str] = None
    legs_budget: Optional[int] = None

    def stage(self, kind, ok, **fields) -> None:
        self.stages.append(Stage(kind=kind, ok=ok, **fields))

    def leg(self, index, plan_ok, diagnostics=None, *, executed=None) -> None:
        """One executed or refused leg, with the raw measurements the guard evaluated."""
        record = {
            "index": int(index),
            "plan_ok": (None if plan_ok is None else bool(plan_ok)),
            # Planner success and physical execution are independent facts: a planner may report
            # failure after dense actions have already advanced the simulator.
            "executed": (bool(plan_ok) if executed is None else bool(executed)),
        }
        for key, value in dict(diagnostics or {}).items():
            record[key] = value
        self.legs.append(record)

    def set_metadata(self, **values) -> None:
        """Store the primitive's already-measured bounded diagnostics for offline attribution."""
        unknown = set(values) - TRACE_METADATA_KEYS
        if unknown:
            raise ValueError(f"unknown motion trace metadata keys: {sorted(unknown)}")
        self.metadata.update(values)

    def stop(self, stop_condition, *, legs_budget=None) -> None:
        if stop_condition not in STOP_CONDITIONS:
            raise ValueError(
                f"unknown loop stop condition {stop_condition!r}; "
                f"declared conditions: {list(STOP_CONDITIONS)}")
        self.stop_condition = str(stop_condition)
        if legs_budget is not None:
            self.legs_budget = int(legs_budget)

    @property
    def failed_leg_index(self) -> Optional[int]:
        for leg in self.legs:
            if not leg.get("plan_ok") or leg.get("deviated"):
                return int(leg["index"])
        return None

    def record(self, *, action_id, tick, status, physics_steps, state_changed) -> dict:
        """The line appended to the host-only trace file."""
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "tool": str(self.tool),
            "action_id": action_id,
            "tick": tick,
            "status": status,
            "physics_steps": physics_steps,
            "state_changed": state_changed,
            "metadata": dict(self.metadata),
            "stop_condition": self.stop_condition,
            "legs_budget": self.legs_budget,
            "legs_attempted": len(self.legs),
            # A leg whose plan was refused executed nothing; a leg that deviated DID execute --
            # that is how the deviation was measured -- so it counts here and still sets
            # failed_leg_index.
            "legs_executed": sum(1 for leg in self.legs if leg.get("executed")),
            "failed_leg_index": self.failed_leg_index,
            "stages": [stage.to_dict() for stage in self.stages],
            "legs": [dict(leg) for leg in self.legs],
        }


__all__ = ["MotionTrace", "Stage", "STAGE_KINDS", "STOP_CONDITIONS",
           "TRACE_METADATA_KEYS", "TRACE_SCHEMA_VERSION"]
