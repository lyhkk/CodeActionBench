"""Typed, provenance-carrying return objects for every benchmark tool (spec §4). Pure: no sim, no VLM.
These make the loop replayable and prevent silently mixing pixels from different frames/cameras."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_VALID_STATUS = {"SUCCESS", "FAILED", "ABORTED"}
_VALID_KIND = {"depth", "point", "scale", "ray"}


class FrameMismatch(ValueError):
    """Raised when two observations that must share a frame do not (wrong camera / tick)."""


@dataclass(frozen=True)
class Observation:
    obs_id: str
    camera: str
    image_ref: str
    # Closed call-time camera record: K/E/size, freshness and the projection convention.
    cam_pose_snapshot: Dict[str, Any]
    tick: int
    annotation: Optional[Dict[str, Any]] = None   # drawing-tool payload (marks, measured overlay
    #                                               facts); NOT camera pose — kept top-level so the
    #                                               field name says what it holds


@dataclass(frozen=True)
class ObservationSet:
    set_id: str
    observations: List[Observation]
    roles: Dict[str, str]               # semantic role -> obs_id
    tick: int

    def __post_init__(self):
        if not self.observations:
            raise ValueError("ObservationSet must contain at least one observation")
        ticks = {o.tick for o in self.observations}
        if ticks != {self.tick}:
            raise ValueError(f"ObservationSet tick mismatch: set tick {self.tick}, obs ticks {ticks}")
        ids = {o.obs_id for o in self.observations}
        missing = {role: obs_id for role, obs_id in self.roles.items() if obs_id not in ids}
        if missing:
            raise ValueError(f"ObservationSet roles reference unknown obs_id(s): {missing}")


@dataclass(frozen=True)
class Estimate:
    value: Any
    kind: str                            # depth | point | scale | ray
    uncertainty: float                   # MANDATORY (spec §4)
    coarse: bool                         # MANDATORY
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _VALID_KIND:
            raise ValueError(f"kind must be one of {_VALID_KIND}, got {self.kind!r}")
        if self.uncertainty is None or not isinstance(self.uncertainty, (int, float)):
            raise ValueError("uncertainty is mandatory and must be a number")
        if not isinstance(self.coarse, bool):
            raise ValueError("coarse is mandatory and must be a bool")


@dataclass(frozen=True)
class ActionResult:
    action_id: str
    commanded: Dict[str, Any]
    achieved: Dict[str, Any]             # ACTUAL executed delta/pose read from proprioception AFTER
    status: str                          # SUCCESS | FAILED | ABORTED
    abort_reason: Optional[str]
    resulting_pose: Dict[str, Any]
    tick: int
    call_status: str = "OK"               # API machinery returned a structured result
    planning: Dict[str, Any] = field(
        default_factory=lambda: {"status": "NOT_REPORTED"})
    execution: Dict[str, Any] = field(
        default_factory=lambda: {"status": "UNKNOWN", "state_changed": None})
    failure: Optional[Dict[str, Any]] = None
    # Canonical raw robot self-state at the action boundary. Kept optional on the Python
    # dataclass so lightweight pure-test fixtures can still construct partial values; every
    # production mutating tool must populate both fields and the wire contract requires them.
    observed_before: Optional[Dict[str, Any]] = None
    observed_after: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        if self.status not in _VALID_STATUS:
            raise ValueError(f"status must be one of {_VALID_STATUS}, got {self.status!r}")
        if self.call_status != "OK":
            raise ValueError("a returned ActionResult must have call_status='OK'")
        if self.status == "SUCCESS" and self.failure is not None:
            raise ValueError("a successful ActionResult cannot carry failure")
        if self.status != "SUCCESS" and self.failure is None:
            stage = str(self.achieved.get("failure_stage") or
                        ("abort_guard" if self.status == "ABORTED" else "unspecified"))
            code = str(self.achieved.get("failure_category") or self.abort_reason or self.status)
            message = str(self.achieved.get("reason") or self.abort_reason or
                          "inspect achieved diagnostics")
            object.__setattr__(self, "failure", {
                "stage": stage, "code": code, "message": message,
            })


@dataclass(frozen=True)
class ObservationPair:
    """Two observations of one camera around one caller-selected robot motion.

    Unlike ObservationSet, the frames intentionally have different ticks.  The motion and
    contact evidence travel with the pair so a later correspondence calculation cannot silently
    consume unrelated frames or an unknown camera move.
    """
    pair_id: str
    before: Observation
    after: Observation
    motion: ActionResult
    camera_delta_world_m: List[float]
    camera_baseline_m: float
    camera_rotation_deg: float
    contact_evidence: Dict[str, Any]
    validity: Dict[str, Any]

    def __post_init__(self):
        if self.before.camera != self.after.camera:
            raise ValueError("ObservationPair must use the same camera before and after motion")
        if self.after.tick < self.before.tick:
            raise ValueError("ObservationPair after.tick must not precede before.tick")
        if self.motion.tick != self.after.tick:
            raise ValueError("ObservationPair motion tick must match the after observation")
        if self.camera_baseline_m < 0:
            raise ValueError("camera_baseline_m must be non-negative")
        required = {"motion_succeeded", "baseline_nonzero", "contact_free"}
        if set(self.validity) != required or any(
                not isinstance(self.validity[name], bool) for name in required):
            raise ValueError(
                "ObservationPair validity must contain exactly boolean "
                "motion_succeeded/baseline_nonzero/contact_free")


def require_same_frame(a: Observation, b: Observation, allow_different_tick: bool = False) -> None:
    """Guard: two observations used together (e.g. a disparity pair) must be the SAME camera. Ticks may
    differ only for a motion pair (allow_different_tick=True). Prevents cross-frame pixel mixing (§4)."""
    if a.camera != b.camera:
        raise FrameMismatch(f"cross-camera use: {a.camera} vs {b.camera}")
    if not allow_different_tick and a.tick != b.tick:
        raise FrameMismatch(f"different ticks not allowed here: {a.tick} vs {b.tick}")
