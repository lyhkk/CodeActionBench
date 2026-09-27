"""Normalize the motion planner's own failure status into a stable, agent-facing taxonomy.

Why this exists: the harness used to report every planning failure as one `straight_line_plan_failure`
label. Measured across 28 failing cells replayed under six solver-effort settings, that one label
covers two situations with opposite meanings:

  * the commanded goal pose has NO inverse-kinematics solution — recovered 0 / 22 times by any
    amount of extra solver effort (more IK seeds, more trajopt seeds, more attempts, graph fallback)
  * a trajectory could not be optimized between two valid states — recovered 4 / 6 times

So the collapsed label discarded the only bit that distinguished a kinematic fact from a solver
outcome. Reporting the distinction is squarely inside the §0.1 corollary-8 boundary: it describes
why the command that was issued did not execute, in physical/planning terms. It must NOT tell the
model what to do next — no "retry", no "pick another pose", no recovery advice. The model reads the
fact and decides.

Vendor strings are deliberately not the contract: `raw` is carried for auditability, `code` is what
the benchmark promises, so swapping planners does not silently change the agent-facing surface.
Pure and simulator-free.
"""
from typing import Any, Mapping, Optional


# code -> factual gloss. Every gloss states a condition, never an instruction.
PLANNER_STATUS_MEANINGS = {
    "goal_pose_unreachable":
        "no inverse-kinematics solution exists for the commanded goal pose from this arm",
    "trajectory_optimization_failed":
        "start and goal are individually valid but no trajectory between them was optimized",
    "graph_search_failed":
        "the sampling-based search found no connecting path",
    "start_state_in_collision":
        "the arm's current configuration is already in collision, so no motion can be planned",
    "start_state_outside_joint_limits":
        "the arm's current configuration is outside its joint limits",
    "start_state_invalid":
        "the arm's current configuration was rejected as a planning start state",
    "orientation_constraint_rejected":
        "the requested orientation-holding constraint was rejected for this query",
    "query_invalid":
        "the planning query itself was malformed or out of range",
    "trajectory_timing_failed":
        "a trajectory was found but could not be given a valid timing",
    "not_attempted":
        "the planner did not attempt this query",
    "planner_status_unavailable":
        "the planner did not report a reason",
}

# Raw cuRobo `MotionGenStatus` values (0.7.8) -> benchmark code.
_RAW_TO_CODE = {
    "IK Fail": "goal_pose_unreachable",
    "Graph Fail": "graph_search_failed",
    "TrajOpt Fail": "trajectory_optimization_failed",
    "Finetune TrajOpt Fail": "trajectory_optimization_failed",
    "dt exceeded maximum allowed trajectory dt": "trajectory_timing_failed",
    "Invalid Query": "query_invalid",
    "Invalid Start State, unknown issue": "start_state_invalid",
    "Start state is colliding with world": "start_state_in_collision",
    "Start state is in self-collision": "start_state_in_collision",
    "Start state is out of joint limits": "start_state_outside_joint_limits",
    "Invalid partial pose metric": "orientation_constraint_rejected",
    "Not Attempted": "not_attempted",
    "Success": None,
}


def normalize_planner_status(raw: Any) -> Optional[str]:
    """Map a planner's own status string to the benchmark code, or None when it means success."""
    if raw is None:
        return "planner_status_unavailable"
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return "planner_status_unavailable"
    if text in _RAW_TO_CODE:
        return _RAW_TO_CODE[text]
    lowered = text.lower()
    if "success" in lowered:
        return None
    # Unknown vendor string: keep it auditable rather than guessing a meaning for it.
    return "planner_status_unavailable"


def planner_diagnostic(result: Any) -> Optional[Mapping[str, Any]]:
    """Agent-facing planner diagnostic from a raw planner result dict, or None if there is nothing
    to report. Reports the condition only — never a suggested action."""
    if not isinstance(result, Mapping):
        return None
    raw = result.get("curobo_status")
    code = normalize_planner_status(raw)
    if code is None:
        return None
    out = {
        "code": code,
        "meaning": PLANNER_STATUS_MEANINGS.get(code,
                                               PLANNER_STATUS_MEANINGS["planner_status_unavailable"]),
        "raw": (str(raw) if raw is not None else None),
    }
    # These are the planner's own measurements of this refusal, not harness-derived advice.
    for key, out_key in (("attempts", "attempts"),
                         ("position_error", "position_error_m"),
                         ("rotation_error", "rotation_error_rad")):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[out_key] = round(float(value), 6)
    valid_query = result.get("valid_query")
    if isinstance(valid_query, bool):
        out["valid_query"] = valid_query
    return out
