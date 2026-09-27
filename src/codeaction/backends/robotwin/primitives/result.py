"""
Standard PrimitiveResult / ResultDict schema helpers.

PrimitiveResult — returned by atomic primitives (internal).
ResultDict      — returned by safe skills (exposed to Executor).

The two are structurally compatible (both have status/details/data); safe
skills add a `stage` field to disambiguate motion vs grasp_verification vs
task_verification failures.
"""

from typing import Any


# ── Status constants ──────────────────────────────────────────────────────

SUCCESS = "SUCCESS"
FAILED = "FAILED"

STAGE_MOTION = "motion"
STAGE_GRASP_VERIFICATION = "grasp_verification"
STAGE_TASK_VERIFICATION = "task_verification"
STAGE_PERCEPTION = "perception"


# ── Builders ──────────────────────────────────────────────────────────────

def make_primitive_result(event: str, status: str, details: str, **data: Any) -> dict:
    """
    Build a PrimitiveResult dict.

    Args:
        event:    Primitive name, e.g. "move_to_pose".
        status:   SUCCESS or FAILED.
        details:  Human-readable explanation (suitable for Planner observation).
        **data:   Arbitrary payload fields (ee_before, ee_after, plan_success, ...).

    Returns:
        {"event", "status", "details", "data"}.
    """
    return {
        "event": event,
        "status": status,
        "details": details,
        # Linguistic aliases keep the schema robust for LLM consumption (kept
        # for backward compatibility with skill_library.make_result).
        "feedback": details,
        "message": details,
        "error": details if status == FAILED else None,
        "data": dict(data),
    }


def make_skill_result(event: str, status: str, stage: str, details: str, **data: Any) -> dict:
    """
    Build a ResultDict for a safe skill.

    Differs from PrimitiveResult in that it carries a `stage` field, which the
    Planner uses to decide what kind of recovery is appropriate.
    """
    r = make_primitive_result(event, status, details, **data)
    r["stage"] = stage
    return r


def get_primitive_feedback(result: Any) -> str:
    """Robust feedback extractor — works on any result-shaped dict or raw string."""
    if not isinstance(result, dict):
        return str(result)
    return (
        result.get("details")
        or result.get("feedback")
        or result.get("message")
        or "No feedback provided."
    )


def is_success(result: Any) -> bool:
    """Convenience predicate."""
    return isinstance(result, dict) and result.get("status") == SUCCESS
