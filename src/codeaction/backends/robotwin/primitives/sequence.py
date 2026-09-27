"""
Primitive sequence runner.

This module exposes TWO runners:

  1. ``execute_primitive_sequence(steps, …)`` — INTERNAL.
     Takes a list of ``(callable, kwargs)`` pairs. Used by safe skills
     (e.g. dual_arm_lift) to compose primitive calls. NOT exposed to the
     LLM.

  2. ``execute_program(TASK_ENV, program, refs, logger)``.
     Takes a structured op-dict program (validated against ALLOWED_OPS),
     dispatches each op to the correct primitive function, supports
     ``save_as`` variable binding and ``$var.field`` references.
     This is the runtime backing ``skill_library.execute_primitive_sequence``,
     which IS exposed to the LLM Executor.

Both runners share the same aggregation logic (motion_completed / grasp_verified /
task_success), so their return ResultDicts are interchangeable from the
Planner's perspective.

Safety contract:
  - motion primitives reset ``TASK_ENV.plan_success = True`` internally;
  - ``is_task_success`` calls ``TASK_ENV.check_success()`` explicitly;
  - primitives do NOT bump ``take_action_cnt``. Only example runners /
    finalization helpers may do so.
"""

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

from .result import (
    SUCCESS, FAILED,
    STAGE_MOTION, STAGE_GRASP_VERIFICATION, STAGE_TASK_VERIFICATION,
    make_skill_result,
)


PrimitiveCall = Tuple[Callable[..., dict], dict]


_STAGE_BY_EVENT = {
    "is_grasp_verified": STAGE_GRASP_VERIFICATION,
    "is_lift_verified":  STAGE_GRASP_VERIFICATION,
    "is_task_success":   STAGE_TASK_VERIFICATION,
}


def execute_primitive_sequence(
    steps: Iterable[PrimitiveCall],
    *,
    event: str = "execute_primitive_sequence",
    abort_on_fail: bool = True,
    logger=None,
) -> dict:
    """
    Run primitives in order.

    Args:
        steps: iterable of `(callable, kwargs_dict)` pairs.
               Each callable should return a PrimitiveResult dict.
        event: top-level event name for the aggregated result.
        abort_on_fail: stop at the first FAILED step.
        logger: optional EpisodeLogger; if provided, each primitive's result
                is forwarded via `logger.log_skill(...)` for trace continuity.

    Returns:
        A ResultDict with `stage`, `status`, `details`, and aggregated `data`.
    """
    steps = list(steps)
    total = len(steps)

    primitive_results: List[dict] = []
    failed_index = None
    failed_step = None

    motion_completed_any = False
    motion_completed_all = True
    grasp_verified: Any = None
    lift_height_delta: Any = None
    task_success: Any = None
    object_before = None
    object_after = None

    for i, item in enumerate(steps):
        if not (isinstance(item, tuple) and len(item) == 2):
            failed_index = i
            failed_step = repr(item)
            primitive_results.append({
                "event": "invalid_step", "status": FAILED,
                "details": f"step {i} is not a (callable, kwargs) tuple: {item!r}",
                "data": {},
            })
            break

        fn, kwargs = item
        try:
            r = fn(**dict(kwargs))
        except Exception as e:
            r = {
                "event": getattr(fn, "__name__", "unknown"),
                "status": FAILED,
                "details": f"primitive raised: {e}",
                "data": dict(kwargs),
            }
        if not isinstance(r, dict):
            r = {
                "event": getattr(fn, "__name__", "unknown"),
                "status": FAILED,
                "details": f"primitive returned non-dict: {r!r}",
                "data": {},
            }
        primitive_results.append(r)

        if logger is not None:
            try:
                logger.log_skill(
                    skill_name=r.get("event", "primitive"),
                    args=kwargs,
                    result=r.get("status"),
                    feedback=r.get("details", ""),
                    step_num=int(r.get("data", {}).get("sim_step", -1)),
                    success=(r.get("status") == SUCCESS),
                    data=r.get("data"),
                )
            except Exception:
                pass  # logger is best-effort

        # Aggregate motion / verification signals
        d = r.get("data") or {}
        if "motion_completed" in d:
            motion_completed_any = True
            motion_completed_all = motion_completed_all and bool(d["motion_completed"])
        ev = r.get("event")
        if ev == "is_grasp_verified":
            grasp_verified = bool(d.get("grasp_verified", False))
        if ev == "is_lift_verified":
            grasp_verified = (grasp_verified if grasp_verified is not None
                              else bool(d.get("lift_verified", False)))
            lift_height_delta = d.get("height_delta", lift_height_delta)
        if ev == "is_task_success":
            task_success = bool(d.get("env_success", False))
        if ev == "get_object_pose":
            if object_before is None:
                object_before = d.get("position")
            object_after = d.get("position", object_after)
        # Track height delta even from grasp verification
        if ev == "is_grasp_verified" and lift_height_delta is None:
            lift_height_delta = d.get("height_delta")

        if r.get("status") != SUCCESS and abort_on_fail:
            failed_index = i
            failed_step = {
                "event": ev,
                "kwargs": kwargs,
                "details": r.get("details"),
            }
            break

    completed = len(primitive_results)
    sequence_ok = (failed_index is None) and all(
        pr.get("status") == SUCCESS for pr in primitive_results
    )

    # Stage of the last failure (if any), otherwise stage of last verification.
    if failed_index is not None:
        last_event = primitive_results[failed_index].get("event")
        stage = _STAGE_BY_EVENT.get(last_event, STAGE_MOTION)
    else:
        last_event = primitive_results[-1].get("event") if primitive_results else None
        stage = _STAGE_BY_EVENT.get(last_event, STAGE_MOTION)

    if sequence_ok:
        details = f"All {completed}/{total} primitives succeeded."
    else:
        details = (f"Sequence aborted at step {failed_index}: "
                   f"{failed_step}" if failed_index is not None
                   else "Sequence completed with non-fatal failures.")

    return make_skill_result(
        event=event,
        status=SUCCESS if sequence_ok else FAILED,
        stage=stage,
        details=details,
        completed_ops=completed,
        total_ops=total,
        failed_op_index=failed_index,
        failed_op=failed_step,
        primitive_results=primitive_results,
        motion_completed=motion_completed_all if motion_completed_any else None,
        grasp_verified=grasp_verified,
        task_success=task_success,
        object_before=object_before,
        object_after=object_after,
        height_delta=lift_height_delta,
    )


# ── Dict-based program runner ───────────────────────────────────────────

def _build_dispatch_table():
    """
    Build {op_name: callable(TASK_ENV, **kwargs) -> PrimitiveResult} table.

    Constructed lazily inside the function to avoid eager import of
    motion / gripper / verification (which pull in envs.utils.action and
    therefore sapien) when only program_schema / validator are needed.
    """
    from . import perception as _perception
    from . import motion as _motion
    from . import gripper as _gripper
    from . import verification as _verification
    from . import official_actions as _official

    # Wrappers — primitives accept positional TASK_ENV, kwargs go through as-is.
    table = {
        # perception
        "get_object_pose":     _perception.get_object_pose,
        "get_gripper_pose":    _perception.get_gripper_pose,
        "get_gripper_state":   _perception.get_gripper_state,
        # motion
        "move_to_pose":        _motion.move_to_pose,
        "move_delta":          _wrap_move_delta(_motion.move_delta),
        "move_both_to_poses":  _motion.move_both_to_poses,
        "move_both_delta":     _wrap_move_both_delta(_motion.move_both_delta),
        "move_to_home":        _motion.move_to_home,
        # gripper
        "open_gripper":        _gripper.open_gripper,
        "close_gripper":       _gripper.close_gripper,
        "wait_steps":          _gripper.wait_steps,
        # verification
        "is_grasp_verified":   _verification.is_grasp_verified,
        "is_lift_verified":    _verification.is_lift_verified,
        "is_task_success":     _verification.is_task_success,
        # Official API wrappers
        "dual_grasp_actor":    _official.dual_grasp_actor,
    }
    return table


def _wrap_move_delta(fn):
    """Accept either ``{arm, dx, dy, dz}`` or ``{arm, delta=[dx,dy,dz]}``."""
    def _call(TASK_ENV, **kwargs):
        if "delta" in kwargs:
            d = kwargs.pop("delta")
            if not isinstance(d, (list, tuple)) or len(d) != 3:
                return _make_arg_error("move_delta", "delta must be length 3", kwargs)
            kwargs["dx"], kwargs["dy"], kwargs["dz"] = float(d[0]), float(d[1]), float(d[2])
        return fn(TASK_ENV, **kwargs)
    _call.__name__ = "move_delta"
    return _call


def _wrap_move_both_delta(fn):
    """No transformation needed; primitive already accepts left_delta/right_delta."""
    def _call(TASK_ENV, **kwargs):
        return fn(TASK_ENV, **kwargs)
    _call.__name__ = "move_both_delta"
    return _call


def _make_arg_error(event: str, message: str, args: dict) -> dict:
    """Build a FAILED PrimitiveResult for an arg-shape error."""
    return {
        "event": event, "status": FAILED,
        "details": f"argument error: {message}",
        "data": {"args": args, "motion_completed": False},
    }




def execute_program(TASK_ENV,
                    program: List[dict],
                    refs: Optional[Sequence[str]] = None,
                    logger=None,
                    event: str = "execute_primitive_sequence") -> dict:
    """
    Validated dict-based program runner.

    Args:
        TASK_ENV: live RoboTwin env (captured by closure, never exposed to LLM).
        program:  list of op-dict entries.
        refs:     optional list of scene reference names for validator V7.
        logger:   optional EpisodeLogger (`log_skill` is used best-effort).
        event:    name for the aggregated ResultDict.

    Returns:
        Aggregated ResultDict with stage / status / details and a `data`
        block containing failed_op_index, failed_op, completed_ops,
        total_ops, primitive_results, motion_completed, grasp_verified,
        task_success, object_before, object_after, height_delta.
    """
    # Local imports — heavier modules (validator, schema dispatch).
    from .program_schema import resolve_program_value
    from .program_validator import validate_program

    # 1. Validate first; abort before any side effects on TASK_ENV.
    val_r = validate_program(program, refs=refs)
    if val_r["status"] != SUCCESS:
        # Forward to logger for trace continuity.
        if logger is not None:
            try:
                logger.log_skill(
                    skill_name="execute_primitive_sequence.validation",
                    args={"program_len": len(program) if isinstance(program, list) else None},
                    result=FAILED,
                    feedback=val_r["details"],
                    step_num=int(getattr(TASK_ENV, "take_action_cnt", -1)),
                    success=False,
                    data=val_r.get("data"),
                )
            except Exception:
                pass
        # Re-wrap as the canonical aggregated schema.
        return make_skill_result(
            event=event,
            status=FAILED,
            stage="validation",
            details=val_r["details"],
            completed_ops=0,
            total_ops=len(program) if isinstance(program, list) else 0,
            failed_op_index=val_r["data"].get("failed_op_index"),
            failed_op={"op": val_r["data"].get("op")},
            primitive_results=[],
            motion_completed=None,
            grasp_verified=None,
            task_success=None,
            object_before=None,
            object_after=None,
            height_delta=None,
            validator_rule=val_r["data"].get("validator_rule"),
        )

    normalized: List[dict] = val_r["data"]["program"]

    # 2. Dispatch table — built lazily.
    try:
        dispatch = _build_dispatch_table()
    except Exception as e:
        return make_skill_result(
            event=event,
            status=FAILED,
            stage="validation",
            details=f"failed to construct primitive dispatch table: {e}",
            completed_ops=0,
            total_ops=len(normalized),
        )

    # 3. Run, accumulating state for $var resolution + aggregated fields.
    program_state: dict = {}
    primitive_results: List[dict] = []
    failed_index: Optional[int] = None
    failed_op: Optional[dict] = None

    motion_completed_any = False
    motion_completed_all = True
    grasp_verified: Any = None
    lift_height_delta: Any = None
    task_success: Any = None
    object_before: Any = None
    object_after: Any = None

    for i, entry in enumerate(normalized):
        op = entry["op"]
        raw_args = entry["args"]
        save_as = entry["save_as"]

        # Resolve $-references against current state.
        resolved_args, err = resolve_program_value(raw_args, program_state)
        if err:
            failed_index = i
            failed_op = {"op": op, "args": raw_args, "save_as": save_as,
                         "details": f"reference resolution failed: {err}"}
            primitive_results.append({
                "event": op, "status": FAILED,
                "details": f"reference resolution failed: {err}",
                "data": {"args": raw_args, "motion_completed": False},
            })
            break

        fn = dispatch.get(op)
        if fn is None:
            failed_index = i
            failed_op = {"op": op, "args": resolved_args, "save_as": save_as,
                         "details": "op passed validation but has no runtime"}
            primitive_results.append({
                "event": op, "status": FAILED,
                "details": (f"op {op!r} has no runtime in primitive dispatch. "
                            f"This is a bug: validator allowed it but no "
                            f"implementation is registered."),
                "data": {"args": resolved_args, "motion_completed": False},
            })
            break

        # Call the primitive.
        try:
            r = fn(TASK_ENV, **resolved_args)
        except TypeError as e:
            r = {
                "event": op, "status": FAILED,
                "details": f"primitive arg mismatch: {e}",
                "data": {"args": resolved_args, "motion_completed": False},
            }
        except Exception as e:
            r = {
                "event": op, "status": FAILED,
                "details": f"primitive raised: {e}",
                "data": {"args": resolved_args, "motion_completed": False},
            }

        if not isinstance(r, dict):
            r = {
                "event": op, "status": FAILED,
                "details": f"primitive returned non-dict: {r!r}",
                "data": {},
            }
        primitive_results.append(r)

        # Best-effort per-primitive logging.
        if logger is not None:
            try:
                logger.log_skill(
                    skill_name=op, args=resolved_args,
                    result=r.get("status"), feedback=r.get("details", ""),
                    step_num=int(r.get("data", {}).get("sim_step", -1)),
                    success=(r.get("status") == SUCCESS),
                    data=r.get("data"),
                )
            except Exception:
                pass

        # Aggregate signals.
        d = r.get("data") or {}
        if "motion_completed" in d:
            motion_completed_any = True
            motion_completed_all = motion_completed_all and bool(d["motion_completed"])
        if op == "is_grasp_verified":
            grasp_verified = bool(d.get("grasp_verified", False))
            if lift_height_delta is None:
                lift_height_delta = d.get("height_delta")
        elif op == "is_lift_verified":
            grasp_verified = grasp_verified if grasp_verified is not None \
                else bool(d.get("lift_verified", False))
            lift_height_delta = d.get("height_delta", lift_height_delta)
        elif op == "is_task_success":
            task_success = bool(d.get("env_success", False))
        elif op == "get_object_pose":
            pos = d.get("position")
            if pos is not None:
                if object_before is None:
                    object_before = pos
                object_after = pos

        # Bind save_as.
        if save_as:
            program_state[save_as] = r

        # Abort on failure.
        if r.get("status") != SUCCESS:
            failed_index = i
            failed_op = {"op": op, "args": resolved_args, "save_as": save_as,
                         "details": r.get("details")}
            break

    completed = len(primitive_results)
    sequence_ok = (failed_index is None) and all(
        pr.get("status") == SUCCESS for pr in primitive_results
    )

    # Stage of the last failure (if any), otherwise stage of last verification.
    if failed_index is not None:
        last_event = primitive_results[failed_index].get("event")
        stage = _STAGE_BY_EVENT.get(last_event, STAGE_MOTION)
    else:
        last_event = primitive_results[-1].get("event") if primitive_results else None
        stage = _STAGE_BY_EVENT.get(last_event, STAGE_MOTION)

    if sequence_ok:
        details = f"All {completed}/{len(normalized)} primitives succeeded."
    elif failed_index is not None:
        details = (f"Sequence aborted at program[{failed_index}]: "
                   f"{failed_op.get('details') if failed_op else 'unknown failure'}")
    else:
        details = "Program completed with non-fatal failures."

    aggregated = make_skill_result(
        event=event,
        status=SUCCESS if sequence_ok else FAILED,
        stage=stage,
        details=details,
        completed_ops=completed,
        total_ops=len(normalized),
        failed_op_index=failed_index,
        failed_op=failed_op,
        primitive_results=primitive_results,
        motion_completed=motion_completed_all if motion_completed_any else None,
        grasp_verified=grasp_verified,
        task_success=task_success,
        object_before=object_before,
        object_after=object_after,
        height_delta=lift_height_delta,
        sim_step=int(getattr(TASK_ENV, "take_action_cnt", -1)),
    )

    # Best-effort aggregated logging.
    if logger is not None:
        try:
            logger.log_skill(
                skill_name=event,
                args={"program_len": len(normalized)},
                result=aggregated["status"],
                feedback=aggregated["details"],
                step_num=aggregated["data"].get("sim_step", -1),
                success=(aggregated["status"] == SUCCESS),
                data=aggregated["data"],
            )
        except Exception:
            pass

    return aggregated
