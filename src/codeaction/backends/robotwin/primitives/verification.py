"""
Verification primitives — privileged checks that distinguish

  motion_success    : did the waypoint sequence complete?
  grasp_success     : did the object actually get held?
  task_success      : did the env's check_success() return True?

These three signals must NEVER collapse into one boolean (see
status/lift_pot_failure_analysis.md).
"""

from typing import Optional

from .result import (
    SUCCESS, FAILED,
    STAGE_GRASP_VERIFICATION, STAGE_TASK_VERIFICATION,
    make_primitive_result,
)
from .perception import get_object_pose, get_gripper_state


# ── Grasp verification ────────────────────────────────────────────────────

def is_grasp_verified(TASK_ENV, object_name: str, z_before: float,
                      min_dz: float = 0.03, arm: Optional[str] = None) -> dict:
    """
    After a grasp + micro-lift, check whether the object is actually held.

    Decision rule:
      SUCCESS iff height_delta >= min_dz AND (if `arm` given) that gripper is closed.

    Note: this is a privileged check; it reads the object's z directly. It is
    NOT the env's task success — call `is_task_success` for that.
    """
    pose_r = get_object_pose(TASK_ENV, object_name)
    if pose_r["status"] != SUCCESS:
        return make_primitive_result(
            "is_grasp_verified", FAILED,
            f"Cannot verify grasp: {pose_r['details']}",
            object_name=object_name, z_before=z_before, height_delta=None,
            grasp_verified=False,
        )
    z_after = float(pose_r["data"]["position"][2])
    dz = z_after - float(z_before)
    height_ok = dz >= float(min_dz)

    gripper_closed: Optional[bool] = None
    gripper_val: Optional[float] = None
    if arm is not None:
        grip_r = get_gripper_state(TASK_ENV, arm)
        if grip_r["status"] == SUCCESS:
            gripper_closed = bool(grip_r["data"].get("is_closed"))
            gripper_val = grip_r["data"].get("gripper_val")
    # If arm not specified, do not penalize for unknown gripper state.
    gripper_ok = True if gripper_closed is None else gripper_closed

    success = height_ok and gripper_ok
    return make_primitive_result(
        "is_grasp_verified", SUCCESS if success else FAILED,
        (f"Grasp verified: dz={dz:.3f}m (>= {min_dz}), gripper_closed={gripper_closed}."
         if success else
         f"Grasp NOT verified: dz={dz:.3f}m (need >= {min_dz}), "
         f"gripper_closed={gripper_closed}."),
        object_name=object_name, arm=arm,
        z_before=float(z_before), z_after=z_after, height_delta=dz,
        height_ok=height_ok, gripper_closed=gripper_closed,
        gripper_val=gripper_val, grasp_verified=success,
    )


# ── Lift verification ─────────────────────────────────────────────────────

def is_lift_verified(TASK_ENV, object_name: str, z_before: float,
                     min_dz: float = 0.10) -> dict:
    """
    After a lift, check the object rose at least `min_dz`.
    """
    pose_r = get_object_pose(TASK_ENV, object_name)
    if pose_r["status"] != SUCCESS:
        return make_primitive_result(
            "is_lift_verified", FAILED,
            f"Cannot verify lift: {pose_r['details']}",
            object_name=object_name, z_before=z_before, height_delta=None,
            lift_verified=False,
        )
    z_after = float(pose_r["data"]["position"][2])
    dz = z_after - float(z_before)
    success = dz >= float(min_dz)

    # Dual-arm tasks: report both grippers' state for the log even though
    # we don't fail on them here (motion-level failure handles that).
    left = get_gripper_state(TASK_ENV, "left")
    right = get_gripper_state(TASK_ENV, "right")
    left_closed = left["data"].get("is_closed") if left["status"] == SUCCESS else None
    right_closed = right["data"].get("is_closed") if right["status"] == SUCCESS else None

    return make_primitive_result(
        "is_lift_verified", SUCCESS if success else FAILED,
        (f"Lift verified: dz={dz:.3f}m (>= {min_dz})."
         if success else
         f"Lift NOT verified: dz={dz:.3f}m (need >= {min_dz})."),
        object_name=object_name,
        z_before=float(z_before), z_after=z_after, height_delta=dz,
        lift_verified=success,
        left_gripper_closed=left_closed, right_gripper_closed=right_closed,
    )


# ── Task-level success ────────────────────────────────────────────────────

def is_task_success(TASK_ENV) -> dict:
    """
    Call TASK_ENV.check_success() (the env's own eval) and report.

    `check_success` returns a fresh boolean. We mirror it onto
    TASK_ENV.eval_success so that the outer eval harness sees the success
    (the harness only watches eval_success and the step counter; env.move()
    never updates eval_success on its own).
    """
    try:
        env_ok = bool(TASK_ENV.check_success())
    except Exception as e:
        return make_primitive_result(
            "is_task_success", FAILED,
            f"check_success raised: {e}",
            env_success=False,
        )

    # Mirror to eval_success so downstream harness logic agrees.
    try:
        if env_ok:
            TASK_ENV.eval_success = True
    except Exception:
        pass

    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "is_task_success", SUCCESS if env_ok else FAILED,
        "Env reports task success." if env_ok else "Env reports task NOT yet successful.",
        env_success=env_ok, sim_step=sim_step,
    )
