"""
Motion primitives — thin wrappers over `env.move(env.<primitive>(...))`.

Contract for every motion primitive in this module:

  1. Reset TASK_ENV.plan_success = True *before* calling env.move().
     (plan_success is sticky-False; it must be re-armed each motion.)
  2. Call env.move(...) which returns True/False/None.
  3. Inspect TASK_ENV.plan_success *after* the call as the source of truth.
  4. Return a PrimitiveResult with ee_before / ee_after / plan_success in `data`.

Pose convention: SAPIEN wxyz = [x, y, z, qw, qx, qy, qz]. See pose_utils.py.

Step-budget note: env.move() goes through env.take_dense_action() which does
NOT increment TASK_ENV.take_action_cnt. The eval loop watches that counter,
so a runner using these primitives must explicitly bump the counter to exit.
See examples/primitive_lift_pot.py for the recommended pattern.
"""

from typing import Optional, Sequence

import numpy as np

from envs.utils.action import Action, ArmTag

from .result import (
    SUCCESS, FAILED, STAGE_MOTION,
    make_primitive_result,
)
from .perception import get_gripper_pose
from . import pose_utils


# ── Helpers ───────────────────────────────────────────────────────────────

def _arm_tag(arm: str) -> ArmTag:
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return ArmTag(arm)


def _safe_ee_pose(TASK_ENV, arm: str):
    """Return the current EE pose for `arm`, or None on failure."""
    r = get_gripper_pose(TASK_ENV, arm)
    return r["data"].get("pose") if r["status"] == SUCCESS else None


def _validate_target_pose(target_pose: Sequence[float]) -> Optional[str]:
    if target_pose is None:
        return "target_pose is None"
    if len(target_pose) != 7:
        return f"target_pose must be length 7, got {len(target_pose)}"
    ok, msg = pose_utils.check_workspace_bounds(target_pose[:3])
    if not ok:
        return msg
    return None


def _validate_delta(dx: float, dy: float, dz: float) -> Optional[str]:
    ok, mag = pose_utils.check_delta_magnitude(dx, dy, dz)
    if not ok:
        return f"delta magnitude {mag:.3f}m exceeds limit {pose_utils.MAX_DELTA_MAGNITUDE}m"
    return None


def _clamp_delta(dx: float, dy: float, dz: float,
                 limit: float = pose_utils.MAX_DELTA_MAGNITUDE):
    """Clamp a displacement to `limit` magnitude, preserving direction.

    Returns (cdx, cdy, cdz, was_clamped, original_magnitude). The 0.3 m cap
    is a control-quality heuristic, not a safety limit — cuRobo still does
    collision-aware planning on the (clamped) target.
    """
    mag = pose_utils.l2_norm(dx, dy, dz)
    if mag <= limit or mag < 1e-9:
        return dx, dy, dz, False, mag
    s = limit / mag
    return dx * s, dy * s, dz * s, True, mag


# ── Single-arm absolute move ──────────────────────────────────────────────

def move_to_pose(TASK_ENV, arm: str, target_pose: Sequence[float]) -> dict:
    """
    Move one arm to absolute pose [x, y, z, qw, qx, qy, qz].
    """
    msg = _validate_target_pose(target_pose)
    if msg:
        return make_primitive_result(
            "move_to_pose", FAILED, f"Invalid target pose: {msg}",
            arm=arm, target_pose=list(target_pose) if target_pose is not None else None,
            plan_success=False,
        )

    ee_before = _safe_ee_pose(TASK_ENV, arm)
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.move_to_pose(_arm_tag(arm), list(target_pose)))
    except Exception as e:
        return make_primitive_result(
            "move_to_pose", FAILED, f"env.move raised: {e}",
            arm=arm, target_pose=list(target_pose),
            ee_before=ee_before, ee_after=_safe_ee_pose(TASK_ENV, arm),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    ee_after = _safe_ee_pose(TASK_ENV, arm)
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_to_pose", SUCCESS if ok else FAILED,
        f"{arm} arm reached target." if ok else f"{arm} arm motion planning failed.",
        arm=arm, target_pose=list(target_pose),
        ee_before=ee_before, ee_after=ee_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Single-arm reach to a TCP (fingertip) target, with internal chunking ──

def reach_tcp(TASK_ENV, arm: str, target_xyz: Sequence[float],
              target_quat: Optional[Sequence[float]] = None,
              max_step: float = pose_utils.MAX_DELTA_MAGNITUDE) -> dict:
    """Reach a FINGERTIP (TCP) target — the single atomic "go there" primitive.

    Targets the TCP, not the wrist, so a grounded grasp point is hit precisely.
    ``target_quat`` None → keep the current TCP orientation (pure reach);
    otherwise the gripper also reorients to ``target_quat`` (grasp pose).

    Distance is handled internally: one cuRobo plan to the full target is tried
    first (keeps cuRobo's curved obstacle avoidance); only if that fails is the
    path split into ≤ ``max_step`` legs (each still cuRobo-planned).  The caller
    issues ONE call regardless of distance and never sees a displacement cap.
    """
    quat_was_explicit = target_quat is not None
    quat_mode = "explicit" if quat_was_explicit else "kept_current"

    gp = get_gripper_pose(TASK_ENV, arm)
    data = gp.get("data") or {}
    ee_pose, tcp_pose = data.get("pose"), data.get("tcp_pose")
    if not ee_pose or not tcp_pose or len(ee_pose) != 7 or len(tcp_pose) != 7:
        return make_primitive_result(
            "reach_tcp", FAILED, "could not read ee/tcp pose.",
            arm=arm, plan_success=False, quat_mode=quat_mode,
            orientation_constraint=quat_was_explicit, failure_stage="proprioception")

    if target_quat is None:
        target_quat = list(tcp_pose[3:])
    if len(target_quat) != 4 or not (isinstance(target_xyz, (list, tuple)) and len(target_xyz) == 3):
        return make_primitive_result(
            "reach_tcp", FAILED,
            f"need target_xyz[3] and target_quat[4]; got xyz={target_xyz!r} quat={target_quat!r}",
            arm=arm, plan_success=False, quat_mode=quat_mode,
            orientation_constraint=quat_was_explicit, failure_stage="argument_validation")
    msg = _validate_target_pose(list(target_xyz) + list(target_quat))
    if msg:
        return make_primitive_result(
            "reach_tcp", FAILED, f"Invalid target: {msg}",
            arm=arm, target_xyz=list(target_xyz), plan_success=False,
            quat_mode=quat_mode, orientation_constraint=quat_was_explicit,
            failure_stage="workspace_check", failure_category="workspace")

    # 1) try the full reach in one cuRobo plan
    ee_target = pose_utils._tcp_target_to_ee_target(ee_pose, tcp_pose, target_xyz, target_quat)
    r = move_to_pose(TASK_ENV, arm, ee_target)
    n_legs = 1
    n_legs_attempted = 0
    failed_leg_index = None
    failure_stage = None
    if r.get("status") != SUCCESS:
        failure_stage = "full_plan"
        # 2) fallback: chunk the TCP path into <= max_step legs
        legs = pose_utils._chunk_positions(list(tcp_pose[:3]), list(target_xyz), max_step)
        n_legs = 0
        for wp in legs:
            n_legs += 1
            n_legs_attempted += 1
            gp_i = get_gripper_pose(TASK_ENV, arm)
            d_i = gp_i.get("data") or {}
            ee_i, tcp_i = d_i.get("pose"), d_i.get("tcp_pose")
            if not ee_i or not tcp_i:
                failure_stage = "chunk_plan"
                failed_leg_index = n_legs
                break
            r = move_to_pose(TASK_ENV, arm,
                             pose_utils._tcp_target_to_ee_target(ee_i, tcp_i, wp, target_quat))
            if r.get("status") != SUCCESS:
                failure_stage = "chunk_plan"
                failed_leg_index = n_legs
                break

    tcp_after = (get_gripper_pose(TASK_ENV, arm).get("data") or {}).get("tcp_pose")
    err = (float(np.linalg.norm(np.asarray(tcp_after[:3], float) - np.asarray(target_xyz, float)))
           if tcp_after else None)
    moved_m = (float(np.linalg.norm(np.asarray(tcp_after[:3], float) - np.asarray(tcp_pose[:3], float)))
               if tcp_after else None)
    ok = r.get("status") == SUCCESS
    planner_detail = r.get("details") or r.get("message") or r.get("error") or ""
    if ok:
        failure_stage = None
        failed_leg_index = None
        failure_category = None
    elif quat_was_explicit:
        failure_category = "orientation_constrained_plan_failure"
    elif failure_stage == "workspace_check":
        failure_category = "workspace"
    else:
        failure_category = "collision_or_joint_limit"
    return make_primitive_result(
        "reach_tcp", SUCCESS if ok else FAILED,
        (f"{arm} TCP → {[round(float(v),4) for v in target_xyz]} "
         f"(orient {'set' if quat_was_explicit else 'kept'}, {n_legs} leg(s), "
         f"tcp_err={err*100:.1f}cm)." if ok and err is not None
         else f"{arm} reach failed: {planner_detail}"),
        arm=arm, target_xyz=list(target_xyz), target_quat=list(target_quat),
        tcp_after=tcp_after, tcp_err_m=err, n_legs=n_legs,
        plan_success=ok, motion_completed=ok,
        quat_mode=quat_mode, orientation_constraint=quat_was_explicit,
        failure_stage=failure_stage, n_legs_attempted=n_legs_attempted,
        failed_leg_index=failed_leg_index, moved_m=moved_m,
        target_distance_remaining_m=err, planner_detail=planner_detail,
        failure_category=failure_category)


# ── Single-arm delta move ─────────────────────────────────────────────────

def move_delta(TASK_ENV, arm: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """Move one arm by a world-frame displacement.

    Over-cap requests are clamped to MAX_DELTA_MAGNITUDE (direction kept) and
    executed, rather than rejected — so the LLM never wastes a turn on an
    over-long request; it just takes another step.
    """
    cdx, cdy, cdz, clamped, orig_mag = _clamp_delta(dx, dy, dz)

    ee_before = _safe_ee_pose(TASK_ENV, arm)
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.move_by_displacement(_arm_tag(arm), x=cdx, y=cdy, z=cdz))
    except Exception as e:
        return make_primitive_result(
            "move_delta", FAILED, f"env.move raised: {e}",
            arm=arm, delta=[cdx, cdy, cdz], requested_delta=[dx, dy, dz],
            clamped=clamped, ee_before=ee_before,
            ee_after=_safe_ee_pose(TASK_ENV, arm), plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    ee_after = _safe_ee_pose(TASK_ENV, arm)
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    clamp_note = (" (capped for control stability — use move_to_pose to reach a far "
                  "point in one call)" if clamped else "")
    return make_primitive_result(
        "move_delta", SUCCESS if ok else FAILED,
        (f"{arm} arm displaced by [{cdx:.3f}, {cdy:.3f}, {cdz:.3f}]{clamp_note}." if ok
         else f"{arm} arm displacement motion planning failed."),
        arm=arm, delta=[cdx, cdy, cdz], requested_delta=[dx, dy, dz],
        clamped=clamped, ee_before=ee_before, ee_after=ee_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Single-arm rotate-in-place (position-hold, world-axis) ───────────────

def rotate_delta(TASK_ENV, arm: str,
                 axis_world: Sequence[float], angle_deg: float) -> dict:
    """
    Rotate the end-effector around a world-frame axis by ``angle_deg``,
    keeping its xyz position fixed (cuRobo PoseCostMetric constraint).

    The motion-planning, IK, and trajectory generation are entirely cuRobo's
    responsibility — this primitive only computes the target quaternion and
    hands a 7-DoF target plus ``constraint_pose=[1,1,1,0,0,0]`` to
    ``TASK_ENV.move``.  No new motion algorithm is introduced.
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "rotate_delta", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm, plan_success=False,
        )

    axis = np.asarray(axis_world, dtype=np.float64)
    if axis.size != 3:
        return make_primitive_result(
            "rotate_delta", FAILED,
            f"axis_world must be length 3, got {axis.size}",
            arm=arm, axis_world=list(axis_world), plan_success=False,
        )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return make_primitive_result(
            "rotate_delta", FAILED, "axis_world is zero-length.",
            arm=arm, axis_world=list(axis_world), plan_success=False,
        )
    axis_unit = axis / axis_norm

    ee_before = _safe_ee_pose(TASK_ENV, arm)
    if ee_before is None or len(ee_before) != 7:
        return make_primitive_result(
            "rotate_delta", FAILED, "Failed to read current ee pose.",
            arm=arm, axis_world=list(axis_world), plan_success=False,
        )
    cur_xyz = list(ee_before[:3])
    cur_quat = list(ee_before[3:7])

    half = float(np.deg2rad(float(angle_deg))) * 0.5
    dq = np.array(
        [np.cos(half), *(np.sin(half) * axis_unit)],
        dtype=np.float64,
    )
    new_quat = pose_utils._quat_mul_wxyz(dq, cur_quat)
    target_pose = cur_xyz + new_quat.tolist()

    # Build Action with constraint_pose in its args.  TASK_ENV.move() reads
    # ``action.args.get("constraint_pose")`` (envs/_base_task.py:959, 976, 987)
    # and forwards it to cuRobo's PoseCostMetric (hold xyz, free rotation).
    action = Action(
        _arm_tag(arm), "move",
        target_pose=target_pose,
        constraint_pose=[1, 1, 1, 0, 0, 0],
    )

    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move((_arm_tag(arm), [action]))
    except Exception as e:
        return make_primitive_result(
            "rotate_delta", FAILED, f"env.move raised: {e}",
            arm=arm, axis_world=list(axis_world), angle_deg=float(angle_deg),
            target_pose=target_pose,
            ee_before=ee_before, ee_after=_safe_ee_pose(TASK_ENV, arm),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    ee_after = _safe_ee_pose(TASK_ENV, arm)
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "rotate_delta", SUCCESS if ok else FAILED,
        (f"{arm} ee rotated {angle_deg:.1f}° around world axis "
         f"[{axis_unit[0]:.3f}, {axis_unit[1]:.3f}, {axis_unit[2]:.3f}]."
         if ok else
         f"{arm} ee rotate-in-place planning failed."),
        arm=arm, axis_world=axis_unit.tolist(), angle_deg=float(angle_deg),
        target_pose=target_pose,
        ee_before=ee_before, ee_after=ee_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Dual-arm absolute move ────────────────────────────────────────────────

def move_both_to_poses(TASK_ENV, left_pose: Sequence[float],
                       right_pose: Sequence[float]) -> dict:
    """
    Move both arms simultaneously to the given absolute poses.
    Both poses must be [x, y, z, qw, qx, qy, qz].
    """
    for name, p in (("left_pose", left_pose), ("right_pose", right_pose)):
        msg = _validate_target_pose(p)
        if msg:
            return make_primitive_result(
                "move_both_to_poses", FAILED, f"Invalid {name}: {msg}",
                left_pose=list(left_pose) if left_pose is not None else None,
                right_pose=list(right_pose) if right_pose is not None else None,
                plan_success=False,
            )

    left_before = _safe_ee_pose(TASK_ENV, "left")
    right_before = _safe_ee_pose(TASK_ENV, "right")
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(
            TASK_ENV.move_to_pose(ArmTag("left"), list(left_pose)),
            TASK_ENV.move_to_pose(ArmTag("right"), list(right_pose)),
        )
    except Exception as e:
        return make_primitive_result(
            "move_both_to_poses", FAILED, f"env.move raised: {e}",
            left_pose=list(left_pose), right_pose=list(right_pose),
            left_ee_before=left_before, right_ee_before=right_before,
            left_ee_after=_safe_ee_pose(TASK_ENV, "left"),
            right_ee_after=_safe_ee_pose(TASK_ENV, "right"),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    left_after = _safe_ee_pose(TASK_ENV, "left")
    right_after = _safe_ee_pose(TASK_ENV, "right")
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_both_to_poses", SUCCESS if ok else FAILED,
        "Both arms reached target poses." if ok
        else "Dual-arm motion planning failed.",
        left_pose=list(left_pose), right_pose=list(right_pose),
        left_ee_before=left_before, right_ee_before=right_before,
        left_ee_after=left_after, right_ee_after=right_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


def reach_both_tcp(TASK_ENV, left_xyz, right_xyz,
                   left_quat=None, right_quat=None) -> dict:
    """Synchronized dual-arm reach: place BOTH fingertips (TCP) at their world
    targets in ONE simultaneous motion (so an object is lifted level, not tilted).

    Each ``*_quat`` is optional — None keeps that arm's current orientation. Both
    TCP targets are converted to EE targets, then executed together via
    ``move_both_to_poses`` (→ env.together_move_to_pose).
    """
    ee_targets = {}
    for arm, xyz, quat in (("left", left_xyz, left_quat), ("right", right_xyz, right_quat)):
        gp = get_gripper_pose(TASK_ENV, arm)
        d = gp.get("data") or {}
        ee, tcp = d.get("pose"), d.get("tcp_pose")
        if not ee or not tcp or len(ee) != 7 or len(tcp) != 7:
            return make_primitive_result(
                "move_both_to_poses", FAILED,
                f"could not read {arm} ee/tcp pose.", plan_success=False)
        q = list(quat) if quat is not None else list(tcp[3:])
        ee_targets[arm] = pose_utils._tcp_target_to_ee_target(ee, tcp, xyz, q)
    return move_both_to_poses(TASK_ENV, ee_targets["left"], ee_targets["right"])


# ── Dual-arm delta move ───────────────────────────────────────────────────

def move_both_delta(TASK_ENV,
                    left_delta: Sequence[float],
                    right_delta: Sequence[float]) -> dict:
    """
    Move both arms simultaneously by world-frame displacements.
    Each delta is [dx, dy, dz].
    """
    for name, d in (("left_delta", left_delta), ("right_delta", right_delta)):
        if d is None or len(d) != 3:
            return make_primitive_result(
                "move_both_delta", FAILED,
                f"Invalid {name}: must be length 3, got {d!r}",
                left_delta=list(left_delta) if left_delta is not None else None,
                right_delta=list(right_delta) if right_delta is not None else None,
                plan_success=False,
            )
        msg = _validate_delta(d[0], d[1], d[2])
        if msg:
            return make_primitive_result(
                "move_both_delta", FAILED, f"Invalid {name}: {msg}",
                left_delta=list(left_delta), right_delta=list(right_delta),
                plan_success=False,
            )

    left_before = _safe_ee_pose(TASK_ENV, "left")
    right_before = _safe_ee_pose(TASK_ENV, "right")
    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(
            TASK_ENV.move_by_displacement(ArmTag("left"),
                                          x=float(left_delta[0]),
                                          y=float(left_delta[1]),
                                          z=float(left_delta[2])),
            TASK_ENV.move_by_displacement(ArmTag("right"),
                                          x=float(right_delta[0]),
                                          y=float(right_delta[1]),
                                          z=float(right_delta[2])),
        )
    except Exception as e:
        return make_primitive_result(
            "move_both_delta", FAILED, f"env.move raised: {e}",
            left_delta=list(left_delta), right_delta=list(right_delta),
            left_ee_before=left_before, right_ee_before=right_before,
            left_ee_after=_safe_ee_pose(TASK_ENV, "left"),
            right_ee_after=_safe_ee_pose(TASK_ENV, "right"),
            plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)
    left_after = _safe_ee_pose(TASK_ENV, "left")
    right_after = _safe_ee_pose(TASK_ENV, "right")
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_both_delta", SUCCESS if ok else FAILED,
        f"Both arms displaced (L={list(left_delta)}, R={list(right_delta)})." if ok
        else "Dual-arm displacement motion planning failed.",
        left_delta=list(left_delta), right_delta=list(right_delta),
        left_ee_before=left_before, right_ee_before=right_before,
        left_ee_after=left_after, right_ee_after=right_after,
        plan_success=ok, motion_completed=ok, sim_step=sim_step,
    )


# ── Home recovery ─────────────────────────────────────────────────────────

def move_to_home(TASK_ENV, arm: Optional[str] = None) -> dict:
    """
    Move one or both arms back to their original (home) pose.
    `arm=None` → both arms (left then right).
    """
    if arm is not None and arm not in ("left", "right"):
        return make_primitive_result(
            "move_to_home", FAILED,
            f"arm must be 'left', 'right', or None; got {arm!r}",
            arm=arm, plan_success=False,
        )

    arms = ("left", "right") if arm is None else (arm,)
    all_ok = True
    sub_results = []
    for a in arms:
        TASK_ENV.plan_success = True
        try:
            TASK_ENV.move(TASK_ENV.back_to_origin(ArmTag(a)))
            arm_ok = bool(TASK_ENV.plan_success)
        except Exception as e:
            arm_ok = False
            sub_results.append({"arm": a, "ok": False, "error": str(e)})
        else:
            sub_results.append({"arm": a, "ok": arm_ok})
        all_ok = all_ok and arm_ok

    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "move_to_home", SUCCESS if all_ok else FAILED,
        f"Home recovery: {sub_results}.",
        arm=arm, sub_results=sub_results,
        plan_success=all_ok, motion_completed=all_ok, sim_step=sim_step,
    )
