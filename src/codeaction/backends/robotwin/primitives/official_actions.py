"""
Official RoboTwin API wrappers.

These primitives call the official TASK_ENV methods (grasp_actor, move,
move_by_displacement) that the expert play_once() uses. They produce
correct grasp poses via the env's internal contact matrices and motion
planner, avoiding hand-written quaternion computation.

Security: TASK_ENV is passed as first arg (captured by dispatch closure),
never exposed to the Executor. The Executor sees only structured op-dicts.
"""

from typing import Optional
from .result import SUCCESS, FAILED, make_primitive_result
from .perception import get_object_pose, get_gripper_state


def _resolve_actor(TASK_ENV, object_name: str):
    """Resolve object_name to a SAPIEN Actor with get_contact_point."""
    # Direct attribute lookup (e.g. TASK_ENV.pot for lift_pot)
    for candidate in ["pot", "target_object", "obj", "actor", "model"]:
        obj = getattr(TASK_ENV, candidate, None)
        if obj is not None and hasattr(obj, "get_contact_point"):
            return obj
    # Scan all attributes for name match
    for attr in dir(TASK_ENV):
        if attr.startswith("_"):
            continue
        obj = getattr(TASK_ENV, attr, None)
        if (obj is not None and hasattr(obj, "get_contact_point")
                and hasattr(obj, "get_pose")):
            name = str(getattr(obj, "name", ""))
            if object_name in name or name in object_name:
                return obj
    return None


def dual_grasp_actor(TASK_ENV,
                     object: str = None,
                     object_name: str = None,
                     left_contact_point_id: int = 0,
                     right_contact_point_id: int = 1,
                     pre_grasp_dis: float = 0.035,
                     grasp_dis: float = 0.0,
                     gripper_pos: float = 0.0,
                     preclose_gripper_pos: float = 0.5,
                     **_extra) -> dict:
    """
    Dual-arm grasp using the official RoboTwin grasp_actor API.

    Mirrors lift_pot.play_once():
        self.move(
            self.grasp_actor(actor, left, pre_grasp_dis=0.035, contact_point_id=0),
            self.grasp_actor(actor, right, pre_grasp_dis=0.035, contact_point_id=1),
        )
    """
    from envs.utils.action import ArmTag

    name = object or object_name
    if not name:
        return make_primitive_result(
            "dual_grasp_actor", FAILED,
            "missing 'object' or 'object_name' arg",
            plan_success=False, method="official_grasp_actor",
        )

    # Record object pose before
    pose_before = get_object_pose(TASK_ENV, name)
    obj_z_before = None
    if pose_before["status"] == SUCCESS:
        obj_z_before = pose_before["data"]["position"][2]

    # Resolve actor
    actor = _resolve_actor(TASK_ENV, name)
    if actor is None:
        return make_primitive_result(
            "dual_grasp_actor", FAILED,
            f"Actor for '{name}' not found in TASK_ENV",
            object=name, plan_success=False, method="official_grasp_actor",
        )

    # Optional: pre-close grippers to half (like play_once does)
    preclose_ok = True
    preclose_error = None
    try:
        TASK_ENV.plan_success = True
        TASK_ENV.move(
            TASK_ENV.close_gripper(ArmTag("left"), pos=float(preclose_gripper_pos)),
            TASK_ENV.close_gripper(ArmTag("right"), pos=float(preclose_gripper_pos)),
        )
        preclose_ok = bool(TASK_ENV.plan_success)
    except Exception as e:
        preclose_ok = False
        preclose_error = str(e)

    # ── Lightweight diagnostics (no planner calls to avoid state corruption) ─
    print(f"\n[dual_grasp_actor] actor={type(actor).__name__}, "
          f"is_pot={actor is getattr(TASK_ENV, 'pot', None)}, "
          f"plan_success={TASK_ENV.plan_success}, "
          f"need_plan={getattr(TASK_ENV, 'need_plan', '?')}")
    for cpid in [int(left_contact_point_id), int(right_contact_point_id)]:
        try:
            cp_matrix = actor.get_contact_point(cpid, "matrix")
            print(f"[dual_grasp_actor] contact_point[{cpid}] "
                  f"matrix={'OK '+str(cp_matrix.shape) if cp_matrix is not None else 'None'}")
        except Exception as e:
            print(f"[dual_grasp_actor] contact_point[{cpid}] ERROR: {e}")

    # ── Build grasp Action sequences manually (call choose_grasp_pose ONCE
    #    per arm, then construct Actions).  This avoids double-planning:
    #    grasp_actor() would call choose_grasp_pose() again internally with
    #    different RRT outcomes, wasting the successful plan. ──────────────
    from envs.utils.action import Action
    TASK_ENV.plan_success = True
    arm_actions = {}  # arm -> (ArmTag, [Action, ...])
    for arm, cpid in [("left", int(left_contact_point_id)),
                      ("right", int(right_contact_point_id))]:
        if not TASK_ENV.plan_success:
            return make_primitive_result(
                "dual_grasp_actor", FAILED,
                f"plan_success became False before planning {arm} arm",
                object=name, plan_success=False, method="official_grasp_actor",
            )
        try:
            cgp = TASK_ENV.choose_grasp_pose(
                actor, arm_tag=arm,
                pre_dis=float(pre_grasp_dis),
                target_dis=float(grasp_dis),
                contact_point_id=cpid,
            )
        except Exception as e:
            return make_primitive_result(
                "dual_grasp_actor", FAILED,
                f"choose_grasp_pose({arm}) raised: {e}",
                object=name, plan_success=False, method="official_grasp_actor",
                arm=arm, contact_point_id=cpid,
            )
        if cgp is None:
            return make_primitive_result(
                "dual_grasp_actor", FAILED,
                f"choose_grasp_pose({arm}) returned None (plan_success was False?)",
                object=name, plan_success=False, method="official_grasp_actor",
            )
        pre_pose, target_pose = cgp
        if pre_pose is None or target_pose is None:
            return make_primitive_result(
                "dual_grasp_actor", FAILED,
                f"Motion planner found no valid path to grasp pose for {arm} arm "
                f"(contact_point_id={cpid}). The expert play_once() would also "
                f"fail on this seed.",
                object=name, plan_success=False, method="official_grasp_actor",
                arm=arm, contact_point_id=cpid,
                preclose_ok=preclose_ok,
            )
        print(f"[dual_grasp_actor] choose_grasp_pose({arm}) OK: "
              f"pre={[round(v,3) for v in pre_pose[:3]]}, "
              f"grasp={[round(v,3) for v in target_pose[:3]]}")
        # Construct Action list (mirrors grasp_actor logic in _base_task.py)
        if pre_pose == target_pose:
            actions = [
                Action(arm, "move", target_pose=pre_pose),
                Action(arm, "close", target_gripper_pos=float(gripper_pos)),
            ]
        else:
            actions = [
                Action(arm, "move", target_pose=pre_pose),
                Action(arm, "move", target_pose=target_pose,
                       constraint_pose=[1, 1, 1, 0, 0, 0]),
                Action(arm, "close", target_gripper_pos=float(gripper_pos)),
            ]
        arm_actions[arm] = (arm, actions)

    left_actions = arm_actions["left"]
    right_actions = arm_actions["right"]

    # Execute synchronized dual-arm grasp
    TASK_ENV.plan_success = True
    try:
        result = TASK_ENV.move(left_actions, right_actions)
    except Exception as e:
        import traceback
        return make_primitive_result(
            "dual_grasp_actor", FAILED,
            f"move() raised during grasp: {e}\\n{traceback.format_exc()}",
            object=name, plan_success=False, method="official_grasp_actor",
        )

    plan_ok = bool(TASK_ENV.plan_success)

    # Record state after
    pose_after = get_object_pose(TASK_ENV, name)
    obj_z_after = None
    if pose_after["status"] == SUCCESS:
        obj_z_after = pose_after["data"]["position"][2]

    left_grip = get_gripper_state(TASK_ENV, "left")
    right_grip = get_gripper_state(TASK_ENV, "right")

    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))

    if not plan_ok:
        return make_primitive_result(
            "dual_grasp_actor", FAILED,
            f"Dual grasp move failed (plan_success=False)",
            object=name, plan_success=False,
            method="official_grasp_actor",
            obj_z_before=obj_z_before, obj_z_after=obj_z_after,
            left_gripper=left_grip.get("data", {}).get("gripper_val"),
            right_gripper=right_grip.get("data", {}).get("gripper_val"),
            sim_step=sim_step,
            preclose_ok=preclose_ok,
            preclose_error=preclose_error,
            preclose_gripper_pos=preclose_gripper_pos,
        )

    return make_primitive_result(
        "dual_grasp_actor", SUCCESS,
        f"Dual grasp succeeded via official API on '{name}'",
        object=name, plan_success=True, motion_completed=True,
        method="official_grasp_actor",
        pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
        gripper_pos=gripper_pos,
        left_contact_point_id=left_contact_point_id,
        right_contact_point_id=right_contact_point_id,
        obj_z_before=obj_z_before, obj_z_after=obj_z_after,
        left_gripper=left_grip.get("data", {}).get("gripper_val"),
        right_gripper=right_grip.get("data", {}).get("gripper_val"),
        sim_step=sim_step,
        preclose_ok=preclose_ok,
        preclose_error=preclose_error,
        preclose_gripper_pos=preclose_gripper_pos,
    )
