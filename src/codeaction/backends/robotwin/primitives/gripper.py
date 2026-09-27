"""
Gripper primitives — open/close via env.move(env.<open|close>_gripper(...)).
"""

from typing import Optional

from envs.utils.action import ArmTag

from .result import SUCCESS, FAILED, make_primitive_result
from .perception import get_gripper_state


def _arm_tag(arm: str) -> ArmTag:
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return ArmTag(arm)


def open_gripper(TASK_ENV, arm: str, pos: float = 1.0) -> dict:
    """
    Open one gripper. `pos` defaults to 1.0 (fully open).
    """
    before = get_gripper_state(TASK_ENV, arm)
    gripper_before = before["data"].get("gripper_val") if before["status"] == SUCCESS else None

    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.open_gripper(_arm_tag(arm), pos=float(pos)))
    except Exception as e:
        return make_primitive_result(
            "open_gripper", FAILED, f"env.move raised: {e}",
            arm=arm, pos=float(pos),
            gripper_before=gripper_before, plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)

    after = get_gripper_state(TASK_ENV, arm)
    gripper_after = after["data"].get("gripper_val") if after["status"] == SUCCESS else None
    is_closed = after["data"].get("is_closed") if after["status"] == SUCCESS else None
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "open_gripper", SUCCESS if ok else FAILED,
        f"{arm} gripper open (val={gripper_after})." if ok
        else f"{arm} open_gripper motion planning failed.",
        arm=arm, pos=float(pos),
        gripper_before=gripper_before, gripper_after=gripper_after,
        is_closed=is_closed, plan_success=ok, sim_step=sim_step,
    )


def close_gripper(TASK_ENV, arm: str, pos: float = 0.0) -> dict:
    """
    Close one gripper. `pos` defaults to 0.0 (fully closed).
    """
    before = get_gripper_state(TASK_ENV, arm)
    gripper_before = before["data"].get("gripper_val") if before["status"] == SUCCESS else None

    TASK_ENV.plan_success = True
    try:
        TASK_ENV.move(TASK_ENV.close_gripper(_arm_tag(arm), pos=float(pos)))
    except Exception as e:
        return make_primitive_result(
            "close_gripper", FAILED, f"env.move raised: {e}",
            arm=arm, pos=float(pos),
            gripper_before=gripper_before, plan_success=False,
        )
    ok = bool(TASK_ENV.plan_success)

    after = get_gripper_state(TASK_ENV, arm)
    gripper_after = after["data"].get("gripper_val") if after["status"] == SUCCESS else None
    is_closed = after["data"].get("is_closed") if after["status"] == SUCCESS else None
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "close_gripper", SUCCESS if ok else FAILED,
        f"{arm} gripper closed (val={gripper_after}, is_closed={is_closed})." if ok
        else f"{arm} close_gripper motion planning failed.",
        arm=arm, pos=float(pos),
        gripper_before=gripper_before, gripper_after=gripper_after,
        is_closed=is_closed, plan_success=ok, sim_step=sim_step,
    )


def wait_steps(TASK_ENV, n: int) -> dict:
    """
    Let physics settle for `n` steps without moving. Caps at 200 for safety.
    Uses TASK_ENV.delay which holds current gripper values.
    """
    n = int(n)
    if n < 0:
        return make_primitive_result(
            "wait_steps", FAILED, f"n must be >= 0, got {n}",
            n=n, steps_waited=0,
        )
    if n > 200:
        return make_primitive_result(
            "wait_steps", FAILED, f"n {n} exceeds 200-step safety cap",
            n=n, steps_waited=0,
        )
    try:
        TASK_ENV.delay(n)
    except Exception as e:
        return make_primitive_result(
            "wait_steps", FAILED, f"TASK_ENV.delay raised: {e}",
            n=n, steps_waited=0,
        )
    sim_step = int(getattr(TASK_ENV, "take_action_cnt", -1))
    return make_primitive_result(
        "wait_steps", SUCCESS, f"Waited {n} physics steps.",
        n=n, steps_waited=n, sim_step=sim_step,
    )
