"""Legacy module path for the public actions that can advance simulation physics.

The set is an action-accounting inventory, not a contact permission table.  Contact identity no
longer changes execution; waypoint stalls report whatever contact coexists with the stop.
"""
from typing import Mapping


PHYSICS_ACTION_TOOLS = frozenset({
    "capture_motion_pair",
    "move_delta",
    "move_both_delta",
    "reach_tcp",
    "reach_both_tcp",
    "probe_contact_along",
    "set_gripper",
})


def is_same_gripper_finger_pair(key, finger_names: Mapping[str, set]) -> bool:
    """Whether an opaque robot-to-robot key is one gripper's two fingers touching each other.

    A closing gripper that meets nothing closes onto itself, and the solver reports that as a
    robot-to-robot contact like any other. It is not a collision by any reading: no third body is
    involved, the motion that caused it is the commanded one, and the arm is where it was.

    Measured over the shadow campaign (`data/contact_shadow/`, 2026-08-17), every one of the nine
    above-threshold events of this shape came from behaviour that must not be interrupted -- the
    experts for click_bell, handover_mic, open_microwave, press_stapler and stack_bowls_three, plus
    a deliberate close on empty air. None was harmful, so this class is excluded here rather than
    weighed. The measurement instrument keeps recording them: `codeaction.motion.contact_features` reports
    every pair, and only this rule drops them.
    """
    if not key or key[0] != "robot" or len(key) != 3:
        return False
    fingers = {arm: set(links or ()) for arm, links in (finger_names or {}).items()}
    return any(set(key[1:]) <= links for links in fingers.values() if links)


__all__ = [
    "PHYSICS_ACTION_TOOLS",
    "is_same_gripper_finger_pair",
]
