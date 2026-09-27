"""The internal numbers the harness still uses, declared to the model in one place.

The harness does not turn a measured quantity into a verdict. Two families of
numeric parameters survive that change, and both belong here rather than inside a tool result:

* **motion-completion conditions** — without them a controller would not stop issuing commands. The leg
  loop stops re-aiming inside `leg_convergence_tolerance_rule`; `reach_tcp`'s bounded correction
  stops refining inside `reach_correction_trigger_m`; the stall counter stops a loop that is no
  longer making progress. A no-progress stop is reported as ``ABORTED/stalled``; these parameters
  identify execution progress only and never classify contact or task intent.
* **safety guards** — the per-command displacement cap, gross trajectory-deviation bounds, and
  coarse world-frame workspace envelope stop physically invalid execution without task semantics.
  The contact-impulse threshold is only the declared sensor floor used in contact reports.

Declaring them once in the initial episode configuration keeps them out of every individual tool
payload while leaving the model able to interpret a guard event, a clamped step, or an
`effective_step_m` that is smaller than the `step_m` it asked for. Pure: no simulator, no toolbox
required.
"""
import math
import os

from codeaction.motion.motion_validation import (DISPLACEMENT_TOLERANCE_FRACTION,
                                       DISPLACEMENT_TOLERANCE_MAX_M,
                                       DISPLACEMENT_TOLERANCE_MIN_M,
                                       PLANNER_INTERPOLATION_DT_S,
                                       REACH_CORRECTION_MAX_ATTEMPTS,
                                       REACH_CORRECTION_TRIGGER_M,
                                       trajectory_deviation_rule)

DEFAULT_MAX_STEP_M = 0.30
DEFAULT_CONTACT_LEG_M = 0.03
CONTACT_IMPULSE_THRESHOLD = 1e-4
STALL_PROGRESS_FLOOR_M = 0.001
STALL_CONSECUTIVE_LEGS = 2
DENSE_STALL_WINDOW_S = 0.4
DENSE_STALL_ANGULAR_FLOOR_RAD = math.radians(1.0)
AIM_PITCH_MIN_DEG = 60.0
AIM_PITCH_MAX_DEG = 90.0
DEFAULT_WORKSPACE_ENVELOPE_WORLD_M = {
    "x_min": -0.8, "x_max": 0.8,
    "y_min": -0.8, "y_max": 0.8,
    "z_min": 0.4, "z_max": 1.5,
}


def validated_aim_pitch(pitch) -> float:
    """Validate one caller-selected world-Y TCP pre-rotation in the embodiment envelope."""
    if isinstance(pitch, bool):
        raise ValueError("camera_aim_pose pitch must be a number, not a boolean")
    try:
        value = float(pitch)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera_aim_pose pitch must be a number") from exc
    if not math.isfinite(value):
        raise ValueError("camera_aim_pose pitch must be finite")
    if not AIM_PITCH_MIN_DEG <= value <= AIM_PITCH_MAX_DEG:
        raise ValueError(
            f"camera_aim_pose pitch {value} is outside the inclusive "
            f"{AIM_PITCH_MIN_DEG}..{AIM_PITCH_MAX_DEG} degree range")
    return value

# Every episode host assigns CODEACTION_CUROBO_TABLE_WORLD=0 at import (mcp_episode_server, task_run,
# reference_scripted_gate, reference_admission_batch), so an episode ALWAYS runs with the fixed table
# cuboid removed. A caller that has to declare what the episode will run under -- the controller,
# which is a different PROCESS and never sets that variable -- must use this constant rather than
# read its own environment. Reading the ambient environment on both sides silently produced two
# different declared parameter blocks for one attempt, hence two different instruction-surface
# hashes, and the sim-side identity preflight refused the run with `instruction_surface`
# Reading the ambient environment on both sides is what produced this failure.
EPISODE_TABLE_WORLD_DISABLED = True


def planner_collision_world(table_world_disabled: bool = None) -> dict:
    """Exactly what the motion planner's world contains for this episode.

    The active benchmark planner checks this arm's self-collision only: episode hosts disable the
    fixed table cuboid before scene boot, and no task object, the other arm, or held geometry is
    ever inserted into cuRobo's world. This is a configuration constant — identical for every
    query of every episode — so it belongs here with the other declared parameters rather than in
    each `check_tcp_pose_reachability` result, where it was 106 bytes repeated on the most
    frequently called tool (97/97 calls identical in the 26.x corpus, 0 reads in agent code).
    Stating it structurally rather than only as prose is what keeps the word "collision-aware" from
    silently implying scene coverage the planner does not have.
    """
    if table_world_disabled is None:
        table_world_disabled = EPISODE_TABLE_WORLD_DISABLED
    return {
        "robot_self": True,
        "table": not bool(table_world_disabled),
        "scene_objects": False,
        "other_arm": False,
        "attached_object": False,
    }


def declared_harness_parameters(toolbox=None, *, orientation_anchor=None,
                                table_world_disabled=None) -> dict:
    """Return the actual declared parameters for one runtime configuration."""
    max_step = getattr(toolbox, "_max_step", None) if toolbox is not None else None
    contact_leg = getattr(toolbox, "_leg", None) if toolbox is not None else None
    live_anchor = (getattr(toolbox, "_orientation_anchor", None)
                   if toolbox is not None else None)
    anchor = live_anchor if orientation_anchor is None else orientation_anchor
    if table_world_disabled is None:
        table_world_disabled = os.environ.get(
            "CODEACTION_CUROBO_TABLE_WORLD", "1").strip().lower() in {"0", "false", "off", "no"}
    table_world_disabled = bool(table_world_disabled)
    live_workspace = getattr(getattr(toolbox, "_pu", None), "DEFAULT_WORKSPACE", None)
    workspace_source = (live_workspace if isinstance(live_workspace, dict)
                        and all(key in live_workspace
                                for key in DEFAULT_WORKSPACE_ENVELOPE_WORLD_M)
                        else DEFAULT_WORKSPACE_ENVELOPE_WORLD_M)
    workspace = {
        key: float(workspace_source[key]) for key in DEFAULT_WORKSPACE_ENVELOPE_WORLD_M
    }
    return {
        "max_step_m": float(max_step if max_step is not None else DEFAULT_MAX_STEP_M),
        "contact_leg_m": float(
            contact_leg if contact_leg is not None else DEFAULT_CONTACT_LEG_M),
        "leg_convergence_tolerance_rule": (
            f"min({DISPLACEMENT_TOLERANCE_MAX_M}, "
            f"max({DISPLACEMENT_TOLERANCE_MIN_M}, "
            f"{DISPLACEMENT_TOLERANCE_FRACTION}*|commanded displacement|)) metres"),
        "reach_correction_trigger_m": float(REACH_CORRECTION_TRIGGER_M),
        "reach_correction_max_attempts": int(REACH_CORRECTION_MAX_ATTEMPTS),
        "planner_interpolation_dt_s": float(PLANNER_INTERPOLATION_DT_S),
        "trajectory_deviation_rule": (
            "one executed leg is refused as a gross departure when "
            f"{trajectory_deviation_rule()}; the bounds in force for the triggering leg are "
            "returned in achieved.guard.bound"),
        "contact_impulse_threshold": float(CONTACT_IMPULSE_THRESHOLD),
        "stall_progress_floor_m": float(STALL_PROGRESS_FLOOR_M),
        "stall_consecutive_legs": int(STALL_CONSECUTIVE_LEGS),
        "dense_stall_window_s": float(DENSE_STALL_WINDOW_S),
        "dense_stall_angular_floor_rad": float(DENSE_STALL_ANGULAR_FLOOR_RAD),
        "orientation_anchor": bool(True if anchor is None else anchor),
        "workspace_envelope_world_m": workspace,
        "workspace_envelope_note": (
            "coarse safety envelope applied before simulation execution by absolute-TCP reach "
            "and reachability calls. It is not an IK-reachability, scene-collision, or "
            "collision-free-space proof."),
        "curobo_world_model": (
            "none; robot self-collision checking remains enabled"
            if table_world_disabled else "legacy fixed-table cuboid"),
        # The structured counterpart of curobo_world_model, and the single declaration of what
        # check_tcp_pose_reachability actually checked. Its results no longer repeat it.
        "planner_collision_world": planner_collision_world(table_world_disabled),
        "contact_handling": (
            "contact identity never permits or interrupts an action. Waypoint motion stops after "
            "stall_consecutive_legs each make less than stall_progress_floor_m progress, then "
            "reports blocked_in_contact, contact_parts, and pose_at_contact from that boundary. "
            "The contact read uses contact_impulse_threshold only as a sensor floor and does not "
            "decide task intent or severity"),
        "note": "These are the harness's own loop-termination conditions and safety-guard bounds. "
                "They decide when a loop stops issuing commands and when a declared safety guard "
                "interrupts the arm; they never decide task success or whether a motion was "
                "accurate enough for your purpose. "
                "workspace_envelope_world_m is a pre-execution safety refusal bound, not a claim "
                "that every point inside is reachable or collision-free. "
                "contact_leg_m also caps probe_contact_along's step_m, which is why a result "
                "reports effective_step_m as its maximum nominal waypoint spacing; a final or "
                "re-aimed leg may be shorter.",
    }
