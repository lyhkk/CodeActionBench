"""
Perception primitives — read-only views of the simulation state.

These delegate to `privileged_perception.get_scene_objects` and the robot's
own EE pose accessors. No side effects on the env.
"""

from typing import Optional

from .result import (
    SUCCESS, FAILED, STAGE_PERCEPTION,
    make_primitive_result,
)


# Lazy import to avoid circulars; privileged_perception lives next to this pkg
def _scene_objects(TASK_ENV) -> dict:
    from privileged_perception import get_scene_objects
    return get_scene_objects(TASK_ENV)


# ── Object pose ───────────────────────────────────────────────────────────

def get_object_pose(TASK_ENV, object_name: str) -> dict:
    """
    Read world pose of `object_name` from SAPIEN.

    Returns a PrimitiveResult whose `data` contains:
        position: [x, y, z]
        orientation: [qw, qx, qy, qz]  (SAPIEN convention)
    """
    objects = _scene_objects(TASK_ENV)
    if object_name not in objects:
        return make_primitive_result(
            "get_object_pose", FAILED,
            f"Object '{object_name}' not found in scene. "
            f"Known names: {list(objects.keys())[:10]}",
            object_name=object_name, position=None, orientation=None,
        )
    info = objects[object_name]
    return make_primitive_result(
        "get_object_pose", SUCCESS,
        f"Pose of '{object_name}' read.",
        object_name=object_name,
        position=list(info["position"]),
        orientation=list(info["orientation"]),
    )


# ── Gripper EE pose ───────────────────────────────────────────────────────

# Approximate maximum gripper opening width (meters).  Used to convert the
# normalised gripper_val [0,1] returned by the robot wrapper into a
# meters-scale `gripper_width_m`, so LLMs can compare it directly with
# `extent_world` numbers from VisionPerception.  Per embodiment the exact
# max width varies; this constant is the typical aloha-agilex value.
_APPROX_MAX_GRIPPER_WIDTH_M = 0.08


def _read_gripper_width(TASK_ENV, arm: str) -> Optional[float]:
    """Best-effort read of the current gripper opening in meters.

    Source of truth is ``robot.get_*_gripper_val()`` which returns a value
    in ``[0, 1]`` (open = 1).  We rescale by an approximate max width so the
    LLM sees a meters-scale number it can compare with ``extent_world``.
    Returns ``None`` on failure.
    """
    try:
        if arm == "left":
            val = float(TASK_ENV.robot.get_left_gripper_val())
        else:
            val = float(TASK_ENV.robot.get_right_gripper_val())
    except Exception:
        return None
    val = max(0.0, min(1.0, val))
    return val * _APPROX_MAX_GRIPPER_WIDTH_M


def get_gripper_pose(TASK_ENV, arm: str) -> dict:
    """
    Return current end-effector pose for `arm` ("left" or "right").

    Pose format follows env.get_*_ee_pose: [x, y, z, qw, qx, qy, qz].
    Additionally returns ``gripper_width_m`` (approximate finger opening in
    meters, derived from the normalised gripper joint value).
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_gripper_pose", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm, pose=None, tcp_pose=None, gripper_width_m=None,
        )
    try:
        if arm == "left":
            pose = list(TASK_ENV.robot.get_left_ee_pose())
            tcp_pose = list(TASK_ENV.robot.get_left_tcp_pose())
        else:
            pose = list(TASK_ENV.robot.get_right_ee_pose())
            tcp_pose = list(TASK_ENV.robot.get_right_tcp_pose())
    except Exception as e:
        return make_primitive_result(
            "get_gripper_pose", FAILED,
            f"Failed to read {arm} EE/TCP pose: {e}",
            arm=arm, pose=None, tcp_pose=None, gripper_width_m=None,
        )
    gripper_width_m = _read_gripper_width(TASK_ENV, arm)
    return make_primitive_result(
        "get_gripper_pose", SUCCESS,
        f"{arm} EE + TCP pose read.",
        arm=arm, pose=pose, tcp_pose=tcp_pose,
        gripper_width_m=(round(gripper_width_m, 4)
                         if gripper_width_m is not None else None),
    )


# ── Gripper open/close state ──────────────────────────────────────────────

def get_gripper_state(TASK_ENV, arm: str) -> dict:
    """
    Return open/closed status and raw gripper value for `arm`.
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_gripper_state", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}",
            arm=arm,
        )
    try:
        if arm == "left":
            val = float(TASK_ENV.robot.get_left_gripper_val())
            is_closed = bool(TASK_ENV.robot.is_left_gripper_close())
        else:
            val = float(TASK_ENV.robot.get_right_gripper_val())
            is_closed = bool(TASK_ENV.robot.is_right_gripper_close())
    except Exception as e:
        return make_primitive_result(
            "get_gripper_state", FAILED,
            f"Failed to read {arm} gripper state: {e}",
            arm=arm,
        )
    return make_primitive_result(
        "get_gripper_state", SUCCESS,
        f"{arm} gripper state: {'closed' if is_closed else 'open'} (val={val:.3f}).",
        arm=arm, gripper_val=val, is_closed=is_closed,
    )


# ── Tactile grasp check (proprioceptive) ─────────────────────────────────

# Stall-depth proxy: if the gripper's normalised val exceeds commanded_pos by
# more than this threshold the fingers are resisted by an object.  Tunable
# after first sim run; 0.04 is conservative for a [0,1]-normalised value.
_GRIP_CONTACT_THRESHOLD = 0.04


def _read_joint_effort(TASK_ENV, arm: str):
    """Gripper joint motor effort (proprioceptive), or None.

    Step-1 investigation found no clean get_qf() / get_drive_force() on the
    gripper joint objects — only get_drive_target() (commanded position) is
    exposed.  Returning None causes the caller to use the stall-depth proxy.

    If a future SAPIEN / robot.py version exposes joint efforts, replace the
    body here with e.g.:
        joints = TASK_ENV.robot.left_gripper if arm == "left" \
                 else TASK_ENV.robot.right_gripper
        qf = joints[0][0].get_qf()
        return float(abs(qf).max())
    """
    return None  # no clean effort accessor -> caller uses stall proxy


def get_grip_force(TASK_ENV, arm: str, commanded_pos: float = 0.0) -> dict:
    """Tactile grasp check: do the fingers feel resistance (object held)?

    Prefers motor joint effort (qf); falls back to stall-depth proxy.
    NEVER reads scene contact impulses (privileged). in_contact=True = held.

    Args:
        TASK_ENV:       the simulation environment (robot accessor required).
        arm:            "left" or "right".
        commanded_pos:  the normalised gripper position that was commanded
                        (0.0 = fully closed, 1.0 = fully open).  The stall
                        proxy computes grip_force = max(0, actual - commanded).
                        Pass the value you used in close_gripper(pos=...).

    Returns a PrimitiveResult whose ``data`` contains:
        arm:          "left" or "right"
        grip_force:   float >= 0  (effort units or stall-depth proxy units)
        in_contact:   bool  — True when grip_force > _GRIP_CONTACT_THRESHOLD
        source:       "joint_effort" | "stall_proxy"
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_grip_force", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}", arm=arm,
        )

    effort = _read_joint_effort(TASK_ENV, arm)
    if effort is not None:
        force = float(effort)
        source = "joint_effort"
    else:
        try:
            val = float(
                TASK_ENV.robot.get_left_gripper_val() if arm == "left"
                else TASK_ENV.robot.get_right_gripper_val()
            )
        except Exception as e:
            return make_primitive_result(
                "get_grip_force", FAILED,
                f"cannot read gripper val: {e}", arm=arm,
            )
        force = max(0.0, val - float(commanded_pos))
        source = "stall_proxy"

    in_contact = force > _GRIP_CONTACT_THRESHOLD
    return make_primitive_result(
        "get_grip_force", SUCCESS,
        f"{arm} grip_force={force:.3f} ({source}), in_contact={in_contact}.",
        arm=arm, grip_force=round(force, 4),
        in_contact=in_contact, source=source,
    )


# ── Contact-impulse grasp verification (replaces the thin-handle stall proxy) ─
#
# The stall proxy false-negatives on a thin handle (the gripper closes almost
# fully around it, so val≈commanded → "no contact" even while the object is
# held).  This reads the SAPIEN contact impulse between the arm's gripper
# fingers and the grasped object instead — a true grasp registers a non-zero
# finger↔object impulse regardless of handle thickness.
#
# NOTE: this is a PRIVILEGED signal (reads sim contacts), kept deliberately
# isolated so it can be swapped later for an honest, vision-based "does the
# object move with the gripper" check.  Threshold tuned from the no-LLM A/B/C
# grasp test (success vs no-contact separation).
_GRASP_IMPULSE_THRESHOLD = 1e-4


def _aggregate_grasp_contact(contacts, finger_links, object_name=None):
    """Pure: from a list of SAPIEN contacts, sum the magnitude of contact
    impulses between this arm's gripper FINGERS and the target object.

    Returns ``(total_impulse, per_finger_impulse, n_contact_points)``.  A
    contact counts only when one body is a finger link and (if ``object_name``
    is given) the other is that object.
    """
    def _mag(imp):
        return float(sum(float(c) ** 2 for c in imp) ** 0.5)

    total = 0.0
    per_finger = {}
    n_pts = 0
    for c in contacts:
        n0 = c.bodies[0].entity.name
        n1 = c.bodies[1].entity.name
        if n0 in finger_links:
            finger, other = n0, n1
        elif n1 in finger_links:
            finger, other = n1, n0
        else:
            continue
        if object_name is not None and other != object_name:
            continue
        imp = sum(_mag(p.impulse) for p in c.points)
        total += imp
        per_finger[finger] = per_finger.get(finger, 0.0) + imp
        n_pts += len(c.points)
    return total, per_finger, n_pts


def get_grasp_contact(TASK_ENV, arm: str, object_name: Optional[str] = None,
                      impulse_thresh: float = _GRASP_IMPULSE_THRESHOLD) -> dict:
    """Grasp check via gripper-finger↔object contact impulse (does NOT
    false-negative on a thin handle, unlike the stall proxy).

    ``in_contact`` is True when the total finger↔object impulse exceeds
    ``impulse_thresh``; ``fingers_in_contact`` (how many of the arm's finger
    links register impulse) is reported for a stricter "both-sided pinch"
    criterion if needed.  Privileged sim signal — temporary; see module note.
    """
    if arm not in ("left", "right"):
        return make_primitive_result(
            "get_grasp_contact", FAILED,
            f"arm must be 'left' or 'right', got {arm!r}", arm=arm)
    try:
        grip = TASK_ENV.robot.left_gripper if arm == "left" else TASK_ENV.robot.right_gripper
        finger_links = {g[0].child_link.get_name() for g in grip}
        contacts = TASK_ENV.scene.get_contacts()
    except Exception as e:
        return make_primitive_result(
            "get_grasp_contact", FAILED, f"could not read contacts: {e}", arm=arm)

    total, per_finger, n_pts = _aggregate_grasp_contact(contacts, finger_links, object_name)
    n_fingers = sum(1 for v in per_finger.values() if v > impulse_thresh)
    # A parallel-jaw grasp = the object pinched between BOTH fingers. Total
    # impulse alone false-positives when a wrongly-oriented gripper RAMS the
    # object with ONE finger (verified no-LLM on the A/B/C grasp test: a no-grasp
    # ram gave total 0.27 with 1 finger, while real grasps gave ~0.10 with 2
    # fingers — both-finger count, not magnitude, tracks the pot-lift ground
    # truth). So require both finger links in contact.
    in_contact = n_fingers >= 2
    return make_primitive_result(
        "get_grasp_contact", SUCCESS,
        f"{arm} grasp: fingers_in_contact={n_fingers}/2, total_impulse={total:.4f}, "
        f"points={n_pts}, in_contact={in_contact}.",
        arm=arm, object_name=object_name,
        total_impulse=round(total, 5), fingers_in_contact=n_fingers,
        n_contact_points=n_pts, per_finger={k: round(v, 5) for k, v in per_finger.items()},
        in_contact=in_contact, source="contact_impulse",
    )


# ── Gripper geometry (coarse constants for LLM clearance reasoning) ──────

# Approximate gripper geometry (meters) for collision-clearance reasoning.
# Coarse on purpose — the LLM combines these with visual judgment, and
# cuRobo enforces actual collision avoidance during execution. Refine per
# embodiment from URDF if precise clearance is ever needed.
_GRIPPER_GEOMETRY = {
    "finger_length_m": 0.09,     # how far fingertips reach past the TCP frame
    "max_opening_m": 0.08,       # fingertip separation when fully open
    "finger_thickness_m": 0.02,  # each finger's thickness
}


def get_gripper_geometry(TASK_ENV, arm: str = "left") -> dict:
    """Approximate gripper dimensions so the LLM can keep clearance and avoid
    ramming the object. Coarse constants; cuRobo still enforces real collision
    avoidance during motion."""
    g = dict(_GRIPPER_GEOMETRY)
    return make_primitive_result(
        "get_gripper_geometry", SUCCESS,
        (f"gripper: fingers reach ~{g['finger_length_m']} m past TCP, "
         f"open to ~{g['max_opening_m']} m, each finger ~"
         f"{g['finger_thickness_m']} m thick."),
        arm=arm, **g,
    )


# ── Camera snapshot ──────────────────────────────────────────────────────

VALID_CAMERAS = ("head_camera", "left_camera", "right_camera")


def get_camera_snapshot(TASK_ENV, camera_name: str, save_path: str) -> dict:
    """
    Render and save an RGB image from the specified camera.

    Returns a PrimitiveResult whose ``data`` contains:
        camera_name: str
        save_path: str   (always a plain str, JSON-safe)

    The image is written to *save_path* on disk.  The caller (TaP runtime)
    is responsible for creating the parent directory beforehand.
    """
    if camera_name not in VALID_CAMERAS:
        return make_primitive_result(
            "get_camera_snapshot", FAILED,
            f"camera_name must be one of {VALID_CAMERAS}, got {camera_name!r}",
            camera_name=camera_name, save_path=None,
        )
    try:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        TASK_ENV.save_camera_rgb(str(save_path), camera_name)
    except Exception as e:
        return make_primitive_result(
            "get_camera_snapshot", FAILED,
            f"Failed to save camera RGB: {e}",
            camera_name=camera_name, save_path=str(save_path),
        )
    return make_primitive_result(
        "get_camera_snapshot", SUCCESS,
        f"Snapshot saved from {camera_name}.",
        camera_name=camera_name, save_path=str(save_path),
    )
