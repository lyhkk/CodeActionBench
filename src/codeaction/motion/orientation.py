"""Pure gripper-orientation geometry for the codeaction tool surface.

The caller supplies the manipulation semantics as two world-frame axes.  This module only converts
those axes to robot pose convention; it does not inspect a scene or evaluate arm reachability.
"""
import numpy as np


# The two caller-selected axes define a rotation only when they are orthogonal. Values outside this
# published tolerance are semantic input errors, not numerical noise that the bridge may silently
# reinterpret. Residuals inside the tolerance are removed to construct a proper rotation, and the
# result reports both normalized caller inputs and whether that numerical correction occurred.
AXIS_ORTHOGONALITY_TOLERANCE = 1e-6


def _unit_axis(value, name):
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite numbers")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        raise ValueError(f"{name} must be non-zero")
    return axis / norm


def _mat_to_quat_wxyz(matrix):
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    R = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w, x, y, z = (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        w, x, y, z = ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        w, x, y, z = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        w, x, y, z = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def grasp_quat_candidates(approach_axis_world, opening_axis_world):
    """Return two wxyz orientations for an approach direction and finger-opening line.

    In this embodiment, TCP local +x is the EE-to-TCP approach axis and local +y is the line
    joining the fingertips.  The input opening axis is an unoriented line, so reversing its sign
    swaps finger identities without changing the parallel-jaw grasp geometry.  The resulting two
    poses differ by 180 degrees of wrist roll and can have different arm reachability.
    """
    approach = _unit_axis(approach_axis_world, "approach_axis_world")
    opening_input = _unit_axis(opening_axis_world, "opening_axis_world")
    input_dot = float(np.dot(opening_input, approach))
    if abs(input_dot) > AXIS_ORTHOGONALITY_TOLERANCE:
        raise ValueError(
            "approach_axis_world and opening_axis_world must be orthogonal after normalization: "
            f"abs(dot)={abs(input_dot):.9g} exceeds "
            f"{AXIS_ORTHOGONALITY_TOLERANCE:.0e}")
    opening = opening_input - input_dot * approach
    opening_norm = float(np.linalg.norm(opening))
    if opening_norm < 1e-12:
        raise ValueError("opening_axis_world cannot define an axis orthogonal to approach_axis_world")
    opening /= opening_norm

    candidates = []
    for label, signed_opening in (("opening_sign_as_given", opening),
                                  ("opening_sign_reversed", -opening)):
        local_z = np.cross(approach, signed_opening)
        rotation = np.column_stack([approach, signed_opening, local_z])
        quat = _mat_to_quat_wxyz(rotation)
        candidates.append({
            "label": label,
            "quat_wxyz": [float(v) for v in quat],
            "approach_axis_world": [float(v) for v in approach],
            "opening_axis_world": [float(v) for v in signed_opening],
        })
    return {
        "candidates": candidates,
        "input_axes": {
            "approach_axis_world_normalized": [float(v) for v in approach],
            "opening_axis_world_normalized": [float(v) for v in opening_input],
            "normalized_dot": input_dot,
            "orthogonality_tolerance": AXIS_ORTHOGONALITY_TOLERANCE,
            "orthogonalization_applied": bool(
                abs(input_dot) > 32.0 * np.finfo(float).eps),
        },
        "opening_axis_input_is_unoriented_line": True,
        "reachability_evaluated": False,
        "note": (
            "Candidates swap finger identities and differ by 180-degree wrist roll; this geometry "
            "conversion inspects no scene or robot configuration and does not rank or select a "
            "candidate. check_tcp_pose_reachability is the separate non-executing interface for a "
            "caller-proposed full TCP pose. Roll identity is not joint-space path or collision "
            "difficulty; the caller chooses the candidate."),
    }

_AXIS_WORDS = (((1, 0, 0), "robot-right (+x)"), ((-1, 0, 0), "robot-left (-x)"),
               ((0, 1, 0), "robot-forward (+y)"), ((0, -1, 0), "robot-backward (-y)"),
               ((0, 0, 1), "up (+z)"), ((0, 0, -1), "down (-z)"))
_ALIGNED_DEG = 20.0


def describe_axis(axis):
    """Plain-language reading of a unit world axis: the nearest named direction, or a blend.

    A quaternion is not legible on its own, and neither is a raw axis triple for a reader who has
    to decide whether a pose does what they intended. This states what the numbers point at; it
    expresses no preference about where they SHOULD point.
    """
    vec = _unit_axis(axis, "axis")
    scored = sorted(((float(np.dot(vec, np.asarray(ref, float))), word)
                     for ref, word in _AXIS_WORDS), reverse=True)
    best_dot, best_word = scored[0]
    angle = float(np.degrees(np.arccos(max(-1.0, min(1.0, best_dot)))))
    if angle <= _ALIGNED_DEG:
        return f"{best_word}, {angle:.0f} deg off"
    second_word = scored[1][1]
    return f"between {best_word} and {second_word}"


# The pose convention in one place. It is a CONSTANT of the interface, not a measurement, so it is
# stated where a caller reads the interface — `get_embodiment().orientation` and the common
# ActionResult note — instead of being copied into every decoded pose. Measured 2026-08-13: the
# identical 330-byte string was attached to every orientation block, four times per motion result
# (two boundary snapshots x two arms), which is 1,038 copies across one local run corpus and a
# single distinct value. A diagram-bearing result that has no other place to say it may cite this
# constant once; nothing may re-attach it per pose.
POSE_CONVENTION = (
    "quaternion is [qw,qx,qy,qz]; it rotates gripper-local axes into the world. "
    "Local +x is the approach axis (the direction the gripper reaches along, TCP ahead of the "
    "wrist); local +y is the line joining the fingertips, so the fingers close along it. The "
    "opening axis is an unoriented line: its sign only swaps which finger is which."
)


def decode_pose_axes(quat_wxyz):
    """Inverse of grasp_quat_candidates: what a live orientation MEANS in world coordinates.

    Same convention (`POSE_CONVENTION`): TCP local +x is the approach axis (the direction the
    gripper reaches along) and local +y is the line joining the fingertips. Reporting these next to
    every pose keeps a quaternion auditable — a caller can see which way the gripper actually faces
    instead of having to invert four numbers in their head. It is a decode of a value the caller
    already has: no target, no preferred direction, no evaluation of whether the orientation suits
    any task. The convention STRING is not repeated here; only the decoded axes are per-pose facts.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quat_wxyz must be four finite values [qw,qx,qy,qz]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quat_wxyz must be non-zero")
    w, x, y, z = q / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    approach = rotation @ np.array([1.0, 0.0, 0.0])
    opening = rotation @ np.array([0.0, 1.0, 0.0])
    return {
        "approach_axis_world": [float(v) for v in approach],
        "opening_axis_world": [float(v) for v in opening],
        "approach_reads_as": describe_axis(approach),
        "opening_reads_as": describe_axis(opening),
    }
