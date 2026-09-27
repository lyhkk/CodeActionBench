"""
Pose / quaternion utilities.

Pose conventions used across RoboTwin:

  SAPIEN wxyz (canonical, used by env.get_*_ee_pose and env.move_to_pose):
      [x, y, z, qw, qx, qy, qz]
      sapien.Pose.q is also [w, x, y, z].

  scipy / many external libs use xyzw:
      [qx, qy, qz, qw]

This module standardises on **wxyz** for everything passed to env.move().
All primitive args expect wxyz; conversion happens here when interfacing
with scipy or external grasp computation that returns xyzw.
"""

from typing import Iterable, List, Sequence

import numpy as np


# ── Conversions ───────────────────────────────────────────────────────────

def quat_wxyz_to_xyzw(q: Sequence[float]) -> List[float]:
    """Convert [qw, qx, qy, qz] → [qx, qy, qz, qw] (scipy convention)."""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return [float(qx), float(qy), float(qz), float(qw)]


def quat_xyzw_to_wxyz(q: Sequence[float]) -> List[float]:
    """Convert [qx, qy, qz, qw] → [qw, qx, qy, qz] (SAPIEN convention)."""
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return [float(qw), float(qx), float(qy), float(qz)]


def pose_wxyz_from_p_q(p: Sequence[float], q_wxyz: Sequence[float]) -> List[float]:
    """Compose a 7-vec wxyz pose from position and wxyz quaternion."""
    return [float(p[0]), float(p[1]), float(p[2]),
            float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])]


# ── Default downward-pointing gripper quaternion (wxyz) ───────────────────
# 180-degree rotation about world X — gripper points straight down.
# In wxyz: [cos(pi/2), sin(pi/2), 0, 0] = [0, 1, 0, 0]
DOWN_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]


# ── Position / displacement helpers ───────────────────────────────────────

def position_of(pose_or_p: Sequence[float]) -> List[float]:
    """Extract the [x, y, z] component from a pose or position."""
    return [float(pose_or_p[0]), float(pose_or_p[1]), float(pose_or_p[2])]


def displaced_pose(pose_wxyz: Sequence[float],
                   dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> List[float]:
    """Return a new wxyz pose displaced in world frame; orientation preserved."""
    if len(pose_wxyz) != 7:
        raise ValueError(f"pose_wxyz must be length 7, got {len(pose_wxyz)}")
    out = list(pose_wxyz)
    out[0] = float(out[0]) + float(dx)
    out[1] = float(out[1]) + float(dy)
    out[2] = float(out[2]) + float(dz)
    return out


def l2_norm(*components: float) -> float:
    return float(np.sqrt(sum(c * c for c in components)))


def pose_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cartesian distance between two poses (ignores orientation)."""
    return l2_norm(a[0] - b[0], a[1] - b[1], a[2] - b[2])


# ── Validation ────────────────────────────────────────────────────────────

# Conservative workspace bounds. Used by safety checks in motion primitives.
DEFAULT_WORKSPACE = {
    "x_min": -0.8, "x_max": 0.8,
    "y_min": -0.8, "y_max": 0.8,
    "z_min": 0.4,  "z_max": 1.5,
}

MAX_DELTA_MAGNITUDE = 0.30  # meters, per single motion primitive


# ── Approach-quaternion computation ───────────────────────────────────────

def compute_approach_quat_wxyz(handle_pos: Sequence[float],
                                object_center: Sequence[float]) -> List[float]:
    """
    Compute a gripper quaternion (wxyz) for a side-approach grasp.

    The gripper's **negative-X axis** is the approach direction (from outside
    toward the object center).  This matches the SAPIEN convention used by
    ``_base_task.get_grasp_pose`` where the ee-to-contact offset is
    ``[-0.12 - pre_dis, 0, 0]`` in the gripper's local frame.

    The gripper's Z axis is aligned with world-up as closely as possible.

    Returns [qw, qx, qy, qz].
    """
    h = np.array(handle_pos[:3], dtype=np.float64)
    c = np.array(object_center[:3], dtype=np.float64)

    # Approach direction: from handle toward center (inward).
    approach = c - h
    approach[2] = 0.0                     # project onto XY plane for a level approach
    norm = float(np.linalg.norm(approach))
    if norm < 1e-6:
        # Degenerate — fall back to straight-down quaternion
        return list(DOWN_QUAT_WXYZ)
    approach /= norm                      # unit inward vector

    # Gripper X axis points *outward* (away from object); approach is -X.
    x_axis = -approach
    z_axis = np.array([0.0, 0.0, 1.0])   # world up

    # Y axis = Z × X (right-hand rule)
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm < 1e-6:
        return list(DOWN_QUAT_WXYZ)
    y_axis /= y_norm

    # Re-orthogonalise Z = X × Y
    z_axis = np.cross(x_axis, y_axis)

    rot = np.column_stack([x_axis, y_axis, z_axis])   # 3×3 rotation matrix

    try:
        import transforms3d as t3d
        q_wxyz = t3d.quaternions.mat2quat(rot)         # returns [w, x, y, z]
    except ImportError:
        # Fallback: manual Shepperd method
        q_wxyz = _mat2quat_fallback(rot)

    return [float(q_wxyz[0]), float(q_wxyz[1]),
            float(q_wxyz[2]), float(q_wxyz[3])]


def _mat2quat_fallback(R) -> List[float]:
    """Shepperd's method — pure-numpy, no transforms3d dependency."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _quat_wxyz_to_rotmat(q_wxyz: Sequence[float]):
    """Convert wxyz quaternion to 3x3 rotation matrix (pure numpy)."""
    w, x, y, z = float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def compute_7d_grasp_poses(
    left_pos: Sequence[float],
    right_pos: Sequence[float],
    object_center: Sequence[float],
    pre_grasp_offset: float = 0.035,
    ee_contact_offset: float = 0.12,
    left_quat_wxyz: Sequence[float] = None,
    right_quat_wxyz: Sequence[float] = None,
) -> dict:
    """
    Build full 7D pre-grasp and grasp poses from 3D handle/contact positions.

    The ee position is offset from the contact point by ``ee_contact_offset``
    along the gripper's **negative X axis** (the approach direction), matching
    ``_base_task.get_grasp_pose``'s convention:
        ``grasp_p = contact_p + R @ [-0.12, 0, 0]``

    The pre-grasp pose adds an additional ``pre_grasp_offset`` along the same
    direction.

    If ``left_quat_wxyz`` / ``right_quat_wxyz`` are provided they are used
    directly as the gripper orientation (e.g. arm-preferred quaternions from
    GRASP_DIRECTION_DIC). Otherwise ``compute_approach_quat_wxyz`` derives
    an orientation from the handle→center geometry.

    Returns dict with keys:
        left_pre_grasp_pose, right_pre_grasp_pose,
        left_grasp_pose, right_grasp_pose
    All in wxyz format [x, y, z, qw, qx, qy, qz].
    """
    quats = {"left": left_quat_wxyz, "right": right_quat_wxyz}
    results = {}
    for side, pos in [("left", left_pos), ("right", right_pos)]:
        q_override = quats[side]
        if q_override is not None:
            q = [float(v) for v in q_override]
        else:
            q = compute_approach_quat_wxyz(pos, object_center)

        # Compute the rotation matrix from the quaternion
        R = _quat_wxyz_to_rotmat(q)
        contact_p = np.array(pos[:3], dtype=np.float64)

        # Grasp ee position: offset along gripper's -X axis (approach dir)
        # Same convention as _base_task.get_grasp_pose: R @ [-offset, 0, 0]
        grasp_p = contact_p + R @ np.array([-ee_contact_offset, 0.0, 0.0])
        grasp_pose = pose_wxyz_from_p_q(grasp_p.tolist(), q)

        # Pre-grasp: further back along the same direction
        pre_p = contact_p + R @ np.array([-(ee_contact_offset + pre_grasp_offset), 0.0, 0.0])
        pre_grasp_pose = pose_wxyz_from_p_q(pre_p.tolist(), q)

        results[f"{side}_pre_grasp_pose"] = pre_grasp_pose
        results[f"{side}_grasp_pose"] = grasp_pose

    return results


def check_workspace_bounds(pose_xyz: Sequence[float],
                           workspace: dict = None) -> tuple:
    """Return (in_bounds: bool, violation_msg: str|None)."""
    ws = workspace or DEFAULT_WORKSPACE
    x, y, z = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
    bad = []
    if x < ws["x_min"] or x > ws["x_max"]: bad.append(f"x={x:.3f}")
    if y < ws["y_min"] or y > ws["y_max"]: bad.append(f"y={y:.3f}")
    if z < ws["z_min"] or z > ws["z_max"]: bad.append(f"z={z:.3f}")
    if bad:
        return False, "out of workspace: " + ", ".join(bad)
    return True, None


def check_delta_magnitude(dx: float, dy: float, dz: float,
                          limit: float = MAX_DELTA_MAGNITUDE) -> tuple:
    """Return (within_limit: bool, magnitude: float)."""
    mag = l2_norm(dx, dy, dz)
    return mag <= limit, mag


# ── Rigid-pose math (wxyz quaternions) — pure, no SAPIEN/envs dependency ────

def _quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: ``a * b``."""
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_conj(q: Sequence[float]) -> list:
    w, x, y, z = [float(v) for v in q]
    return [w, -x, -y, -z]


def _quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Rotate 3-vector ``v`` by wxyz quaternion ``q``:  q * (0,v) * q*."""
    qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    return _quat_mul_wxyz(_quat_mul_wxyz(q, qv), _quat_conj(q))[1:]


def _pose_compose(pa: Sequence[float], pb: Sequence[float]) -> list:
    """Compose two poses ``[x,y,z,qw,qx,qy,qz]``: result = pa ∘ pb."""
    p = np.asarray(pa[:3], dtype=np.float64) + _quat_rotate(pa[3:], pb[:3])
    q = _quat_mul_wxyz(pa[3:], pb[3:])
    return p.tolist() + q.tolist()


def _pose_inverse(pose: Sequence[float]) -> list:
    qc = _quat_conj(pose[3:])
    pinv = -_quat_rotate(qc, pose[:3])
    return pinv.tolist() + list(qc)


def _tcp_target_to_ee_target(ee_pose, tcp_pose, target_xyz, target_quat) -> list:
    """EE pose that places the TCP at ``(target_xyz, target_quat)``.

    The tool offset ``T_ee_tcp = inv(T_w_ee) ∘ T_w_tcp`` is a constant rigid
    transform (read once from the current ee/tcp pair); the EE target is then
    ``T_w_tcp_target ∘ inv(T_ee_tcp)``.  This lets a single reach target the
    FINGERTIP, so dropping the position-only ``move_tcp_to`` does not regress
    grasp accuracy by the ~9 cm tool offset.
    """
    T_ee_tcp = _pose_compose(_pose_inverse([float(v) for v in ee_pose]),
                             [float(v) for v in tcp_pose])
    tcp_target = [float(v) for v in target_xyz] + [float(v) for v in target_quat]
    return _pose_compose(tcp_target, _pose_inverse(T_ee_tcp))


def _chunk_positions(start_xyz, end_xyz, max_step: float = MAX_DELTA_MAGNITUDE) -> list:
    """Split a straight path ``start → end`` into legs of length ≤ ``max_step``.

    Returns waypoints (the last is exactly ``end``); a move already within
    ``max_step`` is a single leg.  Used to chunk a long reach internally so the
    caller issues ONE call and never sees the cap.
    """
    s = np.asarray(start_xyz, dtype=np.float64)
    e = np.asarray(end_xyz, dtype=np.float64)
    dist = float(np.linalg.norm(e - s))
    if dist <= max_step or dist < 1e-9:
        return [e.tolist()]
    n = int(np.ceil(dist / max_step))
    return [(s + (e - s) * (i / n)).tolist() for i in range(1, n + 1)]


def pca_principal_axis(points) -> np.ndarray:
    """Principal (largest-variance) axis of an Nx3 point cloud, as a unit vector
    with a deterministic sign.  For a bar-like handle this is its long axis; the
    grasp closes perpendicular to it.  Honest: computed from the point cloud, not
    from any privileged object pose.
    """
    P = np.asarray(points, dtype=np.float64)
    c = P - P.mean(axis=0)
    cov = (c.T @ c) / max(len(c), 1)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis


def _mat_to_quat_wxyz(R) -> list:
    """Rotation matrix → wxyz quaternion (Shepperd's method, dependency-free)."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0.0:
        s = (t + 1.0) ** 0.5 * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
        w, x, y, z = (R[2, 1]-R[1, 2])/s, 0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s
    elif R[1, 1] > R[2, 2]:
        s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
        w, x, y, z = (R[0, 2]-R[2, 0])/s, (R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s
    else:
        s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
        w, x, y, z = (R[1, 0]-R[0, 1])/s, (R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s
    q = np.array([w, x, y, z], dtype=np.float64)
    return (q / (np.linalg.norm(q) + 1e-12)).tolist()


def grasp_quat_from_handle_axis(handle_axis, approach=(0.0, 0.0, -1.0)) -> list:
    """Grasp orientation (wxyz) for a parallel-jaw gripper grasping a bar.

    Gripper frame (verified in sim: get_tcp_pose = get_ee_pose translated +0.12 m
    along local-x, SAME orientation — so local-x is the EE→TCP "toward-grip" axis):
      local-x = approach = EE→TCP direction. For a TOP-DOWN grasp the wrist is
                ABOVE the grip point, so this points DOWN → ~world -z (default).
                (Using +z put the wrist BELOW the handle → unreachable.)
      local-z = aligned with the handle's long axis,
      local-y = the finger CLOSING direction (⟂ handle, so fingers close ACROSS the bar).
    Built honestly from the perceived handle axis; no privileged pose read.
    """
    x = np.asarray(approach, dtype=np.float64)
    x = x / (np.linalg.norm(x) + 1e-12)
    h = np.asarray(handle_axis, dtype=np.float64)
    z = h - np.dot(h, x) * x                      # handle component ⟂ approach
    if np.linalg.norm(z) < 1e-9:                  # handle ∥ approach → any ⟂ axis
        z = np.array([1.0, 0.0, 0.0]) - x[0] * x
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    R = np.column_stack([x, y, z])                # columns = local x,y,z in world
    return _mat_to_quat_wxyz(R)
