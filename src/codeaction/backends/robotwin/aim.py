"""Depth-free wrist-camera pose geometry.

``camera_aim_pose`` computes exactly one caller-configured TCP pose.  Its pitch is a world-Y
pre-rotation applied to the embodiment's fixed base TCP orientation; it does not rotate a virtual
camera.  The function never selects among alternatives, queries a planner, moves the robot, or
decides whether the resulting view is useful.  The caller owns pitch/standoff selection and
composes this pure transform with
``check_tcp_pose_reachability`` and ``reach_tcp`` when it wants to plan or execute.

The caller also supplies the semantic target.  The transform uses only current calibrated
camera/TCP self-state and the fixed TCP-to-camera mount inferred from that state.  It never reads
scene actors, object ground truth, depth, segmentation, or task-specific nouns.
"""
import numpy as np
from codeaction.backends.robotwin import epipolar as epi
from codeaction.backends.robotwin.geometry import camera_matrices

from codeaction.contracts.harness_parameters import validated_aim_pitch


# Embodiment default, explicitly delivered in the tool schema and echoed in every result.
DEFAULT_PITCH_DEG = 80.0
DEFAULT_STANDOFF_M = 0.28
_CENTERING_ITERATIONS = 2


def ray_plane_intersect(K, E, C, uv, plane_z):
    """World XYZ where the camera ray through ``uv`` meets the horizontal plane ``z=plane_z``."""
    p1 = np.asarray(epi.back_project(K, E, list(uv), 1.0), float)
    C = np.asarray(C, float)
    d = p1 - C
    if abs(d[2]) < 1e-9:
        return None
    t = (float(plane_z) - C[2]) / d[2]
    return (C + t * d).tolist()


def rough_target_from_head(vp, head_uv, plane_z, head_camera="head_camera"):
    """Back-project a caller-selected head pixel onto a caller-supplied horizontal plane."""
    K, E, C = camera_matrices(vp, head_camera)
    return ray_plane_intersect(K, E, C, head_uv, plane_z)


def _camera_pose_world(world_to_camera):
    extrinsic = np.asarray(world_to_camera, dtype=float)
    if extrinsic.shape not in ((3, 4), (4, 4)):
        raise ValueError("camera_E must be a 3x4 or 4x4 world-to-camera rigid transform")
    rotation_world_to_camera = extrinsic[:3, :3]
    translation_world_to_camera = extrinsic[:3, 3]
    rotation_camera_to_world = rotation_world_to_camera.T
    camera_position_world = -rotation_camera_to_world @ translation_world_to_camera
    return camera_position_world, rotation_camera_to_world


def project_world_point(K, camera_position_world, camera_rotation_world, point_world,
                        image_size_hw):
    """Project one world point using an explicit camera pose; pure and JSON-safe."""
    camera_point = np.asarray(camera_rotation_world, dtype=float).T @ (
        np.asarray(point_world, dtype=float) - np.asarray(camera_position_world, dtype=float))
    depth = float(camera_point[2])
    if depth <= 1e-9:
        return {"pixel_uv": None, "depth_m": depth, "in_view": False,
                "center_error_px": None}
    image = np.asarray(K, dtype=float) @ camera_point
    pixel = [float(image[0] / image[2]), float(image[1] / image[2])]
    height, width = [int(value) for value in image_size_hw]
    center_error = float(np.linalg.norm(
        np.asarray(pixel) - np.asarray([width / 2.0, height / 2.0])))
    return {
        "pixel_uv": pixel,
        "depth_m": depth,
        "in_view": bool(0 <= pixel[0] < width and 0 <= pixel[1] < height),
        "center_error_px": center_error,
    }


def compute_camera_aim_pose(*, target_xyz, pitch, standoff, tcp_pose, camera_K, camera_E,
                            image_size_hw, quat_mul, axis_angle_quat, quat_to_rotation,
                            down_quat, iterations=_CENTERING_ITERATIONS):
    """Compute one TCP pose without planning, rendering, scene reads, or robot motion.

    The current TCP/camera pair identifies the fixed TCP-to-camera transform.  The loop
    translates the requested pose in world XY until the candidate camera's optical axis
    intersects the target-height plane at the caller's target; each pass subtracts the residual
    it just measured, so it settles on the first pass and stays there for any iteration count.
    The returned pixel facts are predictions under that unexecuted pose, not post-execution
    measurements.
    """
    target = np.asarray(target_xyz, dtype=float)
    tcp = np.asarray(tcp_pose, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_xyz must be three finite world coordinates")
    if tcp.shape != (7,) or not np.all(np.isfinite(tcp)):
        raise ValueError("tcp_pose must be a finite [x,y,z,qw,qx,qy,qz] pose")
    standoff = float(standoff)
    if not np.isfinite(standoff) or standoff <= 0:
        raise ValueError("standoff must be a positive finite number")
    pitch = validated_aim_pitch(pitch)
    camera_position, camera_rotation = _camera_pose_world(camera_E)
    current_tcp_rotation = quat_to_rotation(tcp[3:])
    tcp_to_camera_translation = current_tcp_rotation.T @ (camera_position - tcp[:3])
    tcp_to_camera_rotation = current_tcp_rotation.T @ camera_rotation

    quat = np.asarray(
        quat_mul(axis_angle_quat([0, 1, 0], pitch), down_quat), dtype=float)
    quat_norm = float(np.linalg.norm(quat))
    if not np.isfinite(quat_norm) or quat_norm < 1e-9:
        raise ValueError("camera aim orientation is invalid")
    quat /= quat_norm
    target_tcp_rotation = quat_to_rotation(quat)
    tcp_xy = target[:2].copy()
    for _ in range(int(iterations)):
        tcp_position = np.asarray([tcp_xy[0], tcp_xy[1], target[2] + standoff])
        candidate_camera_position = (
            tcp_position + target_tcp_rotation @ tcp_to_camera_translation)
        candidate_camera_rotation = target_tcp_rotation @ tcp_to_camera_rotation
        optical_axis = candidate_camera_rotation[:, 2]
        if abs(float(optical_axis[2])) < 1e-9:
            break
        distance = (target[2] - candidate_camera_position[2]) / optical_axis[2]
        look_xy = (candidate_camera_position + distance * optical_axis)[:2]
        tcp_xy = tcp_xy - (look_xy - target[:2])

    tcp_xyz = np.asarray([tcp_xy[0], tcp_xy[1], target[2] + standoff])
    candidate_camera_position = tcp_xyz + target_tcp_rotation @ tcp_to_camera_translation
    candidate_camera_rotation = target_tcp_rotation @ tcp_to_camera_rotation
    projection = project_world_point(
        camera_K, candidate_camera_position, candidate_camera_rotation,
        target, image_size_hw)
    return {
        "pitch_deg": pitch,
        "target_tcp_pose_world": tcp_xyz.tolist() + quat.tolist(),
        "predicted_projection": {
            "pixel_uv": projection["pixel_uv"],
            "depth_m": projection["depth_m"],
            "projected_point_in_frame": projection["in_view"],
            "center_error_px": projection["center_error_px"],
        },
    }
