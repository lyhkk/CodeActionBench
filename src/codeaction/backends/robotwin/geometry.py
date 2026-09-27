"""Small RoboTwin camera and actor geometry adapters used by the formal runtime."""

from __future__ import annotations

import numpy as np


def actor_center(actor) -> np.ndarray:
    """Return an actor or actor-wrapper world position."""
    try:
        position = actor.get_pose().p
    except Exception:
        position = actor.actor.get_pose().p
    return np.asarray(position, dtype=np.float64)


def camera_matrices(vp, camera):
    """Return fresh camera intrinsics, world-to-camera extrinsics, and camera centre."""
    try:
        if hasattr(vp.backend, "_refresh_cameras"):
            vp.backend._refresh_cameras()
        else:
            vp.backend.TASK_ENV._update_render()
            vp.backend.TASK_ENV.cameras.update_picture()
    except Exception:
        pass
    matrices = vp.backend.get_camera_matrices(camera)
    if matrices is None or len(matrices) < 2:
        raise RuntimeError(f"camera matrices unavailable for {camera!r}")
    intrinsics = np.asarray(matrices[0], dtype=float)
    extrinsics = np.asarray(matrices[1], dtype=float)[:3]
    if intrinsics.shape != (3, 3) or extrinsics.shape != (3, 4):
        raise RuntimeError(f"invalid camera matrix shape for {camera!r}")
    centre = -extrinsics[:3, :3].T @ extrinsics[:3, 3]
    return intrinsics, extrinsics, centre
