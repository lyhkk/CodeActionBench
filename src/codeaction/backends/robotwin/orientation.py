"""Pure orientation helpers required by the RoboTwin adapter."""

import numpy as np


def axis_angle_quat(axis, degrees):
    """Return a normalized wxyz quaternion for one axis-angle rotation."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("axis must be finite and non-zero")
    axis = axis / norm
    half = np.deg2rad(float(degrees)) * 0.5
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
