"""Task-agnostic target-progress stop for dense robot transports.

Planner success says that a control sequence was produced and played.  It does not say that the
physical robot kept moving or reached the requested TCP target.  This module samples only the
robot's own TCP pose at a dense transport's configured progress-window boundaries.  Contact is
deliberately absent: callers read anonymous contact facts only after this monitor has
independently established a stall.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class MotionProgressInterrupted(RuntimeError):
    """Internal control-flow signal raised after a measured no-progress window."""


def _quat_distance_rad(left, right):
    try:
        a = np.asarray(left, float)
        b = np.asarray(right, float)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return None


@dataclass(frozen=True)
class StallEvent:
    arms: tuple[str, ...]
    physics_step: int
    window_steps: int


class MotionProgressGuard:
    """Stop an active dense transport when an outstanding target makes no measured progress."""

    def __init__(self, pose_reader, *, window_steps, linear_floor_m,
                 angular_floor_rad, target_tolerance_m):
        self._pose_reader = pose_reader
        self.window_steps = max(1, int(window_steps))
        self.linear_floor_m = float(linear_floor_m)
        self.angular_floor_rad = float(angular_floor_rad)
        self.target_tolerance_m = float(target_tolerance_m)
        self._active = None
        self._event = None

    def begin(self, targets, physics_step):
        if self._active is not None:
            raise RuntimeError("motion progress guard already active")
        arms = {}
        for arm, target in dict(targets or {}).items():
            pose = self._pose_reader(str(arm))
            if not pose or len(pose) < 7 or target is None or len(target) < 3:
                continue
            xyz = np.asarray(target[:3], float)
            target_quat = (np.asarray(target[3:7], float)
                           if len(target) >= 7 else None)
            position_error = float(np.linalg.norm(np.asarray(pose[:3], float) - xyz))
            orientation_error = (
                _quat_distance_rad(pose[3:7], target_quat)
                if target_quat is not None else None)
            arms[str(arm)] = {
                "target_xyz": xyz,
                "target_quat": target_quat,
                "window_start": int(physics_step),
                "window_position_error_m": position_error,
                "window_orientation_error_rad": orientation_error,
            }
        self._active = arms
        self._event = None

    def finish(self):
        event = self._event
        self._active = None
        self._event = None
        return event

    def observe(self, physics_step):
        if self._active is None or self._event is not None:
            return
        stalled = []
        for arm, state in self._active.items():
            elapsed = int(physics_step) - int(state["window_start"])
            # The decision compares the window's two endpoints.  Reading TCP on every 4 ms
            # physics step adds FK/state-query overhead but cannot affect that comparison.
            if elapsed < self.window_steps:
                continue
            pose = self._pose_reader(arm)
            if not pose or len(pose) < 7:
                continue
            position_error = float(np.linalg.norm(
                np.asarray(pose[:3], float) - state["target_xyz"]))
            orientation_error = (
                _quat_distance_rad(pose[3:7], state["target_quat"])
                if state["target_quat"] is not None else None)
            position_outstanding = position_error >= self.target_tolerance_m
            orientation_outstanding = (
                orientation_error is not None
                and orientation_error >= self.angular_floor_rad)
            if not position_outstanding and not orientation_outstanding:
                state["window_start"] = int(physics_step)
                state["window_position_error_m"] = position_error
                state["window_orientation_error_rad"] = orientation_error
                continue
            linear_progress = (
                state["window_position_error_m"] - position_error)
            start_orientation_error = state["window_orientation_error_rad"]
            angular_progress = (
                0.0 if orientation_error is None or start_orientation_error is None
                else start_orientation_error - orientation_error)
            if (linear_progress < self.linear_floor_m
                    and angular_progress < self.angular_floor_rad):
                stalled.append(arm)
            state["window_start"] = int(physics_step)
            state["window_position_error_m"] = position_error
            state["window_orientation_error_rad"] = orientation_error
        if stalled:
            self._event = StallEvent(
                arms=tuple(stalled), physics_step=int(physics_step),
                window_steps=self.window_steps)
            raise MotionProgressInterrupted(
                f"dense transport stalled for {self.window_steps} physics steps")


__all__ = ["MotionProgressGuard", "MotionProgressInterrupted", "StallEvent"]
