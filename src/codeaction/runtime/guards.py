"""Physical safety guards (spec §6). PHYSICAL bounds only — a guard may ONLY reject/clamp/report; it
NEVER modifies the goal, chooses the next action, or compensates the path (spec §0.1). Pure functions."""
import math
from typing import List, Optional, Tuple


def clamp_step(delta_xyz: List[float], max_step: float) -> Tuple[List[float], bool]:
    """Clamp a commanded step to max_step magnitude. Returns (delta, was_clamped). Direction preserved;
    no rerouting."""
    n = math.sqrt(sum(c * c for c in delta_xyz))
    if n <= max_step or n < 1e-12:
        return [float(c) for c in delta_xyz], False
    s = max_step / n
    return [float(c * s) for c in delta_xyz], True


def in_workspace(xyz: List[float], aabb: dict) -> bool:
    """True iff xyz is inside the workspace AABB {'x':(lo,hi),'y':(lo,hi),'z':(lo,hi)}."""
    return all(aabb[k][0] <= xyz[i] <= aabb[k][1] for i, k in enumerate(("x", "y", "z")))


class SequentialArmLock:
    """Default: exactly one arm commanded at a time. An explicit paired call is the sanctioned dual-arm
    exception (e.g. handover). This is a physical-concurrency guard, not a strategy."""
    def __init__(self):
        self._held: Optional[str] = None

    def acquire(self, arm: str) -> bool:
        if self._held is None or self._held == arm:
            self._held = arm
            return True
        return False

    def release(self, arm: str) -> None:
        if self._held == arm:
            self._held = None

    def acquire_pair(self, arm_a: str, arm_b: str) -> bool:
        if self._held is None:
            self._held = f"{arm_a}+{arm_b}"
            return True
        return False

    def release_pair(self, arm_a: str, arm_b: str) -> None:
        if self._held in (f"{arm_a}+{arm_b}", f"{arm_b}+{arm_a}"):
            self._held = None
