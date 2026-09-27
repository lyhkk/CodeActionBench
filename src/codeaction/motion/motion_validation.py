"""Internal motion-loop termination thresholds.

The harness does not calculate model-facing TCP position or orientation errors. Motion results
carry the commanded value plus raw before/final robot snapshots, so the caller can calculate any
difference it needs. The numeric motion parameters here are **loop-termination conditions**:
`displacement_tolerance_m` tells the leg loop when to stop re-aiming at its target, and
`REACH_CORRECTION_TRIGGER_M` tells the bounded convergence correction when to stop refining.
Without them those loops would not terminate. They are declared to the model through the initial
episode configuration, but neither a residual nor an intermediate waypoint is returned.

Keeping this free of simulator imports makes it unit-testable.
"""
from __future__ import annotations

DISPLACEMENT_TOLERANCE_MAX_M = 0.005
DISPLACEMENT_TOLERANCE_MIN_M = 0.002
DISPLACEMENT_TOLERANCE_FRACTION = 0.25

# When the bounded convergence correction inside `reach_tcp` stops refining. Like
# `displacement_tolerance_m` this is a LOOP-TERMINATION parameter, not a verdict: it decides how
# many correction steps run, not whether the call is reported as successful. Declared to the model
# in the initial episode configuration.
REACH_CORRECTION_TRIGGER_M = 0.01

# The three bounds that decide ABORTED/trajectory_deviation. Keeping the values here lets the
# guard implementation and the episode declaration share one tested source of truth.
TRAJECTORY_DEVIATION_ACTUAL_FLOOR_M = 0.09
TRAJECTORY_DEVIATION_ACTUAL_FACTOR = 4.0
TRAJECTORY_DEVIATION_LATERAL_FLOOR_M = 0.06
TRAJECTORY_DEVIATION_LATERAL_FACTOR = 3.0
TRAJECTORY_DEVIATION_REGRESS_FLOOR_M = 0.05
TRAJECTORY_DEVIATION_REGRESS_FACTOR = 2.0

# `reach_tcp` issues at most this many bounded convergence corrections after its primary
# transport, while the residual remains at or above REACH_CORRECTION_TRIGGER_M.
REACH_CORRECTION_MAX_ATTEMPTS = 2

# cuRobo's interpolation period (`envs/robot/planner.py`). This states what a reported trajectory
# sample count counts; a source pin in test_codeaction_motion_validation catches planner drift.
PLANNER_INTERPOLATION_DT_S = 1.0 / 250.0


def trajectory_deviation_bounds(commanded_leg_m: float) -> dict:
    """Return the three trajectory-deviation bounds for one commanded leg length."""
    commanded = float(commanded_leg_m)
    return {
        "actual_leg_m": max(
            TRAJECTORY_DEVIATION_ACTUAL_FLOOR_M,
            TRAJECTORY_DEVIATION_ACTUAL_FACTOR * commanded,
        ),
        "lateral_m": max(
            TRAJECTORY_DEVIATION_LATERAL_FLOOR_M,
            TRAJECTORY_DEVIATION_LATERAL_FACTOR * commanded,
        ),
        "remaining_growth_m": max(
            TRAJECTORY_DEVIATION_REGRESS_FLOOR_M,
            TRAJECTORY_DEVIATION_REGRESS_FACTOR * commanded,
        ),
    }


def trajectory_deviation_rule() -> str:
    """Model-visible formula rendered from the same constants the guard evaluates."""
    return (
        "actual_leg_m > max("
        f"{TRAJECTORY_DEVIATION_ACTUAL_FLOOR_M:g}, "
        f"{TRAJECTORY_DEVIATION_ACTUAL_FACTOR:.1f} * commanded_leg_m) or "
        "lateral_m > max("
        f"{TRAJECTORY_DEVIATION_LATERAL_FLOOR_M:g}, "
        f"{TRAJECTORY_DEVIATION_LATERAL_FACTOR:.1f} * commanded_leg_m) or "
        "target_remaining_after_m > target_remaining_before_m + max("
        f"{TRAJECTORY_DEVIATION_REGRESS_FLOOR_M:g}, "
        f"{TRAJECTORY_DEVIATION_REGRESS_FACTOR:.1f} * commanded_leg_m)"
    )


def displacement_tolerance_m(
    commanded_magnitude_m: float,
    *,
    max_tolerance_m: float = DISPLACEMENT_TOLERANCE_MAX_M,
    min_tolerance_m: float = DISPLACEMENT_TOLERANCE_MIN_M,
    fraction: float = DISPLACEMENT_TOLERANCE_FRACTION,
) -> float:
    """How close to the commanded target the leg loop must get before it stops re-aiming.

    This is the loop's termination condition, not a statement that the displacement was accurate
    enough for the caller's purpose. The result reports raw before/final poses; it does not report
    this threshold, a residual, or any intermediate waypoint.

    Measured on this embodiment (`data/straight_leg_probe`, 485 executed legs): a straight leg in a
    healthy run delivers its commanded step with |achieved - commanded| of p50 1.11 mm / p95
    2.16 mm, and that error is an ABSOLUTE controller floor — it does not shrink with leg length.
    A fixed 2 mm test therefore sits inside the controller's own noise, so a loop that re-aims at
    the target can keep missing it after the motion has already been delivered.

    The 5 mm cap is not a taste call: across the two Arm-1 production batches, every run that
    stopped without terminating had ended 1.88-3.77 mm from its target, and the next cluster of
    genuinely-short runs starts at 39.34 mm — the cap sits in that gap, above the p95 leg error.
    The fraction stops a small command from being rubber-stamped by an absolute number, and the
    floor keeps this never tighter than the historical 2 mm.
    """
    try:
        mag = abs(float(commanded_magnitude_m))
    except (TypeError, ValueError):
        mag = 0.0
    return float(min(max_tolerance_m, max(min_tolerance_m, fraction * mag)))
