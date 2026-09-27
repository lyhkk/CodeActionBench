"""Two-hand pot transport derived from the official ``lift_pot`` task.

The official task ends while both grippers hold the pot in the air.  This sibling keeps the same
pot asset, randomized initial pose and two official handle contact points, then adds one procedural
static raised rest.  Success requires the pot to be upright and centered on that rest at its
raised support height, with both grippers open.  The card separately requires an in-episode
simultaneous left/right gripper-to-pot contact event; otherwise two unrelated single-arm touches
could be mistaken for cooperative support.

No collision object is inserted into the approach corridor.  The raised rest is the declared
destination and the host reference approaches it from above, so this tests transport and stable
release without relying on an obstacle-aware planner that the benchmark intentionally disables.
"""

import numpy as np
import sapien

from envs.lift_pot import lift_pot
from envs.utils import create_box, get_face_prod


class pot_dual_carry_setdown(lift_pot):

    REST_XY = (0.0, -0.25)
    REST_HALF = (0.15, 0.065, 0.035)
    TARGET_TOL_XY_M = (0.055, 0.055)
    TARGET_TOL_Z_M = 0.035
    CARRY_Z_M = 0.92

    def load_actors(self):
        super().load_actors()
        self.pot_start_p = np.array(self.pot.get_pose().p, dtype=float)
        rest_center_z = 0.741 + self.table_z_bias + self.REST_HALF[2]
        self.raised_rest = create_box(
            scene=self,
            pose=sapien.Pose([self.REST_XY[0], self.REST_XY[1], rest_center_z],
                             [1, 0, 0, 0]),
            half_size=self.REST_HALF,
            color=(0.15, 0.35, 0.9),
            name="raised_rest",
            is_static=True,
        )
        # The pot begins supported by the table.  Raising the support surface by twice the rest's
        # half-height raises the same stable pot centre by exactly that amount.
        self.target_pot_z = float(self.pot_start_p[2] + 2.0 * self.REST_HALF[2])
        self.prohibited_area.append([
            self.REST_XY[0] - self.REST_HALF[0],
            self.REST_XY[1] - self.REST_HALF[1],
            self.REST_XY[0] + self.REST_HALF[0],
            self.REST_XY[1] + self.REST_HALF[1],
        ])

    def play_once(self):
        raise NotImplementedError(
            "pot_dual_carry_setdown uses its card-declared host-side reference")

    def check_success(self):
        pot_pose = self.pot.get_pose()
        pot_p = np.array(pot_pose.p, dtype=float)
        rest_p = np.array(self.raised_rest.get_pose().p, dtype=float)
        upright = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        return bool(
            np.all(np.abs(pot_p[:2] - rest_p[:2]) < self.TARGET_TOL_XY_M)
            and abs(pot_p[2] - self.target_pot_z) < self.TARGET_TOL_Z_M
            and upright > 0.8
            and self.is_left_gripper_open()
            and self.is_right_gripper_open())
