"""Extreme clutter/recovery task: `phone_retrieve_restore`.

Upstream `place_phone_stand` is a ONE-segment pick-and-place: grasp the phone, seat it on the
stand. This env adds the two segments that make the clutter/recovery category what it is — an
occluder that must be moved out of the way FIRST, and a scene that must be RESTORED afterwards:

    blocker_displaced -> phone_retrieved -> phone_stable_on_stand -> blocker_restored

The blocker is a procedural box spawned adjacent to the phone on the side facing the robot, so a
top-down approach to the phone is obstructed until it is moved. Its home region is recorded at
setup; success requires it back there. That last term is what distinguishes this from "sweep the
obstacle onto the floor and carry on".

Success deliberately does NOT prescribe where the blocker goes in the meantime, which arm does
what, or the order beyond what physics forces (no fixed arm order, action count, or
expert path). Both a park-and-return and a hand-over-and-return solution satisfy it.

Verifier hygiene (`codeaction/check_success_audit.py` runs over this directory too): `check_success`
reads only attributes assigned in `load_actors`, assigns nothing, and keeps no sticky flag.
"""

import numpy as np
import sapien

from envs.place_phone_stand import place_phone_stand
from envs.utils import create_box


class phone_retrieve_restore(place_phone_stand):

    BLOCKER_HALF = (0.035, 0.035, 0.05)
    BLOCKER_GAP_M = 0.085       # blocker center offset from the phone, toward the robot
    BLOCKER_HOME_TOL_M = 0.07   # "put it back where it was" tolerance on the home region
    PHONE_TOL = (0.045, 0.04, 0.04)   # upstream place_phone_stand tolerance, reused verbatim

    def load_actors(self):
        super().load_actors()

        # Occluder in front of the phone (toward the robot, -y), sized so a straight top-down
        # descent onto the phone is obstructed until it is displaced.
        phone_p = self.phone.get_pose().p
        blocker_xy = (float(phone_p[0]), float(phone_p[1]) - self.BLOCKER_GAP_M)
        self.blocker = create_box(
            scene=self,
            pose=sapien.Pose([blocker_xy[0], blocker_xy[1],
                              0.741 + self.table_z_bias + self.BLOCKER_HALF[2]], [1, 0, 0, 0]),
            half_size=self.BLOCKER_HALF,
            color=(0.15, 0.6, 0.3),
            name="blocker",
            is_static=False,
        )

        # Setup-time record, NOT a lazy first-call snapshot: check_success runs once at finalize,
        # so a home recorded on first call would be the END pose and "restored" would be trivial.
        self.blocker_home_xy = np.array(blocker_xy, dtype=float)

        self.add_prohibit_area(self.blocker, padding=0.05)

    def play_once(self):
        """No scripted expert. Upstream's choreography grasps the phone directly and never touches
        the blocker, so inheriting it would silently skip the two segments this task exists to
        test. The reference is composed host-side."""
        raise NotImplementedError(
            "phone_retrieve_restore has no scripted expert; "
            "compose a host-side reference under the G3 protocol")

    def check_success(self):
        """Terminal state: the phone is seated on the stand (upstream's functional-point relation,
        reused verbatim so the placement bar is identical), the blocker is back in its home region,
        and both grippers are open. Every term is a stable end state, so no latch is needed."""
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        stand_func_pose = np.array(self.stand.get_functional_point(0))
        blocker_p = np.array(self.blocker.get_pose().p)
        return bool(
            np.all(np.abs(phone_func_pose - stand_func_pose)[:3] < np.array(self.PHONE_TOL))
            and np.linalg.norm(blocker_p[:2] - self.blocker_home_xy) < self.BLOCKER_HOME_TOL_M
            and blocker_p[2] > 0.741 + self.table_z_bias
            and self.is_left_gripper_open() and self.is_right_gripper_open())
