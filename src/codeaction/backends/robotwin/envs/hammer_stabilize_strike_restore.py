"""Extreme tool-use task: `hammer_stabilize_strike_restore`.

Upstream `beat_block_hammer` is a ONE-segment task: its success predicate is a terminal pose
relation (hammer head within 2 cm of the block's functional point, plus contact), and its block is
`is_static=True`, so "stabilize the target" is vacuous — a static block cannot be knocked away.

This env turns it into a four-segment goal graph:

    hammer_grasped -> target_stabilized -> valid_impact_latched -> hammer_returned

Three changes, each carrying one segment:

1. **The block is dynamic.** `is_static=False` means the strike genuinely displaces an unheld
   block, so stabilizing it with the other arm becomes physically load-bearing rather than
   decorative. The card's `moved_from_start.max_m` milestone reads that out.
2. **A return pad is added.** A procedural static pad away from the block gives the tool a
   declared home, so the episode does not end the instant the strike lands.
3. **Success is the terminal state AFTER the strike.** `check_success` is hammer-on-return-pad +
   both grippers open + block still near its start. The strike is TRANSIENT and is deliberately
   NOT in this predicate — it is latched out-of-band by `codeaction.verification.verifiers.LatchMonitor`
   (`contact_between` hammer/block), which the task card declares as a REQUIRED latch event.
   Terminal state alone would accept "carry the hammer to the pad and never strike"; the latch is
   what makes the strike necessary.

`load_actors` copies upstream's pose randomization rather than subclassing and mutating it: the
block must be created dynamic in the first place, and removing/recreating a live actor is a scene
mutation whose behaviour on the pinned sim is unverified. The strike target keeps upstream's red
procedural box geometry, so no new asset enters the pack.

Verifier hygiene (`codeaction/check_success_audit.py` runs over this directory too): `check_success`
reads only attributes assigned in `load_actors`, assigns nothing, and keeps no sticky flag.
"""

import numpy as np
import sapien

from envs.beat_block_hammer import beat_block_hammer
from envs.utils import create_actor, create_box, rand_pose


class hammer_stabilize_strike_restore(beat_block_hammer):

    # Return pad: static, on the far side of the table from the block band (upstream randomizes
    # the block over y in [-0.05, 0.15]), so "set the hammer back down next to the block" cannot
    # satisfy the return segment by accident.
    RETURN_PAD_XY = (0.0, 0.22)
    RETURN_PAD_HALF = (0.06, 0.06, 0.005)
    RETURN_TOL_XY_M = 0.06      # hammer center over the pad: pad half-width plus a grasp margin
    BLOCK_STAY_M = 0.08         # the struck block must still be near where it started

    def load_actors(self):
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.15],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )

        # The one geometry change vs upstream: a dynamic target, so stabilization is real.
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box",
            is_static=False,
        )
        self.hammer.set_mass(0.001)

        self.return_pad = create_box(
            scene=self,
            pose=sapien.Pose([self.RETURN_PAD_XY[0], self.RETURN_PAD_XY[1],
                              0.741 + self.table_z_bias], [1, 0, 0, 0]),
            half_size=self.RETURN_PAD_HALF,
            color=(0.2, 0.4, 0.9),
            name="return_pad",
            is_static=True,
        )

        # Setup-time record, NOT a lazy first-call snapshot: check_success runs once at finalize,
        # so a start recorded on first call would be the END pose and the "stayed put" term would
        # be trivially true.
        self.block_start_xy = np.array(block_pose.p[:2], dtype=float)

        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            block_pose.p[0] - 0.05,
            block_pose.p[1] - 0.05,
            block_pose.p[0] + 0.05,
            block_pose.p[1] + 0.05,
        ])
        self.prohibited_area.append([
            self.RETURN_PAD_XY[0] - 0.08, self.RETURN_PAD_XY[1] - 0.08,
            self.RETURN_PAD_XY[0] + 0.08, self.RETURN_PAD_XY[1] + 0.08,
        ])

    def play_once(self):
        """No scripted expert. Inheriting upstream's choreography would encode a fixed arm order
        and skip both the stabilization and the return segment. Success must not be defined by a
        fixed arm order, action count, or expert path. The reference for this task is composed
        host-side."""
        raise NotImplementedError(
            "hammer_stabilize_strike_restore has no scripted expert; "
            "compose a host-side reference under the G3 protocol")

    def check_success(self):
        """TERMINAL state only: the hammer is parked on the return pad with both grippers open and
        the struck block is still near where it started. The transient strike is verified
        out-of-band by the card's required `contact_between` latch event, never here — a predicate
        that is only true during impact cannot be read once at finalize."""
        hammer_p = np.array(self.hammer.get_pose().p)
        pad_p = np.array(self.return_pad.get_pose().p)
        block_p = np.array(self.block.get_pose().p)
        return bool(
            np.all(np.abs(hammer_p[:2] - pad_p[:2]) < self.RETURN_TOL_XY_M)
            and hammer_p[2] > pad_p[2]
            and np.linalg.norm(block_p[:2] - self.block_start_xy) < self.BLOCK_STAY_M
            and self.is_left_gripper_open() and self.is_right_gripper_open())
