"""`grab_roller` with a success predicate that actually binds both grippers to the roller.

The upstream task is bimanual by design: its official language instruction is "use both arms to
grab the roller on the table", and its scripted expert grasps `contact_point_id=0` and `1` of the
same roller in ONE simultaneous dual-arm move. Upstream `check_success` expresses that as
``is_left_gripper_close() and is_right_gripper_close() and roller.z > 0.8``, which is a proxy, not
a binding: both terms read gripper JOINT state, so a one-arm lift with the other gripper closed on
air passes. That gap was recorded on the release card as a KNOWN GAP on 2026-07-23.

This subclass keeps the scene, the randomization, the expert and every upstream conjunct, and adds
the missing term: each gripper must be in live contact with the roller. Nothing is relaxed — the
height and closure terms are unchanged — so a trajectory that satisfied upstream honestly (both
hands on the roller) still satisfies this one. Only the one-arm shortcut stops passing.

Same doctrine as `pick_red_box`: the environment owns scene, expert and success predicate, and the
benchmark verifier calls that predicate rather than reimplementing it.
"""

from envs.grab_roller import grab_roller

from codeaction.backends.robotwin.envs._contact import gripper_contacts_actor


class grab_roller_dual_contact(grab_roller):

    LIFT_Z_M = 0.8

    def check_success(self):
        roller_pose = self.roller.get_pose().p
        return bool(
            self.is_left_gripper_close()
            and self.is_right_gripper_close()
            and float(roller_pose[2]) > self.LIFT_Z_M
            and gripper_contacts_actor(self, "left", self.roller)
            and gripper_contacts_actor(self, "right", self.roller)
        )
