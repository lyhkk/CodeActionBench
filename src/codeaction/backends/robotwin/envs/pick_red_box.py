"""Single-object red-box pickup with an environment-owned success predicate.

This replaces the historical host-constructed ``wrist_grasp_loop`` scene, whose benchmark
verifier only checked vertical displacement.  The task environment now owns all three pieces
needed for normal RoboTwin evaluation: scene construction, a scripted expert, and
``check_success``.

Success means that the box rose at least 5 cm from its setup height, the left gripper is closed,
and a left-gripper link is still in contact with the box.  The contact term binds the requested
arm to the object and rules out both "push it onto something" and "close the left gripper on air
while the right arm lifts it".
"""

import numpy as np

from envs._base_task import Base_Task
from envs.utils import ArmTag, create_box, rand_pose


class pick_red_box(Base_Task):

    BOX_HALF_SIZE = (0.025, 0.025, 0.035)
    MIN_RISE_M = 0.05

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        pose = rand_pose(
            xlim=[-0.25, -0.08],
            ylim=[-0.04, 0.10],
            zlim=[0.741 + self.BOX_HALF_SIZE[2]],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
        )
        self.red_box = create_box(
            scene=self,
            pose=pose,
            half_size=self.BOX_HALF_SIZE,
            color=(1.0, 0.05, 0.05),
            name="red_box",
            is_static=False,
        )
        self.red_box_start_z = float(self.red_box.get_pose().p[2])
        self.add_prohibit_area(self.red_box, padding=0.07)

    def play_once(self):
        left = ArmTag("left")
        self.move(self.grasp_actor(
            self.red_box,
            arm_tag=left,
            pre_grasp_dis=0.08,
            contact_point_id=0,
        ))
        self.move(self.move_by_displacement(left, z=0.12))
        self.info["info"] = {"{A}": "red box", "{a}": str(left)}
        return self.info

    def _left_gripper_contacts_box(self):
        left_names = set(self.robot.left_fix_gripper_name)
        for joint, _multiplier, _offset in self.robot.left_gripper:
            if joint is not None:
                left_names.add(joint.child_link.get_name())

        target_name = self.red_box.get_name()
        for contact in self.scene.get_contacts():
            names = (
                contact.bodies[0].entity.name,
                contact.bodies[1].entity.name,
            )
            if not contact.points or target_name not in names:
                continue
            other = names[1] if names[0] == target_name else names[0]
            if other in left_names:
                return True
        return False

    def check_success(self):
        rise = float(self.red_box.get_pose().p[2]) - self.red_box_start_z
        return bool(
            rise > self.MIN_RISE_M
            and self.is_left_gripper_close()
            and self._left_gripper_contacts_box())
