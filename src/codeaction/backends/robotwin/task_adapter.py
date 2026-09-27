"""Explicit task-class adaptation for CodeAction's RoboTwin backend."""

from __future__ import annotations

from functools import lru_cache


class CodeActionTaskMixin:
    """Use the CodeAction robot/planner while retaining the upstream task implementation."""

    def load_robot(self, **kwargs):
        import sapien.core as sapien

        from codeaction.backends.robotwin.robot import Robot

        if not hasattr(self, "robot"):
            self.robot = Robot(self.scene, self.need_topp, **kwargs)
            self.robot.set_planner(self.scene)
            self.robot.init_joints()
        else:
            self.robot.reset(self.scene, self.need_topp, **kwargs)

        for entity in (self.robot.left_entity, self.robot.right_entity):
            for link in entity.get_links():
                link: sapien.physx.PhysxArticulationLinkComponent
                link.set_mass(1)


@lru_cache(maxsize=None)
def adapt_task_class(upstream_class):
    """Return one stable subclass that changes only the robot-construction seam."""
    if issubclass(upstream_class, CodeActionTaskMixin):
        return upstream_class
    return type(
        f"CodeAction_{upstream_class.__name__}",
        (CodeActionTaskMixin, upstream_class),
        {
            "__module__": __name__,
            "__doc__": (
                f"CodeAction adapter for {upstream_class.__module__}."
                f"{upstream_class.__name__}."
            ),
        },
    )
