"""cuRobo world-model selection without importing cuRobo or the simulator."""
import os


TABLE_WORLD_ENV = "CODEACTION_CUROBO_TABLE_WORLD"


def curobo_world_config(robot_origin_xyz):
    """Return the legacy table world, or no world model when explicitly disabled.

    ``None`` is cuRobo's supported empty-world value. Robot self-collision remains enabled by
    ``MotionGenConfig`` independently of the world model.
    """
    if os.environ.get(TABLE_WORLD_ENV, "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    origin = list(robot_origin_xyz)
    return {
        "cuboid": {
            "table": {
                "dims": [0.7, 2, 0.04],
                "pose": [origin[1], 0.0, 0.74 - origin[2], 1, 0, 0, 0.0],
            },
        }
    }
