"""The agent-facing tool name registry, and the names that must never join it.

There is one tier. Earlier revisions declared a ladder above it -- D1 adding a calibrated workspace,
D2 adding depth -- as tool-name lists with no implementation behind them. Those rungs will not be
built: the benchmark's proposition is that metric scale comes from the robot's own motion and the
resulting image change, so a depth rung would not be a harder setting of this benchmark but a
different one. The lists are gone and the three names they held moved into DENY, which is the
honest encoding of "never exposed" and is a strictly stronger guard than "belongs to a rung we do
not serve". `leak_audit.audit_registry` checks the exposed set against DENY.

`TOOL_SET_ID` in tool_surface.py remains the literal "D0". Every task card pins it by value, so it
is a frozen historical token, not a claim that other tiers exist.
"""

D0_TOOLS = [
    "get_world_frame", "get_embodiment", "get_camera_info", "get_arm_pose", "get_gripper_state",
    "get_robot_state", "grasp_quat_candidates", "check_tcp_pose_reachability",
    "capture_head", "capture_wrist", "capture_evidence_views", "project", "ray",
    "plane_intersect",
    "capture_motion_pair", "triangulate_correspondence", "scale_from_object_size",
    "scale_from_gripper",
    "preview_tcp_pose", "compare_tcp_poses", "draw_marks",
    "move_delta", "probe_contact_along", "move_both_delta", "reach_tcp",
    "reach_both_tcp",
    "set_gripper", "camera_aim_pose",
    "get_grasp_contact",
    "done",
]
# Names that must NEVER be exposed to an agent; only the out-of-band verifier may reach them.
# The first four are object ground truth. The last three are the retired D1/D2 rungs: a calibrated
# workspace, and rendered depth. get_depth and get_segmentation exist on the RoboTwin camera and
# so are reachable by mistake; get_workspace and get_point_cloud have no implementation and are
# listed so that adding one cannot silently become an exposed tool.
DENY = {
    "get_object_pose", "get_scene_objects", "get_segmentation", "actor_center",
    "get_workspace", "get_depth", "get_point_cloud",
}


# Local declarations extend the same registry consumed by MCP and run_code.
from codeaction.extensions import declarations as _extensions
for _name, _entry in _extensions("tool").items():
    if _name in D0_TOOLS and not _entry.get("replace"):
        raise ValueError(f"tool {_name} already exists; declare replace: true")
    if _name not in D0_TOOLS:
        D0_TOOLS.append(_name)
