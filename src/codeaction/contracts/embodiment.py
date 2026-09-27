"""Canonical robot self-description card.

Only pose-independent robot geometry and interface semantics belong here. Scene facts, privileged
task strategies, concrete grasp poses, and configuration-specific planning advice do not.

The `frame` block states the coordinate CONVENTION the interface runs on — which frame every
coordinate lives in, its handedness, what the origin is attached to, and which side each arm is
mounted on. It states no position of anything in the scene. Giving the convention explicitly is the
deliberate counterpart to never asking the model to convert frames: on real hardware that
conversion belongs to tf / the driver stack, not to the policy, and here `CuroboPlanner` does it
(`_trans_from_world_to_base` + the embodiment's `frame_bias`) behind the motion tools. Axis strings
are NOT restated here — they come from `world_frame`, which is their single source (spec §8).
"""
from codeaction.contracts import world_frame as _wf

_CARD = {
    "embodiment": "aloha-agilex (dual-arm)",
    "arms": ["left", "right"],
    "gripper": {
        "type": "parallel-jaw",
        "max_opening_m": 0.08,
        "finger_length_m": 0.09,
        "finger_thickness_m": 0.02,
        "empty_close_finger_gap_m": 0.049,
        # FK envelope observed at empty-close/full-open on the active embodiment. This is the
        # distance between finger-link origins, not the 0.08 m object-clearance specification;
        # the rounded bounds include the solver's small endpoint variation (~0.04899..0.14166 m).
        "finger_gap_range_m": [0.048, 0.142],
        "finger_gap_m": "FK distance between the two finger-link contact points. empty-close "
                        "baseline is ~0.049 m. finger_gap_m - empty_close_finger_gap_m is a raw "
                        "gap difference, not by itself an object width. Interpreting it as a "
                        "width requires additional evidence that the same object blocks both "
                        "fingers during a close and that the relevant contact faces and direction "
                        "are known.",
        "opening_axis": "the fingers open/close along the line joining the two fingertips. Its "
                        "world-frame direction depends on the live gripper pose; every reported "
                        "pose carries an `orientation` block that decodes it, and "
                        "scale_from_gripper reports opening_axis_world from FK.",
        "tcp_to_finger_link_origin_m": 0.0354,
        "finger_link_origin_to_tip_m": 0.070,
        "finger_link_origin_to_side_m": 0.010,
        "contact_geometry": "A contact report gives the world pose of the finger LINK ORIGIN, not "
                            "of the touched point. From the link origin: the fingertip is "
                            "finger_link_origin_to_tip_m along the approach axis, and the finger's "
                            "side face is finger_link_origin_to_side_m (= finger_thickness_m / 2) "
                            "along the opening axis. Which of the two applies depends on which "
                            "face of the finger met the obstacle, which follows from the direction "
                            "the arm was moving and the shape it met — the robot cannot report "
                            "that. Relative to the TCP the link origin sits "
                            "tcp_to_finger_link_origin_m back along the approach axis and "
                            "finger_gap_m / 2 along the opening axis.",
    },
    "tcp": {
        "definition": "TCP = the NOMINAL grasp centre, a computed frame a fixed distance ahead of "
                      "the wrist along the approach axis. It is where motion tools put the "
                      "gripper; it is NOT the physical fingertip plane, which lies a further "
                      "tcp_to_fingertip_plane_m ahead. Reading a contact as 'the TCP is on the "
                      "surface' therefore misplaces the surface by that amount. The EE/flange "
                      "frame sits ~0.11 m behind the TCP along the same axis — motion tools here "
                      "target the TCP.",
        "tcp_to_ee_offset_m": 0.11,
        "tcp_to_fingertip_plane_m": 0.0354,
    },
    "frame": {
        "name": "world",
        # "axes" is injected by get_embodiment straight from world_frame, so the axis strings have
        # exactly one source and cannot drift, and no caller shares a mutable dict with the card.
        "one_frame_only": "every coordinate a tool reports, and every coordinate you pass to a "
                          "tool, is expressed in this one world frame. Nothing here returns or "
                          "accepts a robot-base-frame or object-local coordinate, so you never "
                          "perform a frame conversion: the motion tools convert a world-frame "
                          "target into each arm's own base frame internally. The single "
                          "deliberately non-world quantity is get_camera_info's extrinsic E, and "
                          "it is labelled world_to_camera.",
        "handedness": "right-handed — the cross product of the +x direction with the +y direction "
                      "gives the +z direction. This is what makes an orientation built from two "
                      "chosen axes well defined (see grasp_quat_candidates).",
        "origin": "attached to the scene, not to the robot: it is the same origin for both arms "
                  "and it does not move when an arm moves. Where it sits relative to any surface "
                  "or object is NOT stated here and does not follow from the axis definition.",
        "arm_mounting": "the two arms are mounted side by side on one shared base. The arm named "
                        "`left` sits at negative x and the arm named `right` at positive x, at "
                        "nearly the same y and z — the arm names agree with the axis definition "
                        "rather than with any viewer's point of view. The mounting is not exactly "
                        "mirrored, so read live positions with get_arm_pose rather than assuming "
                        "one arm's pose is the other's reflection.",
        "see_also": "get_world_frame returns the same axis definition plus a head-camera overlay "
                    "that draws it.",
    },
    "orientation": {
        "quaternion": "poses are [x, y, z, qw, qx, qy, qz]; the quaternion is wxyz and rotates "
                      "gripper-local axes into the world.",
        "local_axes": "local +x is the approach axis — the direction the gripper reaches along, "
                      "with the TCP ahead of the wrist. Local +y is the line joining the "
                      "fingertips, so the fingers close along it. Local +z completes the frame.",
        "reading_a_pose": "four numbers do not say which way the gripper faces, so every reported "
                          "pose carries an `orientation` block with approach_axis_world and "
                          "opening_axis_world plus a plain reading of each. grasp_quat_candidates "
                          "goes the other way, turning two chosen world axes into quaternions; the "
                          "two are exact inverses.",
        "consequences": "orientation is load-bearing beyond aesthetics: it decides which face of "
                        "the finger meets an obstacle (see gripper.contact_geometry), whether a "
                        "pose is reachable at all, and which way an opening gripper sweeps. "
                        "Nothing here states a preferred orientation for any task.",
    },
    "cameras": {
        "scope": "qualitative mount semantics only: these strings identify which robot link moves "
                 "each camera and its broad viewing direction. They do not provide a fixed numeric "
                 "camera-to-TCP transform. Use the live calibrated world-to-camera E carried by "
                 "each Observation for image geometry.",
        "head_camera": "fixed head-mounted overview camera; in its image, +x(world, robot-right) "
                       "is image-right.",
        "left_camera": "wrist-mounted on the LEFT arm and forward-looking relative to the wrist, "
                       "not aligned with the TCP approach axis. Each Observation reports the "
                       "live calibrated camera extrinsics.",
        "right_camera": "wrist-mounted on the RIGHT arm; same FORWARD-looking mount as left_camera.",
    },
    "reachability": {
        "note": "Grasp orientation is caller-selected. grasp_quat_candidates converts a chosen "
                "approach axis and finger-opening axis into two roll-equivalent candidates; "
                "check_tcp_pose_reachability evaluates each caller-proposed full TCP pose from "
                "the current robot configuration without executing it. grasp_quat_candidates does "
                "not rank candidates, and neither does this card. Roll identity is not joint-space path or collision "
                "difficulty; the caller chooses which full pose to evaluate or execute.",
    },
    "motion": {
        "max_single_displacement_m": 0.30,
        "note": "per-call displacement cap (control-quality bound; direction preserved on clamp).",
    },
    "units": "meters; quaternions wxyz",
}


def get_embodiment() -> dict:
    card = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _CARD.items()}
    card["frame"]["axes"] = dict(_wf.get_world_frame()["axes"])
    card["prompt_text"] = (
        "All coordinates you read and all you pass are in ONE right-handed world frame "
        "(get_world_frame gives its axes and draws them on a head view); the motion tools convert "
        "a world target into each arm's base frame internally, so you never convert frames. The "
        "arm named left is mounted at negative x, right at positive x. "
        "Embodiment: dual-arm (left/right), parallel-jaw grippers (max opening "
        f"{_CARD['gripper']['max_opening_m']} m). TCP = the nominal grasp centre, not the physical "
        f"fingertip plane, which is a further {_CARD['tcp']['tcp_to_fingertip_plane_m']} m along "
        f"the approach axis (EE flange is ~"
        f"{_CARD['tcp']['tcp_to_ee_offset_m']} m behind the TCP). Poses are "
        "[x,y,z,qw,qx,qy,qz]; gripper-local +x is the approach axis and +y the finger-opening "
        "line, and every reported pose carries an `orientation` block decoding those into world "
        "directions. Wrist cameras are forward-looking "
        "relative to the wrist rather than aligned with the TCP approach axis; every Observation "
        "contains its live camera extrinsics. The gripper opening axis in world coordinates depends "
        "on the current pose. grasp_quat_candidates constructs orientations from caller-selected "
        "approach and opening axes without ranking them; check_tcp_pose_reachability evaluates a "
        "caller-proposed full pose without executing it. "
        f"empty-close finger gap baseline: {_CARD['gripper']['empty_close_finger_gap_m']} m. "
        "Units: meters."
    )
    return card


def get_finger_gap_bounds_m() -> tuple[float, float]:
    """Nominal controllable FK finger-link gap range for the active embodiment.

    This is deliberately separate from ``max_opening_m``: that field is object clearance, while
    this API accepts the FK distance between finger-link origins. Candidate-pose tools use the
    calibrated envelope instead of accepting geometry the active gripper cannot command.
    """
    minimum, maximum = _CARD["gripper"]["finger_gap_range_m"]
    return minimum, maximum
