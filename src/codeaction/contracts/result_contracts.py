"""Machine-checkable wire-result contracts and compact model-facing return signatures.

The exhaustive contract belongs to the harness and tests.  The model receives only ``summary``:
enough to know which decision-bearing fields a call returns without paying for every diagnostic
branch in every tool description.  Runtime validation is dependency-free so the same check runs
inside the remote ``robotwin`` environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from codeaction.motion.orientation import AXIS_ORTHOGONALITY_TOLERANCE


def _obj(properties, required=(), *, additional=False):
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": additional,
    }


def _arr(items=None, *, n=None, low=None, high=None):
    out = {"type": "array"}
    if items is not None:
        out["items"] = items
    if n is not None:
        out.update({"minItems": int(n), "maxItems": int(n)})
    if low is not None:
        out["minItems"] = int(low)
    if high is not None:
        out["maxItems"] = int(high)
    return out


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


STR = {"type": "string"}
BOOL = {"type": "boolean"}
NUM = {"type": "number"}
INT = {"type": "integer"}
ANY = {}
DEFAULT_PIXEL_SIGMA_PX = 2.0
RULER_MIN_PROJECTED_SPAN_PX = 1e-9
DRAW_MARKS_MIN_COUNT = 1
DRAW_MARKS_MAX_COUNT = 12
VEC2 = _arr(NUM, n=2)
VEC3 = _arr(NUM, n=3)
QUAT = _arr(NUM, n=4)
POSE7 = _arr(NUM, n=7)
MATRIX3 = _arr(_arr(NUM, n=3), n=3)
MATRIX34 = _arr(_arr(NUM, n=4), n=3)

# One literal convention feeds the wire schema, runtime payloads, and compact result summaries.
# It describes the OpenCV pinhole matrices returned by SAPIEN; it adds no transform or policy.
CAMERA_CONVENTION_VALUE = {
    "extrinsic": "world_to_camera p_cam=R@p_world+t; columns",
    "camera_axes": "+x right; +y down; +z forward",
    "pixel_coordinates": "[u,v] top-left; zero-based; u right; v down; integer=pixel center",
    "image_size": "size_hw=[H,W]; valid iff 0<=u<W,0<=v<H",
    "distortion": "none; RGB pinhole for K",
}
CAMERA_CONVENTION_SUMMARY = (
    "Convention: E=[R|t] world-to-camera, p_cam=R@p_world+t, column vectors, camera +z forward; "
    "[u,v] top-left zero-based; size_hw=[H,W]; distortion=none.")
CAMERA_CONVENTION = _obj(
    {key: {"const": value} for key, value in CAMERA_CONVENTION_VALUE.items()},
    tuple(CAMERA_CONVENTION_VALUE),
)
CAMERA_READ_FAILURE = {
    "type": "string",
    "enum": [
        "camera_refresh_failed",
        "camera_matrices_unavailable",
        "camera_rgb_size_unavailable",
    ],
}
ARM_POSE_READ_FAILURE = {
    "type": "string",
    "enum": [
        "pose_unavailable",
        "ee_pose_unavailable",
        "tcp_pose_unavailable",
        "orientation_unavailable",
    ],
}
GRIPPER_STATE_READ_FAILURE = {
    "type": "string",
    "enum": [
        "gripper_state_unavailable",
        "pose_unavailable",
        "opening_unavailable",
        "finger_gap_unavailable",
        "gripper_val_unavailable",
        "drive_state_unavailable",
    ],
}

# A decoded orientation carries only per-pose FACTS. The convention that explains how to read them
# is one fixed string (`codeaction.motion.orientation.POSE_CONVENTION`) delivered with the interface, not a
# measurement to repeat on every pose.
ORIENTATION = _obj({
    "approach_axis_world": VEC3,
    "opening_axis_world": VEC3,
    "approach_reads_as": STR,
    "opening_reads_as": STR,
}, ("approach_axis_world", "opening_axis_world", "approach_reads_as",
    "opening_reads_as"))

FINGER_POSE = _obj({
    "xyz": VEC3,
    "quat_wxyz": QUAT,
    "approach_axis_world": VEC3,
    "opening_axis_world": VEC3,
    "approach_reads_as": STR,
    "opening_reads_as": STR,
}, ("xyz", "quat_wxyz", "approach_axis_world", "opening_axis_world",
    "approach_reads_as", "opening_reads_as"))

CONTACT = _obj({
    "total_impulse": NUM,
    "fingers_in_contact": INT,
    # `n_contact_points` was removed 2026-08-05: unlike the impulses (a force magnitude a real
    # gripper can sense) and the finger poses (forward kinematics), the number of contact points a
    # solver generated has no counterpart on a real robot -- it moves with the physics engine and
    # the mesh resolution. Read the per-finger impulse distribution instead.
    "per_finger": _obj({}, additional=NUM),
    "contacting_finger_pose": _obj({}, additional=FINGER_POSE),
    "contact_localization": STR,
    "arm": {"type": "string", "enum": ["left", "right"]},
    "tick": INT,
}, ("total_impulse", "fingers_in_contact", "per_finger"))

CONTACT_READ_FAILURE = {"type": "string", "enum": ["contact_unavailable"]}
_PUBLIC_CONTACT_COMMON = {
    "available": BOOL,
    "read_failures": _arr(CONTACT_READ_FAILURE),
    "contact_impulse_unit": {"const": "N*s"},
    "arm": {"type": "string", "enum": ["left", "right"]},
    "tick": INT,
    "note": STR,
}
PUBLIC_CONTACT_AVAILABLE = _obj({
    **CONTACT["properties"],
    **_PUBLIC_CONTACT_COMMON,
    "available": {"const": True},
    "read_failures": _arr(CONTACT_READ_FAILURE, n=0),
}, (*CONTACT["required"], "contacting_finger_pose", "contact_localization", "available",
    "read_failures", "contact_impulse_unit", "arm", "tick", "note"))
PUBLIC_CONTACT_UNAVAILABLE = _obj({
    **_PUBLIC_CONTACT_COMMON,
    "available": {"const": False},
    "read_failures": _arr(CONTACT_READ_FAILURE, low=1),
}, tuple(_PUBLIC_CONTACT_COMMON))
PUBLIC_CONTACT = {"anyOf": [PUBLIC_CONTACT_AVAILABLE, PUBLIC_CONTACT_UNAVAILABLE]}

CONTACT_SIGNATURE = _obj({
    "finger_count": INT,
    "contacting_fingers": _arr(STR),
    "total_impulse": NUM,
    "per_finger": _obj({}, additional=NUM),
    "contacting_finger_pose": _obj({}, additional=FINGER_POSE),
}, ("finger_count", "contacting_fingers", "total_impulse", "per_finger"))

CAMERA_SNAPSHOT = _obj({
    "available": {"const": True},
    "K": MATRIX3,
    "E": MATRIX34,
    "size_hw": _arr(INT, n=2),
    "freshness": {"const": "call_time_current"},
    "read_failures": _arr(STR, n=0),
    "convention": CAMERA_CONVENTION,
}, ("available", "K", "E", "size_hw", "freshness", "read_failures", "convention"))

CAMERA_INFO = _obj({
    "camera": STR,
    "available": BOOL,
    "K": _nullable(MATRIX3),
    "E": _nullable(MATRIX34),
    "size_hw": _nullable(_arr(INT, n=2)),
    "freshness": {"type": "string", "enum": ["call_time_current", "unavailable"]},
    "read_failures": _arr(CAMERA_READ_FAILURE),
    "convention": CAMERA_CONVENTION,
    "tick": INT,
    "note": STR,
}, ("camera", "available", "K", "E", "size_hw", "freshness", "read_failures",
    "convention", "tick", "note"))


def _camera_summary(text):
    return f"{text} {CAMERA_CONVENTION_SUMMARY}"


def _observation(annotation=ANY, *, require_annotation=False):
    required = ["obs_id", "camera", "cam_pose_snapshot", "tick", "image"]
    if require_annotation:
        required.append("annotation")
    return _obj({
        "obs_id": STR,
        "camera": STR,
        "cam_pose_snapshot": CAMERA_SNAPSHOT,
        "tick": INT,
        "annotation": annotation,
        "image": STR,
    }, required)


OBSERVATION = _observation()
OBSERVATION_SET = _obj({
    "set_id": STR,
    "observations": _arr(OBSERVATION, low=1),
    "roles": _obj({}, additional=STR),
    "tick": INT,
    "obs_ids": _arr(STR, low=1),
    "images": STR,
}, ("set_id", "observations", "roles", "tick", "obs_ids", "images"))

# What stopped the arm. Contact and workspace guards may include their final raw triggering
# evidence; intermediate trajectory measurements stay inside the motion loop.
GUARD = _obj({
    "name": STR,
    "effect": STR,
    "observed": _obj({}, additional=True),
    "bound": _obj({}, additional=True),
}, ("name", "effect"))

# A prediction under one unexecuted caller-configured camera pose.  The boolean is deliberately
# named for the calibrated image-boundary fact; it is not RGB visibility or occlusion evidence.
CAMERA_AIM_PROJECTION = _obj({
    "pixel_uv": _nullable(VEC2),
    "depth_m": NUM,
    "projected_point_in_frame": BOOL,
    "center_error_px": _nullable(NUM),
}, ("pixel_uv", "depth_m", "projected_point_in_frame", "center_error_px"))

# The motion planner's own reason for refusing a leg, normalized by codeaction.motion.planner_status.
# `code` separates a kinematic fact (goal_pose_unreachable — no IK solution exists for that goal
# pose from this arm) from a solver outcome (trajectory_optimization_failed) and from an invalid
# start state; measured, those recover 0/22 and 4/6 respectively under extra solver effort, so
# collapsing them into one label discarded the only distinguishing bit. It states the condition
# only — it never carries a suggested action (§0.1 corollary 8).
PLANNER_STATUS = _obj({
    "code": STR,
    "meaning": STR,
    "raw": _nullable(STR),
    "attempts": NUM,
    "valid_query": BOOL,
    "position_error_m": NUM,
    "rotation_error_rad": NUM,
}, ("code", "meaning"))

FAILURE_FIELDS = {
    "reason": STR,
    "failure_category": STR,
    "failure_stage": STR,
    "planner_detail": STR,
    "planner_status": PLANNER_STATUS,
}

JOINT_STATE = _obj({
    "names": _arr(STR),
    "types": _arr(STR),
    "positions": _nullable(_arr(NUM)),
    "velocities": _nullable(_arr(NUM)),
    "position_units": _arr(STR),
    "velocity_units": _arr(STR),
    "read_failures": _arr(STR),
}, ("names", "types", "positions", "velocities", "position_units", "velocity_units",
    "read_failures"))

# One arm at a boundary. `joint_state` is deliberately ABSENT and additionalProperties stays false,
# so a motion result that carried it would fail validation rather than quietly grow back: the
# boundary snapshot states what the caller acts on (poses, decoded axes, gripper, contact), and
# joint qpos/qvel is pulled from `get_robot_state` by whoever actually wants it.
_ROBOT_ARM_FIELDS = {
    "ee_pose": _nullable(POSE7),
    "tcp_pose": _nullable(POSE7),
    "orientation": _nullable(ORIENTATION),
    "opening_m": _nullable(NUM),
    "finger_gap_m": _nullable(NUM),
    "gripper_val": _nullable(NUM),
    "drive_commanded_closed": _nullable(BOOL),
    "contact": _nullable(CONTACT),
    "read_failures": _arr(STR),
}
_ROBOT_ARM_REQUIRED = tuple(_ROBOT_ARM_FIELDS)
_ROBOT_ARM = _obj(_ROBOT_ARM_FIELDS, _ROBOT_ARM_REQUIRED)
# The pull-only variant: identical plus the named joint vector `get_robot_state` exists to return.
_ROBOT_ARM_WITH_JOINTS = _obj({**_ROBOT_ARM_FIELDS, "joint_state": JOINT_STATE},
                              (*_ROBOT_ARM_REQUIRED, "joint_state"))


def _robot_snapshot_fields(arm_schema):
    return {
        "tick": INT,
        "frame": {"const": "world"},
        "pose_format": {"const": "[x,y,z,qw,qx,qy,qz]"},
        "linear_unit": {"const": "m"},
        "angular_unit": {"const": "rad"},
        "contact_impulse_unit": {"const": "N*s"},
        "arms": _obj({
            "left": arm_schema,
            "right": arm_schema,
        }, ("left", "right")),
    }


_ROBOT_SNAPSHOT_FIELDS = _robot_snapshot_fields(_ROBOT_ARM)
_ROBOT_SNAPSHOT_REQUIRED = tuple(_ROBOT_SNAPSHOT_FIELDS)
ROBOT_SNAPSHOT = _obj(_ROBOT_SNAPSHOT_FIELDS, _ROBOT_SNAPSHOT_REQUIRED)

# Names only the robot's OWN contacting part and arm, never the contacted entity.
_CONTACT_PART = _obj({
    "part": {"type": "string", "enum": ["fingertip", "arm_link", "robot_self"]},
    "arms": _arr({"type": "string", "enum": ["left", "right"]}),
}, ("part", "arms"))

# Present on the runtime's stall branch. A failed contact read is unknown, not false.
STALL_CONTACT_EVIDENCE = {
    "blocked_in_contact": _nullable(BOOL),
    "contact_parts": _arr(_CONTACT_PART),
    "pose_at_contact": _nullable(ROBOT_SNAPSHOT),
}


def _require_stall_contact_evidence(schema):
    """Require all three facts only on the failure branch that actually measured them."""
    schema["allOf"] = [{
        "if": {
            "type": "object",
            "properties": {"failure_category": {"const": "stalled"}},
            "required": ["failure_category"],
        },
        "then": {"type": "object", "required": list(STALL_CONTACT_EVIDENCE)},
    }]
    return schema

def _action(commanded, achieved, resulting_pose):
    return _obj({
        "action_id": STR,
        "commanded": commanded,
        "achieved": achieved,
        "status": {"type": "string", "enum": ["SUCCESS", "FAILED", "ABORTED"]},
        "abort_reason": STR,
        "resulting_pose": resulting_pose,
        "tick": INT,
        "call_status": {"const": "OK"},
        "planning": _obj({
            "status": {"type": "string", "enum": [
                "NOT_APPLICABLE", "NOT_RUN", "SUCCEEDED", "FAILED", "NOT_REPORTED"]},
            "planner_status": STR,
        }, ("status",)),
        "execution": _obj({
            "status": {"type": "string", "enum": [
                "NOT_STARTED", "COMPLETED", "INTERRUPTED", "FAILED", "UNKNOWN"]},
            # Whether a post-action state read was available, kept SEPARATE from `status`. Before
            # 2026-08-05 `COMPLETED` was emitted whenever a TCP could still be read, so a call whose
            # plan had been refused still reported its execution as complete.
            "post_state_observed": _nullable(BOOL),
            "state_changed": _nullable(BOOL),
            # How far the physics engine advanced during THIS call, and whether a non-completing
            # call had already moved the robot. Before tool surface 13 a FAILED result could mean either
            # "nothing happened" or "the arm is now somewhere else", with no way to tell.
            # `partial` is nullable: with no clock and no state read, "no side effects" would be
            # as unearned as "it moved".
            "physics_steps": _nullable(INT),
            "partial": _nullable(BOOL),
        }, ("status", "physics_steps", "partial")),
        "failure": _nullable(_obj({
            "stage": STR, "code": STR, "message": STR,
        }, ("stage", "code", "message"))),
        "observed_before": ROBOT_SNAPSHOT,
        "observed_after": ROBOT_SNAPSHOT,
    }, ("action_id", "commanded", "achieved", "status", "resulting_pose", "tick",
        "call_status", "planning", "execution", "failure", "observed_before",
        "observed_after"))


_UNCERTAINTY_EXCLUDES = _arr(STR, low=1)

PROJECT_PROVENANCE = _obj({
    "tool": {"const": "project"},
    "obs_id": STR,
    "xyz": VEC3,
    "method": {"const": "pinhole projection"},
    "uncertainty_scope": {"const": "transform only"},
    "uncertainty_excludes": _UNCERTAINTY_EXCLUDES,
    "note": STR,
}, ("tool", "obs_id", "xyz", "method", "uncertainty_scope",
    "uncertainty_excludes", "note"))

RAY_PROVENANCE = _obj({
    "tool": {"const": "ray"},
    "obs_id": STR,
    "px": VEC2,
    "image_size_hw": _arr(INT, n=2),
    "pixel_in_frame": BOOL,
    "method": {"const": "pinhole back-projection"},
    "uncertainty_scope": {"const": "transform only"},
    "uncertainty_excludes": _UNCERTAINTY_EXCLUDES,
}, ("tool", "obs_id", "px", "image_size_hw", "pixel_in_frame", "method",
    "uncertainty_scope", "uncertainty_excludes"))

_PLANE_PROVENANCE_COMMON = {
    "tool": {"const": "plane_intersect"},
    "obs_id": STR,
    "px": VEC2,
    "plane_point_xyz": VEC3,
    "plane_normal_xyz": VEC3,
    "plane_offset_sigma_m": NUM,
    "ray_origin": VEC3,
    "ray_dir": VEC3,
    "plane_from": STR,
    "inputs_unverified": _arr(STR, low=1),
    "pixel_sigma_px": NUM,
    "coarse_reason": STR,
    "method": {"const": "ray-plane intersection"},
    "uncertainty_scope": {"const": "pixel annotation + plane offset"},
    "uncertainty_excludes": _UNCERTAINTY_EXCLUDES,
}
_PLANE_PROVENANCE_REQUIRED = tuple(_PLANE_PROVENANCE_COMMON)
PLANE_PROVENANCE = {"anyOf": [
    _obj({
        **_PLANE_PROVENANCE_COMMON,
        "range_m": NUM,
        "obliquity_amplification": NUM,
        "pixel_jacobian_world_m_per_px": _obj({"u": VEC3, "v": VEC3}, ("u", "v")),
        "plane_offset_sensitivity_per_m": VEC3,
        "uncertainty_terms": _obj({
            "pixel_m": NUM,
            "plane_offset_m": NUM,
        }, ("pixel_m", "plane_offset_m")),
    }, (*_PLANE_PROVENANCE_REQUIRED, "range_m", "obliquity_amplification",
        "pixel_jacobian_world_m_per_px", "plane_offset_sensitivity_per_m",
        "uncertainty_terms")),
    _obj({**_PLANE_PROVENANCE_COMMON, "note": STR},
         (*_PLANE_PROVENANCE_REQUIRED, "note")),
]}

_PAIR_VALIDITY = _obj({
    "motion_succeeded": BOOL,
    "baseline_nonzero": BOOL,
    "contact_free": BOOL,
}, ("motion_succeeded", "baseline_nonzero", "contact_free"))

_TRIANGULATION_PROVENANCE_COMMON = {
    "tool": {"const": "triangulate_correspondence"},
    "pair_id": STR,
    "obs_before": STR,
    "obs_after": STR,
    "px_before": VEC2,
    "px_after": VEC2,
    "camera": STR,
    "camera_delta_world_m": VEC3,
    "camera_baseline_m": NUM,
    "camera_rotation_deg": NUM,
    "pair_validity": _PAIR_VALIDITY,
    "correspondence_verified": {"const": False},
    "baseline_source": STR,
    "pixel_sigma_px": NUM,
    "method": {"const": "two-ray midpoint"},
    "uncertainty_scope": {"const": "pixel perturbation + ray gap"},
    "uncertainty_excludes": _UNCERTAINTY_EXCLUDES,
}
_TRIANGULATION_PROVENANCE_REQUIRED = tuple(_TRIANGULATION_PROVENANCE_COMMON)
TRIANGULATION_PROVENANCE = {"anyOf": [
    _obj({
        **_TRIANGULATION_PROVENANCE_COMMON,
        "depth_before_m": NUM,
        "depth_after_m": NUM,
        "ray_gap_m": NUM,
        "triangulation_angle_deg": NUM,
        "pixel_perturbation_uncertainty_m": NUM,
        "ray_gap_uncertainty_m": NUM,
    }, (*_TRIANGULATION_PROVENANCE_REQUIRED, "depth_before_m", "depth_after_m",
        "ray_gap_m", "triangulation_angle_deg", "pixel_perturbation_uncertainty_m",
        "ray_gap_uncertainty_m")),
    _obj({**_TRIANGULATION_PROVENANCE_COMMON, "note": STR},
         (*_TRIANGULATION_PROVENANCE_REQUIRED, "note")),
]}

_OBJECT_SIZE_PROVENANCE_COMMON = {
    "tool": {"const": "scale_from_object_size"},
    "obs_id": STR,
    "bbox": _arr(NUM, n=4),
    "extent_axis": {"enum": ["width", "height", "diagonal"]},
    "known_extent_m": NUM,
    "known_extent_sigma_m": _nullable(NUM),
    "normalized_image_extent": NUM,
    "fx": NUM,
    "fy": NUM,
    "source": {"const": "model_prior"},
    "relative_extent_sigma": _nullable(NUM),
    "method": {"const": "pinhole projected extent prior"},
    "uncertainty_scope": {"const": "caller extent prior only"},
    "uncertainty_excludes": _UNCERTAINTY_EXCLUDES,
}
_OBJECT_SIZE_PROVENANCE_REQUIRED = tuple(_OBJECT_SIZE_PROVENANCE_COMMON)
OBJECT_SIZE_PROVENANCE = {"anyOf": [
    _obj(_OBJECT_SIZE_PROVENANCE_COMMON, _OBJECT_SIZE_PROVENANCE_REQUIRED),
    _obj({**_OBJECT_SIZE_PROVENANCE_COMMON, "note": STR},
         (*_OBJECT_SIZE_PROVENANCE_REQUIRED, "note")),
]}


def _estimate(value, kind, provenance):
    return _obj({
        "value": _nullable(value),
        "kind": {"const": kind},
        "uncertainty": _nullable(NUM),
        "coarse": BOOL,
        "valid": BOOL,
        "frame": STR,
        "units": STR,
        "failure": _nullable(_obj({"code": STR, "message": STR}, ("code", "message"))),
        "provenance": provenance,
    }, ("value", "kind", "uncertainty", "coarse", "valid", "frame", "units",
        "failure", "provenance"))


MOVE_ACHIEVED = _require_stall_contact_evidence(_obj({
    "guard": _nullable(GUARD),
    "contact": CONTACT,
    # Which path discipline actually ran.
    "path_mode": STR,
    **STALL_CONTACT_EVIDENCE,
    **FAILURE_FIELDS,
}, ("path_mode",), additional=False))

PROBE_ACHIEVED = _require_stall_contact_evidence(_obj({
    "direction_unit_world": VEC3,
    "travel_budget_m": NUM,
    # Maximum nominal waypoint spacing. `step_m` is clamped to the configured leg maximum; the
    # final/re-aimed leg can be shorter, and this is neither measured path travel nor an overshoot
    # bound.
    "effective_step_m": NUM,
    "guard": _nullable(GUARD),
    "stop_reason": STR,
    "transition": STR,
    "fingers_added": _arr(STR),
    "fingers_lost": _arr(STR),
    "start_contact": CONTACT_SIGNATURE,
    "end_contact": CONTACT_SIGNATURE,
    **STALL_CONTACT_EVIDENCE,
    **FAILURE_FIELDS,
}, additional=False))

REACH_ARM_ACHIEVED = _require_stall_contact_evidence(_obj({
    "guard": _nullable(GUARD),
    "contact": CONTACT,
    "quat_mode": STR,
    **STALL_CONTACT_EVIDENCE,
    **FAILURE_FIELDS,
}, ("quat_mode",), additional=False))


def _paired_motion_achieved():
    """Shared paired transport result; stall facts stay outside the planner's sync block."""
    schema = _obj({
        "sync": _obj({
            "mode": STR, "plan_success": BOOL, "reason": STR,
            "failure_category": STR, "failure_stage": STR,
            "planner_status": _nullable(_obj({
                "left": _nullable(PLANNER_STATUS),
                "right": _nullable(PLANNER_STATUS),
            })),
            "planner_status_available": BOOL,
            "failed_arm": STR,
        }, ("mode", "planner_status_available")),
        **STALL_CONTACT_EVIDENCE,
        "stalled_arms": _arr({"type": "string", "enum": ["left", "right"]}),
    }, ("sync",))
    schema["allOf"] = [{
        "if": {
            "type": "object",
            "properties": {"sync": {
                "type": "object",
                "properties": {"failure_category": {"const": "stalled"}},
                "required": ["failure_category"],
            }},
            "required": ["sync"],
        },
        "then": {"type": "object", "required": [
            *STALL_CONTACT_EVIDENCE, "stalled_arms",
        ]},
    }]
    return schema

TARGET_PROVENANCE = _obj({
    "derived_by": STR,
    "axes_from_that_value": _arr(STR, low=2),
    "axes_you_supplied": _arr(STR),
    "inputs_unverified": _arr(STR),
    "note": STR,
}, ("derived_by", "axes_from_that_value", "inputs_unverified", "note"))

CAMERA_AIM_POSE = _obj({
    "valid": BOOL,
    "camera": STR,
    "arm": STR,
    "target_xyz": VEC3,
    "target_provenance": _nullable(TARGET_PROVENANCE),
    "pitch_deg": NUM,
    "standoff_m": NUM,
    "source_tick": INT,
    "current_tcp_pose_world": _nullable(POSE7),
    "target_tcp_pose_world": _nullable(POSE7),
    "predicted_projection": _nullable(CAMERA_AIM_PROJECTION),
    "reason": STR,
    "note": STR,
}, ("valid", "camera", "arm", "target_xyz", "target_provenance", "pitch_deg",
    "standoff_m", "source_tick", "current_tcp_pose_world", "target_tcp_pose_world",
    "predicted_projection", "reason", "note"))

GRIPPER_ACHIEVED = _obj({
    "gripper_val": _nullable(NUM),
    "drive_error": _nullable(NUM),
    "opening_m": _nullable(NUM),
    "finger_gap_m": _nullable(NUM),
    # A subtraction, named for what it computed. `object_width_m` would assert that something IS
    # between the fingers and that it is held square-on, neither of which the robot can know.
    "finger_gap_minus_empty_close_m": _nullable(NUM),
    "empty_close_finger_gap_m": _nullable(NUM),
    # Upstream drive state (RoboTwin reports closed below a drive value of 0.2); it describes the
    # COMMAND the actuator settled on, not physical closure on an object.
    "drive_commanded_closed": _nullable(BOOL),
    "contact": CONTACT,
}, ("gripper_val", "opening_m", "finger_gap_m", "contact"))


@dataclass(frozen=True)
class ResultContract:
    schema: Mapping[str, Any]
    summary: str


def _contract(schema, summary):
    return ResultContract(schema=schema, summary=summary)


_WORLD_AXIS_PIXELS = _obj({"+x": _arr(INT, n=2), "+y": _arr(INT, n=2),
                          "+z": _arr(INT, n=2)}, ("+x", "+y", "+z"))
_WORLD_AXIS_LABEL_BOXES = _obj({"+x": _arr(INT, n=4), "+y": _arr(INT, n=4),
                               "+z": _arr(INT, n=4)}, ("+x", "+y", "+z"))
_WORLD_AXIS_VISIBILITY = _obj({"+x": BOOL, "+y": BOOL, "+z": BOOL},
                             ("+x", "+y", "+z"))
_WORLD_FRAME_ANN = _obj({
    "tool": {"const": "get_world_frame"},
    "axes": _obj({
        "+x": {"const": "right"}, "-x": {"const": "left"},
        "+y": {"const": "forward"}, "-y": {"const": "backward"},
        "+z": {"const": "up"}, "-z": {"const": "down"},
    }, ("+x", "-x", "+y", "-y", "+z", "-z")),
    "units": {"const": "meters"},
    "prompt_text": STR,
    "legend": _obj({"+x": STR, "+y": STR, "+z": STR}, ("+x", "+y", "+z")),
    "rendering_space": {"const": "screen_space"},
    "metric_scale": {"const": False},
    "world_to_pixel_correspondence": {"const": False},
    "display_note": STR,
    "render_attempted_axes": {"const": ["+x", "+y", "+z"]},
    "glyph": _obj({
        "origin_px": _arr(INT, n=2),
        "endpoints_px": _WORLD_AXIS_PIXELS,
        "label_boxes_xyxy": _WORLD_AXIS_LABEL_BOXES,
        "visibility": _obj({
            "origin_in_frame": BOOL,
            "endpoints_in_frame": _WORLD_AXIS_VISIBILITY,
            "labels_in_frame": _WORLD_AXIS_VISIBILITY,
            "complete_overlay": BOOL,
        }, ("origin_in_frame", "endpoints_in_frame", "labels_in_frame", "complete_overlay")),
    }, ("origin_px", "endpoints_px", "label_boxes_xyxy", "visibility")),
}, ("tool", "axes", "units", "prompt_text", "legend", "rendering_space", "metric_scale",
    "world_to_pixel_correspondence", "display_note", "render_attempted_axes", "glyph"))

_GRASP_CANDIDATE = _obj({
    "label": STR, "quat_wxyz": QUAT, "approach_axis_world": VEC3,
    "opening_axis_world": VEC3,
}, ("label", "quat_wxyz", "approach_axis_world", "opening_axis_world"))

_GRASP_INPUT_AXES = _obj({
    "approach_axis_world_normalized": VEC3,
    "opening_axis_world_normalized": VEC3,
    "normalized_dot": NUM,
    "orthogonality_tolerance": {"const": AXIS_ORTHOGONALITY_TOLERANCE},
    "orthogonalization_applied": BOOL,
}, ("approach_axis_world_normalized", "opening_axis_world_normalized", "normalized_dot",
    "orthogonality_tolerance", "orthogonalization_applied"))

_EMBODIMENT_AXES = _obj({
    "+x": STR, "-x": STR, "+y": STR, "-y": STR, "+z": STR, "-z": STR,
}, ("+x", "-x", "+y", "-y", "+z", "-z"))

_EMBODIMENT_GRIPPER = _obj({
    "type": STR,
    "max_opening_m": NUM,
    "finger_length_m": NUM,
    "finger_thickness_m": NUM,
    "empty_close_finger_gap_m": NUM,
    "finger_gap_range_m": _arr(NUM, n=2),
    "finger_gap_m": STR,
    "opening_axis": STR,
    "tcp_to_finger_link_origin_m": NUM,
    "finger_link_origin_to_tip_m": NUM,
    "finger_link_origin_to_side_m": NUM,
    "contact_geometry": STR,
}, ("type", "max_opening_m", "finger_length_m", "finger_thickness_m",
    "empty_close_finger_gap_m", "finger_gap_range_m", "finger_gap_m", "opening_axis",
    "tcp_to_finger_link_origin_m", "finger_link_origin_to_tip_m",
    "finger_link_origin_to_side_m", "contact_geometry"))

_EMBODIMENT_TCP = _obj({
    "definition": STR,
    "tcp_to_ee_offset_m": NUM,
    "tcp_to_fingertip_plane_m": NUM,
}, ("definition", "tcp_to_ee_offset_m", "tcp_to_fingertip_plane_m"))

_EMBODIMENT_FRAME = _obj({
    "name": STR,
    "one_frame_only": STR,
    "handedness": STR,
    "origin": STR,
    "arm_mounting": STR,
    "see_also": STR,
    "axes": _EMBODIMENT_AXES,
}, ("name", "one_frame_only", "handedness", "origin", "arm_mounting", "see_also",
    "axes"))

_EMBODIMENT_ORIENTATION = _obj({
    "quaternion": STR,
    "local_axes": STR,
    "reading_a_pose": STR,
    "consequences": STR,
}, ("quaternion", "local_axes", "reading_a_pose", "consequences"))

_EMBODIMENT_CAMERAS = _obj({
    "scope": STR,
    "head_camera": STR,
    "left_camera": STR,
    "right_camera": STR,
}, ("scope", "head_camera", "left_camera", "right_camera"))

_EMBODIMENT_REACHABILITY = _obj({"note": STR}, ("note",))
_EMBODIMENT_MOTION = _obj({
    "max_single_displacement_m": NUM,
    "note": STR,
}, ("max_single_displacement_m", "note"))

_MOTION_PAIR = _obj({
    "pair_id": STR, "before": OBSERVATION, "after": OBSERVATION,
    "motion": _action(_obj({}, additional=True), MOVE_ACHIEVED,
                      _obj({"tcp": _nullable(POSE7)}, ("tcp",))),
    "camera_delta_world_m": VEC3, "camera_baseline_m": NUM,
    "camera_rotation_deg": NUM,
    "contact_evidence": _obj({
        "contact_free": BOOL, "scope": STR, "before": CONTACT, "after": CONTACT,
        "motion_abort_reason": STR,
    }, ("contact_free", "scope", "before", "after")),
    "validity": _PAIR_VALIDITY,
    "obs_ids": _arr(STR, n=2), "images": STR,
}, ("pair_id", "before", "after", "motion", "camera_delta_world_m",
    "camera_baseline_m", "camera_rotation_deg", "contact_evidence", "validity", "obs_ids",
    "images"))

_MARK_ANN = _obj({
    "tool": {"const": "draw_marks"}, "image_size_hw": _arr(INT, n=2),
    "marks": _arr(_obj({
        "px": VEC2, "label": STR, "color": STR, "in_frame": BOOL,
        "label_bbox_px": _arr(NUM, n=4),
    }, ("px", "label", "color", "in_frame")), low=DRAW_MARKS_MIN_COUNT,
        high=DRAW_MARKS_MAX_COUNT),
}, ("tool", "image_size_hw", "marks"))

_GRASP_AXIS = _obj({
    "world": VEC3, "camera": VEC3, "reads_as": STR,
    "angle_to_view_line_deg": NUM, "display": STR,
}, ("world", "camera", "reads_as", "angle_to_view_line_deg", "display"))

_GRASP_FINGER = _obj({
    "link_name": STR, "collision_shape_count": INT, "collision_vertex_count": INT,
    "link_origin_xyz": VEC3, "link_origin_px": _nullable(VEC2),
    "inner_tip_center_xyz": VEC3, "inner_tip_center_px": _nullable(VEC2),
    "display_box_tcp": _obj({"min": VEC3, "max": VEC3}, ("min", "max")),
    "collision_aabb_world": _obj({"min": VEC3, "max": VEC3}, ("min", "max")),
    "final_bbox_px": _nullable(_arr(NUM, n=4)),
    "collision_vertices_projectable": INT, "collision_vertices_in_frame": INT,
    "final_fully_in_frame": BOOL,
}, ("link_name", "collision_shape_count", "collision_vertex_count", "link_origin_xyz",
    "link_origin_px", "inner_tip_center_xyz", "inner_tip_center_px", "display_box_tcp",
    "collision_aabb_world", "final_bbox_px", "collision_vertices_projectable",
    "collision_vertices_in_frame", "final_fully_in_frame"))

_GRASP_FOOTPRINT_ANN = _obj({
    "tool": {"const": "preview_tcp_pose"}, "arm": STR, "tcp_xyz": VEC3,
    "quat_wxyz": QUAT, "finger_gap_m": NUM, "finger_gap_source": STR,
    "sampling": _obj({
        "image_tick": INT,
        "robot_geometry_tick": INT,
        "finger_gap_tick": _nullable(INT),
        "ticks_match": BOOL,
        "robot_command_executed": BOOL,
        "tick_advanced": BOOL,
        "definition": STR,
    }, ("image_tick", "robot_geometry_tick", "finger_gap_tick", "ticks_match",
        "robot_command_executed", "tick_advanced", "definition")),
    "geometry_source": STR,
    "axes": _obj({"approach": _GRASP_AXIS, "opening": _GRASP_AXIS,
                  "lateral": _GRASP_AXIS}, ("approach", "opening", "lateral")),
    "orientation_convention": STR,
    "action_overlay": _obj({
        "approach": _obj({
            "display_length_m": NUM, "start_xyz": VEC3, "start_px": _nullable(VEC2),
            "end_xyz": VEC3, "end_px": _nullable(VEC2),
            "direction_world": VEC3, "projected_length_px": _nullable(NUM),
            "nearly_along_view": BOOL, "angle_to_view_line_deg": NUM,
            "display_start_px": _nullable(VEC2),
            "display_anchor_px": _nullable(VEC2), "display_label": STR,
            "definition": STR,
        }, ("display_length_m", "start_xyz", "start_px", "end_xyz", "end_px",
            "direction_world", "projected_length_px", "nearly_along_view",
            "angle_to_view_line_deg", "display_start_px", "display_anchor_px",
            "display_label", "definition")),
        "closing": _arr(_obj({
            "finger_link": STR, "tip_xyz": VEC3, "tip_px": _nullable(VEC2),
            "direction_world": VEC3, "arrow_end_xyz": VEC3,
            "arrow_end_px": _nullable(VEC2), "render_start_px": _nullable(VEC2),
            "render_end_px": _nullable(VEC2),
        }, ("finger_link", "tip_xyz", "tip_px", "direction_world",
            "arrow_end_xyz", "arrow_end_px", "render_start_px",
            "render_end_px")), n=2),
        "closing_definition": STR,
    }, ("approach", "closing", "closing_definition")),
    "ee_xyz": VEC3, "ee_px": _nullable(VEC2), "tcp_px": _nullable(VEC2),
    "grasp_center_xyz": VEC3, "grasp_center_px": _nullable(VEC2),
    "grasp_center_definition": STR,
    "reference": _obj({
        "source": STR, "input_px": _nullable(VEC2),
        "input_xyz_world": _nullable(VEC3), "projected_px": _nullable(VEC2),
        "grasp_minus_reference_px": _nullable(VEC2),
        "pixel_distance": _nullable(NUM),
        "grasp_minus_reference_world": _nullable(VEC3),
        "world_distance_m": _nullable(NUM), "definition": STR,
    }, ("source", "input_px", "input_xyz_world", "projected_px",
        "grasp_minus_reference_px", "pixel_distance",
        "grasp_minus_reference_world", "world_distance_m", "definition")),
    "finger_link_origin_center_xyz": VEC3,
    "finger_link_origin_center_px": _nullable(VEC2),
    "fingers": _arr(_GRASP_FINGER, n=2),
    "final_collision_bbox_px": _nullable(_arr(NUM, n=4)),
    "image_size_hw": _arr(INT, n=2), "rendered_image_size_hw": _arr(INT, n=2),
    "in_view": BOOL,
}, ("tool", "arm", "tcp_xyz", "quat_wxyz", "finger_gap_m", "finger_gap_source",
    "sampling", "geometry_source", "axes", "orientation_convention", "action_overlay",
    "ee_xyz", "ee_px", "tcp_px", "grasp_center_xyz", "grasp_center_px",
    "grasp_center_definition", "reference", "finger_link_origin_center_xyz",
    "finger_link_origin_center_px", "fingers", "final_collision_bbox_px",
    "image_size_hw", "rendered_image_size_hw", "in_view"))

_POSE_CANDIDATE = _obj({
    "index": INT, "label": STR, "arm": STR, "color_rgb": _arr(INT, n=3),
    "tcp_xyz": VEC3, "quat_wxyz": QUAT, "finger_gap_m": NUM, "finger_gap_source": STR,
    "tcp_px": _nullable(VEC2),
    "grasp_center_xyz": VEC3, "grasp_center_px": _nullable(VEC2),
    "approach_axis_world": VEC3, "opening_axis_world": VEC3,
    "approach_reads_as": STR, "opening_reads_as": STR,
    "approach_angle_to_view_line_deg": NUM,
    "approach_stub_px": _arr(_nullable(VEC2), n=2),
    "label_px": _nullable(VEC2),
    "silhouette_bbox_px": _nullable(_arr(NUM, n=4)),
    "collision_vertex_count": INT, "collision_vertices_in_frame": INT,
    "fully_in_frame": BOOL,
}, ("index", "label", "arm", "color_rgb", "tcp_xyz", "quat_wxyz", "finger_gap_m",
    "finger_gap_source", "tcp_px", "grasp_center_xyz", "grasp_center_px",
    "approach_axis_world", "opening_axis_world", "approach_reads_as", "opening_reads_as",
    "approach_angle_to_view_line_deg", "approach_stub_px", "label_px", "silhouette_bbox_px",
    "collision_vertex_count", "collision_vertices_in_frame", "fully_in_frame"))

# `finger_bounding_boxes_overlap`, `separation_m` and `separation_axis_world` are REQUIRED and nullable
# rather than optional: null is the load-bearing answer for two poses of one arm (alternatives that
# never coexist), and a field that could simply be absent would read as "not computed".
_POSE_CANDIDATE_PAIR = _obj({
    "a": INT, "b": INT, "same_arm": BOOL,
    "grasp_center_delta_world": VEC3, "grasp_center_distance_m": NUM,
    "approach_axis_angle_deg": NUM,
    "finger_bounding_boxes_overlap": _nullable(BOOL),
    "separation_m": _nullable(NUM),
    "separation_axis_world": _nullable(VEC3),
    "projected_bbox_overlap": BOOL,
    "nearer_to_camera": _nullable(INT),
    "definition": STR,
}, ("a", "b", "same_arm", "grasp_center_delta_world", "grasp_center_distance_m",
    "approach_axis_angle_deg", "finger_bounding_boxes_overlap", "separation_m",
    "separation_axis_world", "projected_bbox_overlap", "nearer_to_camera", "definition"))

_POSE_CANDIDATES_ANN = _obj({
    "tool": {"const": "compare_tcp_poses"},
    "candidate_count": INT,
    "candidates": _arr(_POSE_CANDIDATE, low=2, high=4),
    "pairs": _arr(_POSE_CANDIDATE_PAIR, low=1, high=6),
    "sampling": _obj({"robot_geometry_tick": INT}, ("robot_geometry_tick",)),
    "geometry_source": STR, "overlay_legend": STR, "layout_discipline": STR, "scope": STR,
    "image_size_hw": _arr(INT, n=2), "rendered_image_size_hw": _arr(INT, n=2),
    "in_view": BOOL,
}, ("tool", "candidate_count", "candidates", "pairs", "sampling", "geometry_source", "overlay_legend",
    "layout_discipline", "scope", "image_size_hw", "rendered_image_size_hw", "in_view"))

_RULER_ANN = _obj({
    "tool": STR, "arm": STR, "true_dist_m": NUM, "endpoints_px": _arr(_nullable(VEC2), n=2),
    "endpoint_in_frame": _arr(BOOL, n=2), "valid_metric_reference": BOOL,
    "span_px": _nullable(NUM), "camera_depths_m": _arr(NUM, n=2),
    "meters_per_pixel_along_segment": _nullable(NUM), "segment_meaning": STR,
    "opening_axis_world": _nullable(VEC3), "in_view": BOOL,
    "minimum_projected_span_px": NUM, "failure_reason": _nullable(STR), "transfer_limit": STR,
}, ("tool", "arm", "true_dist_m", "endpoints_px", "endpoint_in_frame",
    "valid_metric_reference", "span_px", "camera_depths_m",
    "meters_per_pixel_along_segment", "segment_meaning", "opening_axis_world", "in_view",
    "minimum_projected_span_px", "failure_reason", "transfer_limit"))

_INTERNAL_TRACE = _obj({
    "index": INT, "tool": STR, "ok": BOOL, "status": STR, "action_id": STR,
    "tick": INT, "abort_reason": STR, "failure_category": STR, "stop_reason": STR,
    "transition": STR, "obs_ids": _arr(STR), "error": STR,
    # Recorded for the archive and STRIPPED from the model-visible projection: the primitive's
    # own arguments (so a later reading knows which arm acted and where it was sent) and where it
    # ended. Bounded by the sandbox; `args_omitted` says why the arguments are absent.
    "args": {"type": "object"}, "args_omitted": STR,
    "achieved_pose": {"type": "object"},
}, ("index", "tool", "ok"))

_RUN_CODE = _obj({
    "ok": BOOL, "error": _nullable(STR), "stdout": STR, "value": ANY, "result_assigned": BOOL,
    "tool_calls": INT, "obs_ids": _arr(STR), "internal_trace": _arr(_INTERNAL_TRACE),
    "filesystem_policy": {"const": "structurally_denied"}, "namespace_reset": BOOL,
    "path": STR,
    # Any nested atomic-action abort stops the child before it can issue another primitive.
    # The outer code call is recoverable and retains the exact ActionResult here.
    "interrupted_action": {"type": "object"},
    # Present only when an atomic action abort stopped the block. These repeat the stopping
    # action's own vocabulary so ONE classifier covers direct calls and code blocks; `status`
    # here is the block's outcome, while `interrupted_action.status` is the action's own.
    "status": {"type": "string", "enum": ["ABORTED"]},
    "abort_reason": STR,
    "failure": _obj({"stage": STR, "code": STR, "message": STR},
                    ("stage", "code", "message")),
    # Present only when code captured more images than the transport attached. Absent means
    # nothing was withheld, which is why it is optional rather than required-and-nullable: unlike
    # a measurement that can be unavailable, "no images withheld" has no missing case.
    "images_withheld": _obj({
        "count": INT, "obs_ids": _arr(STR), "note": STR,
    }, ("count", "obs_ids", "note")),
}, ("ok", "stdout", "value", "result_assigned", "tool_calls", "obs_ids",
    "internal_trace", "filesystem_policy", "namespace_reset"))


RESULT_CONTRACTS = {
    "get_world_frame": _contract(_observation(_WORLD_FRAME_ANN, require_annotation=True),
                                  _camera_summary("Annotated Observation.")),
    "get_embodiment": _contract(_obj({
        "embodiment": STR, "arms": {"const": ["left", "right"]},
        "gripper": _EMBODIMENT_GRIPPER, "tcp": _EMBODIMENT_TCP,
        "frame": _EMBODIMENT_FRAME,
        "orientation": _EMBODIMENT_ORIENTATION, "cameras": _EMBODIMENT_CAMERAS,
        "reachability": _EMBODIMENT_REACHABILITY, "motion": _EMBODIMENT_MOTION,
        "units": STR, "prompt_text": STR,
    }, ("embodiment", "arms", "gripper", "tcp", "frame", "orientation", "cameras",
        "reachability", "motion", "units", "prompt_text")),
        "EmbodimentCard."),
    "get_camera_info": _contract(CAMERA_INFO, _camera_summary("CameraInfo.")),
    "get_arm_pose": _contract(_obj({
        "arm": STR, "ee_pose": _nullable(POSE7), "tcp_pose": _nullable(POSE7),
        "orientation": _nullable(ORIENTATION),
        "read_failures": _arr(ARM_POSE_READ_FAILURE),
        "tick": INT, "note": STR,
    }, ("arm", "ee_pose", "tcp_pose", "orientation", "read_failures", "tick", "note")),
        "ArmPose."),
    "get_gripper_state": _contract(_obj({
        "arm": STR, "opening_m": _nullable(NUM), "finger_gap_m": _nullable(NUM),
        "gripper_val": _nullable(NUM), "drive_commanded_closed": _nullable(BOOL), "tick": INT,
        "read_failures": _arr(GRIPPER_STATE_READ_FAILURE), "note": STR,
    }, ("arm", "opening_m", "finger_gap_m", "gripper_val", "drive_commanded_closed", "tick",
        "read_failures", "note")),
        "GripperState."),
    "get_robot_state": _contract(_obj({
        **_ROBOT_SNAPSHOT_FIELDS,
        # Unlike the canonical action snapshots, this read-only tool may return a caller-selected
        # subset of arms — and it is the ONLY tool that returns joint_state.
        "arms": _obj({"left": _ROBOT_ARM_WITH_JOINTS, "right": _ROBOT_ARM_WITH_JOINTS}),
        "note": STR,
    }, (*_ROBOT_SNAPSHOT_REQUIRED, "note")),
        "RobotState."),
    "grasp_quat_candidates": _contract(_obj({
        "candidates": _arr(_GRASP_CANDIDATE, n=2),
        "input_axes": _GRASP_INPUT_AXES,
        "opening_axis_input_is_unoriented_line": BOOL,
        "reachability_evaluated": {"const": False}, "note": STR,
    }, ("candidates", "input_axes", "opening_axis_input_is_unoriented_line",
        "reachability_evaluated", "note")),
        "Two unranked wxyz candidates."),
    # `planner_found_trajectory`, not `reachable`: a refusal is this planner's answer for this
    # query from this configuration, not a global reachability fact.
    # `planning_wall_s` was removed 2026-08-05 -- how long our planner took is harness timing, not
    # anything the robot senses, and a jittering float in the context is a reproducibility hazard.
    # `collision_world` and `note` left the result in tool surface 28: what the planner's world contains is
    # episode configuration (`harness_parameters.planner_collision_world`, declared once), and the
    # two fixed notes only restated `stage`, which is already a closed enum the description defines.
    # 97/97 calls in the 26.x corpus carried byte-identical copies of both, 0 agent reads.
    "check_tcp_pose_reachability": _contract(_obj({
        "arm": STR, "target_xyz": VEC3, "target_quat": QUAT, "quat_mode": STR,
        "planner_found_trajectory": _nullable(BOOL), "planner_status": STR,
        "trajectory_sample_count": INT, "interpolation_dt_s": NUM,
        "planner_diagnostic": _nullable(PLANNER_STATUS),
        "stage": {
            "type": "string",
            "enum": ["planner_query", "workspace_prefilter", "planner_exception"],
            "description": (
                "planner_query = the self-collision-aware single-arm planner ran; "
                "workspace_prefilter = the coarse workspace envelope rejected the target before "
                "planning; planner_exception = the planner query raised"),
        },
        "state_unchanged": BOOL, "tick": INT,
        "reason": STR,
    }, ("arm", "target_xyz", "planner_found_trajectory", "planner_status",
        "trajectory_sample_count", "interpolation_dt_s", "planner_diagnostic", "stage",
        "state_unchanged", "tick")),
        "Planner query result."),
    "get_grasp_contact": _contract(PUBLIC_CONTACT, "ContactRead."),
    "capture_head": _contract(OBSERVATION, _camera_summary("Observation.")),
    "capture_wrist": _contract(OBSERVATION_SET, _camera_summary("ObservationSet.")),
    "capture_evidence_views": _contract(OBSERVATION_SET, _camera_summary("ObservationSet.")),
    "project": _contract(_estimate(VEC2, "pixel_point", PROJECT_PROVENANCE),
        "Estimate<pixel_point>."),
    "ray": _contract(_estimate(
        _obj({"origin": VEC3, "dir": VEC3}, ("origin", "dir")), "ray", RAY_PROVENANCE),
        "Estimate<ray>."),
    "plane_intersect": _contract(_estimate(VEC3, "world_point", PLANE_PROVENANCE),
        "Estimate<world_point>."),
    "capture_motion_pair": _contract(_MOTION_PAIR,
        _camera_summary("ObservationPair.")),
    "triangulate_correspondence": _contract(
        _estimate(VEC3, "world_point", TRIANGULATION_PROVENANCE),
        "Estimate<world_point>."),
    "scale_from_object_size": _contract(_estimate(NUM, "depth", OBJECT_SIZE_PROVENANCE),
        "Estimate<depth>."),
    "scale_from_gripper": _contract(_observation(_RULER_ANN), _camera_summary(
        "Observation ruler; use valid_metric_reference, endpoints_px, span_px and true_dist_m.")),
    "draw_marks": _contract(_observation(_MARK_ANN, require_annotation=True), _camera_summary(
        "Annotated Observation; marks report px,label,color,in_frame.")),
    "preview_tcp_pose": _contract(_observation(
        _GRASP_FOOTPRINT_ANN, require_annotation=True),
        "Annotated Observation; translucent physical finger shapes show the terminal gripper, a "
        "display-only approach arrow points into its terminal TCP, two shaded inner-tip markers "
        "carry inward closing arrows toward the grasp center, and an optional reference is marked. "
        "annotation.sampling identifies the same-tick image/robot-geometry inputs and the "
        "live-vs-caller gap source. It gives no collision or reachability verdict. "
        + CAMERA_CONVENTION_SUMMARY),
    "compare_tcp_poses": _contract(
        _observation(_POSE_CANDIDATES_ANN, require_annotation=True),
        _camera_summary("Annotated Observation; required annotation contains candidates, pairs, "
                        "and the robot-geometry sample tick.")),
    "move_delta": _contract(_action(
        _obj({"arm": STR, "delta": VEC3, "clamped_to": VEC3, "path": STR}, ("arm", "delta")),
        MOVE_ACHIEVED, _obj({"tcp": _nullable(POSE7)}, ("tcp",))),
        "ActionResult."),
    "probe_contact_along": _contract(_action(
        _obj({"arm": STR, "direction_xyz": VEC3, "distance_m": NUM, "step_m": NUM,
              "clamped_to": VEC3}, ("arm", "direction_xyz", "distance_m", "step_m")),
        PROBE_ACHIEVED, _obj({"tcp": _nullable(POSE7)}, ("tcp",))),
        "ActionResult."),
    "move_both_delta": _contract(_action(
        _obj({"left_delta": VEC3, "right_delta": VEC3,
              "clamped_to": _obj({"left": VEC3, "right": VEC3})},
             ("left_delta", "right_delta", "clamped_to")),
        _paired_motion_achieved(),
        _obj({"left_tcp": _nullable(POSE7), "right_tcp": _nullable(POSE7)},
             ("left_tcp", "right_tcp"))),
        "ActionResult."),
    "reach_tcp": _contract(_action(
        _obj({"arm": STR, "target_xyz": VEC3, "target_quat": QUAT,
              "target_provenance": _nullable(TARGET_PROVENANCE)}, ("arm", "target_xyz")),
        REACH_ARM_ACHIEVED, _obj({"tcp": _nullable(POSE7)}, ("tcp",))),
        "ActionResult."),
    "reach_both_tcp": _contract(_action(
        _obj({"left_xyz": VEC3, "right_xyz": VEC3, "left_quat": QUAT, "right_quat": QUAT},
             ("left_xyz", "right_xyz")),
        _paired_motion_achieved(),
        _obj({"left_tcp": _nullable(POSE7), "right_tcp": _nullable(POSE7)},
             ("left_tcp", "right_tcp"))),
        "ActionResult."),
    "set_gripper": _contract(_action(
        _obj({"arm": STR, "pos": NUM}, ("arm", "pos")), GRIPPER_ACHIEVED,
        _obj({"tcp": _nullable(POSE7)}, ("tcp",))),
        "ActionResult."),
    "camera_aim_pose": _contract(
        CAMERA_AIM_POSE,
        "CameraAimPose."),
    "run_code": _contract(_RUN_CODE,
        "CodeResult{ok,error,stdout,value,result_assigned,internal_trace}; ok is code execution "
        "only. images_withheld appears when code captured more images than were attached. A "
        "nested unexpected-contact or contact-read-unavailable abort stops the code block, "
        "preserves its ActionResult in interrupted_action, and leaves the episode active."),
    "write_file": _contract(_obj({
        "ok": BOOL, "path": STR, "bytes": INT,
        "filesystem_policy": {"const": "structurally_denied"},
    }, ("ok", "path", "bytes", "filesystem_policy")),
        "FileWrite{ok,path,bytes}; virtual .py/.md workspace only."),
    "read_file": _contract(_obj({
        "ok": BOOL, "path": STR, "content": STR,
        "start_byte": INT, "end_byte": INT, "total_bytes": INT, "returned_bytes": INT,
        "next_offset_bytes": _nullable(INT), "eof": BOOL,
        "filesystem_policy": {"const": "structurally_denied"},
    }, ("ok", "path", "content", "start_byte", "end_byte", "total_bytes",
        "returned_bytes", "next_offset_bytes", "eof", "filesystem_policy")),
        "FileRead chunk; continue from next_offset_bytes until eof."),
    "list_files": _contract(_obj({
        "ok": BOOL, "files": _arr(_obj({"path": STR, "bytes": INT}, ("path", "bytes"))),
        "total_bytes": INT,
        "limits": _obj({"max_files": INT, "max_file_bytes": INT, "max_total_bytes": INT,
                        "max_path_bytes": INT, "max_path_parts": INT,
                        "suffixes": _arr(STR), "default_read_chunk_bytes": INT,
                        "max_read_chunk_bytes": INT},
                       ("max_files", "max_file_bytes", "max_total_bytes", "max_path_bytes",
                        "max_path_parts", "suffixes", "default_read_chunk_bytes",
                        "max_read_chunk_bytes")),
        "filesystem_policy": {"const": "structurally_denied"},
    }, ("ok", "files", "total_bytes", "limits", "filesystem_policy")),
        "FileList{files[{path,bytes}],total_bytes,limits}; virtual workspace only."),
    "run_program": _contract(_RUN_CODE,
        "CodeResult plus path; ok is program execution only, not nested robot-action success."),
    "done": _contract(_obj({"ok": BOOL, "note": STR}, ("ok", "note")),
        "Neutral acknowledgement only; no verifier verdict is returned."),
}


class ResultContractError(RuntimeError):
    """A tool implementation returned data that violates its declared wire contract."""


_TYPE_CHECKS = {
    "null": lambda value: value is None,
    "object": lambda value: isinstance(value, Mapping),
    "array": lambda value: isinstance(value, (list, tuple)),
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(float(value)),
}


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    if not schema:
        return []
    if "anyOf" in schema:
        branches = [_validate(value, branch, path) for branch in schema["anyOf"]]
        if any(not problems for problems in branches):
            return []
        return [f"{path} matches no allowed shape"]
    problems = []
    for branch in schema.get("allOf", ()):
        condition = branch.get("if") if isinstance(branch, Mapping) else None
        if condition is None:
            problems.extend(_validate(value, branch, path))
            continue
        matched = not _validate(value, condition, path)
        selected = branch.get("then") if matched else branch.get("else")
        if isinstance(selected, Mapping):
            problems.extend(_validate(value, selected, path))
    if "const" in schema and value != schema["const"]:
        return [f"{path} must equal {schema['const']!r}, got {value!r}"]
    expected = schema.get("type")
    if expected:
        check = _TYPE_CHECKS.get(expected)
        if check is None:
            return [f"{path} uses unsupported schema type {expected!r}"]
        if not check(value):
            return [f"{path} must be {expected}, got {type(value).__name__}"]
    choices = schema.get("enum")
    if choices is not None and value not in choices:
        return [f"{path} must be one of {list(choices)}, got {value!r}"]
    if expected == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for name in required:
            if name not in value:
                problems.append(f"{path}.{name} is required")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            child = f"{path}.{name}"
            if name in properties:
                problems.extend(_validate(item, properties[name], child))
            elif extra is False:
                problems.append(f"{child} is not declared")
            elif isinstance(extra, Mapping):
                problems.extend(_validate(item, extra, child))
    elif expected == "array":
        low, high = schema.get("minItems"), schema.get("maxItems")
        if low is not None and len(value) < int(low):
            problems.append(f"{path} needs at least {low} items")
        if high is not None and len(value) > int(high):
            problems.append(f"{path} allows at most {high} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                problems.extend(_validate(item, item_schema, f"{path}[{index}]"))
    return problems


def validate_result(tool: str, payload: Mapping[str, Any]) -> None:
    """Fail closed when a successful tool implementation violates its wire contract."""
    contract = RESULT_CONTRACTS.get(str(tool))
    if contract is None:
        # Test/fixture registries may deliberately add private tools.  Completeness of the shipped
        # surface is enforced by the inventory assertion above; an unknown private name has no
        # public contract to enforce.
        return
    problems = _validate(payload, contract.schema, "$result")
    if problems:
        detail = "; ".join(problems[:12])
        if len(problems) > 12:
            detail += f"; ... {len(problems) - 12} more"
        raise ResultContractError(f"{tool} output contract violation: {detail}")


def result_summary(tool: str) -> str:
    return RESULT_CONTRACTS[str(tool)].summary


def result_keys(tool: str) -> tuple:
    """Top-level key names of a tool's wire result, read off the declared schema.

    Derived rather than written by hand so a description can never drift from the contract it
    describes. The summaries say what a result MEANS; without the key names the caller has to
    guess how to subscript it, and guessing is what actually failed in live runs -- `.obs_id`
    used as an attribute on a dict, and `grasp_quat_candidates` read as `quat_wxyz_candidates`
    instead of `candidates`. Each wrong guess costs a tool call, and tool calls are the budget.
    """
    schema = RESULT_CONTRACTS[str(tool)].schema
    properties, required = _root_properties(schema)
    if not properties:
        return ()
    # Required keys first, in declaration order, then the optional ones: the model reads left to
    # right and the always-present fields are the ones it can subscript unconditionally.
    return (tuple(k for k in properties if k in required)
            + tuple(k for k in properties if k not in required))


def _root_properties(schema: Any):
    """Merge top-level object variants for compact result-key documentation."""
    if not isinstance(schema, Mapping):
        return {}, set()
    properties = schema.get("properties")
    if properties:
        return dict(properties), set(schema.get("required") or ())
    variants = [branch for branch in schema.get("anyOf", ())
                if isinstance(branch, Mapping) and branch.get("properties")]
    if not variants:
        return {}, set()
    merged = {}
    required = None
    for branch in variants:
        merged.update(branch["properties"])
        branch_required = set(branch.get("required") or ())
        required = branch_required if required is None else required & branch_required
    return merged, (required or set())


def _container_children(schema: Any):
    """(marker, child names) when a declared property is a container the caller must subscript.

    `[]` marks an array of objects, so `fingers[]{link_name, ...}` reads as "index it, then take
    one of these". Nullable properties are declared as anyOf, so follow that to the branch that
    carries the structure before deciding.
    """
    def unwrap(node):
        if not isinstance(node, Mapping):
            return {}
        if "anyOf" in node:
            for branch in node["anyOf"]:
                resolved = unwrap(branch)
                if resolved.get("type") in ("object", "array"):
                    return resolved
            return {}
        return node

    resolved = unwrap(schema)
    if resolved.get("type") == "array":
        item = unwrap(resolved.get("items") or {})
        if item.get("type") == "object" and item.get("properties"):
            return "[]", tuple(item["properties"])
        return None
    if resolved.get("type") == "object" and resolved.get("properties"):
        return "", tuple(resolved["properties"])
    return None


def result_key_outline(tool: str) -> tuple:
    """`result_keys`, with every top-level container expanded ONE level.

    Depth 1 alone is the layer that carries no information for exactly the tools that need it
    most: `preview_tcp_pose` declares 88 leaf fields and delivers six top-level names, of
    which the only decision-bearing one is the opaque `annotation`; the motion tools bury every
    outcome under `achieved`. A caller that reads the depth-1 line still has to guess the
    subscript, and guessing is what live runs actually failed at -- 5 of 6 observed run_code
    shape errors. One more level reaches the leaves that decide something.

    Positions are deliberately NOT offered: a JSON object is name-keyed, and the model-visible
    projection SORTS keys and drops the middle when a result exceeds its byte ceiling, so any
    positional assumption would break precisely when the result is largest.
    """
    schema = RESULT_CONTRACTS[str(tool)].schema
    properties, _required = _root_properties(schema)
    if not properties:
        return ()
    out = []
    for key in result_keys(tool):
        found = _container_children(properties.get(key, {}))
        if found is None:
            out.append(key)
        else:
            marker, names = found
            out.append(f"{key}{marker}{{{', '.join(names)}}}")
    return tuple(out)


def output_schema(tool: str) -> dict:
    return dict(RESULT_CONTRACTS[str(tool)].schema)


if set(RESULT_CONTRACTS) != {
        "get_world_frame", "get_embodiment", "get_camera_info", "get_arm_pose",
        "get_gripper_state", "get_robot_state", "grasp_quat_candidates",
        "check_tcp_pose_reachability",
        "get_grasp_contact", "capture_head",
        "capture_wrist", "capture_evidence_views", "project", "ray", "plane_intersect",
        "capture_motion_pair", "triangulate_correspondence", "scale_from_object_size",
        "scale_from_gripper", "draw_marks", "preview_tcp_pose", "compare_tcp_poses",
        "move_delta", "probe_contact_along", "move_both_delta", "reach_tcp",
        "reach_both_tcp", "set_gripper", "camera_aim_pose", "run_code",
        "write_file", "read_file", "list_files", "run_program", "done",
}:
    raise RuntimeError("result contract inventory drift")


from codeaction.extensions import declarations as _extensions
for _name, _entry in _extensions("tool").items():
    if _name in RESULT_CONTRACTS and not _entry.get("replace"):
        raise ValueError(f"tool {_name} already exists; declare replace: true")
    RESULT_CONTRACTS[_name] = ResultContract(_entry["output_schema"], _entry["returns"])
