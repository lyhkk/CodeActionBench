"""Single-source OpenAI tool schemas for the D0 tool set + typed-result serialization (spec §5/§6).
Descriptions document semantics, fixed internal mechanics, and failure modes, but never strategy or
a recommended call order (spec §0.1: a strategic hint would smuggle a workflow back in). Pixels are
always [u,v] in the image's OWN pixel coordinates (each Observation reports size_hw)."""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

from codeaction.contracts import embodiment as _emb
from codeaction.motion import orientation as _orientation
from codeaction.contracts.harness_parameters import AIM_PITCH_MAX_DEG, AIM_PITCH_MIN_DEG
from codeaction.contracts.result_contracts import (DEFAULT_PIXEL_SIGMA_PX, DRAW_MARKS_MAX_COUNT,
                                      DRAW_MARKS_MIN_COUNT, RESULT_CONTRACTS,
                                      RULER_MIN_PROJECTED_SPAN_PX, output_schema,
                                      result_key_outline, result_summary, validate_result)
from codeaction.contracts.types import ActionResult, Estimate, Observation, ObservationSet, ObservationPair

_ARM = {"type": "string", "enum": ["left", "right"]}
_ARM_OR_NONE = {"type": "string", "enum": ["left", "right", "none"],
                "description": "left/right labels the active arm; none returns view roles only"}
_CAM = {"type": "string", "enum": ["head_camera", "left_camera", "right_camera"]}
_WRIST_VIEWS = {"type": "string", "enum": ["active", "both"],
                "default": "active",
                "description": ("omitted/default active = active arm wrist only; both = left and "
                                "right wrist views")}
_EVIDENCE_VIEWS = {
    "type": "array",
    "items": {"type": "string", "enum": ["overview", "left_wrist", "right_wrist"]},
    "minItems": 1,
    "maxItems": 3,
    "uniqueItems": True,
    "default": ["overview", "left_wrist", "right_wrist"],
    "description": ("non-empty unique subset of overview, left_wrist, right_wrist; "
                    "omitted returns all three in that order"),
}
_PX = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
       "description": "[u,v] pixel in the referenced observation's own coordinates"}
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
        "description": "world [x,y,z] meters"}
_AXIS = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
         "description": "world-frame direction [x,y,z]; magnitude is ignored"}
_GRASP_AXIS_DOT_TOLERANCE_TEXT = f"{_orientation.AXIS_ORTHOGONALITY_TOLERANCE:.0e}"
_GRASP_APPROACH_AXIS = {
    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
    "description": ("world approach [x,y,z], normalized before use; abs(dot) with the normalized "
                    f"opening axis must be <= {_GRASP_AXIS_DOT_TOLERANCE_TEXT}")}
_GRASP_OPENING_AXIS = {
    "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
    "description": ("unoriented world opening line [x,y,z], normalized before use; to stay "
                    "orthogonal, abs(dot) with the normalized approach axis must be <= "
                    f"{_GRASP_AXIS_DOT_TOLERANCE_TEXT}")}
_DELTA = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
          "description": "world-frame displacement [dx,dy,dz] meters"}
_ARMS = {"type": "array", "items": _ARM, "minItems": 1, "maxItems": 2,
         "description": "subset of arms to report; duplicates are ignored"}
_PREVIEW_FINGER_GAP_MIN_M, _PREVIEW_FINGER_GAP_MAX_M = _emb.get_finger_gap_bounds_m()

RUN_CODE_INTERFACE_NOTE = (
    "Direct calls and run_code are equally supported; choose whichever interface fits the step. "
    "The initial composition contract is authoritative for limits and reset effects."
)

# Retained for historical replay and the explicitly non-release code-first experiment. The
# default reference/vendor surfaces no longer deliver these names; see InterfaceProfile.
PROGRAM_TOOL_NAMES = ("write_file", "read_file", "list_files", "run_program")

NONCONTACT_SCALE_NOTE = (
    "Three independent non-contact metric-scale sources are available: calibrated camera motion "
    "(`capture_motion_pair` then `triangulate_correspondence`), visible FK gripper geometry "
    "(`scale_from_gripper`), and a caller-supplied object-size prior "
    "(`scale_from_object_size`, always coarse). These obtain metric evidence without deliberate "
    "scene contact. A camera sensing move is still a real guarded robot motion: the caller "
    "chooses its direction and distance, and the result reports achieved motion and observed "
    "finger contact."
)

GRASP_EVIDENCE_NOTE = (
    "Generic grasp-geometry and contact evidence tools are available in the tool surface: "
    "grasp_quat_candidates converts caller-chosen approach/opening axes to robot quaternions; "
    "preview_tcp_pose shows where an uncommitted pose would put the gripper in the camera image, "
    "and compare_tcp_poses does the same for 2-4 poses at once plus their relative geometry; "
    "probe_contact_along measures changes "
    "in the contacting-finger set along a caller-chosen direction; get_grasp_contact reports the "
    "current unfiltered finger-world contact state. Every contact report carries the "
    "forward-kinematics world POSE of each contacting finger link — position and decoded "
    "orientation — so a touch is a metric observation of a world point and not only a stop signal."
)


def _obj(props, required):
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


TOOL_SPECS = {
    "get_world_frame": ("World-frame axis definition (meters) + a fresh head-camera view with a "
                        "labelled +X/+Y/+Z screen-space legend. The legend is non-metric: its "
                        "pixel positions, arrow angles, and lengths are display-only, are not "
                        "projected world points, and provide no world-to-pixel correspondence. "
                        "The required annotation separately reports origin, endpoint, label, and "
                        "complete-overlay visibility.", _obj({}, [])),
    "get_embodiment": ("Robot self-description card: gripper geometry, TCP definition, qualitative "
                       "camera-mount semantics, motion caps, and orientation-construction / "
                       "reachability interfaces. Numeric camera geometry comes from each live "
                       "Observation extrinsic E. It contains no preferred grasp direction, "
                       "candidate ranking, or scene-specific motion strategy.", _obj({}, [])),
    "get_camera_info": ("Read one camera's calibration and RGB image size without reading depth. "
                        "Wrist E is described as call-time current only after a successful camera "
                        "refresh; a legal-camera read failure returns available=false, explicit "
                        "freshness/read_failures, and null unavailable fields without ending the "
                        "episode. Each Observation carries its own snapshot for image geometry.",
                        _obj({"camera": _CAM}, ["camera"])),
    "get_arm_pose": ("Current EE and TCP pose [x,y,z,qw,qx,qy,qz] of one arm (proprioception). "
                     "Missing, malformed, non-finite, or failed pose reads return explicit null "
                     "fields and stable read_failures without changing tick or ending the episode.",
                     _obj({"arm": _ARM}, ["arm"])),
    "get_gripper_state": ("Gripper state: drive-derived opening_m and gripper_val (normalized "
                          "drive value, 1.0 = fully open — both track the COMMAND when the "
                          "fingers are physically blocked) and finger_gap_m = FK finger-link "
                          "distance (physical; compare against get_embodiment().gripper."
                          "empty_close_finger_gap_m). Pose, drive, and FK read failures remain "
                          "separate explicit nulls with stable read_failures and do not change tick.",
                          _obj({"arm": _ARM}, ["arm"])),
    "get_robot_state": ("Batched raw robot self-state with explicit world frame and units. Each "
                        "requested arm reports named joint positions/velocities with per-joint "
                        "units, TCP/EE [x,y,z,qw,qx,qy,qz], physical finger gap, drive state, "
                        "and anonymous contact impulse. Unavailable sensor reads remain explicit "
                        "nulls and appear in read_failures. This is robot proprioception/contact "
                        "only; it does not report scene objects, scores, or depth.",
                        _obj({"arms": _ARMS}, [])),
    "grasp_quat_candidates": ("Convert caller-chosen TCP approach/opening axes into two wxyz "
                              "orientations separated by 180-degree wrist roll. This pure "
                              "robot-frame conversion does not inspect the scene or test "
                              "reachability, and it does not rank or select a candidate. Inputs are normalized; "
                              f"abs(dot(normalized approach, normalized opening)) must be <= "
                              f"{_GRASP_AXIS_DOT_TOLERANCE_TEXT}. Larger values are rejected; a "
                              "smaller numerical residual is orthogonalized and disclosed in "
                              "result.input_axes.",
                              _obj({"approach_axis_world": _GRASP_APPROACH_AXIS,
                                    "opening_axis_world": _GRASP_OPENING_AXIS},
                                   ["approach_axis_world", "opening_axis_world"])),
    "check_tcp_pose_reachability": ("Run one non-executing, self-collision-aware single-arm "
                                    "motion-planning query from the arm's current joint "
                                    "configuration. target_quat omitted keeps the current nominal "
                                    "TCP orientation. What the planner's world contains is fixed "
                                    "for the episode and declared once as planner_collision_world "
                                    "in the initial episode configuration, so no result repeats "
                                    "it. stage says which path produced this result. The two normal "
                                    "values are planner_query, meaning the motion planner actually "
                                    "ran the self-collision-aware query, and workspace_prefilter, "
                                    "meaning the coarse safety envelope rejected the target before "
                                    "any planner query ran; reason then names the violated bound. "
                                    "planner_exception means the planner raised and reason carries "
                                    "the error. trajectory_sample_count is the "
                                    "number of samples in the planner's interpolated trajectory at "
                                    "interpolation_dt_s, not a path length. On refusal, "
                                    "planner_diagnostic.code/meaning distinguishes a reported "
                                    "goal_pose_unreachable no-IK condition from trajectory-"
                                    "optimization or graph-search solver outcomes, and preserves "
                                    "native status, attempt count, valid_query, and endpoint "
                                    "residuals when available. These are facts from this query, not proof "
                                    "of global infeasibility; they can change after either arm "
                                    "moves. Targets outside workspace_envelope_world_m declared in "
                                    "the initial episode configuration are rejected before the query; "
                                    "that coarse safety envelope is not a reachability or collision-free "
                                    "proof. No trajectory is executed and no scene actor is read.",
                                    _obj({"arm": _ARM, "target_xyz": _XYZ,
                                          "target_quat": {
                                              "type": "array", "items": {"type": "number"},
                                              "minItems": 4, "maxItems": 4,
                                              "description": "wxyz"}},
                                         ["arm", "target_xyz"])),
    "get_grasp_contact": ("Finger↔world contact impulse (unfiltered by identity; finger↔finger "
                          "self-touch excluded — contact with any part of the robot's own body is "
                          "not a world contact), plus the forward-kinematics world POSE of each "
                          "contacting finger link, orientation included. fingers_in_contact "
                          "counts the fingers whose impulse exceeds the contact threshold declared "
                          "in the initial episode configuration, and per_finger carries the "
                          "impulses themselves; two "
                          "fingers in contact means two anonymous world contacts, not contact with "
                          "one object and not a grasp. Contact is reported wherever it occurs; "
                          "nothing here names, classifies, or scores what was touched. available=true "
                          "means the contact backend was read, including a real zero-contact result; "
                          "available=false with contact_unavailable means no contact value was "
                          "measured. contact_impulse_unit, arm, and tick are always returned.",
                          _obj({"arm": _ARM}, ["arm"])),
    "capture_head": ("Capture the fixed head-camera RGB overview → Observation {obs_id, size_hw, "
                     "K/E snapshot, tick}; image attached. No depth channel. The head view gives "
                     "workspace-wide context such as appearance, coarse location, and scene changes.",
                     _obj({}, [])),
    "capture_wrist": ("Capture wrist-camera RGB close view(s) → ObservationSet. "
                      "Serialized shape: {observations:[Observation,...], "
                      "roles:{role:obs_id}, obs_ids:[...], tick}. "
                      "views omitted/default='active' returns the active arm wrist camera; "
                      "views='both' returns "
                      "left and right wrist cameras. active_arm may be left, right, or none; "
                      "active/opposite roles are labelled only when an active arm is provided. "
                      "Wrist-camera optical axes follow their calibrated arm mounts and current "
                      "poses. No depth channel.",
                      _obj({"active_arm": _ARM_OR_NONE, "views": _WRIST_VIEWS}, ["active_arm"])),
    "capture_evidence_views": ("Capture a selected same-tick RGB bundle from the head, left "
                               "wrist, and/or right wrist cameras → ObservationSet with "
                               "serialized shape "
                               "{observations:[Observation,...], roles:{role:obs_id}, "
                               "obs_ids:[...], tick}. views omitted returns overview, left_wrist, "
                               "and right_wrist in that order; a non-empty unique subset returns "
                               "only those roles. active_wrist/opposite_wrist are included only "
                               "when active_arm is left or right and the corresponding selected "
                               "wrist view exists. Each image is 2D RGB with its own K/E snapshot; "
                               "no depth channel.",
                               _obj({"active_arm": _ARM_OR_NONE,
                                     "views": _EVIDENCE_VIEWS}, [])),
    "project": ("Project a caller-supplied world point through one observation's calibrated "
                "K/E snapshot → Estimate(value=[u,v] pixel; None if behind the camera). "
                "uncertainty=0 means deterministic transform only: it excludes caller world-point "
                "error and camera calibration error, and does not verify that the point is a real "
                "scene point.", _obj({"obs_id": {"type": "string"}, "xyz": _XYZ},
                                     ["obs_id", "xyz"])),
    "ray": ("Back-project a pixel of an observation → Estimate(value={origin, dir} world ray). "
            "A single ray carries direction only and contains no scene depth; origin and dir are "
            "not an object XYZ position. valid means the calibrated ray is numerically computable, "
            "not that px sampled the RGB: provenance.pixel_in_frame and image_size_hw distinguish "
            "an in-frame pixel from a mathematically extrapolated coordinate. uncertainty=0 means "
            "deterministic transform only and excludes caller pixel-annotation and camera "
            "calibration error.",
            _obj({"obs_id": {"type": "string"}, "px": _PX}, ["obs_id", "px"])),
    "plane_intersect": ("Intersect a pixel's ray with a caller-supplied plane of ANY orientation "
                        "(a point on the plane + its normal) → Estimate(value=[x,y,z]). The "
                        "plane definition is supplied entirely by the caller; no scene surface "
                        "or acquisition method is assumed, and this tool cannot check whether "
                        "the plane you supplied corresponds to anything physical, so the result "
                        "is always marked coarse. Fails (value None) when the ray is parallel to "
                        "the plane or the intersection lies behind the camera. The returned "
                        "uncertainty combines your pixel_sigma_px with your declared "
                        "plane_offset_sigma_m. pixel_sigma_px omitted/default=2.0 is an independent "
                        "isotropic 1-sigma annotation error in u/v; 0 asserts an exact pixel. The "
                        "tool propagates the complete local u/v ray-plane Jacobian. Both terms "
                        "grow with grazing geometry, and provenance returns both sensitivities.",
                        _obj({"obs_id": {"type": "string"}, "px": _PX,
                              "plane_point_xyz": _XYZ,
                              "plane_normal_xyz": {
                                  "type": "array", "items": {"type": "number"},
                                  "minItems": 3, "maxItems": 3,
                                  "description": "world-frame plane normal [x,y,z]; magnitude "
                                                 "is ignored"},
                              "plane_offset_sigma_m": {
                                  "type": "number", "minimum": 0.0,
                                  "description": "your own 1-sigma uncertainty, in meters, about "
                                                 "where this plane sits along its normal. The "
                                                 "harness cannot derive it — a plane you assumed "
                                                 "and a plane you measured are indistinguishable "
                                                 "here — so it enters the error model only "
                                                 "because you declare it. 0.0 asserts the plane "
                                                 "position is exact."},
                              "pixel_sigma_px": {
                                  "type": "number", "minimum": 0.0,
                                  "default": DEFAULT_PIXEL_SIGMA_PX,
                                  "description": "caller-owned independent isotropic 1-sigma "
                                                 "annotation error in both u and v pixels; "
                                                 "omitted/default=2.0; 0.0 asserts exact px"}},
                             ["obs_id", "px", "plane_point_xyz", "plane_normal_xyz",
                              "plane_offset_sigma_m"])),
    "capture_motion_pair": ("NON-CONTACT calibrated motion acquisition. Captures the selected "
                          "arm's "
                          "wrist RGB before and after one REAL guarded world-frame displacement "
                          "chosen entirely by the caller. Returns an ObservationPair with both "
                          "images, achieved camera translation/rotation, ActionResult, and "
                          "finger-contact evidence. It does not identify a target, match pixels, "
                          "or choose the motion. Its nested motion returns only raw before/final "
                          "robot poses, not an error or intermediate waypoints. Unlike "
                          "probe_contact_along, this is not a contact measurement. validity "
                          "separately reports motion_succeeded, baseline_nonzero, and contact_free.",
                          _obj({"arm": _ARM,
                                "dx": {"type": "number", "description": "world x meters"},
                                "dy": {"type": "number", "description": "world y meters"},
                                "dz": {"type": "number", "description": "world z meters"}},
                               ["arm", "dx", "dy", "dz"])),
    "triangulate_correspondence": ("NON-CONTACT calibrated two-view geometry for one "
                                    "capture_motion_pair ObservationPair. The caller supplies the "
                                    "same scene point in the before and after images; the tool "
                                    "returns its world [x,y,z], pixel-propagated uncertainty, ray "
                                    "gap, triangulation angle, achieved pair validity, and "
                                    "correspondence_verified=false. Top-level valid means only that "
                                    "the numerical geometry is solvable; it does not verify the "
                                    "correspondence or acquisition quality; coarse=false only says "
                                    "no object-size prior was used. pixel_sigma_px is your "
                                    "independent isotropic 1-sigma annotation error in u/v; "
                                    "omitted/default=2.0 and 0.0 asserts exact pixels.",
                                    _obj({"pair_id": {"type": "string"},
                                          "px_before": _PX, "px_after": _PX,
                                          "pixel_sigma_px": {
                                              "type": "number", "minimum": 0.0,
                                              "default": DEFAULT_PIXEL_SIGMA_PX,
                                              "description": "caller-owned independent isotropic "
                                                             "1-sigma annotation error in u/v "
                                                             "pixels"}},
                                         ["pair_id", "px_before", "px_after"])),
    "scale_from_object_size": ("NON-CONTACT COARSE depth from a real projected bbox extent prior "
                               "YOU supply. extent_axis=width uses |x2-x1|/fx, height uses "
                               "|y2-y1|/fy, and diagonal uses their normalized hypotenuse; "
                               "Z≈known_extent_m/normalized_image_extent. known_extent_m must "
                               "describe that same apparent projected bbox extent, not an "
                               "unrelated object dimension. "
                               "The prior is yours to justify: unless the task states an object's "
                               "size, commonsense product sizes may not match this scene's "
                               "objects. When the optional extent sigma is omitted, uncertainty "
                               "is unknown/null rather than guessed. Returned "
                               "Estimate is always coarse=true.",
                               _obj({"obs_id": {"type": "string"},
                                     "bbox": {"type": "array", "items": {"type": "number"},
                                              "minItems": 4, "maxItems": 4,
                                              "description": "[x1,y1,x2,y2] pixel box"},
                                     "extent_axis": {"type": "string",
                                                     "enum": ["width", "height", "diagonal"]},
                                     "known_extent_m": {"type": "number", "minimum": 1e-6,
                                                        "description": "meters for the selected "
                                                                       "extent_axis"},
                                     "known_extent_sigma_m": {
                                         "type": "number", "minimum": 0.0,
                                         "description": "caller 1-sigma error in meters; omit if "
                                                        "unknown"}},
                                    ["obs_id", "bbox", "extent_axis", "known_extent_m"])),
    "scale_from_gripper": ("NON-CONTACT visible robot ruler. Captures the caller-selected RGB "
                           "camera at the current tick and overlays the selected arm's two live "
                           "FK fingertips with their true 3D separation. Returns exact endpoint "
                           "pixels, pixel span, camera depths, opening-axis direction, and whether "
                           "both endpoints are inside the image. A valid reference also requires "
                           f"finite span_px>{RULER_MIN_PROJECTED_SPAN_PX:g}; otherwise "
                           "failure_reason is endpoint_out_of_frame or "
                           "degenerate_projected_span. `valid_metric_reference=false` "
                           "means no visible scale evidence was obtained and meters-per-pixel is "
                           "withheld. It does not move the robot or touch the scene. Transferring "
                           "a valid local pixel scale to a target "
                           "assumes a similar camera range and orientation.",
                           _obj({"camera": _CAM, "arm": _ARM}, ["camera", "arm"])),
    "draw_marks": ("Overlay labelled circles at pixels YOU choose on an observation → a new "
                   "annotated Observation (image attached). This is a model-selected pixel "
                   f"evidence check, not detection. {DRAW_MARKS_MIN_COUNT}–"
                   f"{DRAW_MARKS_MAX_COUNT} marks.",
                   _obj({"obs_id": {"type": "string"},
                         "pixels": {"type": "array", "items": _PX,
                                    "minItems": DRAW_MARKS_MIN_COUNT,
                                    "maxItems": DRAW_MARKS_MAX_COUNT},
                         "labels": {"type": "array", "items": {"type": "string"},
                                    "minItems": DRAW_MARKS_MIN_COUNT,
                                    "maxItems": DRAW_MARKS_MAX_COUNT}},
                        ["obs_id", "pixels", "labels"])),
    "preview_tcp_pose": ("SEE WHERE A TCP POSE YOU HAVE NOT COMMITTED TO WOULD PUT THE GRIPPER, "
                             "in this camera image, before anything moves. Nothing executes and no "
                             "arm state changes. Reachability answers a different question: a pose "
                             "the planner can solve may still sit inside a surface or beside the "
                             "object, and only this image shows that, against the scene pixels you "
                             "can already see. "
                             "Actual finger collision-shape silhouettes become transparent claws; "
                             "their inner tips are shaded 3D markers with inward closing arrows to "
                             "the green grasp_center. A cyan display-only arrow beside the gripper "
                             "is parallel to the proposed approach direction and ends at a square "
                             "display anchor, not TCP or a scene point; its label gives the 3D "
                             "angle to the camera view line. projected_length_px and "
                             "nearly_along_view report when the 2D cue is weak. Its fixed physical "
                             "reference length is not a planned standoff. Full "
                             "approach/opening/lateral vectors remain numeric. "
                             "finger_gap_m is the FK distance between the two finger-link origins. "
                             "A caller value must be inside the active embodiment's nominal "
                             f"[{_PREVIEW_FINGER_GAP_MIN_M:.3f}, "
                             f"{_PREVIEW_FINGER_GAP_MAX_M:.3f}] m range; when omitted, the live "
                             "current-tick FK gap is used. The input Observation must also be from "
                             "the current tick, so the RGB, camera calibration, robot collision "
                             "geometry, and any live gap are not mixed across robot states. "
                             "annotation.sampling reports those ticks and whether the gap came "
                             "from the caller or live FK. Optional reference_px marks a "
                             "caller-selected IMAGE pixel; optional reference_xyz_world projects a "
                             "caller-selected WORLD point instead. They are mutually exclusive. The "
                             "overlay reports their pixel offset; a pixel alone is a camera ray, "
                             "not a 3D origin, so world offset exists only for reference_xyz_world. "
                             "This is pose geometry/alignment evidence only: no object identity or "
                             "scene pose is read, and collision and "
                             "reachability are not evaluated.",
                             _obj({"obs_id": {"type": "string"}, "arm": _ARM,
                                   "tcp_xyz": _XYZ,
                                   "quat_wxyz": {"type": "array", "items": {"type": "number"},
                                                 "minItems": 4, "maxItems": 4},
                                   "finger_gap_m": {
                                       "type": "number",
                                       "minimum": _PREVIEW_FINGER_GAP_MIN_M,
                                       "maximum": _PREVIEW_FINGER_GAP_MAX_M,
                                       "description": "candidate FK distance between the two "
                                                      "finger-link origins in metres; bounded by "
                                                      "the active embodiment's nominal controllable "
                                                      "range"},
                                   "reference_px": _PX,
                                   "reference_xyz_world": _XYZ},
                                  ["obs_id", "arm", "tcp_xyz", "quat_wxyz"])),
    "compare_tcp_poses": ("Pure comparison of 2-4 caller-proposed terminal TCP poses; nothing "
                             "executes or selects a candidate. The Observation must be current-tick. "
                             "Live robot finger collision geometry and any omitted live FK gaps are "
                             "sampled at annotation.sampling.robot_geometry_tick, equal to the "
                             "Observation tick. Each candidate keeps its pose, gap source, projected "
                             "finger outlines, grasp centre and approach axis. pairs reports "
                             "grasp-centre delta/distance, approach angle, and separation_m between "
                             "finger bounding boxes. Positive separation_m proves at least that gap; "
                             "zero/negative sets finger_bounding_boxes_overlap=true but does not "
                             "prove true collision or penetration. Those box fields are null for "
                             "same-arm alternatives. projected_bbox_overlap and nearer_to_camera are "
                             "2D legibility facts only. Scope is terminal fingers: no wrist/arm/scene, "
                             "reachability, planning, grasp quality, ranking or recommendation. "
                             "A caller gap must be within the active embodiment's nominal "
                             f"[{_PREVIEW_FINGER_GAP_MIN_M:.3f}, "
                             f"{_PREVIEW_FINGER_GAP_MAX_M:.3f}] m range.",
                             _obj({"obs_id": {"type": "string"},
                                   "candidates": {
                                       "type": "array", "minItems": 2, "maxItems": 4,
                                       "description": "candidate terminal poses to compare",
                                       "items": _obj(
                                           {"arm": _ARM, "tcp_xyz": _XYZ,
                                            "quat_wxyz": {"type": "array",
                                                          "items": {"type": "number"},
                                                          "minItems": 4, "maxItems": 4},
                                            "finger_gap_m": {"type": "number",
                                                             "minimum": _PREVIEW_FINGER_GAP_MIN_M,
                                                             "maximum": _PREVIEW_FINGER_GAP_MAX_M,
                                                             "description": "omit to use that "
                                                                            "arm's live gap"}},
                                           ["arm", "tcp_xyz", "quat_wxyz"])}},
                                  ["obs_id", "candidates"])),
    "move_delta": ("Guarded world-frame displacement relative to one arm's current nominal TCP "
                   "(≤0.30 m, "
                   "clamped). path selects a fixed path discipline, not a quality level. "
                   "'waypoint_legs' (default) re-aims every leg from the latest measured TCP toward "
                   "the fixed endpoint and, when orientation_anchor=true in the initial episode "
                   "configuration (the default), holds the call-start orientation at every "
                   "waypoint. Tracking drift can move later waypoint "
                   "targets off the original start-to-end line, and the planner-chosen trajectory "
                   "within each leg can bow laterally; a leg can be refused even when a different "
                   "route to the endpoint exists. 'single_plan' plans exactly once to that endpoint "
                   "without caller-imposed intermediate waypoints or a Cartesian-line constraint. "
                   "Contact identity does not permit or interrupt either mode. A refused leg "
                   "reports achieved.planner_status.code when available from a native "
                   "current-call diagnostic; otherwise failure_stage, failure_category, and "
                   "planner_detail still report the refusal boundary. ABORTED with "
                   "trajectory_deviation applies the rule declared in the initial episode "
                   "configuration and returns the triggering applied limit in "
                   "achieved.guard.bound. achieved.path_mode states what actually ran. FAILED can "
                   "follow a partial move; read "
                   "execution.physics_steps, execution.partial, and observed_after before "
                   "recovery. Final measured TCP pose may differ from the command.",
                   _obj({"arm": _ARM, "dx": {"type": "number"}, "dy": {"type": "number"},
                         "dz": {"type": "number"},
                         "path": {"type": "string",
                                  "enum": ["waypoint_legs", "single_plan"]}}, ["arm"])),
    "probe_contact_along": ("Guarded contact measurement along a caller-supplied world-frame "
                             "direction. The caller supplies arm, direction_xyz, distance_m, and "
                             "step_m; the tool does not locate a target or choose a direction or "
                             "orientation. With orientation_anchor=true in the initial episode "
                             "configuration (the default), it holds the call-start TCP orientation "
                             "and compares "
                             "the contacting-finger set after each completed leg, stopping on a "
                             "change including contact gain or loss. Contact is anonymous and "
                             "unfiltered by object identity. SUCCESS means only that this set "
                             "changed; FAILED covers invalid/pre-read failure, no post-leg change "
                             "within the budget, or leg failure; inspect the failure fields. "
                             "achieved.effective_step_m is the maximum nominal "
                             "waypoint spacing after clamping to contact_leg_m; a final or re-aimed "
                             "leg may be shorter. It is not a measured travelled "
                             "distance or overshoot bound. achieved.travel_budget_m is the "
                             "clamped requested budget, not distance travelled. "
                             "A refused leg "
                             "reports achieved.planner_status.code when available from a native "
                             "current-call diagnostic; otherwise the stable failure fields remain. "
                             "ABORTED/trajectory_deviation uses "
                             "the declared motion safety rule.",
                             _obj({"arm": _ARM,
                                   "direction_xyz": _AXIS,
                                   "distance_m": {"type": "number",
                                                   "minimum": 1e-9,
                                                   "description": "positive travel budget in meters; clamped to the motion cap"},
                                   "step_m": {"type": "number",
                                              "minimum": 0.002,
                                              "description": "positive nominal waypoint spacing in meters; clamped to the configured leg maximum and returned as achieved.effective_step_m"}},
                                  ["arm", "direction_xyz", "distance_m", "step_m"])),
    "move_both_delta": ("Synchronized world-frame displacement of both arms through one paired "
                        "primitive call. Each delta is clamped to the per-call cap. Because a "
                        "paired primitive may execute the successful arm when only one arm's plan "
                        "succeeds and still return FAILED, FAILED does not imply zero motion: read "
                        "execution.partial, execution.physics_steps, and both arms in "
                        "observed_after before recovery. achieved.sync.planner_status carries a "
                        "normalized current-call refusal diagnostic per arm, including optional "
                        "native raw evidence. planner_status_available is always explicit; false "
                        "means this call produced no refusal diagnostic to return, including when "
                        "planning succeeded. Final measured TCP poses may differ from the "
                        "commands.",
                        _obj({"left_delta": _DELTA, "right_delta": _DELTA},
                             ["left_delta", "right_delta"])),
    "reach_tcp": ("Transport one arm's nominal TCP/grasp centre to an absolute world target. The "
                  "fixed internal procedure attempts one full plan; if that is refused, it tries "
                  "chunked waypoint plans toward the same target; after a completed primary "
                  "transport it may run up to two bounded orientation-holding correction plans when "
                  "the residual exceeds the threshold declared in the initial episode "
                  "configuration. These planner-selected trajectories are not promised Cartesian "
                  "lines. target_quat omitted "
                  "keeps the call-start TCP orientation; supplied means a full 6-DoF target. The "
                  "top-level status and planning.status scope the primary transport, including "
                  "chunk fallback; a "
                  "correction refusal or exhausted correction budget does not retroactively change "
                  "a completed transport to FAILED. Conversely, FAILED may be partial: read "
                  "execution.partial, execution.physics_steps, and observed_after. The Harness "
                  "does not turn the final residual into a verdict. The final measured TCP pose "
                  "can differ; no position/angle error is returned, so compute it from commanded "
                  "and observed_after. Raw observed_before/observed_after remain available. Native "
                  "planner evidence is returned on transport refusal when available. Targets "
                  "outside workspace_envelope_world_m declared in the initial episode "
                  "configuration are rejected before simulation execution; that coarse safety "
                  "envelope is not a reachability or collision-free proof.",
                  _obj({"arm": _ARM, "target_xyz": _XYZ,
                        "target_quat": {"type": "array", "items": {"type": "number"},
                                        "minItems": 4, "maxItems": 4,
                                        "description": "wxyz"}}, ["arm", "target_xyz"])),
    "reach_both_tcp": ("Synchronized dual-arm transport of both nominal TCP/grasp centres through "
                       "one paired primitive call. Each arm may keep its current orientation or use "
                       "an explicit wxyz quaternion. A paired failure can occur after one or both "
                       "arms moved, so FAILED does not imply zero motion: read execution.partial, "
                       "execution.physics_steps, and both arms in observed_after before recovery. "
                       "achieved.sync.planner_status carries a normalized current-call refusal "
                       "diagnostic per arm, including optional native raw evidence. "
                       "planner_status_available is always explicit; false means this call produced "
                       "no refusal diagnostic to return, including when planning succeeded. Final "
                       "measured TCP poses may differ "
                       "from the targets. Targets outside workspace_envelope_world_m declared in "
                       "the initial episode configuration are rejected before simulation execution; "
                       "that coarse safety envelope is not a reachability or collision-free proof.",
                       _obj({"left_xyz": _XYZ, "right_xyz": _XYZ,
                             "left_quat": {"type": "array", "items": {"type": "number"},
                                           "minItems": 4, "maxItems": 4,
                                           "description": "wxyz"},
                             "right_quat": {"type": "array", "items": {"type": "number"},
                                            "minItems": 4, "maxItems": 4,
                                            "description": "wxyz"}},
                            ["left_xyz", "right_xyz"])),
    "set_gripper": ("Command one gripper to a normalized drive position: pos 1.0 is fully open, "
                    "0.0 is fully closed, and intermediate values are allowed. achieved reports "
                    "drive readback, physical finger gap, and anonymous per-finger contact. An "
                    "obstruction can make the physical gap disagree with the commanded drive; "
                    "actuation SUCCESS alone is not grasp evidence. The episode tick advances only "
                    "when execution evidence shows real actuation; with a simulator clock this "
                    "means this call advanced physics. execution.physics_steps and "
                    "execution.partial report that evidence.",
                    _obj({"arm": _ARM,
                          "pos": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                         ["arm", "pos"])),
    "camera_aim_pose": ("Pure geometry: for one caller-supplied wrist camera, world point, pitch, "
                        "and standoff, compensate the calibrated TCP-to-camera mount and return "
                        "exactly one unexecuted target_tcp_pose_world plus its predicted point "
                        "projection. It does not select candidates, query a planner, move the "
                        "robot, change tick, test collision, inspect RGB visibility/occlusion, or "
                        "decide task success. pitch is the world-Y pre-rotation defined by its "
                        "parameter schema; the caller chooses it and standoff. valid=false means "
                        "current TCP self-state was unavailable, not that the pose is unreachable. "
                        "projected_point_in_frame tests only calibrated image bounds. The returned "
                        "pose is directly compatible with check_tcp_pose_reachability and "
                        "reach_tcp; the caller decides whether and when to use it.",
                        _obj({
                            "camera": {"type": "string",
                                       "enum": ["left_camera", "right_camera"]},
                            "target_xyz": _XYZ,
                            "pitch": {"type": "number", "minimum": AIM_PITCH_MIN_DEG,
                                      "maximum": AIM_PITCH_MAX_DEG, "default": 80.0,
                                      "description": "single caller-selected world-Y "
                                                     "pre-rotation in degrees applied to the "
                                                     "embodiment base TCP orientation; 90 is in "
                                                     "the top-down TCP family and 60 is 30 "
                                                     "degrees shallower"},
                            "standoff": {"type": "number", "minimum": 1e-6,
                                         "default": 0.28,
                                         "description": "target-plane to TCP height in metres"},
                        }, ["camera", "target_xyz"])),
    "run_code": ("Execute Python in an isolated sandbox; every tool is pre-bound as a "
                 "keyword-only function returning "
                 "a plain dict. Also pre-bound: np, math, load_image(obs_id) → HxWx3 uint8 pixel "
                 "copy, and print(). `import numpy as np` and `import math` are accepted but "
                 "unnecessary; other imports, dunder attributes, eval/open/getattr are rejected. "
                 "Assign `result` to return a consolidated value; print() alone returns "
                 "value=null with result_assigned=false. Omitting result is valid only when a "
                 "block intentionally defines persistent variables/functions for later calls. "
                 "Internal payloads needed later must be copied into `result`; internal_trace only "
                 "summarizes calls. Variables and defs persist unless namespace_reset=true. Images "
                 "captured or explicitly loaded with load_image(obs_id) inside code are attached "
                 "to the next message.",
                 _obj({"code": {"type": "string"}}, ["code"])),
    "write_file": ("Create or replace one agent-owned `.py` or `.md` text file in the restricted "
                   "virtual workspace. This does not write to the "
                   "benchmark host, source tree, task package, verifier, or result directory.",
                   _obj({"path": {"type": "string"}, "content": {"type": "string"}},
                        ["path", "content"])),
    "read_file": ("Read one UTF-8 byte-aligned chunk from a virtual file. Continue from "
                  "next_offset_bytes until eof; a chunk may be shorter than max_bytes to preserve "
                  "UTF-8 and the model-result limit. No benchmark or host file is addressable.",
                  _obj({"path": {"type": "string"},
                        "offset_bytes": {"type": "integer", "minimum": 0, "default": 0},
                        "max_bytes": {"type": "integer", "minimum": 4, "maximum": 16384,
                                      "default": 8192}}, ["path"])),
    "list_files": ("List agent-owned files in this episode's restricted virtual workspace. "
                   "This cannot enumerate any host or benchmark directory.", _obj({}, [])),
    "run_program": ("Execute one stored `.py` file under the same sandbox and result contract as "
                    "run_code. The file persists independently of execution-namespace resets.",
                    _obj({"path": {"type": "string"}}, ["path"])),
    "done": ("Finish the attempt: report what you did and whether you believe the task is "
             "complete. This ends the episode.",
             _obj({"report": {"type": "string"},
                   "success_claim": {"type": "boolean"}}, ["report"])),
}

# Concise model-facing purposes.  The full field inventory and every implementation branch are
# owned by result_contracts.py and engineering documentation; repeating them here makes tool
# selection harder without adding a decision the model can take before the call.
MODEL_TOOL_PURPOSES = {
    "get_world_frame": ("Return the metric world-axis definition and a non-metric screen-space "
                        "legend on a fresh head RGB; it provides no world-to-pixel correspondence."),
    "get_embodiment": ("Return pose-independent robot, gripper and TCP geometry plus qualitative "
                        "camera-mount semantics; live numeric camera geometry is in Observation E."),
    "get_camera_info": "Return the camera's current K, world-to-camera E, image size and tick.",
    "get_arm_pose": "Read one arm's current EE and nominal TCP world poses with read failures.",
    "get_gripper_state": ("Read drive-derived gripper state and physical FK finger_gap_m with "
                           "source-specific read failures; a blocked gripper can make them disagree."),
    "get_robot_state": ("Read named raw joint qpos/qvel, poses, gripper state and anonymous "
                        "contact for selected arms, with frame, units and explicit read failures."),
    "grasp_quat_candidates": ("Convert caller-chosen world approach/opening axes into two wxyz "
                               "unranked roll-equivalent gripper orientations after declared "
                               "orthogonality validation; no reachability test."),
    "check_tcp_pose_reachability": ("Run one non-executing single-arm motion-planning query from "
                                     "the current joint state; false is not global infeasibility."),
    "get_grasp_contact": ("Read available/unavailable anonymous finger-to-world contact and "
                           "contacting finger-link FK poses; two contacts do not identify one object."),
    "capture_head": "Capture calibrated head-camera RGB; no depth.",
    "capture_wrist": "Capture one or both calibrated wrist RGB views with role labels; no depth.",
    "capture_evidence_views": ("Capture a selected same-tick subset of head and wrist RGB views; "
                               "no depth."),
    "project": "Project a world XYZ into one observation's pixel frame.",
    "ray": "Back-project one observation pixel to a calibrated world ray; it has no scene depth.",
    "plane_intersect": ("Intersect an observation ray with a caller-defined plane. The harness "
                        "cannot verify that the plane represents a physical surface."),
    "capture_motion_pair": ("Capture wrist RGB before and after one real guarded caller-selected "
                            "motion; no target or correspondence is selected."),
    "triangulate_correspondence": ("Triangulate caller-selected corresponding pixels from one "
                                   "motion pair; correspondence correctness is not verified."),
    "scale_from_object_size": "Estimate coarse depth from a caller-supplied real-size prior.",
    "scale_from_gripper": "Capture an RGB view with the selected arm's live FK fingertip ruler.",
    "draw_marks": "Draw 1-12 caller-selected labelled pixels on an observation.",
    "preview_tcp_pose": ("See where an uncommitted TCP pose would put the gripper in the camera "
                             "image, against the scene you can already see, before anything moves. "
                             "No collision or reachability verdict."),
    "compare_tcp_poses": ("See how 2-4 uncommitted TCP poses stand relative to EACH OTHER, in one "
                             "image and as numbers: grasp-centre offset, approach-axis angle, and "
                             "box-conservative clearance between terminal gripper geometry (null "
                             "between two poses of the same arm). No arm link, scene object, "
                             "reachability or grasp-quality verdict."),
    "move_delta": ("Execute a guarded world-frame arm displacement given a delta RELATIVE to "
                   "where the arm is now."),
    "probe_contact_along": ("Move along a caller-selected direction until finger-contact set "
                            "changes or travel ends. SUCCESS only means that set changed."),
    "move_both_delta": ("Execute one synchronized dual-arm displacement and report whether a "
                        "failed paired call already moved either arm."),
    "reach_tcp": "Execute guarded planner-backed nominal-TCP transport to an ABSOLUTE world target.",
    "reach_both_tcp": ("Execute synchronized dual-arm nominal-TCP transport and report whether a "
                       "failed paired call already moved either arm."),
    "set_gripper": ("Set a normalized gripper drive position and read back drive, physical gap "
                    "and contact. SUCCESS is actuation completion, not grasp evidence."),
    "camera_aim_pose": ("Compute one caller-configured wrist-camera TCP pose and predicted point "
                        "projection; pure geometry with no planning, selection, or execution."),
    "run_code": ("Execute restricted persistent Python with public tools, np, math and "
                 "load_image(obs_id). Assign result for anything you want back -- internal tool "
                 "payloads are not returned. ok describes code execution only; nested robot "
                 "outcomes remain in internal_trace/result."),
    "write_file": "Create or replace one .py or .md file in the isolated virtual workspace.",
    "read_file": "Read one reconstructable .py or .md chunk from the virtual workspace.",
    "list_files": "List files, byte totals and limits of the isolated virtual workspace.",
    "run_program": ("Execute a stored .py file under the same sandbox contract as run_code; ok "
                    "describes program execution, not nested robot-action success."),
    "done": "End the episode and record the report; the acknowledgement contains no verifier verdict.",
}

if set(MODEL_TOOL_PURPOSES) != set(TOOL_SPECS):
    raise ValueError("model-facing tool-purpose inventory drift")
# The compact purpose inventory is for audit/catalog views. The authored description is the one
# model-facing explanation; prepending the purpose makes every call pay twice for the same fact.
# Purpose remains a fallback so a tool can never ship without a description.
TOOL_SPECS = {
    name: ((authored or MODEL_TOOL_PURPOSES[name]), params)
    for name, (authored, params) in TOOL_SPECS.items()
}

# The exhaustive, machine-checked shapes live in result_contracts.py.  Only these compact
# decision-bearing signatures are delivered to the model.
TOOL_RETURNS = {name: result_summary(name) for name in TOOL_SPECS}


def _augment_with_returns(specs, returns):
    """Fold the declared return contract into each tool's delivered description (single source).

    Completeness is enforced in both directions so a new tool cannot ship with an undocumented
    return shape, and a stale return entry cannot outlive its tool.
    """
    undocumented = sorted(set(specs) - set(returns))
    if undocumented:
        raise ValueError(f"tools missing a declared return contract: {undocumented}")
    orphaned = sorted(set(returns) - set(specs))
    if orphaned:
        raise ValueError(f"return contracts for unknown tools: {orphaned}")
    out = {}
    for name, (description, params) in specs.items():
        text = f"{description}\n\nReturns: {returns[name].rstrip('.')}."
        # The key names come from the declared schema (result_key_outline), so this line cannot
        # drift from the contract. Without it the caller has to guess how to subscript the result
        # -- measured live as `.obs_id` on a dict and `quat_wxyz_candidates` for `candidates`.
        # Containers are expanded one level, because depth 1 alone names `annotation` / `achieved`
        # without saying what is inside them, which is the layer every decision actually needs.
        keys = result_key_outline(name)
        if keys:
            text += "\nResult keys (dict): " + ", ".join(keys) + "."
        out[name] = (text, params)
    return out


# The AUTHORED text, before the schema-derived return contract and key outline are appended. The
# anti-bloat gate belongs on this: it is the part a human writes and can pad, while the appended
# lines are generated from the declared schema and are already bounded by it. Measuring the merged
# string made a tool's prose allowance shrink as its RESULT grew, which penalises exactly the rich,
# well-declared results the surface wants.
from codeaction.extensions import declarations as _extensions
for _name, _entry in _extensions("tool").items():
    if _name in TOOL_SPECS and not _entry.get("replace"):
        raise ValueError(f"tool {_name} already exists; declare replace: true")
    TOOL_SPECS[_name] = (_entry["description"], _entry["input_schema"])
    TOOL_RETURNS[_name] = _entry["returns"]

AUTHORED_TOOL_SPECS = {name: text for name, (text, _) in TOOL_SPECS.items()}
TOOL_SPECS = _augment_with_returns(TOOL_SPECS, TOOL_RETURNS)


def build_openai_tools(names):
    """OpenAI `tools=` payload for the given tool names (single source: TOOL_SPECS)."""
    out = []
    for n in names:
        if n not in TOOL_SPECS:
            continue
        desc, params = TOOL_SPECS[n]
        out.append({"type": "function",
                    "function": {"name": n, "description": desc, "parameters": params}})
    return out


def get_output_schema(name):
    """Return the exhaustive internal wire-result schema for tests and MCP-capable transports."""
    return output_schema(name)


# `effective_step_m` used to be dropped here while `step_m` was silently clamped to the
# configured leg maximum. Preserve the maximum nominal waypoint spacing so the caller knows which
# control request ran; it is not measured path travel, and a final/re-aimed leg can be shorter.
_SERIALIZE_DROP_KEYS = {"measured_from"}
# Declared failure signals kept even when None: an Estimate returns value=None on failure (schema
# text promises it), so the null-drop compaction must not remove the key and contradict the schema.
_SERIALIZE_KEEP_NULL_KEYS = {
    "value", "uncertainty", "failure", "known_extent_sigma_m", "relative_extent_sigma",
    # Decision-bearing unavailable measurements stay explicit; optional diagnostics may still be
    # compacted.  Missing and measured-null must not collapse into the same model observation.
    "state_changed", "physics_steps", "partial", "center_error_px",
    "gripper_val", "opening_m", "finger_gap_m", "drive_commanded_closed",
    "finger_gap_minus_empty_close_m", "drive_error", "effective_step_m",
    "ee_pose", "tcp_pose", "tcp", "left_tcp", "right_tcp", "orientation", "reachable",
    "positions", "velocities", "contact",
    "tcp_px", "ee_px", "grasp_center_px", "finger_link_origin_center_px",
    "final_collision_bbox_px", "link_origin_px", "inner_tip_center_px", "final_bbox_px",
    "standoff_bbox_px", "standoff_tcp_px", "tip_px", "span_px",
    # preview_tcp_pose may legitimately place some or all of a candidate behind the camera.
    # These required-nullable fields must remain present so "not projectable" does not become an
    # output-contract violation.
    "start_px", "end_px", "projected_length_px", "display_start_px",
    "display_anchor_px", "arrow_end_px", "render_start_px", "render_end_px",
    "meters_per_pixel_along_segment", "opening_axis_world", "failure_reason",
    # camera_aim_pose keeps an unavailable current pose/prediction distinct from an implementation
    # that forgot the field.  The pure tool never emits planner or execution fields.
    "target_provenance", "current_tcp_pose_world", "target_tcp_pose_world",
    "predicted_projection", "pixel_uv", "center_error_px",
    # preview_tcp_pose.reference is a closed causal record. Null means the selected reference kind
    # cannot provide that quantity; absence would instead mean the implementation forgot it.
    "input_px", "input_xyz_world", "projected_px", "grasp_minus_reference_px",
    "pixel_distance", "grasp_minus_reference_world", "world_distance_m", "finger_gap_tick",
    # compare_tcp_poses: null is the ANSWER for a same-arm pair (two poses of one arm are
    # alternatives that never coexist, so their boxes are not compared) and for a pair whose silhouettes
    # do not overlap. Compacting these away would read as "not computed".
    "finger_bounding_boxes_overlap", "separation_m", "separation_axis_world", "nearer_to_camera",
    "silhouette_bbox_px", "label_px",
    # Motion-contract repair: these nulls are evidence, not optional decoration. A missing
    # planner diagnostic is different from a producer forgetting the required field; a probe with
    # no observed contact must not look like an older implementation that never measured it.
    "planner_found_trajectory", "planner_diagnostic", "physics_step",
    "blocked_in_contact", "pose_at_contact",
    # get_camera_info remains a successful read-only call when sensor data is unavailable.
    "K", "E", "size_hw",
}


def _san(x):
    if isinstance(x, dict):
        return {str(k): _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, np.ndarray):
        return _san(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        return _san(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def _compact(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in _SERIALIZE_DROP_KEYS:
                continue
            vv = _compact(v)
            if vv is None and k not in _SERIALIZE_KEEP_NULL_KEYS:
                continue
            out[k] = vv
        return out
    if isinstance(x, list):
        return [_compact(v) for v in x]
    return x


def serialize(result):
    """Typed tool result → JSON-safe dict for the role=tool message. Observations flag that their
    image arrives as an attached message (the transport concern lives in the runner)."""
    if isinstance(result, ObservationSet):
        d = _compact(_san(dataclasses.asdict(result)))
        for obs in d["observations"]:
            obs.pop("image_ref", None)
            obs["image"] = "attached in the next message"
        d["obs_ids"] = [obs.obs_id for obs in result.observations]
        d["images"] = "attached in the next message"
        return d
    if isinstance(result, ObservationPair):
        d = _compact(_san(dataclasses.asdict(result)))
        for key in ("before", "after"):
            d[key].pop("image_ref", None)
            d[key]["image"] = "attached in this tool result"
        d["obs_ids"] = [result.before.obs_id, result.after.obs_id]
        d["images"] = "attached in this tool result"
        return d
    if dataclasses.is_dataclass(result):
        d = _compact(_san(dataclasses.asdict(result)))
        if isinstance(result, Estimate):
            provenance = d.get("provenance") or {}
            tool = str(provenance.get("tool") or "")
            valid = d.get("value") is not None
            if result.kind == "point" and tool == "project":
                wire_kind, frame, units = (
                    "pixel_point", f"image:{provenance.get('obs_id', 'unknown')}", "px")
            elif result.kind == "point":
                wire_kind, frame, units = "world_point", "world", "m"
            elif result.kind == "ray":
                wire_kind, frame, units = "ray", "world", "origin:m;dir:unitless"
            elif result.kind == "depth":
                wire_kind, frame, units = "depth", "camera_ray", "m"
            else:
                wire_kind, frame, units = result.kind, "unspecified", "unspecified"
            note = str(provenance.get("note") or "estimate unavailable")
            code = "ESTIMATE_UNAVAILABLE"
            if "parallel" in note:
                code = "RAY_PARALLEL_TO_PLANE"
            elif "behind" in note:
                code = "BEHIND_CAMERA"
            elif "degenerate" in note:
                code = "DEGENERATE_GEOMETRY"
            d.update({
                "kind": wire_kind,
                "valid": valid,
                "frame": frame,
                "units": units,
                "failure": None if valid else {"code": code, "message": note},
            })
        if isinstance(result, Observation):
            d["image"] = "attached in the next message"
            d.pop("image_ref", None)          # a server path is meaningless to the model
        return d
    if isinstance(result, dict):
        return _compact(_san(result))
    return {"value": _compact(_san(result))}


def serialize_tool_result(name, result):
    """Serialize one successful implementation result and enforce its named output contract."""
    payload = serialize(result)
    validate_result(name, payload)
    return payload


def json_safe(value):
    """Normalize arbitrary composition/control data to strict-JSON values."""
    return _san(value)


def to_json(payload) -> str:
    return json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False)
