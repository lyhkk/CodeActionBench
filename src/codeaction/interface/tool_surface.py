"""Versioned tool-set and per-interface delivered-surface identity.

The D0 registry owns primitive membership and definitions.  Task cards select its exact
``{id, version, sha256}``; they do not copy tool names.  Control/composition tools, transport
metadata, and gateway-only extras are declared transforms outside the base digest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from codeaction.contracts.identity import sha256_json
from codeaction.interface.interface_extras import BASH_EXEC_DEFINITION
from codeaction.interface.registry import D0_TOOLS
from codeaction.interface.schemas import (PROGRAM_TOOL_NAMES, TOOL_SPECS, build_openai_tools,
                             get_output_schema)


# A frozen historical token: every task card pins it by value, and the D1/D2 rungs it once
# contrasted with have been removed (see interface/registry.py). It names this tool set, not
# a position on a ladder.
TOOL_SET_ID = "D0"
# 2.0.0 — every tool declares its return contract in the delivered description; `plane_intersect`
# requires `plane_offset_sigma_m` and always returns coarse; contact payloads carry the FK world
# position of each contacting finger link.
# 3.0.0 — self-contact now excludes the robot's WHOLE body, not just finger-to-finger, so a finger
# resting on its own wrist camera no longer reads as a world contact or trips contact-abort; the
# contact payload reports the contacting link's full POSE (`contacting_finger_pose`) instead of a
# bare position; every reported pose carries an `orientation` block decoding its quaternion into
# world approach/opening axes. The first changes when a guard fires and the second renames a field,
# so this is a new tested unit — 2.x results never aggregate with 3.x. (2.0.0 produced no runs.)
# 3.1.0 — additive: plane_intersect names its unverified input, and a commanded target that is a
# value this harness derived from such an input says so (exact float match, never fuzzy). The
# conditionality of a derived value did not previously survive being passed to a motion tool.
# 3.3.0 — TCP motion SUCCESS now requires measured position/orientation postconditions; planner
# success alone is not reported as tool success, and the delivered return contract exposes the
# postcondition diagnostics.
# 4.0.0 — exhaustive wire-result schemas are machine-validated in the production ToolBox path;
# model-facing descriptions are compact decision signatures; non-finite estimates use strict JSON
# validity/failure fields; motion results separate call/planning/execution/conditions; paired
# displacement and camera aiming gate SUCCESS on measured postconditions.
# 4.1.0 — `move_delta` decides SUCCESS from the MEASURED distance to the commanded target instead
# of from which loop exit fired. Its convergence test was a fixed 2 mm, which measurement places
# INSIDE this controller's own per-leg error (p50 1.11 mm / p95 2.16 mm — an absolute floor that
# does not shrink with leg length), so a displacement that had already been delivered could fail to
# terminate and be reported FAILED: across the two Arm-1 production batches every such run ended
# 1.88-3.77 mm from its target, while genuinely-short runs start at 39.34 mm. Tolerance is now
# `min(5 mm, max(2 mm, 0.25*|commanded|))`, and only the leg-budget/stall bookkeeping exits are
# upgraded to SUCCESS when the measured postcondition holds — planner failure, trajectory deviation
# and contact abort keep their own verdicts. This changes what SUCCESS means for one tool without
# changing any delivered description, so `base_tool_set_sha256` is UNCHANGED and the version is the
# only thing separating the two units: **`move_delta` outcomes never aggregate across 4.0.0/4.1.0.**
# 4.2.0 — multi-leg motion gets an orientation anchor, and residual rotation fails the call.
# `_straight_step` was called with no orientation target, so each leg held whatever the previous
# leg had drifted to: production logged 292 deg of cumulative uncommanded rotation over 36 calls
# (max 71.6 in one), and 66% of calls were issued from the resulting up-pointing wrist regime where
# plan failure runs 69%. Every leg of `move_delta` / `probe_contact_along` / `probe_contact_z` now
# holds the quaternion read at CALL START. Measured on the real tool: a clean displacement drifts
# 0.11-0.17 deg (was 1.30 mean / 34.9 max), and among cells both behaviours complete, drift falls
# 6x at identical delivered displacement. Because these tools command no rotation, leftover
# rotation is undelivered command: SUCCESS now also requires <= 5 deg (the same tolerance
# `reach_tcp` already uses). That reclassifies motions which used to report SUCCESS only by
# rotating the wrist 22-35 deg to get there. **`move_delta` and the contact probes never aggregate
# across 4.1.0 and 4.2.0.**
# 4.3.0 — a leg refused by the planner now reports WHY, as a normalized `achieved.planner_status`.
# Every planning refusal used to arrive as one `straight_line_plan_failure`, but the planner's own
# status separates two situations with opposite implications: measured over 28 failing cells
# replayed under six solver-effort settings, a goal pose with no IK solution recovered 0/22 while a
# trajectory-optimization failure recovered 4/6. `envs/robot/planner.py` now passes its raw status
# through additively (existing callers read status/position/velocity and are untouched), and
# `codeaction/planner_status.py` maps vendor strings to a stable code so swapping planners cannot
# silently change the agent-facing surface. Codes state the CONDITION only — never an action
# (§0.1 corollary 8), which `test_codeaction_planner_status` enforces against an advice word list.
# Also declares what 4.2.0 shipped undeclared: `move_delta`'s postcondition carries orientation
# fields, so its contract is the full POSTCONDITION rather than the position-only one.
# 4.4.0 — new tool `check_direction_feasibility`: plan, without executing, a hold-orientation
# displacement along a caller-chosen direction at caller-chosen distances, from the arm's CURRENT
# configuration. `check_tcp_pose_reachability` answers "can I reach absolute pose P", which is a
# different question: 31% of measured (position, orientation, direction) triples flip between IK
# branches of the SAME end-effector pose, so displacement feasibility belongs to the live joint
# configuration and cannot be derived from a pose — nor from any closed-form conditioning number
# (a directional-manipulability metric was tested and did not separate plannable from failed).
# Results are per-distance with the planner's own refusal code and deliberately carry NO maximum
# reach, because feasibility is measurably non-monotonic in distance. The caller supplies the
# distances; the harness picks none of them (§0.1: expose evidence, do not choose).
# 4.4.1 — declaration fix, no behaviour change: `moved_m` is returned by every leg-based tool
# but was declared per consumer, so PROBE_ACHIEVED missed it. An undeclared field is fatal
# under 4.0.0 wire validation, so any probe_contact_along call that could read both TCPs --
# the normal case, and a step in the constructive reference solution -- ended the episode
# with episode_fatal. It now lives once in FAILURE_FIELDS, and one real payload per
# leg-based tool is pushed through serialize_tool_result so the class cannot recur.
# 4.4.1 note (no version change): the orientation anchor is now a DECLARED harness parameter
# (`CODEACTION_ORIENTATION_ANCHOR`, default on = shipped behaviour), so its task-level effect can
# be A/B'd without editing production. It changes no delivered description, so the surface
# digest and the default behaviour are both unchanged and no re-pin is needed. Two runs that
# differ are separated at the RUN level: `orientation_anchor` is written into the transcript
# `meta` record beside `frame_retention`, and a run with it OFF is not a 4.4.1 result.
# 5.0.0 — the harness stops judging. Every field that compared a measured quantity against a
# harness-chosen bound and turned the answer into an agent-visible verdict is gone: the TCP
# postconditions (10 mm / 5 deg), the camera centring threshold (80 px), the gripper drive-target
# test (0.05), and the `conditions` blocks derived from them. None of those bounds had a
# measurement behind them, and the question they answered — "is this close enough?" — depends on
# what the caller intends to do next, which the harness structurally cannot know (§0.1).
# What replaces them is not less information but more: `achieved.residual` carries the same errors
# plus the SIGNED per-axis error and the delivered fraction, alongside the operands
# (`commanded_target_xyz`, `tcp_after_xyz`, `tcp_before`/`tcp_after`) they were computed from, so
# every derived number is recomputable by the reader. Doing the arithmetic is a bridge service;
# setting the bound is not.
# Top-level `status` keeps its three values and changes meaning to non-numeric facts: SUCCESS = the
# call executed and returned measurements, FAILED = it could not execute, ABORTED = a guard stopped
# it. Guards now report themselves as `achieved.guard{name, observed, bound, effect}` — a guard is
# an ACTION that stops the arm, not an evaluation of how well the motion went. `execution.status`
# no longer says COMPLETED merely because a TCP could still be read after a refused plan; whether a
# post-action state read was available is the separate `execution.post_state_observed`.
# Renames that stop a field from claiming more than its evidence: `reachable`/`feasible` ->
# `planner_found_trajectory` (a refusal is this planner's answer for this query — 31% of measured
# triples flip between IK branches of the same pose), `is_closed` -> `drive_commanded_closed`
# (upstream drive state, not physical closure). Removals: `planning_wall_s` (harness timing, not
# anything the robot senses, and a jittering float in the model's context is a reproducibility
# hazard), `n_contact_points` (a solver artefact with no real-robot counterpart), and
# `in_contact_both_fingers` (reads as "holding something"; the per-finger impulses say what is
# actually known). Additions: `achieved.residual`, `achieved.guard`, `drive_error`,
# `finger_gap_minus_empty_close_m` (named for the subtraction it performs — `object_width_m` would
# assert an object exists and is held square-on), `probe_contact_along.effective_step_m` (the leg
# actually walked; `step_m` is clamped to the configured maximum), and the `aim_camera` candidate
# evaluation echo.
# **No 4.x result aggregates with 5.x**: what SUCCESS means changed for every motion tool.
# 5.1.0 — closes the mechanical and delivered-surface gaps in the 5.0.0 migration. The common
# ActionResult no longer accepts an open `conditions` object, paired schemas no longer admit
# `postcondition_success`, and the model-facing purpose lines consistently describe residuals as
# caller evidence. the reference scaffold and the vendor agent now receive the actual harness parameter object in their
# hashed instruction surface, including orientation-anchor state. `move_delta` tests convergence
# immediately after each leg instead of using the final residual to promote an ABORTED result to
# SUCCESS. This is a breaking correction to both the result and instruction contracts, so 5.0 and
# 5.1 results never aggregate.
# 5.2.0 — `preview_tcp_pose` now draws what the gripper physically occupies. The old result
# called finger-link FK origins "fingertips" and rendered only their joining segment, so a caller
# could not see the finger bodies, TCP-to-EE offset, orientation axes, or the volume swept by its
# proposed final approach. It now derives simplified perspective finger boxes from the live SAPIEN
# collision vertices, marks the collision-derived inner-tip midpoint as `grasp_center`, labels
# local +x/+y/+z as approach/opening/lateral, and optionally draws a standoff ghost plus straight
# approach arrow. The payload reports per-finger link origins, inner-tip reference points, display
# and collision bounds, TCP/EE, grasp center, and decoded axes; it makes no collision, reachability,
# object, or grasp-quality judgment. The schema/result contract and image semantics are breaking,
# so 5.1 and 5.2 results never aggregate.
# 6.0.0 — unexpected contact is now an episode terminal rather than a recoverable ActionResult.
# Every motion is watched per physics step for new robot-to-world, self, or arm-to-arm contact.
# probe_contact_along alone permits new finger-to-world contact on the selected arm; any other new
# contact stops execution, locks the tool surface, and records a scoreable unintended_collision
# outcome. Benchmark episodes
# disable the fixed cuRobo table world and retain robot self-collision, so a privileged table model
# can neither leak the plane through reachability queries nor spend repeated solver attempts trying
# to route around a contact that the benchmark defines as failure. This changes motion semantics,
# the instruction contract, and the tested unit, so no 5.x result aggregates with 6.0.
# 6.1.0 — every mutating result now carries canonical observed_before/observed_after robot
# snapshots for BOTH arms: named joint qpos/qvel with units, EE/TCP poses, gripper reads, anonymous
# contact, tick, frame, and explicit read-failure codes. SUCCESS, FAILED, ABORTED, and the terminal
# unintended-collision payload retain the same boundary evidence; unavailable data is a required
# null rather than a missing field. Existing achieved/resulting_pose fields remain for compact
# decisions and compatibility, but their raw operands are now always present and recomputable.
# Contact impulse and contacting-finger FK poses are no longer rounded at the API boundary. This is
# a wire-result contract change, so 6.0 and 6.1 results never aggregate.
# 7.0.0 — motion tools stop calculating model-facing position/angle errors, executed deltas,
# delivered fractions, per-leg counters, and intermediate trajectory diagnostics. Segmented
# planning, convergence checks, and trajectory guards still use their measurements internally;
# results report only the command, execution/failure facts, contact evidence where applicable, and
# canonical raw observed_before/observed_after snapshots. Tool descriptions explicitly warn that
# the final measured pose may differ from the command and tell the caller to compare the raw poses
# if it needs a difference. This removes required wire fields, so no 6.x result aggregates with
# 7.0.
# Also in 7.0.0 (folded in before any 7.0 batch ran, so it needs no version of its own):
# `get_embodiment` carries a `frame` block stating the coordinate CONVENTION this interface runs
# on. Every coordinate in every argument and every result is in ONE world frame, so the caller
# never performs a frame conversion — the motion tools convert a world target into each arm's own
# base frame internally; the frame is right-handed; its origin is attached to the scene rather than
# to the robot; and the arm named `left` is mounted at negative x, `right` at positive x, NOT as an
# exact mirror of the other (measured from the two embodiment `frame_bias` vectors, which differ by
# ~9 mm in the mirrored component). Axis strings are still sourced from `world_frame`, so they keep
# exactly one definition. The convention was previously implicit: cuRobo's contract requires
# targets in the ROBOT BASE frame and `CuroboPlanner` performs that conversion behind the tools,
# but nothing said so to the model. Declaring it is the deliberate counterpart to never asking the
# model to transform frames — on real hardware that conversion belongs to tf / the driver stack,
# not to the policy, so pushing it onto the agent would be harder than reality rather than more
# faithful to it. The block states no position of anything in the scene, adds no tool and changes
# no result semantics; it moves the surface digest only because `get_embodiment`'s output schema
# gains a required `frame` property.
# Also in 7.0.0, before any 7.0 batch ran, candidate-pose drawing becomes one visually layered
# evidence tool. `draw_tcp_frame` leaves the agent-facing surface because it drew WORLD axes at the
# live TCP, duplicating get_world_frame while its name suggested TCP-local axes; the method remains
# available to internal diagnostics. `preview_tcp_pose` keeps collision-derived physical
# geometry and true perspective as one terminal-action overlay. Per-shape collision silhouettes
# become translucent claws; a fixed display-only approach arrow points into the terminal TCP, and
# shaded inner-tip markers carry two inward closing arrows toward the grasp center. There is no
# synthetic palm, TCP triad, separate orientation panel, or public `approach_distance_m`; complete
# approach/opening/lateral vectors remain numeric. A caller may add either an image pixel or a
# world point as a reference; the result reports grasp-minus-reference pixel displacement and, only
# for a world point, world displacement. It applies no alignment, collision, reachability, or
# grasp-quality verdict. This changes the delivered tool set and result schema but is folded into
# 7.0 because no 7.0 episode predates it.
# 7.1.0 — additive: `compare_tcp_poses` renders 2-4 caller-proposed terminal gripper poses in ONE
# image and returns their pairwise relative geometry. No existing tool's description, arguments,
# result, or semantics change, so 7.0 and 7.1 results aggregate for every tool except this new one.
# It is a separate tool rather than a batch mode of `preview_tcp_pose` for three reasons. (a)
# The two images have opposite ink budgets: per-candidate detail (per-shape fills, inner-tip
# markers, closing arrows, an angle-labelled approach arrow) is exactly what makes a second
# candidate unreadable, so each candidate here is reduced to two collision outlines, a labelled
# grasp centre, and a short approach stub. (b) The pairwise payload has no place in a
# single-candidate contract. (c) A `combine` flag would make the RETURN TYPE depend on an argument
# value, and on a hybrid surface the model subscripts these payloads in `run_code`, so the same code
# would be right or wrong according to a flag. What is NOT added is a per-candidate batch mode: a
# loop over `preview_tcp_pose` inside `run_code` already returns one image per candidate.
# The relative NUMBERS are the reason the tool exists, not the shared canvas: two projected
# silhouettes can coincide in the image and be far apart in depth, so an image alone cannot answer
# "do these two collide". `separation_m` answers it in meters by a separating-axis test between the
# candidates' finger bounding boxes, and reports the direction of the guarantee: a positive value is
# a proven gap the true shapes can only exceed, a non-positive one is a box overlap the true shapes
# may not have. It is null between two poses of one arm, which are alternatives that never coexist.
# Also in 7.1.0, `preview_tcp_pose` and `compare_tcp_poses` declare the parallel-jaw
# precondition they always had (grasp centre = inner-tip midpoint; opening = one scalar gap) instead
# of failing incidentally inside geometry extraction. On this embodiment the check never fires, so
# no delivered description changes. Layout in the new overlay is SELF-aware only — it avoids its own
# drawn pixels and the image border. Scene-aware placement would require detecting objects, which is
# the capability this substrate withholds, so the limit is declared rather than treated as a gap.
# 7.2.0 — additive text only: `run_code`/`run_program` now state two return-contract facts the
# surface previously left to inference. (a) Internal tool PAYLOADS are not returned — `internal_trace`
# carries each internal call's name plus a compact status, so a value produced mid-block that the
# caller still needs must be assigned into `result` or it is lost to the model. Nothing in the old
# text said this, and a reader could reasonably assume the trace held the data; the same class of
# guess already cost 5 of 6 observed run_code shape errors. (b) `result` is read AFTER an ordinary
# code exception and the namespace is not reset, so filling it incrementally preserves whatever was
# set before a traceback — while a timeout or an ended attempt kills the sandbox child and returns
# `value=null`. (b) is what makes the safe idiom (`result = {}` then fill) discoverable instead of
# folklore.
# Also folded into 7.2.0 before any 7.2 batch ran: the code result gains an optional
# `images_withheld{count, obs_ids, note}`, emitted when code captured more images than the
# transport attached. Previously the caller received EVERY obs_id but only the last few images with
# nothing marking the difference, so a partial view read as a complete one — the history-eviction
# path has left a visible marker all along and this is its missing counterpart. It lives in the
# payload rather than in an extra text message because the MCP transport forwards only image refs,
# so a text-only note would reach the reference scaffold and vanish on the vendor agent. The note also states the recovery
# path (the observations stay addressable by obs_id), which is evidence, not strategy.
# No tool, argument, or existing field changes, so 7.1 and 7.2 results aggregate.
# 7.3.0 — additive text only: the schema-derived "Result keys" line now expands every top-level
# container ONE level, so a description says `annotation{..., grasp_center_xyz, ...}` and
# `achieved{..., failure_category, planner_status}` instead of the opaque `annotation` / `achieved`.
# Depth 1 alone was the layer carrying no information for exactly the tools that need it most:
# `preview_tcp_pose` declares 88 leaf fields behind six top-level names, and every motion
# outcome sits under `achieved`. A caller reading the old line still had to guess the subscript,
# which is what live runs failed at (5 of 6 observed run_code shape errors). Arrays of objects are
# marked `name[]{...}` so "index it, then take one of these" is explicit. Positions are deliberately
# NOT offered: a JSON object is name-keyed, and the model-visible projection sorts keys and drops
# the middle past its byte ceiling, so any positional assumption would break precisely when the
# result is largest. The line remains generated from the declared schema, so it cannot drift from
# the contract. +5.3k bytes measured. Nothing else changes, so 7.2 and 7.3 results aggregate.
# 7.4.0 — additive: the paired-motion `achieved.sync` block declares `failed_arm`, the side whose
# target was refused. `reach_both_tcp`'s workspace-bounds abort has always emitted it and the
# schema never listed it, so `additionalProperties: False` turned an ordinary out-of-bounds
# dual-arm target into `episode_fatal` / `tool_runtime_error` with origin=environment — the whole
# episode lost to a validator, not to the robot. It ended the 2026-08-07 lift_pot attempt at step
# 13. A paired command genuinely fails for ONE side, so the field is declared rather than dropped.
# No behaviour and no other field changes, so 7.3 and 7.4 results aggregate.
# 8.0.0 — BREAKING, and deliberately a controlled A/B: the two pose-preview tools are renamed and
# their opening sentence is rewritten from what they DRAW to what they ANSWER.
# `draw_grasp_footprint` -> `preview_tcp_pose`, `draw_pose_candidates` -> `compare_tcp_poses`.
# Behaviour, arguments, payload and rendered image are byte-identical; only the name and the first
# sentence move, which is what makes this a clean single-variable experiment.
# The motivation is measured, not aesthetic. Across the 2026-08-07 batch (12 attempts, 4 tasks) the
# model called `check_tcp_pose_reachability` 33 times and these two tools ZERO times, then collided
# in 12/12 attempts with 89% of its motion delivering nothing. It validated every pose against the
# one oracle that structurally cannot see what it was about to hit — benchmark episodes run with
# `curobo_world_model: none`, so a planner "solvable" says nothing about whether the target sits
# inside the table — while these tools, which project the real finger collision geometry onto the
# scene pixels, were never tried. Two candidate causes for that: the names described the MECHANISM
# (`draw`, `footprint`) rather than the question, and the delivered first sentence led with what
# gets rendered while the sentence "see where an uncommitted pose would land, before it moves"
# appeared nowhere on the surface.
# `check_*` was rejected as a prefix: those tools return a planner verdict and these deliberately
# return none, so borrowing the prefix would invite exactly the over-reading their `scope` field
# works to prevent. `draw_marks` and `draw_camera_rays` keep the `draw_` prefix because they really
# are "render what I tell you"; the split is meaningful rather than inconsistent.
# What is NOT done, and must not be: telling the model to be careful, to understand the 3D space,
# or to check a pose before moving. That is procedure — §0.1 corollary 4 and Reflection P6 — and it
# would delete the very ability P3 measures. Collision-ends-the-attempt and the three scale sources
# are already delivered verbatim in the system prompt, so no information is being added here; only
# the tool's own purpose, which is semantics, is now stated where it was previously absent.
# Tool NAMES change, so 7.x and 8.0 never aggregate for these two tools.
# 9.0.0 — BREAKING surface reduction. `open_gripper` and `close_gripper` were exact behavioural
# aliases because both accepted the full normalized 0..1 drive range and the simulator normalizes
# both action labels to the same gripper command. They become one required-`pos`
# `set_gripper(arm, pos)` tool, preserving continuous control without two misleading defaults.
# `check_direction_feasibility` is removed: it batch-expanded a direction into absolute targets
# for the same planner family as `check_tcp_pose_reachability`, yet explicitly could not predict
# the segmented straight execution used by `move_delta`; it saw 0 calls in tool surface 8.0.0 and 2 in 305
# local transcripts. Three non-public compatibility methods (`probe_contact_z`,
# `horizontal_plane_intersect`, `table_plane_intersect`) and two registry-only names with no schema
# or implementation (`draw_camera_rays`, `draw_reachable_ladder`) are deleted as dead surface.
# The paired non-contact geometry, candidate comparison, and synchronized dual-arm primitives stay:
# recent 8.0.0 use plus earlier live runs show distinct evidence or synchronization semantics.
# The per-physics-step contact monitor still interrupts the current open-loop motion, but no longer
# declares the whole attempt failed: direct calls return recoverable ABORTED/UNEXPECTED_CONTACT;
# run_code stops that code block, retains the same ActionResult, and yields control to the next
# agent turn. This reverses tool surface 6.0's tested-unit policy while preserving its immediate motion stop.
# Contact/task quality remains for the agent and out-of-band verifier, so 8.x and 9.0 never aggregate.
# 10.0.0 — BREAKING aim_camera input safety envelope for the current aloha-agilex embodiment.
# `pitches` now accepts 1..4 finite candidates in the inclusive 60..90 degree range in both the
# schema and runtime. Each candidate still changes the whole arm configuration and can encounter
# contact; the downward-facing range is an input bound, not a collision-free-motion guarantee.
# 10.1.0 — clarify aim_camera failure attribution without adding a second visibility boolean.
# The public `in_view` field is explicitly the calibrated-projection fact
# `projected_point_in_frame`; failed calls now distinguish `UNREACHABLE` (no candidate plan) from
# `NO_FRAMING_POSE` (a reachable candidate still projects the point outside the image).
# 11.0.0 — BREAKING aim_camera causal-contract correction. Candidate pitches are now generated and
# planner-queried from one unchanged state without execution; only one selected in-frame candidate
# can produce a physical motion. Results expose every generated TCP/EE pose, predicted projection,
# planner answer, deterministic selection rule, exact 0/1 physical execution, and final measured
# projection. No-execution paths no longer advance the physical observation tick; fractional
# cdist_ok_px values are preserved. A contact interruption retains the exact selected command in
# action_context. 10.x and 11.x results must not aggregate.
# 12.0.0 — BREAKING atomicity correction. The composite `aim_camera` tool is replaced by pure
# `camera_aim_pose(camera,target_xyz,pitch,standoff)`, which computes exactly one caller-selected
# TCP pose and predicted projection. It performs no candidate enumeration, policy selection,
# planner query, robot execution, or final-state verdict. The agent explicitly composes the result
# with `check_tcp_pose_reachability`, `reach_tcp`, and post-motion observation/project calls. This
# keeps deterministic camera-mount compensation in the bridge while returning pitch/standoff
# choice, plan/execute composition, recovery, and sufficiency judgment to the tested agent.
# Also folded into 12.0.0 before release qualification: preview_tcp_pose now rejects stale RGB,
# binds caller-selected finger gaps to the active embodiment, preserves every required nullable
# reference field, and reports image/robot/gap sampling ticks in its required annotation.
# 11.x and 12.x results must not aggregate.
# 12.1.0 — retire the non-atomic get_task control API. The two D0 descriptions that identify where
# contact and leg-limit parameters are declared now point to the proactive initial episode
# configuration. Their inputs, results, implementation, and verdict semantics are unchanged; the
# definition digest moves because agent-visible prose is part of the tested surface.
# 13.0.0 — BREAKING motion-API contract repair from the 2026-08-08 audit. Every mutating tool now
# advances the episode tick only on real execution and reports call-local physics steps plus partial
# side effects. Host-only traces preserve internal stages without expanding the Agent result.
# `move_delta` exposes waypoint_legs versus a true one-call single_plan; `reach_tcp` declares its
# full-plan/chunked-waypoint/correction mechanics; paired failures expose per-arm planner
# availability; contact
# probing distinguishes requested budget, nominal waypoint spacing and first observed contact; the
# reachability query returns native diagnostics and its exact collision-world scope. The
# `move_delta.path` tokens intentionally change from straight/free to waypoint_legs/single_plan;
# targets and strategy ownership are otherwise unchanged. 12.x and 13.x results MUST NOT aggregate.
# 13.0.1 — fairness clarification discovered by the final tool surface 13 audit. The initial episode
# configuration now declares the exact world-frame workspace envelope used to reject absolute TCP
# queries before execution. Motion descriptions also make native waypoint planner diagnostics
# explicitly conditional. Unexpected-contact conversions now leave one ABORTED host motion trace under the
# original action id. No target selection, planning discipline, or task semantics changed.
# 14.0.0 — BREAKING orientation/embodiment contract repair. grasp_quat_candidates no longer
# recommends quaternion-distance candidate selection or labels one candidate primary. Caller axes
# must be orthogonal after normalization within the declared 1e-6 tolerance; larger discrepancies
# are rejected, while accepted numerical residual correction is disclosed in input_axes. The
# EmbodimentCard now has a closed nested result schema, treats finger-gap subtraction as raw
# evidence rather than object width, and distinguishes qualitative camera-mount semantics from the
# live numeric Observation extrinsic. 13.x and 14.x results MUST NOT aggregate.
# 15.0.0 — BREAKING camera-calibration contract repair. Every successful Observation now carries
# an explicit fresh/available snapshot plus the complete OpenCV projection, pixel, image-size and
# no-distortion convention. get_camera_info no longer swallows a refresh failure or labels stale E
# current: legal-camera sensor failures return explicit nullable fields and read_failures without
# ending the episode. Camera size is read from RGB rather than the optional depth-backed helper,
# and capture_wrist's omitted views default is machine-declared. 14.x and 15.x results MUST NOT
# aggregate.
# 16.0.0 — BREAKING world-frame overlay repair. get_world_frame no longer projects a fixed,
# known-length world triad into the same calibrated Observation: that combination was a synthetic
# metric fiducial even though its anchor coordinate was omitted. It now draws a non-metric
# screen-space glyph independent of K/E and returns no world anchor, depth, or physical marker
# length. The required, closed annotation separately reports origin, endpoint, label, and complete
# visibility and explicitly denies world-to-pixel correspondence. 15.x and 16.x results MUST NOT
# aggregate.
# 17.0.0 — BREAKING robot self-state/contact-read repair. Lightweight arm-pose and gripper-state
# reads now return stable source-specific read_failures for explicit nulls. get_grasp_contact has
# a self-contained available/unavailable envelope with N*s, arm and tick, so backend failure can
# no longer masquerade as measured zero contact. The shared contact reader also stops actions
# recoverably when contact state is unavailable instead of disabling the safety boundary. No
# object identity, grasp verdict, action selection, or recovery policy was added. 16.x and 17.x
# results MUST NOT aggregate.
# 18.0.0 — BREAKING Estimate evidence-contract repair. The five Estimate-producing tools now use
# producer-specific closed provenance schemas with required inputs, method, and uncertainty scope.
# project/ray zero uncertainty covers only the deterministic calibrated transform; it does not
# erase caller annotation/world-point or camera-calibration error. ray also reports image_size_hw
# and pixel_in_frame while retaining mathematically valid out-of-frame extrapolation. No depth,
# object validation, target selection, or action policy was added. 17.x and 18.x results MUST NOT
# aggregate.
# 19.0.0 — BREAKING information-density repair. The authored description is now the single
# model-facing semantic explanation; compact catalog purposes are no longer prepended, and return
# summaries no longer restate those semantics. Closed result keys remain generated from the same
# schemas. Camera conventions and Estimate provenance use shorter self-describing fixed values;
# fields, computations, failure meanings, and control behavior are unchanged. 18.x and 19.x
# results MUST NOT aggregate.
# 20.0.0 — BREAKING plane-intersection uncertainty repair. plane_intersect exposes caller-owned
# pixel_sigma_px with a declared default and echoes it in closed provenance. Pixel uncertainty now
# uses the complete local u/v ray-plane Jacobian instead of an fx-only scalar approximation. Plane
# selection, surface truth, and usability remain caller decisions. 19.x and 20.x results MUST NOT
# aggregate.
# 21.0.0 — BREAKING triangulation evidence repair. Pixel uncertainty is caller-owned; numerical
# validity is explicitly separate from required motion/baseline/contact facts and from the fixed
# fact that the harness did not verify correspondence. No matcher, quality gate, or candidate
# selection was added. 20.x and 21.x results MUST NOT aggregate.
# 22.0.0 — BREAKING metric-scale repair. Object-size depth now binds a caller extent prior to an
# explicit projected bbox axis and uses caller uncertainty (or required unknown/null); the visible
# gripper ruler rejects a finite projected span at or below its declared numerical epsilon. No
# object prior, bbox, target, or camera action is selected by the harness.
# 23.0.0 — BREAKING draw-marks contract repair. The existing 1..12 legibility bound now constrains
# both input arrays and output marks, and annotation is required on every successful result. Pixel
# selection, labels, out-of-frame acceptance, and rendering behavior are unchanged.
# 24.0.0 — BREAKING compare-pose truthfulness repair. Conservative terminal finger-box overlap is
# named as such, current-tick robot geometry sampling is explicit, and candidate gap bounds come
# from the active embodiment. Comparison remains pure and does not rank, plan, or execute.
# 25.0.0 — BREAKING composition/files contract repair. Effective composition limits and hard-stop
# reset effects are delivered before the first call; virtual-file failures have stable subtypes;
# read_file uses bounded UTF-8 byte chunks that survive model-result projection and can reconstruct
# every legal file. Direct calls, code blocks, and saved programs remain separate atomic choices.
# 26.0.0 — BREAKING motion/camera-description density repair. Common ActionResult/contact facts are
# delivered once in the initial instruction contract instead of being copied across motion tools
# and return summaries. Each tool keeps its unique success/failure/partial/fallback semantics;
# schemas, computations, thresholds, side effects, and result fields are unchanged.
# 26.1.0 — additive evidence capture selector. `capture_evidence_views.views` accepts a non-empty
# unique subset of overview/left_wrist/right_wrist while omission preserves the three-view result.
# The return type and same-tick contract are unchanged; the delivered input schema has changed.
# 27.0.0 — BREAKING boundary-snapshot density repair. `observed_before/observed_after` and every
# decoded `orientation` block stop carrying two things the caller never acts on. (a) `joint_state`
# leaves the snapshot entirely; `get_robot_state` remains its single source and is unchanged. It was
# ~73% of every snapshot while being wrong twice over: on a single-URDF dual-arm embodiment
# `robot.left_entity is robot.right_entity`, so BOTH arms reported the identical whole-robot
# 38-joint vector (mobile base and unused leader chains included, 22 of 38 never moving), and no
# joint limit or link geometry is published anywhere, so the numbers could not be interpreted.
# Measured across two run corpora: 0 reads of `joint_state` from a snapshot, versus `status` 342,
# `resulting_pose` 46 — and `resulting_pose.tcp` is already the same value as
# `observed_after.arms[arm].tcp_pose` (121/121 identical). (b) The fixed 330-byte `convention`
# prose leaves every orientation block for `codeaction.motion.orientation.POSE_CONVENTION`, stated once in
# `get_embodiment().orientation` and the common ActionResult note; only `preview_tcp_pose`'s single
# `orientation_convention` field still cites it. Decoded axes, poses, gripper scalars, contact,
# frame/units and read failures are unchanged, and the common note now gives the access path.
# Per-motion-result cost falls 22,367 -> 6,464 bytes. This removes required wire fields, so no 26.x
# result aggregates with 27.0.
# 27.1.0 — additive historical-image replay. Existing `load_image(obs_id)` calls inside run_code or
# run_program now include that observation in the next model-visible image projection. No new public
# tool or robot capability is added; repeated references still send one image by last occurrence.
# 28.0.0 — BREAKING reachability-density repair: 27.0's rule applied to the most frequently called
# tool. `check_tcp_pose_reachability` stops returning `collision_world` and `note`. What the
# planner's world contains is EPISODE CONFIGURATION — identical for every query of every episode —
# so it is declared once as `planner_collision_world` beside the other harness parameters instead
# of 106 bytes per call; and the two fixed notes only restated `stage`, which is a closed enum
# (`planner_query` / `workspace_prefilter` / `planner_exception`) whose meanings the tool
# description now defines, with `reason` naming the violated bound or the planner error. Measured
# on the 26.x corpus: 97/97 calls carried byte-identical copies of both, while agent code read
# `planner_found_trajectory` 58 times and `collision_world` 0 times. Per-call payload 617 -> 177
# bytes. No planner behaviour, query scope, decision field, or refusal diagnostic changes. This
# removes required wire fields, so no 27.x result aggregates with 28.0.
# 29.0.0 — recoverable contact results add bounded settle evidence and a standard head
# Observation; set_gripper joins the complete physics-action contact policy.
# 30.0.0 — a code block stopped by a recoverable safety abort now reports that abort in the
# action vocabulary at its top level, which extends the same-turn cancellation barrier to
# run_code/run_program on both tracks.
# 31.0.0 — one gripper's two fingers touching each other is no longer a contact for the
# rule. Measured over the shadow campaign: all nine above-threshold events of that shape
# came from behaviour that must not be interrupted, none from a harmful one.
# 32.0.0 — a contact stop now reports which of the robot's own parts stopped, its own
# pose at the detecting step, and a plain-language note on what that means next. The
# contacted entity remains host-only.
# 33.0.0 — a two-leg stall now reports whether robot contact coexisted with the stop,
# which of the robot's own parts were then in contact, and its own boundary pose. The
# contacted entity remains host-only, and the report makes no causal claim.
# 34.0.0 — contact identity no longer permits or interrupts motion. Waypoint stalls are the
# single contact-related stop and carry the three factual fields introduced in 33.0.0. Any
# atomic ABORTED result, regardless of cause, retains the same-turn composition barrier.
# 35.0.0 — the same measured-progress stop covers dense single- and dual-arm transports. Planner
# completion no longer reports SUCCESS when the commanded TCP target remains materially unmet.
# 36.0.0 — dense progress means reduction of the commanded target error, not TCP path length.
# This prevents collision-induced jitter or a closed loop from being mistaken for progress.
# 37.0.0 — a completed primary transport that leaves the target outstanding and cannot execute a
# correction is a measured execution stall, not a planner-success result or a contact classifier.
# 38.0.0 — camera_aim_pose now points the wrist CAMERA at the target rather than the TCP. Its
# centring step subtracted the residual from the target instead of from the pose under test, which
# is a reflection: an even iteration count returned the pose unchanged, and the shipped count is
# two. The returned pose was therefore the target's own xy at the requested standoff, so the
# camera sat at its mount offset from there and its optical axis met the target plane 7 to 30 cm
# away, depending on pitch. Callers that used the pose to place the gripper got what they asked
# for either way; callers that moved there to look got the target outside the frame about
# seven times in ten.
TOOL_SET_VERSION = "38.0.0"
TOOL_SET_MEMBERS = tuple(
    name for name in D0_TOOLS if name != "done" and name in TOOL_SPECS
)
INTERFACE_REFERENCE = "reference-mcp"
INTERFACE_REFERENCE_CODE_FIRST = "reference-code-first"
INTERFACE_VENDOR_DIRECT = "vendor-mcp-direct"
INTERFACE_VENDOR_GATEWAY = "vendor-mcp-gateway"

# The code-first arm of the interface-shape comparison. The model is handed the composition tools plus one same-tick
# three-camera evidence bundle, and reaches the remaining primitives as a documented Python library
# inside run_code. The sandbox now returns up to six images per block, so one
# capture_evidence_views call (three images) no longer exceeds its evidence channel.
CODE_FIRST_DELIVERED_PRIMITIVES = ("capture_evidence_views",)

# Both reference profiles run the reference-scaffold scaffold and its control contract; they differ only in
# which primitives are delivered as model-facing schemas. Every consumer should test membership
# here rather than equality against one id.
REFERENCE_INTERFACE_PROFILES = (INTERFACE_REFERENCE, INTERFACE_REFERENCE_CODE_FIRST)

DEFAULT_ALWAYS_LOAD_TOOLS = (
    "run_code", "capture_motion_pair",
    "triangulate_correspondence", "scale_from_gripper", "scale_from_object_size",
    "grasp_quat_candidates", "check_tcp_pose_reachability", "preview_tcp_pose",
    "probe_contact_along", "get_grasp_contact",
)

class ToolSurfaceError(ValueError):
    pass


@dataclass(frozen=True)
class InterfaceProfile:
    id: str
    transport: str
    task_delivery: str
    submittable: bool
    transforms: tuple[dict, ...]
    extras: tuple[str, ...] = ()
    # Which D0 primitives are delivered as model-facing tool schemas. None = all of them (the
    # default surface). A subset means the rest reach the model only as a documented Python
    # library inside run_code — the CAPABILITY set is unchanged, so `base_tool_set_sha256` is
    # unchanged too and only `delivered_sha256` moves. That difference is the whole point: the
    # interface SHAPE becomes a declared benchmark variable instead of an unexamined constant.
    delivered_primitives: Optional[tuple[str, ...]] = None
    # True when the profile is meaningless without run_code (the primitives live behind it).
    requires_hybrid: bool = False
    # Retained only for an explicit non-release experiment. Release/default profiles expose
    # run_code as their sole composition tool.
    delivered_program_tools: tuple[str, ...] = ()


INTERFACE_PROFILES = {
    INTERFACE_REFERENCE: InterfaceProfile(
        id=INTERFACE_REFERENCE,
        transport="openai-tools",
        task_delivery="initial_prompt",
        submittable=True,
        transforms=(
            {"id": "hybrid-composition-tools", "version": "2.0.0",
             "applied_by": "reference_agent"},
            {"id": "done-control-tool", "version": "1.0.0",
             "applied_by": "reference_agent"},
        ),
    ),
    INTERFACE_REFERENCE_CODE_FIRST: InterfaceProfile(
        id=INTERFACE_REFERENCE_CODE_FIRST,
        transport="openai-tools",
        task_delivery="initial_prompt",
        submittable=False,          # experimental arm; not a leaderboard surface yet
        transforms=(
            {"id": "hybrid-composition-tools", "version": "2.0.0",
             "applied_by": "reference_agent"},
            {"id": "code-first-primitive-library", "version": "1.1.0",
             "applied_by": "reference_agent"},
            {"id": "done-control-tool", "version": "1.0.0",
             "applied_by": "reference_agent"},
        ),
        delivered_primitives=CODE_FIRST_DELIVERED_PRIMITIVES,
        requires_hybrid=True,
        delivered_program_tools=PROGRAM_TOOL_NAMES,
    ),
    INTERFACE_VENDOR_DIRECT: InterfaceProfile(
        id=INTERFACE_VENDOR_DIRECT,
        transport="mcp",
        task_delivery="initial_prompt",
        submittable=True,
        transforms=(
            {"id": "hybrid-composition-tools", "version": "2.0.0",
             "applied_by": "episode_server"},
            {"id": "anthropic-always-load-meta", "version": "1.0.0",
             "applied_by": "episode_server"},
            {"id": "done-control-tool", "version": "1.0.0",
             "applied_by": "episode_server"},
        ),
    ),
    INTERFACE_VENDOR_GATEWAY: InterfaceProfile(
        id=INTERFACE_VENDOR_GATEWAY,
        transport="mcp",
        task_delivery="initial_prompt",
        submittable=False,
        transforms=(
            {"id": "hybrid-composition-tools", "version": "2.0.0",
             "applied_by": "episode_server"},
            {"id": "anthropic-always-load-meta", "version": "1.0.0",
             "applied_by": "episode_server"},
            {"id": "done-control-tool", "version": "1.0.0",
             "applied_by": "episode_server"},
            {"id": "bash-exec-extra", "version": "1.0.0",
             "applied_by": "gateway"},
        ),
        extras=("bash_exec",),
    ),
}


def _schema_definition(name: str) -> dict:
    try:
        description, input_schema = TOOL_SPECS[name]
    except KeyError as exc:
        raise ToolSurfaceError(f"unknown tool definition {name!r}") from exc
    return {"name": name, "description": description, "inputSchema": input_schema}


def base_payload(members: Iterable[str] = TOOL_SET_MEMBERS) -> dict:
    out = {}
    for name in members:
        definition = _schema_definition(str(name))
        out[definition["name"]] = {
            "description": definition["description"],
            "inputSchema": definition["inputSchema"],
            "outputSchema": get_output_schema(definition["name"]),
        }
    return out


BASE_TOOL_SET_SHA256 = sha256_json(base_payload())
PINNED_TOOL_SET = {
    "id": TOOL_SET_ID,
    "version": TOOL_SET_VERSION,
    "sha256": BASE_TOOL_SET_SHA256,
}


def resolve_tool_set(reference: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(reference, Mapping):
        raise ToolSurfaceError("tool_surface.tool_set must be an object")
    expected = PINNED_TOOL_SET
    actual = {key: reference.get(key) for key in ("id", "version", "sha256")}
    if actual != expected:
        raise ToolSurfaceError(
            f"unknown or stale tool set: expected {expected}, got {actual}")
    return TOOL_SET_MEMBERS


def resolve_interface_profile(profile_id: str) -> InterfaceProfile:
    try:
        return INTERFACE_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ToolSurfaceError(f"unknown interface profile {profile_id!r}") from exc


def tool_set_drift(reference) -> dict | None:
    """None when the card's pinned tool set matches the live code; else both sides.

    The pin is a CROSS-CHECK, not the identity: the comparison identity always records the
    surface computed from the code that actually ran, so a modified tool set self-declares
    through its hash. Drift therefore only decides submission eligibility, never runnability.
    """
    actual = ({key: reference.get(key) for key in ("id", "version", "sha256")}
              if isinstance(reference, Mapping) else None)
    if actual == PINNED_TOOL_SET:
        return None
    return {"card": actual, "code": dict(PINNED_TOOL_SET)}


def validate_card_tool_surface(card_surface: Mapping[str, Any], *,
                               strict_pins: bool = True) -> dict:
    if not isinstance(card_surface, Mapping):
        raise ToolSurfaceError("tool_surface must be an object")
    allowed = {"tool_set", "hybrid", "web_access", "interface_profiles"}
    unknown = sorted(set(card_surface) - allowed)
    if unknown:
        raise ToolSurfaceError(f"tool_surface contains unknown keys: {unknown}")
    drift = tool_set_drift(card_surface.get("tool_set"))
    if drift is not None and strict_pins:
        resolve_tool_set(card_surface.get("tool_set"))   # raises the canonical message
    if not isinstance(card_surface.get("hybrid"), bool):
        raise ToolSurfaceError("tool_surface.hybrid must be boolean")
    if card_surface.get("web_access") is not False:
        raise ToolSurfaceError("tool_surface.web_access must be false")
    profiles = card_surface.get("interface_profiles")
    expected_profiles = list(INTERFACE_PROFILES)
    if profiles != expected_profiles:
        raise ToolSurfaceError(
            f"tool_surface.interface_profiles must equal {expected_profiles}")
    report = {
        "tool_set": dict(PINNED_TOOL_SET),
        "hybrid": card_surface["hybrid"],
        "web_access": False,
        "interface_profiles": expected_profiles,
    }
    if drift is not None:
        report["tool_set_drift"] = drift
    return report


def delivered_primitive_names(profile_id: str) -> list[str]:
    """Primitives this profile delivers as model-facing tool schemas (a subset for code-first)."""
    profile = resolve_interface_profile(profile_id)
    if profile.delivered_primitives is None:
        return list(TOOL_SET_MEMBERS)
    unknown = [n for n in profile.delivered_primitives if n not in TOOL_SET_MEMBERS]
    if unknown:
        raise ToolSurfaceError(
            f"{profile_id} delivers primitives outside the pinned tool set: {unknown}")
    # keep the pinned registry order so delivery order never depends on how the subset was typed
    return [name for name in TOOL_SET_MEMBERS if name in set(profile.delivered_primitives)]


def ordered_tool_names(profile_id: str, *, hybrid: bool) -> list[str]:
    profile = resolve_interface_profile(profile_id)
    if profile.requires_hybrid and not hybrid:
        raise ToolSurfaceError(
            f"{profile_id} delivers its primitives through run_code and requires hybrid=True")
    composition = (["run_code"] + list(profile.delivered_program_tools)) if hybrid else []
    names = delivered_primitive_names(profile_id)
    names = composition + names + ["done"]
    names.extend(profile.extras)
    return names


def _mcp_definitions(names: Iterable[str], *, apply_always_load: bool) -> list[dict]:
    names = list(names)
    eager = set(DEFAULT_ALWAYS_LOAD_TOOLS) if apply_always_load else set()
    out = []
    for name in names:
        if name == "bash_exec":
            out.append(dict(BASH_EXEC_DEFINITION))
            continue
        definition = _schema_definition(name)
        if name in eager:
            definition["_meta"] = {"anthropic/alwaysLoad": True}
        out.append(definition)
    return out


def delivered_definitions(profile_id: str, *, hybrid: bool) -> list[dict]:
    profile = resolve_interface_profile(profile_id)
    names = ordered_tool_names(profile_id, hybrid=hybrid)
    if profile.transport == "openai-tools":
        return build_openai_tools(names)
    return _mcp_definitions(names, apply_always_load=True)


def mcp_server_definitions(profile_id: str, *, hybrid: bool) -> list[dict]:
    """Definitions emitted by the episode server before any agent-side transform.

    The reference container receives MCP schema objects and converts them to the native
    OpenAI-compatible function shape recorded by ``delivered_definitions``. Vendor profiles are
    already model-facing MCP definitions; only gateway-owned extras are absent at the server.
    """
    profile = resolve_interface_profile(profile_id)
    names = ordered_tool_names(profile_id, hybrid=hybrid)
    if profile_id == INTERFACE_REFERENCE:
        return _mcp_definitions(names, apply_always_load=False)
    server_names = [name for name in names if name not in profile.extras]
    return _mcp_definitions(server_names, apply_always_load=True)


def surface_identity(profile_id: str, *, hybrid: bool,
                     runtime_registry_names: Optional[Iterable[str]] = None) -> dict:
    profile = resolve_interface_profile(profile_id)
    if runtime_registry_names is not None:
        runtime = set(runtime_registry_names)
        # Two separate obligations, and conflating them broke the code-first arm: every DELIVERED
        # primitive must be backed by the runtime, and the runtime must expose nothing outside the
        # PINNED set. On the default surface the two sets coincide, so this is unchanged there; on
        # code-first the agent's registry holds only what it was handed while the episode host
        # still backs all thirty, and both are correct.
        missing = sorted(set(delivered_primitive_names(profile_id)) - runtime)
        undeclared = sorted((runtime & set(TOOL_SPECS)) - set(TOOL_SET_MEMBERS))
        if missing or undeclared:
            raise ToolSurfaceError(
                f"runtime registry drift: missing={missing}, undeclared={undeclared}")
    definitions = delivered_definitions(profile_id, hybrid=hybrid)
    return {
        "tool_set": dict(PINNED_TOOL_SET),
        "base_tool_set_sha256": sha256_json(base_payload()),
        "delivered_sha256": sha256_json(definitions),
        "ordered_names": ordered_tool_names(profile_id, hybrid=hybrid),
        "task_delivery": profile.task_delivery,
        "transforms": [dict(item) for item in profile.transforms],
        "extras": list(profile.extras),
        "interface_profile": profile.id,
        "submittable": profile.submittable,
    }


def assert_surface_preflight(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    keys = (
        "tool_set", "base_tool_set_sha256", "delivered_sha256", "ordered_names",
        "task_delivery", "transforms", "extras", "interface_profile",
    )
    mismatch = [key for key in keys if expected.get(key) != observed.get(key)]
    if mismatch:
        raise ToolSurfaceError(f"tool surface preflight mismatch: {mismatch}")


def surface_is_submittable(interface_profile: str) -> bool:
    """Whether attempts on this interface may enter a benchmark submission."""
    return resolve_interface_profile(interface_profile).submittable
