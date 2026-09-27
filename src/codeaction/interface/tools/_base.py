"""Module surface and the private substrate every tool family stands on.

Underscored because it is NOT a peer of the capability modules beside it: they are four
orthogonal tool families, this is the layer beneath all four.
"""
"""Sim-touching D0 tool layer (spec §6; leak fixes §7; governing invariant §0.1).

The ToolBox holds env/vp by CLOSURE — the agent never receives an env handle. Every perception/
geometry/scale tool returns a typed object (types.py); motion/gripper tools return ActionResult with
raw proprioception snapshots before and after execution; info tools return plain read-only
dicts. No object GT anywhere: no scene poses, no scene lists, no rendered depth; grasp contact is
finger↔anything, UNFILTERED by object identity (leak fix #1). Guards may only reject / clamp /
abort / report — they never modify the goal or choose the next action.

Contact semantics: contact identity never decides whether an action is allowed. Transport motions
stop only when their measured progress meets the shared stall rule. At that boundary they report
``blocked_in_contact``, ``contact_parts`` and ``pose_at_contact``; an unavailable read stays null.
Finger contact tools keep their own requested measurement stop.

`move_delta(path="waypoint_legs")` uses target-anchored constrained waypoint legs
(≤ contact_leg_m each, cuRobo constraint_pose=[1,1,1,0,0,0]); each leg re-aims from the latest
measured TCP toward the fixed endpoint, and its planner-selected trajectory can bow laterally.
`path="single_plan"` makes exactly one unconstrained-path planner call to the same endpoint.
`reach_tcp` attempts a full free plan, falls
back to chunked waypoint plans after refusal, and after completed transport may make at most two
orientation-holding correction plans. None promises a Cartesian line. Its public status scopes the
primary transport. The final pose can differ from the
command. A planner-complete transport that cannot make measured progress to its still-outstanding
target returns ``ABORTED/stalled``; smaller delivered-vs-commanded differences remain raw for the
caller to judge.

Production construction runs only in the sim runtime because primitives/, servo/, and probes/ resolve
there. Local tests import this module with isolated fakes for contract and accounting behavior; live
planner/physics integration still requires the remote gate and smoke."""
from dataclasses import replace
from functools import wraps
from pathlib import Path
import os
import time

import numpy as np

from codeaction.contracts.types import Observation, ObservationSet, ObservationPair, Estimate, ActionResult
from codeaction.runtime.guards import clamp_step, SequentialArmLock
from codeaction.contracts import world_frame as _wf
from codeaction.contracts import embodiment as _emb
from codeaction.backends.robotwin import aim as _aim
from codeaction.backends.robotwin import scale as _scale
from codeaction.motion import orientation as _orientation
from codeaction.motion import planner_status as _planner_status
from codeaction.motion import motion_validation as _mv
from codeaction.interface.registry import D0_TOOLS
from codeaction.runtime.argcheck import checked_registry
from codeaction.contracts import harness_parameters as _harness
from codeaction.interface.schemas import to_json
from codeaction.contracts.result_contracts import (CAMERA_CONVENTION_VALUE, DEFAULT_PIXEL_SIGMA_PX,
                                      DRAW_MARKS_MAX_COUNT, DRAW_MARKS_MIN_COUNT,
                                      RULER_MIN_PROJECTED_SPAN_PX)
from codeaction.motion.motion_validation import (REACH_CORRECTION_TRIGGER_M,
                                       displacement_tolerance_m)
from codeaction.motion.motion_report import ExecutionEvidence, execution_block
from codeaction.motion.motion_progress import MotionProgressGuard, MotionProgressInterrupted
from codeaction.motion.motion_trace import MotionTrace
from codeaction.motion.collision_monitor import CollisionMonitor, read_scene_contacts
from codeaction.contracts.failures import ContactReadUnavailable, PhysicalTimeBudgetExhausted

VALID_CAMERAS = ("head_camera", "left_camera", "right_camera")
_MAX_POSE_CANDIDATES = 4  # legibility bound for compare_tcp_poses
_CONTACT_IMPULSE_EPS = _harness.CONTACT_IMPULSE_THRESHOLD

# Wong colour-blind-safe hues, one per candidate index, chosen to stay distinguishable from the
# single-candidate overlay's own palette (magenta fingers / cyan approach / yellow closing).
_CANDIDATE_COLORS = ((86, 180, 233), (230, 159, 0), (0, 158, 115), (204, 121, 167))
_BOX_OVERLAP_COLOR = (240, 70, 70)

# What the planner checks is EPISODE CONFIGURATION, not a per-query measurement: it is identical
# for every call of every episode, so it is declared once with the other harness parameters
# (`codeaction.contracts.harness_parameters.planner_collision_world`) and no longer copied into each result.

# Stable display colours shared by the non-metric world-frame legend and pose previews.  Colour
# identifies an axis; it does not define a world point, depth, or physical length.
_WF_AXIS_COLORS = {"x": (225, 60, 60), "y": (60, 180, 75), "z": (70, 110, 235)}


ORIENTATION_ANCHOR_ENV = "CODEACTION_ORIENTATION_ANCHOR"

def _observed_action(fn):
    """Attach one canonical before/after robot snapshot to every ActionResult path."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        observed_before = self._robot_snapshot()
        result = fn(self, *args, **kwargs)
        return self._attach_action_observations(result, observed_before)
    return wrapped


def _fingerprint_changed(before, after, tol=1e-9):
    """Whether two robot-state fingerprints differ. None when neither read produced a number."""
    if before is None or after is None:
        return None
    comparable = False
    for entry_before, entry_after in zip(before, after):
        if entry_before[0] != entry_after[0]:
            return True
        for left, right in zip(entry_before[1:], entry_after[1:]):
            if left is None or right is None:
                # A read that appeared or vanished between the two samples is itself a change.
                if (left is None) != (right is None):
                    return True
                continue
            values_left = ([float(left)] if isinstance(left, (int, float))
                           else [float(v) for v in left])
            values_right = ([float(right)] if isinstance(right, (int, float))
                            else [float(v) for v in right])
            if len(values_left) != len(values_right):
                return True
            comparable = True
            if any(abs(a - b) > tol for a, b in zip(values_left, values_right)):
                return True
    return False if comparable else None


def orientation_anchor_enabled():
    """Whether every straight leg holds the CALL-START orientation (shipped default: yes).

    Exposed as a declared harness parameter so the anchor's task-level effect can be measured
    without editing production, the way `FRAME_RETENTION` is a declared scaffold parameter. Off
    reproduces pre-4.2.0 semantics, where each leg held whatever the previous leg drifted to.
    """
    raw = os.environ.get(ORIENTATION_ANCHOR_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "off", "no"}



class ToolBoxBase:
    """Camera, observation, physics-stepping, pose/contact reads and drawing primitives.

    Also holds every class-level default the tool families read off `self`.
    """
    enforce_result_contracts = True
    _orientation_anchor = True
    _STALL_PROGRESS_FLOOR_M = _harness.STALL_PROGRESS_FLOOR_M
    _STALL_CONSECUTIVE_LEGS = _harness.STALL_CONSECUTIVE_LEGS
    _ECHO_DECIMALS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    _ECHO_MIN_AXES = 2
    def __init__(self, env, vp, out_dir, max_step=0.30, contact_leg_m=0.03):
        from codeaction.backends.robotwin.primitives import motion, gripper, perception, pose_utils
        from codeaction.backends.robotwin.primitives.result import SUCCESS as PRIM_OK
        from codeaction.backends.robotwin.orientation import axis_angle_quat
        from codeaction.backends.robotwin.perception import _invert_extrinsic
        self._env, self._vp = env, vp
        self._m, self._g, self._p, self._pu = motion, gripper, perception, pose_utils
        self._collision_monitor = CollisionMonitor(env, _CONTACT_IMPULSE_EPS)
        # Read the stopping pose from the SAME snapshot the boundaries use, once per
        # interruption rather than once per physics step.
        self._collision_monitor.attach_pose_reader(self._robot_snapshot)
        self._axisq, self._invert = axis_angle_quat, _invert_extrinsic
        self._OK = PRIM_OK
        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._max_step = float(max_step)
        self._leg = float(contact_leg_m)
        # Declared harness parameter, read ONCE per episode and reported in the transcript `meta`
        # record alongside frame_retention. Default is the shipped 4.2.0+ behaviour; turning it off
        # reproduces the pre-4.2.0 leg semantics so the anchor can be A/B'd without a code edit.
        # It changes no delivered description, so `base_tool_set_sha256` is unchanged and two runs
        # are distinguished at the RUN level by `orientation_anchor` in run_meta -- never silently.
        self._orientation_anchor = orientation_anchor_enabled()
        self._lock = SequentialArmLock()
        self._tick = 0                    # bumps ONLY on executed sim-mutating actions
        # Host-injected physics clock (StepObserver.step_count), set by attach_sim_step_source.
        # None means no clock in this process (unit fixtures, probes) -- the measured-state
        # fallback in `_state_fingerprint` then decides whether anything executed.
        self._sim_step_fn = None
        self._sim_state_fn = None
        self._episode_terminal_check = None
        self._obs = {}                    # obs_id -> Observation
        self._pairs = {}                  # pair_id -> ObservationPair
        self._n_obs = 0
        self._n_pair = 0
        self._n_ann = 0
        self._n_act = 0
        get_timestep = getattr(getattr(env, "scene", None), "get_timestep", None)
        sim_dt = float(get_timestep()) if callable(get_timestep) else None
        window_steps = (None if sim_dt is None or sim_dt <= 0 else
                        int(np.ceil(_harness.DENSE_STALL_WINDOW_S / sim_dt)))
        self._motion_progress = (
            None if window_steps is None else MotionProgressGuard(
                self._tcp,
                window_steps=window_steps,
                linear_floor_m=_harness.STALL_PROGRESS_FLOOR_M,
                angular_floor_rad=_harness.DENSE_STALL_ANGULAR_FLOOR_RAD,
                target_tolerance_m=REACH_CORRECTION_TRIGGER_M))
        self._motion_progress_step_wrapped = False

    # ── internals ─────────────────────────────────────────────────────────
    def _kec(self, camera):
        from codeaction.backends.robotwin.geometry import camera_matrices
        return camera_matrices(self._vp, camera)

    def _refresh_camera_strict(self):
        """Refresh RGB/calibration state and propagate failure to the codeaction read boundary."""
        self._env._update_render()
        self._env.cameras.update_picture()

    def _read_camera_matrices(self, camera):
        matrices = self._vp.backend.get_camera_matrices(camera)
        if matrices is None or len(matrices) < 2:
            raise RuntimeError("camera matrices unavailable")
        K = np.asarray(matrices[0], dtype=float)
        E = np.asarray(matrices[1], dtype=float)[:3]
        if K.shape != (3, 3) or E.shape != (3, 4):
            raise RuntimeError("camera matrix shape invalid")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(E)):
            raise RuntimeError("camera matrices contain non-finite values")
        return K, E

    def _read_rgb_size(self, camera):
        rgb = self._env.cameras.get_rgb()
        frame = np.asarray(rgb[camera]["rgb"])
        if frame.ndim < 2 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise RuntimeError("camera RGB size unavailable")
        return [int(frame.shape[0]), int(frame.shape[1])]

    @staticmethod
    def _camera_snapshot(K, E, size_hw):
        return {
            "available": True,
            "K": np.asarray(K).tolist(),
            "E": np.asarray(E).tolist(),
            "size_hw": [int(size_hw[0]), int(size_hw[1])],
            "freshness": "call_time_current",
            "read_failures": [],
            "convention": dict(CAMERA_CONVENTION_VALUE),
        }

    def _get_obs(self, obs_id):
        if obs_id not in self._obs:
            raise KeyError(f"unknown obs_id {obs_id!r} — pass an id returned by capture_*/draw tools")
        return self._obs[obs_id]

    def _get_pair(self, pair_id):
        if pair_id not in self._pairs:
            raise KeyError(f"unknown pair_id {pair_id!r} — pass an id returned by "
                           "capture_motion_pair")
        return self._pairs[pair_id]

    def image_path(self, obs_id):
        """Runner/sandbox-side lookup of an observation's image file (never exposed as a tool)."""
        return self._get_obs(obs_id).image_ref

    def _observe(self, side):
        """EVAL-SIDE observability (out-of-band, like the verifier): after every sim-mutating
        action, snapshot head + that arm's wrist camera to observer/tick_NNN_*.png. The model
        NEVER sees these — not registered in _obs, not referenced in any payload; when the model
        wants to see, it must call a capture_* tool (its perception cadence is measured behaviour)."""
        try:
            d = self._out / "observer"
            d.mkdir(exist_ok=True)
            self._env._update_render()
            self._env.cameras.update_picture()
            for cam in ("head_camera", f"{side}_camera"):
                # The action counter is in the name because a call that executed nothing no longer
                # advances the tick: two attempts can now share one tick, and the earlier frames
                # must not be overwritten by the later attempt.
                self._env.save_camera_rgb(
                    str(d / f"tick_{self._tick:03d}_{self._n_act:03d}_{cam}.png"), cam)
        except Exception:
            pass

    def _observe_both(self):
        self._observe("left")
        self._observe("right")

    def _fresh(self, o):
        if o.tick != self._tick:
            raise ValueError(f"stale observation {o.obs_id} (tick {o.tick} != now {self._tick}) — "
                             "the arm moved since this frame; call capture_* again")

    def _tcp(self, arm):
        d = self._p.get_gripper_pose(self._env, arm).get("data") or {}
        return d.get("tcp_pose")

    # ── execution evidence: did this call actually do anything ────────────
    def attach_sim_step_source(self, fn):
        """Episode hosts hand the ToolBox the same physics-step counter they give the recorder.

        `env.take_action_cnt` cannot serve here: planned trajectories execute through
        `take_dense_action`, which never increments it. `StepObserver` wraps `env.scene.step` and
        therefore counts every physics step of every motion path.
        """
        self._sim_step_fn = fn
        guard = getattr(self, "_motion_progress", None)
        scene = getattr(getattr(self, "_env", None), "scene", None)
        if (guard is not None and scene is not None
                and not getattr(self, "_motion_progress_step_wrapped", False)):
            original_step = scene.step

            def progress_observed_step():
                original_step()
                step = self._physics_step()
                if step is not None:
                    guard.observe(step)

            scene.step = progress_observed_step
            self._motion_progress_step_wrapped = True

    def attach_sim_state_source(self, fn):
        """Attach StepObserver state so recovery leaves the terminal physics step unused."""
        if not callable(fn):
            raise TypeError("simulation state source must be callable")
        self._sim_state_fn = fn

    def attach_episode_terminal_check(self, fn):
        """Attach the host's synchronous terminal latch check at the D0 registry boundary.

        Physical time is deliberately checked after a D0 tool returns, rather than thrown from
        inside an inherited primitive.  The tool is the public atomic operation: it may cross the
        threshold while completing, but the registry terminates the episode before another direct
        or ``run_code`` internal tool can begin.
        """
        if not callable(fn):
            raise TypeError("episode terminal check must be callable")
        self._episode_terminal_check = fn

    def _raise_if_episode_terminal(self):
        check = getattr(self, "_episode_terminal_check", None)
        if callable(check):
            check()

    def _physics_step(self):
        """Current physics-step count, or None when this process has no clock attached."""
        fn = getattr(self, "_sim_step_fn", None)
        if fn is None:
            return None
        try:
            return int(fn())
        except Exception:
            return None

    def _raise_if_motion_interrupted(self):
        """Surface a latched monitor stop even when an upstream primitive swallowed its signal."""
        monitor = getattr(self, "_collision_monitor", None)
        raise_if_interrupted = getattr(monitor, "raise_if_interrupted", None)
        if callable(raise_if_interrupted):
            raise_if_interrupted()

    def _run_with_dense_progress(self, targets, fn):
        """Run one inherited dense controller while the shared measured-progress guard is active."""
        guard = getattr(self, "_motion_progress", None)
        step = self._physics_step()
        if guard is None or step is None:
            return fn(), None
        guard.begin(targets, step)
        result = None
        try:
            try:
                result = fn()
            except MotionProgressInterrupted:
                # Some inherited primitives catch Exception and convert it to FAILED; others may
                # let the internal control-flow signal through. The latched event is authoritative.
                pass
        finally:
            event = guard.finish()
        return result, event

    @staticmethod
    def _physics_delta(before, after):
        """Steps advanced between two reads; None whenever either read was unavailable."""
        if before is None or after is None:
            return None
        return max(0, int(after) - int(before))

    def _state_fingerprint(self, arms=("left", "right")):
        """Raw robot state used as the CLOCKLESS execution evidence: joint positions, TCP pose and
        FK finger gap per arm. Unreadable components stay None, which keeps "we could not tell"
        distinct from "nothing moved"."""
        out = []
        for arm in arms:
            try:
                entity = (self._env.robot.left_entity if arm == "left"
                          else self._env.robot.right_entity)
                qpos = [float(v) for v in entity.get_qpos()]
            except Exception:
                qpos = None
            try:
                tcp = self._tcp(arm)
            except Exception:
                tcp = None
            try:
                gap = self._finger_gap(arm)
            except Exception:
                gap = None
            out.append((arm, qpos, tcp, gap))
        return tuple(out)

    def _record_motion_trace(self, record):
        """Append one host-only motion record beside the observer frames.

        Same standing as the GT probe: privileged, analysis-only, and unreachable from the tool
        surface (no filesystem tool addresses it). Every failure is swallowed — a diagnostic that
        can break a motion is worse than no diagnostic.
        """
        try:
            with open(self._out / "motion_trace.jsonl", "a", encoding="utf-8") as stream:
                stream.write(to_json(record) + "\n")
        except Exception:
            pass

    def _advance_tick_if_executed(self, evidence):
        """The ONLY place the episode state clock moves.

        A call that changed nothing keeps the tick, so existing observations stay valid and the
        action timeline stays truthful. Attempts are still individually addressable: `action_id`
        numbers every call, executed or not.
        """
        if not evidence.advanced:
            return False
        self._tick += 1
        return True

    @staticmethod
    def _finite_float_list(value):
        if value is None:
            return None
        try:
            out = [float(v) for v in np.asarray(value).reshape(-1)]
        except Exception:
            return None
        return out if all(np.isfinite(v) for v in out) else None

    @staticmethod
    def _joint_type(joint):
        for name in ("get_type", "type"):
            try:
                value = getattr(joint, name)
                value = value() if callable(value) else value
                if value is not None:
                    return str(value).lower()
            except Exception:
                continue
        return "unknown"

    @staticmethod
    def _joint_position_unit(joint_type):
        kind = str(joint_type).lower()
        if "revolute" in kind or "continuous" in kind:
            return "rad"
        if "prismatic" in kind:
            return "m"
        return "native"

    def _joint_state(self, arm):
        """Raw articulation qpos/qvel with a name, type, and unit for every reported value."""
        failures = []
        entity = None
        try:
            robot = self._env.robot
            entity = robot.left_entity if arm == "left" else robot.right_entity
        except Exception:
            failures.extend([
                "articulation_unavailable",
                "joint_metadata_unavailable",
                "joint_positions_unavailable",
                "joint_velocities_unavailable",
            ])

        joints = []
        if entity is not None:
            try:
                joints = list(entity.get_active_joints())
            except Exception:
                failures.append("joint_metadata_unavailable")

        positions = None
        velocities = None
        if entity is not None:
            try:
                positions = self._finite_float_list(entity.get_qpos())
            except Exception:
                positions = None
            if positions is None:
                failures.append("joint_positions_unavailable")
            try:
                velocities = self._finite_float_list(entity.get_qvel())
            except Exception:
                velocities = None
            if velocities is None:
                failures.append("joint_velocities_unavailable")

        count = max(len(joints), len(positions or ()), len(velocities or ()))
        if count and not joints:
            failures.append("joint_metadata_unavailable")
        names, types = [], []
        for index in range(count):
            joint = joints[index] if index < len(joints) else None
            try:
                name = str(joint.get_name()) if joint is not None else f"joint_{index}"
            except Exception:
                name = f"joint_{index}"
                failures.append("joint_name_unavailable")
            names.append(name)
            joint_type = self._joint_type(joint) if joint is not None else "unknown"
            if joint_type == "unknown":
                failures.append("joint_type_unavailable")
            types.append(joint_type)
        if joints and len(joints) != count:
            failures.append("joint_metadata_length_mismatch")

        position_units = [self._joint_position_unit(kind) for kind in types]
        velocity_units = [f"{unit}/s" if unit != "native" else "native"
                          for unit in position_units]
        return {
            "names": names,
            "types": types,
            "positions": positions,
            "velocities": velocities,
            "position_units": position_units,
            "velocity_units": velocity_units,
            "read_failures": list(dict.fromkeys(failures)),
        }

    @staticmethod
    def _finite_number(value):
        try:
            value = float(value)
        except Exception:
            return None
        return value if np.isfinite(value) else None

    @staticmethod
    def _boolean(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return None

    def _arm_pose_read(self, arm):
        """Validated pose-only read shared by the lightweight API and boundary snapshot."""
        failures = []
        pose_source_available = True
        try:
            result = self._p.get_gripper_pose(self._env, arm)
            pose = result.get("data") or {}
            if not isinstance(pose, dict):
                raise TypeError("pose data is not an object")
        except Exception:
            pose = {}
            pose_source_available = False
            failures.append("pose_unavailable")

        ee_pose = self._finite_float_list(pose.get("pose"))
        if ee_pose is None or len(ee_pose) != 7:
            ee_pose = None
            failures.append("ee_pose_unavailable")
        tcp_pose = self._finite_float_list(pose.get("tcp_pose"))
        if tcp_pose is None or len(tcp_pose) != 7:
            tcp_pose = None
            failures.append("tcp_pose_unavailable")
        orientation = self._pose_axes(tcp_pose)
        if orientation is None:
            failures.append("orientation_unavailable")
        return {
            "ee_pose": ee_pose,
            "tcp_pose": tcp_pose,
            "orientation": orientation,
            "_opening_m": self._finite_number(pose.get("gripper_width_m")),
            "_pose_source_available": pose_source_available,
            "read_failures": list(dict.fromkeys(failures)),
        }

    def _gripper_state_read(self, arm, pose_read=None):
        """Validated drive/pose/FK reads with one stable failure code per unavailable field."""
        failures = []
        try:
            result = self._p.get_gripper_state(self._env, arm)
            state = result.get("data") or {}
            if not isinstance(state, dict):
                raise TypeError("gripper state data is not an object")
        except Exception:
            state = {}
            failures.append("gripper_state_unavailable")
        if pose_read is None:
            try:
                result = self._p.get_gripper_pose(self._env, arm)
                pose = result.get("data") or {}
                if not isinstance(pose, dict):
                    raise TypeError("gripper pose data is not an object")
                opening = self._finite_number(pose.get("gripper_width_m"))
            except Exception:
                opening = None
                failures.append("pose_unavailable")
        else:
            opening = pose_read["_opening_m"]
            if not pose_read["_pose_source_available"]:
                failures.append("pose_unavailable")
        try:
            finger_gap = self._finite_number(self._finger_gap(arm))
        except Exception:
            finger_gap = None

        drive_value = self._finite_number(state.get("gripper_val"))
        drive_closed = self._boolean(state.get("is_closed"))
        if opening is None:
            failures.append("opening_unavailable")
        if finger_gap is None:
            failures.append("finger_gap_unavailable")
        if drive_value is None:
            failures.append("gripper_val_unavailable")
        if drive_closed is None:
            failures.append("drive_state_unavailable")
        return {
            "opening_m": opening,
            "finger_gap_m": finger_gap,
            "gripper_val": drive_value,
            "drive_commanded_closed": drive_closed,
            "read_failures": list(dict.fromkeys(failures)),
        }

    def _snapshot_arm(self, arm, include_joint_state=False):
        """One arm's robot-observable state; unavailable reads remain explicit nulls.

        Joint qpos/qvel is a PULL, not a push: `include_joint_state` is on only for
        `get_robot_state`, which exists to be asked for it. Measured 2026-08-13 across two run
        corpora, `joint_state` was ~73% of every boundary snapshot and 0 callers ever read it FROM a
        snapshot (the one real read in 1,019 agent code blocks came from `get_robot_state`), while
        the vector it carried was wrong twice over: on a single-URDF dual-arm embodiment
        `robot.left_entity is robot.right_entity`, so both arms reported the identical whole-robot
        38-joint vector — mobile base and the unused leader chains included — and 22 of those 38
        never move. The interface publishes no joint limits or link geometry, so the numbers were
        also uninterpretable by construction.
        """
        pose = self._arm_pose_read(arm)
        gripper = self._gripper_state_read(arm, pose_read=pose)
        failures = list(pose["read_failures"]) + list(gripper["read_failures"])

        try:
            contact = self._contact(arm)
            if not isinstance(contact, dict):
                contact = None
        except Exception:
            contact = None
        if contact is None:
            failures.append("contact_unavailable")

        joint_state = None
        if include_joint_state:
            joint_state = self._joint_state(arm)
            failures.extend(joint_state["read_failures"])
        snapshot = {
            "ee_pose": pose["ee_pose"],
            "tcp_pose": pose["tcp_pose"],
            "orientation": pose["orientation"],
            "opening_m": gripper["opening_m"],
            "finger_gap_m": gripper["finger_gap_m"],
            "gripper_val": gripper["gripper_val"],
            "drive_commanded_closed": gripper["drive_commanded_closed"],
            "contact": contact,
            "read_failures": list(dict.fromkeys(failures)),
        }
        if joint_state is not None:
            snapshot["joint_state"] = joint_state
        return snapshot

    def _robot_snapshot(self, include_joint_state=False):
        """Canonical raw boundary snapshot. Both arms are always present."""
        return {
            "tick": int(getattr(self, "_tick", 0)),
            "frame": "world",
            "pose_format": "[x,y,z,qw,qx,qy,qz]",
            "linear_unit": "m",
            "angular_unit": "rad",
            "contact_impulse_unit": "N*s",
            "arms": {arm: self._snapshot_arm(arm, include_joint_state=include_joint_state)
                     for arm in ("left", "right")},
        }

    @staticmethod
    def _resulting_pose_from_snapshot(result, snapshot):
        pose = dict(result.resulting_pose or {})
        arms = snapshot["arms"]
        if "left_tcp" in pose or "right_tcp" in pose:
            return {"left_tcp": arms["left"]["tcp_pose"],
                    "right_tcp": arms["right"]["tcp_pose"]}
        if "tcp" not in pose:
            return pose
        arm = result.commanded.get("arm")
        camera = result.commanded.get("camera")
        if arm not in ("left", "right") and isinstance(camera, str):
            arm = camera.split("_", 1)[0]
        return {"tcp": arms[arm]["tcp_pose"]} if arm in arms else pose

    def _attach_action_observations(self, result, observed_before):
        """Decorate a direct ActionResult or an ObservationPair's nested motion."""
        observed_after = self._robot_snapshot()

        def attach(action):
            return replace(
                action,
                observed_before=observed_before,
                observed_after=observed_after,
                resulting_pose=self._resulting_pose_from_snapshot(action, observed_after),
                tick=observed_after["tick"],
            )

        if isinstance(result, ActionResult):
            return attach(result)
        if isinstance(result, ObservationPair):
            result = replace(result, motion=attach(result.motion))
            if hasattr(self, "_pairs"):
                self._pairs[result.pair_id] = result
        return result

    def _robot_link_names(self):
        """Every link of both arms. Contact with the robot's OWN body is self-touch, not evidence
        about the scene: the arm can rest a finger on its wrist camera, which would otherwise be
        reported as a world contact and could trip contact-abort."""
        cached = getattr(self, "_robot_links", None)
        if cached is not None:
            return cached
        names = set()
        for attr in ("left_entity", "right_entity"):
            entity = getattr(getattr(self._env, "robot", None), attr, None)
            for getter in ("get_links", "links"):
                links = getattr(entity, getter, None)
                links = links() if callable(links) else links
                if links:
                    for link in links:
                        try:
                            names.add(link.get_name())
                        except Exception:
                            continue
                    break
        self._robot_links = names
        return names

    def _contact(self, arm):
        """Finger↔WORLD contact impulse + the FK world position of each contacting finger link.

        Unfiltered by object identity (leak fix #1). Finger↔finger self-contact is EXCLUDED — a
        fully-closed empty gripper presses its own fingers together, which is proprioception
        (self-touch), not world contact; counting it would disarm contact-abort and mask real
        touches. (Standalone aggregation: the primitives' reader counts self-pinch, and main-flow
        modules stay untouched.)

        `contacting_finger_world_xyz` makes a touch a MEASUREMENT rather than only a stop signal:
        a contact event localizes a world point to within the finger's own geometry. The reported
        position is the contacting link's forward-kinematics origin — the robot's own body, which a
        real arm knows without any sensor beyond joint encoders — NOT the simulator's contact-solver
        point, which would be scene ground truth. Combine it with
        `get_embodiment().gripper.finger_length_m / finger_thickness_m` to bound the offset from the
        link origin to the touched surface. It names nothing about WHAT was touched.
        """
        grip = self._env.robot.left_gripper if arm == "left" else self._env.robot.right_gripper
        finger_links = {}
        for g in grip:
            link = g[0].child_link
            finger_links.setdefault(link.get_name(), link)
        fingers = set(finger_links)
        own = self._robot_link_names() | fingers
        contacts = read_scene_contacts(self._env)
        total, per = 0.0, {}
        for c in contacts:
            n0 = c.bodies[0].entity.name
            n1 = c.bodies[1].entity.name
            if n0 in fingers and n1 not in own:
                finger = n0
            elif n1 in fingers and n0 not in own:
                finger = n1
            else:
                continue                       # self-touch: not a world contact
            imp = sum(float(sum(float(v) ** 2 for v in p.impulse) ** 0.5) for p in c.points)
            total += imp
            per[finger] = per.get(finger, 0.0) + imp
        n_fingers = sum(1 for v in per.values() if v > _CONTACT_IMPULSE_EPS)
        world_xyz = {}
        for name, impulse in per.items():
            if impulse <= _CONTACT_IMPULSE_EPS:
                continue
            link = finger_links.get(name)
            if link is None:
                continue
            try:
                pose = link.get_pose()
                entry = {"xyz": [float(v) for v in pose.p]}
                quat = [float(v) for v in pose.q]
                entry["quat_wxyz"] = quat
                entry.update(_orientation.decode_pose_axes(quat))
            except Exception:
                continue
            world_xyz[name] = entry
        # `n_contact_points` and `in_contact_both_fingers` were dropped 2026-08-05: the first is a
        # solver artefact with no real-robot counterpart (it moves with the physics engine and the
        # mesh resolution), the second reads as "it is holding something" while meaning only that
        # two finger links each touched some anonymous world geometry.
        out = {
            "total_impulse": float(total),
            "fingers_in_contact": n_fingers,
            "per_finger": {k: float(v) for k, v in per.items()},
        }
        if world_xyz:
            out["contacting_finger_pose"] = world_xyz
            out["contact_localization"] = (
                "forward-kinematics world pose of each available contacting finger-link origin; "
                "compare its keys with above-threshold per_finger keys to detect a partial FK "
                "localization read. It bounds a touch to the link geometry, not a solver contact "
                "point, object identity, contact face, or grasp verdict.")
        return out

    def _touching(self, c):
        return c["fingers_in_contact"] >= 1 or c["total_impulse"] > _CONTACT_IMPULSE_EPS

    def _contact_signature(self, contact):
        """Compact, object-agnostic contact evidence for closed-loop manipulation.

        Carries the contacting links' FK world positions so a probe's start/end signatures are
        metric evidence, not just a count (see `_contact`).
        """
        c = contact or {}
        per_finger = dict(c.get("per_finger") or {})
        contacting = sorted(k for k, v in per_finger.items()
                            if float(v) > _CONTACT_IMPULSE_EPS)
        out = {"finger_count": int(c.get("fingers_in_contact", len(contacting)) or 0),
               "contacting_fingers": contacting,
               "total_impulse": float(c.get("total_impulse", 0.0) or 0.0),
               "per_finger": per_finger}
        poses = c.get("contacting_finger_pose")
        if poses:
            out["contacting_finger_pose"] = dict(poses)
        return out

    def _contact_transition_label(self, start, end):
        before = int(start.get("finger_count", 0))
        after = int(end.get("finger_count", 0))
        before_fingers = set(start.get("contacting_fingers") or ())
        after_fingers = set(end.get("contacting_fingers") or ())
        if before == after and before_fingers == after_fingers:
            return f"{before}_finger_unchanged"
        return f"{before}_to_{after}_fingers"

    def _straight_leg_diagnostics(self, cur, step_target, new, target):
        """Detect a gross real-TCP departure from one commanded Cartesian leg.

        This is a containment guard, not a planner-success proxy: the simulator may report a
        successful plan while collision/tracking moves the measured TCP somewhere very different.
        The guard cannot undo that first physical deviation, but it prevents the leg loop from
        compounding it with more commands.
        """
        cur = np.asarray(cur, float)
        step_target = np.asarray(step_target, float)
        new = np.asarray(new, float)
        target = np.asarray(target, float)
        command = step_target - cur
        commanded_m = float(np.linalg.norm(command))
        actual = new - cur
        actual_m = float(np.linalg.norm(actual))
        if commanded_m > 1e-12:
            axis = command / commanded_m
            longitudinal_m = float(np.dot(actual, axis))
            lateral_m = float(np.linalg.norm(actual - longitudinal_m * axis))
        else:
            longitudinal_m, lateral_m = 0.0, actual_m
        remaining_before_m = float(np.linalg.norm(target - cur))
        remaining_after_m = float(np.linalg.norm(target - new))
        bounds = _mv.trajectory_deviation_bounds(commanded_m)
        reasons = []
        if actual_m > bounds["actual_leg_m"]:
            reasons.append("actual_leg_too_large")
        if lateral_m > bounds["lateral_m"]:
            reasons.append("excessive_lateral_motion")
        if remaining_after_m > remaining_before_m + bounds["remaining_growth_m"]:
            reasons.append("moved_away_from_target")
        return {"deviated": bool(reasons),
                "reasons": reasons,
                "commanded_leg_m": commanded_m,
                "actual_leg_m": actual_m,
                "longitudinal_m": longitudinal_m,
                "lateral_m": lateral_m,
                "target_remaining_before_m": remaining_before_m,
                "target_remaining_after_m": remaining_after_m,
                "bounds": bounds}

    # Progress floor of the stall counter and how many consecutive legs must sit below it. Both are
    # loop-termination bookkeeping, declared in the initial episode configuration alongside the
    # other internal parameters — they decide when the loop stops issuing commands, never whether
    # the motion was any good.
    _STALL_PROGRESS_FLOOR_M = _harness.STALL_PROGRESS_FLOOR_M
    _STALL_CONSECUTIVE_LEGS = _harness.STALL_CONSECUTIVE_LEGS

    def _guard_record(self, abort_reason, *, target_xyz=None,
                      workspace_message=None, leg_diagnostics=None):
        """Return the guard identity and final control-flow consequence.

        Intermediate leg measurements stay inside the motion loop. Workspace rejection and gross
        trajectory deviation retain their triggering evidence.
        """
        if not abort_reason:
            return None
        common = {"name": str(abort_reason),
                  "effect": "the commanded motion stopped before its endpoint"}
        if abort_reason == "workspace" or str(abort_reason).startswith("workspace"):
            return {**common,
                    "effect": "the command was rejected before any motion was executed",
                    "observed": {"target_xyz": (list(target_xyz)
                                                if target_xyz is not None else None),
                                 "detail": workspace_message},
                    "bound": {"envelope": dict(self._pu.DEFAULT_WORKSPACE),
                              "basis": "safety envelope around the robot itself, like a joint "
                                       "limit; being outside it is not evidence that the pose is "
                                       "kinematically unreachable"}}
        if abort_reason == "trajectory_deviation" and leg_diagnostics:
            measured_keys = (
                "commanded_leg_m", "actual_leg_m", "longitudinal_m", "lateral_m",
                "target_remaining_before_m", "target_remaining_after_m",
            )
            return {
                **common,
                "observed": {key: leg_diagnostics.get(key) for key in measured_keys},
                "bound": {
                    **dict(leg_diagnostics.get("bounds") or {}),
                    "rule": _mv.trajectory_deviation_rule(),
                },
            }
        return {**common, "observed": {}, "bound": {}}

    def _aid(self):
        self._n_act += 1
        return f"act_{self._n_act:03d}"

    def _straight_failure_fields(self, reason):
        if not reason:
            stage, category, detail = None, None, None
        elif reason == "leg plan failed":
            stage = "waypoint_leg_plan"
            category = "waypoint_leg_plan_failure"
            detail = ("an orientation-holding waypoint-leg plan was refused from the current arm "
                      "configuration; its planner-selected trajectory was not a promised "
                      "Cartesian line, and this refusal does not prove that no other route exists")
        elif reason == "single plan failed":
            stage = "single_plan"
            category = "single_plan_failure"
            detail = ("the single planning query to the commanded endpoint was refused from the "
                      "current arm configuration; no caller-imposed Cartesian path constraint was "
                      "applied, and this refusal does not prove that no other route exists")
        elif reason == "correction plan failed before the commanded target was reached":
            stage, category = "correction_plan", "correction_plan_failure"
            detail = ("the primary transport executed but the bounded correction plan was "
                      "refused before the commanded target was reached")
        elif reason == "leg budget exhausted":
            stage, category = "waypoint_leg_budget", "leg_budget_exhausted"
            detail = "the waypoint-leg budget ended before the commanded target was reached"
        elif str(reason).startswith("stalled"):
            stage, category = "motion_execution", "stalled"
            detail = "the commanded transport stopped making measured progress before its target"
        elif reason == "trajectory deviation":
            stage, category = "waypoint_leg_execution", "trajectory_deviation"
            detail = ("the measured TCP departed grossly from the commanded waypoint leg; "
                      "the remaining legs were stopped and the physical state was not rolled back")
        elif reason == "no_contact_within_dz":
            stage, category = "probe_budget", "no_contact_within_dz"
            detail = "no finger contact was observed before the signed dz budget ended"
        elif reason == "no_contact_change_within_distance":
            stage, category = "probe_budget", "no_contact_change_within_distance"
            detail = "the contacting-finger set did not change within the caller's travel budget"
        elif reason in {"no tcp read", "dz is zero", "direction_xyz must be non-zero",
                        "distance_m must be > 0", "step_m must be >= 0.002"}:
            stage = "pre_check"
            category = "invalid_input" if reason != "no tcp read" else "pre_read"
            detail = str(reason)
        elif reason == "arm_lock":
            stage, category, detail = "arm_lock", "arm_lock", "another arm command is active"
        else:
            stage, category, detail = "execution", "execution_failure", str(reason)

        fields = {"failure_stage": stage,
                  "failure_category": category,
                  "planner_detail": detail}
        # The planner's own reason for refusing, normalized (codeaction.motion.planner_status). Reported only
        # when the failure IS a planning refusal, and only as a condition -- never as advice.
        if stage in ("waypoint_leg_plan", "single_plan", "correction_plan"):
            fields["planner_status"] = getattr(self, "_last_planner_diag", None)
        return fields

    @staticmethod
    def _target_error_m(tcp, target_xyz):
        """Measured terminal position error, or None when either input is unavailable."""
        if not tcp or target_xyz is None:
            return None
        try:
            return float(np.linalg.norm(
                np.asarray(tcp[:3], float) - np.asarray(target_xyz[:3], float)))
        except (TypeError, ValueError):
            return None

    def _paired_stall_evidence(self, stalled_arms):
        """Merge the same three anonymous contact facts for a paired motion boundary."""
        arms = [str(arm) for arm in stalled_arms if arm in ("left", "right")]
        readings = [self._stall_contact_evidence(arm) for arm in arms]
        blocked_values = [reading.get("blocked_in_contact") for reading in readings]
        if any(value is True for value in blocked_values):
            blocked = True
        elif any(value is None for value in blocked_values):
            blocked = None
        else:
            blocked = False
        parts = []
        for reading in readings:
            for part in reading.get("contact_parts") or []:
                if part not in parts:
                    parts.append(part)
        pose = next((reading.get("pose_at_contact") for reading in readings
                     if reading.get("pose_at_contact") is not None), None)
        return {"blocked_in_contact": blocked,
                "contact_parts": parts,
                "pose_at_contact": pose,
                "stalled_arms": arms}

    def _stall_contact_evidence(self, arm):
        reader = getattr(getattr(self, "_collision_monitor", None),
                         "current_contact_evidence", None)
        if not callable(reader):
            return {"blocked_in_contact": None,
                    "contact_parts": [], "pose_at_contact": None}
        return reader(arm)

    def _cam_centre(self, o):
        E = np.asarray(o.cam_pose_snapshot["E"], float)[:3]
        return -E[:3, :3].T @ E[:3, 3]

    def _finger_links(self, arm):
        """The two distinct gripper finger links, in the embodiment's stable gripper order."""
        grip = self._env.robot.left_gripper if arm == "left" else self._env.robot.right_gripper
        seen, links = set(), []
        for g in grip:
            link = g[0].child_link
            nm = link.get_name()
            if nm in seen:
                continue
            seen.add(nm)
            links.append(link)
        return links

    def _finger_points(self, arm):
        """World positions of the two finger-link origins (FK robot self-geometry)."""
        return [np.asarray(link.get_pose().p, float) for link in self._finger_links(arm)]

    def _finger_gap(self, arm):
        """FK distance between the two finger-link origins.

        Contact or an obstruction can stop physical closure while the drive-derived opening_m
        follows the command. The raw gap and its empty-close difference do not identify an object,
        contact face, or width; the caller interprets them with separate contact/visual evidence.
        """
        pts = self._finger_points(arm)
        return float(np.linalg.norm(pts[0] - pts[1])) if len(pts) >= 2 else None

    def _project_pt(self, o, xyz):
        from codeaction.backends.robotwin import epipolar as epi
        p = epi.project(o.cam_pose_snapshot["K"], o.cam_pose_snapshot["E"], list(xyz))
        return None if p is None else [float(p[0]), float(p[1])]

    def _quat_wxyz_to_rot(self, quat_wxyz):
        q = np.asarray(quat_wxyz, float)
        if q.shape != (4,):
            raise ValueError("quat_wxyz must be [qw,qx,qy,qz]")
        n = float(np.linalg.norm(q))
        if n < 1e-9:
            raise ValueError("quat_wxyz must be non-zero")
        w, x, y, z = q / n
        return np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], float)

    def _pose_matrix(self, pose):
        """Convert a SAPIEN pose (or a small pose-compatible test fixture) to a 4x4 matrix."""
        if hasattr(pose, "to_transformation_matrix"):
            return np.asarray(pose.to_transformation_matrix(), float)
        p = np.asarray(pose.p, float)
        q = np.asarray(pose.q, float)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self._quat_wxyz_to_rot(q)
        matrix[:3, 3] = p
        return matrix

    @staticmethod
    def _transform_points(points, matrix):
        points = np.asarray(points, float).reshape(-1, 3)
        matrix = np.asarray(matrix, float)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def _finger_collision_geometry(self, arm):
        """Return live finger collision-shape vertices in world coordinates.

        Collision geometry is robot self-knowledge.  This deliberately does not read scene
        objects, contact-solver points, visual meshes, or privileged object poses.
        """
        records = []
        for link in self._finger_links(arm):
            pose_getter = getattr(link, "get_entity_pose", None)
            link_pose = pose_getter() if callable(pose_getter) else link.get_pose()
            link_world = self._pose_matrix(link_pose)
            shapes = []
            for shape in link.get_collision_shapes():
                vertices = np.asarray(shape.get_vertices(), float).reshape(-1, 3)
                if not len(vertices):
                    continue
                scale = np.asarray(shape.get_scale(), float)
                vertices = vertices * scale
                vertices = self._transform_points(
                    vertices, self._pose_matrix(shape.get_local_pose()))
                shapes.append(self._transform_points(vertices, link_world))
            if not shapes:
                raise ValueError(
                    f"finger link {link.get_name()!r} has no readable collision vertices")
            records.append({"link_name": link.get_name(),
                            "link_origin_world": link_world[:3, 3].copy(),
                            "shapes_world": shapes})
        if len(records) != 2:
            raise ValueError(f"expected two finger links for {arm}, got {len(records)}")
        return records

    @staticmethod
    def _require_parallel_jaw(tool_name):
        """Declare, at the entry of the tools that assume it, that the gripper has two opposed jaws.

        `grasp_center` is DEFINED as the midpoint between two inner-tip faces and the opening degree
        of freedom is a single scalar gap, so these overlays are meaningful only for a parallel jaw.
        The assumption was previously implicit and would have surfaced as an incidental "expected two
        finger links" error deep inside geometry extraction. Stating it here turns a hidden
        structural assumption into a declared precondition that fails loudly and says why. It is not
        an abstraction over other end effectors: a three-finger hand or a suction cup needs a
        different overlay, not a different branch of this one.
        """
        kind = str((_emb.get_embodiment().get("gripper") or {}).get("type", "unknown"))
        if kind != "parallel-jaw":
            raise ValueError(
                f"{tool_name} is defined for a parallel-jaw gripper (two opposed fingers closing "
                f"along one opening axis, grasp center = inner-tip midpoint); the active embodiment "
                f"declares gripper type {kind!r}")

    def _candidate_gripper_geometry(self, arm, tcp_xyz, quat_wxyz, finger_gap_m=None):
        """Live finger collision geometry re-posed at a caller-proposed TCP pose and finger gap.

        Shared by every candidate-pose overlay so one pose means one geometry regardless of which
        tool drew it. The live collision shapes are expressed in the live TCP frame and only the
        opening degree of freedom is changed, which preserves each shape exactly and keeps
        TCP/orientation semantics separate from the arm's current configuration. Pure robot
        self-knowledge: no scene object, contact solver, or privileged pose is read.
        """
        tcp = np.asarray(tcp_xyz, float)
        if tcp.shape != (3,) or not np.all(np.isfinite(tcp)):
            raise ValueError("tcp_xyz must be three finite world coordinates [x,y,z]")
        quat = np.asarray(quat_wxyz, float)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)) or float(np.linalg.norm(quat)) < 1e-9:
            raise ValueError("quat_wxyz must be four finite numbers with non-zero norm")
        if finger_gap_m is None:
            gap = self._finger_gap(arm)
            if gap is None:
                raise ValueError("could not read live finger_gap_m; pass finger_gap_m explicitly")
            gap_source = "live FK finger_gap_m"
        else:
            gap = float(finger_gap_m)
            gap_source = "caller"
        gap_min_m, gap_max_m = _emb.get_finger_gap_bounds_m()
        if not np.isfinite(gap) or gap < gap_min_m or gap > gap_max_m:
            raise ValueError(
                f"finger_gap_m must be within the active embodiment's nominal "
                f"[{gap_min_m:.3f},{gap_max_m:.3f}] m range")
        live_tcp = self._tcp(arm)
        if not live_tcp or len(live_tcp) != 7:
            raise ValueError(f"could not read live {arm} TCP pose")
        live_geometry = self._finger_collision_geometry(arm)
        live_tcp_xyz = np.asarray(live_tcp[:3], float)
        live_R = self._quat_wxyz_to_rot(live_tcp[3:])
        R = self._quat_wxyz_to_rot(quat_wxyz)

        origins_tcp = [live_R.T @ (rec["link_origin_world"] - live_tcp_xyz)
                       for rec in live_geometry]
        origin_center_tcp = 0.5 * (origins_tcp[0] + origins_tcp[1])
        opening_tcp = np.asarray([0.0, 1.0, 0.0])
        signed = [float(np.dot(origin - origin_center_tcp, opening_tcp))
                  for origin in origins_tcp]
        if abs(signed[0] - signed[1]) < 1e-9:
            raise ValueError("live finger origins are degenerate along TCP opening axis")
        sides = [-1.0, 1.0] if signed[0] < signed[1] else [1.0, -1.0]
        live_origin_delta = origins_tcp[1] - origins_tcp[0]
        transverse_delta = (live_origin_delta
                            - float(np.dot(live_origin_delta, opening_tcp)) * opening_tcp)
        transverse_m = float(np.linalg.norm(transverse_delta))
        if gap <= transverse_m:
            raise ValueError("finger_gap_m is smaller than the fingers' fixed transverse offset")
        opening_half_gap = 0.5 * float((gap * gap - transverse_m * transverse_m) ** 0.5)

        fingers = []
        for rec, origin_tcp, side in zip(live_geometry, origins_tcp, sides):
            shift = side * opening_half_gap - float(np.dot(
                origin_tcp - origin_center_tcp, opening_tcp))
            shifted_origin_tcp = origin_tcp + shift * opening_tcp
            shape_tcp_parts, shapes_world = [], []
            for shape_world in rec["shapes_world"]:
                shape_tcp = (np.asarray(shape_world, float) - live_tcp_xyz) @ live_R
                shape_tcp = shape_tcp + shift * opening_tcp
                shape_tcp_parts.append(shape_tcp)
                shapes_world.append(shape_tcp @ R.T + tcp)
            vertices_tcp = np.concatenate(shape_tcp_parts, axis=0)
            box_min_tcp = vertices_tcp.min(axis=0)
            box_max_tcp = vertices_tcp.max(axis=0)
            box_tcp = np.asarray([
                [x, y, z]
                for x in (box_min_tcp[0], box_max_tcp[0])
                for y in (box_min_tcp[1], box_max_tcp[1])
                for z in (box_min_tcp[2], box_max_tcp[2])
            ])
            inner_y = box_max_tcp[1] if side < 0 else box_min_tcp[1]
            inner_tip_tcp = np.asarray([
                box_max_tcp[0], inner_y, 0.5 * (box_min_tcp[2] + box_max_tcp[2])])
            fingers.append({
                "link_name": rec["link_name"],
                "shapes_world": shapes_world,
                "shape_tcp_parts": shape_tcp_parts,
                "vertex_count": int(sum(len(part) for part in shape_tcp_parts)),
                "box_min_tcp": box_min_tcp,
                "box_max_tcp": box_max_tcp,
                "box_world": box_tcp @ R.T + tcp,
                # Oriented-box descriptor for the separating-axis clearance test: the same display
                # box, expressed as centre + half extents + world axes rather than as 8 corners.
                "box_center_world": tcp + R @ (0.5 * (box_min_tcp + box_max_tcp)),
                "box_half_extent": 0.5 * (box_max_tcp - box_min_tcp),
                "box_axes_world": np.asarray([R[:, 0], R[:, 1], R[:, 2]]),
                "link_origin_world": tcp + R @ shifted_origin_tcp,
                "inner_tip_world": tcp + R @ inner_tip_tcp,
            })
        return {
            "arm": arm, "tcp": tcp, "quat": quat, "R": R,
            "gap": gap, "gap_source": gap_source,
            "fingers": fingers,
            "origin_center_world": tcp + R @ origin_center_tcp,
            "grasp_center": 0.5 * (fingers[0]["inner_tip_world"]
                                   + fingers[1]["inner_tip_world"]),
        }

    @staticmethod
    def _box_separation(a, b):
        """Separating-axis clearance between two oriented boxes, in meters.

        Returns (separation_m, axis_world). Positive = a PROVEN gap of at least that size along
        `axis_world`; the true clearance between the enclosed collision shapes is at least this,
        because each box contains its shapes. Zero or negative = the boxes overlap, and the
        magnitude is the smallest translation that would separate them — which the true shapes may
        not need, since a box is larger than what it encloses. So "no overlap" is sound and
        "overlap" is conservative, which is the safe way round for a clearance question.
        """
        centers = [item["box_center_world"] for item in (a, b)]
        halves = [item["box_half_extent"] for item in (a, b)]
        axes = [item["box_axes_world"] for item in (a, b)]
        delta = centers[1] - centers[0]
        tests = [axes[0][i] for i in range(3)] + [axes[1][i] for i in range(3)]
        tests += [np.cross(axes[0][i], axes[1][j]) for i in range(3) for j in range(3)]
        best_sep, best_axis = None, np.asarray([1.0, 0.0, 0.0])
        for axis in tests:
            norm = float(np.linalg.norm(axis))
            if norm < 1e-9:            # parallel box axes produce a degenerate cross product
                continue
            unit = axis / norm
            reach = sum(float(halves[k][i]) * abs(float(np.dot(axes[k][i], unit)))
                        for k in (0, 1) for i in range(3))
            separation = abs(float(np.dot(delta, unit))) - reach
            if best_sep is None or separation > best_sep:
                best_sep, best_axis = separation, unit
        return float(best_sep if best_sep is not None else 0.0), best_axis

    @staticmethod
    def _convex_hull_2d(points):
        """Monotone-chain hull of projected points; dependency-free and deterministic."""
        pts = sorted({(float(p[0]), float(p[1])) for p in points})
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return ((a[0] - o[0]) * (b[1] - o[1])
                    - (a[1] - o[1]) * (b[0] - o[0]))

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def _project_points(self, observation, points):
        """Vectorized form of `_project_pt`; returns (pixels-or-None, camera depths)."""
        points = np.asarray(points, float).reshape(-1, 3)
        K = np.asarray(observation.cam_pose_snapshot["K"], float)
        E = np.asarray(observation.cam_pose_snapshot["E"], float)[:3]
        camera = points @ E[:3, :3].T + E[:3, 3]
        pixels = [None] * len(points)
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        if np.any(valid):
            image = camera[valid] @ K.T
            image = image[:, :2] / image[:, 2:3]
            for index, pixel in zip(np.flatnonzero(valid), image):
                pixels[int(index)] = [float(pixel[0]), float(pixel[1])]
        return pixels, camera[:, 2]

    def _annotated(self, base, im, note):
        self._n_ann += 1
        obs_id = f"{base.obs_id}_a{self._n_ann:02d}"
        path = self._out / f"{obs_id}.png"
        im.save(str(path))
        o = Observation(obs_id=obs_id, camera=base.camera, image_ref=str(path),
                        cam_pose_snapshot=dict(base.cam_pose_snapshot), tick=base.tick,
                        annotation=note)
        self._obs[obs_id] = o
        return o

    # ── world / robot info (read-only) ────────────────────────────────────
