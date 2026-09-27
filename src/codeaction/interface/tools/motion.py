"""Every tool that commits motion: straight legs, reaches, probes, gripper and aim."""
from codeaction.interface.tools._base import (
    ActionResult,
    ExecutionEvidence,
    MotionTrace,
    PhysicalTimeBudgetExhausted,
    REACH_CORRECTION_TRIGGER_M,
    _aim,
    _emb,
    _fingerprint_changed,
    _mv,
    _observed_action,
    _planner_status,
    clamp_step,
    displacement_tolerance_m,
    execution_block,
    np)


class MotionTools:
    def _straight_step(self, arm, target_xyz, target_quat=None):
        """One Cartesian step to a TCP target under cuRobo `constraint_pose=[1,1,1,0,0,0]`.

        That metric holds ORIENTATION for this one solve; it does not guarantee a straight Cartesian
        path or zero lateral drift (measured: up to 6.4 cm lateral deviation, and identical with the
        metric off). Callers running several legs must pass an explicit `target_quat` — leaving it
        None makes each leg hold whatever the previous leg drifted to, which accumulates without
        bound. Returns plan ok.
        """
        from envs.utils.action import Action, ArmTag
        d = self._p.get_gripper_pose(self._env, arm).get("data") or {}
        ee, tcp = d.get("pose"), d.get("tcp_pose")
        if not ee or not tcp:
            return False
        q = list(target_quat) if target_quat is not None else list(tcp[3:])
        ee_target = self._pu._tcp_target_to_ee_target(ee, tcp, list(target_xyz), q)
        self._env.plan_success = True
        self._last_planner_diag = None
        try:
            self._env.move((ArmTag(arm), [Action(ArmTag(arm), "move", target_pose=ee_target,
                                                 constraint_pose=[1, 1, 1, 0, 0, 0])]))
        except PhysicalTimeBudgetExhausted:
            raise
        except Exception:
            return False
        ok = bool(self._env.plan_success)
        if not ok:
            self._last_planner_diag = self._planner_diag(arm)
        return ok

    def _planner_diag(self, arm):
        """Why the planner refused, from the planner's own status. `left/right_move_to_pose` appends
        the raw plan result to `left/right_joint_path` BEFORE it checks the status, so the reason
        survives the collapse to a boolean without the harness re-planning or holding extra state."""
        try:
            path = (self._env.left_joint_path if arm == "left" else self._env.right_joint_path)
            return _planner_status.planner_diagnostic(path[-1]) if path else None
        except Exception:
            return None

    def _planner_path_cursor(self):
        """Per-arm raw planner-list lengths before one primitive call."""
        cursor = {}
        for arm in ("left", "right"):
            try:
                path = (self._env.left_joint_path if arm == "left"
                        else self._env.right_joint_path)
                cursor[arm] = len(path)
            except Exception:
                cursor[arm] = None
        return cursor

    def _planner_entries_since(self, cursor):
        """Raw planner entries appended by this call only; never reuse a previous refusal."""
        out = {}
        for arm in ("left", "right"):
            start = dict(cursor or {}).get(arm)
            if start is None:
                out[arm] = []
                continue
            try:
                path = (self._env.left_joint_path if arm == "left"
                        else self._env.right_joint_path)
                out[arm] = list(path[int(start):])
            except Exception:
                out[arm] = []
        return out

    @staticmethod
    def _planner_diagnostics_from_entries(entries):
        """Last refusal diagnostic per arm from one call's bounded raw-entry slice."""
        out = {}
        for arm in ("left", "right"):
            diagnostics = [
                diagnostic for diagnostic in (
                    _planner_status.planner_diagnostic(entry)
                    for entry in dict(entries or {}).get(arm, ()))
                if diagnostic is not None
            ]
            out[arm] = diagnostics[-1] if diagnostics else None
        return out

    def _move_delta_single_plan(self, arm, dx, dy, dz):
        """`move_delta(path="single_plan")`: exactly one plan to the clamped endpoint."""
        aid = self._aid()
        if not self._lock.acquire(arm):
            tcp = self._tcp(arm)
            return ActionResult(action_id=aid,
                                commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                                           "path": "single_plan"},
                                achieved={"reason": "arm_lock",
                                          "path_mode": "single_plan",
                                          **self._straight_failure_fields("arm_lock")},
                                status="ABORTED", abort_reason="arm_lock",
                                resulting_pose={"tcp": tcp}, tick=self._tick,
                                execution=execution_block(ExecutionEvidence(
                                    attempted=False, physics_steps=0, state_changed=False,
                                    interrupted=False, completed=False,
                                    post_state_observed=tcp is not None)))
        try:
            (cdx, cdy, cdz), clamped = clamp_step([float(dx), float(dy), float(dz)],
                                                  self._max_step)
            tcp0 = self._tcp(arm)
            if not tcp0:
                return ActionResult(
                    action_id=aid,
                    commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                               "path": "single_plan"},
                    achieved={"reason": "no tcp read",
                              "path_mode": "single_plan",
                              **self._straight_failure_fields("no tcp read")},
                    status="FAILED", abort_reason=None,
                    resulting_pose={"tcp": None}, tick=self._tick,
                    execution=execution_block(ExecutionEvidence(
                        attempted=False, physics_steps=0, state_changed=False,
                        interrupted=False, completed=False, post_state_observed=False)))
            target = (np.asarray(tcp0[:3], float) + np.array([cdx, cdy, cdz])).tolist()
            q_anchor = list(tcp0[3:])
            pose = self._p.get_gripper_pose(self._env, arm).get("data") or {}
            ee_pose = pose.get("pose")
            if not ee_pose:
                return ActionResult(
                    action_id=aid,
                    commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                               "path": "single_plan"},
                    achieved={"reason": "no tcp read", "path_mode": "single_plan",
                              **self._straight_failure_fields("no tcp read")},
                    status="FAILED", abort_reason=None, resulting_pose={"tcp": tcp0},
                    tick=self._tick,
                    execution=execution_block(ExecutionEvidence(
                        attempted=False, physics_steps=0, state_changed=False,
                        interrupted=False, completed=False, post_state_observed=True)))
            ee_target = self._pu._tcp_target_to_ee_target(
                ee_pose, tcp0, target, q_anchor)
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint((arm,))
            planner_cursor = self._planner_path_cursor()
            self._env.plan_success = True
            self._last_planner_diag = None
            r, dense_stall = self._run_with_dense_progress(
                {arm: target},
                lambda: self._m.move_to_pose(self._env, arm, ee_target))
            r = r or {}
            self._raise_if_motion_interrupted()
            ok = r.get("status") == self._OK
            if not ok:
                self._last_planner_diag = self._planner_diagnostics_from_entries(
                    self._planner_entries_since(planner_cursor)).get(arm)
            tcp1 = self._tcp(arm)
            steps_after = self._physics_step()
            physics_steps = self._physics_delta(steps_before, steps_after)
            target_error = self._target_error_m(tcp1, target)
            converge_tol = displacement_tolerance_m(
                float(np.linalg.norm(np.asarray(target, float)
                                     - np.asarray(tcp0[:3], float))))
            stalled = bool(dense_stall is not None or (
                ok and target_error is not None and target_error >= converge_tol))
            stall_contact_evidence = (
                self._stall_contact_evidence(arm) if stalled else {})
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint((arm,)))
            evidence = ExecutionEvidence(
                attempted=True,
                physics_steps=physics_steps,
                state_changed=state_changed,
                interrupted=stalled,
                completed=bool(ok and tcp1 is not None and not stalled),
                post_state_observed=tcp1 is not None,
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe(arm)
            reason = ("stalled (target remained after completed plan)" if stalled
                      else None if ok else "single plan failed")
            status = "ABORTED" if stalled else "SUCCESS" if ok else "FAILED"
            failure_fields = self._straight_failure_fields(reason)
            trace = MotionTrace("move_delta")
            trace.stage("primary_plan", ok=(ok or dense_stall is not None),
                        physics_steps=physics_steps,
                        planner_status=(None if ok else self._last_planner_diag))
            trace.stop("stalled" if stalled else "converged" if ok else "leg_plan_refused")
            result = ActionResult(
                action_id=aid,
                commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                           "clamped_to": [cdx, cdy, cdz] if clamped else None,
                           "path": "single_plan"},
                achieved={"reason": reason,
                          "path_mode": "single_plan",
                          "contact": self._contact(arm),
                          "guard": self._guard_record("stalled" if stalled else None),
                          **stall_contact_evidence,
                          **failure_fields},
                status=status, abort_reason="stalled" if stalled else None,
                resulting_pose={"tcp": tcp1}, tick=self._tick,
                planning={"status": "SUCCEEDED" if ok or stalled else "FAILED"},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": str(failure_fields.get("failure_stage") or "planning"),
                    "code": ("MOTION_STALLED" if stalled else
                             str(failure_fields.get("failure_category") or "PLAN_FAILED")),
                    "message": str(reason or "the single plan did not execute"),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release(arm)

    @_observed_action
    def move_delta(self, arm, dx=0.0, dy=0.0, dz=0.0, path="waypoint_legs"):
        """World-frame displacement of the TCP. `path` selects the path discipline, and the two
        differ in what they guarantee, not in quality:

        path="waypoint_legs" (default): each leg re-aims from the latest measured TCP toward the
        fixed endpoint. With orientation_anchor enabled (the shipped default), every leg holds the
        call-start orientation. Tracking drift can move later targets off the original line; the
        planner-selected trajectory can also bow laterally. A leg can be refused while another
        endpoint route could exist.

        path="single_plan": exactly one collision-aware plan to the same endpoint, with the start
        orientation as a goal and no caller-imposed path constraint or intermediate waypoint. The
        result reports achieved.path_mode."""
        path_mode = str(path)
        if path_mode not in ("waypoint_legs", "single_plan"):
            tcp = self._tcp(arm)
            return ActionResult(
                action_id=self._aid(),
                commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                           "path": path_mode},
                achieved={"reason": "unknown path mode",
                          "path_mode": path_mode,
                          **self._straight_failure_fields("unknown path mode")},
                status="ABORTED",
                abort_reason="path must be 'waypoint_legs' or 'single_plan'",
                resulting_pose={"tcp": tcp}, tick=self._tick,
                execution=execution_block(ExecutionEvidence(
                    attempted=False, physics_steps=0, state_changed=False,
                    interrupted=False, completed=False,
                    post_state_observed=tcp is not None)))
        if path_mode == "single_plan":
            return self._move_delta_single_plan(arm, dx, dy, dz)
        aid = self._aid()
        if not self._lock.acquire(arm):
            tcp = self._tcp(arm)
            return ActionResult(action_id=aid, commanded={"arm": arm, "delta": [dx, dy, dz]},
                                achieved={"reason": "arm_lock",
                                          "path_mode": "waypoint_legs",
                                          **self._straight_failure_fields("arm_lock")},
                                status="ABORTED",
                                abort_reason="arm_lock", resulting_pose={"tcp": tcp},
                                tick=self._tick,
                                execution=execution_block(ExecutionEvidence(
                                    attempted=False, physics_steps=0, state_changed=False,
                                    interrupted=False, completed=False,
                                    post_state_observed=tcp is not None)))
        try:
            (cdx, cdy, cdz), clamped = clamp_step([float(dx), float(dy), float(dz)],
                                                  self._max_step)
            mag = float(np.hypot(np.hypot(cdx, cdy), cdz))
            tcp0 = self._tcp(arm)
            if not tcp0:
                return ActionResult(action_id=aid, commanded={"arm": arm, "delta": [dx, dy, dz]},
                                    achieved={"reason": "no tcp read",
                                              "path_mode": "waypoint_legs",
                                              **self._straight_failure_fields("no tcp read")},
                                    status="FAILED", abort_reason=None,
                                    resulting_pose={"tcp": None}, tick=self._tick,
                                    execution=execution_block(ExecutionEvidence(
                                        attempted=False, physics_steps=0, state_changed=False,
                                        interrupted=False, completed=False,
                                        post_state_observed=False)))
            target = np.asarray(tcp0[:3], float) + np.array([cdx, cdy, cdz])
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint((arm,))
            trace = MotionTrace("move_delta")
            status, abort_reason, reason = "SUCCESS", None, None
            budget = 3 * max(1, int(np.ceil(mag / self._leg)))
            iters, stalls = 0, 0
            last_leg_diag = None
            stall_contact_evidence = None
            converge_tol = displacement_tolerance_m(mag)
            # Orientation anchor: every leg holds the quaternion read at CALL START. Passing None
            # made each leg re-read the already-drifted live quat as its own hold target, so an
            # error one leg introduced became the next leg's goal and nothing ever pulled the wrist
            # back — this tool never commands rotation, yet production logged 292 deg of cumulative
            # uncommanded rotation, max 71.6 deg in one call.
            q_anchor = list(tcp0[3:])
            while mag > 1e-12:
                cur = np.asarray(self._tcp(arm)[:3], float)
                rem = target - cur
                rn = float(np.linalg.norm(rem))
                if rn < converge_tol:
                    break                                   # converged: delivered the command
                if iters >= budget:
                    status, abort_reason, reason = (
                        "ABORTED", "leg_budget", "leg budget exhausted")
                    break
                step_target = cur + rem * (min(self._leg, rn) / rn)
                leg_steps_before = self._physics_step()
                ok = self._straight_step(arm, step_target.tolist(),
                                         q_anchor if self._orientation_anchor else None)
                iters += 1
                new_tcp = self._tcp(arm)
                if not new_tcp:
                    leg_physics_steps = self._physics_delta(
                        leg_steps_before, self._physics_step())
                    trace.leg(index=iters, plan_ok=ok,
                              executed=bool(leg_physics_steps))
                    status, reason = "FAILED", "no tcp read"
                    break
                new = np.asarray(new_tcp[:3], float)
                leg_diag = self._straight_leg_diagnostics(cur, step_target, new, target)
                last_leg_diag = leg_diag
                leg_physics_steps = self._physics_delta(
                    leg_steps_before, self._physics_step())
                trace.leg(
                    index=iters,
                    plan_ok=ok,
                    executed=(leg_physics_steps > 0 if leg_physics_steps is not None
                              else leg_diag["actual_leg_m"] > 1e-9),
                    diagnostics={
                        **leg_diag,
                        "physics_steps": leg_physics_steps,
                    },
                )
                if leg_diag["deviated"]:
                    status, abort_reason, reason = (
                        "ABORTED", "trajectory_deviation", "trajectory deviation")
                    break
                if not ok:
                    status, reason = "FAILED", "leg plan failed"
                    break
                # Test the loop's own stop condition immediately after the leg. A delivered final
                # leg must not fall through into stall bookkeeping and then be reclassified from a
                # derived final error after the loop has ended.
                if float(np.linalg.norm(target - new)) < converge_tol:
                    break
                if float(np.linalg.norm(new - cur)) < self._STALL_PROGRESS_FLOOR_M:
                    stalls += 1
                    if stalls >= self._STALL_CONSECUTIVE_LEGS:
                        stall_contact_evidence = (
                            self._stall_contact_evidence(arm))
                        status, abort_reason, reason = (
                            "ABORTED", "stalled", "stalled (no progress 2 legs)")
                        break
                else:
                    stalls = 0
            tcp1 = self._tcp(arm)
            physics_steps = self._physics_delta(steps_before, self._physics_step())
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint((arm,)))
            evidence = ExecutionEvidence(
                attempted=iters > 0,
                physics_steps=physics_steps,
                state_changed=state_changed,
                interrupted=status == "ABORTED",
                completed=status == "SUCCESS",
                post_state_observed=tcp1 is not None,
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe(arm)
            guard = self._guard_record(
                abort_reason,
                leg_diagnostics=(last_leg_diag
                                 if abort_reason == "trajectory_deviation" else None))
            if status == "SUCCESS":
                stop_condition = "converged"
            elif reason == "leg plan failed":
                stop_condition = "leg_plan_refused"
            elif reason == "leg budget exhausted":
                stop_condition = "leg_budget_exhausted"
            elif reason == "no tcp read":
                stop_condition = "no_tcp_read"
            elif reason and str(reason).startswith("stalled"):
                stop_condition = "stalled"
            elif abort_reason == "trajectory_deviation":
                stop_condition = "trajectory_deviation"
            else:
                stop_condition = "not_started"
            trace.stage(
                "waypoint_legs",
                ok=status == "SUCCESS",
                physics_steps=physics_steps,
                planner_status=(getattr(self, "_last_planner_diag", None)
                                if reason == "leg plan failed" else None),
            )
            trace.stop(stop_condition, legs_budget=budget)
            failure_fields = self._straight_failure_fields(reason)
            result = ActionResult(
                action_id=aid,
                commanded={"arm": arm, "delta": [float(dx), float(dy), float(dz)],
                           "clamped_to": [cdx, cdy, cdz] if clamped else None,
                           "path": "waypoint_legs"},
                achieved={"reason": reason,
                          "path_mode": "waypoint_legs",
                          "guard": guard,
                          "contact": self._contact(arm),
                          **(stall_contact_evidence or {}),
                          **failure_fields},
                status=status, abort_reason=abort_reason,
                resulting_pose={"tcp": tcp1}, tick=self._tick,
                planning={"status": (
                    "FAILED" if reason == "leg plan failed" else
                    "NOT_RUN" if iters == 0 else "SUCCEEDED")},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": str(failure_fields.get("failure_stage")
                                 or ("abort_guard" if abort_reason else "execution")),
                    "code": str(failure_fields.get("failure_category")
                                or abort_reason or "MOTION_DID_NOT_EXECUTE"),
                    "message": str(reason or abort_reason or "the motion did not execute"),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release(arm)

    @_observed_action
    def probe_contact_along(self, arm, direction_xyz, distance_m, step_m):
        """Measure a contacting-finger-set change along a caller-selected world direction.

        The primitive keeps the current TCP orientation and has no object identity, target
        locator, direction chooser, or grasp policy. With orientation_anchor enabled (the shipped
        default), every leg holds the call-start orientation. A contact transition is evidence
        returned to the caller, not a task-specific recovery decision. Sets are compared after
        completed legs; the collision monitor separately records the first permitted contact step.
        """
        aid = self._aid()
        commanded = {"arm": arm,
                     "direction_xyz": list(direction_xyz),
                     "distance_m": float(distance_m),
                     "step_m": float(step_m)}
        direction = np.asarray(direction_xyz, float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            reason = "direction_xyz must be non-zero"
        elif float(distance_m) <= 0.0:
            reason = "distance_m must be > 0"
        elif float(step_m) < 0.002:
            reason = "step_m must be >= 0.002"
        else:
            reason = None
        if reason:
            tcp = self._tcp(arm)
            return ActionResult(
                action_id=aid, commanded=commanded,
                achieved={"reason": reason,
                          "stop_reason": None, "transition": None,
                          "start_contact": None, "end_contact": None,
                          **self._straight_failure_fields(reason)},
                status="FAILED", abort_reason=None, resulting_pose={"tcp": tcp}, tick=self._tick,
                execution=execution_block(ExecutionEvidence(
                    attempted=False, physics_steps=0, state_changed=False,
                    interrupted=False, completed=False,
                    post_state_observed=tcp is not None)))
        if not self._lock.acquire(arm):
            tcp = self._tcp(arm)
            return ActionResult(
                action_id=aid, commanded=commanded,
                achieved={"reason": "arm_lock",
                          "stop_reason": None, "transition": None,
                          "start_contact": None, "end_contact": None,
                          **self._straight_failure_fields("arm_lock")},
                status="ABORTED", abort_reason="arm_lock",
                resulting_pose={"tcp": tcp}, tick=self._tick,
                execution=execution_block(ExecutionEvidence(
                    attempted=False, physics_steps=0, state_changed=False,
                    interrupted=False, completed=False,
                    post_state_observed=tcp is not None)))
        try:
            tcp0 = self._tcp(arm)
            if not tcp0:
                return ActionResult(
                    action_id=aid, commanded=commanded,
                    achieved={"reason": "no tcp read",
                              "stop_reason": None, "transition": None,
                              "start_contact": None, "end_contact": None,
                              **self._straight_failure_fields("no tcp read")},
                    status="FAILED", abort_reason=None,
                    resulting_pose={"tcp": None}, tick=self._tick,
                    execution=execution_block(ExecutionEvidence(
                        attempted=False, physics_steps=0, state_changed=False,
                        interrupted=False, completed=False, post_state_observed=False)))

            unit = direction / norm
            travel = min(float(distance_m), self._max_step)
            target = np.asarray(tcp0[:3], float) + unit * travel
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint((arm,))
            trace = MotionTrace("probe_contact_along")
            leg = min(float(step_m), self._leg)
            budget = 3 * max(1, int(np.ceil(travel / leg)))
            start_contact = self._contact_signature(self._contact(arm))
            end_contact = start_contact
            status, abort_reason = "FAILED", None
            reason, stop_reason = None, None
            iters, stalls = 0, 0
            last_leg_diag = None
            stall_contact_evidence = None
            q_anchor = list(tcp0[3:])       # hold the CALL-START orientation, not each leg's drift
            while True:
                cur_tcp = self._tcp(arm)
                if not cur_tcp:
                    reason = "no tcp read"
                    break
                cur = np.asarray(cur_tcp[:3], float)
                rem = target - cur
                rn = float(np.linalg.norm(rem))
                if rn < 0.002:
                    reason = "no_contact_change_within_distance"
                    break
                if iters >= budget:
                    reason = "leg budget exhausted"
                    break
                step_target = cur + rem * (min(leg, rn) / rn)
                leg_steps_before = self._physics_step()
                ok = self._straight_step(arm, step_target.tolist(),
                                         q_anchor if self._orientation_anchor else None)
                iters += 1
                new_tcp = self._tcp(arm)
                if not new_tcp:
                    leg_physics_steps = self._physics_delta(
                        leg_steps_before, self._physics_step())
                    trace.leg(index=iters, plan_ok=ok,
                              executed=bool(leg_physics_steps))
                    reason = "no tcp read"
                    break
                new = np.asarray(new_tcp[:3], float)
                leg_diag = self._straight_leg_diagnostics(cur, step_target, new, target)
                last_leg_diag = leg_diag
                leg_physics_steps = self._physics_delta(
                    leg_steps_before, self._physics_step())
                trace.leg(
                    index=iters,
                    plan_ok=ok,
                    executed=(leg_physics_steps > 0 if leg_physics_steps is not None
                              else leg_diag["actual_leg_m"] > 1e-9),
                    diagnostics={**leg_diag, "physics_steps": leg_physics_steps},
                )
                if leg_diag["deviated"]:
                    status, abort_reason, reason = (
                        "ABORTED", "trajectory_deviation", "trajectory deviation")
                    break
                if not ok:
                    reason = "leg plan failed"
                    break
                if float(np.linalg.norm(new - cur)) < self._STALL_PROGRESS_FLOOR_M:
                    stalls += 1
                    if stalls >= self._STALL_CONSECUTIVE_LEGS:
                        stall_contact_evidence = (
                            self._stall_contact_evidence(arm))
                        status, abort_reason, reason = (
                            "ABORTED", "stalled", "stalled (no progress 2 legs)")
                        break
                else:
                    stalls = 0
                end_contact = self._contact_signature(self._contact(arm))
                if end_contact["contacting_fingers"] != start_contact["contacting_fingers"]:
                    status, reason, stop_reason = "SUCCESS", None, "contact_signature_changed"
                    break

            tcp1 = self._tcp(arm)
            end_contact = self._contact_signature(self._contact(arm))
            physics_steps = self._physics_delta(steps_before, self._physics_step())
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint((arm,)))
            procedure_completed = (status == "SUCCESS"
                                   or reason == "no_contact_change_within_distance")
            evidence = ExecutionEvidence(
                attempted=iters > 0, physics_steps=physics_steps,
                state_changed=state_changed, interrupted=status == "ABORTED",
                completed=procedure_completed, post_state_observed=tcp1 is not None,
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe(arm)
            start_set = set(start_contact["contacting_fingers"])
            end_set = set(end_contact["contacting_fingers"])
            transition = (self._contact_transition_label(start_contact, end_contact)
                          if start_set != end_set else None)
            if status == "SUCCESS":
                trace_stop = "contact_signature_changed"
            elif reason == "no_contact_change_within_distance":
                trace_stop = "travel_budget_exhausted"
            elif reason == "leg budget exhausted":
                trace_stop = "leg_budget_exhausted"
            elif reason == "leg plan failed":
                trace_stop = "leg_plan_refused"
            elif reason == "no tcp read":
                trace_stop = "no_tcp_read"
            elif reason and str(reason).startswith("stalled"):
                trace_stop = "stalled"
            elif abort_reason == "trajectory_deviation":
                trace_stop = "trajectory_deviation"
            else:
                trace_stop = "not_started"
            trace.stage(
                "waypoint_legs", ok=procedure_completed, physics_steps=physics_steps,
                planner_status=(getattr(self, "_last_planner_diag", None)
                                if reason == "leg plan failed" else None),
            )
            trace.stop(trace_stop, legs_budget=budget)
            achieved = {
                "reason": reason,
                "stop_reason": stop_reason,
                "transition": transition,
                "fingers_added": sorted(end_set - start_set),
                "fingers_lost": sorted(start_set - end_set),
                "start_contact": start_contact,
                "end_contact": end_contact,
                "direction_unit_world": unit.tolist(),
                "travel_budget_m": travel,
                "effective_step_m": leg,
                "guard": self._guard_record(
                    abort_reason,
                    leg_diagnostics=(last_leg_diag
                                     if abort_reason == "trajectory_deviation" else None)),
                **(stall_contact_evidence or {}),
                **self._straight_failure_fields(reason),
            }
            result = ActionResult(
                action_id=aid, commanded=commanded, achieved=achieved,
                status=status, abort_reason=abort_reason,
                resulting_pose={"tcp": tcp1}, tick=self._tick,
                planning={"status": (
                    "FAILED" if reason == "leg plan failed" else
                    "NOT_RUN" if iters == 0 else "SUCCEEDED")},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": str(achieved.get("failure_stage") or "execution"),
                    "code": str(achieved.get("failure_category") or
                                abort_reason or "CONTACT_SET_UNCHANGED"),
                    "message": str(reason or abort_reason or
                                   "contact set did not change"),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release(arm)

    @_observed_action
    def move_both_delta(self, left_delta, right_delta):
        """Synchronized dual-arm world-frame displacement. This is the explicit paired exception
        to the default one-arm-at-a-time guard. A failed paired call may already have moved an arm;
        execution evidence and both post-state snapshots carry that fact."""
        aid = self._aid()
        if len(left_delta) != 3 or len(right_delta) != 3:
            raise ValueError("left_delta and right_delta must each be [dx,dy,dz]")
        l_delta, l_clamped = clamp_step([float(v) for v in left_delta], self._max_step)
        r_delta, r_clamped = clamp_step([float(v) for v in right_delta], self._max_step)
        commanded = {"left_delta": [float(v) for v in left_delta],
                     "right_delta": [float(v) for v in right_delta],
                     "clamped_to": {"left": l_delta if l_clamped else None,
                                    "right": r_delta if r_clamped else None}}
        if not self._lock.acquire_pair("left", "right"):
            left_tcp, right_tcp = self._tcp("left"), self._tcp("right")
            return ActionResult(action_id=aid, commanded=commanded,
                                achieved={"sync": {"mode": "simultaneous",
                                                   "reason": "arm_lock",
                                                   "failure_stage": "arm_lock",
                                                   "failure_category": "arm_lock",
                                                   "planner_status_available": False}},
                                status="ABORTED", abort_reason="arm_lock",
                                resulting_pose={"left_tcp": left_tcp,
                                                "right_tcp": right_tcp},
                                tick=self._tick,
                                execution=execution_block(ExecutionEvidence(
                                    attempted=False, physics_steps=0, state_changed=False,
                                    interrupted=False, completed=False,
                                    post_state_observed=bool(left_tcp and right_tcp))))
        try:
            tcp0 = {"left": self._tcp("left"), "right": self._tcp("right")}
            if not tcp0["left"] or not tcp0["right"]:
                return ActionResult(action_id=aid, commanded=commanded,
                                    achieved={"sync": {"mode": "simultaneous",
                                                       "reason": "no tcp read",
                                                       "failure_stage": "pre_read",
                                                       "failure_category": "pre_read",
                                                       "planner_status_available": False}},
                                    status="FAILED", abort_reason=None,
                                    resulting_pose={"left_tcp": tcp0["left"],
                                                    "right_tcp": tcp0["right"]},
                                    tick=self._tick,
                                    execution=execution_block(ExecutionEvidence(
                                        attempted=False, physics_steps=0, state_changed=False,
                                        interrupted=False, completed=False,
                                        post_state_observed=False)))
            targets = {
                "left": (np.asarray(tcp0["left"][:3], float)
                         + np.asarray(l_delta, float)).tolist(),
                "right": (np.asarray(tcp0["right"][:3], float)
                          + np.asarray(r_delta, float)).tolist(),
            }
            tolerances = {
                "left": displacement_tolerance_m(float(np.linalg.norm(l_delta))),
                "right": displacement_tolerance_m(float(np.linalg.norm(r_delta))),
            }
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint(("left", "right"))
            planner_cursor = self._planner_path_cursor()
            r, dense_stall = self._run_with_dense_progress(
                targets, lambda: self._m.move_both_delta(self._env, l_delta, r_delta))
            r = r or {}
            self._raise_if_motion_interrupted()
            ok = r.get("status") == self._OK
            data = r.get("data") or {}
            tcp1 = {"left": self._tcp("left"), "right": self._tcp("right")}
            stalled_arms = list(dense_stall.arms) if dense_stall is not None else [
                arm for arm in ("left", "right")
                if ok and self._target_error_m(tcp1[arm], targets[arm]) is not None
                and self._target_error_m(tcp1[arm], targets[arm]) >= tolerances[arm]
            ]
            stalled = bool(stalled_arms)
            stall_contact_evidence = (
                self._paired_stall_evidence(stalled_arms) if stalled else {})
            physics_steps = self._physics_delta(steps_before, self._physics_step())
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint(("left", "right")))
            evidence = ExecutionEvidence(
                attempted=True, physics_steps=physics_steps, state_changed=state_changed,
                interrupted=stalled,
                completed=bool(ok and not stalled and tcp1["left"] and tcp1["right"]),
                post_state_observed=bool(tcp1["left"] and tcp1["right"]),
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe_both()
            planner_diagnostics = self._planner_diagnostics_from_entries(
                self._planner_entries_since(planner_cursor))
            planner_status_available = any(
                value is not None for value in planner_diagnostics.values())
            reason = ("stalled before one or both commanded targets were reached" if stalled
                      else None if ok else str(r.get("details") or r.get("message") or
                                              "dual-arm displacement motion planning failed"))
            status = "ABORTED" if stalled else "SUCCESS" if ok else "FAILED"
            trace = MotionTrace("move_both_delta")
            trace.stage(
                "primary_plan", ok=(ok or dense_stall is not None), physics_steps=physics_steps,
                planner_status=(planner_diagnostics if planner_status_available else None),
            )
            trace.stop("stalled" if stalled else "converged" if ok else "leg_plan_refused")
            result = ActionResult(
                action_id=aid,
                commanded=commanded,
                achieved={"sync": {"mode": "simultaneous",
                                   "plan_success": bool(data.get("plan_success", ok)),
                                   "reason": reason,
                                   "failure_category": ("stalled" if stalled else None if ok
                                                        else "paired_plan_failure"),
                                   "failure_stage": ("motion_execution" if stalled else
                                                     None if ok else "paired_plan"),
                                   "planner_status": (planner_diagnostics
                                                      if planner_status_available else None),
                                   "planner_status_available": planner_status_available},
                          **stall_contact_evidence},
                status=status,
                abort_reason="stalled" if stalled else None,
                resulting_pose={"left_tcp": tcp1["left"], "right_tcp": tcp1["right"]},
                tick=self._tick,
                planning={"status": "SUCCEEDED" if ok or stalled else "FAILED"},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": "motion_execution" if stalled else "planning",
                    "code": "MOTION_STALLED" if stalled else "PAIRED_PLAN_FAILED",
                    "message": str(reason),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=result.status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release_pair("left", "right")

    @_observed_action
    def reach_tcp(self, arm, target_xyz, target_quat=None):
        """Transport the nominal TCP/grasp centre to a world target.

        The primitive tries one full plan and then chunked waypoint plans after refusal; completed
        transport can receive at most two orientation-holding correction plans. These are
        planner-selected trajectories, not promised Cartesian lines. Public success requires the
        measured TCP to complete the transport. target_quat None keeps the call-start orientation. Workspace violations are
        rejected before simulation execution (ABORTED, workspace).
        """
        aid = self._aid()
        ok, msg = self._pu.check_workspace_bounds(list(target_xyz))
        if not ok:
            tcp = self._tcp(arm)
            return ActionResult(action_id=aid,
                                commanded={"arm": arm, "target_xyz": list(target_xyz)},
                                achieved={"quat_mode": ("explicit" if target_quat is not None
                                                        else "kept_current"),
                                          "failure_stage": "workspace_check",
                                          "planner_detail": str(msg),
                                          "guard": self._guard_record(
                                              "workspace", target_xyz=list(target_xyz),
                                              workspace_message=str(msg)),
                                          "failure_category": "workspace"},
                                status="ABORTED",
                                abort_reason=f"workspace: {msg}",
                                execution={"status": "NOT_STARTED",
                                           "post_state_observed": True,
                                           "state_changed": False,
                                           "physics_steps": 0,
                                           "partial": False},
                                resulting_pose={"tcp": tcp}, tick=self._tick)
        if not self._lock.acquire(arm):
            tcp = self._tcp(arm)
            return ActionResult(action_id=aid,
                                commanded={"arm": arm, "target_xyz": list(target_xyz)},
                                achieved={"quat_mode": ("explicit" if target_quat is not None
                                                        else "kept_current"),
                                          "failure_stage": "arm_lock",
                                          "planner_detail": "another arm command is active",
                                          "failure_category": "arm_lock"},
                                status="ABORTED",
                                abort_reason="arm_lock",
                                resulting_pose={"tcp": tcp}, tick=self._tick,
                                execution=execution_block(ExecutionEvidence(
                                    attempted=False, physics_steps=0, state_changed=False,
                                    interrupted=False, completed=False,
                                    post_state_observed=tcp is not None)))
        try:
            tcp0 = self._tcp(arm)
            if not tcp0:
                return ActionResult(
                    action_id=aid,
                    commanded={"arm": arm, "target_xyz": list(target_xyz),
                               "target_quat": (list(target_quat)
                                               if target_quat is not None else None)},
                    achieved={"quat_mode": ("explicit" if target_quat is not None
                                            else "kept_current"),
                              "reason": "no tcp read", "failure_stage": "pre_read",
                              "failure_category": "pre_read"},
                    status="FAILED", abort_reason=None, resulting_pose={"tcp": None},
                    tick=self._tick,
                    execution=execution_block(ExecutionEvidence(
                        attempted=False, physics_steps=0, state_changed=False,
                        interrupted=False, completed=False, post_state_observed=False)))
            q_anchor = list(target_quat) if target_quat is not None else list(tcp0[3:])
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint((arm,))
            planner_cursor = self._planner_path_cursor()
            r, dense_stall = self._run_with_dense_progress(
                {arm: list(target_xyz) + q_anchor},
                lambda: self._m.reach_tcp(
                    self._env, arm, list(target_xyz),
                    list(target_quat) if target_quat is not None else None))
            r = r or {}
            self._raise_if_motion_interrupted()
            ok = r.get("status") == self._OK
            r_data = r.get("data") or {}
            transport_entries = self._planner_entries_since(planner_cursor).get(arm, [])
            n_legs_attempted = int(r_data.get("n_legs_attempted") or 0)
            trace = MotionTrace("reach_tcp")

            def entry_ok(entry):
                if not isinstance(entry, dict):
                    return None
                status_text = str(entry.get("status") or "").strip().lower()
                if status_text == "success":
                    return True
                if status_text:
                    return False
                raw = entry.get("curobo_status")
                return None if raw is None else _planner_status.normalize_planner_status(raw) is None

            primary_entry = transport_entries[0] if transport_entries else None
            primary_diag = _planner_status.planner_diagnostic(primary_entry)
            primary_ok = (False if n_legs_attempted else
                          (entry_ok(primary_entry) if primary_entry is not None else bool(ok)))
            trace.stage("primary_plan", ok=primary_ok, planner_status=primary_diag)
            transport_diagnostics = [primary_diag] if primary_diag is not None else []
            failed_leg_index = r_data.get("failed_leg_index")
            for index in range(1, n_legs_attempted + 1):
                entry = (transport_entries[index]
                         if index < len(transport_entries) else None)
                leg_diag = _planner_status.planner_diagnostic(entry)
                if leg_diag is not None:
                    transport_diagnostics.append(leg_diag)
                leg_ok = entry_ok(entry)
                if leg_ok is None:
                    leg_ok = failed_leg_index != index
                trace.stage("chunk_leg", ok=leg_ok, index=index,
                            planner_status=leg_diag)
                trace.leg(index=index, plan_ok=leg_ok, executed=leg_ok)

            # Bounded convergence correction: the original primitive receives the caller's raw
            # optional quaternion, while corrections hold the CALL-START orientation anchor.
            corrections_attempted = 0
            corrections_executed = 0
            correction_refused = False
            if ok and dense_stall is None:
                for _ in range(_mv.REACH_CORRECTION_MAX_ATTEMPTS):
                    cur = self._tcp(arm)
                    err = float(np.linalg.norm(np.asarray(cur[:3], float)
                                               - np.asarray(target_xyz, float))) if cur else None
                    if err is None or err < REACH_CORRECTION_TRIGGER_M:
                        break
                    corrections_attempted += 1
                    correction_steps_before = self._physics_step()
                    correction_cursor = self._planner_path_cursor()
                    corrected, correction_stall = self._run_with_dense_progress(
                        {arm: list(target_xyz) + q_anchor},
                        lambda: self._straight_step(arm, list(target_xyz), q_anchor))
                    corrected = bool(corrected)
                    correction_steps = self._physics_delta(
                        correction_steps_before, self._physics_step())
                    correction_entries = self._planner_entries_since(
                        correction_cursor).get(arm, [])
                    correction_diag = next((
                        diagnostic for diagnostic in (
                            _planner_status.planner_diagnostic(entry)
                            for entry in reversed(correction_entries))
                        if diagnostic is not None), None)
                    if correction_diag is None and not corrected:
                        correction_diag = getattr(self, "_last_planner_diag", None)
                    trace.stage("correction", ok=(corrected or correction_stall is not None),
                                index=corrections_attempted,
                                physics_steps=correction_steps,
                                planner_status=correction_diag)
                    if correction_stall is not None:
                        dense_stall = correction_stall
                        break
                    if not corrected:
                        correction_refused = True
                        break
                    corrections_executed += 1
            tcp1 = self._tcp(arm)
            c1 = self._contact(arm)
            final_error = (float(np.linalg.norm(np.asarray(tcp1[:3], float)
                                                - np.asarray(target_xyz, float)))
                           if tcp1 else None)
            stalled = bool(dense_stall is not None or (
                ok
                and final_error is not None
                and final_error >= REACH_CORRECTION_TRIGGER_M
                and (correction_refused or
                     corrections_attempted >= _mv.REACH_CORRECTION_MAX_ATTEMPTS)))
            if dense_stall is not None:
                stop_condition = "stalled"
            elif not ok:
                stop_condition = "leg_plan_refused"
            elif correction_refused:
                stop_condition = "stalled"
            elif stalled:
                stop_condition = "correction_budget_exhausted"
            elif final_error is None:
                stop_condition = "no_tcp_read"
            else:
                stop_condition = "converged"
            trace.stop(stop_condition)
            trace.set_metadata(
                n_legs=r_data.get("n_legs"),
                n_legs_attempted=n_legs_attempted,
                moved_m=r_data.get("moved_m"),
                target_distance_remaining_m=r_data.get("target_distance_remaining_m"),
                corrections_attempted=corrections_attempted,
                corrections_executed=corrections_executed,
            )
            physics_steps = self._physics_delta(steps_before, self._physics_step())
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint((arm,)))
            completed = bool(ok and not correction_refused and not stalled
                             and final_error is not None)
            evidence = ExecutionEvidence(
                attempted=True, physics_steps=physics_steps, state_changed=state_changed,
                interrupted=stalled, completed=completed,
                post_state_observed=tcp1 is not None,
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe(arm)
            planner_detail = (r_data.get("planner_detail") or r.get("details") or
                              r.get("message") or
                              "the planner did not return a trajectory; inspect "
                              "achieved.planner_status for the normalized refusal reason when "
                              "available")
            if stalled:
                reason = ("stalled (target remained after the completed transport and no "
                          "correction could execute)" if correction_refused else
                          "stalled (target remained after bounded corrections)")
                status, abort_reason = "ABORTED", "stalled"
            elif final_error is None:
                reason = "no tcp read"
                status, abort_reason = "FAILED", None
            else:
                reason = None if ok else str(planner_detail)
                status, abort_reason = ("SUCCESS", None) if ok else ("FAILED", None)
            planner_diagnostic = (transport_diagnostics[-1]
                                  if transport_diagnostics and not ok else None)
            failure_fields = self._straight_failure_fields(reason)
            if not ok and not stalled:
                failure_fields = {
                    "planner_detail": str(planner_detail),
                    "failure_category": r_data.get("failure_category"),
                    "failure_stage": r_data.get("failure_stage"),
                }
            stall_contact_evidence = (
                self._stall_contact_evidence(arm) if stalled else {})
            result = ActionResult(
                action_id=aid,
                commanded={"arm": arm, "target_xyz": list(target_xyz),
                           "target_quat": list(target_quat) if target_quat is not None else None,
                           "target_provenance": self._target_provenance(target_xyz)},
                achieved={"contact": c1,
                          "reason": reason,
                          "quat_mode": r_data.get("quat_mode", "explicit" if target_quat is not None
                                                  else "kept_current"),
                          "planner_detail": (failure_fields.get("planner_detail")
                                             if status != "SUCCESS" else None),
                          "failure_category": (failure_fields.get("failure_category")
                                               if status != "SUCCESS" else None),
                          "failure_stage": (failure_fields.get("failure_stage")
                                            if status != "SUCCESS" else None),
                          "planner_status": planner_diagnostic,
                          "guard": self._guard_record(abort_reason),
                          **stall_contact_evidence},
                status=status,
                abort_reason=abort_reason,
                resulting_pose={"tcp": tcp1}, tick=self._tick,
                planning={"status": "SUCCEEDED" if ok or stalled else "FAILED"},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": str(failure_fields.get("failure_stage") or "planning"),
                    "code": ("MOTION_STALLED" if stalled else
                             "PLAN_FAILED"),
                    "message": str(reason),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=result.status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release(arm)

    @_observed_action
    def reach_both_tcp(self, left_xyz, right_xyz, left_quat=None, right_quat=None):
        """Synchronized dual-arm nominal-TCP transport in one paired primitive call.

        A failed paired call may already have moved one or both arms; execution evidence and both
        post-state snapshots carry that fact.
        """
        aid = self._aid()
        left_xyz, right_xyz = list(left_xyz), list(right_xyz)
        commanded = {"left_xyz": left_xyz, "right_xyz": right_xyz,
                     "left_quat": list(left_quat) if left_quat is not None else None,
                     "right_quat": list(right_quat) if right_quat is not None else None}
        for side, xyz in (("left", left_xyz), ("right", right_xyz)):
            ok, msg = self._pu.check_workspace_bounds(list(xyz))
            if not ok:
                left_tcp, right_tcp = self._tcp("left"), self._tcp("right")
                return ActionResult(
                    action_id=aid,
                    commanded=commanded,
                    achieved={"sync": {"mode": "simultaneous",
                                       "failure_stage": "workspace_check",
                                       "failure_category": "workspace",
                                       "failed_arm": side,
                                       "reason": str(msg),
                                       "planner_status_available": False}},
                    status="ABORTED",
                    abort_reason=f"workspace: {side}: {msg}",
                    resulting_pose={"left_tcp": left_tcp, "right_tcp": right_tcp},
                    tick=self._tick,
                    execution=execution_block(ExecutionEvidence(
                        attempted=False, physics_steps=0, state_changed=False,
                        interrupted=False, completed=False,
                        post_state_observed=bool(left_tcp and right_tcp))))
        if not self._lock.acquire_pair("left", "right"):
            left_tcp, right_tcp = self._tcp("left"), self._tcp("right")
            return ActionResult(action_id=aid, commanded=commanded,
                                achieved={"sync": {"mode": "simultaneous",
                                                   "failure_stage": "arm_lock",
                                                   "failure_category": "arm_lock",
                                                   "reason": "another arm command is active",
                                                   "planner_status_available": False}},
                                status="ABORTED", abort_reason="arm_lock",
                                resulting_pose={"left_tcp": left_tcp,
                                                "right_tcp": right_tcp},
                                tick=self._tick,
                                execution=execution_block(ExecutionEvidence(
                                    attempted=False, physics_steps=0, state_changed=False,
                                    interrupted=False, completed=False,
                                    post_state_observed=bool(left_tcp and right_tcp))))
        try:
            tcp0 = {"left": self._tcp("left"), "right": self._tcp("right")}
            if not tcp0["left"] or not tcp0["right"]:
                return ActionResult(action_id=aid, commanded=commanded,
                                    achieved={"sync": {"mode": "simultaneous",
                                                       "failure_stage": "pre_read",
                                                       "failure_category": "pre_read",
                                                       "reason": "no tcp read",
                                                       "planner_status_available": False}},
                                    status="FAILED", abort_reason=None,
                                    resulting_pose={"left_tcp": tcp0["left"],
                                                    "right_tcp": tcp0["right"]},
                                    tick=self._tick,
                                    execution=execution_block(ExecutionEvidence(
                                        attempted=False, physics_steps=0, state_changed=False,
                                        interrupted=False, completed=False,
                                        post_state_observed=False)))
            steps_before = self._physics_step()
            fingerprint_before = self._state_fingerprint(("left", "right"))
            planner_cursor = self._planner_path_cursor()
            targets = {
                "left": left_xyz + (list(left_quat) if left_quat is not None
                                      else list(tcp0["left"][3:])),
                "right": right_xyz + (list(right_quat) if right_quat is not None
                                        else list(tcp0["right"][3:])),
            }
            r, dense_stall = self._run_with_dense_progress(
                targets,
                lambda: self._m.reach_both_tcp(
                    self._env, left_xyz, right_xyz,
                    list(left_quat) if left_quat is not None else None,
                    list(right_quat) if right_quat is not None else None))
            r = r or {}
            self._raise_if_motion_interrupted()
            ok = r.get("status") == self._OK
            data = r.get("data") or {}
            tcp1 = {"left": self._tcp("left"), "right": self._tcp("right")}
            stalled_arms = list(dense_stall.arms) if dense_stall is not None else [
                arm for arm in ("left", "right")
                if ok and self._target_error_m(tcp1[arm], targets[arm]) is not None
                and self._target_error_m(tcp1[arm], targets[arm])
                >= REACH_CORRECTION_TRIGGER_M
            ]
            stalled = bool(stalled_arms)
            stall_contact_evidence = (
                self._paired_stall_evidence(stalled_arms) if stalled else {})
            physics_steps = self._physics_delta(steps_before, self._physics_step())
            state_changed = _fingerprint_changed(
                fingerprint_before, self._state_fingerprint(("left", "right")))
            evidence = ExecutionEvidence(
                attempted=True, physics_steps=physics_steps, state_changed=state_changed,
                interrupted=stalled,
                completed=bool(ok and not stalled and tcp1["left"] and tcp1["right"]),
                post_state_observed=bool(tcp1["left"] and tcp1["right"]),
            )
            self._advance_tick_if_executed(evidence)
            if evidence.advanced:
                self._observe_both()
            planner_diagnostics = self._planner_diagnostics_from_entries(
                self._planner_entries_since(planner_cursor))
            planner_status_available = any(
                value is not None for value in planner_diagnostics.values())
            reason = ("stalled before one or both commanded targets were reached" if stalled
                      else None if ok else str(r.get("details") or r.get("message") or
                                              "dual-arm TCP motion planning failed"))
            status = "ABORTED" if stalled else "SUCCESS" if ok else "FAILED"
            trace = MotionTrace("reach_both_tcp")
            trace.stage(
                "primary_plan", ok=(ok or dense_stall is not None), physics_steps=physics_steps,
                planner_status=(planner_diagnostics if planner_status_available else None),
            )
            trace.stop("stalled" if stalled else "converged" if ok else "leg_plan_refused")
            result = ActionResult(
                action_id=aid,
                commanded=commanded,
                achieved={"sync": {"mode": "simultaneous",
                                   "plan_success": bool(data.get("plan_success", ok)),
                                   "reason": reason,
                                   "failure_category": ("stalled" if stalled else None if ok
                                                        else "paired_plan_failure"),
                                   "failure_stage": ("motion_execution" if stalled else
                                                     None if ok else "paired_plan"),
                                   "planner_status": (planner_diagnostics
                                                      if planner_status_available else None),
                                   "planner_status_available": planner_status_available},
                          **stall_contact_evidence},
                status=status,
                abort_reason="stalled" if stalled else None,
                resulting_pose={"left_tcp": tcp1["left"], "right_tcp": tcp1["right"]},
                tick=self._tick,
                planning={"status": "SUCCEEDED" if ok or stalled else "FAILED"},
                execution=execution_block(evidence),
                failure=(None if status == "SUCCESS" else {
                    "stage": "motion_execution" if stalled else "planning",
                    "code": "MOTION_STALLED" if stalled else "PAIRED_PLAN_FAILED",
                    "message": str(reason),
                }))
            self._record_motion_trace(trace.record(
                action_id=aid, tick=self._tick, status=result.status,
                physics_steps=physics_steps, state_changed=state_changed))
            return result
        finally:
            self._lock.release_pair("left", "right")

    def _gripper_action(self, arm, pos):
        aid = self._aid()
        gap_before = self._finger_gap(arm)
        state_before = self._p.get_gripper_state(self._env, arm).get("data") or {}
        drive_before = state_before.get("gripper_val")
        steps_before = self._physics_step()
        # The upstream open/close wrappers both accept the complete normalized 0..1 range and
        # envs.utils.action normalizes either label to the same gripper drive command. Reuse one
        # wrapper internally; the public surface exposes the actual operation as set_gripper.
        r = self._g.open_gripper(self._env, arm, pos=float(pos))
        st = self._p.get_gripper_state(self._env, arm).get("data") or {}
        gp = self._p.get_gripper_pose(self._env, arm).get("data") or {}
        contact = self._contact(arm)
        actuation_completed = r.get("status") == self._OK
        drive_value = st.get("gripper_val")
        drive_error = (abs(float(drive_value) - float(pos))
                       if drive_value is not None else None)
        gap_after = self._finger_gap(arm)
        gap_changed = (None if gap_before is None or gap_after is None
                       else abs(float(gap_after) - float(gap_before)) > 1e-6)
        drive_changed = (None if drive_before is None or drive_value is None
                         else abs(float(drive_value) - float(drive_before)) > 1e-9)
        comparable_changes = [value for value in (gap_changed, drive_changed)
                              if value is not None]
        state_changed = (any(comparable_changes) if comparable_changes else None)
        physics_steps = self._physics_delta(steps_before, self._physics_step())
        evidence = ExecutionEvidence(
            attempted=True, physics_steps=physics_steps, state_changed=state_changed,
            interrupted=False, completed=actuation_completed,
            post_state_observed=gap_after is not None or drive_value is not None,
        )
        self._advance_tick_if_executed(evidence)
        if evidence.advanced:
            self._observe(arm)
        # A subtraction, named for what it computed. Calling it `object_width_m` would assert that
        # something IS between the fingers and that it is being held square-on — neither of which
        # the robot can know. The reader combines it with the contact report and decides.
        empty_close = (_emb.get_embodiment().get("gripper") or {}).get(
            "empty_close_finger_gap_m")
        gap_minus_empty_close = (
            float(gap_after) - float(empty_close)
            if gap_after is not None and empty_close is not None else None)
        result = ActionResult(
            action_id=aid, commanded={"arm": arm, "pos": float(pos)},
            achieved={"gripper_val": drive_value,
                      "drive_error": drive_error,
                      "opening_m": gp.get("gripper_width_m"),
                      "finger_gap_m": gap_after,
                      "finger_gap_minus_empty_close_m": gap_minus_empty_close,
                      "empty_close_finger_gap_m": empty_close,
                      "drive_commanded_closed": st.get("is_closed"),
                      "contact": contact},
            status="SUCCESS" if actuation_completed else "FAILED",
            abort_reason=None, resulting_pose={"tcp": self._tcp(arm)}, tick=self._tick,
            planning={"status": "NOT_APPLICABLE"},
            execution=execution_block(evidence),
            failure=(None if actuation_completed else {
                "stage": "execution", "code": "GRIPPER_ACTUATION_FAILED",
                "message": "gripper primitive did not complete",
            }))
        trace = MotionTrace("set_gripper")
        trace.stage("gripper_actuation", ok=actuation_completed,
                    physics_steps=physics_steps)
        trace.stop("converged" if actuation_completed else "actuation_failed")
        self._record_motion_trace(trace.record(
            action_id=aid, tick=self._tick, status=result.status,
            physics_steps=physics_steps, state_changed=state_changed))
        return result

    @_observed_action
    def set_gripper(self, arm, pos):
        """Set normalized gripper drive position; 1.0 is open and 0.0 is closed.

        Tick advances only when execution evidence shows real actuation.
        """
        return self._gripper_action(arm, pos)

    def camera_aim_pose(self, camera, target_xyz, pitch=None, standoff=None):
        """Compute one caller-configured wrist-camera TCP pose without planning or motion.

        ``pitch`` is the world-Y pre-rotation applied to the embodiment's fixed base TCP
        orientation; it is not a virtual-camera rotation.
        """
        if camera not in ("left_camera", "right_camera"):
            raise ValueError("camera_aim_pose supports left_camera/right_camera; "
                             "head_camera is fixed")
        side = camera.split("_")[0]
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be three finite world coordinates")
        pitch_deg = _aim.validated_aim_pitch(
            _aim.DEFAULT_PITCH_DEG if pitch is None else pitch)
        standoff_m = float(_aim.DEFAULT_STANDOFF_M if standoff is None else standoff)
        if not np.isfinite(standoff_m) or standoff_m <= 0:
            raise ValueError("standoff must be a positive finite number")

        tcp_pose = self._tcp(side)
        common = {
            "valid": False,
            "camera": camera,
            "arm": side,
            "target_xyz": target.tolist(),
            "target_provenance": self._target_provenance(target.tolist()),
            "pitch_deg": pitch_deg,
            "standoff_m": standoff_m,
            "source_tick": self._tick,
            "current_tcp_pose_world": tcp_pose,
            "target_tcp_pose_world": None,
            "predicted_projection": None,
        }
        if not tcp_pose or len(tcp_pose) != 7:
            return {**common, "reason": f"could not read current {side} TCP pose",
                    "note": "No planner query or robot motion occurred."}

        K, E, _ = self._kec(camera)
        image_size = self._vp.backend.get_image_size(camera)
        computed = _aim.compute_camera_aim_pose(
            target_xyz=target.tolist(), pitch=pitch_deg, standoff=standoff_m,
            tcp_pose=tcp_pose, camera_K=K, camera_E=E, image_size_hw=image_size,
            quat_mul=self._pu._quat_mul_wxyz, axis_angle_quat=self._axisq,
            quat_to_rotation=self._pu._quat_wxyz_to_rotmat,
            down_quat=np.asarray(self._pu.DOWN_QUAT_WXYZ, dtype=float))
        return {
            **common,
            "valid": True,
            "target_tcp_pose_world": computed["target_tcp_pose_world"],
            "predicted_projection": computed["predicted_projection"],
            "reason": "one camera aim pose computed",
            "note": (
                "Pure geometry only: no candidate selection, planner query, robot motion, "
                "collision verdict, RGB visibility verdict, or task-success verdict."),
        }

    # ── runner surface ────────────────────────────────────────────────────
