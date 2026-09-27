"""Read-only queries about the robot and the scene as the model may see them."""
from codeaction.interface.tools._base import (
    CAMERA_CONVENTION_VALUE,
    ContactReadUnavailable,
    VALID_CAMERAS,
    _WF_AXIS_COLORS,
    _emb,
    _mv,
    _orientation,
    _planner_status,
    _wf,
    np)


class SensingTools:
    def get_world_frame(self):
        """Return the canonical world-axis card on a fresh head view.

        The corner glyph is screen-space and non-metric.  It labels the card's axes but is not a
        projection of a world point and provides no world-to-pixel correspondence.
        """
        card = _wf.get_world_frame()
        base = self._capture_camera("head_camera")
        from PIL import Image, ImageDraw, ImageFont
        im = Image.open(base.image_ref).convert("RGB")
        dr = ImageDraw.Draw(im)
        width, height = im.size
        wf_axes = card["axes"]

        def in_frame(p):
            return 0 <= p[0] < width and 0 <= p[1] < height

        def arrow(start, end, color, line_width=4, head_len=10, head_half=5):
            a, b = np.asarray(start, float), np.asarray(end, float)
            direction = b - a
            norm = float(np.linalg.norm(direction))
            if norm < 1.0:
                return False
            direction /= norm
            normal = np.asarray([-direction[1], direction[0]])
            dr.line([tuple(a), tuple(b)], fill=color, width=line_width)
            dr.polygon([tuple(b),
                        tuple(b - head_len * direction + head_half * normal),
                        tuple(b - head_len * direction - head_half * normal)], fill=color)
            return True

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=11)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", size=11)
            except OSError:
                try:
                    font = ImageFont.load_default(size=11)
                except TypeError:
                    font = ImageFont.load_default()

        occupied = []

        def badge(at, text, color, offsets):
            box0 = dr.textbbox((0, 0), text, font=font, stroke_width=1)
            bw, bh = box0[2] - box0[0] + 6, box0[3] - box0[1] + 6
            chosen = None
            for dx, dy in offsets:
                x = min(max(1.0, float(at[0]) + dx), max(1.0, width - bw - 1.0))
                y = min(max(1.0, float(at[1]) + dy), max(1.0, height - bh - 1.0))
                candidate = (x, y, x + bw, y + bh)
                if not any(not (candidate[2] + 2 < q[0] or q[2] + 2 < candidate[0]
                                   or candidate[3] + 2 < q[1] or q[3] + 2 < candidate[1])
                           for q in occupied):
                    chosen = candidate
                    break
            if chosen is None:
                chosen = candidate
            occupied.append(chosen)
            x, y = chosen[:2]
            dr.rectangle(chosen, fill=color, outline="white", width=1)
            dr.text((x + 3 - box0[0], y + 3 - box0[1]), text, font=font, fill="white",
                    stroke_width=1, stroke_fill=(0, 0, 0))
            return [int(round(value)) for value in chosen]

        # A conventional corner glyph, sized only in pixels.  Its directions and lengths are
        # deliberately independent of K, E, scene geometry, and physical units.
        axis_px = max(8, min(28, int(round(min(width, height) * 0.24))))
        margin_px = max(4, min(12, int(round(min(width, height) * 0.10))))
        origin_px = [margin_px + axis_px, margin_px + axis_px]
        endpoints = {
            "x": [origin_px[0] + axis_px, origin_px[1]],
            "y": [origin_px[0] - int(round(0.70 * axis_px)),
                  origin_px[1] + int(round(0.55 * axis_px))],
            "z": [origin_px[0], origin_px[1] - axis_px],
        }
        attempted_axes = []
        for ax in ("x", "y", "z"):
            attempted_axes.append(f"+{ax}")
            arrow(origin_px, endpoints[ax], _WF_AXIS_COLORS[ax])
        if in_frame(origin_px):
            u, v = int(origin_px[0]), int(origin_px[1])
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill="white",
                       outline=(20, 20, 20), width=2)

        axis_offsets = {"x": [(6, -7), (-28, -7), (6, 4)],
                        "y": [(-20, -18), (5, -7), (-20, 4)],
                        "z": [(5, -16), (-24, -16), (5, 2)]}
        label_boxes = {
            ax: badge(endpoints[ax], f"+{ax.upper()}", _WF_AXIS_COLORS[ax], axis_offsets[ax])
            for ax in ("x", "y", "z")
        }

        def box_in_frame(box):
            return 0 <= box[0] <= box[2] < width and 0 <= box[1] <= box[3] < height

        endpoint_visibility = {f"+{ax}": bool(in_frame(endpoints[ax]))
                               for ax in ("x", "y", "z")}
        label_visibility = {f"+{ax}": bool(box_in_frame(label_boxes[ax]))
                            for ax in ("x", "y", "z")}
        origin_visibility = bool(in_frame(origin_px))

        color_names = {"x": "red", "y": "green", "z": "blue"}
        return self._annotated(base, im, {
            "tool": "get_world_frame",
            "axes": card["axes"], "units": card["units"], "prompt_text": card["prompt_text"],
            "legend": {f"+{ax}": f"{wf_axes['+' + ax]} ({name})"
                       for ax, name in color_names.items()},
            "rendering_space": "screen_space",
            "metric_scale": False,
            "world_to_pixel_correspondence": False,
            "display_note": "Pixel glyph only; arrow angles and lengths do not encode camera "
                            "projection, world position, depth, or physical scale.",
            "render_attempted_axes": attempted_axes,
            "glyph": {
                "origin_px": [int(value) for value in origin_px],
                "endpoints_px": {f"+{ax}": [int(value) for value in endpoints[ax]]
                                 for ax in ("x", "y", "z")},
                "label_boxes_xyxy": {f"+{ax}": label_boxes[ax]
                                     for ax in ("x", "y", "z")},
                "visibility": {
                    "origin_in_frame": origin_visibility,
                    "endpoints_in_frame": endpoint_visibility,
                    "labels_in_frame": label_visibility,
                    "complete_overlay": bool(
                        origin_visibility and all(endpoint_visibility.values())
                        and all(label_visibility.values())),
                },
            },
        })

    def get_embodiment(self):
        return _emb.get_embodiment()

    def get_camera_info(self, camera):
        if camera not in VALID_CAMERAS:
            raise ValueError(f"camera must be one of {VALID_CAMERAS}")
        result = {
            "camera": camera,
            "available": False,
            "K": None,
            "E": None,
            "size_hw": None,
            "freshness": "unavailable",
            "read_failures": [],
            "convention": dict(CAMERA_CONVENTION_VALUE),
            "tick": self._tick,
            "note": ("Legal-camera read failure is nonfatal; use each Observation snapshot for "
                     "its image geometry."),
        }
        try:
            self._refresh_camera_strict()
        except Exception:
            result["read_failures"] = ["camera_refresh_failed"]
            return result

        result["freshness"] = "call_time_current"
        try:
            K, E = self._read_camera_matrices(camera)
        except Exception:
            result["read_failures"].append("camera_matrices_unavailable")
        else:
            result["K"] = K.tolist()
            result["E"] = E.tolist()
        try:
            result["size_hw"] = self._read_rgb_size(camera)
        except Exception:
            result["read_failures"].append("camera_rgb_size_unavailable")
        result["available"] = not result["read_failures"]
        return result

    # A number is "ours" only if it is ours written down: equal, or equal after rounding to
    # millimetre precision or finer. Coarser rounding is excluded — at centimetre precision a
    # match stops meaning "you copied my value" and starts meaning "you picked a round number".
    _ECHO_DECIMALS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    _ECHO_MIN_AXES = 2

    @classmethod
    def _is_written_echo(cls, ours, theirs):
        """True when `theirs` is `ours` transcribed, possibly with fewer decimals."""
        if theirs == ours:
            return True
        return any(round(ours, d) == theirs for d in cls._ECHO_DECIMALS)

    def _register_unverified_point(self, xyz, tool, unverified_inputs):
        """Remember a world point we derived from an input the caller asserted rather than measured.

        The conditionality of such a point does not survive being passed on: the model receives
        three floats and hands them to a motion tool, and by then nothing records that they rest on
        an unverified assumption. Registering the value here lets the consuming tool say so.
        """
        ledger = getattr(self, "_unverified_points", None)
        if ledger is None:
            ledger = self._unverified_points = []
        ledger.append({"value": [float(v) for v in xyz], "derived_by": tool,
                       "inputs_unverified": list(unverified_inputs),
                       "tick": getattr(self, "_tick", None)})
        del ledger[:-32]

    def _target_provenance(self, xyz):
        """Report when a commanded target carries axes we ourselves derived from an assumption.

        Matching is transcription, never proximity: an axis counts only when the commanded number
        IS our number, possibly written with fewer decimals (see `_is_written_echo`). A target is
        rarely copied whole — the axes that come from us and the axes the caller replaced are
        reported separately, because they do not have the same standing as evidence. This states
        where the numbers came from and nothing about whether the target is a good one.
        """
        if not xyz:
            return None
        try:
            target = [float(v) for v in xyz]
        except (TypeError, ValueError):
            return None
        if len(target) != 3:
            return None
        names = ("x", "y", "z")
        for item in reversed(getattr(self, "_unverified_points", []) or []):
            hit = [i for i in range(3) if self._is_written_echo(item["value"][i], target[i])]
            if len(hit) < self._ECHO_MIN_AXES:
                continue
            from_us = [names[i] for i in hit]
            replaced = [n for n in names if n not in from_us]
            out = {"derived_by": item["derived_by"],
                   "axes_from_that_value": from_us,
                   "inputs_unverified": list(item["inputs_unverified"]),
                   "note": f"the {'/'.join(from_us)} of this target {'is' if len(from_us) == 1 else 'are'} "
                           f"a value this harness derived from an input you asserted rather than "
                           f"measured; it was not verified against the scene"}
            if replaced:
                out["axes_you_supplied"] = replaced
                out["note"] += (f"; the {'/'.join(replaced)} did not come from that value and this "
                                f"harness has no record of where it came from")
            return out
        return None

    def _pose_axes(self, pose7):
        """Decode a reported pose's quaternion into world axes (see codeaction.motion.orientation)."""
        if not pose7 or len(pose7) != 7:
            return None
        try:
            return _orientation.decode_pose_axes(list(pose7)[3:])
        except Exception:
            return None

    def get_arm_pose(self, arm):
        pose = self._arm_pose_read(arm)
        return {"arm": arm, "ee_pose": pose["ee_pose"], "tcp_pose": pose["tcp_pose"],
                "orientation": pose["orientation"],
                "read_failures": pose["read_failures"],
                "tick": self._tick,
                "note": ("nominal TCP grasp centre; pose=[x,y,z,qw,qx,qy,qz]; orientation "
                         "decodes the quaternion")}

    def get_gripper_state(self, arm):
        state = self._gripper_state_read(arm)
        return {"arm": arm, "opening_m": state["opening_m"],
                "finger_gap_m": state["finger_gap_m"],
                "gripper_val": state["gripper_val"],
                "drive_commanded_closed": state["drive_commanded_closed"],
                "read_failures": state["read_failures"],
                "tick": self._tick,
                "note": ("finger_gap_m is physical FK; opening_m/gripper_val are drive-derived "
                         "and may track the command when blocked")}

    def get_robot_state(self, arms=None):
        """Batched raw robot self-state for one or both arms."""
        if arms is None:
            arms = ["left", "right"]
        if isinstance(arms, str):
            arms = [arms]
        selected, seen = [], set()
        for arm in arms:
            if arm in seen:
                continue
            if arm not in ("left", "right"):
                raise ValueError("arms entries must be 'left' or 'right'")
            seen.add(arm)
            selected.append(arm)
        snapshot = self._robot_snapshot(include_joint_state=True)
        snapshot["arms"] = {arm: snapshot["arms"][arm] for arm in selected}
        snapshot["note"] = (
            "robot self-state only; unavailable reads are null and named in read_failures; "
            "no scene identity, depth, or task score. This is the only place joint_state is "
            "returned: motion results carry the boundary snapshot without it")
        return snapshot

    def grasp_quat_candidates(self, approach_axis_world, opening_axis_world):
        """Pure conversion of caller-selected semantic grasp axes to wxyz orientations."""
        return _orientation.grasp_quat_candidates(approach_axis_world, opening_axis_world)

    def check_tcp_pose_reachability(self, arm, target_xyz, target_quat=None):
        """Run one self-collision-aware planner query without executing its trajectory."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must be three finite world coordinates")
        pose = self._p.get_gripper_pose(self._env, arm).get("data") or {}
        ee_pose, tcp_pose = pose.get("pose"), pose.get("tcp_pose")
        if not ee_pose or not tcp_pose or len(ee_pose) != 7 or len(tcp_pose) != 7:
            raise ValueError(f"could not read current {arm} EE/TCP pose")
        if target_quat is None:
            quat = np.asarray(tcp_pose[3:], dtype=float)
            quat_mode = "kept_current"
        else:
            quat = np.asarray(target_quat, dtype=float)
            quat_mode = "explicit"
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError("target_quat must be four finite wxyz values")
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm < 1e-9:
            raise ValueError("target_quat must be non-zero")
        quat /= quat_norm

        in_bounds, bounds_note = self._pu.check_workspace_bounds(target.tolist())
        if not in_bounds:
            return {
                "arm": arm,
                "target_xyz": target.tolist(),
                "target_quat": quat.tolist(),
                "quat_mode": quat_mode,
                "planner_found_trajectory": False,
                "planner_status": "OUT_OF_WORKSPACE",
                "trajectory_sample_count": 0,
                "interpolation_dt_s": _mv.PLANNER_INTERPOLATION_DT_S,
                "planner_diagnostic": None,
                "stage": "workspace_prefilter",
                "state_unchanged": True,
                "tick": self._tick,
                "reason": str(bounds_note),
            }

        entity = (self._env.robot.left_entity if arm == "left"
                  else self._env.robot.right_entity)
        planner = (self._env.robot.left_plan_path if arm == "left"
                   else self._env.robot.right_plan_path)
        qpos_before = np.asarray(entity.get_qpos(), dtype=float)
        tcp_before = np.asarray(tcp_pose, dtype=float)
        tick_before = self._tick
        ee_target = self._pu._tcp_target_to_ee_target(
            ee_pose, tcp_pose, target.tolist(), quat.tolist())
        try:
            planned = planner(ee_target)
            planner_error = None
        except Exception as exc:
            planned = {}
            planner_error = f"{type(exc).__name__}: {exc}"

        qpos_after = np.asarray(entity.get_qpos(), dtype=float)
        tcp_after_raw = self._tcp(arm)
        tcp_after = np.asarray(tcp_after_raw, dtype=float) if tcp_after_raw else None
        state_unchanged = (self._tick == tick_before
                           and np.allclose(qpos_before, qpos_after, atol=1e-10, rtol=0)
                           and tcp_after is not None
                           and np.allclose(tcp_before, tcp_after, atol=1e-10, rtol=0))
        if not state_unchanged:
            raise RuntimeError("non-executing planner changed robot state")

        planner_status = planned.get("status") if isinstance(planned, dict) else None
        path = planned.get("position") if isinstance(planned, dict) else None
        trajectory_sample_count = int(np.asarray(path).shape[0]) if path is not None else 0
        found = None if planner_error else planner_status == "Success"
        diagnostic = (_planner_status.planner_diagnostic(planned)
                      if not planner_error and found is False else None)
        return {"arm": arm, "target_xyz": target.tolist(),
                "target_quat": quat.tolist(), "quat_mode": quat_mode,
                "planner_found_trajectory": found,
                "planner_status": planner_status or "ERROR",
                "trajectory_sample_count": trajectory_sample_count,
                "interpolation_dt_s": _mv.PLANNER_INTERPOLATION_DT_S,
                "planner_diagnostic": diagnostic,
                "stage": "planner_exception" if planner_error else "planner_query",
                "state_unchanged": True, "tick": self._tick,
                "reason": planner_error}

    def get_grasp_contact(self, arm):
        """Finger contact impulse, UNFILTERED by object identity (leak fix #1): reports the fingers
        touching ANYTHING, with the forward-kinematics pose of each one that is. Two fingers in
        contact means two anonymous world contacts, not contact with one object and not a grasp;
        `fingers_in_contact` counts the fingers whose impulse exceeds the harness contact
        threshold declared in the initial episode configuration, and `per_finger` carries the
        impulses themselves."""
        common = {"contact_impulse_unit": "N*s", "arm": arm, "tick": self._tick}
        try:
            contact = self._contact(arm)
        except ContactReadUnavailable:
            return {
                "available": False,
                "read_failures": ["contact_unavailable"],
                **common,
                "note": "Contact state was not measured; no zero-contact claim was made.",
            }
        return {
            "available": True,
            "read_failures": [],
            **common,
            **contact,
            "contacting_finger_pose": dict(contact.get("contacting_finger_pose") or {}),
            "contact_localization": str(contact.get("contact_localization") or (
                "No above-threshold contacting finger-link pose was available; an empty map with "
                "fingers_in_contact=0 is a measured empty contact state.")),
            "note": "Anonymous finger-to-world contact measurement; not a grasp verdict.",
        }

    # ── perception ────────────────────────────────────────────────────────
