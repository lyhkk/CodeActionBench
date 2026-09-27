"""Camera capture, image-plane geometry, and the annotated overlays the model reads."""
from codeaction.interface.tools._base import (
    DEFAULT_PIXEL_SIGMA_PX,
    DRAW_MARKS_MAX_COUNT,
    DRAW_MARKS_MIN_COUNT,
    Estimate,
    Observation,
    ObservationPair,
    ObservationSet,
    RULER_MIN_PROJECTED_SPAN_PX,
    VALID_CAMERAS,
    _observed_action,
    _scale,
    _wf,
    np)


class VisionTools:
    def _capture_camera(self, camera):
        """2D RGB frame + K/E snapshot at THIS instant → Observation. The agent states object
        pixels/boxes itself (no detect tool, spec §1)."""
        if camera not in VALID_CAMERAS:
            raise ValueError(f"camera must be one of {VALID_CAMERAS}")
        self._n_obs += 1
        obs_id = f"obs_{self._n_obs:03d}"
        path = self._out / f"{obs_id}_{camera}.png"
        self._env.save_camera_rgb(str(path), camera)
        # save_camera_rgb refreshes before reading RGB. Read K/E only after that succeeds and
        # derive H/W from this exact file rather than the backend's optional depth-based helper.
        K, E = self._read_camera_matrices(camera)
        from PIL import Image
        with Image.open(path) as image:
            width, height = image.size
        o = Observation(obs_id=obs_id, camera=camera, image_ref=str(path),
                        cam_pose_snapshot=self._camera_snapshot(K, E, [height, width]),
                        tick=self._tick)
        self._obs[obs_id] = o
        return o

    def capture_head(self):
        """Fixed head-camera overview."""
        return self._capture_camera("head_camera")

    def _active_arm_or_none(self, active_arm):
        if active_arm in (None, "none"):
            return None
        if active_arm not in ("left", "right"):
            raise ValueError("active_arm must be 'left', 'right', or 'none'")
        return active_arm

    def capture_wrist(self, active_arm, views="active"):
        """Wrist-camera close view(s), labelled by active/opposite role."""
        active_arm = self._active_arm_or_none(active_arm)
        if views not in ("active", "both"):
            raise ValueError("views must be 'active' or 'both'")
        if active_arm is None and views != "both":
            raise ValueError("active_arm='none' is only valid with views='both'")
        if active_arm is None:
            left = self._capture_camera("left_camera")
            right = self._capture_camera("right_camera")
            return ObservationSet(
                set_id=f"obs_set_{self._n_obs:03d}",
                observations=[left, right],
                roles={"left_wrist": left.obs_id, "right_wrist": right.obs_id},
                tick=self._tick)
        active_cam = f"{active_arm}_camera"
        observations = [self._capture_camera(active_cam)]
        roles = {"active_wrist": observations[0].obs_id,
                 f"{active_arm}_wrist": observations[0].obs_id}
        if views == "both":
            other_arm = "right" if active_arm == "left" else "left"
            observations.append(self._capture_camera(f"{other_arm}_camera"))
            roles["opposite_wrist"] = observations[1].obs_id
            roles[f"{other_arm}_wrist"] = observations[1].obs_id
        return ObservationSet(set_id=f"obs_set_{self._n_obs:03d}", observations=observations,
                              roles=roles, tick=self._tick)

    def capture_evidence_views(self, active_arm=None, views=None):
        """Selected same-tick head/wrist camera bundle."""
        active_arm = self._active_arm_or_none(active_arm)
        requested = (["overview", "left_wrist", "right_wrist"]
                     if views is None else views)
        if not isinstance(requested, list):
            raise ValueError("views must be a non-empty list")
        if not requested:
            raise ValueError("views must contain at least one view")
        if any(not isinstance(view, str) for view in requested):
            raise ValueError("views must contain only view-name strings")
        if len(requested) > 3 or len(set(requested)) != len(requested):
            raise ValueError("views must contain 1-3 unique view names")
        cameras = {
            "overview": "head_camera",
            "left_wrist": "left_camera",
            "right_wrist": "right_camera",
        }
        unknown = [view for view in requested if view not in cameras]
        if unknown:
            raise ValueError(f"unknown evidence view(s): {unknown}")
        observations = [self._capture_camera(cameras[view]) for view in requested]
        roles = {view: observation.obs_id
                 for view, observation in zip(requested, observations)}
        if active_arm is not None:
            active_role = f"{active_arm}_wrist"
            opposite_arm = "right" if active_arm == "left" else "left"
            opposite_role = f"{opposite_arm}_wrist"
            if active_role in roles:
                roles["active_wrist"] = roles[active_role]
            if opposite_role in roles:
                roles["opposite_wrist"] = roles[opposite_role]
        return ObservationSet(
            set_id=f"obs_set_{self._n_obs:03d}",
            observations=observations,
            roles=roles,
            tick=self._tick)

    # ── geometry / scale (return Estimate) ────────────────────────────────
    def project(self, obs_id, xyz):
        """World point → pixel in this observation (uses the obs's OWN K/E snapshot)."""
        o = self._get_obs(obs_id)
        val = self._project_pt(o, xyz)
        return Estimate(value=val, kind="point", uncertainty=0.0, coarse=False,
                        provenance={"tool": "project", "obs_id": obs_id, "xyz": list(xyz),
                                    "method": "pinhole projection",
                                    "uncertainty_scope": "transform only",
                                    "uncertainty_excludes": [
                                        "caller world point", "camera calibration"],
                                    "note": "pixel [u,v]; null=behind camera"})

    def ray(self, obs_id, px):
        """Pixel → world ray {origin, dir} of this observation's camera (calibration only)."""
        o = self._get_obs(obs_id)
        from codeaction.backends.robotwin import epipolar as epi
        C = self._cam_centre(o)
        p1 = np.asarray(epi.back_project(o.cam_pose_snapshot["K"], o.cam_pose_snapshot["E"],
                                         [float(px[0]), float(px[1])], 1.0), float)
        d = p1 - C
        d = d / np.linalg.norm(d)
        height, width = [int(v) for v in o.cam_pose_snapshot["size_hw"]]
        u, v = float(px[0]), float(px[1])
        return Estimate(value={"origin": C.tolist(), "dir": d.tolist()}, kind="ray",
                        uncertainty=0.0, coarse=False,
                        provenance={"tool": "ray", "obs_id": obs_id,
                                    "px": [u, v],
                                    "image_size_hw": [height, width],
                                    "pixel_in_frame": 0.0 <= u < width and 0.0 <= v < height,
                                    "method": "pinhole back-projection",
                                    "uncertainty_scope": "transform only",
                                    "uncertainty_excludes": [
                                        "caller pixel annotation", "camera calibration"]})

    def plane_intersect(self, obs_id, px, plane_point_xyz, plane_normal_xyz,
                        plane_offset_sigma_m, pixel_sigma_px=DEFAULT_PIXEL_SIGMA_PX):
        """Pixel ray ∩ caller-supplied plane of ANY orientation (point + normal) → world point.

        The plane definition is supplied entirely by the caller; no scene surface or acquisition
        method is assumed. This general form subsumes the internal axis-aligned compatibility helper.

        Error model (spec §4: uncertainty must be derived from a DECLARED error model, never a flat
        constant). Two independent terms:
          * pixel — caller-owned isotropic u/v sigma propagated by the exact local Jacobian
          * plane — caller-owned 1-sigma plane offset propagated along the ray
        The plane term normally DOMINATES: a plane the caller assumed rather than measured can be
        wrong by tens of centimeters, while the pixel term is ~1 cm. Reporting only the pixel term
        would rank an invented plane above an honestly-declared size prior, which is the harness
        silently ordering scale sources for the model. The harness cannot verify the plane's
        provenance, so the Estimate is ALWAYS `coarse=True`; both sensitivities are returned.
        """
        o = self._get_obs(obs_id)
        K = o.cam_pose_snapshot["K"]
        p0 = np.asarray(plane_point_xyz, float)
        n = np.asarray(plane_normal_xyz, float)
        if p0.shape != (3,) or not np.all(np.isfinite(p0)):
            raise ValueError("plane_point_xyz must be three finite world coordinates")
        if n.shape != (3,) or not np.all(np.isfinite(n)):
            raise ValueError("plane_normal_xyz must be a finite 3-vector")
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-9:
            raise ValueError("plane_normal_xyz must be non-zero")
        n = n / n_norm
        sigma_plane = float(plane_offset_sigma_m)
        if not np.isfinite(sigma_plane) or sigma_plane < 0.0:
            raise ValueError("plane_offset_sigma_m must be a finite non-negative number of meters")
        sigma_pixel = float(pixel_sigma_px)
        if not np.isfinite(sigma_pixel) or sigma_pixel < 0.0:
            raise ValueError("pixel_sigma_px must be a finite non-negative number of pixels")
        C = self._cam_centre(o)
        from codeaction.backends.robotwin import epipolar as epi
        u, v = float(px[0]), float(px[1])
        E = o.cam_pose_snapshot["E"]
        p1 = np.asarray(epi.back_project(K, E, [u, v], 1.0), float)
        ray_vec = p1 - C
        ray_dir = ray_vec / float(np.linalg.norm(ray_vec))
        base_prov = {"tool": "plane_intersect", "obs_id": obs_id,
                     "px": [u, v],
                     "plane_point_xyz": p0.tolist(),
                     "plane_normal_xyz": n.tolist(),
                     "plane_offset_sigma_m": sigma_plane,
                     "ray_origin": C.tolist(),
                     "ray_dir": ray_dir.tolist(),
                     "inputs_unverified": ["plane_point_xyz"],
                     "pixel_sigma_px": sigma_pixel,
                     "plane_from": "caller-supplied; unverified",
                     "coarse_reason": "caller plane unverified",
                     "method": "ray-plane intersection",
                     "uncertainty_scope": "pixel annotation + plane offset",
                     "uncertainty_excludes": [
                         "camera calibration", "pixel error beyond caller isotropic sigma",
                         "plane error beyond declared offset"]}
        denom = float(np.dot(ray_dir, n))
        if abs(denom) < 1e-9:
            return Estimate(value=None, kind="point", uncertainty=float("inf"), coarse=True,
                            provenance={**base_prov, "note": "ray parallel to the plane"})
        t = float(np.dot(p0 - C, n)) / denom
        if t <= 0:
            return Estimate(value=None, kind="point", uncertainty=float("inf"), coarse=True,
                            provenance={**base_prov,
                                        "note": "intersection is behind the camera"})
        xyz = (C + t * ray_dir).tolist()
        rng = float(t)
        obliquity = 1.0 / abs(denom)
        # The back-projected depth-1 vector is affine in u/v. Differentiating
        # P=C+A*r/(n.r) gives the exact local ray-plane Jacobian for this K/E and plane.
        ray_u = np.asarray(epi.back_project(K, E, [u + 1.0, v], 1.0), float) - p1
        ray_v = np.asarray(epi.back_project(K, E, [u, v + 1.0], 1.0), float) - p1
        raw_denom = float(np.dot(ray_vec, n))
        plane_numer = float(np.dot(p0 - C, n))

        def derivative(delta_ray):
            return plane_numer * (
                delta_ray * raw_denom - ray_vec * float(np.dot(delta_ray, n))) \
                / (raw_denom * raw_denom)

        jac_u = derivative(ray_u)
        jac_v = derivative(ray_v)
        pixel_term = sigma_pixel * float(np.sqrt(
            np.dot(jac_u, jac_u) + np.dot(jac_v, jac_v)))
        plane_term = sigma_plane * obliquity
        unc = float(np.hypot(pixel_term, plane_term))
        # d(point)/d(plane offset along +n): shifting the plane by δ along n moves the
        # intersection by δ/(d·n) along the ray direction.
        sensitivity = (ray_dir / denom).tolist()
        self._register_unverified_point(xyz, "plane_intersect", ["plane_point_xyz"])
        return Estimate(value=xyz, kind="point", uncertainty=unc, coarse=True,
                        provenance={**base_prov,
                                    "range_m": rng,
                                    "obliquity_amplification": round(obliquity, 3),
                                    "pixel_jacobian_world_m_per_px": {
                                        "u": jac_u.tolist(), "v": jac_v.tolist()},
                                    "plane_offset_sensitivity_per_m": sensitivity,
                                    "uncertainty_terms": {"pixel_m": pixel_term,
                                                          "plane_offset_m": plane_term}})

    @_observed_action
    def capture_motion_pair(self, arm, dx, dy, dz):
        """Capture one wrist camera before and after one caller-selected guarded arm motion.

        This acquisition contains the whole real motion.  It deliberately does not identify an
        image target or match pixels: after seeing the returned frames, the model supplies its own
        correspondence to triangulate_correspondence.
        """
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        delta = np.asarray([dx, dy, dz], dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("dx, dy, and dz must be finite meters")
        if float(np.linalg.norm(delta)) < 1e-9:
            raise ValueError("the caller-selected camera motion must be non-zero")

        camera = f"{arm}_camera"
        before = self._capture_camera(camera)
        contact_before = self._contact(arm)
        motion = self.move_delta(arm, dx=float(dx), dy=float(dy), dz=float(dz))
        after = self._capture_camera(camera)
        contact_after = self._contact(arm)

        C0, C1 = self._cam_centre(before), self._cam_centre(after)
        camera_delta = C1 - C0
        R0 = np.asarray(before.cam_pose_snapshot["E"], dtype=float)[:3, :3]
        R1 = np.asarray(after.cam_pose_snapshot["E"], dtype=float)[:3, :3]
        relative_rotation = R1 @ R0.T
        cos_angle = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_deg = float(np.degrees(np.arccos(cos_angle)))
        contact_observed = self._touching(contact_before) or self._touching(contact_after)
        contact_evidence = {
            "contact_free": not contact_observed,
            "scope": "finger-world contact checked before/after each guarded leg; a grazing "
                     "transient between checks can be missed",
            "before": contact_before,
            "after": contact_after,
            "motion_abort_reason": motion.abort_reason,
        }
        self._n_pair += 1
        pair = ObservationPair(
            pair_id=f"motion_pair_{self._n_pair:03d}", before=before, after=after,
            motion=motion, camera_delta_world_m=camera_delta.tolist(),
            camera_baseline_m=float(np.linalg.norm(camera_delta)),
            camera_rotation_deg=rotation_deg, contact_evidence=contact_evidence,
            validity={"motion_succeeded": motion.status == "SUCCESS",
                      "baseline_nonzero": float(np.linalg.norm(camera_delta)) > 1e-9,
                      "contact_free": not contact_observed})
        self._pairs[pair.pair_id] = pair
        return pair

    def triangulate_correspondence(self, pair_id, px_before, px_after,
                                   pixel_sigma_px=DEFAULT_PIXEL_SIGMA_PX):
        """Triangulate model-selected corresponding pixels from a capture_motion_pair result."""
        pair = self._get_pair(pair_id)
        e = _scale.triangulate_correspondence(
            pair.before.cam_pose_snapshot["K"], pair.before.cam_pose_snapshot["E"], px_before,
            pair.after.cam_pose_snapshot["K"], pair.after.cam_pose_snapshot["E"], px_after,
            pixel_sigma_px)
        return Estimate(
            value=e.value, kind=e.kind, uncertainty=e.uncertainty, coarse=e.coarse,
            provenance={**e.provenance, "tool": "triangulate_correspondence",
                        "pair_id": pair_id,
                        "obs_before": pair.before.obs_id, "obs_after": pair.after.obs_id,
                        "camera": pair.before.camera,
                        "camera_delta_world_m": pair.camera_delta_world_m,
                        "camera_baseline_m": pair.camera_baseline_m,
                        "camera_rotation_deg": pair.camera_rotation_deg,
                        "pair_validity": dict(pair.validity),
                        "correspondence_verified": False,
                        "baseline_source": "achieved camera extrinsics"})

    def scale_from_object_size(self, obs_id, bbox, extent_axis, known_extent_m,
                               known_extent_sigma_m=None):
        """COARSE depth from a caller-selected projected bbox extent and real prior."""
        o = self._get_obs(obs_id)
        K = np.asarray(o.cam_pose_snapshot["K"], dtype=float)
        fx, fy = float(K[0][0]), float(K[1][1])
        e = _scale.scale_from_object_size(
            fx, fy, bbox, extent_axis, known_extent_m, known_extent_sigma_m)
        return Estimate(value=e.value, kind=e.kind, uncertainty=e.uncertainty, coarse=e.coarse,
                        provenance={**e.provenance, "tool": "scale_from_object_size",
                                    "obs_id": obs_id, "bbox": list(bbox)})

    def scale_from_gripper(self, camera, arm):
        """Capture a caller-selected RGB camera and overlay the live FK fingertip ruler."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        base = self._capture_camera(camera)
        return self._gripper_ruler_observation(base, arm, tool_name="scale_from_gripper")

    # ── drawing (known anchors only; params from the model, no smart defaults) ──
    def draw_marks(self, obs_id, pixels, labels):
        """Labelled open circles at the model's OWN pixels (self-check / communicate a pick)."""
        if len(pixels) != len(labels):
            raise ValueError("pixels and labels must have the same length")
        if not DRAW_MARKS_MIN_COUNT <= len(pixels) <= DRAW_MARKS_MAX_COUNT:
            raise ValueError(
                f"{DRAW_MARKS_MIN_COUNT}..{DRAW_MARKS_MAX_COUNT} marks (legibility bound)")
        from PIL import Image, ImageDraw
        from codeaction.backends.robotwin import epipolar as epi
        base = self._get_obs(obs_id)
        im = Image.open(base.image_ref).convert("RGB")
        dr = ImageDraw.Draw(im)
        width, height = im.size
        used = []
        for i, (px, lab) in enumerate(zip(pixels, labels)):
            name, rgb = epi.PICK_COLORS[i % len(epi.PICK_COLORS)]
            u, v = int(round(px[0])), int(round(px[1]))
            in_frame = 0 <= float(px[0]) < width and 0 <= float(px[1]) < height
            r = 9
            dr.ellipse([u - r, v - r, u + r, v + r], outline=rgb, width=4)
            label = str(lab)
            tx, ty = u + r + 4, v - r
            try:
                tb = dr.textbbox((tx, ty), label)
                dr.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1],
                             fill=(0, 0, 0))
            except Exception:
                tb = None
            dr.text((tx, ty), label, fill=rgb)
            used.append({"px": [float(px[0]), float(px[1])],
                         "label": label,
                         "color": name,
                         "in_frame": bool(in_frame),
                         "label_bbox_px": list(tb) if tb else None})
        return self._annotated(base, im, {"tool": "draw_marks", "marks": used,
                                          "image_size_hw": [height, width]})

    def _gripper_ruler_observation(self, o, arm, tool_name):
        """Render current FK fingertip geometry into an observation from the current tick."""
        self._fresh(o)
        pts = self._finger_points(arm)
        if len(pts) < 2:
            raise ValueError(f"could not read two finger links for {arm}")
        p0, p1 = pts[0], pts[1]
        dist = float(np.linalg.norm(p1 - p0))
        axis = ((p1 - p0) / dist).tolist() if dist > 1e-9 else None
        a, b = self._project_pt(o, p0.tolist()), self._project_pt(o, p1.tolist())
        from PIL import Image, ImageDraw
        im = Image.open(o.image_ref).convert("RGB")
        dr = ImageDraw.Draw(im)
        width, height = im.size
        endpoint_in_frame = [bool(px is not None and 0 <= float(px[0]) < width
                                  and 0 <= float(px[1]) < height) for px in (a, b)]
        span_px = (float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
                   if a is not None and b is not None else None)
        span_valid = (span_px is not None and np.isfinite(span_px)
                      and span_px > RULER_MIN_PROJECTED_SPAN_PX)
        valid_metric_reference = all(endpoint_in_frame) and span_valid
        failure_reason = None
        if not all(endpoint_in_frame):
            failure_reason = "endpoint_out_of_frame"
        elif not span_valid:
            failure_reason = "degenerate_projected_span"
        E = np.asarray(o.cam_pose_snapshot["E"], dtype=float)[:3]
        camera_depths = [float((E[:3, :3] @ p + E[:3, 3])[2]) for p in (p0, p1)]
        if a and b:
            dr.line([tuple(map(int, a)), tuple(map(int, b))], fill=(240, 210, 30), width=3)
            for px in (a, b):    # ring each fingertip so the endpoints read as fingertips
                dr.ellipse([px[0] - 4, px[1] - 4, px[0] + 4, px[1] + 4],
                           outline=(240, 210, 30), width=2)
            dr.text((int((a[0] + b[0]) / 2) + 5, int((a[1] + b[1]) / 2)),
                    f"{dist:.3f} m", fill=(240, 210, 30))
            dr.text((int((a[0] + b[0]) / 2) + 5, int((a[1] + b[1]) / 2) + 12),
                    "finger-opening axis", fill=(240, 210, 30))
        # result self-interpretation: the segment joins the two fingertips, so it IS the finger-
        # opening axis; report that axis as a world vector (FK self-knowledge, no scene facts)
        return self._annotated(o, im, {"tool": tool_name, "arm": arm,
                                       "true_dist_m": round(dist, 4),
                                       "endpoints_px": [a, b],
                                       "endpoint_in_frame": endpoint_in_frame,
                                       "valid_metric_reference": valid_metric_reference,
                                       "span_px": span_px,
                                       "camera_depths_m": camera_depths,
                                       "meters_per_pixel_along_segment": (
                                           dist / span_px
                                           if valid_metric_reference else None),
                                       "segment_meaning": "the segment joins the two fingertips = "
                                                          "the finger-OPENING axis",
                                       "opening_axis_world": ([round(float(v), 3) for v in axis]
                                                              if axis else None),
                                       "in_view": all(endpoint_in_frame),
                                       "minimum_projected_span_px":
                                           RULER_MIN_PROJECTED_SPAN_PX,
                                       "failure_reason": failure_reason,
                                       "transfer_limit": "pixel scale transfers only to image "
                                                         "features at a similar camera range and "
                                                         "orientation"})

