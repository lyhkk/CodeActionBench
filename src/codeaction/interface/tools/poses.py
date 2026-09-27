"""Preview and comparison of candidate TCP poses before any motion is committed."""
from codeaction.interface.tools._base import (
    _BOX_OVERLAP_COLOR,
    _CANDIDATE_COLORS,
    _MAX_POSE_CANDIDATES,
    _emb,
    _orientation,
    np)


class PoseTools:
    def preview_tcp_pose(self, obs_id, arm, tcp_xyz, quat_wxyz, finger_gap_m=None,
                             reference_px=None, reference_xyz_world=None):
        """Project a proposed gripper pose using the robot's live collision geometry.

        A caller reference can make alignment auditable without turning this tool into a detector:
        an image pixel remains only a pixel/ray, while a supplied world point also permits a world
        offset. A fixed-length approach arrow and two finger-closing arrows are display-only
        direction cues, not a planned path, collision query, reachability check, or grasp-quality
        verdict.
        """
        self._require_parallel_jaw("preview_tcp_pose")
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        o = self._get_obs(obs_id)
        self._fresh(o)
        source_tick = int(self._tick)
        tcp = np.asarray(tcp_xyz, float)
        if tcp.shape != (3,):
            raise ValueError("tcp_xyz must be [x,y,z]")
        if reference_px is not None and reference_xyz_world is not None:
            raise ValueError("pass at most one of reference_px and reference_xyz_world")
        reference_input_px = None
        reference_input_xyz = None
        if reference_px is not None:
            reference_input_px = np.asarray(reference_px, float)
            if reference_input_px.shape != (2,) or not np.all(np.isfinite(reference_input_px)):
                raise ValueError("reference_px must be two finite image coordinates [u,v]")
        elif reference_xyz_world is not None:
            reference_input_xyz = np.asarray(reference_xyz_world, float)
            if reference_input_xyz.shape != (3,) or not np.all(np.isfinite(reference_input_xyz)):
                raise ValueError("reference_xyz_world must be three finite world coordinates")
        if finger_gap_m is not None:
            gap_min_m, gap_max_m = _emb.get_finger_gap_bounds_m()
            requested_gap = float(finger_gap_m)
            if (not np.isfinite(requested_gap)
                    or requested_gap < gap_min_m or requested_gap > gap_max_m):
                raise ValueError(
                    f"finger_gap_m must be within the active embodiment's nominal "
                    f"[{gap_min_m:.3f},{gap_max_m:.3f}] m range")
        geometry = self._candidate_gripper_geometry(arm, tcp, quat_wxyz, finger_gap_m)
        gap, gap_source = geometry["gap"], geometry["gap_source"]
        R = geometry["R"]
        approach_axis = R[:, 0]
        opening_axis = R[:, 1]
        lateral_axis = R[:, 2]
        axes = _orientation.decode_pose_axes(quat_wxyz)
        finger_records = []
        inner_tip_centers_world = []
        all_final_pixels = []
        all_final_in_frame = []
        approach_display_length = 0.10
        approach_start = tcp - approach_display_length * approach_axis
        ee_offset = float(_emb.get_embodiment()["tcp"]["tcp_to_ee_offset_m"])

        from PIL import Image, ImageDraw
        base = Image.open(o.image_ref).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(overlay, "RGBA")
        box_draw_records = []
        width, height = base.size
        finger_colors = [(225, 78, 178), (225, 78, 178)]
        def bbox(pixels):
            pixels = [p for p in pixels if p is not None]
            if not pixels:
                return None
            xs, ys = [p[0] for p in pixels], [p[1] for p in pixels]
            return [round(min(xs), 2), round(min(ys), 2),
                    round(max(xs), 2), round(max(ys), 2)]

        for index, finger in enumerate(geometry["fingers"]):
            origin_world = finger["link_origin_world"]
            final_shapes = []
            collision_pixels = []
            shape_draw_parts = []
            vertex_count = 0
            projectable_count = 0
            in_frame_count = 0
            for final_world in finger["shapes_world"]:
                final_px, shape_depths = self._project_points(o, final_world)
                vertex_count += len(final_px)
                projectable_count += sum(p is not None for p in final_px)
                in_frame_flags = [bool(p is not None and 0 <= p[0] < width
                                       and 0 <= p[1] < height) for p in final_px]
                in_frame_count += sum(in_frame_flags)
                all_final_in_frame.extend(in_frame_flags)
                all_final_pixels.extend(p for p in final_px if p is not None)
                collision_pixels.extend(p for p in final_px if p is not None)
                final_shapes.append(final_world)
                shape_draw_parts.append({
                    "hull": self._convex_hull_2d(
                        [p for p in final_px if p is not None]),
                    "depth": float(np.mean(shape_depths)),
                })

            # A local-axis-aligned box derived from all live collision vertices keeps the physical
            # dimensions and camera perspective while removing mesh-detail clutter. It is a
            # display abstraction, not a replacement collision model.
            box_min_tcp = finger["box_min_tcp"]
            box_max_tcp = finger["box_max_tcp"]
            box_px, box_depths = self._project_points(o, finger["box_world"])
            box_draw_records.append({
                "kind": "finger", "index": index, "pixels": box_px, "depths": box_depths,
                "collision_hull": self._convex_hull_2d(collision_pixels),
                "shape_parts": shape_draw_parts,
            })

            inner_tip_world = finger["inner_tip_world"]
            inner_tip_centers_world.append(inner_tip_world)
            origin_px = self._project_pt(o, origin_world)
            inner_tip_px = self._project_pt(o, inner_tip_world)
            final_vertices = np.concatenate(final_shapes, axis=0)
            finger_records.append({
                "link_name": finger["link_name"],
                "collision_shape_count": len(final_shapes),
                "collision_vertex_count": vertex_count,
                "link_origin_xyz": [round(float(v), 5) for v in origin_world],
                "link_origin_px": origin_px,
                "inner_tip_center_xyz": [round(float(v), 5) for v in inner_tip_world],
                "inner_tip_center_px": inner_tip_px,
                "display_box_tcp": {
                    "min": [round(float(v), 5) for v in box_min_tcp],
                    "max": [round(float(v), 5) for v in box_max_tcp],
                },
                "collision_aabb_world": {
                    "min": [round(float(v), 5) for v in final_vertices.min(axis=0)],
                    "max": [round(float(v), 5) for v in final_vertices.max(axis=0)],
                },
                "final_bbox_px": bbox(box_px),
                "collision_vertices_projectable": projectable_count,
                "collision_vertices_in_frame": in_frame_count,
                "final_fully_in_frame": in_frame_count == vertex_count,
            })

        # Render the physical collision shapes themselves as translucent claws. Their projected
        # hulls are depth-sorted; the TCP-local bounding boxes remain metadata only.
        box_draw_records.sort(
            key=lambda record: float(np.mean(record["depths"])), reverse=True)
        for record in box_draw_records:
            color = finger_colors[record["index"]]
            for part in sorted(record["shape_parts"], key=lambda item: item["depth"],
                               reverse=True):
                if len(part["hull"]) >= 3:
                    dr.polygon(part["hull"], fill=(*color, 38))
            hull = record["collision_hull"]
            if len(hull) >= 3:
                dr.line(hull + [hull[0]], fill=(*color, 225), width=2)
            # Local +x maximum is the forward tip face.
            pixels = record["pixels"]
            if not all(p is not None for p in pixels):
                continue
            tip_face = [pixels[i] for i in (4, 5, 7, 6, 4)]
            dr.line(tip_face, fill=(90, 255, 125, 220), width=2)
        final_bbox = bbox(all_final_pixels)

        tcp_px = self._project_pt(o, tcp)
        ee_xyz = tcp - ee_offset * approach_axis
        ee_px = self._project_pt(o, ee_xyz)
        approach_start_px = self._project_pt(o, approach_start)
        origin_center_world = geometry["origin_center_world"]
        origin_center_px = self._project_pt(o, origin_center_world)
        grasp_center = geometry["grasp_center"]
        grasp_center_px = self._project_pt(o, grasp_center)
        camera_rotation = np.asarray(o.cam_pose_snapshot["E"], float)[:3, :3]
        approach_camera = camera_rotation @ approach_axis
        approach_view_angle_deg = float(np.degrees(np.arccos(
            max(0.0, min(1.0, abs(float(approach_camera[2])))))))
        approach_projected_length_px = (
            None if approach_start_px is None or tcp_px is None
            else round(float(np.linalg.norm(
                np.asarray(tcp_px, float) - np.asarray(approach_start_px, float))), 3))
        approach_nearly_along_view = approach_view_angle_deg < 20.0
        reference = {
            "source": "none",
            "input_px": None,
            "input_xyz_world": None,
            "projected_px": None,
            "grasp_minus_reference_px": None,
            "pixel_distance": None,
            "grasp_minus_reference_world": None,
            "world_distance_m": None,
            "definition": "optional caller evidence; an image pixel is a camera ray, not a 3D "
                          "origin. This tool reports offsets and applies no acceptance threshold.",
        }
        if reference_input_px is not None:
            reference["source"] = "caller_image_pixel"
            reference["input_px"] = [float(v) for v in reference_input_px]
            reference["projected_px"] = [float(v) for v in reference_input_px]
        elif reference_input_xyz is not None:
            reference["source"] = "caller_world_point"
            reference["input_xyz_world"] = [float(v) for v in reference_input_xyz]
            reference["projected_px"] = self._project_pt(o, reference_input_xyz)
            world_offset = grasp_center - reference_input_xyz
            reference["grasp_minus_reference_world"] = [
                round(float(v), 6) for v in world_offset]
            reference["world_distance_m"] = round(float(np.linalg.norm(world_offset)), 6)
        if grasp_center_px is not None and reference["projected_px"] is not None:
            pixel_offset = np.asarray(grasp_center_px, float) - np.asarray(
                reference["projected_px"], float)
            reference["grasp_minus_reference_px"] = [
                round(float(v), 3) for v in pixel_offset]
            reference["pixel_distance"] = round(float(np.linalg.norm(pixel_offset)), 3)

        def arrow_polygon(start_px, end_px, color, width_px=4.0, outline=None):
            if start_px is None or end_px is None:
                return
            start_v, end_v = np.asarray(start_px, float), np.asarray(end_px, float)
            delta = end_v - start_v
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                return
            unit = delta / norm
            normal = np.asarray([-unit[1], unit[0]])
            head_len = min(12.0, max(7.0, norm * 0.30))
            neck = end_v - head_len * unit
            shaft = [start_v + 0.5 * width_px * normal,
                     neck + 0.5 * width_px * normal,
                     neck - 0.5 * width_px * normal,
                     start_v - 0.5 * width_px * normal]
            dr.polygon([tuple(v) for v in shaft], fill=(*color, 225),
                       outline=outline)
            dr.polygon([tuple(end_v), tuple(neck + width_px * normal),
                        tuple(neck - width_px * normal)], fill=(*color, 255),
                       outline=outline)

        def perspective_arrow(start_px, end_px, color):
            """A tapered screen-space arrow: narrow at its tail, broad at its display anchor."""
            if start_px is None or end_px is None:
                return
            start_v, end_v = np.asarray(start_px, float), np.asarray(end_px, float)
            delta = end_v - start_v
            norm = float(np.linalg.norm(delta))
            if norm < 1e-6:
                return
            unit = delta / norm
            normal = np.asarray([-unit[1], unit[0]])
            head_len = min(14.0, max(9.0, norm * 0.32))
            neck = end_v - head_len * unit

            def parts(extra):
                start_half = 1.5 + extra
                neck_half = 3.0 + extra
                head_half = 6.5 + extra
                shaft = [
                    start_v + start_half * normal,
                    neck + neck_half * normal,
                    neck - neck_half * normal,
                    start_v - start_half * normal,
                ]
                head = [end_v, neck + head_half * normal, neck - head_half * normal]
                return shaft, head

            halo_shaft, halo_head = parts(2.0)
            dr.polygon([tuple(v) for v in halo_shaft], fill=(20, 35, 42, 205))
            dr.polygon([tuple(v) for v in halo_head], fill=(20, 35, 42, 225))
            shaft, head = parts(0.0)
            dr.polygon([tuple(v) for v in shaft], fill=(*color, 225))
            dr.polygon([tuple(v) for v in head], fill=(*color, 255))

        # Keep the approach cue outside the dense terminal geometry. The arrow is parallel to the
        # true projected approach direction, but its square endpoint is explicitly a display
        # anchor—not TCP, a scene point, or a robot part.
        approach_display_start_px = approach_start_px
        approach_display_anchor_px = tcp_px
        if approach_start_px is not None and tcp_px is not None and all_final_pixels:
            true_start = np.asarray(approach_start_px, float)
            true_end = np.asarray(tcp_px, float)
            projected_delta = true_end - true_start
            projected_norm = float(np.linalg.norm(projected_delta))
            if projected_norm > 1e-6:
                projected_unit = projected_delta / projected_norm
                projected_normal = np.asarray([-projected_unit[1], projected_unit[0]])
                pixels_v = [np.asarray(pixel, float) for pixel in all_final_pixels]
                candidates = []
                for sign in (-1.0, 1.0):
                    outward = sign * projected_normal
                    extent = max(0.0, max(float(np.dot(pixel - true_end, outward))
                                          for pixel in pixels_v))
                    anchor = true_end + (extent + 14.0) * outward
                    start = anchor - min(34.0, max(24.0, projected_norm)) * projected_unit
                    margin = min(
                        *(float(point[0]) for point in (anchor, start)),
                        *(float(point[1]) for point in (anchor, start)),
                        *(float(width - 1 - point[0]) for point in (anchor, start)),
                        *(float(height - 1 - point[1]) for point in (anchor, start)),
                    )
                    candidates.append((margin, start, anchor))
                _, display_start, display_anchor = max(candidates, key=lambda item: item[0])
                approach_display_start_px = display_start.tolist()
                approach_display_anchor_px = display_anchor.tolist()

        perspective_arrow(approach_display_start_px, approach_display_anchor_px,
                          (35, 215, 245))
        approach_display_label = f"A {approach_view_angle_deg:.1f}deg"
        if approach_display_anchor_px is not None:
            u, v = approach_display_anchor_px
            dr.rectangle([u - 5, v - 5, u + 5, v + 5],
                         fill=(20, 35, 42, 225), outline=(35, 215, 245, 255), width=2)
            label_box = dr.textbbox((0, 0), approach_display_label, stroke_width=1)
            label_width = label_box[2] - label_box[0]
            label_height = label_box[3] - label_box[1]
            label_x = u + 8
            if label_x + label_width >= width:
                label_x = u - 8 - label_width
            label_y = max(1.0, min(float(height - label_height - 1), v - label_height - 7))
            dr.text((label_x, label_y), approach_display_label,
                    fill=(85, 235, 255, 255), stroke_width=2,
                    stroke_fill=(20, 35, 42, 235))

        # The two inner-tip markers and inward arrows show the physical close action directly.
        # Clip the rendered shafts to the marker edges in image space: the full physical direction
        # remains in the payload, while the visual cue does not cover either endpoint marker.
        closing_records = []
        for finger, inner_tip_world in zip(finger_records, inner_tip_centers_world):
            closing_end_world = grasp_center
            tip_px = finger["inner_tip_center_px"]
            closing_end_px = grasp_center_px
            if tip_px is not None:
                u, v = tip_px
                for radius, color in ((6, (40, 35, 55, 210)),
                                      (5, (232, 120, 205, 245)),
                                      (3, (252, 175, 225, 255))):
                    dr.ellipse([u - radius, v - radius, u + radius, v + radius],
                               fill=color)
                dr.ellipse([u - 3, v - 3, u - 1, v - 1],
                           fill=(255, 245, 255, 255))
            render_start_px = tip_px
            render_end_px = closing_end_px
            if tip_px is not None and closing_end_px is not None:
                tip_v = np.asarray(tip_px, float)
                center_v = np.asarray(closing_end_px, float)
                screen_delta = center_v - tip_v
                screen_norm = float(np.linalg.norm(screen_delta))
                if screen_norm > 9.0:
                    screen_unit = screen_delta / screen_norm
                    render_start_px = (tip_v + 4.0 * screen_unit).tolist()
                    render_end_px = (center_v - 2.0 * screen_unit).tolist()
            arrow_polygon(render_start_px, render_end_px, (255, 205, 35),
                          width_px=4.0, outline=(55, 45, 20, 235))
            closing_direction = grasp_center - inner_tip_world
            closing_direction /= max(1e-9, float(np.linalg.norm(closing_direction)))
            closing_records.append({
                "finger_link": finger["link_name"],
                "tip_xyz": finger["inner_tip_center_xyz"],
                "tip_px": tip_px,
                "direction_world": [round(float(v), 6) for v in closing_direction],
                "arrow_end_xyz": [round(float(v), 5) for v in closing_end_world],
                "arrow_end_px": closing_end_px,
                "render_start_px": render_start_px,
                "render_end_px": render_end_px,
            })

        reference_projected_px = reference["projected_px"]
        # Do not stack a second marker on an already aligned grasp point; the numeric residual
        # remains in the payload, and a duplicate ring would hide the action arrows.
        if (grasp_center_px is not None and reference_projected_px is not None
                and (reference["pixel_distance"] is None
                     or reference["pixel_distance"] > 6.0)):
            u, v = reference_projected_px
            dr.ellipse([u - 4, v - 4, u + 4, v + 4],
                       outline=(255, 220, 55, 245), width=2)
        if grasp_center_px:
            u, v = grasp_center_px
            dr.ellipse([u - 4, v - 4, u + 4, v + 4],
                       fill=(20, 20, 20, 105), outline=(90, 255, 125, 255), width=2)
            dr.line([(u - 6, v), (u + 6, v)], fill=(90, 255, 125, 255), width=2)
            dr.line([(u, v - 6), (u, v + 6)], fill=(90, 255, 125, 255), width=2)

        axis_payload = {}
        for name, direction in (("approach", approach_axis),
                                ("opening", opening_axis),
                                ("lateral", lateral_axis)):
            camera_axis = camera_rotation @ direction
            view_angle_deg = float(np.degrees(np.arccos(
                max(0.0, min(1.0, abs(float(camera_axis[2])))))))
            axis_payload[name] = {
                "world": [round(float(v), 6) for v in direction],
                "camera": [round(float(v), 6) for v in camera_axis],
                "reads_as": _orientation.describe_axis(direction),
                "angle_to_view_line_deg": round(view_angle_deg, 3),
                "display": ("scene_approach_arrow" if name == "approach"
                            else "scene_closing_arrows" if name == "opening"
                            else "payload_only"),
            }
        if self._tick != source_tick:
            raise RuntimeError("preview_tcp_pose changed the robot-state tick")
        im = Image.alpha_composite(base, overlay).convert("RGB")
        return self._annotated(o, im, {"tool": "preview_tcp_pose",
                                       "arm": arm,
                                       "tcp_xyz": [float(v) for v in tcp],
                                       "quat_wxyz": [float(v) for v in quat_wxyz],
                                       "finger_gap_m": round(gap, 4),
                                       "finger_gap_source": gap_source,
                                       "sampling": {
                                           "image_tick": int(o.tick),
                                           "robot_geometry_tick": source_tick,
                                           "finger_gap_tick": (
                                               source_tick if finger_gap_m is None else None),
                                           "ticks_match": int(o.tick) == source_tick,
                                           "robot_command_executed": False,
                                           "tick_advanced": False,
                                           "definition": "RGB/calibration and live robot geometry "
                                                         "are sampled at one harness tick; "
                                                         "finger_gap_tick is null when the caller "
                                                         "supplied finger_gap_m. This preview sends "
                                                         "no robot command and does not advance "
                                                         "the harness tick.",
                                       },
                                       "geometry_source": "true projected hull of live robot "
                                                          "finger collision vertices with "
                                                          "translucent per-shape fill; no palm, "
                                                          "bounding-box fill, scene geometry, or "
                                                          "occlusion claim",
                                       "axes": axis_payload,
                                       "orientation_convention": _orientation.POSE_CONVENTION,
                                       "action_overlay": {
                                           "approach": {
                                               "display_length_m": approach_display_length,
                                               "start_xyz": [round(float(v), 5)
                                                             for v in approach_start],
                                               "start_px": approach_start_px,
                                               "end_xyz": [round(float(v), 5) for v in tcp],
                                               "end_px": tcp_px,
                                               "direction_world": [
                                                   round(float(v), 6)
                                                   for v in approach_axis],
                                               "projected_length_px":
                                                   approach_projected_length_px,
                                               "nearly_along_view":
                                                   approach_nearly_along_view,
                                               "angle_to_view_line_deg":
                                                   round(approach_view_angle_deg, 3),
                                               "display_start_px":
                                                   approach_display_start_px,
                                               "display_anchor_px":
                                                   approach_display_anchor_px,
                                               "display_label":
                                                   approach_display_label,
                                               "definition": "display-only arrow parallel to the "
                                                             "true projected approach direction; "
                                                             "its square endpoint is a displaced "
                                                             "display anchor, not TCP, a scene "
                                                             "point, or a robot part",
                                           },
                                           "closing": closing_records,
                                           "closing_definition": "one arrow from each physical "
                                                                 "inner-tip marker toward the "
                                                                 "collision-derived grasp center; "
                                                                 "rendered shafts are clipped to "
                                                                 "the endpoint-marker edges",
                                       },
                                       "ee_xyz": [round(float(v), 5) for v in ee_xyz],
                                       "ee_px": ee_px,
                                       "tcp_px": tcp_px,
                                       "grasp_center_xyz": [round(float(v), 5)
                                                            for v in grasp_center],
                                       "grasp_center_px": grasp_center_px,
                                       "grasp_center_definition": "midpoint between the two "
                                                                  "collision-derived finger "
                                                                  "inner-tip face centers",
                                       "reference": reference,
                                       "finger_link_origin_center_xyz": [round(float(v), 5)
                                                                          for v in origin_center_world],
                                       "finger_link_origin_center_px": origin_center_px,
                                       "fingers": finger_records,
                                       "final_collision_bbox_px": final_bbox,
                                       "image_size_hw": [height, width],
                                       "rendered_image_size_hw": [height, width],
                                       "in_view": bool(all_final_in_frame)
                                                  and all(all_final_in_frame)})

    def compare_tcp_poses(self, obs_id, candidates):
        """Draw 2-4 caller-proposed terminal gripper poses in ONE image and report their RELATIVE
        geometry.

        This exists for the question a per-candidate image cannot answer: how two or more proposed
        terminal poses stand with respect to EACH OTHER. It is deliberately not a batch mode of
        `preview_tcp_pose` — the per-candidate detail that tool draws (per-shape translucent
        fills, inner-tip markers, two closing arrows, an angle-labelled approach arrow) becomes
        noise once a second candidate shares the canvas, so each candidate here is reduced to its
        two collision-shape outlines, its grasp centre and a short approach stub, and everything
        removed from the image stays in the payload as a number.

        The relative numbers are the point, not the picture: two projected silhouettes can overlap
        completely in the image and be far apart in depth. Pairwise separating-axis clearance is a
        conservative bounding-box fact in meters; box overlap does not prove true-shape collision.

        Scope, stated because a reader will otherwise over-read the result: this compares TERMINAL
        GRIPPER geometry only — no wrist, forearm, or other arm link is posed here, because that
        would need an IK solve rather than a projection. It reads no scene object, evaluates no
        reachability, plans no path, and issues no grasp-quality verdict. Two candidates on the SAME
        arm are alternatives that never coexist, so their boxes are not compared.
        """
        self._require_parallel_jaw("compare_tcp_poses")
        o = self._get_obs(obs_id)
        self._fresh(o)
        source_tick = int(self._tick)
        if not isinstance(candidates, (list, tuple)):
            raise ValueError("candidates must be a list of candidate pose objects")
        if not (2 <= len(candidates) <= _MAX_POSE_CANDIDATES):
            raise ValueError(
                f"candidates must hold 2-{_MAX_POSE_CANDIDATES} poses, got {len(candidates)}; "
                "one pose alone has no relative geometry — preview_tcp_pose renders that")
        allowed = {"arm", "tcp_xyz", "quat_wxyz", "finger_gap_m"}
        geometries = []
        for index, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError(f"candidates[{index}] must be an object with {sorted(allowed)}")
            unknown = sorted(set(item) - allowed)
            if unknown:
                raise ValueError(f"candidates[{index}] has unknown key(s) {unknown}; "
                                 f"accepted: {sorted(allowed)}")
            missing = [key for key in ("arm", "tcp_xyz", "quat_wxyz") if key not in item]
            if missing:
                raise ValueError(f"candidates[{index}] is missing {missing}")
            if item["arm"] not in ("left", "right"):
                raise ValueError(f"candidates[{index}].arm must be 'left' or 'right'")
            geometries.append(self._candidate_gripper_geometry(
                item["arm"], item["tcp_xyz"], item["quat_wxyz"], item.get("finger_gap_m")))

        from PIL import Image, ImageDraw
        base = Image.open(o.image_ref).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(overlay, "RGBA")
        width, height = base.size
        camera_rotation = np.asarray(o.cam_pose_snapshot["E"], float)[:3, :3]
        drawn = []
        for index, geometry in enumerate(geometries):
            color = _CANDIDATE_COLORS[index]
            finger_hulls, candidate_pixels = [], []
            vertex_count = in_frame_count = 0
            for finger in geometry["fingers"]:
                finger_pixels = []
                for shape_world in finger["shapes_world"]:
                    shape_px, _ = self._project_points(o, shape_world)
                    vertex_count += len(shape_px)
                    for pixel in shape_px:
                        if pixel is None:
                            continue
                        finger_pixels.append(pixel)
                        if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                            in_frame_count += 1
                finger_hulls.append(self._convex_hull_2d(finger_pixels))
                candidate_pixels.extend(finger_pixels)
            approach_axis = geometry["R"][:, 0]
            _, box_depths = self._project_points(
                o, [finger["box_center_world"] for finger in geometry["fingers"]])
            drawn.append({
                "index": index, "color": color, "geometry": geometry,
                "hulls": finger_hulls,
                "pixels": np.asarray(candidate_pixels, float).reshape(-1, 2),
                "tcp_px": self._project_pt(o, geometry["tcp"]),
                "approach_tail_px": self._project_pt(
                    o, geometry["tcp"] - 0.10 * approach_axis),
                "grasp_center_px": self._project_pt(o, geometry["grasp_center"]),
                "mean_depth": float(np.mean(box_depths)),
                "vertex_count": vertex_count,
                "in_frame_count": in_frame_count,
            })

        # Far candidates first so a nearer silhouette's outline stays unbroken over a farther one.
        for record in sorted(drawn, key=lambda item: item["mean_depth"], reverse=True):
            color = record["color"]
            for hull in record["hulls"]:
                if len(hull) >= 3:
                    dr.polygon(hull, fill=(*color, 26))
                    dr.line(hull + [hull[0]], fill=(*color, 235), width=2)

        # Layout is SELF-aware, never scene-aware: the tool knows where its own ink landed and
        # nothing else. Knowing where the object or the text is would require detecting them, which
        # is exactly the capability this substrate withholds. So a stub is placed to clear every
        # candidate's silhouette and the image border, and no further.
        all_ink = np.concatenate([record["pixels"] for record in drawn
                                  if len(record["pixels"])] or [np.zeros((0, 2))])
        for record in drawn:
            start_px, anchor_px = self._offset_direction_stub(
                record["approach_tail_px"], record["tcp_px"], record["pixels"], all_ink,
                (width, height))
            record["approach_stub_px"] = [start_px, anchor_px]
            if start_px is not None and anchor_px is not None:
                self._draw_stub_arrow(dr, start_px, anchor_px, record["color"])

        for record in drawn:
            record["label_px"] = None
            pixel = record["grasp_center_px"]
            if pixel is None:
                continue
            u, v = pixel
            color = record["color"]
            dr.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(20, 22, 28, 215))
            dr.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(*color, 255))
            # Two near-coincident candidates are exactly the case this tool is for, and it is also
            # the case where two labels land on top of each other, so the label leans away from the
            # OTHER candidates' ink by the same self-aware rule the stubs use.
            others = [item["pixels"] for item in drawn
                      if item is not record and len(item["pixels"])]
            others = np.concatenate(others) if others else np.zeros((0, 2))
            record["label_px"] = self._offset_label_anchor(
                pixel, others, (width, height))
            dr.text(tuple(record["label_px"]), f"C{record['index'] + 1}",
                    fill=(*color, 255), stroke_width=2, stroke_fill=(20, 22, 28, 235))

        pairs = []
        for a in range(len(drawn)):
            for b in range(a + 1, len(drawn)):
                pairs.append(self._candidate_pair_record(drawn[a], drawn[b]))

        # A connector is drawn only for a single pair. With three or four candidates the six
        # possible connectors would obscure the silhouettes they are meant to relate, and the same
        # numbers are in `pairs` either way.
        if len(drawn) == 2 and all(record["grasp_center_px"] is not None for record in drawn):
            pair = pairs[0]
            start = np.asarray(drawn[0]["grasp_center_px"], float)
            end = np.asarray(drawn[1]["grasp_center_px"], float)
            boxes_overlap = bool(pair["finger_bounding_boxes_overlap"])
            line_color = _BOX_OVERLAP_COLOR if boxes_overlap else (225, 228, 235)
            dr.line([tuple(start), tuple(end)], fill=(*line_color, 200), width=2)
            middle = 0.5 * (start + end)
            text = f"{pair['grasp_center_distance_m']:.3f} m"
            if boxes_overlap:
                text += " BOX OVERLAP"
            dr.text((middle[0] + 6, middle[1] + 4), text, fill=(*line_color, 255),
                    stroke_width=2, stroke_fill=(20, 22, 28, 235))

        self._draw_candidate_legend(dr, drawn, pairs, (width, height))
        im = Image.alpha_composite(base, overlay).convert("RGB")

        candidate_records = []
        for record in drawn:
            geometry = record["geometry"]
            approach_axis = geometry["R"][:, 0]
            opening_axis = geometry["R"][:, 1]
            axes = _orientation.decode_pose_axes(geometry["quat"])
            view_angle = float(np.degrees(np.arccos(max(0.0, min(1.0, abs(float(
                (camera_rotation @ approach_axis)[2])))))))
            pixels = record["pixels"]
            candidate_records.append({
                "index": record["index"],
                "label": f"C{record['index'] + 1}",
                "arm": geometry["arm"],
                "color_rgb": [int(v) for v in record["color"]],
                "tcp_xyz": [round(float(v), 5) for v in geometry["tcp"]],
                "quat_wxyz": [float(v) for v in geometry["quat"]],
                "finger_gap_m": round(geometry["gap"], 4),
                "finger_gap_source": geometry["gap_source"],
                "tcp_px": record["tcp_px"],
                "grasp_center_xyz": [round(float(v), 5) for v in geometry["grasp_center"]],
                "grasp_center_px": record["grasp_center_px"],
                "approach_axis_world": [round(float(v), 6) for v in approach_axis],
                "opening_axis_world": [round(float(v), 6) for v in opening_axis],
                "approach_reads_as": axes["approach_reads_as"],
                "opening_reads_as": axes["opening_reads_as"],
                "approach_angle_to_view_line_deg": round(view_angle, 3),
                "approach_stub_px": record["approach_stub_px"],
                "label_px": record["label_px"],
                "silhouette_bbox_px": (
                    None if not len(pixels)
                    else [round(float(pixels[:, 0].min()), 2),
                          round(float(pixels[:, 1].min()), 2),
                          round(float(pixels[:, 0].max()), 2),
                          round(float(pixels[:, 1].max()), 2)]),
                "collision_vertex_count": record["vertex_count"],
                "collision_vertices_in_frame": record["in_frame_count"],
                "fully_in_frame": record["in_frame_count"] == record["vertex_count"],
            })
        in_view = all(item["fully_in_frame"] for item in candidate_records)
        return self._annotated(o, im, {
            "tool": "compare_tcp_poses",
            "candidate_count": len(candidate_records),
            "candidates": candidate_records,
            "pairs": pairs,
            "sampling": {"robot_geometry_tick": source_tick},
            "geometry_source": "true projected outline of live robot finger collision vertices, "
                               "re-posed at each caller-proposed TCP pose and finger gap; one "
                               "colour per candidate, no scene geometry and no occlusion claim",
            "overlay_legend": "per candidate: two collision-shape outlines, a filled grasp-centre "
                              "dot with its C-label, and a short approach stub parallel to the "
                              "projected approach direction whose square anchor is a displaced "
                              "display point, not TCP or a scene point. A grasp-centre connector "
                              "is drawn only when exactly two candidates are given",
            "layout_discipline": "stub and legend placement avoid the tool's OWN drawn pixels and "
                                 "the image border only. This overlay cannot see objects, text, or "
                                 "anything else in the scene, so a clear image is not guaranteed",
            "scope": "terminal gripper geometry of caller-proposed poses. No wrist/forearm/arm link "
                     "is posed, no scene object is read, and this is not a reachability, planning, "
                     "or grasp-quality result",
            "image_size_hw": [height, width],
            "rendered_image_size_hw": [height, width],
            "in_view": in_view,
        })

    def _candidate_pair_record(self, a, b):
        """Relative geometry of two drawn candidates — the numbers an image cannot carry."""
        geometry_a, geometry_b = a["geometry"], b["geometry"]
        same_arm = geometry_a["arm"] == geometry_b["arm"]
        delta = geometry_b["grasp_center"] - geometry_a["grasp_center"]
        approach_a = geometry_a["R"][:, 0]
        approach_b = geometry_b["R"][:, 0]
        approach_angle = float(np.degrees(np.arccos(
            max(-1.0, min(1.0, float(np.dot(approach_a, approach_b)))))))
        separation, axis, boxes_overlap = None, None, None
        if not same_arm:
            best = None
            for finger_a in geometry_a["fingers"]:
                for finger_b in geometry_b["fingers"]:
                    value, unit = self._box_separation(finger_a, finger_b)
                    if best is None or value < best[0]:
                        best = (value, unit)
            separation = round(float(best[0]), 5)
            axis = [round(float(v), 6) for v in best[1]]
            boxes_overlap = bool(best[0] <= 0.0)
        overlap = False
        nearer = None
        boxes = []
        for record in (a, b):
            pixels = record["pixels"]
            boxes.append(None if not len(pixels) else
                         (float(pixels[:, 0].min()), float(pixels[:, 1].min()),
                          float(pixels[:, 0].max()), float(pixels[:, 1].max())))
        if boxes[0] is not None and boxes[1] is not None:
            overlap = bool(boxes[0][0] <= boxes[1][2] and boxes[1][0] <= boxes[0][2]
                           and boxes[0][1] <= boxes[1][3] and boxes[1][1] <= boxes[0][3])
            if overlap:
                nearer = a["index"] if a["mean_depth"] <= b["mean_depth"] else b["index"]
        return {
            "a": a["index"], "b": b["index"],
            "same_arm": same_arm,
            "grasp_center_delta_world": [round(float(v), 5) for v in delta],
            "grasp_center_distance_m": round(float(np.linalg.norm(delta)), 5),
            "approach_axis_angle_deg": round(approach_angle, 3),
            "finger_bounding_boxes_overlap": boxes_overlap,
            "separation_m": separation,
            "separation_axis_world": axis,
            "projected_bbox_overlap": overlap,
            "nearer_to_camera": nearer,
            "definition": (
                "delta is b minus a between the two collision-derived grasp centres. "
                "separation_m is the separating-axis clearance between the two candidates' finger "
                "bounding boxes: positive means a proven gap of at least that size along "
                "separation_axis_world, and because each box contains its collision shapes the true "
                "clearance is at least this; zero or negative means the bounding boxes overlap and "
                "the magnitude is the smallest box-separating translation; the true shapes may not "
                "overlap and may not need that translation. Both are null for two poses of the SAME "
                "arm, which are alternatives that "
                "never coexist. projected_bbox_overlap is an image-legibility warning about the "
                "two silhouettes' pixel bounding boxes and states nothing about 3D."),
        }

    @staticmethod
    def _offset_direction_stub(tail_px, head_px, own_pixels, all_pixels, size):
        """Place a short screen arrow parallel to a projected direction, clear of drawn ink.

        Returns (start_px, anchor_px) or (None, None) when the direction does not project.
        """
        if tail_px is None or head_px is None:
            return None, None
        width, height = size
        tail = np.asarray(tail_px, float)
        head = np.asarray(head_px, float)
        delta = head - tail
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return None, None
        unit = delta / norm
        normal = np.asarray([-unit[1], unit[0]])
        own = np.asarray(own_pixels, float).reshape(-1, 2)
        others = np.asarray(all_pixels, float).reshape(-1, 2)
        best = None
        for sign in (-1.0, 1.0):
            outward = sign * normal
            extent = 0.0
            if len(own):
                extent = max(0.0, float(np.max((own - head) @ outward)))
            anchor = head + (extent + 14.0) * outward
            start = anchor - 30.0 * unit
            probes = np.asarray([anchor, start, 0.5 * (anchor + start)])
            margin = float(np.min([probes[:, 0].min(), probes[:, 1].min(),
                                   width - 1 - probes[:, 0].max(),
                                   height - 1 - probes[:, 1].max()]))
            clearance = 1e6
            if len(others):
                clearance = float(np.min(np.linalg.norm(
                    others[None, :, :] - probes[:, None, :], axis=2)))
            score = min(margin, clearance)
            if best is None or score > best[0]:
                best = (score, start, anchor)
        return best[1].tolist(), best[2].tolist()

    @staticmethod
    def _offset_label_anchor(pixel, other_pixels, size, offset=9.0):
        """Diagonal text anchor beside a marker, leaning away from other candidates' pixels."""
        width, height = size
        origin = np.asarray(pixel, float)
        others = np.asarray(other_pixels, float).reshape(-1, 2)
        best = None
        for du in (offset, -offset - 14.0):
            for dv in (-offset - 5.0, offset):
                anchor = origin + np.asarray([du, dv])
                margin = float(min(anchor[0], anchor[1],
                                   width - 14.0 - anchor[0], height - 10.0 - anchor[1]))
                clearance = 1e6
                if len(others):
                    clearance = float(np.min(np.linalg.norm(others - anchor, axis=1)))
                score = min(margin, clearance)
                if best is None or score > best[0]:
                    best = (score, anchor)
        return best[1].tolist()

    @staticmethod
    def _draw_stub_arrow(dr, start_px, anchor_px, color):
        start = np.asarray(start_px, float)
        end = np.asarray(anchor_px, float)
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            return
        unit = delta / norm
        normal = np.asarray([-unit[1], unit[0]])
        head_len = min(11.0, max(7.0, norm * 0.30))
        neck = end - head_len * unit
        dr.line([tuple(start), tuple(neck)], fill=(20, 22, 28, 205), width=6)
        dr.line([tuple(start), tuple(neck)], fill=(*color, 235), width=3)
        dr.polygon([tuple(end), tuple(neck + 5.5 * normal), tuple(neck - 5.5 * normal)],
                   fill=(*color, 255), outline=(20, 22, 28, 225))
        dr.rectangle([end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3],
                     outline=(20, 22, 28, 225), width=1)

    @staticmethod
    def _draw_candidate_legend(dr, drawn, pairs, size):
        """Index/arm/colour key plus the pairwise verdicts, in the emptiest corner of OWN ink."""
        width, height = size
        rows = [f"C{record['index'] + 1}  {record['geometry']['arm']}" for record in drawn]
        for pair in pairs:
            name = f"C{pair['a'] + 1}xC{pair['b'] + 1}"
            if pair["finger_bounding_boxes_overlap"] is None:
                rows.append(f"{name}  same arm, {pair['grasp_center_distance_m']:.3f} m apart")
            elif pair["finger_bounding_boxes_overlap"]:
                rows.append(f"{name}  BOX OVERLAP by {abs(pair['separation_m']):.3f} m")
            else:
                rows.append(f"{name}  clear by >= {pair['separation_m']:.3f} m")
        line_height = 14
        box_width = 8 + 7 * max(len(row) for row in rows) + 22
        box_height = 8 + line_height * len(rows)
        margin = 6
        corners = {
            "top_left": (margin, margin),
            "top_right": (width - box_width - margin, margin),
            "bottom_left": (margin, height - box_height - margin),
            "bottom_right": (width - box_width - margin, height - box_height - margin),
        }
        ink = np.concatenate([record["pixels"] for record in drawn
                              if len(record["pixels"])] or [np.zeros((0, 2))])
        best = None
        for name, (x, y) in corners.items():
            if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                continue
            covered = 0
            if len(ink):
                covered = int(np.count_nonzero(
                    (ink[:, 0] >= x) & (ink[:, 0] <= x + box_width)
                    & (ink[:, 1] >= y) & (ink[:, 1] <= y + box_height)))
            if best is None or covered < best[0]:
                best = (covered, name, x, y)
        if best is None:
            return
        _, _, x, y = best
        dr.rectangle([x, y, x + box_width, y + box_height], fill=(18, 20, 26, 190),
                     outline=(150, 156, 168, 190), width=1)
        for row_index, text in enumerate(rows):
            text_y = y + 4 + row_index * line_height
            if row_index < len(drawn):
                color = drawn[row_index]["color"]
                dr.rectangle([x + 5, text_y + 3, x + 15, text_y + 10], fill=(*color, 255))
            else:
                color = (225, 228, 235)
            dr.text((x + 20, text_y), text, fill=(*color, 255))

    # ── motion + gripper (guarded; return ActionResult) ───────────────────
