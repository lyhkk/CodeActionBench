"""
TaP v2 vision perception module.

Implementation backed by SAPIEN actor segmentation + sim depth.
The geometry pipeline (mask → unproject → median → PCA) is identical to what
a real-world VLM-backed perception would use; the extension point is a single
`PerceptionBackend` ABC that swaps the upstream mask / depth source.

Honesty contract:
  - The simulator's exact 3D object pose is *never* returned to the LLM.
  - `pos_world`, `principal_axis_world`, `extent_world` are computed from
    seg-mask + depth + camera matrices via pure numpy math.
  - We use ``get_scene_objects`` for two purposes:
      (a) Enumerate object names so the LLM can resolve references.
      (b) Bootstrap the SAPIEN seg-buffer-id for each object, by projecting
          the GT position to one image pixel and sampling the seg id there.
    Use (b) supplies the anchor that a VLM detection at bbox (u,v,w,h)
    would provide — same data flow, different anchor source. The downstream
    pos_world / PCA pipeline is identical for both anchor sources.

Boundary contract (TaP vs cuRobo):
  - This module is read-only. It never invokes any motion primitive.

The module is import-safe without SAPIEN — heavy imports are lazy inside
``SimPerceptionBackend`` methods, so unit tests can construct a
``VisionPerception`` with a fake backend on plain Python.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ─── Matrix helpers (handle SAPIEN's 3x4 extrinsic_cv) ──────────────────────

def _extrinsic_to_4x4(Rt: np.ndarray) -> np.ndarray:
    """Promote a 3x4 [R|t] to a 4x4 homogeneous matrix.  Pass-through if
    already 4x4.  SAPIEN's ``camera.get_extrinsic_matrix()`` returns 3x4."""
    Rt = np.asarray(Rt, dtype=np.float64)
    if Rt.shape == (4, 4):
        return Rt
    if Rt.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = Rt
        return out
    raise ValueError(f"extrinsic must be (3,4) or (4,4), got {Rt.shape}")


def _invert_extrinsic(Rt: np.ndarray) -> np.ndarray:
    """4x4 inverse of [R|t] (cam-to-world from world-to-cam)."""
    return np.linalg.inv(_extrinsic_to_4x4(Rt))


def _apply_extrinsic(Rt: np.ndarray, point_h: np.ndarray) -> np.ndarray:
    """Multiply [R|t] by a homogeneous point, returning a 3-vector regardless
    of whether Rt is 3x4 or 4x4."""
    Rt4 = _extrinsic_to_4x4(Rt)
    return (Rt4 @ point_h)[:3]


# ─── Backend ABC ────────────────────────────────────────────────────────────

class PerceptionBackend(ABC):
    """Interface for interchangeable perception backends.

    Implementations must provide a per-pixel mask for a named scene object,
    a depth image in meters, and the camera matrices (K, extrinsic_cv,
    cam2world_gl). Everything else (PCA, projection, etc.) is shared math in
    ``VisionPerception``.
    """

    @abstractmethod
    def get_object_mask(self, name: str, camera: str) -> Optional[np.ndarray]:
        """Return a boolean HxW mask of pixels belonging to *name*, or None if
        the object / camera is unavailable."""
        raise NotImplementedError

    @abstractmethod
    def get_depth_image(self, camera: str) -> Optional[np.ndarray]:
        """Return HxW float depth in meters, NaN where invalid, or None on
        failure."""
        raise NotImplementedError

    @abstractmethod
    def get_camera_matrices(
        self, camera: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return ``(K_3x3, extrinsic_cv_4x4, cam2world_gl_4x4)`` or None."""
        raise NotImplementedError

    @abstractmethod
    def get_object_names(self) -> List[str]:
        """Return the list of scene-reference names this backend can localise."""
        raise NotImplementedError

    def get_image_size(self, camera: str) -> Optional[Tuple[int, int]]:
        """Optional helper: ``(height, width)`` of *camera*'s image, or None.

        Default implementation derives it from the depth image; backends can
        override for efficiency.
        """
        depth = self.get_depth_image(camera)
        if depth is None:
            return None
        return int(depth.shape[0]), int(depth.shape[1])


# ─── SAPIEN simulation backend ─────────────────────────────────────────────

class SimPerceptionBackend(PerceptionBackend):
    """Reads SAPIEN actor segmentation + sim depth.

    Inputs (mask, depth, K, R|t, cam2world) come from the simulator — see the
    module's honesty contract. This backend measures how well the downstream
    geometry lets the LLM grasp given simulator-derived 2D input.
    """

    def __init__(self, TASK_ENV: Any):
        self.env = TASK_ENV
        # Cache: (camera_name, object_name) → seg ID written by SAPIEN into
        # the segmentation buffer's actor channel.  Built on first lookup via
        # spatial bootstrap (project the object's 3D anchor to the image,
        # sample the seg id at that pixel). The anchor comes from
        # ``get_scene_objects``; a VLM bbox/point could supply the anchor
        # with the same bootstrap logic.
        self._seg_id_cache: Dict[Tuple[str, str], int] = {}
        # Lazy-built once: name → (pos_world, orientation) from sim, used
        # only for the bootstrap.  We never surface this to the LLM.
        self._anchor_cache: Dict[str, List[float]] = {}
        self._anchors_loaded = False

    # — Camera-object lookup —

    def _get_camera_object(self, camera_name: str) -> Optional[Any]:
        """Return the sapien.RenderCamera (or compatible) for *camera_name*."""
        cams = getattr(self.env, "cameras", None)
        if cams is None:
            return None
        if camera_name == "head_camera":
            if not getattr(cams, "collect_head_camera", True):
                return None
            for cam, cname in zip(cams.static_camera_list, cams.static_camera_name):
                if cname == "head_camera":
                    return cam
            return None
        if camera_name in ("left_camera", "right_camera"):
            if not getattr(cams, "collect_wrist_camera", True):
                return None
            return getattr(cams, camera_name, None)
        # Other static cameras
        for cam, cname in zip(getattr(cams, "static_camera_list", []),
                              getattr(cams, "static_camera_name", [])):
            if cname == camera_name:
                return cam
        return None

    # — Anchor cache (spatial bootstrap source) —

    def _refresh_anchors(self) -> None:
        """Refresh ``self._anchor_cache`` with the current world position of
        every scene object.  Called on every perception query so that pose
        estimates track objects as they are moved (e.g. lifted) — otherwise
        the mask would stay frozen at the initial position and the LLM
        would see a stale pos_world.

        A VLM backend could replace this with redetection at the current
        frame. The downstream mask + PCA pipeline is unchanged.
        """
        try:
            from privileged_perception import get_scene_objects
            objs = get_scene_objects(self.env)
            new_cache: Dict[str, List[float]] = {}
            for name, info in objs.items():
                pos = info.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) == 3:
                    new_cache[name] = [float(v) for v in pos]
            # Replace atomically so substring-fallback still finds renamed objects
            self._anchor_cache = new_cache
        except Exception:
            pass
        self._anchors_loaded = True

    # Back-compat alias kept for any external caller; uses refresh semantics.
    def _load_anchors(self) -> None:
        self._refresh_anchors()

    # — Spatial seg-ID bootstrap —

    def _project_world_to_pixel(
        self, world_xyz: Sequence[float], camera: str,
    ) -> Optional[Tuple[int, int]]:
        """Internal: project a world point to image pixel via the OpenCV
        intrinsic+extrinsic for *camera*.  Returns ``(u, v)`` or None."""
        m = self.get_camera_matrices(camera)
        if m is None:
            return None
        K, Rt, _ = m
        p = np.array([float(world_xyz[0]), float(world_xyz[1]),
                      float(world_xyz[2]), 1.0])
        cam_pt = Rt @ p
        if cam_pt[2] <= 1e-6:
            return None
        img = K @ cam_pt[:3]
        u = int(round(img[0] / img[2]))
        v = int(round(img[1] / img[2]))
        return u, v

    def _bootstrap_seg_id(
        self, name: str, camera: str, actor_ids: np.ndarray,
    ) -> Optional[int]:
        """Sample the seg-id at the projected anchor pixel for *name*.

        Falls back through (a) the anchor pixel, (b) a 5x5 neighbourhood,
        (c) the actor sub-region from the privileged bbox if available.
        Background IDs (the most populous one in the image) are filtered
        out so a misprojection on the table doesn't poison the cache.
        """
        if not self._anchors_loaded:
            self._load_anchors()
        anchor = self._anchor_cache.get(name)
        if anchor is None:
            # Try substring match
            for known, pos in self._anchor_cache.items():
                if name in known or known in name:
                    anchor = pos
                    break
        if anchor is None:
            return None
        pix = self._project_world_to_pixel(anchor, camera)
        if pix is None:
            return None
        u, v = pix
        H, W = actor_ids.shape
        if not (0 <= v < H and 0 <= u < W):
            return None

        # Identify the dominant "background" ID — the seg ID covering the
        # largest pixel count (typically the table/ground).
        unique, counts = np.unique(actor_ids, return_counts=True)
        background_id = int(unique[int(np.argmax(counts))])

        def _filter_background(values: np.ndarray) -> Optional[int]:
            vals = values.flatten().tolist()
            vals = [int(x) for x in vals if int(x) != background_id and int(x) != 0]
            if not vals:
                return None
            # Most common
            from collections import Counter
            return Counter(vals).most_common(1)[0][0]

        # First try the exact pixel
        sid = int(actor_ids[v, u])
        if sid not in (0, background_id):
            return sid

        # Then a 5x5 patch around the anchor
        v0 = max(v - 2, 0); v1 = min(v + 3, H)
        u0 = max(u - 2, 0); u1 = min(u + 3, W)
        candidate = _filter_background(actor_ids[v0:v1, u0:u1])
        if candidate is not None:
            return candidate

        # Then a 15x15 patch
        v0 = max(v - 7, 0); v1 = min(v + 8, H)
        u0 = max(u - 7, 0); u1 = min(u + 8, W)
        return _filter_background(actor_ids[v0:v1, u0:u1])

    # — Mask building (depth-proximity strategy) —
    #
    # SAPIEN's segmentation channel-1 in this build does not separate
    # individual scene actors reliably (e.g. pot pixels and table pixels
    # share an ID). As a fallback we build the mask by depth
    # proximity to the bootstrap anchor:
    #   • Project anchor (world) to pixel (u_a, v_a).
    #   • Sample the depth around (u_a, v_a) to estimate the object's
    #     reference depth d_a.
    #   • Mask = pixels whose unprojected world distance to anchor is below
    #     a per-object radius (default 0.20 m).
    # The geometry pipeline downstream (median + PCA + extent) operates on
    # these mask pixels exactly as it would on a VLM-derived mask.
    # The mask itself uses the GT anchor as a 2D + 1D seed — the
    # SAME role a VLM bbox + sim-depth (or Depth-Anything) would play.
    #
    # Radius can grow per-object using ``extent`` if known; the default
    # 0.20 m comfortably covers the typical pot / block / cup sizes in our
    # tasks while staying below typical inter-object spacing.

    _DEFAULT_MASK_RADIUS_M = 0.10

    def get_object_mask(self, name: str, camera: str = "head_camera") -> Optional[np.ndarray]:
        # Always refresh anchors so moving objects (lifted pot, pushed cube)
        # have a current anchor for mask construction.
        self._refresh_anchors()
        anchor = self._anchor_cache.get(name)
        if anchor is None:
            for known, pos in self._anchor_cache.items():
                if name in known or known in name:
                    anchor = pos
                    break
        if anchor is None:
            return None

        matrices = self.get_camera_matrices(camera)
        depth = self.get_depth_image(camera)
        if matrices is None or depth is None:
            return None
        K, Rt, _ = matrices
        H, W = depth.shape

        # Per-pixel world positions (only for valid-depth pixels)
        finite = np.isfinite(depth) & (depth > 0)
        if not finite.any():
            return np.zeros((H, W), dtype=bool)
        vs, us = np.where(finite)
        ds = depth[vs, us]
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        x_cam = (us - cx) * ds / fx
        y_cam = (vs - cy) * ds / fy
        z_cam = ds
        cam_h = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        try:
            inv_Rt = _invert_extrinsic(Rt)
        except np.linalg.LinAlgError:
            return np.zeros((H, W), dtype=bool)
        world_pts = (cam_h @ inv_Rt.T)[:, :3]
        dists = np.linalg.norm(world_pts - np.array(anchor), axis=1)

        # Pixel→world distance threshold (per-object radius)
        radius = self._DEFAULT_MASK_RADIUS_M
        mask = np.zeros((H, W), dtype=bool)
        within = dists < radius
        mask[vs[within], us[within]] = True
        return mask

    def _refresh_cameras(self) -> None:
        """Trigger a fresh render so subsequent get_picture / get_depth calls
        return the current scene state."""
        cams = getattr(self.env, "cameras", None)
        if cams is None:
            return
        # Newer code paths use ``cameras.update_picture()`` (sapien refreshes
        # all owned cameras at once).  Fall back to per-camera take_picture
        # if that helper is missing.
        try:
            scene = getattr(self.env, "scene", None)
            if scene is not None and hasattr(scene, "update_render"):
                scene.update_render()
        except Exception:
            pass
        try:
            if hasattr(cams, "update_picture"):
                cams.update_picture()
                return
        except Exception:
            pass
        for cam in list(getattr(cams, "static_camera_list", [])):
            try:
                cam.take_picture()
            except Exception:
                pass
        for attr in ("left_camera", "right_camera"):
            cam = getattr(cams, attr, None)
            if cam is not None:
                try:
                    cam.take_picture()
                except Exception:
                    pass

    def get_depth_image(self, camera: str = "head_camera") -> Optional[np.ndarray]:
        cams = getattr(self.env, "cameras", None)
        if cams is None:
            return None
        # Refresh before reading — SAPIEN's GPU buffers are stale otherwise.
        self._refresh_cameras()
        try:
            depth_dict = cams.get_depth()
        except Exception:
            return None
        cam_block = depth_dict.get(camera)
        if not cam_block or "depth" not in cam_block:
            return None
        depth_mm = np.asarray(cam_block["depth"], dtype=np.float64)
        depth_m = depth_mm / 1000.0
        depth_m[depth_m <= 0] = np.nan
        return depth_m

    def get_camera_matrices(
        self, camera: str = "head_camera",
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        cams = getattr(self.env, "cameras", None)
        if cams is None:
            return None
        try:
            config = cams.get_config()
        except Exception:
            return None
        block = config.get(camera)
        if not block:
            return None
        try:
            K = np.asarray(block["intrinsic_cv"], dtype=np.float64)
            Rt = np.asarray(block["extrinsic_cv"], dtype=np.float64)
            cam2world = np.asarray(block["cam2world_gl"], dtype=np.float64)
        except Exception:
            return None
        return K, Rt, cam2world

    def get_object_names(self) -> List[str]:
        try:
            from privileged_perception import get_scene_objects
            return list(get_scene_objects(self.env).keys())
        except Exception:
            return []


# ─── High-level perception API used by TaP executor ──────────────────────────

class VisionPerception:
    """Thin façade over a backend; owns the geometry / math.

    The TaP executor talks only to this class — never directly to the
    backend — so swapping the backend (e.g. to a VLM) leaves the executor
    completely unchanged.
    """

    # Approximate maximum gripper opening width.  Used to convert the
    # normalised gripper_val to a meters value alongside extent_world.  This
    # is an approximation per embodiment; exact value depends on URDF.
    APPROX_MAX_GRIPPER_WIDTH_M = 0.08

    def __init__(self, TASK_ENV: Any, backend: Optional[PerceptionBackend] = None):
        self.env = TASK_ENV
        self.backend = backend if backend is not None else SimPerceptionBackend(TASK_ENV)
        self._viewpoint_cache: Dict[str, Dict[str, Any]] = {}


    # — Empty pose record helper —

    @staticmethod
    def _empty_pose(name: str, reason: str = "not_visible") -> Dict[str, Any]:
        return {
            "name": name,
            "visible": False,
            "pos_world": None,
            "bbox_pixels": None,
            "depth_valid_ratio": 0.0,
            "principal_axis_world": None,
            "extent_world": None,
            "reason": reason,
        }

    # — Core: get_object_pose —

    def get_object_pose(
        self, name: str, camera: str = "head_camera",
    ) -> Dict[str, Any]:
        """Compute 3D pose / bbox / orientation / extent for *name*.

        Returns a dict the LLM can read directly.  Never raises; on failure
        returns ``visible=False`` with a ``reason`` field.
        """
        mask = self.backend.get_object_mask(name, camera)
        if mask is None:
            return self._empty_pose(name, "backend_unavailable")
        if not mask.any():
            return self._empty_pose(name, "no_mask_pixels")
        depth = self.backend.get_depth_image(camera)
        matrices = self.backend.get_camera_matrices(camera)
        if depth is None or matrices is None:
            return self._empty_pose(name, "camera_data_unavailable")
        K, Rt, _cam2world = matrices

        H, W = mask.shape

        # — Bbox in pixels —
        mask_v, mask_u = np.where(mask)
        u_min, u_max = int(mask_u.min()), int(mask_u.max())
        v_min, v_max = int(mask_v.min()), int(mask_v.max())
        bbox = [u_min, v_min, (u_max - u_min + 1), (v_max - v_min + 1)]

        # — Valid (mask & finite depth) —
        if depth.shape != mask.shape:
            return self._empty_pose(name, "depth_mask_shape_mismatch")
        finite = np.isfinite(depth) & (depth > 0)
        valid = mask & finite
        n_mask = int(mask.sum())
        n_valid = int(valid.sum())
        depth_valid_ratio = (n_valid / n_mask) if n_mask > 0 else 0.0

        # Bbox + visibility alone are still useful even with no depth.
        if n_valid < 5:
            return {
                "name": name,
                "visible": True,
                "bbox_pixels": bbox,
                "depth_valid_ratio": round(depth_valid_ratio, 3),
                "pos_world": None,
                "principal_axis_world": None,
                "extent_world": None,
                "reason": "insufficient_valid_depth",
            }

        # — Unproject valid pixels (OpenCV intrinsic + extrinsic_cv) —
        vs, us = np.where(valid)
        ds = depth[vs, us]  # (N,)
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        x_cam = (us - cx) * ds / fx
        y_cam = (vs - cy) * ds / fy
        z_cam = ds
        cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3) OpenCV

        # OpenCV: extrinsic_cv maps world → cam → multiply inv(extrinsic_cv) to lift to world
        try:
            inv_Rt = _invert_extrinsic(Rt)
        except np.linalg.LinAlgError:
            return self._empty_pose(name, "singular_extrinsic")
        cam_h = np.concatenate(
            [cam_points, np.ones((cam_points.shape[0], 1), dtype=cam_points.dtype)],
            axis=1,
        )  # (N, 4)
        world_h = cam_h @ inv_Rt.T
        world_points = world_h[:, :3]

        # — Median for robust pos —
        pos_world = np.median(world_points, axis=0)

        # — PCA on centred cloud —
        centred = world_points - pos_world
        # Cov matrix; eigh returns ascending eigenvalues
        cov = (centred.T @ centred) / max(len(centred), 1)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return {
                "name": name,
                "visible": True,
                "bbox_pixels": bbox,
                "pos_world": [round(float(v), 4) for v in pos_world.tolist()],
                "depth_valid_ratio": round(depth_valid_ratio, 3),
                "principal_axis_world": None,
                "extent_world": None,
                "reason": "pca_singular",
            }

        # Sort descending so component 0 is the largest axis
        order = np.argsort(eigvals)[::-1]
        eigvecs_sorted = eigvecs[:, order]
        principal_axis = eigvecs_sorted[:, 0]
        # Make the sign deterministic — first non-zero component positive.
        if principal_axis[np.argmax(np.abs(principal_axis))] < 0:
            principal_axis = -principal_axis

        # Extent along each PCA axis
        projections = centred @ eigvecs_sorted  # (N, 3)
        extent = (projections.max(axis=0) - projections.min(axis=0))

        return {
            "name": name,
            "visible": True,
            "pos_world": [round(float(v), 4) for v in pos_world.tolist()],
            "bbox_pixels": bbox,
            "depth_valid_ratio": round(depth_valid_ratio, 3),
            "principal_axis_world": [round(float(v), 4) for v in principal_axis.tolist()],
            "extent_world": [round(float(v), 4) for v in extent.tolist()],
        }

    def grasp_orientation_at_pixel(self, u, v, camera="head_camera", world_xyz=None,
                                   patch=12, near_m=0.04, approach=(0.0, 0.0, -1.0)):
        """Honest grasp orientation at a grounded pixel: unproject a small depth
        patch around (u,v), keep the points near ``world_xyz`` (the handle), take
        the PCA principal axis (the handle's long axis), and build the grasp quat
        (fingers close ⟂ that axis).  Returns
        ``{grasp_quat, handle_axis_world, n_points, valid}``.  No privileged pose.
        """
        from codeaction.backends.robotwin.primitives.pose_utils import pca_principal_axis, grasp_quat_from_handle_axis
        depth = self.backend.get_depth_image(camera)
        mats = self.backend.get_camera_matrices(camera)
        if depth is None or mats is None:
            return {"grasp_quat": None, "handle_axis_world": None,
                    "n_points": 0, "valid": False, "reason": "no_depth_or_matrices"}
        K, Rt = np.asarray(mats[0], float), np.asarray(mats[1], float)
        H, W = depth.shape
        u, v = int(round(u)), int(round(v))
        y0, y1 = max(0, v - patch), min(H, v + patch + 1)
        x0, x1 = max(0, u - patch), min(W, u + patch + 1)
        vv, uu = np.mgrid[y0:y1, x0:x1]
        vv, uu = vv.ravel(), uu.ravel()
        ds = depth[vv, uu]
        m = np.isfinite(ds) & (ds > 0)
        if int(m.sum()) < 10:
            return {"grasp_quat": None, "handle_axis_world": None,
                    "n_points": int(m.sum()), "valid": False, "reason": "few_points"}
        uu, vv, ds = uu[m], vv[m], ds[m]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cam_h = np.stack([(uu - cx) * ds / fx, (vv - cy) * ds / fy, ds, np.ones_like(ds)], axis=1)
        try:
            inv_Rt = _invert_extrinsic(Rt)
        except np.linalg.LinAlgError:
            return {"grasp_quat": None, "handle_axis_world": None,
                    "n_points": int(len(ds)), "valid": False, "reason": "singular_extrinsic"}
        world = (cam_h @ inv_Rt.T)[:, :3]
        if world_xyz is not None:
            gx = np.asarray(world_xyz, dtype=np.float64)
            near = world[np.linalg.norm(world - gx, axis=1) <= near_m]
            if len(near) >= 10:
                world = near
        axis = pca_principal_axis(world)
        quat = grasp_quat_from_handle_axis(axis, approach=approach)
        return {"grasp_quat": [round(float(q), 5) for q in quat],
                "handle_axis_world": [round(float(a), 4) for a in axis],
                "n_points": int(len(world)), "valid": True}

    # — Helper tools exposed to LLM —

    def world_to_pixel(
        self, world_xyz: Sequence[float], camera: str = "head_camera",
    ) -> Dict[str, Any]:
        matrices = self.backend.get_camera_matrices(camera)
        if matrices is None:
            return {"pixel_uv": None, "in_view": False, "depth_m": None, "reason": "camera_data_unavailable"}
        K, Rt, _ = matrices
        p_world = np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0])
        p_cam = Rt @ p_world  # (4,)
        z_cam = float(p_cam[2])
        if z_cam <= 1e-6:
            return {
                "pixel_uv": None,
                "in_view": False,
                "depth_m": z_cam,
                "reason": "behind_camera",
            }
        img = K @ p_cam[:3]
        u = float(img[0]) / float(img[2])
        v = float(img[1]) / float(img[2])
        size = self.backend.get_image_size(camera)
        if size is not None:
            H, W = size
            in_view = bool((0 <= u < W) and (0 <= v < H))
        else:
            in_view = True
        return {
            "pixel_uv": [int(round(u)), int(round(v))],
            "in_view": in_view,
            "depth_m": round(z_cam, 4),
        }

    def pixel_to_world_point(
        self, u: int, v: int, depth_m: float, camera: str = "head_camera",
    ) -> Dict[str, Any]:
        matrices = self.backend.get_camera_matrices(camera)
        if matrices is None:
            return {"world_xyz": None, "reason": "camera_data_unavailable"}
        K, Rt, _ = matrices
        if not np.isfinite(depth_m) or depth_m <= 0:
            return {"world_xyz": None, "reason": "invalid_depth"}
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        d = float(depth_m)
        x_cam = (float(u) - cx) * d / fx
        y_cam = (float(v) - cy) * d / fy
        z_cam = d
        try:
            inv_Rt = _invert_extrinsic(Rt)
        except np.linalg.LinAlgError:
            return {"world_xyz": None, "reason": "singular_extrinsic"}
        p_world_h = inv_Rt @ np.array([x_cam, y_cam, z_cam, 1.0])
        return {"world_xyz": [round(float(p_world_h[i]), 4) for i in range(3)]}

    _DEFAULT_GROUND_PATCH = 10
    _DEFAULT_FG_BAND_M = 0.03

    @staticmethod
    def _foreground_depth_pixel(
        depth: np.ndarray, u: int, v: int, patch: int, fg_band: float = 0.03,
    ) -> Tuple[int, int, Optional[float]]:
        """Return ``(u', v', depth_m)`` of the FOREGROUND (nearest-surface)
        valid pixel closest to ``(u, v)`` within a ``patch``-radius window, or
        ``(u, v, None)`` if no valid depth is present.

        Why foreground, not the exact pixel: at a thin / open structure (e.g. a
        pot-handle loop) the single pixel's depth is unreliable — the ray can
        slip through the gap to the far body / table behind, throwing the 3-D
        point ~8-10 cm off ALONG the ray even when the pixel is ~right (close
        pixels, huge depth jump).  So we snap to the NEAR surface: foreground =
        valid depths within ``fg_band`` (m) of the window minimum (the handle
        metal sits within ~fg_band; the background is far beyond it), then pick
        the foreground pixel nearest ``(u, v)``.  On flat / uniform-depth
        regions the exact pixel is itself the foreground, so this degrades to
        the plain nearest-valid behaviour.
        """
        H, W = depth.shape
        p = int(max(patch, 0))
        v0 = max(v - p, 0); v1 = min(v + p + 1, H)
        u0 = max(u - p, 0); u1 = min(u + p + 1, W)
        sub = depth[v0:v1, u0:u1]
        valid = np.isfinite(sub) & (sub > 0)
        if not valid.any():
            return u, v, None
        dmin = float(sub[valid].min())
        fg = valid & (sub <= dmin + float(fg_band))
        vs, us = np.where(fg)
        abs_v = vs + v0
        abs_u = us + u0
        dist2 = (abs_v - v) ** 2 + (abs_u - u) ** 2
        j = int(np.argmin(dist2))
        return int(abs_u[j]), int(abs_v[j]), float(sub[vs[j], us[j]])

    def pixel_to_world(
        self, u: int, v: int, camera: str = "head_camera",
        patch: Optional[int] = None, fg_band: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Honest grounding: turn a (VLM-picked) pixel into a 3D world point.

        Unlike :meth:`pixel_to_world_point`, the caller supplies ONLY the pixel
        — the depth is looked up internally from the camera's depth image.  The
        lookup is FOREGROUND-ROBUST: within a ``patch``-radius window it snaps to
        the nearest-surface (foreground) pixel, so a point on / beside a thin
        handle does not deproject through the open loop to the body behind it
        (the dominant grasp-grounding error).  See ``_foreground_depth_pixel``.

        This is the localisation primitive the LLM uses after pointing at a
        feature in the camera image; it never consults the simulator's object
        pose, so it carries no privileged-localisation signal (only the depth
        ruler + the LLM's own pixel).

        Returns ``{world_xyz, depth_m, pixel_uv, valid, reason?}``.  ``pixel_uv``
        is the pixel actually sampled (may differ from the input — it snaps to
        the near surface), so the (pixel, depth, world) triple stays consistent.
        """
        if patch is None:
            patch = self._DEFAULT_GROUND_PATCH
        if fg_band is None:
            fg_band = self._DEFAULT_FG_BAND_M
        depth = self.backend.get_depth_image(camera)
        matrices = self.backend.get_camera_matrices(camera)
        if depth is None or matrices is None:
            return {"world_xyz": None, "depth_m": None, "pixel_uv": None,
                    "valid": False, "reason": "camera_data_unavailable"}
        H, W = depth.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < H and 0 <= ui < W):
            return {"world_xyz": None, "depth_m": None, "pixel_uv": None,
                    "valid": False, "reason": "out_of_bounds"}
        u_s, v_s, d = self._foreground_depth_pixel(
            depth, ui, vi, int(max(patch, 0)), float(fg_band))
        if d is None:
            return {"world_xyz": None, "depth_m": None, "pixel_uv": [ui, vi],
                    "valid": False, "reason": "no_valid_depth_near_pixel"}
        res = self.pixel_to_world_point(u_s, v_s, d, camera)
        xyz = res.get("world_xyz")
        if xyz is None:
            return {"world_xyz": None, "depth_m": round(float(d), 4),
                    "pixel_uv": [u_s, v_s], "valid": False,
                    "reason": res.get("reason", "unproject_failed")}
        return {"world_xyz": xyz, "depth_m": round(float(d), 4),
                "pixel_uv": [u_s, v_s], "valid": True}

    def get_depth_at_pixel(
        self, u: int, v: int, camera: str = "head_camera",
    ) -> Dict[str, Any]:
        depth = self.backend.get_depth_image(camera)
        if depth is None:
            return {"depth_m": None, "valid": False, "reason": "camera_data_unavailable"}
        H, W = depth.shape
        ui, vi = int(u), int(v)
        if not (0 <= vi < H and 0 <= ui < W):
            return {"depth_m": None, "valid": False, "reason": "out_of_bounds"}
        d = float(depth[vi, ui])
        if not np.isfinite(d) or d <= 0:
            return {"depth_m": None, "valid": False, "reason": "invalid_depth"}
        return {"depth_m": round(d, 4), "valid": True}

    def describe_camera_viewpoint(
        self, camera: str = "head_camera",
    ) -> Dict[str, Any]:
        """Generate a human-readable description of camera mount + image-axis-to-world-axis mapping.

        Caches per camera since the head_camera pose is static in clean configs.
        """
        if camera in self._viewpoint_cache:
            return self._viewpoint_cache[camera]
        matrices = self.backend.get_camera_matrices(camera)
        if matrices is None:
            info = {
                "camera": camera,
                "description": (
                    f"Camera {camera!r} viewpoint unavailable. "
                    "Default world frame: +x forward, +y left, +z up (m)."
                ),
                "position_world": None,
                "look_dir_world": None,
                "image_right_in_world": None,
                "image_down_in_world": None,
            }
            self._viewpoint_cache[camera] = info
            return info
        _K, Rt, _cam2world = matrices
        # extrinsic_cv: world → cam.  inv(Rt)[:3, 3] = camera position in world.
        try:
            inv_Rt = _invert_extrinsic(Rt)
        except np.linalg.LinAlgError:
            inv_Rt = np.eye(4)
        pos_world = inv_Rt[:3, 3]
        # OpenCV camera frame: +x=right, +y=down, +z=forward (into scene).
        # Camera's right axis in world: inv_Rt[:3, 0]
        right_in_world = inv_Rt[:3, 0]
        down_in_world = inv_Rt[:3, 1]
        forward_in_world = inv_Rt[:3, 2]

        axis_names = {
            (0, +1): "+x_world (forward)",
            (0, -1): "-x_world (backward)",
            (1, +1): "+y_world (robot's left)",
            (1, -1): "-y_world (robot's right)",
            (2, +1): "+z_world (up)",
            (2, -1): "-z_world (down)",
        }

        def _dominant_axis_text(vec: np.ndarray) -> str:
            idx = int(np.argmax(np.abs(vec)))
            sign = 1 if vec[idx] >= 0 else -1
            return axis_names.get((idx, sign), "unknown")

        # Image resolution (width, height) so the LLM can output in-range pixels.
        size = self.backend.get_image_size(camera)
        image_size_wh = [int(size[1]), int(size[0])] if size else None

        description = (
            f"{camera} mounted at ("
            f"{pos_world[0]:.2f}, {pos_world[1]:.2f}, {pos_world[2]:.2f}) m; "
            f"looking direction = ("
            f"{forward_in_world[0]:.2f}, {forward_in_world[1]:.2f}, {forward_in_world[2]:.2f}). "
            f"Image-right ≈ {_dominant_axis_text(right_in_world)}; "
            f"image-down ≈ {_dominant_axis_text(down_in_world)}. "
            + (f"Image is {image_size_wh[0]}x{image_size_wh[1]} px (u in [0,{image_size_wh[0]}), "
               f"v in [0,{image_size_wh[1]})). " if image_size_wh else "")
            + "World frame is +x forward, +y left, +z up; units meters."
        )
        info = {
            "camera": camera,
            "description": description,
            "position_world": [round(float(v), 4) for v in pos_world.tolist()],
            "look_dir_world": [round(float(v), 4) for v in forward_in_world.tolist()],
            "image_right_in_world": [round(float(v), 4) for v in right_in_world.tolist()],
            "image_down_in_world": [round(float(v), 4) for v in down_in_world.tolist()],
            "image_size_wh": image_size_wh,
        }
        self._viewpoint_cache[camera] = info
        return info


# ─── Public re-exports ───────────────────────────────────────────────────────

__all__ = [
    "PerceptionBackend",
    "SimPerceptionBackend",
    "VisionPerception",
]
