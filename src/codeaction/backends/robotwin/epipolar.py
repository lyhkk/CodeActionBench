"""
Epipolar depth from a known wrist-camera nudge (GT-free) + the colored depth-tick drawing.

Two-view geometry, ported from axis_bench/s2a_link.py + s2a_epipolar_ticks.py (kept self-contained
here so the servo loop does not import axis_bench internals). Camera matrices follow
VisionPerception.get_camera_matrices: K (3x3 OpenCV intrinsic) and Ext = extrinsic_cv world->cam
[R|t] (3x4 or 4x4; we use the top 3 rows).

Pipeline (used by wrist_grasp_loop's `epi` source):
  uv0 = VLM points the target in wrist frame0 (the anchor ray)
  known lateral nudge (move_delta along the camera's image-right) -> wrist frame1
  uv1 = VLM points the same target in frame1
  depth = two-view triangulation (continuous) AND reach-bounded snap (the `approach` mode: snap uv1
          to the nearest candidate along the anchor ray within [reach_min, reach] -> can't blow up)
  H = back_project(uv0, depth)   # the grasp point, in world; no GT / no rendered depth
draw_depth_ticks renders the colored candidate dots on frame1 (the "different-colour depth-value
points" image) so the estimate is auditable.
"""
from __future__ import annotations

import numpy as np

# A small, distinctly-nameable palette for the depth ticks (near = cool, far = warm).
COLORS = [
    ("blue", (40, 90, 230)), ("cyan", (0, 190, 200)), ("green", (40, 175, 70)),
    ("lime", (150, 205, 40)), ("yellow", (240, 210, 30)), ("orange", (240, 150, 30)),
    ("red", (225, 55, 55)), ("magenta", (210, 60, 200)), ("white", (245, 245, 245)),
]

# PICK palette: deliberately EXCLUDES the red family so the tick names never collide with a RED
# target object (the s2a color-name confound: a red object makes the VLM mis-pick the "red" dot).
PICK_COLORS = [
    ("blue", (30, 90, 235)), ("cyan", (0, 195, 205)), ("green", (35, 170, 70)),
    ("lime", (150, 205, 45)), ("yellow", (240, 215, 30)), ("white", (250, 250, 250)),
    ("black", (15, 15, 15)),
]


def pick_candidates(K0, E0, uv0, K1, E1, depth_lo, depth_hi, n):
    """n evenly-spaced candidate depths in [depth_lo, depth_hi] with their frame-1 projections.
    Returns a list of dicts {color, depth, uv} for those that project in front of cam1."""
    out = []
    zs = np.linspace(depth_lo, depth_hi, n)
    for i, z in enumerate(zs):
        p = project(K1, E1, back_project(K0, E0, uv0, float(z)))
        if p is None:
            continue
        out.append({"color": PICK_COLORS[i % len(PICK_COLORS)][0], "depth": float(z),
                    "uv": [float(p[0]), float(p[1])]})
    return out


def draw_pick_ticks(img_in, img_out, cands, uv0_note=None):
    """Draw the PICK candidates as labelled colored dots along the epipolar line (the auditable
    'colored depth-value points' frame the VLM picks from). Best-effort; never raises."""
    try:
        from PIL import Image, ImageDraw
        cmap = {n: rgb for n, rgb in PICK_COLORS}
        im = Image.open(img_in).convert("RGB")
        d = ImageDraw.Draw(im)
        pts = [tuple(map(int, map(round, c["uv"]))) for c in cands]
        if len(pts) >= 2:
            d.line(pts, fill=(120, 120, 120), width=1)
        for c, (u, v) in zip(cands, pts):
            rgb = cmap.get(c["color"], (200, 200, 200))
            d.ellipse([u - 7, v - 7, u + 7, v + 7], fill=rgb, outline=(255, 255, 255), width=2)
            d.text((u + 9, v - 6), f'{c["color"]} {c["depth"]*100:.0f}cm', fill=rgb)
        im.save(img_out)
        return True
    except Exception as e:
        print(f"[epi] draw_pick_ticks failed: {e}", flush=True)
        return False


def parse_pick_color(text, names):
    """Pull a color name from the VLM's (free-text + JSON tail) answer; returns index in `names`
    or -1. Tolerates synonyms."""
    import re as _re, json as _json
    b = (text or "").strip()
    b = _re.sub(r"^```(?:json)?\s*", "", b, flags=_re.I)
    b = _re.sub(r"\s*```$", "", b).strip()
    obj = None
    try:
        obj = _json.loads(b)
    except Exception:
        for c in reversed(_re.findall(r"\{.*?\}", b, flags=_re.DOTALL)):
            try:
                obj = _json.loads(c); break
            except Exception:
                continue
    syn = {"lightblue": "cyan", "teal": "cyan", "sky": "cyan", "gold": "yellow", "grey": "black",
           "gray": "black", "dark": "black"}
    col = ""
    if isinstance(obj, dict):
        col = str(obj.get("color", "")).strip().lower()
    col = syn.get(col.replace(" ", ""), col)
    return names.index(col) if col in names else -1


def back_project(K0, Ext0, uv0, Z):
    """World point on camera-0's ray through pixel uv0 at camera-z depth Z."""
    E0 = np.asarray(Ext0)[:3]
    R0, t0 = E0[:3, :3], E0[:3, 3]
    ray = np.linalg.inv(np.asarray(K0)) @ np.array([uv0[0], uv0[1], 1.0])
    return R0.T @ (Z * ray - t0)


def project(K1, Ext1, Xw):
    """Project world point Xw into camera 1; None if behind the camera."""
    E1 = np.asarray(Ext1)[:3]
    p = E1[:3, :3] @ np.asarray(Xw, float) + E1[:3, 3]
    if p[2] <= 1e-6:
        return None
    img = np.asarray(K1) @ p
    return np.array([img[0] / img[2], img[1] / img[2]])


def triangulate_point(K0, E0, uv0, K1, E1, uv1):
    """General two-view midpoint triangulation (any known camera motion). Returns
    (world X, depth in cam0) or (None, None) on a degenerate configuration."""
    E0 = np.asarray(E0)[:3]; R0, t0 = E0[:3, :3], E0[:3, 3]; C0 = -R0.T @ t0
    E1 = np.asarray(E1)[:3]; R1, t1 = E1[:3, :3], E1[:3, 3]; C1 = -R1.T @ t1
    d0 = R0.T @ (np.linalg.inv(np.asarray(K0)) @ np.array([uv0[0], uv0[1], 1.0])); d0 /= np.linalg.norm(d0)
    d1 = R1.T @ (np.linalg.inv(np.asarray(K1)) @ np.array([uv1[0], uv1[1], 1.0])); d1 /= np.linalg.norm(d1)
    A = np.array([[d0 @ d0, -d0 @ d1], [d0 @ d1, -d1 @ d1]])
    bb = np.array([d0 @ (C1 - C0), d1 @ (C1 - C0)])
    try:
        st = np.linalg.solve(A, bb)
    except np.linalg.LinAlgError:
        return None, None
    X = ((C0 + st[0] * d0) + (C1 + st[1] * d1)) / 2.0
    return X, float((R0 @ X + t0)[2])


def reach_snap_depth(K0, E0, uv0, K1, E1, uv1, reach_min, reach, reach_k):
    """`approach` mode: place candidate depths in [reach_min, reach] along cam0's ray through uv0,
    project each into cam1, and snap to the candidate whose projection is nearest uv1. Returns
    (snapped_depth, perp_px, candidates[(Z, proj_uv)], picked_idx). Reach-bounded → never blows up."""
    allz = np.linspace(reach_min, reach, reach_k)
    cands = []
    for z in allz:
        p = project(K1, E1, back_project(K0, E0, uv0, float(z)))
        if p is not None:
            cands.append((float(z), p))
    if not cands:
        return None, None, [], -1
    arr = np.array([p for _, p in cands])
    d2 = np.linalg.norm(arr - np.asarray(uv1, float), axis=1)
    j = int(np.argmin(d2))
    return cands[j][0], float(d2[j]), cands, j


def draw_depth_ticks(img_in, img_out, K0, E0, uv0, K1, E1, uv1, depth_lo, depth_hi,
                     n_ticks=9, picked_depth=None):
    """Draw n_ticks evenly-spaced colored candidate dots (with depth labels in cm) along the anchor
    ray's projection into frame1, mark the VLM's uv1, and ring the candidate nearest the picked
    depth. Saves the auditable "colored depth-value points" frame. Best-effort; never raises."""
    try:
        from PIL import Image, ImageDraw
        im = Image.open(img_in).convert("RGB")
        d = ImageDraw.Draw(im)
        zs = np.linspace(depth_lo, depth_hi, n_ticks)
        pts = []
        for z in zs:
            p = project(K1, E1, back_project(K0, E0, uv0, float(z)))
            pts.append(None if p is None else (float(p[0]), float(p[1])))
        live = [p for p in pts if p is not None]
        if len(live) >= 2:
            d.line(live, fill=(20, 20, 20), width=1)
        for i, (z, p) in enumerate(zip(zs, pts)):
            if p is None:
                continue
            c = COLORS[i % len(COLORS)][1]
            u, v = int(round(p[0])), int(round(p[1]))
            d.ellipse([u - 6, v - 6, u + 6, v + 6], fill=c, outline=(255, 255, 255), width=1)
            d.text((u + 8, v - 6), f"{z*100:.0f}", fill=c)
        # picked candidate ring
        if picked_depth is not None:
            pp = project(K1, E1, back_project(K0, E0, uv0, float(picked_depth)))
            if pp is not None:
                u, v = int(round(pp[0])), int(round(pp[1]))
                d.ellipse([u - 10, v - 10, u + 10, v + 10], outline=(0, 0, 0), width=2)
        # VLM's frame-1 point (the snap target)
        if uv1 is not None:
            u, v = int(round(uv1[0])), int(round(uv1[1]))
            d.line([u - 8, v, u + 8, v], fill=(0, 0, 0), width=2)
            d.line([u, v - 8, u, v + 8], fill=(0, 0, 0), width=2)
        im.save(img_out)
        return True
    except Exception as e:
        print(f"[epi] draw_depth_ticks failed: {e}", flush=True)
        return False
