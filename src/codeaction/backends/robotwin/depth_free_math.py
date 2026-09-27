"""Pure metric math for the depth-free pipeline — every scale = f(known FK motion, image change).

No sim, no VLM here: functions take numbers / VLM box-text and return numbers. This keeps the
module dependency-free and locally unit-testable (tests/test_depth_free_math.py, numpy only).
Knob semantics + the failure→knob map live in docs/depth_free_method_card.md; the knobs themselves
in depth_free_config.DepthFreeConfig."""
import re
import json
import math
from typing import Optional, Sequence, Tuple


# Generic "last balanced JSON object in free text" extractor. Mirrors vlm_client._last_json on
# purpose — this module must NOT import the VLM client (wrong dependency direction + would break the
# "no VLM here" contract and local testability). Keep the two in sync if either changes.
def _last_json(text: str) -> Optional[dict]:
    b = (text or "").strip()
    b = re.sub(r"^```(?:json)?\s*", "", b, flags=re.I)
    b = re.sub(r"\s*```$", "", b).strip()
    try:
        obj = json.loads(b)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for c in reversed(re.findall(r"\{.*?\}", b, flags=re.DOTALL)):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def estimate_depth(fx: float, baseline: float, disparity: float) -> Optional[float]:
    """Z = fx*b/d from a KNOWN same-camera baseline b (FK) and a measured box-centroid disparity d
    (px). None when d≈0 (no measurable shift → cannot invert)."""
    if disparity is None or abs(disparity) < 1e-6:
        return None
    return float(fx) * float(baseline) / float(disparity)


def centroid_disparity(c0: Sequence[float], c1: Sequence[float]) -> float:
    """Pixel distance between two box centroids (the image change from a known move)."""
    return float(math.hypot(c1[0] - c0[0], c1[1] - c0[1]))


def target_baseline(d_star_frac: float, W: int, b0: float, d0: float,
                    b_floor: float, b_cap: float) -> Optional[float]:
    """Disparity targeting: pick baseline b* so the object shifts d* = d_star_frac*W px between
    frames — large enough to spread the epipolar marks, small enough to stay in-frame. Linear in
    disparity: b* = d*·b0/d0, clamped to [b_floor, b_cap]. None when d0≈0."""
    if d0 is None or abs(d0) < 1e-6:
        return None
    d_star = float(d_star_frac) * float(W)
    b_star = d_star * float(b0) / float(d0)
    return float(min(max(b_star, b_floor), b_cap))


def looming_depth(da: float, s0: float, s1: float, min_growth_frac: float) -> Optional[float]:
    """Axial scale-from-motion: descend a KNOWN da (axis-projected FK move), the object's box
    scalar grows s0→s1, so Z0 = da*s1/(s1-s0). None when growth < min_growth_frac*s0 (saturated /
    too close → defer to touch)."""
    if s1 is None or s0 <= 0 or (s1 - s0) < float(min_growth_frac) * float(s0):
        return None
    return float(da) * float(s1) / (float(s1) - float(s0))


def depth_interval(z_est: float, kappa: float, depth_min: float, depth_max: float
                   ) -> Tuple[float, float]:
    """Pick search interval [z_est/kappa, z_est*kappa] clamped to the reachable depth band."""
    lo = max(float(depth_min), float(z_est) / float(kappa))
    hi = min(float(depth_max), float(z_est) * float(kappa))
    return float(lo), float(hi)


def box_from_text(text: str, W: int, H: int, norm: float = 1000.0) -> Optional[list]:
    """Extract {"box":[x1,y1,x2,y2]} (0..norm normalised) from VLM free text → pixel
    [x1,y1,x2,y2] with x1<x2, y1<y2. None if absent/malformed."""
    obj = _last_json(text)
    b = obj.get("box") if isinstance(obj, dict) else None
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return None
    try:
        xs = sorted([float(b[0]) / norm * W, float(b[2]) / norm * W])
        ys = sorted([float(b[1]) / norm * H, float(b[3]) / norm * H])
    except (TypeError, ValueError):
        return None
    return [xs[0], ys[0], xs[1], ys[1]]


def box_centroid(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_diag(box: Sequence[float]) -> float:
    """Near-isotropic size scalar (diagonal) — less sensitive to one clipped side than sqrt(area)."""
    return float(math.hypot(box[2] - box[0], box[3] - box[1]))
