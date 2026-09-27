"""Typed scale tools (spec §3/§4). Each returns an Estimate with mandatory uncertainty + coarse flag +
provenance. The HARNESS never decides whether an estimate is 'good enough' (spec §0.1) — it just
returns it; the model chooses whether to verify further. Reuses servo/depth_free_math for the math.
Uncertainty is DERIVED from a declared error model (spec §4), never a flat confidence constant: the
pixel sources propagate their pixel noise through Z = fx*L/px, while the object-size source uses
the caller's extent sigma and keeps omission explicitly unknown."""
import math

import numpy as np

from codeaction.contracts.types import Estimate

_SIGMA_PX = 2.0          # motion/ruler producers not yet migrated to caller-owned uncertainty


def _import_dfm():
    from codeaction.backends.robotwin import depth_free_math as dfm
    return dfm


def scale_from_motion(fx, baseline_achieved, disparity_px) -> Estimate:
    """Z = fx*b/d. b is the ACHIEVED baseline (from an ActionResult.achieved, never the command).
    sigma_Z = Z*sigma_px/d — a small disparity yields an honestly large uncertainty."""
    dfm = _import_dfm()
    z = dfm.estimate_depth(fx, baseline_achieved, disparity_px)
    unc = abs(z) * _SIGMA_PX / abs(float(disparity_px)) if z else float("inf")
    return Estimate(value=z, kind="depth", uncertainty=unc, coarse=False,
                    provenance={"formula": "fx*b/d", "baseline_achieved": baseline_achieved,
                                "disparity_px": disparity_px, "sigma_px": _SIGMA_PX})


def scale_from_object_size(fx, fy, bbox, extent_axis, known_extent_m,
                           known_extent_sigma_m=None) -> Estimate:
    """COARSE depth from the caller's real prior for one selected projected bbox extent."""
    fx, fy = float(fx), float(fy)
    coords = [float(value) for value in bbox]
    extent_m = float(known_extent_m)
    sigma_m = None if known_extent_sigma_m is None else float(known_extent_sigma_m)
    if len(coords) != 4 or not all(math.isfinite(value) for value in (fx, fy, *coords)):
        raise ValueError("fx, fy and bbox must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    if extent_axis not in ("width", "height", "diagonal"):
        raise ValueError("extent_axis must be width, height or diagonal")
    if not math.isfinite(extent_m) or extent_m <= 0.0:
        raise ValueError("known_extent_m must be finite and positive")
    if sigma_m is not None and (not math.isfinite(sigma_m) or sigma_m < 0.0):
        raise ValueError("known_extent_sigma_m must be finite and non-negative")
    width_normalized = abs(coords[2] - coords[0]) / fx
    height_normalized = abs(coords[3] - coords[1]) / fy
    normalized_extent = {
        "width": width_normalized,
        "height": height_normalized,
        "diagonal": math.hypot(width_normalized, height_normalized),
    }[extent_axis]
    z = extent_m / normalized_extent if normalized_extent > 0.0 else None
    relative_sigma = sigma_m / extent_m if sigma_m is not None else None
    unc = abs(z) * relative_sigma if z is not None and relative_sigma is not None else float("inf")
    provenance = {
        "extent_axis": extent_axis,
        "known_extent_m": extent_m,
        "known_extent_sigma_m": sigma_m,
        "normalized_image_extent": normalized_extent,
        "fx": fx,
        "fy": fy,
        "source": "model_prior",
        "relative_extent_sigma": relative_sigma,
        "method": "pinhole projected extent prior",
        "uncertainty_scope": "caller extent prior only",
        "uncertainty_excludes": [
            "camera calibration", "bbox annotation", "size-prior shape/view mismatch"],
    }
    if z is None:
        provenance["note"] = "selected bbox extent is zero"
    return Estimate(value=z, kind="depth", uncertainty=unc, coarse=True,
                    provenance=provenance)


def scale_from_gripper(fx, opening_m, span_px) -> Estimate:
    """Depth using the known gripper opening as an in-image ruler: Z = fx*opening/span.
    sigma_Z = Z*sigma_px/span. Declared assumption: the fingertip segment is ~parallel to the
    image plane (tilt foreshortens span_px and overestimates Z)."""
    z = float(fx) * float(opening_m) / float(span_px) if span_px else None
    unc = abs(z) * _SIGMA_PX / abs(float(span_px)) if z else float("inf")
    return Estimate(value=z, kind="depth", uncertainty=unc, coarse=False,
                    provenance={"formula": "fx*opening/span", "opening_m": opening_m,
                                "span_px": span_px, "sigma_px": _SIGMA_PX,
                                "assumes": "fronto-parallel fingertip segment"})


def _world_ray(K, E, px):
    E = np.asarray(E, dtype=float)[:3]
    R, t = E[:3, :3], E[:3, 3]
    C = -R.T @ t
    d = R.T @ (np.linalg.inv(np.asarray(K, dtype=float))
               @ np.asarray([float(px[0]), float(px[1]), 1.0]))
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        return None, None
    return C, d / norm


def _two_ray_midpoint(K0, E0, px0, K1, E1, px1):
    C0, d0 = _world_ray(K0, E0, px0)
    C1, d1 = _world_ray(K1, E1, px1)
    if C0 is None or C1 is None:
        return None
    dot = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
    A = np.asarray([[1.0, -dot], [dot, -1.0]], dtype=float)
    rhs = np.asarray([np.dot(d0, C1 - C0), np.dot(d1, C1 - C0)], dtype=float)
    if not np.all(np.isfinite(A)) or float(np.linalg.cond(A)) > 1e12:
        return None
    try:
        s, t = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    p0, p1 = C0 + float(s) * d0, C1 + float(t) * d1
    point = 0.5 * (p0 + p1)
    E0a, E1a = np.asarray(E0, dtype=float)[:3], np.asarray(E1, dtype=float)[:3]
    z0 = float((E0a[:3, :3] @ point + E0a[:3, 3])[2])
    z1 = float((E1a[:3, :3] @ point + E1a[:3, 3])[2])
    if z0 <= 1e-6 or z1 <= 1e-6:
        return None
    angle = math.degrees(math.acos(float(np.clip(abs(dot), 0.0, 1.0))))
    return {"point": point, "depth_before_m": z0, "depth_after_m": z1,
            "ray_gap_m": float(np.linalg.norm(p0 - p1)),
            "triangulation_angle_deg": angle}


def triangulate_correspondence(K0, E0, px0, K1, E1, px1, pixel_sigma_px) -> Estimate:
    """General calibrated two-view triangulation for model-selected corresponding pixels.

    The uncertainty is derived by perturbing each supplied image coordinate by the declared
    annotation noise and measuring the resulting 3D displacement.  Half of the closest-ray gap
    is also retained as an observed cross-view consistency term.
    """
    sigma_px = float(pixel_sigma_px)
    if not math.isfinite(sigma_px) or sigma_px < 0.0:
        raise ValueError("pixel_sigma_px must be finite and non-negative")
    base = _two_ray_midpoint(K0, E0, px0, K1, E1, px1)
    provenance = {
        "px_before": [float(px0[0]), float(px0[1])],
        "px_after": [float(px1[0]), float(px1[1])],
        "pixel_sigma_px": sigma_px,
        "method": "two-ray midpoint",
        "uncertainty_scope": "pixel perturbation + ray gap",
        "uncertainty_excludes": [
            "camera calibration", "caller correspondence choice", "scene change"],
    }
    if base is None:
        return Estimate(value=None, kind="point", uncertainty=float("inf"), coarse=False,
                        provenance={**provenance, "note": "degenerate or behind-camera rays"})

    deviations = []
    offsets = ((-sigma_px, 0.0), (sigma_px, 0.0),
               (0.0, -sigma_px), (0.0, sigma_px))
    for which in (0, 1):
        for du, dv in offsets:
            p0 = [float(px0[0]), float(px0[1])]
            p1 = [float(px1[0]), float(px1[1])]
            target = p0 if which == 0 else p1
            target[0] += du
            target[1] += dv
            perturbed = _two_ray_midpoint(K0, E0, p0, K1, E1, p1)
            if perturbed is not None:
                deviations.append(float(np.linalg.norm(
                    perturbed["point"] - base["point"])))
    pixel_unc = max(deviations) if deviations else float("inf")
    ray_gap_term = 0.5 * float(base["ray_gap_m"])
    uncertainty = max(pixel_unc, ray_gap_term)
    return Estimate(
        value=base["point"].tolist(), kind="point", uncertainty=uncertainty, coarse=False,
        provenance={**provenance,
                    "depth_before_m": base["depth_before_m"],
                    "depth_after_m": base["depth_after_m"],
                    "ray_gap_m": base["ray_gap_m"],
                    "triangulation_angle_deg": base["triangulation_angle_deg"],
                    "pixel_perturbation_uncertainty_m": pixel_unc,
                    "ray_gap_uncertainty_m": ray_gap_term})
