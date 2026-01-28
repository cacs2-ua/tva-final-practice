from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2 as cv

Array = np.ndarray


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _to_uint8(x: Array) -> Array:
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    xf -= xf.min()
    denom = xf.max() - xf.min()
    if denom < 1e-6:
        denom = 1.0
    return (255.0 * xf / denom).clip(0, 255).astype(np.uint8)


def _green_channel(img: Array) -> Array:
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 2:
        return _to_uint8(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        return _to_uint8(img[:, :, 1])  # OpenCV BGR -> green
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _clahe_u8(gray: Array, clip_limit: float, tile_grid_size: int) -> Array:
    clahe = cv.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return clahe.apply(gray)


def _make_line_se(length: int, angle_deg: float, thickness: int = 1) -> Array:
    """
    Create a LINE structuring element by drawing a line in a square kernel.

    angle_deg in [0, 180).
    """
    length = int(length)
    thickness = int(max(1, thickness))
    k = length
    k = max(k, 3)
    if k % 2 == 0:
        k += 1

    se = np.zeros((k, k), dtype=np.uint8)
    c = (k - 1) / 2.0
    theta = np.deg2rad(angle_deg)

    half = (k - 1) / 2.0
    dx = np.cos(theta) * half
    dy = np.sin(theta) * half

    x0 = int(round(c - dx))
    y0 = int(round(c - dy))
    x1 = int(round(c + dx))
    y1 = int(round(c + dy))

    cv.line(se, (x0, y0), (x1, y1), color=1, thickness=thickness)
    return se


def _otsu_threshold_1d(values_u8: Array) -> int:
    """
    Otsu threshold computed on a 1D array of uint8 values.
    Returns threshold in [0,255].
    """
    v = values_u8.astype(np.uint8).ravel()
    if v.size == 0:
        return 0

    hist = np.bincount(v, minlength=256).astype(np.float64)
    total = v.size

    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 0

    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break

        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thr = t
    return int(thr)


def _component_elongation(coords_yx: Array) -> float:
    """
    coords_yx: (N,2) array of [y,x].
    Returns sqrt(lambda_max / lambda_min) (>=1). If degenerate => 1.
    """
    if coords_yx.shape[0] < 5:
        return 1.0
    y = coords_yx[:, 0].astype(np.float64)
    x = coords_yx[:, 1].astype(np.float64)
    y -= y.mean()
    x -= x.mean()
    C = np.cov(np.stack([x, y], axis=0))  # 2x2
    vals, _ = np.linalg.eigh(C)
    vals = np.sort(vals)
    lam_min = float(max(vals[0], 1e-9))
    lam_max = float(max(vals[1], 1e-9))
    return float(np.sqrt(lam_max / lam_min))


def _filter_components(
    binary_u8: Array,
    *,
    min_area: int,
    max_area: int,
    min_elongation: float,
    max_extent: float,
    large_area_keep: int,
    verbose: bool = False,
) -> Array:
    """
    Keep components that look vessel-like:
      - area in [min_area, max_area]
      - either elongated enough OR big enough (large_area_keep)
      - extent <= max_extent (avoid filled blobs)
    """
    m = (binary_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(binary_u8, dtype=np.uint8)

    kept = 0
    removed = 0

    for lab in range(1, num):
        area = int(stats[lab, cv.CC_STAT_AREA])
        x = int(stats[lab, cv.CC_STAT_LEFT])
        y = int(stats[lab, cv.CC_STAT_TOP])
        w = int(stats[lab, cv.CC_STAT_WIDTH])
        h = int(stats[lab, cv.CC_STAT_HEIGHT])

        if area < int(min_area) or area > int(max_area):
            removed += 1
            continue

        extent = float(area) / float(max(1, w * h))
        coords = np.column_stack(np.where(labels == lab))  # (N,2) y,x
        elong = _component_elongation(coords)

        keep = True
        if (elong < float(min_elongation)) and (area < int(large_area_keep)):
            keep = False
        if extent > float(max_extent) and elong < (float(min_elongation) + 0.5):
            keep = False

        if keep:
            out[labels == lab] = 255
            kept += 1
        else:
            removed += 1

    if verbose:
        print(f"[CC filter] num_components={num-1} kept={kept} removed={removed}")
    return out


@dataclass
class VesselMorphParams:
    # CLAHE
    use_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile: int = 8

    # Background suppression (morphological opening on inverted image)
    bg_open_frac_of_diag: float = 0.08   # ~77 for DRIVE diag ~960
    bg_open_min: int = 31
    bg_open_max: int = 151

    # Multi-scale / multi-orientation line enhancement
    angles_deg: Tuple[int, ...] = tuple(range(0, 180, 15))  # 12 angles
    line_frac_of_diag: Tuple[float, ...] = (0.012, 0.020, 0.028, 0.036)  # multi-scale lengths
    line_len_min: int = 9
    line_len_max: int = 41
    line_thickness: int = 1

    # Thresholding
    thresh_method: str = "otsu"   # "otsu" or "percentile"
    thresh_percentile: float = 92.0
    thresh_min: int = 5           # avoid tiny thresholds

    # Postprocessing
    post_close_ksize: int = 3
    post_open_ksize: int = 3

    # Connected components filtering
    cc_min_area: int = 25
    cc_max_area_frac: float = 0.15   # max area as fraction of FOV area
    cc_min_elongation: float = 2.2
    cc_max_extent: float = 0.45
    cc_large_area_keep: int = 500

    # Final optional thinning-like cleanup (simple)
    final_erode_iters: int = 0

    # Debug
    verbose: bool = False


def segment_vessels_morphology(
    img_bgr_or_gray: Array,
    fov_mask_u8: Array,
    params: Optional[VesselMorphParams] = None,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Array]]]:
    """
    Pipeline D:
      1) FOV mask
      2) Green channel + optional CLAHE
      3) Invert (vessels become bright)
      4) Background suppression via morphological opening
      5) Multi-orientation + multi-scale white top-hat with line SEs, take max
      6) Threshold
      7) Morph cleanup (close/open)
      8) Connected components + shape filters
    """
    p = params or VesselMorphParams()
    verbose = bool(p.verbose)

    g = _green_channel(img_bgr_or_gray)
    fov = (fov_mask_u8 > 0)

    if g.shape[:2] != fov_mask_u8.shape[:2]:
        raise ValueError(f"Shape mismatch: image={g.shape}, fov_mask={fov_mask_u8.shape}")

    # Fill outside-FOV with median inside-FOV to avoid border artifacts
    inside_vals = g[fov]
    med = int(np.median(inside_vals)) if inside_vals.size else 0
    g_filled = g.copy()
    g_filled[~fov] = med

    if verbose:
        print(f"[Input] shape={g.shape} median_inside_fov={med} "
              f"min/max_inside={int(inside_vals.min())}/{int(inside_vals.max())} "
              f"fov_area={int(fov.sum())}")

    # Optional CLAHE
    if p.use_clahe:
        g_enh = _clahe_u8(g_filled, p.clahe_clip_limit, p.clahe_tile)
    else:
        g_enh = g_filled

    # Invert so vessels become bright
    inv = (255 - g_enh).astype(np.uint8)

    # Background suppression (opening on inv)
    h, w = inv.shape[:2]
    diag = float(np.hypot(h, w))
    bg_k = _ensure_odd(int(round(diag * float(p.bg_open_frac_of_diag))))
    bg_k = int(np.clip(bg_k, p.bg_open_min, p.bg_open_max))
    bg_se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (bg_k, bg_k))
    inv_bg = cv.morphologyEx(inv, cv.MORPH_OPEN, bg_se)
    inv_norm = cv.subtract(inv, inv_bg)  # vessels remain bright

    if verbose:
        v = inv_norm[fov]
        print(f"[BG suppress] bg_k={bg_k} inv_norm min/max/mean={int(v.min())}/{int(v.max())}/{float(v.mean()):.2f}")

    # Build lengths from diag fractions
    lengths: List[int] = []
    for frac in p.line_frac_of_diag:
        L = _ensure_odd(int(round(diag * float(frac))))
        L = int(np.clip(L, p.line_len_min, p.line_len_max))
        lengths.append(L)
    lengths = sorted(set(lengths))

    angles = list(map(int, p.angles_deg))

    if verbose:
        print(f"[Line SE] lengths={lengths} angles={angles} thickness={p.line_thickness}")

    # Multi-orientation, multi-scale white top-hat (bright lines) then max
    best = np.zeros_like(inv_norm, dtype=np.uint8)
    for L in lengths:
        for ang in angles:
            se = _make_line_se(L, ang, thickness=p.line_thickness)
            resp = cv.morphologyEx(inv_norm, cv.MORPH_TOPHAT, se)
            best = cv.max(best, resp)

    resp = best
    resp_in = resp[fov]

    if verbose:
        print(f"[Response] resp min/max/mean inside_fov={int(resp_in.min())}/{int(resp_in.max())}/{float(resp_in.mean()):.2f}")

    # Threshold
    if p.thresh_method.lower() == "otsu":
        thr = _otsu_threshold_1d(resp_in)
        thr = max(int(thr), int(p.thresh_min))
        method_desc = f"otsu(thr={thr})"
    elif p.thresh_method.lower() == "percentile":
        thr = int(np.percentile(resp_in, float(p.thresh_percentile))) if resp_in.size else 0
        thr = max(int(thr), int(p.thresh_min))
        method_desc = f"pct{p.thresh_percentile}(thr={thr})"
    else:
        raise ValueError("thresh_method must be 'otsu' or 'percentile'")

    seg0 = (resp >= thr).astype(np.uint8) * 255
    seg0[~fov] = 0

    if verbose:
        fg = int((seg0 > 0).sum())
        print(f"[Threshold] method={method_desc} foreground_pixels={fg}")

    # Morph cleanup
    ck = _ensure_odd(p.post_close_ksize)
    ok = _ensure_odd(p.post_open_ksize)
    close_se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ck, ck))
    open_se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok, ok))

    seg1 = cv.morphologyEx(seg0, cv.MORPH_CLOSE, close_se)
    seg1 = cv.morphologyEx(seg1, cv.MORPH_OPEN, open_se)
    seg1[~fov] = 0

    # CC filtering
    max_area = int(float(p.cc_max_area_frac) * float(max(1, fov.sum())))
    seg2 = _filter_components(
        seg1,
        min_area=p.cc_min_area,
        max_area=max_area,
        min_elongation=p.cc_min_elongation,
        max_extent=p.cc_max_extent,
        large_area_keep=p.cc_large_area_keep,
        verbose=verbose,
    )

    # Optional final erosion (sometimes helps remove tiny islands)
    seg3 = seg2
    if p.final_erode_iters > 0:
        e_se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        seg3 = cv.erode(seg3, e_se, iterations=int(p.final_erode_iters))
        seg3[~fov] = 0

    debug: Dict[str, Array] = {}
    if return_debug:
        debug = {
            "green": g,
            "green_filled": g_filled,
            "clahe": g_enh,
            "inv": inv,
            "inv_bg": inv_bg,
            "inv_norm": inv_norm,
            "response": resp,
            "seg0_raw": seg0,
            "seg1_morph": seg1,
            "seg2_cc_filtered": seg2,
            "final": seg3,
            "fov": (fov_mask_u8 > 0).astype(np.uint8) * 255,
        }

    return (seg3, debug) if return_debug else seg3
