from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask

Array = np.ndarray


# -----------------------------
# Debug helpers
# -----------------------------
def _dbg(verbose: bool, *args):
    if verbose:
        print(*args)


def _to_uint8(x: Array) -> Array:
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    xf -= xf.min()
    den = float(xf.max() - xf.min())
    if den < 1e-6:
        den = 1.0
    y = (255.0 * (xf / den)).clip(0, 255).astype(np.uint8)
    return y


def _get_green_or_gray(img: Array) -> Array:
    if img.ndim == 2:
        return _to_uint8(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        # OpenCV loads BGR; green channel usually best for vessels
        return _to_uint8(img[:, :, 1])
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _masked_otsu_threshold(values_u8: Array) -> int:
    """
    Compute Otsu threshold from a 1D uint8 array of values (already masked).
    """
    if values_u8.size == 0:
        return 128
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = float(values_u8.size)

    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    w_f = 0.0

    var_max = -1.0
    thr = 128

    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break

        sum_b += float(t * hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f)
        if var_between > var_max:
            var_max = var_between
            thr = t

    return int(thr)


def _line_kernel(length: int, angle_deg: float) -> Array:
    """
    Create a 2D line structuring element of given odd length and orientation.
    """
    length = int(length)
    if length < 3:
        length = 3
    if length % 2 == 0:
        length += 1

    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2

    # Endpoint vector
    rad = np.deg2rad(angle_deg)
    dx = int(round((length // 2) * np.cos(rad)))
    dy = int(round((length // 2) * np.sin(rad)))

    x1, y1 = c - dx, c - dy
    x2, y2 = c + dx, c + dy

    cv.line(k, (x1, y1), (x2, y2), color=1, thickness=1)
    return k


def _retinex_bilateral(gray_u8: Array, *, d: int = 5, sigma_color: float = 35.0, sigma_space: float = 35.0) -> Array:
    """
    Retinex-style illumination correction inspired by the paper:
      R(x) = log(I+1) - log(L+1), where L is edge-preserving (bilateral) smooth.
    Output: uint8 in [0,255].
    """
    I = gray_u8.astype(np.float32)
    L = cv.bilateralFilter(gray_u8, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space)).astype(np.float32)

    R = np.log1p(I) - np.log1p(L)
    R = _to_uint8(R)
    return R


def _clahe(gray_u8: Array, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Array:
    clahe = cv.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray_u8)


def _multiscale_multiorient_blackhat(
    gray_u8: Array,
    *,
    lengths: List[int],
    angles_deg: List[float],
    blur_ksize: int = 3,
) -> Array:
    """
    Vesselness-like map using multi-scale, multi-orientation morphological BLACKHAT.
    Dark vessels on brighter background -> blackhat highlights them.
    """
    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        g = cv.GaussianBlur(gray_u8, (blur_ksize, blur_ksize), 0)
    else:
        g = gray_u8

    best = np.zeros_like(g, dtype=np.uint8)

    for L in lengths:
        for a in angles_deg:
            k = _line_kernel(L, a)
            bh = cv.morphologyEx(g, cv.MORPH_BLACKHAT, k)
            best = cv.max(best, bh)

    return best


@dataclass
class CCFilterStats:
    kept: int
    removed: int
    total: int
    removed_reasons: Dict[str, int]


def _component_shape_metrics(component_mask_u8: Array) -> Dict[str, float]:
    """
    Compute basic shape descriptors for a single component mask (0/1 or 0/255).
    Returns: area, perimeter, circularity, elongation, bbox_ar, extent
    """
    m = (component_mask_u8 > 0).astype(np.uint8)
    area = float(m.sum())

    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "area": area,
            "perimeter": 0.0,
            "circularity": 1.0,
            "elongation": 1.0,
            "bbox_ar": 1.0,
            "extent": 1.0,
        }

    cnt = max(contours, key=cv.contourArea)
    per = float(cv.arcLength(cnt, True))

    # circularity: 1 for circle, lower for elongated
    circ = (4.0 * np.pi * area) / (per * per + 1e-6)

    x, y, w, h = cv.boundingRect(cnt)
    bbox_ar = float(max(w, h)) / float(min(w, h) + 1e-6)
    extent = float(area) / float(w * h + 1e-6)

    # elongation via PCA on coordinates
    ys, xs = np.where(m > 0)
    if xs.size < 5:
        elong = bbox_ar
    else:
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        pts -= pts.mean(axis=0, keepdims=True)
        cov = (pts.T @ pts) / float(max(1, pts.shape[0] - 1))
        evals, _ = np.linalg.eigh(cov)
        evals = np.sort(evals)  # ascending
        # major/minor axis ratio
        elong = float(np.sqrt((evals[1] + 1e-9) / (evals[0] + 1e-9)))

    return {
        "area": float(area),
        "perimeter": float(per),
        "circularity": float(circ),
        "elongation": float(elong),
        "bbox_ar": float(bbox_ar),
        "extent": float(extent),
    }


def _filter_connected_components(
    bin_mask_u8: Array,
    *,
    min_area: int = 30,
    min_area_if_elongated: int = 15,
    min_elongation: float = 2.2,
    max_circularity: float = 0.38,
    max_extent_blob: float = 0.70,
    connectivity: int = 8,
    verbose: bool = False,
) -> Tuple[Array, CCFilterStats]:
    """
    Pipeline E core: CC filtering as structural enforcement.
    We REMOVE components that are:
      - too small
      - too round/compact (blob-like): high circularity + low elongation OR high extent
    """
    m = (bin_mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=connectivity)

    removed_reasons: Dict[str, int] = {}
    out = np.zeros_like(m, dtype=np.uint8)

    total_cc = max(0, num - 1)
    kept = 0
    removed = 0

    _dbg(verbose, f"[CC] total components (excluding bg) = {total_cc}")

    for cc_id in range(1, num):
        area = int(stats[cc_id, cv.CC_STAT_AREA])
        cc_mask = (labels == cc_id).astype(np.uint8)

        metrics = _component_shape_metrics(cc_mask)

        # Rules:
        # 1) tiny noise
        if area < int(min_area_if_elongated):
            removed += 1
            removed_reasons["too_small_hard"] = removed_reasons.get("too_small_hard", 0) + 1
            continue

        # 2) small but not elongated enough
        if area < int(min_area) and metrics["elongation"] < float(min_elongation):
            removed += 1
            removed_reasons["too_small_not_elongated"] = removed_reasons.get("too_small_not_elongated", 0) + 1
            continue

        # 3) blob-like: round + not elongated
        if metrics["circularity"] > float(max_circularity) and metrics["elongation"] < float(min_elongation):
            removed += 1
            removed_reasons["too_round_compact"] = removed_reasons.get("too_round_compact", 0) + 1
            continue

        # 4) very “filled” bbox -> tends to be lesions/blobs rather than line structures
        if metrics["extent"] > float(max_extent_blob) and metrics["elongation"] < float(min_elongation):
            removed += 1
            removed_reasons["high_extent_blob"] = removed_reasons.get("high_extent_blob", 0) + 1
            continue

        # Keep
        out[labels == cc_id] = 1
        kept += 1

    out_u8 = (out * 255).astype(np.uint8)

    _dbg(
        verbose,
        f"[CC] kept={kept} removed={removed} total={total_cc} reasons={removed_reasons}"
    )

    return out_u8, CCFilterStats(kept=kept, removed=removed, total=total_cc, removed_reasons=removed_reasons)


# -----------------------------
# Pipeline E main API
# -----------------------------
def vessel_segmentation(
    input_image: Union[str, Array],
    *,
    verbose: bool = False,
    # Upstream segmenter knobs
    retinex_d: int = 5,
    retinex_sigma_color: float = 35.0,
    retinex_sigma_space: float = 35.0,
    clahe_clip: float = 2.2,
    clahe_tile: Tuple[int, int] = (8, 8),
    bh_lengths: Tuple[int, ...] = (9, 13, 17),
    bh_angle_step_deg: int = 15,
    bh_blur_ksize: int = 3,
    # Binarization knobs
    thr_offset: int = 0,  # shift Otsu a bit (+ -> stricter, - -> more vessels)
    post_open_ksize: int = 3,
    post_close_ksize: int = 3,
    # CC filter knobs (Pipeline E core)
    cc_min_area: int = 30,
    cc_min_area_if_elongated: int = 15,
    cc_min_elongation: float = 2.2,
    cc_max_circularity: float = 0.38,
    cc_max_extent_blob: float = 0.70,
) -> Array:
    """
    Pipeline E:
      Upstream unsupervised segmenter (Retinex-ish + multi-orient black-hat + threshold)
      THEN Connected Components filtering as structural enforcement.

    Returns:
      segmented_image: uint8 0/255 (same HxW as input)
    """
    # Load
    if isinstance(input_image, str):
        img = cv.imread(input_image, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_image}")
    else:
        img = input_image

    # FOV
    fov = compute_fov_mask(img)
    fov_bool = (fov > 0)
    _dbg(verbose, f"[E] FOV coverage = {fov_bool.mean():.3f}")

    g = _get_green_or_gray(img)

    # Retinex-style correction
    r = _retinex_bilateral(
        g,
        d=retinex_d,
        sigma_color=retinex_sigma_color,
        sigma_space=retinex_sigma_space,
    )
    _dbg(verbose, f"[E] Retinex: min={int(r.min())} max={int(r.max())} mean={float(r.mean()):.2f}")

    # Local contrast
    c = _clahe(r, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
    _dbg(verbose, f"[E] CLAHE: min={int(c.min())} max={int(c.max())} mean={float(c.mean()):.2f}")

    # Vesselness-ish via multiscale multi-orientation black-hat
    angles = list(np.arange(0, 180, int(bh_angle_step_deg), dtype=np.int32).tolist())
    vesselness = _multiscale_multiorient_blackhat(
        c,
        lengths=list(map(int, bh_lengths)),
        angles_deg=list(map(float, angles)),
        blur_ksize=int(bh_blur_ksize),
    )

    # Normalize inside FOV (for stable thresholding)
    v = vesselness.copy()
    v[~fov_bool] = 0

    vals = v[fov_bool].astype(np.uint8)
    thr = _masked_otsu_threshold(vals)
    thr = int(np.clip(thr + int(thr_offset), 0, 255))

    p50 = int(np.percentile(vals, 50)) if vals.size else 0
    p90 = int(np.percentile(vals, 90)) if vals.size else 0
    _dbg(verbose, f"[E] Vesselness: p50={p50} p90={p90} otsu_thr={thr} (offset={thr_offset})")

    initial = (v > thr).astype(np.uint8) * 255
    initial[~fov_bool] = 0

    _dbg(verbose, f"[E] Initial bin: fg_frac={float((initial>0).mean()):.4f}")

    # Light morphology to reduce specks & reconnect micro-gaps (still upstream)
    if post_open_ksize and post_open_ksize >= 3:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (post_open_ksize, post_open_ksize))
        initial = cv.morphologyEx(initial, cv.MORPH_OPEN, k)

    if post_close_ksize and post_close_ksize >= 3:
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (post_close_ksize, post_close_ksize))
        initial = cv.morphologyEx(initial, cv.MORPH_CLOSE, k)

    _dbg(verbose, f"[E] After morph: fg_frac={float((initial>0).mean()):.4f}")

    # -----------------------------
    # Pipeline E KEY step: CC filtering
    # -----------------------------
    filtered, cc_stats = _filter_connected_components(
        initial,
        min_area=cc_min_area,
        min_area_if_elongated=cc_min_area_if_elongated,
        min_elongation=cc_min_elongation,
        max_circularity=cc_max_circularity,
        max_extent_blob=cc_max_extent_blob,
        connectivity=8,
        verbose=verbose,
    )

    # Optional final gentle reconnect (kept minimal to not bloat vessels)
    # (If you want, you can raise post_close_ksize instead of doing more here.)
    filtered[~fov_bool] = 0
    filtered = (filtered > 0).astype(np.uint8) * 255

    _dbg(verbose, f"[E] Final: fg_frac={float((filtered>0).mean()):.4f} dtype={filtered.dtype} unique={np.unique(filtered)[:5]}")

    return filtered
