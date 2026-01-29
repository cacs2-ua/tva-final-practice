from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Literal, Any

import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask

Array = np.ndarray
ThresholdMode = Literal["otsu", "adaptive"]
EnhanceMode = Literal["blackhat", "tophat_invert"]


@dataclass
class VesselSegDebug:
    bgr: Array
    fov_mask: Array
    green: Array
    green_in_fov: Array
    clahe: Array
    vesselness: Array
    binary_raw: Array
    binary_clean: Array
    final: Array
    params: Dict[str, Any]
    stats: Dict[str, Any]


def _to_uint8(img: Array) -> Array:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    x = x - float(x.min())
    denom = float(x.max() - x.min())
    if denom < 1e-6:
        denom = 1.0
    x = (255.0 * x / denom).clip(0, 255).astype(np.uint8)
    return x


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k

def _remove_small_components(mask_01: Array, min_area: int) -> Tuple[Array, Dict[str, Any]]:
    """
    Keep only connected components with area >= min_area.
    Input: 0/1 uint8.
    Output: 0/1 uint8.
    """
    m = (mask_01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)

    kept = np.zeros_like(m, dtype=np.uint8)
    kept_count = 0
    removed_count = 0
    areas_kept = []

    for i in range(1, num):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area >= int(min_area):
            kept[labels == i] = 1
            kept_count += 1
            areas_kept.append(area)
        else:
            removed_count += 1

    info = {
        "cc_total": int(num - 1),
        "cc_kept": int(kept_count),
        "cc_removed": int(removed_count),
        "min_area": int(min_area),
        "areas_kept_min": int(min(areas_kept)) if areas_kept else 0,
        "areas_kept_max": int(max(areas_kept)) if areas_kept else 0,
    }
    return kept, info

def _line_se(length: int, angle_deg: float, thickness: int = 1) -> Array:
    """
    Create a line-shaped structuring element (0/1 uint8) by drawing a line
    on a square canvas. Used for oriented morphological black-hat/top-hat.
    """
    length = int(max(3, length))
    thickness = int(max(1, thickness))
    k = np.zeros((length, length), dtype=np.uint8)

    c = (length - 1) / 2.0
    rad = np.deg2rad(angle_deg)

    dx = (length * 0.45) * np.cos(rad)
    dy = (length * 0.45) * np.sin(rad)

    x0 = int(round(c - dx))
    y0 = int(round(c - dy))
    x1 = int(round(c + dx))
    y1 = int(round(c + dy))

    cv.line(k, (x0, y0), (x1, y1), 1, thickness=thickness)
    return k


def _otsu_threshold_from_values(values_uint8: Array) -> int:
    """
    Otsu threshold computed ONLY from the provided 0..255 uint8 values.
    This avoids bias from pixels outside the FOV mask.
    """
    v = values_uint8.reshape(-1).astype(np.uint8)
    if v.size == 0:
        return 128

    hist = np.bincount(v, minlength=256).astype(np.float64)
    total = float(v.size)

    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    thr = 128

    for t in range(256):
        w_b += hist[t]
        if w_b <= 0:
            continue
        w_f = total - w_b
        if w_f <= 0:
            break
        sum_b += float(t) * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) * (m_b - m_f)
        if between > max_var:
            max_var = between
            thr = t

    return int(thr)

def _percentile_threshold_from_values(values_uint8: Array, target_fg_frac: float) -> int:
    """
    Pick a threshold so that roughly `target_fg_frac` of pixels are >= thr.
    (Unsupervised, robust when Otsu is unstable on some images.)
    """
    v = values_uint8.reshape(-1).astype(np.uint8)
    if v.size == 0:
        return 128
    t = float(np.clip(float(target_fg_frac), 0.001, 0.499))
    q = np.percentile(v, 100.0 * (1.0 - t))
    return int(np.clip(int(round(q)), 0, 255))

def _quick_cleanup_and_stats(
    binary01: Array,
    fov01: Array,
    morph_open_ksize: int,
    morph_close_ksize: int,
    min_cc_area: int,
) -> Tuple[float, int]:
    """
    Fast approximation of your final cleanup to guide auto-thresholding.
    Returns (fg_frac_inside_fov, cc_total_after_small_cc_removal).
    """
    m = (binary01 > 0).astype(np.uint8)
    m[fov01 == 0] = 0

    ok_open = _ensure_odd(morph_open_ksize)
    ok_close = _ensure_odd(morph_close_ksize)
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok_open, ok_open))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok_close, ok_close))

    m255 = (m * 255).astype(np.uint8)
    m255 = cv.morphologyEx(m255, cv.MORPH_OPEN, k_open)
    m255 = cv.morphologyEx(m255, cv.MORPH_CLOSE, k_close)
    m_clean = (m255 > 0).astype(np.uint8)

    m_clean, cc_info = _remove_small_components(m_clean, min_area=int(min_cc_area))

    denom = float(np.maximum(1, int(fov01.sum())))
    fg_frac = float(m_clean.sum()) / denom
    cc_total = int(cc_info.get("cc_total", 0))
    return fg_frac, cc_total

def _auto_select_otsu_threshold(
    vesselness: Array,
    fov01: Array,
    thr_otsu: int,
    otsu_offset: int,
    morph_open_ksize: int,
    morph_close_ksize: int,
    min_cc_area: int,
    *,
    fg_min: float = 0.020,
    fg_max: float = 0.180,
    fg_target: float = 0.070,
    sweep_min: int = -32,
    sweep_max: int = 12,
    sweep_step: int = 2,
    use_percentile_fallback: bool = True,
    verbose: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """
    Fixes the 'single-image collapse' when Otsu+fixed offset yields an extreme fg fraction.
    1) Start with (thr_otsu + otsu_offset)
    2) If fg fraction is out-of-range, sweep offsets (existing hyperparameter) to hit fg_target
    3) If still bad, fallback to percentile threshold (unsupervised)
    Returns (thr_final, meta_dict).
    """
    def clip_thr(t: int) -> int:
        return int(np.clip(int(t), 0, 255))

    # initial (current behavior)
    thr0 = clip_thr(thr_otsu + int(otsu_offset))
    fg0, cc0 = _quick_cleanup_and_stats((vesselness >= thr0).astype(np.uint8), fov01, morph_open_ksize, morph_close_ksize, min_cc_area)

    if fg_min <= fg0 <= fg_max:
        return thr0, {"method": "otsu_fixed_offset", "thr_otsu": int(thr_otsu), "thr": int(thr0), "fg": float(fg0), "cc": int(cc0)}

    # sweep offsets (TRY DIFFERENT EXISTING HYPERPARAMETER VALUES)
    best = (abs(fg0 - fg_target) + 5e-5 * cc0, thr0, fg0, cc0, int(otsu_offset))

    for off in range(int(otsu_offset) + int(sweep_min), int(otsu_offset) + int(sweep_max) + 1, int(sweep_step)):
        thr = clip_thr(int(thr_otsu) + int(off))
        fg, cc = _quick_cleanup_and_stats((vesselness >= thr).astype(np.uint8), fov01, morph_open_ksize, morph_close_ksize, min_cc_area)
        score = abs(fg - fg_target) + 5e-5 * cc

        if score < best[0]:
            best = (score, thr, fg, cc, off)

    _, thr_best, fg_best, cc_best, off_best = best

    if fg_min <= fg_best <= fg_max:
        if verbose:
            print(f"[AutoThr] Otsu fixed-offset bad (fg={fg0:.4f}). Using offset sweep -> off={off_best}, thr={thr_best}, fg={fg_best:.4f}, cc={cc_best}")
        return int(thr_best), {"method": "otsu_offset_sweep", "thr_otsu": int(thr_otsu), "thr": int(thr_best), "off": int(off_best), "fg": float(fg_best), "cc": int(cc_best)}

    # fallback: percentile threshold (unsupervised)
    if use_percentile_fallback:
        vals = vesselness[fov01 > 0].astype(np.uint8)
        thr_p = _percentile_threshold_from_values(vals, target_fg_frac=float(fg_target))
        fg_p, cc_p = _quick_cleanup_and_stats((vesselness >= thr_p).astype(np.uint8), fov01, morph_open_ksize, morph_close_ksize, min_cc_area)

        if verbose:
            print(f"[AutoThr] Sweep still extreme (fg={fg_best:.4f}). Using percentile -> thr={thr_p}, fg={fg_p:.4f}, cc={cc_p}")

        return int(thr_p), {"method": "percentile", "thr": int(thr_p), "fg_target": float(fg_target), "fg": float(fg_p), "cc": int(cc_p)}

    # last resort: best sweep
    if verbose:
        print(f"[AutoThr] Using best sweep (still extreme) -> off={off_best}, thr={thr_best}, fg={fg_best:.4f}, cc={cc_best}")
    return int(thr_best), {"method": "otsu_offset_sweep_extreme", "thr_otsu": int(thr_otsu), "thr": int(thr_best), "off": int(off_best), "fg": float(fg_best), "cc": int(cc_best)}


def segment_vessels_pipeline_c(
    img_bgr_or_gray: Array,
    *,
    # FOV mask params (you can tune these)
    fov_close_ksize: Optional[int] = None,
    fov_open_ksize: Optional[int] = None,
    fov_blur_ksize: int = 7,
    fov_blur_kind: str = "median",
    fov_do_convex_hull: bool = True,
    fov_ellipse_mode: str = "auto",
    fov_ellipse_scale: float = 1.01,
    # Pipeline C params
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    enhance_mode: EnhanceMode = "blackhat",
    line_lengths: Tuple[int, ...] = (9, 13, 17),
    line_thickness: int = 1,
    n_angles: int = 12,  # 12 -> 0..165 step 15 deg
    add_disk_blackhat: bool = True,
    disk_sizes: Tuple[int, ...] = (9, 13),
    threshold_mode: ThresholdMode = "otsu",
    otsu_offset: int = -5,  # negative => more sensitive
    adaptive_block_size: int = 31,
    adaptive_C: int = -2,
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 5,
    min_cc_area: int = 30,
    verbose: bool = False,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, VesselSegDebug]]:
    """
    Pipeline C (OpenCV-only baseline):
      - FOV mask
      - green channel
      - CLAHE
      - vessel enhancement via morphology (black-hat / top-hat after inversion)
        with multi-orientation line kernels (+ optional disk black-hat)
      - threshold (Otsu on FOV pixels OR adaptive)
      - cleanup (open/close + remove small CCs)
    Returns uint8 0/255 mask.
    """
    if img_bgr_or_gray is None:
        raise ValueError("img_bgr_or_gray is None")

    img = img_bgr_or_gray
    if img.ndim == 2:
        bgr = cv.cvtColor(_to_uint8(img), cv.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] >= 3:
        bgr = _to_uint8(img[:, :, :3])
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    h, w = bgr.shape[:2]

    # --- FOV mask ---
    fov_mask = compute_fov_mask(
        bgr,
        blur_ksize=fov_blur_ksize,
        blur_kind=fov_blur_kind,
        close_ksize=fov_close_ksize,
        open_ksize=fov_open_ksize,
        do_convex_hull=fov_do_convex_hull,
        ellipse_mode=fov_ellipse_mode,   # "off"/"auto"/"force"
        ellipse_scale=fov_ellipse_scale,
        return_debug=False,
    )
    fov01 = (fov_mask > 0).astype(np.uint8)
    fov_area_ratio = float(fov01.mean())

    # green channel (fundus: green has best vessel contrast)
    green = bgr[:, :, 1].copy()
    green_in_fov = green.copy()
    green_in_fov[fov01 == 0] = 0

    # --- CLAHE inside FOV (avoid amplifying outside background) ---
    tile = int(max(2, clahe_tile_grid_size))
    clahe = cv.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(tile, tile))

    # Fill outside-FOV with a robust inside intensity before CLAHE (prevents border artifacts)
    inside = green[fov01 > 0]
    fill_val = int(np.median(inside)) if inside.size else 0
    green_filled = green.copy()
    green_filled[fov01 == 0] = fill_val

    clahe_img = clahe.apply(green_filled)
    clahe_in_fov = clahe_img.copy()
    clahe_in_fov[fov01 == 0] = 0

    # --- Vessel enhancement (multi-orientation morphology) ---
    angles = np.linspace(0, 180, num=int(max(2, n_angles)), endpoint=False)

    vesselness = np.zeros((h, w), dtype=np.uint8)

    def apply_enhancement(src_u8: Array, se: Array) -> Array:
        if enhance_mode == "blackhat":
            return cv.morphologyEx(src_u8, cv.MORPH_BLACKHAT, se)
        elif enhance_mode == "tophat_invert":
            inv = cv.bitwise_not(src_u8)
            return cv.morphologyEx(inv, cv.MORPH_TOPHAT, se)
        else:
            raise ValueError("enhance_mode must be 'blackhat' or 'tophat_invert'")

    # oriented lines
    for L in line_lengths:
        L = int(max(3, L))
        for a in angles:
            se = _line_se(L, float(a), thickness=line_thickness)
            resp = apply_enhancement(clahe_in_fov, se)
            vesselness = cv.max(vesselness, resp)

    # optional isotropic disk black-hat (helps thick vessels)
    if add_disk_blackhat:
        for d in disk_sizes:
            d = int(max(3, d))
            se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (d, d))
            resp = apply_enhancement(clahe_in_fov, se)
            vesselness = cv.max(vesselness, resp)

    vesselness[fov01 == 0] = 0

    # small denoise
    vesselness = cv.GaussianBlur(vesselness, (3, 3), 0)

    # --- Threshold ---
    if threshold_mode == "otsu":
        vals = vesselness[fov01 > 0]
        thr_otsu = _otsu_threshold_from_values(vals)

        thr2, thr_meta = _auto_select_otsu_threshold(
            vesselness=vesselness,
            fov01=fov01,
            thr_otsu=thr_otsu,
            otsu_offset=otsu_offset,
            morph_open_ksize=morph_open_ksize,
            morph_close_ksize=morph_close_ksize,
            min_cc_area=min_cc_area,
            verbose=verbose,
        )

        binary01 = (vesselness >= int(thr2)).astype(np.uint8)
        thr_used = thr_meta
    elif threshold_mode == "adaptive":
        bs = _ensure_odd(adaptive_block_size)
        thr_img = cv.adaptiveThreshold(
            vesselness,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            bs,
            int(adaptive_C),
        )
        binary01 = (thr_img > 0).astype(np.uint8)
        thr_used = {"adaptive_block_size": bs, "adaptive_C": int(adaptive_C)}
    else:
        raise ValueError("threshold_mode must be 'otsu' or 'adaptive'")

    binary01[fov01 == 0] = 0

    # --- Cleanup ---
    ok_open = _ensure_odd(morph_open_ksize)
    ok_close = _ensure_odd(morph_close_ksize)
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok_open, ok_open))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok_close, ok_close))

    bin255 = (binary01 * 255).astype(np.uint8)
    bin255 = cv.morphologyEx(bin255, cv.MORPH_OPEN, k_open)
    bin255 = cv.morphologyEx(bin255, cv.MORPH_CLOSE, k_close)
    binary01_clean = (bin255 > 0).astype(np.uint8)

    binary01_clean, cc_info = _remove_small_components(binary01_clean, min_area=min_cc_area)
    binary01_clean[fov01 == 0] = 0

    final = (binary01_clean * 255).astype(np.uint8)

    # --- Debug prints ---
    if verbose:
        vvals = vesselness[fov01 > 0]
        vmin = int(vvals.min()) if vvals.size else 0
        vmax = int(vvals.max()) if vvals.size else 0
        vmean = float(vvals.mean()) if vvals.size else 0.0
        fg_raw = int(binary01.sum())
        fg_clean = int(binary01_clean.sum())

        print("[PipelineC] shape =", (h, w), "FOV_area_ratio =", round(fov_area_ratio, 4))
        print("[PipelineC] CLAHE clipLimit =", clahe_clip_limit, "tile =", tile)
        print("[PipelineC] enhance_mode =", enhance_mode,
              "| line_lengths =", line_lengths, "line_thickness =", line_thickness,
              "n_angles =", int(n_angles),
              "| add_disk_blackhat =", bool(add_disk_blackhat), "disk_sizes =", disk_sizes)
        print("[PipelineC] vesselness stats (in FOV): min/max/mean =", vmin, "/", vmax, "/", round(vmean, 3))
        print("[PipelineC] threshold_mode =", threshold_mode, "| thr_used =", thr_used)
        print("[PipelineC] foreground px raw =", fg_raw, "| after clean =", fg_clean)
        print("[PipelineC] CC info:", cc_info)

    if not return_debug:
        return final

    dbg = VesselSegDebug(
        bgr=bgr,
        fov_mask=(fov01 * 255).astype(np.uint8),
        green=green,
        green_in_fov=green_in_fov,
        clahe=clahe_in_fov,
        vesselness=vesselness,
        binary_raw=(binary01 * 255).astype(np.uint8),
        binary_clean=(binary01_clean * 255).astype(np.uint8),
        final=final,
        params={
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid_size": clahe_tile_grid_size,
            "enhance_mode": enhance_mode,
            "line_lengths": line_lengths,
            "line_thickness": line_thickness,
            "n_angles": n_angles,
            "add_disk_blackhat": add_disk_blackhat,
            "disk_sizes": disk_sizes,
            "threshold_mode": threshold_mode,
            "otsu_offset": otsu_offset,
            "adaptive_block_size": adaptive_block_size,
            "adaptive_C": adaptive_C,
            "morph_open_ksize": morph_open_ksize,
            "morph_close_ksize": morph_close_ksize,
            "min_cc_area": min_cc_area,
        },
        stats={
            "fov_area_ratio": fov_area_ratio,
            "thr_used": thr_used,
            **cc_info,
        },
    )
    return final, dbg


def vessel_segmentation(input_image_path: Union[str, "os.PathLike[str]"], *, verbose: bool = False) -> Array:
    """
    Drop-in replacement for the course skeleton: reads path, returns 0/255 uint8 segmentation.
    """
    img = cv.imread(str(input_image_path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_image_path}")

    seg = segment_vessels_pipeline_c(
        img,
        # reasonable defaults for DRIVE-like images
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=8,
        enhance_mode="blackhat",
        line_lengths=(9, 13, 17),
        n_angles=12,
        add_disk_blackhat=True,
        disk_sizes=(9, 13),
        threshold_mode="otsu",
        otsu_offset=-5,
        morph_open_ksize=3,
        morph_close_ksize=5,
        min_cc_area=30,
        verbose=verbose,
        return_debug=False,
    )
    return seg
