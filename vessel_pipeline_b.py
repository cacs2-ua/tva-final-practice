from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask

Array = np.ndarray


# ---------------------------
# Helpers (robust + debuggable)
# ---------------------------

def _to_uint8(gray: Array) -> Array:
    if gray is None:
        raise ValueError("Input is None")
    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(np.float32)
    g = g - float(g.min())
    denom = float(g.max() - g.min())
    if denom < 1e-6:
        denom = 1.0
    return np.clip(255.0 * (g / denom), 0, 255).astype(np.uint8)


def _green_or_gray(img: Array) -> Array:
    if img.ndim == 2:
        return _to_uint8(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        # OpenCV loads BGR; green channel is index 1
        return _to_uint8(img[:, :, 1])
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _auto_line_length(h: int, w: int) -> int:
    # DRIVE ~ 768x584 => good default line length around 9-15.
    diag = float(np.hypot(h, w))
    L = int(round(diag * 0.018))  # ~16 for DRIVE
    L = max(9, min(L, 25))
    return _ensure_odd(L)


def _line_kernel(length: int, angle_deg: float, thickness: int = 1) -> Array:
    """
    Create an odd-size square kernel with a centered line at angle_deg.
    Used as a structuring element for directional morphology.
    """
    length = _ensure_odd(length)
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2

    # Build endpoints for a line through center
    # Use a radius slightly less than half to keep endpoints inside bounds
    r = c - 1
    theta = np.deg2rad(angle_deg)
    dx = int(round(r * np.cos(theta)))
    dy = int(round(r * np.sin(theta)))

    x1, y1 = c - dx, c - dy
    x2, y2 = c + dx, c + dy

    cv.line(k, (x1, y1), (x2, y2), color=1, thickness=int(thickness))
    return k


def _clahe(gray_u8: Array, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> Array:
    clahe = cv.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(map(int, tile_grid)))
    return clahe.apply(gray_u8)


def _multi_orient_morph(
    img_u8: Array,
    op: int,
    *,
    line_length: int,
    angles: List[float],
) -> Array:
    """
    Apply a morphologyEx operation across multiple line orientations and take the pixelwise maximum response.
    This implements the "multi-directional" morphology idea (as in morphology-based vessel papers).
    """
    best = np.zeros_like(img_u8)
    for ang in angles:
        k = _line_kernel(line_length, ang, thickness=1)
        resp = cv.morphologyEx(img_u8, op, k)
        best = cv.max(best, resp)
    return best


def _percentile_score(enh_u8: Array, fov: Array) -> float:
    """
    Score an enhancement map to choose between candidates.
    We want strong separation between "vessel-like responses" and background, but not overly dense.
    """
    m = (fov > 0)
    vals = enh_u8[m].astype(np.float32)
    if vals.size < 100:
        return -1e9

    p50 = float(np.percentile(vals, 50))
    p95 = float(np.percentile(vals, 95))
    p995 = float(np.percentile(vals, 99.5))

    # how strong are the brightest responses vs typical background
    sep = (p995 - p50) / (p50 + 1.0)

    # penalize if "too many" pixels are super high (likely boosting non-vessel background)
    thr = p995
    dense = float(np.mean(vals >= thr))  # should be tiny
    score = sep - 8.0 * dense

    return float(score)


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-6)


def _kmeans_fit_centers(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    attempts: int = 5,
    max_iter: int = 50,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, float]:
    """
    Fit k-means using OpenCV on X (float32 NxD). Returns (centers, compactness).
    """
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    cv.setRNGSeed(int(seed))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, int(max_iter), float(eps))
    compactness, labels, centers = cv.kmeans(
        X,
        int(k),
        None,
        criteria,
        int(attempts),
        flags=cv.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32), float(compactness)


def _assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each row of X to nearest center (Euclidean). Returns labels (N,).
    """
    # distances: (N,K)
    # (x-c)^2 = x^2 + c^2 - 2xc; but N is small enough for direct.
    diffs = X[:, None, :] - centers[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


def _remove_small_components(mask_u8: Array, min_area: int) -> Array:
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8)

    out = np.zeros_like(mask_u8)
    for lab in range(1, num):
        area = int(stats[lab, cv.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == lab] = 255
    return out

def _cleanup_binary(
    mask_u8: Array,
    fov: Array,
    *,
    h: int,
    w: int,
    verbose: bool,
    morph_close_ksize: int = 3,
    morph_close_iter: int = 1,
    morph_open_ksize: int = 3,
    morph_open_iter: int = 1,
    cc_min_area: Optional[int] = None,
) -> Array:
    """
    Cleanup:
      - closing to improve continuity
      - opening to remove specks
      - connected components filtering (remove small objects)
      - enforce FOV
    Defaults are EXACTLY the previous behavior.
    """
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    fov_u8 = (fov > 0).astype(np.uint8) * 255

    morph_close_ksize = _ensure_odd(int(morph_close_ksize))
    morph_open_ksize = _ensure_odd(int(morph_open_ksize))
    morph_close_iter = max(0, int(morph_close_iter))
    morph_open_iter = max(0, int(morph_open_iter))

    # morphology
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))

    m = mask_u8
    if morph_close_iter > 0:
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k_close, iterations=morph_close_iter)
    if morph_open_iter > 0:
        m = cv.morphologyEx(m, cv.MORPH_OPEN, k_open, iterations=morph_open_iter)

    # CC filtering
    if cc_min_area is None:
        # EXACTLY the old rule:
        min_area = max(40, int(round(0.00006 * h * w)))
    else:
        min_area = max(1, int(cc_min_area))

    m2 = _remove_small_components(m, min_area=min_area)

    # enforce FOV hard
    m2[fov_u8 == 0] = 0

    if verbose:
        frac = float(np.mean(m2 > 0))
        print(
            f"[PipelineB][cleanup] "
            f"close={morph_close_ksize}x{morph_close_ksize} it={morph_close_iter} | "
            f"open={morph_open_ksize}x{morph_open_ksize} it={morph_open_iter} | "
            f"min_area={min_area} | vessel_frac={frac:.4f}"
        )

    return m2



# ---------------------------
# Pipeline B (Morphology → KMeans → Cleanup)
# ---------------------------

def vessel_segmentation(
    input_image: Union[str, Array],
    *,
    verbose: bool = False,

    # --- cleanup / CC ---
    cc_min_area: Optional[int] = None,

    # --- CLAHE ---
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),

    # --- Enhancement ---
    enhance_mode: str = "auto",  # "auto" | "blackhat" | "tophat"
    line_lengths: Optional[Tuple[int, ...]] = None,  # None => auto length (old behavior)
    n_angles: int = 12,

    # --- KMeans ---
    k: int = 2,
    use_grad: bool = True,

    # --- FOV params ---
    fov_blur_kind: str = "median",
    fov_blur_ksize: int = 7,
    fov_close_ksize: Optional[int] = None,
    fov_open_ksize: Optional[int] = None,
    fov_do_convex_hull: bool = True,
    fov_ellipse_mode: str = "auto",
    fov_ellipse_scale: float = 1.01,

    # --- Morph cleanup params ---
    morph_close_iter: int = 1,
    morph_close_ksize: int = 3,
    morph_open_iter: int = 1,
    morph_open_ksize: int = 3,
) -> Array:
    """
    Pipeline B: Morphology → k-means → cleanup

    IMPORTANT:
    - All defaults preserve the old behavior.
    - This function now accepts hyperparameters via **PARAMS.
    """

    def _as_tile_grid(x) -> Tuple[int, int]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return (int(x[0]), int(x[1]))
        return (int(x), int(x))

    def _parse_line_lengths(hh: int, ww: int):
        if line_lengths is None:
            return [_auto_line_length(hh, ww)]  # EXACT old behavior
        if isinstance(line_lengths, (int, np.integer)):
            return [_ensure_odd(int(line_lengths))]
        # tuple/list of ints
        out = []
        for L in line_lengths:
            out.append(_ensure_odd(int(L)))
        return out

    # --- Load ---
    if isinstance(input_image, str):
        img = cv.imread(input_image, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {input_image}")
    else:
        img = input_image

    g = _green_or_gray(img)
    h, w = g.shape[:2]

    # --- FOV mask ---
    fov = compute_fov_mask(
        img,
        blur_kind=str(fov_blur_kind),
        blur_ksize=int(fov_blur_ksize),
        close_ksize=fov_close_ksize,
        open_ksize=fov_open_ksize,
        do_hole_fill=True,
        do_convex_hull=bool(fov_do_convex_hull),
        ellipse_mode=str(fov_ellipse_mode),
        ellipse_scale=float(fov_ellipse_scale),
        ellipse_min_fill_ratio=0.975,   # keep your robust default
        return_debug=False,
    )
    fov_u8 = (fov > 0).astype(np.uint8) * 255
    n_fov = int(np.sum(fov_u8 > 0))

    if verbose:
        print(f"[PipelineB] image shape={g.shape} dtype={g.dtype}")
        print(f"[PipelineB] FOV pixels={n_fov} ({n_fov / float(h*w + 1e-9):.3f} of image)")

    # --- CLAHE ---
    tg = _as_tile_grid(clahe_tile_grid_size)
    clahe = _clahe(g, clip_limit=float(clahe_clip_limit), tile_grid=tg)
    clahe[fov_u8 == 0] = 0

    # --- Enhancement (multi-length + multi-angle; choose best) ---
    n_angles = int(n_angles)
    n_angles = max(1, n_angles)
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False).tolist()

    lengths = _parse_line_lengths(h, w)

    mode = str(enhance_mode).strip().lower()
    if mode in ("invert+tophat", "tophat"):
        mode = "tophat"
    elif mode == "blackhat":
        mode = "blackhat"
    else:
        mode = "auto"

    best_score = -1e18
    best_enh = None
    best_desc = "none"

    inv = cv.bitwise_not(clahe)

    for L in lengths:
        if mode in ("auto", "blackhat"):
            enh_blackhat = _multi_orient_morph(clahe, cv.MORPH_BLACKHAT, line_length=L, angles=angles)
            sc = _percentile_score(enh_blackhat, fov_u8)
            if sc > best_score:
                best_score = sc
                best_enh = enh_blackhat
                best_desc = f"blackhat|L={L}|angles={n_angles}"

        if mode in ("auto", "tophat"):
            enh_tophat = _multi_orient_morph(inv, cv.MORPH_TOPHAT, line_length=L, angles=angles)
            sc = _percentile_score(enh_tophat, fov_u8)
            if sc > best_score:
                best_score = sc
                best_enh = enh_tophat
                best_desc = f"tophat(inv)|L={L}|angles={n_angles}"

    if best_enh is None:
        # safety fallback
        best_enh = np.zeros_like(clahe)
        best_desc = "fallback_zero"

    enh = best_enh
    enh[fov_u8 == 0] = 0

    if verbose:
        print(f"[PipelineB] enhance_mode={enhance_mode} -> chosen={best_desc} score={best_score:.4f}")
        vals = enh[fov_u8 > 0]
        if vals.size > 0:
            print(
                f"[PipelineB] enh stats in FOV: "
                f"min={int(vals.min())} p50={int(np.percentile(vals,50))} "
                f"p95={int(np.percentile(vals,95))} max={int(vals.max())}"
            )

    # --- Features for KMeans ---
    idx = np.flatnonzero((fov_u8 > 0).ravel())
    if idx.size < 200:
        if verbose:
            print("[PipelineB] Too few FOV pixels; returning empty mask.")
        return np.zeros((h, w), dtype=np.uint8)

    enh_flat = enh.ravel()[idx].astype(np.float32)

    feats = [enh_flat]

    if bool(use_grad):
        clahe_f = clahe.astype(np.float32)
        gx = cv.Sobel(clahe_f, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(clahe_f, cv.CV_32F, 0, 1, ksize=3)
        grad = cv.magnitude(gx, gy)
        grad_u8 = _to_uint8(grad)
        grad_flat = grad_u8.ravel()[idx].astype(np.float32)
        feats.append(grad_flat)

    X = np.stack(feats, axis=1).astype(np.float32)  # NxD

    # Standardize (keep deterministic)
    rng = np.random.default_rng(0)
    n = X.shape[0]
    n_fit = min(120_000, n)
    sel = rng.choice(n, size=n_fit, replace=False) if n_fit < n else np.arange(n)
    X_fit = X[sel]

    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0) + 1e-6
    Xs_fit = _standardize(X_fit, mean, std).astype(np.float32)
    Xs_all = _standardize(X, mean, std).astype(np.float32)

    # --- KMeans ---
    k = int(k)
    k = max(2, k)

    centers, compactness = _kmeans_fit_centers(Xs_fit, k, seed=0, attempts=5, max_iter=60, eps=1e-3)
    labels_all = _assign_to_centers(Xs_all, centers)

    # Pick vessel cluster: highest mean in ORIGINAL enhanced feature
    means = []
    for lab in range(k):
        if np.any(labels_all == lab):
            means.append(float(enh_flat[labels_all == lab].mean()))
        else:
            means.append(-1e18)
    vessel_lab = int(np.argmax(means))

    if verbose:
        print(f"[PipelineB] kmeans: k={k} compactness={compactness:.2f} use_grad={bool(use_grad)}")
        print(f"[PipelineB] cluster mean(enh): {means} -> vessel_lab={vessel_lab}")

    # Build mask
    mask = np.zeros((h * w,), dtype=np.uint8)
    mask[idx[labels_all == vessel_lab]] = 255
    mask = mask.reshape((h, w))

    # --- Cleanup (now hyperparameterized) ---
    mask = _cleanup_binary(
        mask,
        fov_u8,
        h=h,
        w=w,
        verbose=verbose,
        morph_close_ksize=morph_close_ksize,
        morph_close_iter=morph_close_iter,
        morph_open_ksize=morph_open_ksize,
        morph_open_iter=morph_open_iter,
        cc_min_area=cc_min_area,
    )

    if verbose:
        print(f"[PipelineB] output unique={set(np.unique(mask).tolist())} shape={mask.shape}")

    return mask

