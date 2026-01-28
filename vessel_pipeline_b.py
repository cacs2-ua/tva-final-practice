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


def _cleanup_binary(mask_u8: Array, fov: Array, *, h: int, w: int, verbose: bool) -> Array:
    """
    Cleanup as in the Pipeline B description:
      - closing to improve continuity
      - opening to remove specks
      - connected components filtering (remove small objects)
      - enforce FOV
    """
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255
    fov_u8 = (fov > 0).astype(np.uint8) * 255

    # mild morphology
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    m = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, k_close, iterations=1)
    m = cv.morphologyEx(m, cv.MORPH_OPEN, k_open, iterations=1)

    # remove tiny CCs (relative to image size)
    min_area = max(40, int(round(0.00006 * h * w)))  # ~27 for DRIVE; we clamp to 40
    m2 = _remove_small_components(m, min_area=min_area)

    # enforce FOV hard
    m2[fov_u8 == 0] = 0

    if verbose:
        frac = float(np.mean(m2 > 0))
        print(f"[PipelineB][cleanup] min_area={min_area} | vessel_frac={frac:.4f}")

    return m2


# ---------------------------
# Pipeline B (Morphology → KMeans → Cleanup)
# ---------------------------

def vessel_segmentation(
    input_image: Union[str, Array],
    *,
    verbose: bool = False,
) -> Array:
    """
    Pipeline B: Morphology → k-means → cleanup

    Steps:
      1) FOV mask (retina mask)
      2) green channel
      3) CLAHE contrast normalization
      4) vessel boosting via multi-orientation black-hat / top-hat (pick best)
      5) k-means clustering on per-pixel features (enhanced intensity + gradient)
      6) cleanup morphology + CC filtering
    Returns:
      binary mask uint8 (0/255) same HxW as input.
    """
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
    # Use robust defaults from your existing fov_mask.py
    fov = compute_fov_mask(
        img,
        blur_kind="median",
        blur_ksize=7,
        close_ksize=None,   # auto
        open_ksize=None,    # auto
        do_hole_fill=True,
        do_convex_hull=True,
        ellipse_mode="auto",
        ellipse_scale=1.01,
        ellipse_min_fill_ratio=0.975,
        return_debug=False,
    )
    fov_u8 = (fov > 0).astype(np.uint8) * 255
    n_fov = int(np.sum(fov_u8 > 0))

    if verbose:
        print(f"[PipelineB] image shape={g.shape} dtype={g.dtype}")
        print(f"[PipelineB] FOV pixels={n_fov} ({n_fov / float(h*w + 1e-9):.3f} of image)")

    # --- CLAHE ---
    clahe = _clahe(g, clip_limit=2.0, tile_grid=(8, 8))
    clahe[fov_u8 == 0] = 0  # don’t propagate outside

    # --- Multi-orientation morphology boosting (try both) ---
    line_len = _auto_line_length(h, w)
    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    # Candidate A: black-hat on CLAHE (vessels are dark lines => black-hat highlights them)
    enh_blackhat = _multi_orient_morph(clahe, cv.MORPH_BLACKHAT, line_length=line_len, angles=angles)

    # Candidate B: invert + top-hat (sometimes works better if vessels become bright)
    inv = cv.bitwise_not(clahe)
    enh_tophat = _multi_orient_morph(inv, cv.MORPH_TOPHAT, line_length=line_len, angles=angles)

    score_a = _percentile_score(enh_blackhat, fov_u8)
    score_b = _percentile_score(enh_tophat, fov_u8)

    if score_b > score_a:
        enh = enh_tophat
        enh_mode = "invert+tophat"
        enh_score = score_b
    else:
        enh = enh_blackhat
        enh_mode = "blackhat"
        enh_score = score_a

    if verbose:
        print(f"[PipelineB] line_len={line_len} angles={len(angles)}")
        print(f"[PipelineB] enh scores: blackhat={score_a:.4f} | invert+tophat={score_b:.4f} -> chosen={enh_mode} ({enh_score:.4f})")
        vals = enh[fov_u8 > 0]
        print(f"[PipelineB] enh stats in FOV: min={int(vals.min())} p50={int(np.percentile(vals,50))} p95={int(np.percentile(vals,95))} max={int(vals.max())}")

    enh[fov_u8 == 0] = 0

    # --- Feature image for clustering ---
    # Feature 1: enhanced intensity (strong vesselness response)
    # Feature 2: gradient magnitude on CLAHE (vessel edges help clustering)
    clahe_f = clahe.astype(np.float32)
    gx = cv.Sobel(clahe_f, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(clahe_f, cv.CV_32F, 0, 1, ksize=3)
    grad = cv.magnitude(gx, gy)
    grad_u8 = _to_uint8(grad)

    # Flatten only inside FOV
    idx = np.flatnonzero((fov_u8 > 0).ravel())
    if idx.size < 200:
        # pathological; return empty
        if verbose:
            print("[PipelineB] Too few FOV pixels; returning empty mask.")
        return np.zeros((h, w), dtype=np.uint8)

    enh_flat = enh.ravel()[idx].astype(np.float32)
    grad_flat = grad_u8.ravel()[idx].astype(np.float32)

    X = np.stack([enh_flat, grad_flat], axis=1).astype(np.float32)

    # Standardize features (important for kmeans stability)
    # Fit on a subset for speed, then assign all pixels.
    rng = np.random.default_rng(0)
    n = X.shape[0]
    n_fit = min(120_000, n)  # speed cap
    sel = rng.choice(n, size=n_fit, replace=False) if n_fit < n else np.arange(n)
    X_fit = X[sel]

    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0) + 1e-6
    Xs_fit = _standardize(X_fit, mean, std).astype(np.float32)
    Xs_all = _standardize(X, mean, std).astype(np.float32)

    # --- KMeans ---
    k = 2
    centers, compactness = _kmeans_fit_centers(Xs_fit, k, seed=0, attempts=5, max_iter=60, eps=1e-3)
    labels_all = _assign_to_centers(Xs_all, centers)

    # Pick vessel cluster: the one with higher mean in the ORIGINAL enhanced feature (not standardized)
    m0 = float(enh_flat[labels_all == 0].mean()) if np.any(labels_all == 0) else -1.0
    m1 = float(enh_flat[labels_all == 1].mean()) if np.any(labels_all == 1) else -1.0
    vessel_lab = 1 if m1 > m0 else 0

    if verbose:
        print(f"[PipelineB] kmeans: k={k} compactness={compactness:.2f}")
        print(f"[PipelineB] cluster mean(enh): lab0={m0:.2f} lab1={m1:.2f} -> vessel_lab={vessel_lab}")

    # Build mask
    mask = np.zeros((h * w,), dtype=np.uint8)
    mask[idx[labels_all == vessel_lab]] = 255
    mask = mask.reshape((h, w))

    # --- Cleanup ---
    mask = _cleanup_binary(mask, fov_u8, h=h, w=w, verbose=verbose)

    if verbose:
        print(f"[PipelineB] output unique={set(np.unique(mask).tolist())} shape={mask.shape}")

    return mask
