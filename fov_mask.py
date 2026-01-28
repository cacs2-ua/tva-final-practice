from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Literal

import numpy as np
import cv2 as cv

Array = np.ndarray


@dataclass
class FOVMaskDebug:
    """Optional debug outputs to inspect intermediate steps."""
    green_or_gray: Array
    blurred: Array
    otsu: Array
    candidate_chosen: Array
    after_morph: Array
    largest_cc: Array
    filled: Array
    convex_hull: Array
    ellipse_mask: Array
    final: Array
    chosen_inverted: bool
    scores: Dict[str, float]
    ellipse_used: bool
    ellipse_reason: str


def _to_uint8(gray: Array) -> Array:
    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(np.float32)
    g = g - g.min()
    denom = (g.max() - g.min()) if (g.max() - g.min()) > 1e-6 else 1.0
    return (255.0 * g / denom).clip(0, 255).astype(np.uint8)

def _flatfield_for_otsu(g: Array) -> Array:
    """
    Remove slow illumination gradients (vignetting) so global thresholding
    doesn't 'eat' the dark retinal rim.
    """
    h, w = g.shape[:2]
    sigma = 0.12 * float(min(h, w))  # large blur = illumination field
    bg = cv.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # flat-field: g / bg
    corr = cv.divide(g, bg, scale=255.0)
    corr = cv.normalize(corr, None, 0, 255, cv.NORM_MINMAX)
    return corr.astype(np.uint8)


def _to_gray_or_green(img: Array) -> Array:
    """
    Convert input (BGR/RGB/GRAY) to a single-channel image.
    For fundus images, green channel usually gives best contrast.
    """
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        # OpenCV loads as BGR by default
        gray = img[:, :, 1]  # green channel
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    return _to_uint8(gray)


def _ensure_odd(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def _auto_ksizes(h: int, w: int) -> Tuple[int, int]:
    """
    Auto choose morphology kernel sizes based on image diagonal.
    Tuned to be "safe" for common fundus sizes (e.g., DRIVE 768x584).
    """
    diag = float(np.hypot(h, w))
    close_k = _ensure_odd(int(round(diag * 0.035)))  # ~33 for DRIVE
    open_k = _ensure_odd(int(round(diag * 0.012)))   # ~11 for DRIVE
    close_k = max(close_k, 25)
    open_k = max(open_k, 9)
    return close_k, open_k


def _morph_cleanup(mask: Array, close_ksize: int, open_ksize: int) -> Array:
    """Close gaps then open to remove specks."""
    close_ksize = _ensure_odd(close_ksize)
    open_ksize = _ensure_odd(open_ksize)

    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_ksize, close_ksize))
    open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_ksize, open_ksize))

    m = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_k)
    m = cv.morphologyEx(m, cv.MORPH_OPEN, open_k)
    return m


def _largest_connected_component(mask: Array) -> Array:
    """Return a binary mask (0/255) containing only the largest foreground CC."""
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    areas = stats[1:, cv.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == best_idx] = 255
    return out


def _fill_holes(mask: Array) -> Array:
    """
    Fill holes inside a binary object using flood-fill on the background.
    Input/Output are 0/255 uint8.
    More robust than using only (0,0) as seed.
    """
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    if h == 0 or w == 0:
        return m

    # find a background seed on the border
    border_coords = []
    border_coords += [(0, x) for x in range(w)]
    border_coords += [(h - 1, x) for x in range(w)]
    border_coords += [(y, 0) for y in range(h)]
    border_coords += [(y, w - 1) for y in range(h)]

    seed = None
    for (yy, xx) in border_coords:
        if m[yy, xx] == 0:
            seed = (xx, yy)  # floodFill uses (x,y)
            break
    if seed is None:
        # mask covers the whole border; nothing to flood-fill safely
        return m

    inv = cv.bitwise_not(m)
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv.floodFill(ff, flood_mask, seedPoint=seed, newVal=0)

    holes = (ff > 0).astype(np.uint8) * 255
    filled = cv.bitwise_or(m, holes)
    return filled


def _convex_hull(mask: Array) -> Array:
    """Return convex hull of the largest object as 0/255 mask."""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m, dtype=np.uint8)

    cnt = max(contours, key=cv.contourArea)
    hull = cv.convexHull(cnt)
    out = np.zeros_like(m, dtype=np.uint8)
    cv.drawContours(out, [hull], -1, 255, thickness=cv.FILLED)
    return out


def _candidate_score(candidate_mask: Array) -> float:
    """
    Score a candidate FOV mask.
    Heuristics (good FOV):
      - large area but not the whole image
      - centered
      - low border contact
      - reasonably compact
    """
    m = (candidate_mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    area = float(m.sum())
    area_ratio = area / float(h * w)

    if area < 10:
        return -1e9

    border = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    border_ratio = float(border.mean())

    ys, xs = np.where(m > 0)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    center_dist = np.sqrt((cy - (h / 2.0)) ** 2 + (cx - (w / 2.0)) ** 2)
    center_dist_norm = center_dist / np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)

    circ = 0.0
    m255 = (m * 255).astype(np.uint8)
    contours, _ = cv.findContours(m255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        A = float(cv.contourArea(cnt))
        P = float(cv.arcLength(cnt, True))
        circ = (4.0 * np.pi * A) / (P * P + 1e-6)

    area_penalty = 0.6 if (area_ratio < 0.10 or area_ratio > 0.98) else 0.0

    score = (
        + 1.5 * area_ratio
        - 2.0 * border_ratio
        - 0.8 * center_dist_norm
        + 0.4 * circ
        - area_penalty
    )
    return float(score)


def _fit_ellipse_mask_from_contour(mask: Array, scale: float = 1.02) -> Tuple[Array, bool, str]:
    """
    Fit an ellipse to the outer contour and return an ellipse mask.
    scale > 1 expands ellipse slightly (useful to recover small border bites).
    """
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(m, dtype=np.uint8), False, "no_contours"

    cnt = max(contours, key=cv.contourArea)
    if len(cnt) < 5:
        return m.copy(), False, "too_few_points_for_ellipse"

    (cx, cy), (MA, ma), angle = cv.fitEllipse(cnt)  # widths (full axis lengths)
    rx = max(1, int(round((MA * 0.5) * scale)))
    ry = max(1, int(round((ma * 0.5) * scale)))

    out = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(out, (int(round(cx)), int(round(cy))), (rx, ry), angle, 0, 360, 255, thickness=cv.FILLED)
    return out, True, "fitEllipse"

def _fit_ellipse_mask_from_ring_edges(
    g: Array,
    mask: Array,
    *,
    scale: float = 1.01,
    ring_width: int = 25,
) -> Tuple[Array, bool, str]:
    """
    Fit ellipse from Canny edges located in a thin ring around the mask border.
    This avoids bias from a truncated (bitten) filled region.
    """
    g = _to_uint8(g)
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]

    ring_width = max(5, int(ring_width))
    k = _ensure_odd(ring_width)
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))

    dil = cv.dilate(m, se)
    ero = cv.erode(m, se)
    ring = cv.bitwise_and(dil, cv.bitwise_not(ero))  # border band

    # Auto Canny thresholds from median intensity INSIDE the mask
    inside = g[m > 0]
    if inside.size < 50:
        return np.zeros_like(m), False, "ring:too_few_inside_pixels"
    med = float(np.median(inside))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))

    edges = cv.Canny(g, lower, upper)
    edges = cv.bitwise_and(edges, ring)

    ys, xs = np.where(edges > 0)
    if ys.size < 80:
        return np.zeros_like(m), False, f"ring:not_enough_edge_points ({ys.size})"

    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

    # fitEllipse needs >=5 points (we have plenty)
    (cx, cy), (MA, ma), angle = cv.fitEllipse(pts)

    rx = max(1, int(round((MA * 0.5) * scale)))
    ry = max(1, int(round((ma * 0.5) * scale)))

    out = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(out, (int(round(cx)), int(round(cy))), (rx, ry), angle, 0, 360, 255, thickness=cv.FILLED)
    return out, True, "ring:edge_fitEllipse"



def _should_use_ellipse(mask: Array, min_fill_ratio: float = 0.975) -> Tuple[bool, str]:
    """
    Detect the classic 'straight bite' problem:
    the shape is convex, so convex hull won't help, but area is noticeably below
    the best-fitting enclosing shape.

    We compare mask area vs minimum enclosing circle area.
    If the ratio is too low, it suggests a 'cut' happened.
    """
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "no_contours"

    cnt = max(contours, key=cv.contourArea)
    A = float(cv.contourArea(cnt))
    if A < 100:
        return False, "tiny_area"

    (x, y), r = cv.minEnclosingCircle(cnt)
    circle_area = float(np.pi * (r * r) + 1e-6)
    fill_ratio = A / circle_area

    if fill_ratio < min_fill_ratio:
        return True, f"low_fill_ratio_vs_enclosing_circle ({fill_ratio:.3f} < {min_fill_ratio:.3f})"
    return False, f"ok_fill_ratio ({fill_ratio:.3f})"


EllipseMode = Literal["off", "auto", "force"]

def _fit_ellipse_mask_from_radial_edges(
    g: Array,
    coarse_mask: Array,
    *,
    scale: float = 1.01,
    n_angles: int = 360,
    smooth_k: int = 9,
) -> Tuple[Array, bool, str]:
    """
    Fit an ellipse using boundary points detected from the ORIGINAL image (g),
    by finding the strongest outward intensity drop along radial rays.
    This is robust when the coarse mask has a 'chord' cut (Otsu/vignetting failure).
    """
    g = _to_uint8(g)
    m = (coarse_mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]

    contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(m, dtype=np.uint8), False, "radial:no_contours"

    cnt = max(contours, key=cv.contourArea)
    if cv.contourArea(cnt) < 100:
        return np.zeros_like(m, dtype=np.uint8), False, "radial:tiny_area"

    # Use minEnclosingCircle center as a stable center guess (better than centroid when truncated)
    (cx, cy), r_guess = cv.minEnclosingCircle(cnt)
    cx = float(cx)
    cy = float(cy)
    r_guess = float(r_guess)

    if r_guess < 5:
        cx, cy = (w / 2.0), (h / 2.0)
        r_guess = 0.5 * min(h, w)

    # Smooth kernel for 1D profiles
    smooth_k = max(3, int(smooth_k))
    if smooth_k % 2 == 0:
        smooth_k += 1
    kernel = np.ones(smooth_k, dtype=np.float32) / float(smooth_k)

    pts = []

    for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        dx = float(np.cos(theta))
        dy = float(np.sin(theta))

        # Max radius until image boundary
        r_candidates = []
        if abs(dx) > 1e-6:
            r_candidates.append((0.0 - cx) / dx)
            r_candidates.append(((w - 1.0) - cx) / dx)
        if abs(dy) > 1e-6:
            r_candidates.append((0.0 - cy) / dy)
            r_candidates.append(((h - 1.0) - cy) / dy)

        r_candidates = [rr for rr in r_candidates if rr > 0]
        if not r_candidates:
            continue
        rmax = min(r_candidates)
        if rmax < 5:
            continue

        # Search window around guessed radius
        r_start = int(max(5, 0.55 * r_guess))
        r_end = int(min(rmax - 2, 1.25 * r_guess))
        if r_end <= r_start + 4:
            r_start = 1
            r_end = int(rmax - 2)

        rr = np.arange(r_start, r_end + 1, dtype=np.int32)
        xs = np.clip(np.rint(cx + rr * dx).astype(np.int32), 0, w - 1)
        ys = np.clip(np.rint(cy + rr * dy).astype(np.int32), 0, h - 1)

        prof = g[ys, xs].astype(np.float32)
        if prof.size < 6:
            continue

        # Smooth the profile to reduce vessel/noise spikes
        prof_s = np.convolve(prof, kernel, mode="same")

        # We want the strongest OUTWARD drop: inside(bright) -> outside(dark)
        drops = prof_s[:-1] - prof_s[1:]  # positive when going darker
        idx = int(np.argmax(drops))
        r_edge = int(rr[idx])

        x = int(np.clip(round(cx + r_edge * dx), 0, w - 1))
        y = int(np.clip(round(cy + r_edge * dy), 0, h - 1))
        pts.append([x, y])

    if len(pts) < 5:
        return np.zeros_like(m, dtype=np.uint8), False, "radial:too_few_points"

    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    (ecx, ecy), (MA, ma), angle = cv.fitEllipse(pts_np)

    rx = max(1, int(round((MA * 0.5) * scale)))
    ry = max(1, int(round((ma * 0.5) * scale)))

    out = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(
        out,
        (int(round(ecx)), int(round(ecy))),
        (rx, ry),
        angle,
        0,
        360,
        255,
        thickness=cv.FILLED,
    )
    return out, True, "radial:edge_drop_fitEllipse"



def compute_fov_mask(
    img_or_path: Union[str, Array],
    *,
    blur_ksize: int = 7,
    blur_kind: str = "median",     # "median" or "gaussian"
    close_ksize: Optional[int] = None,  # None => auto
    open_ksize: Optional[int] = None,   # None => auto
    do_hole_fill: bool = True,
    do_convex_hull: bool = True,
    ellipse_mode: EllipseMode = "auto",  # off / auto / force
    ellipse_scale: float = 1.02,         # slightly enlarge to recover small bites
    ellipse_min_fill_ratio: float = 0.975,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, FOVMaskDebug]]:
    """
    Compute a clean binary Field-Of-View (FOV) mask (0/255 uint8) for a fundus image.

    Fixes supported (exactly what we discussed):
      1) stronger closing (auto close/open sizes are bigger by default)
      2) convex hull (optional)
      3) ellipse fitting (robust for convex 'bites' that hull cannot fix)

    Returns:
      mask (uint8 0/255), and optionally debug object.
    """
    # Load if needed
    if isinstance(img_or_path, str):
        img = cv.imread(img_or_path, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_or_path}")
    else:
        img = img_or_path

    g = _to_gray_or_green(img)
    h, w = g.shape[:2]

    # Auto kernel sizes if not provided
    if close_ksize is None or open_ksize is None:
        auto_close, auto_open = _auto_ksizes(h, w)
        if close_ksize is None:
            close_ksize = auto_close
        if open_ksize is None:
            open_ksize = auto_open

    # Blur
    blur_ksize = _ensure_odd(blur_ksize)
    if blur_kind.lower() == "median":
        blurred = cv.medianBlur(g, blur_ksize)
    elif blur_kind.lower() == "gaussian":
        blurred = cv.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    else:
        raise ValueError("blur_kind must be 'median' or 'gaussian'.")
    g_corr = _flatfield_for_otsu(blurred)
    # Otsu threshold
    _, otsu = cv.threshold(g_corr, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Two candidates: otsu foreground or inverted foreground
    cand_a = otsu
    cand_b = cv.bitwise_not(otsu)

    # Morph cleanup each candidate (this is where close fixes bites)
    cand_a_m = _morph_cleanup(cand_a, close_ksize=close_ksize, open_ksize=open_ksize)
    cand_b_m = _morph_cleanup(cand_b, close_ksize=close_ksize, open_ksize=open_ksize)

    # Keep largest CC
    cand_a_l = _largest_connected_component(cand_a_m)
    cand_b_l = _largest_connected_component(cand_b_m)

    # Score both
    score_a = _candidate_score(cand_a_l)
    score_b = _candidate_score(cand_b_l)

    if score_b > score_a:
        chosen = cand_b_l
        chosen_inverted = True
        scores = {"normal": score_a, "inverted": score_b}
        after_morph = cand_b_m
    else:
        chosen = cand_a_l
        chosen_inverted = False
        scores = {"normal": score_a, "inverted": score_b}
        after_morph = cand_a_m

    filled = _fill_holes(chosen) if do_hole_fill else chosen.copy()
    hull = _convex_hull(filled) if do_convex_hull else filled.copy()

    # Ellipse handling (robust for convex straight "bites")
    ellipse_used = False
    ellipse_reason = "off"
    ellipse_mask = np.zeros_like(hull)

    if ellipse_mode == "force":
        ellipse_mask, ellipse_used, ellipse_reason = _fit_ellipse_mask_from_contour(hull, scale=ellipse_scale)
        final = ellipse_mask
    elif ellipse_mode == "auto":
        need, why = _should_use_ellipse(hull, min_fill_ratio=ellipse_min_fill_ratio)
        ellipse_reason = why
        if need:
            # NEW: fit from real image boundary (robust to the 'chord' failure)
            ellipse_mask, ellipse_used, fit_why = _fit_ellipse_mask_from_ring_edges(
                g_corr, hull, scale=ellipse_scale, ring_width=25
            )
            if ellipse_used:
                ellipse_reason = f"{ellipse_reason} | {fit_why}"
                final = ellipse_mask
            else:
                # fallback to your old contour-based ellipse
                ellipse_mask, ellipse_used, fit_why2 = _fit_ellipse_mask_from_contour(hull, scale=ellipse_scale)
                ellipse_reason = f"{ellipse_reason} | fallback:{fit_why2}"
                final = ellipse_mask if ellipse_used else hull
        else:
            final = hull

    else:  # "off"
        final = hull

    final = (final > 0).astype(np.uint8) * 255

    if not return_debug:
        return final

    dbg = FOVMaskDebug(
        green_or_gray=g,
        blurred=blurred,
        otsu=otsu,
        candidate_chosen=chosen,
        after_morph=after_morph,
        largest_cc=chosen,
        filled=filled,
        convex_hull=hull,
        ellipse_mask=ellipse_mask,
        final=final,
        chosen_inverted=chosen_inverted,
        scores=scores,
        ellipse_used=ellipse_used,
        ellipse_reason=ellipse_reason,
    )
    return final, dbg
