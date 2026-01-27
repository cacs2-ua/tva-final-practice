from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

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
    chosen_inverted: bool
    scores: Dict[str, float]


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

    if gray.dtype != np.uint8:
        # Normalize to uint8 safely
        g = gray.astype(np.float32)
        g = g - g.min()
        denom = (g.max() - g.min()) if (g.max() - g.min()) > 1e-6 else 1.0
        g = (255.0 * g / denom).clip(0, 255).astype(np.uint8)
        gray = g
    return gray


def _morph_cleanup(mask: Array,
                   close_ksize: int = 25,
                   open_ksize: int = 9) -> Array:
    """Close gaps then open to remove specks."""
    if close_ksize % 2 == 0 or open_ksize % 2 == 0:
        raise ValueError("Kernel sizes must be odd.")

    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_ksize, close_ksize))
    open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_ksize, open_ksize))

    m = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_k)
    m = cv.morphologyEx(m, cv.MORPH_OPEN, open_k)
    return m


def _largest_connected_component(mask: Array) -> Array:
    """Return a binary mask (0/255) containing only the largest foreground CC."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    num, labels, stats, _ = cv.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    # stats: [label, x, y, w, h, area] for labels 0..num-1
    areas = stats[1:, cv.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == best_idx] = 255
    return out


def _fill_holes(mask: Array) -> Array:
    """
    Fill holes inside a binary object using flood-fill on the background.
    Input/Output are 0/255 uint8.
    """
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]

    # Flood fill from border on inverted mask to find external background
    inv = cv.bitwise_not(m)
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv.floodFill(ff, flood_mask, (0, 0), 0)  # fill external background to 0

    # Holes are the remaining white regions in ff
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
      - low border contact (FOV usually doesn't touch border if black margin exists)
      - reasonably compact (circular-ish)
    """
    m = (candidate_mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    area = float(m.sum())
    area_ratio = area / float(h * w)

    if area < 10:
        return -1e9

    # Border contact ratio
    border = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    border_ratio = float(border.mean())  # fraction of border that is foreground

    # Centroid distance to image center (normalized)
    ys, xs = np.where(m > 0)
    cy, cx = float(np.mean(ys)), float(np.mean(xs))
    center_dist = np.sqrt((cy - (h / 2.0)) ** 2 + (cx - (w / 2.0)) ** 2)
    center_dist_norm = center_dist / np.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)

    # Compactness via contour circularity (if possible)
    circ = 0.0
    m255 = (m * 255).astype(np.uint8)
    contours, _ = cv.findContours(m255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        A = float(cv.contourArea(cnt))
        P = float(cv.arcLength(cnt, True))
        circ = (4.0 * np.pi * A) / (P * P + 1e-6)  # 0..1-ish

    # Prefer:
    # - area_ratio in [~0.15 .. ~0.95]
    # - small border_ratio
    # - small center_dist_norm
    # - higher circ
    # Combine into a single score:
    if area_ratio < 0.10 or area_ratio > 0.98:
        area_penalty = 0.6
    else:
        area_penalty = 0.0

    score = (
        + 1.5 * area_ratio
        - 2.0 * border_ratio
        - 0.8 * center_dist_norm
        + 0.4 * circ
        - area_penalty
    )
    return float(score)


def compute_fov_mask(
    img_or_path: Union[str, Array],
    *,
    blur_ksize: int = 7,
    blur_kind: str = "median",   # "median" or "gaussian"
    close_ksize: int = 25,
    open_ksize: int = 9,
    do_hole_fill: bool = True,
    do_convex_hull: bool = True,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, FOVMaskDebug]]:
    """
    Compute a clean binary Field-Of-View (FOV) mask (0/255 uint8) for a fundus image.

    Pipeline:
      1) green channel / grayscale
      2) blur
      3) Otsu threshold
      4) try both polarities (normal + inverted), pick best by heuristic score
      5) morphology close + open
      6) largest connected component
      7) optional hole filling
      8) optional convex hull

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

    # Blur
    if blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be odd.")
    if blur_kind.lower() == "median":
        blurred = cv.medianBlur(g, blur_ksize)
    elif blur_kind.lower() == "gaussian":
        blurred = cv.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    else:
        raise ValueError("blur_kind must be 'median' or 'gaussian'.")

    # Otsu threshold
    _, otsu = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Two candidates: otsu foreground or inverted foreground
    cand_a = otsu
    cand_b = cv.bitwise_not(otsu)

    # Morph cleanup each candidate
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
    else:
        chosen = cand_a_l
        chosen_inverted = False
        scores = {"normal": score_a, "inverted": score_b}

    filled = _fill_holes(chosen) if do_hole_fill else chosen.copy()
    hull = _convex_hull(filled) if do_convex_hull else filled.copy()

    final = (hull > 0).astype(np.uint8) * 255

    if not return_debug:
        return final

    dbg = FOVMaskDebug(
        green_or_gray=g,
        blurred=blurred,
        otsu=otsu,
        candidate_chosen=chosen,
        after_morph=(cand_b_m if chosen_inverted else cand_a_m),
        largest_cc=chosen,
        filled=filled,
        convex_hull=hull,
        chosen_inverted=chosen_inverted,
        scores=scores,
    )
    return final, dbg
