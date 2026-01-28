# Pipeline A: Illumination correction -> vessel enhancement -> (GrabCut-style) graph cut refinement
# No deep learning / no supervised learning.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2 as cv

from pathlib import Path


try:
    # If your project already has a robust FOV detector, we will reuse it.
    from fov_mask import compute_fov_mask  # type: ignore
except Exception:
    compute_fov_mask = None  # fallback will be used

def _load_img_if_path(img_or_path):
    """
    Accept either:
      - np.ndarray image
      - str / Path path to an image on disk
    Returns np.ndarray (uint8).
    """
    if isinstance(img_or_path, (str, Path)):
        p = Path(img_or_path)
        im = cv.imread(str(p), cv.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"Could not read image at: {p}")
        return im
    return img_or_path

@dataclass
class PipelineAParams:
    # --- safety / early exits ---
    blank_std_thresh: float = 1.0          # if std(gray) < this AND max(gray) small -> blank
    blank_max_thresh: int = 2
    min_vessel_score: float = 0.03         # if vesselness max < this -> return empty

    # --- illumination correction / contrast ---
    bg_sigma: float = 25.0                 # background estimation for illumination correction
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # --- vessel enhancement (multi-scale black-hat) ---
    blackhat_ksizes: Tuple[int, int, int] = (9, 15, 23)  # odd sizes
    smooth_sigma: float = 1.2

    # --- GrabCut (GraphCut) refinement ---
    grabcut_iters: int = 3
    pr_fg_quantile: float = 0.80           # probable FG threshold quantile
    fg_quantile: float = 0.93              # sure FG threshold quantile
    bg_quantile: float = 0.55              # sure BG threshold quantile
    min_fg_pixels: int = 40                # ensure some FG seeds exist
    min_pr_fg_pixels: int = 200            # ensure some probable FG exists

    # --- postprocessing ---
    min_component_area: int = 20
    open_ksize: int = 3                    # remove specks
    close_ksize: int = 3                   # connect tiny gaps

    # --- fallback FOV ---
    fov_nonzero_thresh: int = 1            # for fallback FOV from green channel
    fov_close_ksize: int = 21


def _dbg(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)


def _to_green(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        # works for both RGB and BGR because G is channel 1 in both
        return img[:, :, 1]
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _fallback_fov_mask(g: np.ndarray, params: PipelineAParams) -> np.ndarray:
    # basic: anything non-black, then close to get one blob, then take largest CC
    thr = (g > int(params.fov_nonzero_thresh)).astype(np.uint8) * 255
    if thr.sum() == 0:
        return np.zeros_like(g, dtype=np.uint8)

    k = int(params.fov_close_ksize)
    if k % 2 == 0:
        k += 1
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
    closed = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv.connectedComponentsWithStats((closed > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return (closed > 0).astype(np.uint8) * 255

    # largest non-background component
    areas = stats[1:, cv.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    fov = (labels == idx).astype(np.uint8) * 255
    return fov


def _compute_fov(g: np.ndarray, params: PipelineAParams, verbose: bool, debug: Dict[str, Any]) -> np.ndarray:
    if compute_fov_mask is not None:
        try:
            fov = compute_fov_mask(g)  # project function
            fov = (fov > 0).astype(np.uint8) * 255
            debug["fov_source"] = "compute_fov_mask"
            return fov
        except Exception as e:
            _dbg(verbose, f"[PipelineA] compute_fov_mask failed -> fallback. Reason: {e!r}")
            debug["fov_source"] = "fallback_due_to_exception"

    fov = _fallback_fov_mask(g, params)
    debug["fov_source"] = "fallback"
    return fov


def _illumination_correct_and_clahe(g: np.ndarray, params: PipelineAParams) -> np.ndarray:
    g_f = g.astype(np.float32)
    bg = cv.GaussianBlur(g_f, (0, 0), sigmaX=float(params.bg_sigma), sigmaY=float(params.bg_sigma))
    corr = g_f - bg
    # normalize to 8-bit
    corr_u8 = cv.normalize(corr, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    clahe = cv.createCLAHE(clipLimit=float(params.clahe_clip),
                           tileGridSize=(int(params.clahe_tile), int(params.clahe_tile)))
    eq = clahe.apply(corr_u8)
    return eq


def _vessel_enhance(eq_u8: np.ndarray, fov_u8: np.ndarray, params: PipelineAParams) -> np.ndarray:
    # Multi-scale black-hat to highlight dark thin structures
    scores = []
    for k in params.blackhat_ksizes:
        k = int(k)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        bh = cv.morphologyEx(eq_u8, cv.MORPH_BLACKHAT, kernel)
        scores.append(bh.astype(np.float32))

    v = np.maximum.reduce(scores) if len(scores) > 1 else scores[0]
    v = v / 255.0

    # smooth a bit
    if float(params.smooth_sigma) > 0:
        v = cv.GaussianBlur(v, (0, 0), sigmaX=float(params.smooth_sigma), sigmaY=float(params.smooth_sigma))

    # zero outside FOV
    v[fov_u8 == 0] = 0.0

    # normalize inside FOV for stability
    inside = v[fov_u8 > 0]
    if inside.size > 0:
        vmin = float(np.min(inside))
        vmax = float(np.max(inside))
        if vmax > vmin + 1e-6:
            v = (v - vmin) / (vmax - vmin)
            v[fov_u8 == 0] = 0.0
    return v


def _ensure_min_seeds(score: np.ndarray, fov: np.ndarray, mask: np.ndarray, label: int, min_pixels: int) -> None:
    # Add top-K pixels as seeds if too few exist.
    cur = int((mask == label).sum())
    if cur >= int(min_pixels):
        return
    inside = np.where(fov > 0)
    if inside[0].size == 0:
        return
    vals = score[inside]
    if vals.size == 0:
        return
    k = int(min_pixels - cur)
    k = min(k, vals.size)
    if k <= 0:
        return
    # indices of top-k by score
    order = np.argpartition(vals, -k)[-k:]
    ys = inside[0][order]
    xs = inside[1][order]
    mask[ys, xs] = label


def _postprocess(bin_u8: np.ndarray, params: PipelineAParams) -> np.ndarray:
    m = (bin_u8 > 0).astype(np.uint8) * 255

    # opening/closing
    ok = int(params.open_ksize)
    ck = int(params.close_ksize)
    if ok >= 3:
        if ok % 2 == 0:
            ok += 1
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok, ok))
        m = cv.morphologyEx(m, cv.MORPH_OPEN, k)
    if ck >= 3:
        if ck % 2 == 0:
            ck += 1
        k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ck, ck))
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, k)

    # remove tiny components
    n, labels, stats, _ = cv.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return m

    out = np.zeros_like(m)
    for i in range(1, n):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area >= int(params.min_component_area):
            out[labels == i] = 255
    return out


def vessel_segmentation(
    img: np.ndarray,
    verbose: bool = False,
    params: Optional[PipelineAParams] = None,
    return_debug: bool = False,
):
    """
    Returns a binary vessel mask (uint8, values {0,255}).

    Key fixes for your failing tests:
    - Hard blank-image early exit -> ALWAYS returns all zeros.
    - Vesselness sanity check (max score too small) -> returns all zeros.
    - Robust GrabCut seeding + fallback seed enforcement so synthetic lines are detected.
    """
    if params is None:
        params = PipelineAParams()

    debug: Dict[str, Any] = {}

    img = _load_img_if_path(img)

    g = _to_green(img)
    g_u8 = g.astype(np.uint8)

    # ---- (1) blank / near-blank early exit ----
    g_std = float(np.std(g_u8))
    g_max = int(np.max(g_u8)) if g_u8.size else 0
    debug["g_std"] = g_std
    debug["g_max"] = g_max
    _dbg(verbose, f"[PipelineA] green std={g_std:.3f}, max={g_max}")

    if (g_std < float(params.blank_std_thresh)) and (g_max <= int(params.blank_max_thresh)):
        out = np.zeros(g_u8.shape, dtype=np.uint8)
        debug["early_exit"] = "blank_image"
        if return_debug:
            return out, debug
        return out

    # ---- (2) FOV ----
    fov_u8 = _compute_fov(g_u8, params, verbose, debug)
    fov_area = int((fov_u8 > 0).sum())
    debug["fov_area"] = fov_area
    _dbg(verbose, f"[PipelineA] FOV source={debug.get('fov_source')} area={fov_area}")

    if fov_area == 0:
        out = np.zeros(g_u8.shape, dtype=np.uint8)
        debug["early_exit"] = "empty_fov"
        if return_debug:
            return out, debug
        return out

    # ---- (3) illumination correction + CLAHE ----
    eq = _illumination_correct_and_clahe(g_u8, params)

    # ---- (4) vessel enhancement ----
    score = _vessel_enhance(eq, fov_u8, params)
    smax = float(np.max(score)) if score.size else 0.0
    debug["vessel_score_max"] = smax
    _dbg(verbose, f"[PipelineA] vessel_score max={smax:.4f}")

    # If no vessel signal at all, return empty (fixes blank-like images + avoids all-foreground grabcut)
    if smax < float(params.min_vessel_score):
        out = np.zeros(g_u8.shape, dtype=np.uint8)
        debug["early_exit"] = "no_vessel_signal"
        if return_debug:
            return out, debug
        return out

    # ---- (5) build GrabCut mask (GraphCut) ----
    inside = score[fov_u8 > 0]
    pr_thr = float(np.quantile(inside, float(params.pr_fg_quantile)))
    fg_thr = float(np.quantile(inside, float(params.fg_quantile)))
    bg_thr = float(np.quantile(inside, float(params.bg_quantile)))
    debug.update({"pr_thr": pr_thr, "fg_thr": fg_thr, "bg_thr": bg_thr})
    _dbg(verbose, f"[PipelineA] thresholds: bg={bg_thr:.4f}, pr_fg={pr_thr:.4f}, fg={fg_thr:.4f}")

    # init: background everywhere, probable background inside FOV
    gc = np.full(g_u8.shape, cv.GC_BGD, dtype=np.uint8)
    gc[fov_u8 > 0] = cv.GC_PR_BGD

    # probable FG / sure FG / sure BG
    prob_fg = (score >= pr_thr) & (fov_u8 > 0)
    sure_fg = (score >= fg_thr) & (fov_u8 > 0)
    sure_bg = (score <= bg_thr) & (fov_u8 > 0)

    gc[prob_fg] = cv.GC_PR_FGD
    gc[sure_fg] = cv.GC_FGD
    gc[sure_bg] = cv.GC_BGD
    gc[fov_u8 == 0] = cv.GC_BGD

    # enforce a minimum number of seeds so synthetic doesn’t end up with zero FG seeds
    _ensure_min_seeds(score, fov_u8, gc, cv.GC_FGD, int(params.min_fg_pixels))
    _ensure_min_seeds(score, fov_u8, gc, cv.GC_PR_FGD, int(params.min_pr_fg_pixels))

    debug["n_sure_fg"] = int((gc == cv.GC_FGD).sum())
    debug["n_pr_fg"] = int((gc == cv.GC_PR_FGD).sum())
    debug["n_sure_bg"] = int((gc == cv.GC_BGD).sum())
    _dbg(verbose, f"[PipelineA] seeds: sure_fg={debug['n_sure_fg']}, pr_fg={debug['n_pr_fg']}")

    # ---- (6) GrabCut ----
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv.grabCut(
            img if img.ndim == 3 else cv.cvtColor(g_u8, cv.COLOR_GRAY2BGR),
            gc,
            None,
            bgdModel,
            fgdModel,
            int(params.grabcut_iters),
            mode=cv.GC_INIT_WITH_MASK,
        )
        debug["grabcut_ok"] = True
    except Exception as e:
        # If grabcut fails, we fallback to thresholding score (still should pass synthetic test).
        debug["grabcut_ok"] = False
        debug["grabcut_error"] = repr(e)
        _dbg(verbose, f"[PipelineA] grabCut failed -> fallback threshold. Reason: {e!r}")

        thr = float(np.quantile(inside, 0.90))
        out = ((score >= thr) & (fov_u8 > 0)).astype(np.uint8) * 255
        out = _postprocess(out, params)
        if return_debug:
            return out, debug
        return out

    # Extract FG + Probable FG, then clamp to FOV
    out = np.where((gc == cv.GC_FGD) | (gc == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    out[fov_u8 == 0] = 0

    # ---- (7) postprocess ----
    out = _postprocess(out, params)

    debug["out_nz"] = int((out > 0).sum())
    _dbg(verbose, f"[PipelineA] output nz={debug['out_nz']}")

    if return_debug:
        return out, debug
    return out
