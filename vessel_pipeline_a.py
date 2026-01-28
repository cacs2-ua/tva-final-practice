from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask


Array = np.ndarray


@dataclass
class PipelineAParams:
    # --- FOV mask ---
    fov_close_ksize: int = 21
    fov_open_ksize: int = 11
    fov_do_convex_hull: bool = True
    fov_ellipse_mode: str = "auto"   # "off" | "auto" | "force"
    fov_ellipse_scale: float = 1.01

    # --- Retinex-like correction (paper-inspired) ---
    bilateral_d: int = 9
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 25.0
    retinex_eps: float = 1.0  # for log(I+eps)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8

    # --- Vesselness (Frangi/Hessian substitute for Local Phase) ---
    frangi_sigmas: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
    frangi_beta: float = 0.5
    frangi_c_percentile: float = 90.0  # dynamic c from S distribution
    # if you want to bias toward thin vessels, increase weight on small sigmas:
    sigma_weight_power: float = 0.0    # 0 => equal weight, >0 biases large sigmas

    # --- GraphCut (GrabCut) ---
    grabcut_iters: int = 5
    # quantiles for init mask inside FOV:
    q_sure_fg: float = 0.995
    q_prob_fg: float = 0.970
    q_prob_bg: float = 0.300

    # --- Postprocess ---
    post_open_ksize: int = 3
    post_close_ksize: int = 3
    min_component_area: int = 30  # remove tiny blobs (not too aggressive)


def _to_uint8(x: Array) -> Array:
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    xf = xf - float(xf.min())
    denom = float(xf.max() - xf.min())
    if denom < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    y = (255.0 * xf / denom).clip(0, 255).astype(np.uint8)
    return y


def _green_or_gray(img: Array) -> Array:
    if img.ndim == 2:
        return _to_uint8(img)
    if img.ndim == 3 and img.shape[2] >= 3:
        return _to_uint8(img[:, :, 1])  # green channel (best vessel contrast)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _retinex_like(gray_u8: Array, p: PipelineAParams) -> Array:
    """
    Retinex approximation from paper:
      R(x) = log(I(x)+eps) - log(Bilateral(I)(x)+eps)
    Then normalize and apply CLAHE.
    """
    I = gray_u8.astype(np.float32)
    L = cv.bilateralFilter(I, d=int(p.bilateral_d),
                           sigmaColor=float(p.bilateral_sigma_color),
                           sigmaSpace=float(p.bilateral_sigma_space))
    eps = float(p.retinex_eps)
    R = np.log(I + eps) - np.log(L + eps)

    # Normalize to uint8
    R_u8 = _to_uint8(R)

    # CLAHE (helps with inhomogeneity / local contrast)
    clahe = cv.createCLAHE(
        clipLimit=float(p.clahe_clip_limit),
        tileGridSize=(int(p.clahe_tile_grid), int(p.clahe_tile_grid))
    )
    out = clahe.apply(R_u8)
    return out


def _hessian_second_derivatives(img_f32: Array) -> Tuple[Array, Array, Array]:
    """
    Compute second derivatives (Ixx, Iyy, Ixy) using Sobel on a smoothed image.
    """
    Ixx = cv.Sobel(img_f32, cv.CV_32F, 2, 0, ksize=3)
    Iyy = cv.Sobel(img_f32, cv.CV_32F, 0, 2, ksize=3)
    Ixy = cv.Sobel(img_f32, cv.CV_32F, 1, 1, ksize=3)
    return Ixx, Iyy, Ixy


def _frangi_vesselness_2d(gray_u8: Array, p: PipelineAParams, *, verbose: bool = False) -> Array:
    """
    Practical substitute for Local Phase enhancement:
    Multi-scale Hessian/Frangi vesselness on an inverted image (vessels become bright ridges).
    Returns vesselness in [0,1] float32.
    """
    # Invert: vessels are dark in green channel, so invert to make vessels bright
    inv = (255 - gray_u8).astype(np.float32) / 255.0

    sigmas = list(p.frangi_sigmas)
    beta = float(p.frangi_beta)
    eps = 1e-12

    vesselness_max = np.zeros_like(inv, dtype=np.float32)

    for s in sigmas:
        # Smooth
        k = int(max(3, 2 * int(round(3 * s)) + 1))
        sm = cv.GaussianBlur(inv, (k, k), float(s))

        Ixx, Iyy, Ixy = _hessian_second_derivatives(sm)

        # Scale normalization (Frangi uses sigma^2)
        scale = float(s * s)
        Ixx *= scale
        Iyy *= scale
        Ixy *= scale

        # Eigenvalues of Hessian for each pixel (2x2 symmetric):
        # [Ixx Ixy; Ixy Iyy]
        tmp = np.sqrt((Ixx - Iyy) ** 2 + 4.0 * (Ixy ** 2))
        l1 = 0.5 * (Ixx + Iyy - tmp)
        l2 = 0.5 * (Ixx + Iyy + tmp)

        # Sort so |l1| <= |l2|
        abs_l1 = np.abs(l1)
        abs_l2 = np.abs(l2)
        swap = abs_l1 > abs_l2
        l1s = l1.copy()
        l2s = l2.copy()
        l1s[swap], l2s[swap] = l2[swap], l1[swap]
        l1, l2 = l1s, l2s

        # For bright tubular structures: l2 should be negative (ridge)
        # If l2 > 0 -> background
        mask = (l2 < 0).astype(np.float32)

        Rb = (l1 / (l2 + eps)) ** 2
        S2 = l1 ** 2 + l2 ** 2

        # Dynamic c from percentile of S (robust)
        c = np.percentile(np.sqrt(S2), float(p.frangi_c_percentile))
        c = float(max(c, 1e-6))

        V = np.exp(-Rb / (2.0 * (beta ** 2))) * (1.0 - np.exp(-S2 / (2.0 * (c ** 2))))
        V *= mask

        # Optional sigma weighting
        if p.sigma_weight_power != 0.0:
            V *= (float(s) ** float(p.sigma_weight_power))

        vesselness_max = np.maximum(vesselness_max, V.astype(np.float32))

    # Normalize to [0,1]
    vmax = float(np.max(vesselness_max))
    if vmax > 1e-12:
        vesselness = (vesselness_max / vmax).clip(0.0, 1.0).astype(np.float32)
    else:
        vesselness = vesselness_max.clip(0.0, 1.0).astype(np.float32)

    if verbose:
        vv = vesselness
        print("[PipelineA][Vesselness] min/max:", float(vv.min()), float(vv.max()))
        for q in (50, 75, 90, 95, 97, 99):
            print(f"[PipelineA][Vesselness] p{q}:", float(np.percentile(vv, q)))

    return vesselness


def _bbox_from_mask(mask_u8: Array) -> Tuple[int, int, int, int]:
    """
    Bounding box (x,y,w,h) around non-zero region. If empty, returns full image.
    """
    ys, xs = np.where(mask_u8 > 0)
    h, w = mask_u8.shape[:2]
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, (x1 - x0 + 1), (y1 - y0 + 1)


def _init_grabcut_mask(vesselness: Array, fov_u8: Array, p: PipelineAParams, *, verbose: bool = False) -> Array:
    """
    Initialize GrabCut mask using vesselness quantiles.
    """
    gc = np.full(fov_u8.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)

    fov = (fov_u8 > 0)
    gc[~fov] = cv.GC_BGD  # sure background outside FOV

    vals = vesselness[fov]
    if vals.size < 10:
        return gc

    q_sure_fg = float(np.quantile(vals, float(p.q_sure_fg)))
    q_prob_fg = float(np.quantile(vals, float(p.q_prob_fg)))
    q_prob_bg = float(np.quantile(vals, float(p.q_prob_bg)))

    sure_fg = fov & (vesselness >= q_sure_fg)
    prob_fg = fov & (vesselness >= q_prob_fg)
    prob_bg = fov & (vesselness <= q_prob_bg)

    gc[prob_bg] = cv.GC_PR_BGD
    gc[prob_fg] = cv.GC_PR_FGD
    gc[sure_fg] = cv.GC_FGD

    if verbose:
        print("[PipelineA][GrabCut init] q_prob_bg:", q_prob_bg, "q_prob_fg:", q_prob_fg, "q_sure_fg:", q_sure_fg)
        unique, counts = np.unique(gc, return_counts=True)
        print("[PipelineA][GrabCut init] label counts:", dict(zip(unique.tolist(), counts.tolist())))

    return gc


def _postprocess(binary_u8: Array, fov_u8: Array, p: PipelineAParams, *, verbose: bool = False) -> Array:
    """
    Light cleanup: open/close + remove tiny components, keep only within FOV.
    """
    out = (binary_u8 > 0).astype(np.uint8) * 255
    fov = (fov_u8 > 0)

    # Morph
    ok = max(1, int(p.post_open_ksize))
    ck = max(1, int(p.post_close_ksize))
    open_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ok, ok))
    close_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ck, ck))

    out = cv.morphologyEx(out, cv.MORPH_OPEN, open_k)
    out = cv.morphologyEx(out, cv.MORPH_CLOSE, close_k)

    # Remove tiny CCs (not too aggressive!)
    m = (out > 0).astype(np.uint8)
    num, labels, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        keep = np.zeros(num, dtype=bool)
        keep[0] = False
        minA = int(p.min_component_area)
        for i in range(1, num):
            area = int(stats[i, cv.CC_STAT_AREA])
            if area >= minA:
                keep[i] = True

        out2 = np.zeros_like(out, dtype=np.uint8)
        for i in range(1, num):
            if keep[i]:
                out2[labels == i] = 255
        out = out2

    out[~fov] = 0

    if verbose:
        nz = int((out > 0).sum())
        print("[PipelineA][Post] nonzero pixels:", nz)

    return out


def vessel_segmentation(
    input_image: Union[str, Array],
    *,
    verbose: bool = False,
    params: Optional[PipelineAParams] = None,
) -> Array:
    """
    REQUIRED API: must exist and return a binary 0/255 uint8 mask of vessels.

    input_image: path (str) OR already-loaded image (np.ndarray).
    """
    p = params or PipelineAParams()

    # Load image
    if isinstance(input_image, str):
        img = cv.imread(input_image, cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read input image: {input_image}")
    else:
        img = input_image

    if verbose:
        print("[PipelineA] img shape:", img.shape, "dtype:", img.dtype)

    g = _green_or_gray(img)
    H, W = g.shape[:2]

    # --- FOV mask ---
    fov_u8 = compute_fov_mask(
        img,
        close_ksize=int(p.fov_close_ksize),
        open_ksize=int(p.fov_open_ksize),
        do_convex_hull=bool(p.fov_do_convex_hull),
        ellipse_mode=str(p.fov_ellipse_mode),
        ellipse_scale=float(p.fov_ellipse_scale),
        return_debug=False,
    )

    if verbose:
        fov_ratio = float((fov_u8 > 0).mean())
        print("[PipelineA] FOV area ratio:", fov_ratio)

    # Crop to FOV bbox for speed
    x, y, w, h = _bbox_from_mask(fov_u8)
    g_crop = g[y:y+h, x:x+w]
    fov_crop = fov_u8[y:y+h, x:x+w]

    # Mask outside FOV
    g_crop_masked = g_crop.copy()
    g_crop_masked[fov_crop == 0] = 0

    # --- Retinex-like + CLAHE ---
    ret = _retinex_like(g_crop_masked, p)
    ret[fov_crop == 0] = 0

    if verbose:
        print("[PipelineA][Retinex] min/max:", int(ret.min()), int(ret.max()))

    # --- Vessel enhancement (Frangi substitute) ---
    vesselness = _frangi_vesselness_2d(ret, p, verbose=verbose)  # float [0,1]
    vesselness[fov_crop == 0] = 0.0

    # --- GraphCut using GrabCut ---
    # Build a 3-channel feature image for GrabCut:
    # channel0 = retinex/clahe, channel1 = vesselness*255, channel2 = retinex/clahe
    v255 = (vesselness * 255.0).clip(0, 255).astype(np.uint8)
    feat = cv.merge([ret, v255, ret])

    gc_mask = _init_grabcut_mask(vesselness, fov_crop, p, verbose=verbose)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect_dummy = (0, 0, feat.shape[1], feat.shape[0])  # ignored in INIT_WITH_MASK
    cv.grabCut(
        feat,
        gc_mask,
        rect_dummy,
        bgdModel,
        fgdModel,
        iterCount=int(p.grabcut_iters),
        mode=cv.GC_INIT_WITH_MASK,
    )

    seg = np.where((gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    seg[fov_crop == 0] = 0

    if verbose:
        print("[PipelineA][GrabCut] seg nonzero:", int((seg > 0).sum()))

    # --- Postprocess ---
    seg = _postprocess(seg, fov_crop, p, verbose=verbose)

    # Paste back to full resolution
    out = np.zeros((H, W), dtype=np.uint8)
    out[y:y+h, x:x+w] = seg
    out[fov_u8 == 0] = 0

    if verbose:
        print("[PipelineA] output unique:", np.unique(out).tolist()[:10])

    return out
