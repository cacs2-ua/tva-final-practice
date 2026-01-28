import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask
from vessel_pipeline_a import vessel_segmentation, PipelineAParams


def _make_synthetic_fundus(h=256, w=256, *, seed=0):
    """
    Create a synthetic 'fundus-like' image:
    - circular FOV
    - background illumination
    - a few dark vessel-like lines
    Returns (bgr_img, fov_gt_u8, vessel_gt_u8)
    """
    rng = np.random.default_rng(seed)

    # FOV ground truth
    fov = np.zeros((h, w), dtype=np.uint8)
    cv.circle(fov, (w // 2, h // 2), int(min(h, w) * 0.43), 255, -1)

    # Base green channel: inside FOV medium intensity, outside black
    g = np.zeros((h, w), dtype=np.uint8)
    g[fov > 0] = 160

    # Add smooth illumination gradient inside FOV
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (20.0 * (xx.astype(np.float32) / max(1, w - 1))).astype(np.float32)
    g = np.clip(g.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    g[fov == 0] = 0

    # Draw vessel-like dark lines (GT)
    vessel_gt = np.zeros((h, w), dtype=np.uint8)
    lines = [
        ((w//2 - 70, h//2 - 10), (w//2 + 70, h//2 + 10)),
        ((w//2 - 30, h//2 - 80), (w//2 + 10, h//2 + 80)),
        ((w//2 - 80, h//2 + 40), (w//2 + 60, h//2 + 20)),
    ]
    for (x0, y0), (x1, y1) in lines:
        cv.line(vessel_gt, (x0, y0), (x1, y1), 255, thickness=2)

    vessel_gt[fov == 0] = 0

    # Apply vessels to image (darken)
    g2 = g.copy()
    g2[vessel_gt > 0] = 60

    # Add mild noise
    noise = rng.normal(0, 6, size=(h, w)).astype(np.float32)
    g2 = np.clip(g2.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    g2[fov == 0] = 0

    # Build BGR
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :, 1] = g2
    return bgr, fov, vessel_gt


def test_fov_mask_binary_and_nonempty():
    img, fov_gt, _ = _make_synthetic_fundus()
    mask = compute_fov_mask(img, close_ksize=21, open_ksize=11, do_convex_hull=True)

    assert mask.dtype == np.uint8
    vals = set(np.unique(mask).tolist())
    assert vals.issubset({0, 255})
    assert (mask > 0).sum() > 1000


def test_pipeline_a_output_binary_shape_and_dtype():
    img, fov_gt, _ = _make_synthetic_fundus()
    out = vessel_segmentation(img, verbose=False)

    assert out.shape == (img.shape[0], img.shape[1])
    assert out.dtype == np.uint8
    vals = set(np.unique(out).tolist())
    assert vals.issubset({0, 255})


def test_pipeline_a_respects_fov_outside_is_zero():
    img, fov_gt, _ = _make_synthetic_fundus()
    out = vessel_segmentation(img, verbose=False)

    # Outside GT FOV should be zero (strict)
    assert int((out[fov_gt == 0] > 0).sum()) == 0


def test_pipeline_a_detects_some_vessels_on_easy_synthetic():
    img, fov_gt, vessel_gt = _make_synthetic_fundus()

    # Use slightly stronger grabcut iterations for stability in synthetic
    params = PipelineAParams(grabcut_iters=5)

    out = vessel_segmentation(img, verbose=False, params=params)

    # Should detect something inside FOV
    nz = int((out > 0).sum())
    assert nz > 50, f"Too few detected vessel pixels: {nz}"

    # Should not explode to fill most FOV
    fov_area = int((fov_gt > 0).sum())
    assert nz < int(0.60 * fov_area), f"Detected vessel area too large: {nz} vs fov_area={fov_area}"


def test_pipeline_a_blank_image_returns_empty_or_near_empty():
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out = vessel_segmentation(img, verbose=False)

    # On blank, we expect no vessels
    assert int((out > 0).sum()) == 0
