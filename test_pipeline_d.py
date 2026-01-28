import numpy as np
import cv2 as cv

from vessel_morph_pipeline import (
    VesselMorphParams,
    segment_vessels_morphology,
)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def _make_synthetic_fundus(h=512, w=512, seed=0):
    """
    Synthetic "fundus-like" image:
      - bright background inside circular FOV
      - dark vessel lines (so they become bright after inversion)
      - noise
    """
    rng = np.random.default_rng(seed)

    # FOV mask (circle)
    fov = np.zeros((h, w), dtype=np.uint8)
    cv.circle(fov, (w // 2, h // 2), int(min(h, w) * 0.42), 255, thickness=-1)

    # Base fundus intensity inside FOV
    g = np.zeros((h, w), dtype=np.uint8)
    g[fov > 0] = 160

    # Add gentle illumination gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (20.0 * (xx / w) + 10.0 * (yy / h)).astype(np.float32)
    g = np.clip(g.astype(np.float32) + grad, 0, 255).astype(np.uint8)

    # Ground truth vessels mask
    gt = np.zeros((h, w), dtype=np.uint8)

    # Draw vessels as dark lines in green channel + mark GT as 255
    lines = [
        ((80, 260), (430, 260), 2),      # horizontal
        ((256, 80), (256, 430), 2),      # vertical
        ((130, 130), (380, 380), 2),     # diag
        ((380, 140), (150, 390), 1),     # other diag thinner
    ]
    for (x0, y0), (x1, y1), thick in lines:
        cv.line(g, (x0, y0), (x1, y1), color=70, thickness=thick)   # dark vessel
        cv.line(gt, (x0, y0), (x1, y1), color=255, thickness=thick)

    # Add noise inside FOV
    noise = rng.normal(0, 10, size=(h, w)).astype(np.float32)
    g2 = g.astype(np.float32)
    g2[fov > 0] = np.clip(g2[fov > 0] + noise[fov > 0], 0, 255)
    g = g2.astype(np.uint8)

    # Build BGR image with green channel used
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = g
    return img, fov, gt


def test_pipeline_d_output_is_binary_uint8():
    img, fov, gt = _make_synthetic_fundus()
    params = VesselMorphParams(verbose=False)
    out = segment_vessels_morphology(img, fov, params=params, return_debug=False)

    assert out.dtype == np.uint8
    vals = set(np.unique(out).tolist())
    assert vals.issubset({0, 255})
    assert 0 in vals


def test_pipeline_d_synthetic_iou_reasonable():
    img, fov, gt = _make_synthetic_fundus()

    # Make it a bit easier/consistent for tests:
    params = VesselMorphParams(
        verbose=False,
        thresh_method="percentile",
        thresh_percentile=90.0,
        post_close_ksize=3,
        post_open_ksize=3,
        cc_min_area=10,
        cc_min_elongation=1.8,
        cc_max_extent=0.60,
        cc_max_area_frac=0.25,
    )

    out = segment_vessels_morphology(img, fov, params=params, return_debug=False)
    score = _iou(out, gt)

    # Synthetic data should be quite easy: expect decent overlap
    assert score > 0.55, f"IoU too low on synthetic vessels: {score:.3f}"


def test_cc_filter_removes_round_blob():
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    fov = np.zeros((h, w), dtype=np.uint8)
    cv.circle(fov, (w // 2, h // 2), 110, 255, thickness=-1)

    # Create green: bright background
    g = np.zeros((h, w), dtype=np.uint8)
    g[fov > 0] = 170

    # Vessel: dark line (good)
    cv.line(g, (40, 130), (220, 130), 60, thickness=2)

    # Blob: dark circle (junk)
    cv.circle(g, (180, 70), 18, 60, thickness=-1)

    img[:, :, 1] = g

    params = VesselMorphParams(
        verbose=False,
        thresh_method="percentile",
        thresh_percentile=88.0,
        cc_min_area=10,
        cc_min_elongation=2.2,   # enforce elongation so blob is removed
        cc_max_extent=0.50,
        cc_max_area_frac=0.25,
    )

    out = segment_vessels_morphology(img, fov, params=params, return_debug=False)

    # Check blob center should be mostly background
    blob_patch = out[70-5:70+6, 180-5:180+6]
    assert float((blob_patch > 0).mean()) < 0.25, "Round blob was not removed as expected"
