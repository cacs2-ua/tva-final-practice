import numpy as np
import cv2 as cv
import pytest

from vessel_pipeline_b import vessel_segmentation


pytestmark = pytest.mark.test_pipeline_b


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def _synthetic_fundus(seed: int = 0):
    """
    Build a fundus-like image:
      - dark background
      - bright circular FOV
      - darker vessel-like lines inside
    Returns: (img_bgr, gt_mask)
    """
    rng = np.random.default_rng(seed)
    h, w = 420, 520
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Bright-ish fundus disk in green channel
    center = (w // 2, h // 2)
    radius = 180
    cv.circle(img, center, radius, (0, 175, 0), thickness=-1)

    # Add mild shading (simulate illumination)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    shade = np.clip(1.0 - (dist / (radius * 1.2)), 0, 1)
    shade_u8 = (60 * shade).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + shade_u8.astype(np.int16), 0, 255).astype(np.uint8)

    # Ground-truth vessel mask
    gt = np.zeros((h, w), dtype=np.uint8)

    # Draw vessel-like lines (dark) inside the disk
    for _ in range(18):
        ang = float(rng.uniform(0, np.pi))
        r1 = float(rng.uniform(30, radius - 10))
        r2 = float(rng.uniform(30, radius - 10))

        x1 = int(center[0] + r1 * np.cos(ang))
        y1 = int(center[1] + r1 * np.sin(ang))
        x2 = int(center[0] - r2 * np.cos(ang))
        y2 = int(center[1] - r2 * np.sin(ang))

        thickness = int(rng.integers(1, 3))
        cv.line(gt, (x1, y1), (x2, y2), 255, thickness=thickness)
        cv.line(img, (x1, y1), (x2, y2), (0, 35, 0), thickness=thickness)

    # Add gaussian noise in green channel
    noise = rng.normal(0, 8, size=(h, w)).astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    img[:, :, 1] = np.clip(g + noise, 0, 255).astype(np.uint8)

    # Slight blur
    img[:, :, 1] = cv.GaussianBlur(img[:, :, 1], (5, 5), 0)

    return img, gt


def test_pipeline_b_output_binary_uint8():
    img, gt = _synthetic_fundus(0)
    out = vessel_segmentation(img, verbose=False)

    assert out.dtype == np.uint8
    assert out.shape == gt.shape
    vals = set(np.unique(out).tolist())
    assert vals.issubset({0, 255})


def test_pipeline_b_recovers_vessels_reasonable_iou():
    img, gt = _synthetic_fundus(1)
    out = vessel_segmentation(img, verbose=False)

    score = _iou(out, gt)

    # Synthetic test: expect decent overlap, not perfection.
    assert score > 0.45, f"IoU too low on synthetic vessels: {score:.4f}"


def test_pipeline_b_deterministic():
    img, _ = _synthetic_fundus(2)
    out1 = vessel_segmentation(img, verbose=False)
    out2 = vessel_segmentation(img, verbose=False)
    assert np.array_equal(out1, out2), "Pipeline B should be deterministic (fixed RNG)."
