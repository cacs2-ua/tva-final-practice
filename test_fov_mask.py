import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def test_mask_is_binary_uint8():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv.circle(img, (128, 128), 90, (40, 180, 40), thickness=-1)  # green-ish disk
    mask = compute_fov_mask(img, do_hole_fill=True, do_convex_hull=True)

    assert mask.dtype == np.uint8
    vals = set(np.unique(mask).tolist())
    assert vals.issubset({0, 255})
    assert 0 in vals and 255 in vals


def test_circle_fov_high_iou_with_noise_and_holes():
    rng = np.random.default_rng(0)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Ideal circle FOV
    gt = np.zeros((h, w), dtype=np.uint8)
    cv.circle(gt, (w // 2, h // 2), 200, 255, thickness=-1)

    # Create fundus-like content inside the FOV
    base = np.zeros((h, w), dtype=np.uint8)
    base[gt > 0] = 150
    noise = rng.normal(0, 12, size=(h, w)).astype(np.float32)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Add holes inside the FOV (simulate gaps)
    holes = gt.copy()
    for _ in range(40):
        x = int(rng.integers(w // 2 - 150, w // 2 + 150))
        y = int(rng.integers(h // 2 - 150, h // 2 + 150))
        r = int(rng.integers(6, 18))
        cv.circle(holes, (x, y), r, 0, thickness=-1)

    noisy[holes == 0] = 0

    # Add random bright specks outside FOV
    for _ in range(300):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if gt[y, x] == 0:
            noisy[y, x] = 255

    # Put into green channel (like real fundus)
    img[:, :, 1] = noisy

    mask = compute_fov_mask(
        img,
        blur_kind="median",
        blur_ksize=9,
        close_ksize=21,
        open_ksize=9,
        do_hole_fill=True,
        do_convex_hull=True,
    )

    score = _iou(mask, gt)
    assert score > 0.95, f"IoU too low: {score}"


def test_ellipse_fov_detected():
    rng = np.random.default_rng(1)
    h, w = 480, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    gt = np.zeros((h, w), dtype=np.uint8)
    cv.ellipse(gt, (w // 2, h // 2), (240, 170), 0, 0, 360, 255, thickness=-1)

    base = np.zeros((h, w), dtype=np.uint8)
    base[gt > 0] = 140
    noise = rng.normal(0, 10, size=(h, w)).astype(np.float32)
    g = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img[:, :, 1] = g

    mask = compute_fov_mask(
        img,
        blur_kind="gaussian",
        blur_ksize=9,
        close_ksize=21,
        open_ksize=11,
        do_hole_fill=True,
        do_convex_hull=True,
    )

    score = _iou(mask, gt)
    assert score > 0.93, f"IoU too low on ellipse: {score}"


def test_works_on_grayscale_input():
    h, w = 300, 300
    gray = np.zeros((h, w), dtype=np.uint8)
    cv.circle(gray, (150, 150), 110, 160, thickness=-1)

    mask = compute_fov_mask(gray, do_hole_fill=True, do_convex_hull=True)
    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 255})
