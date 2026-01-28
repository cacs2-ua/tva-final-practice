import numpy as np
import cv2 as cv

from fov_mask import compute_fov_mask
from vessel_pipeline_c import segment_vessels_pipeline_c


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)


def _f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0)
    gt = (gt > 0)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return float(2 * tp) / float(2 * tp + fp + fn + 1e-9)


# ----------------------------
# FOV MASK TESTS (from your earlier test_fov_mask.py)
# ----------------------------

def test_fov_mask_is_binary_uint8():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv.circle(img, (128, 128), 90, (40, 180, 40), thickness=-1)
    mask = compute_fov_mask(img, do_hole_fill=True, do_convex_hull=True)

    assert mask.dtype == np.uint8
    vals = set(np.unique(mask).tolist())
    assert vals.issubset({0, 255})
    assert 0 in vals and 255 in vals


def test_fov_circle_high_iou_with_noise_and_holes():
    rng = np.random.default_rng(0)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    gt = np.zeros((h, w), dtype=np.uint8)
    cv.circle(gt, (w // 2, h // 2), 200, 255, thickness=-1)

    base = np.zeros((h, w), dtype=np.uint8)
    base[gt > 0] = 150
    noise = rng.normal(0, 12, size=(h, w)).astype(np.float32)
    noisy = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    holes = gt.copy()
    for _ in range(40):
        x = int(rng.integers(w // 2 - 150, w // 2 + 150))
        y = int(rng.integers(h // 2 - 150, h // 2 + 150))
        r = int(rng.integers(6, 18))
        cv.circle(holes, (x, y), r, 0, thickness=-1)

    noisy[holes == 0] = 0

    for _ in range(300):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if gt[y, x] == 0:
            noisy[y, x] = 255

    img[:, :, 1] = noisy

    mask = compute_fov_mask(
        img,
        blur_kind="median",
        blur_ksize=7,
        close_ksize=25,
        open_ksize=9,
        do_hole_fill=True,
        do_convex_hull=True,
    )

    score = _iou(mask, gt)
    assert score > 0.95, f"IoU too low: {score}"


def test_fov_works_on_grayscale_input():
    h, w = 300, 300
    gray = np.zeros((h, w), dtype=np.uint8)
    cv.circle(gray, (150, 150), 110, 160, thickness=-1)

    mask = compute_fov_mask(gray, do_hole_fill=True, do_convex_hull=True)
    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 255})


# ----------------------------
# PIPELINE C TESTS
# ----------------------------

def _make_synthetic_fundus_with_vessels(seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # FOV disk
    fov = np.zeros((h, w), dtype=np.uint8)
    cv.circle(fov, (w // 2, h // 2), 210, 255, thickness=-1)

    # background (green channel)
    bg = np.zeros((h, w), dtype=np.uint8)
    bg[fov > 0] = 160
    noise = rng.normal(0, 10, size=(h, w)).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # synthetic vessels: dark curvy-ish poly-lines
    gt_vessels = np.zeros((h, w), dtype=np.uint8)
    for k in range(12):
        x0 = int(rng.integers(w//2 - 120, w//2 + 120))
        y0 = int(rng.integers(h//2 - 120, h//2 + 120))
        pts = [(x0, y0)]
        for _ in range(6):
            x0 = int(np.clip(x0 + rng.integers(-50, 51), 0, w-1))
            y0 = int(np.clip(y0 + rng.integers(-50, 51), 0, h-1))
            pts.append((x0, y0))
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        thick = int(rng.integers(1, 3))
        cv.polylines(gt_vessels, [pts], isClosed=False, color=255, thickness=thick)

    # apply vessels into bg (darker)
    vessel_dark = bg.copy()
    vessel_dark[gt_vessels > 0] = np.clip(vessel_dark[gt_vessels > 0].astype(np.int16) - 70, 0, 255).astype(np.uint8)

    # add bright optic disc-ish blob (common false-positive source)
    cv.circle(vessel_dark, (w//2 + 70, h//2 - 40), 35, 230, thickness=-1)

    # outside fov black
    vessel_dark[fov == 0] = 0

    img[:, :, 1] = vessel_dark  # green channel
    return img, fov, gt_vessels


def test_pipeline_c_output_is_binary_and_masked():
    img, fov, _gt = _make_synthetic_fundus_with_vessels(seed=1)
    seg, dbg = segment_vessels_pipeline_c(img, return_debug=True, verbose=False)

    assert seg.dtype == np.uint8
    vals = set(np.unique(seg).tolist())
    assert vals.issubset({0, 255})

    # must be zero outside FOV
    assert int(seg[fov == 0].sum()) == 0

    # FOV mask also binary
    fov_vals = set(np.unique(dbg.fov_mask).tolist())
    assert fov_vals.issubset({0, 255})


def test_pipeline_c_detects_some_vessels_reasonably():
    img, fov, gt = _make_synthetic_fundus_with_vessels(seed=2)

    seg = segment_vessels_pipeline_c(
        img,
        threshold_mode="otsu",
        otsu_offset=-5,
        min_cc_area=15,
        morph_close_ksize=5,
        return_debug=False,
        verbose=False,
    )

    # evaluate only inside true FOV
    seg_in = seg.copy()
    seg_in[fov == 0] = 0
    gt_in = gt.copy()
    gt_in[fov == 0] = 0

    # We use a forgiving threshold because this is a heuristic baseline.
    f1 = _f1(seg_in, gt_in)
    assert f1 > 0.25, f"F1 too low for synthetic vessels: {f1}"
