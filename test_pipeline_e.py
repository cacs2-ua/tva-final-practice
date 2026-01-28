import numpy as np
import cv2 as cv

from vessel_pipeline_e import vessel_segmentation


def _make_synth_fundus(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)

    img = np.zeros((h, w, 3), dtype=np.uint8)

    # FOV disk (brighter interior)
    fov = np.zeros((h, w), dtype=np.uint8)
    cv.circle(fov, (w // 2, h // 2), int(min(h, w) * 0.42), 255, thickness=-1)

    base = np.zeros((h, w), dtype=np.uint8)
    base[fov > 0] = 160

    noise = rng.normal(0, 8, size=(h, w)).astype(np.float32)
    g = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Draw vessel-like dark lines inside FOV
    for a in [10, 35, 75, 110, 150]:
        x0 = int(w * 0.2)
        y0 = int(h * 0.5)
        x1 = int(w * 0.8)
        y1 = int(h * 0.5)
        M = cv.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
        pts = np.array([[x0, y0], [x1, y1]], dtype=np.float32)
        pts = cv.transform(np.array([pts]), M)[0].astype(int)
        cv.line(g, tuple(pts[0]), tuple(pts[1]), color=60, thickness=2)

    # Add a dark round blob (should be removed by CC filtering)
    blob_center = (int(w * 0.70), int(h * 0.35))
    cv.circle(g, blob_center, 10, 50, thickness=-1)

    # Apply FOV cutoff
    g[fov == 0] = 0

    img[:, :, 1] = g  # put in green channel
    return img, blob_center


def test_pipeline_e_output_is_binary_uint8_and_shape(tmp_path):
    img, _ = _make_synth_fundus()
    p = tmp_path / "synth.png"
    assert cv.imwrite(str(p), img)

    out = vessel_segmentation(str(p), verbose=False)
    assert out.dtype == np.uint8
    assert out.shape[:2] == img.shape[:2]
    vals = set(np.unique(out).tolist())
    assert vals.issubset({0, 255})


def test_pipeline_e_removes_round_blob_keeps_lines(tmp_path):
    img, blob_center = _make_synth_fundus()
    p = tmp_path / "synth.png"
    assert cv.imwrite(str(p), img)

    out = vessel_segmentation(
        str(p),
        verbose=False,
        # slightly lenient threshold so both lines+blob get detected upstream,
        # then CC filter must remove blob
        thr_offset=-5,
        cc_min_area=25,
        cc_min_area_if_elongated=12,
        cc_min_elongation=2.0,
        cc_max_circularity=0.42,
        cc_max_extent_blob=0.72,
    )

    # Check blob area mostly removed
    bx, by = blob_center
    patch = out[max(0, by-12):by+13, max(0, bx-12):bx+13]
    blob_fg = float((patch > 0).mean())
    assert blob_fg < 0.35, f"Blob not removed enough (fg_frac={blob_fg:.3f})"

    # Check that we still have some vessels detected overall
    fg_frac = float((out > 0).mean())
    assert fg_frac > 0.003, f"Too few vessel pixels detected (fg_frac={fg_frac:.4f})"


def test_pipeline_e_verbose_runs(tmp_path, capsys):
    img, _ = _make_synth_fundus()
    p = tmp_path / "synth.png"
    assert cv.imwrite(str(p), img)

    out = vessel_segmentation(str(p), verbose=True)
    captured = capsys.readouterr().out
    assert out is not None
    assert "[E]" in captured or "[CC]" in captured
