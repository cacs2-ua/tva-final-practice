from __future__ import annotations

import cv2 as cv

from fov_mask import compute_fov_mask
from vessel_morph_pipeline import segment_vessels_morphology, VesselMorphParams


def vessel_segmentation(input_image: str, *, verbose: bool = False, return_debug: bool = False):
    """
    Pipeline D wrapper: FOV mask -> morphology vessel segmentation.
    Returns uint8 mask (0/255).
    """
    img = cv.imread(str(input_image), cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_image}")

    # 1) FOV mask
    fov_mask = compute_fov_mask(
        img,
        blur_kind="median",
        blur_ksize=7,
        close_ksize=None,
        open_ksize=None,
        do_hole_fill=True,
        do_convex_hull=True,
        ellipse_mode="auto",
        ellipse_scale=1.01,
        return_debug=False,
    )

    # 2) Pipeline D params
    params = VesselMorphParams(
        verbose=verbose,
        use_clahe=True,
        clahe_clip_limit=2.5,
        clahe_tile=8,
        post_close_ksize=3,
        post_open_ksize=3,
        cc_min_area=25,
        cc_min_elongation=2.2,
        cc_max_extent=0.45,
        cc_max_area_frac=0.15,
    )

    out = segment_vessels_morphology(img, fov_mask, params=params, return_debug=return_debug)
    return out
