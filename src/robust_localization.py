
"""Robust RGB-D surface localization with confidence checks."""

import numpy as np

from src.camera_geometry import (
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
)
from src.object_perception import detect_red_object


def select_robust_surface(
    object_mask,
    depth_m,
    min_valid_pixels=10,
    min_span_fraction=0.5,
    base_tolerance_m=0.001,
    mad_multiplier=3.0,
):
    """Estimate a flat front surface from noisy or missing depth."""

    mask = np.asarray(object_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)

    if mask.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and depth shapes must match")

    if not np.any(mask):
        raise ValueError("Object mask is empty")

    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be positive")

    if not 0 < min_span_fraction <= 1:
        raise ValueError("Invalid min_span_fraction")

    if base_tolerance_m <= 0 or mad_multiplier <= 0:
        raise ValueError("Invalid robust filtering parameters")

    # 1. Keep valid depth measurements within RGB mask
    valid = (
        mask
        & np.isfinite(depth)
        & (depth > 0)
    )

    valid_count = int(np.count_nonzero(valid))

    if valid_count < min_valid_pixels:
        raise ValueError("Insufficient valid object depth")

    values = depth[valid]

    # 2. Robust depth center and noise scale
    median_depth = float(np.median(values))

    mad = float(np.median(
        np.abs(values - median_depth)
    ))

    estimated_sigma = 1.4826 * mad

    tolerance = max(
        base_tolerance_m,
        mad_multiplier * estimated_sigma,
    )

    # 3. Select depth inliers around the median
    inliers = (
        valid
        & (np.abs(depth - median_depth) <= tolerance)
    )

    inlier_count = int(np.count_nonzero(inliers))

    if inlier_count < min_valid_pixels:
        raise ValueError("Insufficient surface inliers")

    # 4. Measure spatial coverage of remaining surface
    object_rows, object_cols = np.where(mask)
    rows, cols = np.where(inliers)

    object_span_u = (
        object_cols.max() - object_cols.min() + 1
    )
    object_span_v = (
        object_rows.max() - object_rows.min() + 1
    )

    inlier_span_u = cols.max() - cols.min() + 1
    inlier_span_v = rows.max() - rows.min() + 1

    span_u_fraction = inlier_span_u / object_span_u
    span_v_fraction = inlier_span_v / object_span_v

    if (
        span_u_fraction < min_span_fraction
        or span_v_fraction < min_span_fraction
    ):
        raise ValueError("Insufficient spatial surface coverage")

    # 5. Continuous 2D centroid
    # The centroid does not need to have valid depth itself.
    u = float(cols.mean())
    v = float(rows.mean())

    # 6. Robust surface depth
    z = float(np.median(depth[inliers]))

    diagnostics = {
        "valid_pixels": valid_count,
        "inlier_pixels": inlier_count,
        "estimated_sigma_mm": estimated_sigma * 1000,
        "tolerance_mm": tolerance * 1000,
        "span_u_fraction": float(span_u_fraction),
        "span_v_fraction": float(span_v_fraction),
    }

    return u, v, z, diagnostics


def estimate_robust_target(
    rgb,
    depth_m,
    K,
    view_matrix,
    vertical_offset_px=1.0,
):
    """Estimate a target World position from RGB-D data."""

    if depth_m.shape != rgb.shape[:2]:
        raise ValueError("RGB and depth shapes do not match")

    # RGB-only object detection
    object_mask = detect_red_object(rgb)

    # Robust depth-supported surface estimate
    u, v, z, diagnostics = select_robust_surface(
        object_mask,
        depth_m,
    )

    # +1 px is the experimentally observed TinyRenderer
    # alignment correction, not a universal camera rule.
    p_optical = pixel_to_camera(
        u,
        v + vertical_offset_px,
        z,
        K,
    )

    p_gl = optical_to_opengl(p_optical)
    p_world = opengl_to_world(p_gl, view_matrix)

    return p_world, diagnostics
