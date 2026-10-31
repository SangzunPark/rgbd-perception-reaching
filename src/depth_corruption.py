
"""Controlled corruption of RGB-D depth measurements."""

import numpy as np


def validate_depth(depth_m):
    """Validate and return a 2D depth array."""

    depth = np.asarray(depth_m, dtype=float)

    if depth.ndim != 2:
        raise ValueError("Depth image must be 2D")

    return depth


def add_gaussian_depth_noise(depth_m, sigma_m, rng):
    """Add zero-mean Gaussian noise to valid depth pixels."""

    depth = validate_depth(depth_m)

    if not np.isfinite(sigma_m) or sigma_m < 0:
        raise ValueError("sigma_m must be finite and >= 0")

    # Copy: never modify original depth data
    corrupted = depth.copy()

    # Only valid, positive depth measurements
    valid = np.isfinite(corrupted) & (corrupted > 0)

    if sigma_m > 0:
        noise = rng.normal(
            loc=0.0,
            scale=sigma_m,
            size=np.count_nonzero(valid),
        )

        corrupted[valid] += noise

        # Non-positive depths are invalid measurements
        corrupted[valid & (corrupted <= 0)] = np.nan

    return corrupted


def apply_random_depth_dropout(depth_m, probability, rng):
    """Replace randomly selected valid depths with NaN."""

    depth = validate_depth(depth_m)

    if (
        not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("probability must be between 0 and 1")

    corrupted = depth.copy()

    valid = np.isfinite(corrupted) & (corrupted > 0)

    # Independent random draw for every pixel
    random_values = rng.random(size=depth.shape)

    drop_mask = valid & (random_values < probability)

    corrupted[drop_mask] = np.nan

    return corrupted


def apply_depth_patch_dropout(
    depth_m,
    center_u,
    center_v,
    radius,
):
    """Replace a square depth region with NaN."""

    depth = validate_depth(depth_m)

    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a non-negative integer")

    h, w = depth.shape

    if not (
        0 <= center_u < w
        and 0 <= center_v < h
    ):
        raise ValueError("Patch center outside image")

    corrupted = depth.copy()

    # Radius 1 -> 3x3 patch
    # Radius 2 -> 5x5 patch
    y0 = max(0, center_v - radius)
    y1 = min(h, center_v + radius + 1)

    x0 = max(0, center_u - radius)
    x1 = min(w, center_u + radius + 1)

    corrupted[y0:y1, x0:x1] = np.nan

    return corrupted
