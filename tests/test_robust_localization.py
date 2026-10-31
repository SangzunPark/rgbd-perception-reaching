
import numpy as np
import pytest

from src.robust_localization import select_robust_surface


def test_clean_surface():
    mask = np.ones((7, 7), dtype=bool)
    depth = np.full((7, 7), 0.95)

    u, v, z, info = select_robust_surface(mask, depth)

    assert u == pytest.approx(3.0)
    assert v == pytest.approx(3.0)
    assert z == pytest.approx(0.95)
    assert info["inlier_pixels"] == 49


def test_low_depth_outlier_rejected():
    mask = np.ones((7, 7), dtype=bool)
    depth = np.full((7, 7), 0.95)

    # A single 20 mm negative outlier
    depth[0, 0] = 0.93

    _, _, z, info = select_robust_surface(mask, depth)

    assert z == pytest.approx(0.95)
    assert info["inlier_pixels"] == 48


def test_center_missing_depth():
    mask = np.ones((9, 9), dtype=bool)
    depth = np.full((9, 9), 0.95)

    # Remove the 3x3 center region
    depth[3:6, 3:6] = np.nan

    u, v, z, info = select_robust_surface(mask, depth)

    # The center pixel itself is NaN, but the
    # estimated geometric center remains valid.
    assert np.isnan(depth[4, 4])
    assert u == pytest.approx(4.0)
    assert v == pytest.approx(4.0)
    assert z == pytest.approx(0.95)
    assert info["inlier_pixels"] == 72


def test_one_sided_depth_support_rejected():
    mask = np.ones((11, 11), dtype=bool)
    depth = np.full((11, 11), np.nan)

    # Only one column remains: 11 valid pixels
    depth[:, 0] = 0.975

    with pytest.raises(
        ValueError,
        match="spatial surface coverage",
    ):
        select_robust_surface(mask, depth)


def test_all_depth_missing():
    mask = np.ones((7, 7), dtype=bool)
    depth = np.full((7, 7), np.nan)

    with pytest.raises(ValueError):
        select_robust_surface(mask, depth)


def test_gaussian_noise_stability():
    mask = np.ones((21, 21), dtype=bool)

    rng = np.random.default_rng(42)
    depth = 0.95 + rng.normal(
        0.0, 0.003, size=mask.shape
    )

    u, v, z, _ = select_robust_surface(mask, depth)

    assert abs(z - 0.95) < 0.001
    assert abs(u - 10.0) < 0.7
    assert abs(v - 10.0) < 0.7


def test_shape_mismatch():
    mask = np.ones((7, 7), dtype=bool)
    depth = np.ones((5, 5), dtype=float)

    with pytest.raises(ValueError):
        select_robust_surface(mask, depth)
