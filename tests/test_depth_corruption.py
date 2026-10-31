
import numpy as np
import pytest

from src.depth_corruption import (
    add_gaussian_depth_noise,
    apply_random_depth_dropout,
    apply_depth_patch_dropout,
)


def test_zero_gaussian_noise():
    depth = np.ones((5, 5), dtype=float)

    rng = np.random.default_rng(42)

    corrupted = add_gaussian_depth_noise(
        depth, sigma_m=0.0, rng=rng
    )

    np.testing.assert_array_equal(corrupted, depth)


def test_gaussian_noise_reproducibility():
    depth = np.ones((10, 10), dtype=float)

    result_1 = add_gaussian_depth_noise(
        depth, 0.003, np.random.default_rng(42)
    )

    result_2 = add_gaussian_depth_noise(
        depth, 0.003, np.random.default_rng(42)
    )

    np.testing.assert_array_equal(result_1, result_2)
    assert not np.array_equal(result_1, depth)


def test_random_dropout_all():
    depth = np.ones((5, 5), dtype=float)

    corrupted = apply_random_depth_dropout(
        depth, 1.0, np.random.default_rng(42)
    )

    assert np.all(np.isnan(corrupted))


def test_random_dropout_none():
    depth = np.ones((5, 5), dtype=float)

    corrupted = apply_random_depth_dropout(
        depth, 0.0, np.random.default_rng(42)
    )

    np.testing.assert_array_equal(corrupted, depth)


def test_patch_dropout():
    depth = np.ones((5, 5), dtype=float)

    corrupted = apply_depth_patch_dropout(
        depth, center_u=2, center_v=2, radius=1
    )

    assert np.count_nonzero(np.isnan(corrupted)) == 9
    assert np.isfinite(corrupted[0, 0])


def test_original_depth_unchanged():
    depth = np.ones((5, 5), dtype=float)
    original = depth.copy()

    add_gaussian_depth_noise(
        depth, 0.003, np.random.default_rng(42)
    )

    apply_random_depth_dropout(
        depth, 0.5, np.random.default_rng(42)
    )

    apply_depth_patch_dropout(
        depth, 2, 2, 1
    )

    np.testing.assert_array_equal(depth, original)


def test_invalid_parameters():
    depth = np.ones((5, 5), dtype=float)

    with pytest.raises(ValueError):
        add_gaussian_depth_noise(
            depth, -0.001, np.random.default_rng(42)
        )

    with pytest.raises(ValueError):
        apply_random_depth_dropout(
            depth, 1.5, np.random.default_rng(42)
        )

    with pytest.raises(ValueError):
        apply_depth_patch_dropout(
            depth, 2, 2, -1
        )
