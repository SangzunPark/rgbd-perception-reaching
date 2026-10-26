
import numpy as np
import pytest

from src.camera_geometry import (
    compute_intrinsics,
    pixel_to_camera,
)


# Test 1: Camera intrinsics
def test_compute_intrinsics():
    K = compute_intrinsics(320, 240, 60)

    assert K.shape == (3, 3)

    np.testing.assert_allclose(
        K[0, 0], 207.8460969, atol=1e-6
    )

    np.testing.assert_allclose(
        K[1, 1], 207.8460969, atol=1e-6
    )


# Test 2: Image center
def test_center_pixel():
    K = compute_intrinsics(320, 240, 60)

    point = pixel_to_camera(160, 120, 0.8, K)

    np.testing.assert_allclose(
        point, [0, 0, 0.8], atol=1e-6
    )


# Test 3: Off-center pixel
def test_off_center_pixel():
    K = compute_intrinsics(320, 240, 60)

    point = pixel_to_camera(180, 130, 0.8, K)

    np.testing.assert_allclose(
        point,
        [0.07698004, 0.03849002, 0.8],
        atol=1e-6,
    )


# Test 4: Invalid depth
def test_invalid_depth():
    K = compute_intrinsics(320, 240, 60)

    for invalid_depth in [0, -1, np.nan]:
        with pytest.raises(ValueError):
            pixel_to_camera(160, 120, invalid_depth, K)
