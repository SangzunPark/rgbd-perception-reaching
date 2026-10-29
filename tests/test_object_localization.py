import numpy as np
import pytest
import pybullet as p

from src.object_localization import (
    estimate_target,
    local_median_depth,
)
from src.camera_geometry import compute_intrinsics


def make_test_view():
    return p.computeViewMatrix(
        cameraEyePosition=[0, 0, 1.0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
    )


def make_simple_rgb_depth():
    """
    5x5 synthetic scene:
    - center 3x3 block is red object
    - depth = 1.0 everywhere on object
    """
    rgb = np.zeros((5, 5, 3), dtype=float)
    depth = np.ones((5, 5), dtype=float)

    # red object: 3x3 centered block
    rgb[1:4, 1:4, 0] = 0.9
    rgb[1:4, 1:4, 1] = 0.1
    rgb[1:4, 1:4, 2] = 0.1

    return rgb, depth


def test_local_median_depth_basic():
    depth = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    mask = np.ones((3, 3), dtype=bool)

    z = local_median_depth(depth, mask, u=1, v=1, radius=1)

    assert z == pytest.approx(1.0)


def test_estimate_target_single():
    rgb, depth = make_simple_rgb_depth()
    K = compute_intrinsics(5, 5, 90)
    view = make_test_view()

    p_world = estimate_target(
        rgb,
        depth,
        K,
        view,
        depth_method="single",
    )

    expected = np.array([-0.2, 0.2, 0.0])
    np.testing.assert_allclose(p_world, expected, atol=1e-6)


def test_estimate_target_median():
    rgb, depth = make_simple_rgb_depth()
    K = compute_intrinsics(5, 5, 90)
    view = make_test_view()

    p_world = estimate_target(
        rgb,
        depth,
        K,
        view,
        depth_method="median",
        radius=1,
    )

    expected = np.array([-0.2, 0.2, 0.0])
    np.testing.assert_allclose(p_world, expected, atol=1e-6)


def test_estimate_target_shape_mismatch():
    rgb = np.zeros((5, 5, 3), dtype=float)
    depth = np.ones((4, 5), dtype=float)
    K = compute_intrinsics(5, 4, 90)
    view = make_test_view()

    with pytest.raises(ValueError):
        estimate_target(rgb, depth, K, view)


def test_estimate_target_invalid_depth_method():
    rgb, depth = make_simple_rgb_depth()
    K = compute_intrinsics(5, 5, 90)
    view = make_test_view()

    with pytest.raises(ValueError):
        estimate_target(
            rgb,
            depth,
            K,
            view,
            depth_method="unknown",
        )


def test_estimate_target_median_recovers_nan_center():
    rgb, depth = make_simple_rgb_depth()
    K = compute_intrinsics(5, 5, 90)
    view = make_test_view()

    # center pixel becomes invalid
    depth[2, 2] = np.nan

    # single should fail
    with pytest.raises(ValueError):
        estimate_target(
            rgb,
            depth,
            K,
            view,
            depth_method="single",
        )

    # median should recover from surrounding valid object depths
    p_world = estimate_target(
        rgb,
        depth,
        K,
        view,
        depth_method="median",
        radius=1,
    )

    expected = np.array([-0.2, 0.2, 0.0])
    np.testing.assert_allclose(p_world, expected, atol=1e-6)