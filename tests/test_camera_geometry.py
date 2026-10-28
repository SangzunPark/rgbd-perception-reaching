
import numpy as np
import pytest
import pybullet as p

from src.camera_geometry import (
    compute_intrinsics,
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
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


# Test 5: optical_to_opengl
def test_optical_to_opengl():
    point_optical = np.array([0.10, -0.05, 0.95])

    point_opengl = optical_to_opengl(point_optical)

    expected = np.array([0.10, 0.05, -0.95])

    np.testing.assert_allclose(
        point_opengl, expected, atol=1e-9
    )


# Test 6: opengl_to_world
def test_opengl_to_world():
    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 1.0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
    )

    point_opengl = np.array([0.10, 0.05, -0.95])

    point_world = opengl_to_world(point_opengl, view)

    expected = np.array([0.10, 0.05, 0.05])

    np.testing.assert_allclose(
        point_world, expected, atol=1e-6
    )


# Test 7: world_camera_roundtrip
def test_world_camera_roundtrip():
    """World -> OpenGL -> World for a tilted camera."""

    view = p.computeViewMatrix(
        cameraEyePosition=[0.3, -0.2, 1.2],
        cameraTargetPosition=[0.05, 0.1, 0.0],
        cameraUpVector=[0, 1, 0],
    )

    V = np.asarray(view).reshape(4, 4, order="F")

    original_world = np.array([0.10, 0.05, 0.05])

    # World -> OpenGL
    point_h = np.append(original_world, 1.0)
    point_gl = (V @ point_h)[:3]

    # OpenGL -> World
    recovered_world = opengl_to_world(point_gl, view)

    np.testing.assert_allclose(
        recovered_world,
        original_world,
        atol=1e-6,
    )

# Test 8: full_geometry_roundtrip
def test_full_geometry_roundtrip():
    """Known World point -> image -> reconstructed World."""

    K = compute_intrinsics(320, 240, 60)

    view = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 1.0],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
    )

    V = np.asarray(view).reshape(4, 4, order="F")

    gt_top = np.array([0.10, 0.05, 0.05])

    # World -> OpenGL
    p_gl = (V @ np.append(gt_top, 1.0))[:3]

    # OpenGL -> Optical
    p_opt = np.array([p_gl[0], -p_gl[1], -p_gl[2]])

    # Optical 3D -> continuous image pixel coordinates
    z = p_opt[2]
    u = K[0, 0] * p_opt[0] / z + K[0, 2]
    v = K[1, 1] * p_opt[1] / z + K[1, 2]

    # Image -> Optical -> OpenGL -> World
    recovered_opt = pixel_to_camera(u, v, z, K)
    recovered_gl = optical_to_opengl(recovered_opt)
    recovered_world = opengl_to_world(recovered_gl, view)

    np.testing.assert_allclose(
        recovered_world,
        gt_top,
        atol=1e-6,
    )
