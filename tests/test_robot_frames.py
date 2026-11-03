
"""Tests for World / Robot Base coordinate transformations."""

import numpy as np
import pybullet as p
import pytest

from src.robot_scene import (
    ROBOT_BASE,
    create_robot_scene,
    close_robot_scene,
)
from src.robot_frames import (
    make_transform,
    get_robot_base_transform,
    base_to_world,
    world_to_base,
    define_approach_point,
    estimate_target_frames,
)


@pytest.fixture
def scene():
    s = create_robot_scene()
    try:
        yield s
    finally:
        close_robot_scene(s)


def test_rotated_frame_transform():
    # Rotate the frame 90 degrees around Z
    quaternion = p.getQuaternionFromEuler(
        [0, 0, np.pi / 2]
    )

    T = make_transform([1, 2, 3], quaternion)

    world = base_to_world([1, 0, 0], T)

    assert np.allclose(world, [1, 3, 3], atol=1e-6)

    recovered = world_to_base(world, T)

    assert np.allclose(
        recovered, [1, 0, 0], atol=1e-6
    )


def test_actual_robot_base_frame(scene):
    T = get_robot_base_transform(scene)

    assert T.shape == (4, 4)
    assert np.allclose(
        T[:3, 3], ROBOT_BASE, atol=1e-6
    )
    assert np.allclose(
        T[:3, :3], np.eye(3), atol=1e-6
    )


def test_approach_target_coordinates(scene):
    T = get_robot_base_transform(scene)

    # Geometry example only, not a controller command
    example = np.array([0.10, 0.05, 0.05])

    approach = define_approach_point(
        example, height_m=0.10
    )

    assert np.allclose(
        approach, [0.10, 0.05, 0.15]
    )

    approach_base = world_to_base(approach, T)

    assert np.allclose(
        approach_base, [0.60, 0.05, 0.15],
        atol=1e-6,
    )


def test_method_r_target_frame_roundtrip(scene):
    result = estimate_target_frames(scene)

    T = result["T_world_from_base"]

    reconstructed = base_to_world(
        result["target_base_m"], T
    )

    assert np.allclose(
        reconstructed,
        result["target_world_m"],
        atol=1e-8,
    )

    # Sanity check against scene geometry, evaluation only
    expected = np.array([0.10, 0.05, 0.05])

    assert np.linalg.norm(
        result["target_world_m"] - expected
    ) < 0.015

    assert result["diagnostics"]["inlier_pixels"] >= 10


def test_invalid_approach_inputs():
    with pytest.raises(ValueError):
        define_approach_point(
            [0.1, 0.05, 0.05], height_m=-0.1
        )

    with pytest.raises(ValueError):
        define_approach_point(
            [np.nan, 0.05, 0.05]
        )
