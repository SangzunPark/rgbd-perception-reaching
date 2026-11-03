
"""Tests for Panda FK and position-only IK."""

import numpy as np
import pybullet as p
import pytest

from src.robot_scene import (
    create_robot_scene,
    close_robot_scene,
)
from src.robot_frames import estimate_target_frames
from src.robot_kinematics import (
    get_ee_world_position,
    get_arm_joint_angles,
    forward_kinematics,
    solve_inverse_kinematics,
)


@pytest.fixture
def scene():
    s = create_robot_scene()
    try:
        yield s
    finally:
        close_robot_scene(s)


def test_fk_at_initial_pose(scene):
    angles = get_arm_joint_angles(scene)
    expected = get_ee_world_position(scene)

    actual = forward_kinematics(scene, angles)

    assert np.allclose(
        actual, expected, atol=1e-6
    )


def test_fk_restores_original_joint_state(scene):
    original = get_arm_joint_angles(scene)

    altered = original.copy()
    altered[0] += 0.2

    forward_kinematics(scene, altered)

    after = get_arm_joint_angles(scene)

    assert np.allclose(
        after, original, atol=1e-8
    )


def test_ik_for_known_reachable_ee_pose(scene):
    # The current robot pose is known to be reachable.
    target = get_ee_world_position(scene)

    result = solve_inverse_kinematics(
        scene, target
    )

    assert result["converged"]
    assert result["ik_error_m"] <= 0.010
    assert len(result["joint_angles_rad"]) == 7

    fk = forward_kinematics(
        scene, result["joint_angles_rad"]
    )

    assert np.linalg.norm(fk - target) <= 0.010


def test_perception_driven_ik(scene):
    frames = estimate_target_frames(scene)

    # This is the estimated approach point, not GT.
    target = frames["approach_world_m"]

    result = solve_inverse_kinematics(
        scene, target
    )

    assert result["converged"], (
        f"IK failed with error "
        f"{result['ik_error_m'] * 1000:.3f} mm"
    )

    assert np.allclose(
        result["target_world_m"], target
    )

    # Every returned arm angle must respect URDF limits
    for index, angle in zip(
        scene.arm_joint_indices,
        result["joint_angles_rad"],
    ):
        info = p.getJointInfo(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )

        assert info[8] - 1e-6 <= angle <= info[9] + 1e-6


def test_invalid_kinematics_inputs(scene):
    with pytest.raises(ValueError):
        forward_kinematics(
            scene, [0.0, 0.0]
        )

    with pytest.raises(ValueError):
        solve_inverse_kinematics(
            scene, [np.nan, 0.0, 0.0]
        )

    with pytest.raises(ValueError):
        solve_inverse_kinematics(
            scene, [0.1, 0.0, 0.2],
            tolerance_m=-0.01,
        )
