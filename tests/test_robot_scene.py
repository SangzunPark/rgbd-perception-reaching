
"""Day 9-1: robot setup and camera invariants."""

import numpy as np
import pybullet as p
import pytest

from src.object_perception import detect_red_object
from src.robot_scene import (
    ARM_NAMES,
    EE_LINK_NAME,
    HOME_ARM_Q,
    ROBOT_BASE,
    capture_sensor_rgbd,
    close_robot_scene,
    create_robot_scene,
)


@pytest.fixture
def scene():
    robot_scene = create_robot_scene()
    try:
        yield robot_scene
    finally:
        close_robot_scene(robot_scene)


def test_robot_joints_and_ee(scene):
    assert len(scene.arm_joint_indices) == 7
    assert len(scene.finger_joint_indices) == 2

    names = [
        p.getJointInfo(
            scene.robot_id, i,
            physicsClientId=scene.client_id,
        )[1].decode()
        for i in scene.arm_joint_indices
    ]

    assert tuple(names) == ARM_NAMES

    ee_name = p.getJointInfo(
        scene.robot_id,
        scene.ee_link_index,
        physicsClientId=scene.client_id,
    )[12].decode()

    assert ee_name == EE_LINK_NAME



def test_robot_base_and_initial_joint_angles(scene):
    # 1. Get base inertial-frame pose in World
    base_position, base_orientation = (
        p.getBasePositionAndOrientation(
            scene.robot_id,
            physicsClientId=scene.client_id,
        )
    )

    # 2. Get local inertial-frame offset from URDF
    dynamics = p.getDynamicsInfo(
        scene.robot_id,
        -1,  # -1 means the base link
        physicsClientId=scene.client_id,
    )

    local_inertial_pos = dynamics[3]
    local_inertial_orn = dynamics[4]

    # 3. Convert inertial frame back to base link frame
    inverse_pos, inverse_orn = p.invertTransform(
        local_inertial_pos,
        local_inertial_orn,
    )

    link_position, _ = p.multiplyTransforms(
        base_position,
        base_orientation,
        inverse_pos,
        inverse_orn,
    )

    # 4. Verify the actual base link position
    assert np.allclose(
        link_position, ROBOT_BASE, atol=1e-6
    )

    # 5. Verify the initial joint angles
    angles = [
        p.getJointState(
            scene.robot_id, i,
            physicsClientId=scene.client_id,
        )[0]
        for i in scene.arm_joint_indices
    ]

    assert np.allclose(
        angles, HOME_ARM_Q, atol=1e-6
    )

    # 6. Verify finite end-effector position
    ee_position = p.getLinkState(
        scene.robot_id,
        scene.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=scene.client_id,
    )[4]

    assert np.isfinite(ee_position).all()



def test_cube_is_on_floor(scene):
    cube_position, _ = p.getBasePositionAndOrientation(
        scene.cube_id,
        physicsClientId=scene.client_id,
    )

    assert np.allclose(
        cube_position[:2],
        [0.10, 0.05],
        atol=0.002,
    )
    assert cube_position[2] == pytest.approx(
        0.025, abs=0.002
    )


def test_existing_sensor_conventions_and_visibility(scene):
    rgb, depth_m, view, K = capture_sensor_rgbd(
        scene
    )

    assert rgb.shape == (240, 320, 3)
    assert depth_m.shape == (240, 320)
    assert view.shape == (16,)
    assert K.shape == (3, 3)

    assert rgb.min() >= 0
    assert rgb.max() <= 1
    assert np.isfinite(depth_m).all()

    red_mask = detect_red_object(rgb)
    assert red_mask.any()

    assert np.median(
        depth_m[red_mask]
    ) == pytest.approx(0.95, abs=0.03)
