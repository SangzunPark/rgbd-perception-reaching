
"""Tests for simulated Panda joint position control."""

import numpy as np
import pytest

from src.robot_scene import create_robot_scene, close_robot_scene
from src.robot_kinematics import (
    forward_kinematics,
    get_arm_joint_angles,
    get_ee_world_position,
)
from src.reaching_controller import (
    make_calibration_joint_target,
    validate_joint_target,
    run_joint_position_control,
    save_controller_results,
)


@pytest.fixture
def scene():
    s = create_robot_scene()
    try:
        yield s
    finally:
        close_robot_scene(s)


def test_calibration_target_is_reachable(scene):
    goal_q = make_calibration_joint_target(scene)

    assert goal_q.shape == (7,)

    initial = get_ee_world_position(scene)
    goal_world = forward_kinematics(scene, goal_q)

    # The calibration should produce a real change in EE position
    assert np.linalg.norm(goal_world - initial) > 0.01


def test_position_control_reduces_ee_error(scene):
    goal_q = make_calibration_joint_target(scene)
    goal_world = forward_kinematics(scene, goal_q)

    result = run_joint_position_control(
        scene,
        goal_q,
        goal_world,
    )

    initial_error = np.linalg.norm(
        result["initial_ee_world_m"] - goal_world
    )
    final_error = np.linalg.norm(
        result["final_ee_world_m"] - goal_world
    )

    assert final_error < initial_error
    assert result["steps"] >= 240
    assert len(result["history"]) > 2

    # Physical state should actually change, not merely an FK result
    assert not np.allclose(
        get_arm_joint_angles(scene),
        result["history"][0]["step"] * np.ones(7),
    )


def test_controller_history_and_artifacts(scene, tmp_path):
    goal_q = make_calibration_joint_target(scene)
    goal_world = forward_kinematics(scene, goal_q)

    result = run_joint_position_control(
        scene, goal_q, goal_world
    )

    history = result["history"]

    assert history[0]["step"] == 0
    assert history[-1]["step"] == result["steps"]

    times = [r["time_s"] for r in history]
    assert all(b > a for a, b in zip(times, times[1:]))

    paths = save_controller_results(result, tmp_path)

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)


def test_invalid_controller_inputs(scene):
    initial = get_arm_joint_angles(scene)
    goal_world = get_ee_world_position(scene)

    with pytest.raises(ValueError):
        validate_joint_target(scene, [0.0] * 6)

    with pytest.raises(ValueError):
        validate_joint_target(
            scene, np.full(7, np.nan)
        )

    with pytest.raises(ValueError):
        run_joint_position_control(
            scene,
            initial,
            [np.nan, 0.0, 0.0],
        )

    with pytest.raises(ValueError):
        run_joint_position_control(
            scene,
            initial,
            goal_world,
            max_steps=10,
            ramp_steps=240,
        )
