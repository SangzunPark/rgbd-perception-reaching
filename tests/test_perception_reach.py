
"""Step 9-5: motor integration, failure gates, and metric separation."""

import json

import numpy as np
import pytest

import src.run_perception_reach as integration
from src.robot_scene import create_robot_scene, close_robot_scene
from src.robot_kinematics import get_arm_joint_angles, get_ee_world_position


@pytest.fixture
def scene():
    s = create_robot_scene()
    try:
        yield s
    finally:
        close_robot_scene(s)


def test_three_errors_are_not_scalar_additive():
    errors = integration.evaluate_errors_mm(
        estimated_top=[0.100, 0.0, 0.05],
        true_top=[0.102, 0.0, 0.05],
        actual_ee=[0.110, 0.0, 0.15],
        height_m=0.10,
    )
    assert errors["perception_error_mm"] == pytest.approx(2.0)
    assert errors["control_error_mm"] == pytest.approx(10.0)
    assert errors["reaching_error_mm"] == pytest.approx(8.0)


def test_perception_rejection_prevents_motion(scene, monkeypatch):
    initial_q = get_arm_joint_angles(scene)
    initial_ee = get_ee_world_position(scene)

    def reject(*args, **kwargs):
        raise ValueError("Insufficient surface inliers")

    monkeypatch.setattr(integration, "estimate_target_frames", reject)

    def must_not_execute(*args, **kwargs):
        raise AssertionError("Motor/IK must not be used after rejection")

    monkeypatch.setattr(integration, "solve_inverse_kinematics", must_not_execute)
    monkeypatch.setattr(integration, "run_joint_position_control", must_not_execute)

    result = integration.run_perception_reach(scene)
    assert result["status"] == "perception_rejected"
    assert not result["command_issued"]
    assert np.allclose(initial_q, get_arm_joint_angles(scene))
    assert np.allclose(initial_ee, get_ee_world_position(scene))


def test_ik_failure_prevents_motor_command(scene, monkeypatch):
    def failed_ik(*args, **kwargs):
        return {
            "converged": False,
            "joint_angles_rad": get_arm_joint_angles(scene),
            "fk_world_m": get_ee_world_position(scene),
            "ik_error_m": 0.1,
        }

    def must_not_drive(*args, **kwargs):
        raise AssertionError("Controller called after IK failure")

    monkeypatch.setattr(integration, "solve_inverse_kinematics", failed_ik)
    monkeypatch.setattr(integration, "run_joint_position_control", must_not_drive)

    result = integration.run_perception_reach(scene)
    assert result["status"] == "ik_failed"
    assert not result["command_issued"]


def test_actual_rgbd_to_motor_pipeline(scene, tmp_path):
    result = integration.run_perception_reach(scene)

    assert result["command_issued"]
    assert result["status"] in {"control_converged", "control_failed"}
    assert result["simulated_steps"] > 0
    assert len(result["history"]) > 1

    # The commanded reference comes from Method R, not GT.
    estimated = np.asarray(result["estimated_top_world_m"])
    commanded = np.asarray(result["commanded_approach_world_m"])
    assert np.allclose(commanded - estimated, [0, 0, 0.10])

    errors = result["errors_mm"]
    assert np.isfinite(list(errors.values())).all()
    assert errors["control_error_mm"] == pytest.approx(
        1000 * np.linalg.norm(result["final_ee_world_m"] - commanded)
    )

    json_path, csv_path = integration.save_reach_result(result, tmp_path)
    assert json_path.is_file()
    assert csv_path.is_file()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["command_issued"] is True
    assert "history" not in saved
