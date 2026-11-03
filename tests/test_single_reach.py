
"""Tests for the single-scene reaching demonstration and audit."""

import numpy as np
import pytest

from src.robot_scene import (
    create_robot_scene,
    close_robot_scene,
)
from src.run_perception_reach import run_perception_reach
from src.run_single_reach import (
    analyze_single_reach,
    save_demo_analysis,
)


def example_result():
    """Artificial geometry case with known error vectors."""
    return {
        "status": "control_converged",
        "command_issued": True,
        "estimated_top_world_m": np.array([0.100, 0.0, 0.05]),
        "true_top_world_m_before": np.array([0.102, 0.0, 0.05]),
        "true_top_world_m_after": np.array([0.102, 0.0, 0.05]),
        "commanded_approach_world_m": np.array([0.100, 0.0, 0.15]),
        "final_ee_world_m": np.array([0.110, 0.0, 0.15]),
        "ik_fk_world_m": np.array([0.100, 0.0, 0.15]),
        "approach_height_m": 0.10,
        "ik_error_mm": 0.0,
        "errors_mm": {
            "perception_error_mm": 2.0,
            "control_error_mm": 10.0,
            "reaching_error_mm": 8.0,
        },
        "simulated_steps": 10,
        "simulated_time_s": 10 / 240,
        "history": [
            {
                "step": 0,
                "time_s": 0.0,
                "ee_x_m": 0.0,
                "ee_y_m": 0.0,
                "ee_z_m": 0.30,
                "ee_error_mm": 180.278,
            },
            {
                "step": 10,
                "time_s": 10 / 240,
                "ee_x_m": 0.110,
                "ee_y_m": 0.0,
                "ee_z_m": 0.15,
                "ee_error_mm": 10.0,
            },
        ],
    }


def test_vector_error_identity():
    analysis = analyze_single_reach(example_result())

    assert analysis["perception_error_mm"] == pytest.approx(2.0)
    assert analysis["control_error_mm"] == pytest.approx(10.0)
    assert analysis["reaching_error_mm"] == pytest.approx(8.0)

    assert analysis["vector_closure_mm"] < 1e-8


def test_inconsistent_metrics_are_rejected():
    case = example_result()

    case["errors_mm"]["control_error_mm"] = 25.0

    with pytest.raises(
        ValueError, match="Saved metric mismatch"
    ):
        analyze_single_reach(case)


@pytest.fixture(scope="module")
def completed_reach():
    scene = create_robot_scene()

    try:
        yield run_perception_reach(scene)
    finally:
        close_robot_scene(scene)


def test_actual_single_scene_reaching(completed_reach):
    assert completed_reach["status"] == "control_converged"
    assert completed_reach["command_issued"] is True

    analysis = analyze_single_reach(completed_reach)

    assert analysis["reaching_error_mm"] < 10.0
    assert analysis["cube_displacement_mm"] < 1.0
    assert analysis["vector_closure_mm"] < 1e-8


def test_demo_plots_and_report(completed_reach, tmp_path):
    out = tmp_path / "outputs"

    report = (
        tmp_path
        / "reports"
        / "perception_driven_reaching_integration.md"
    )

    trajectory, error_plot, report_path, _ = (
        save_demo_analysis(
            completed_reach,
            output_dir=out,
            report_path=report,
        )
    )

    for path in (trajectory, error_plot, report_path):
        assert path.is_file()
        assert path.stat().st_size > 0

    text = report_path.read_text(encoding="utf-8")

    assert "Perception error" in text
    assert "Final reaching error" in text
    assert "not for target commands" in text
