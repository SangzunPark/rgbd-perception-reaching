
"""Perception-driven reaching: frozen Method R -> IK -> Panda motors."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from src.robot_scene import (
    CUBE_HALF, OUTPUT_DIR, close_robot_scene, create_robot_scene,
    render_overview,
)
from src.robot_frames import define_approach_point, estimate_target_frames
from src.robot_kinematics import (
    get_arm_joint_angles, get_ee_world_position, solve_inverse_kinematics,
)
from src.reaching_controller import run_joint_position_control


def cube_top_world_for_evaluation(scene):
    """Read simulator GT for evaluation ONLY, never for a motor target."""
    pos, orn = p.getBasePositionAndOrientation(
        scene.cube_id, physicsClientId=scene.client_id
    )
    rotation = np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return np.asarray(pos) + rotation @ np.array([0.0, 0.0, CUBE_HALF])


def evaluate_errors_mm(estimated_top, true_top, actual_ee, height_m):
    """Separate perception, motor-control, and final reaching errors."""
    estimated_top = np.asarray(estimated_top, dtype=float)
    true_top = np.asarray(true_top, dtype=float)
    actual_ee = np.asarray(actual_ee, dtype=float)
    estimated_approach = define_approach_point(estimated_top, height_m)
    true_approach = define_approach_point(true_top, height_m)
    return {
        "perception_error_mm": float(1000 * np.linalg.norm(estimated_top - true_top)),
        "control_error_mm": float(1000 * np.linalg.norm(actual_ee - estimated_approach)),
        "reaching_error_mm": float(1000 * np.linalg.norm(actual_ee - true_approach)),
    }


def run_perception_reach(scene, approach_height_m=0.10):
    """Run one reach. GT is never used to compute the motor command."""
    initial_q = get_arm_joint_angles(scene)
    initial_ee = get_ee_world_position(scene)
    result = {
        "status": "not_started",
        "command_issued": False,
        "initial_joint_angles_rad": initial_q,
        "initial_ee_world_m": initial_ee,
        "history": [],
    }

    # (1) Frozen perception; rejection means no movement.
    try:
        frames = estimate_target_frames(scene, approach_height_m)
    except ValueError as exc:
        result.update(status="perception_rejected", reason=str(exc))
        return result

    result.update(
        estimated_top_world_m=frames["target_world_m"],
        commanded_approach_world_m=frames["approach_world_m"],
        approach_height_m=approach_height_m,
        perception_diagnostics=frames["diagnostics"],
    )

    # (2) IK uses the estimated World-frame approach point.
    try:
        ik = solve_inverse_kinematics(scene, frames["approach_world_m"])
    except (ValueError, RuntimeError) as exc:
        result.update(status="ik_failed", reason=str(exc))
        return result

    result.update(
        ik_joint_angles_rad=ik["joint_angles_rad"],
        ik_fk_world_m=ik["fk_world_m"],
        ik_error_mm=float(1000 * ik["ik_error_m"]),
    )
    if not ik["converged"]:
        result.update(status="ik_failed", reason="IK tolerance not met")
        return result

    # Log GT only for later evaluation. Neither IK nor controller sees it.
    true_top_before = cube_top_world_for_evaluation(scene)

    # (3) Execute joint-space motor control.
    result["command_issued"] = True
    motion = run_joint_position_control(
        scene, ik["joint_angles_rad"], frames["approach_world_m"]
    )

    # (4) GT comparison ONLY after executing the motor command.
    true_top_after = cube_top_world_for_evaluation(scene)
    result.update(
        status="control_converged" if motion["converged"] else "control_failed",
        final_ee_world_m=motion["final_ee_world_m"],
        final_joint_angles_rad=motion["final_joint_angles_rad"],
        simulated_steps=motion["steps"],
        simulated_time_s=motion["simulated_time_s"],
        true_top_world_m_before=true_top_before,
        true_top_world_m_after=true_top_after,
        cube_displacement_mm=float(1000 * np.linalg.norm(
            true_top_after - true_top_before
        )),
        errors_mm=evaluate_errors_mm(
            frames["target_world_m"], true_top_before,
            motion["final_ee_world_m"], approach_height_m,
        ),
        history=motion["history"],
    )
    return result


def json_ready(value):
    """Convert NumPy-containing output to JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_reach_result(result, output_dir=OUTPUT_DIR):
    """Persist summary and, if a motor command ran, the trace CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "perception_reach_summary.json"
    summary = {k: v for k, v in result.items() if k != "history"}
    summary_path.write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )

    trace_path = output_dir / "perception_reach_trajectory.csv"
    if result["history"]:
        with trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result["history"][0].keys())
            writer.writeheader()
            writer.writerows(result["history"])
    elif trace_path.exists():
        trace_path.unlink()  # Avoid stale success traces after rejection

    return summary_path, trace_path if result["history"] else None


def main():
    scene = create_robot_scene()
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.imsave(
            OUTPUT_DIR / "perception_reach_before.png", render_overview(scene)
        )
        result = run_perception_reach(scene)
        plt.imsave(
            OUTPUT_DIR / "perception_reach_after.png", render_overview(scene)
        )
        summary_path, trace_path = save_reach_result(result)

        print("Status:", result["status"])
        print("Motor command issued:", result["command_issued"])
        if "reason" in result:
            print("Reason:", result["reason"])
        if "estimated_top_world_m" in result:
            print("Method R top (World, m):", np.round(result["estimated_top_world_m"], 6))
            print("Commanded approach (World, m):", np.round(result["commanded_approach_world_m"], 6))
        if "ik_error_mm" in result:
            print("IK-FK error (mm):", round(result["ik_error_mm"], 3))
        if result["command_issued"]:
            print("Final EE (World, m):", np.round(result["final_ee_world_m"], 6))
            print("Controller simulated steps:", result["simulated_steps"])
            for name, value in result["errors_mm"].items():
                print(f"{name}: {value:.3f}")
            print("Cube displacement (mm):", round(result["cube_displacement_mm"], 3))
        print("Saved:", summary_path)
        if trace_path is not None:
            print("Saved:", trace_path)
    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
