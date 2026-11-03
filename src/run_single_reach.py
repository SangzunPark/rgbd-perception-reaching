
"""Document and visualize one perception-driven Panda reaching run.

Reuses the frozen perception method, IK, and joint controller.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.robot_scene import (
    ROOT, OUTPUT_DIR, close_robot_scene, create_robot_scene, render_overview,
)
from src.run_perception_reach import (
    run_perception_reach, save_reach_result,
)

REPORT_PATH = ROOT / "reports" / "perception_driven_reaching_integration.md"


def _point(result, key):
    """Return a finite World-coordinate vector."""
    value = np.asarray(result[key], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid 3D point: {key}")
    return value


def analyze_single_reach(result):
    """Audit final reaching errors and trajectory against raw coordinates."""
    if not result.get("command_issued"):
        raise ValueError("No reaching motion was commanded")

    estimated = _point(result, "estimated_top_world_m")
    gt_before = _point(result, "true_top_world_m_before")
    gt_after = _point(result, "true_top_world_m_after")
    commanded = _point(result, "commanded_approach_world_m")
    final_ee = _point(result, "final_ee_world_m")
    ik_fk = _point(result, "ik_fk_world_m")

    height = float(result["approach_height_m"])
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Invalid approach height")

    z_offset = np.array([0.0, 0.0, height])

    if not np.allclose(
        commanded, estimated + z_offset, atol=1e-9, rtol=0
    ):
        raise ValueError(
            "Approach target is inconsistent with perception"
        )

    gt_approach = gt_before + z_offset

    error_p = estimated - gt_before
    error_c = final_ee - commanded
    error_r = final_ee - gt_approach

    metrics = {
        "perception_error_mm": 1000.0 * float(np.linalg.norm(error_p)),
        "control_error_mm": 1000.0 * float(np.linalg.norm(error_c)),
        "reaching_error_mm": 1000.0 * float(np.linalg.norm(error_r)),
    }

    # Verify stored metrics against reconstructed errors.
    for name, measured in metrics.items():
        if not np.isclose(
            measured,
            float(result["errors_mm"][name]),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(f"Saved metric mismatch: {name}")

    ik_error_mm = 1000.0 * float(
        np.linalg.norm(ik_fk - commanded)
    )

    if not np.isclose(
        ik_error_mm,
        float(result["ik_error_mm"]),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "Saved IK error does not match FK coordinates"
        )

    history = result["history"]

    if not history or int(history[0]["step"]) != 0:
        raise ValueError("Missing start of trajectory")

    if int(history[-1]["step"]) != int(result["simulated_steps"]):
        raise ValueError(
            "Trajectory does not end at the final simulation step"
        )

    final_sample = np.array([
        history[-1]["ee_x_m"],
        history[-1]["ee_y_m"],
        history[-1]["ee_z_m"],
    ], dtype=float)

    if not np.allclose(
        final_sample, final_ee, atol=1e-8, rtol=0
    ):
        raise ValueError(
            "Final trajectory sample and EE pose disagree"
        )

    return {
        **metrics,
        "ik_error_mm": ik_error_mm,
        "vector_closure_mm": 1000.0 * float(
            np.linalg.norm(error_r - (error_p + error_c))
        ),
        "cube_displacement_mm": 1000.0 * float(
            np.linalg.norm(gt_after - gt_before)
        ),
        "gt_approach_world_m": gt_approach,
    }


def save_demo_analysis(
    result,
    output_dir=OUTPUT_DIR,
    report_path=REPORT_PATH,
):
    """Save 3D trajectory, error curve, and research-style report."""
    analysis = analyze_single_reach(result)

    output_dir = Path(output_dir)
    report_path = Path(report_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    history = result["history"]

    times = np.array([
        row["time_s"] for row in history
    ], dtype=float)

    ee_xyz = np.array([
        [
            row["ee_x_m"],
            row["ee_y_m"],
            row["ee_z_m"],
        ]
        for row in history
    ], dtype=float)

    # Plot 1: Actual measured EE trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        ee_xyz[:, 0],
        ee_xyz[:, 1],
        ee_xyz[:, 2],
        label="Measured EE path",
    )

    ax.scatter(
        *ee_xyz[0],
        marker="o",
        s=60,
        label="Start",
    )

    ax.scatter(
        *ee_xyz[-1],
        marker="x",
        s=80,
        label="Final EE",
    )

    ax.scatter(
        *_point(result, "commanded_approach_world_m"),
        marker="^",
        s=65,
        label="Commanded target",
    )

    ax.scatter(
        *analysis["gt_approach_world_m"],
        marker="+",
        s=80,
        label="GT reference (evaluation only)",
    )

    ax.set(
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    ax.set_title(
        "Perception-driven reaching: measured EE path"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    trajectory_png = (
        output_dir / "single_reach_trajectory_3d.png"
    )
    fig.savefig(trajectory_png, dpi=150)
    plt.close(fig)

    # Plot 2: EE position error over simulated time
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        times,
        [row["ee_error_mm"] for row in history],
        label="EE to commanded target",
    )
    ax.axhline(
        10.0,
        linestyle="--",
        label="10 mm controller threshold",
    )

    ax.set(
        xlabel="Simulated time (s)",
        ylabel="Position error (mm)",
        title="Reaching control error over time",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    error_png = (
        output_dir / "single_reach_control_error.png"
    )
    fig.savefig(error_png, dpi=150)
    plt.close(fig)

    # Research report: keep single-scene claims scoped
    report = f"""# Perception-Driven Reaching: Single-Scene Integration

## Research scope

One simulated 5 cm red cube, one fixed overhead RGB-D camera,
and one Panda robot in PyBullet.

The frozen Method R estimates the cube's top-face center.
Its World-frame output plus a {result['approach_height_m']:.3f} m
vertical approach offset is passed to position-only IK and
joint-space position control.

Simulator GT is used only for evaluation, not for target commands.

## Recorded result

| Quantity | Value |
|---|---:|
| Pipeline status | `{result['status']}` |
| Motor command issued | `{result['command_issued']}` |
| Perception error | {analysis['perception_error_mm']:.3f} mm |
| IK-FK position error | {analysis['ik_error_mm']:.3f} mm |
| Control error | {analysis['control_error_mm']:.3f} mm |
| Final reaching error | {analysis['reaching_error_mm']:.3f} mm |
| Cube displacement | {analysis['cube_displacement_mm']:.3f} mm |
| Simulated steps | {result['simulated_steps']} |
| Simulated time | {result['simulated_time_s']:.3f} s |
| Vector error identity residual | {analysis['vector_closure_mm']:.9f} mm |

## Geometry and interpretation

- The commanded approach position is derived from Method R, not GT.
- Perception error compares the estimated top-face center with
  the pre-motion simulator GT top-face center.
- Control error measures the final EE relative to the commanded
  approach. It includes IK approximation and motor tracking error.
- Final reaching error compares EE with the GT approach,
  using the same approach-height offset.
- Vector errors obey `e_reaching = e_control + e_perception`.
  Their Euclidean lengths need not add.
- The EE trajectory is produced through joint-space position control
  and is not guaranteed to be a Cartesian straight line.

## Outputs

- `outputs/robot_reaching/perception_reach_before.png`
- `outputs/robot_reaching/perception_reach_after.png`
- `outputs/robot_reaching/perception_reach_summary.json`
- `outputs/robot_reaching/perception_reach_trajectory.csv`
- `outputs/robot_reaching/single_reach_trajectory_3d.png`
- `outputs/robot_reaching/single_reach_control_error.png`

## Reproducibility

From the repository root:

```bash
uv run python -m src.run_single_reach
uv run python -m pytest -q
```

## Limitations

This is a single-scene, simulation-only, open-loop perception
demonstration.

It does not establish performance across unseen scenes, safe motion,
collision-free trajectories, real robot transfer, grasping, or
closed-loop visual servoing.

The 10 mm controller threshold is a provisional implementation
criterion, not a robotic safety standard or final task success
threshold.

Multi-scene A/G/R comparisons and robustness transfer remain
future evaluation work.
"""

    report_path.write_text(
        report, encoding="utf-8"
    )

    return trajectory_png, error_png, report_path, analysis


def main():
    scene = create_robot_scene()

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        plt.imsave(
            OUTPUT_DIR / "perception_reach_before.png",
            render_overview(scene),
        )

        result = run_perception_reach(scene)

        plt.imsave(
            OUTPUT_DIR / "perception_reach_after.png",
            render_overview(scene),
        )

        summary_path, trace_path = save_reach_result(
            result, OUTPUT_DIR
        )

        print("Pipeline status:", result["status"])
        print(
            "Motor command issued:",
            result["command_issued"],
        )
        print("Saved:", summary_path)

        if trace_path is None:
            print(
                "No movement: analysis skipped. Reason:",
                result.get("reason", "unknown"),
            )
            return

        plot1, plot2, report, metrics = save_demo_analysis(
            result
        )

        for name in (
            "perception_error_mm",
            "ik_error_mm",
            "control_error_mm",
            "reaching_error_mm",
            "cube_displacement_mm",
            "vector_closure_mm",
        ):
            print(f"{name}: {metrics[name]:.6f}")

        print("Simulated steps:", result["simulated_steps"])

        for path in (trace_path, plot1, plot2, report):
            print("Saved:", path)

    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
