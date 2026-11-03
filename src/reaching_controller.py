
"""Joint-space position control in the Panda simulator."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from src.robot_scene import (
    OUTPUT_DIR,
    create_robot_scene,
    close_robot_scene,
    render_overview,
)
from src.robot_kinematics import (
    forward_kinematics,
    get_arm_joint_angles,
    get_ee_world_position,
)


DT = 1.0 / 240.0


def make_calibration_joint_target(scene):
    """Create a small, known-reachable joint configuration."""

    target = get_arm_joint_angles(scene).copy()

    # Small movements from the initial pose
    target[0] += 0.20
    target[1] += 0.20
    target[3] += 0.15

    return validate_joint_target(scene, target)


def validate_joint_target(scene, joint_angles):
    """Check finite angles and the Panda URDF limits."""

    q = np.asarray(joint_angles, dtype=float)

    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("Expected seven finite joint angles")

    for index, angle in zip(scene.arm_joint_indices, q):
        info = p.getJointInfo(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )

        if not (info[8] <= angle <= info[9]):
            raise ValueError(
                f"Joint {index} violates URDF limits"
            )

    return q


def run_joint_position_control(
    scene,
    target_joint_angles,
    target_world,
    max_steps=1200,
    ramp_steps=240,
    settle_steps=30,
    position_tolerance_m=0.010,
    sample_every=10,
):
    """Move arm joints using simulated position motors.

    target_world is a commanded World-frame reference,
    never a ground-truth measurement.

    The function changes the simulated joint states over time.
    """

    goal = validate_joint_target(
        scene, target_joint_angles
    )

    target = np.asarray(target_world, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Invalid World target")

    if (
        not isinstance(max_steps, int)
        or not isinstance(ramp_steps, int)
        or not isinstance(settle_steps, int)
        or not isinstance(sample_every, int)
        or ramp_steps < 1
        or settle_steps < 1
        or sample_every < 1
        or max_steps < ramp_steps + settle_steps
    ):
        raise ValueError("Invalid simulation step parameters")

    if (
        not np.isfinite(position_tolerance_m)
        or position_tolerance_m <= 0
    ):
        raise ValueError("Invalid position tolerance")

    start_q = get_arm_joint_angles(scene)
    initial_ee = get_ee_world_position(scene)

    # Maximum forces/torques declared in the Panda URDF
    max_forces = []

    for index in scene.arm_joint_indices:
        info = p.getJointInfo(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )
        force = float(info[10])

        if force <= 0 or not np.isfinite(force):
            raise ValueError(
                f"Invalid motor force for joint {index}"
            )

        max_forces.append(force)

    # Hold fingers in their current open configuration
    finger_indices = (9, 10)

    finger_targets = [
        p.getJointState(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )[0]
        for index in finger_indices
    ]

    p.setJointMotorControlArray(
        scene.robot_id,
        list(finger_indices),
        controlMode=p.POSITION_CONTROL,
        targetPositions=finger_targets,
        forces=[20.0, 20.0],
        physicsClientId=scene.client_id,
    )

    history = []
    stable_count = 0
    converged = False

    def measure(step):
        """Read actual physical simulation state."""

        ee = get_ee_world_position(scene)

        states = p.getJointStates(
            scene.robot_id,
            list(scene.arm_joint_indices),
            physicsClientId=scene.client_id,
        )

        actual_q = np.array([
            state[0] for state in states
        ])
        velocities = np.array([
            state[1] for state in states
        ])

        ee_error = float(np.linalg.norm(ee - target))
        joint_error = float(np.max(np.abs(goal - actual_q)))
        joint_speed = float(np.max(np.abs(velocities)))

        record = {
            "step": step,
            "time_s": step * DT,
            "ee_x_m": float(ee[0]),
            "ee_y_m": float(ee[1]),
            "ee_z_m": float(ee[2]),
            "ee_error_mm": ee_error * 1000.0,
            "max_joint_error_rad": joint_error,
            "max_joint_speed_rad_s": joint_speed,
        }

        return record

    history.append(measure(0))

    last_step = 0

    for step in range(1, max_steps + 1):

        # Linearly interpolate commanded joint positions
        alpha = min(step / ramp_steps, 1.0)

        commanded_q = (
            (1.0 - alpha) * start_q
            + alpha * goal
        )

        # Position motor commands -- not a joint state reset
        p.setJointMotorControlArray(
            scene.robot_id,
            list(scene.arm_joint_indices),
            controlMode=p.POSITION_CONTROL,
            targetPositions=commanded_q.tolist(),
            targetVelocities=[0.0] * 7,
            forces=max_forces,
            positionGains=[0.05] * 7,
            velocityGains=[1.0] * 7,
            physicsClientId=scene.client_id,
        )

        # Advance the physics simulation by DT
        p.stepSimulation(
            physicsClientId=scene.client_id
        )

        record = measure(step)
        last_step = step

        # Require settled motion after the ramp is finished
        is_stable = (
            step >= ramp_steps
            and record["ee_error_mm"]
                <= position_tolerance_m * 1000
            and record["max_joint_error_rad"] <= 0.05
            and record["max_joint_speed_rad_s"] <= 0.10
        )

        if is_stable:
            stable_count += 1
        else:
            stable_count = 0

        if (
            step % sample_every == 0
            or step == max_steps
            or stable_count >= settle_steps
        ):
            history.append(record)

        if stable_count >= settle_steps:
            converged = True
            break

    return {
        "converged": converged,
        "steps": last_step,
        "simulated_time_s": last_step * DT,
        "initial_ee_world_m": initial_ee,
        "target_world_m": target,
        "final_ee_world_m": get_ee_world_position(scene),
        "final_joint_angles_rad": get_arm_joint_angles(scene),
        "target_joint_angles_rad": goal,
        "history": history,
    }


def save_controller_results(result, output_dir):
    """Save controller metrics and position-error plot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = result["history"]

    csv_path = output_dir / "control_calibration_trajectory.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

    error_m = np.linalg.norm(
        result["final_ee_world_m"]
        - result["target_world_m"]
    )

    summary = {
        "experiment": "joint_space_controller_calibration",
        "perception_used": False,
        "converged": bool(result["converged"]),
        "steps": int(result["steps"]),
        "simulated_time_s": result["simulated_time_s"],
        "initial_ee_world_m": (
            result["initial_ee_world_m"].tolist()
        ),
        "target_world_m": result["target_world_m"].tolist(),
        "final_ee_world_m": (
            result["final_ee_world_m"].tolist()
        ),
        "target_joint_angles_rad": (
            result["target_joint_angles_rad"].tolist()
        ),
        "final_joint_angles_rad": (
            result["final_joint_angles_rad"].tolist()
        ),
        "final_ee_error_mm": float(error_m * 1000),
    }

    json_path = output_dir / "control_calibration_summary.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        [r["time_s"] for r in history],
        [r["ee_error_mm"] for r in history],
    )
    ax.axhline(
        y=10.0, linestyle="--",
        label="10 mm controller tolerance",
    )
    ax.set_xlabel("Simulated time (s)")
    ax.set_ylabel("EE position error (mm)")
    ax.set_title("Panda Joint Controller Calibration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plot_path = output_dir / "control_calibration_error.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return csv_path, json_path, plot_path


def main():
    scene = create_robot_scene()

    try:
        # Start from the known initial pose
        start_q = get_arm_joint_angles(scene)

        # Test target generated from joint angles, not GT
        goal_q = make_calibration_joint_target(scene)

        # FK establishes a reachable reference point
        goal_world = forward_kinematics(scene, goal_q)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        plt.imsave(
            OUTPUT_DIR / "control_calibration_before.png",
            render_overview(scene),
        )

        result = run_joint_position_control(
            scene,
            target_joint_angles=goal_q,
            target_world=goal_world,
        )

        plt.imsave(
            OUTPUT_DIR / "control_calibration_after.png",
            render_overview(scene),
        )

        paths = save_controller_results(result, OUTPUT_DIR)

        initial_error = np.linalg.norm(
            result["initial_ee_world_m"] - goal_world
        ) * 1000

        final_error = np.linalg.norm(
            result["final_ee_world_m"] - goal_world
        ) * 1000

        print("Initial joint angles (rad):")
        print(np.round(start_q, 4))

        print("\nCommanded joint angles (rad):")
        print(np.round(goal_q, 4))

        print("\nInitial EE World (m):")
        print(np.round(result["initial_ee_world_m"], 6))

        print("\nFK-derived target World (m):")
        print(np.round(goal_world, 6))

        print("\nFinal simulated EE World (m):")
        print(np.round(result["final_ee_world_m"], 6))

        print("\nInitial EE error (mm):", round(initial_error, 3))
        print("Final EE error (mm):", round(final_error, 3))
        print("Controller converged:", result["converged"])
        print("Simulated steps:", result["steps"])
        print(
            "Simulated time (s):",
            round(result["simulated_time_s"], 3),
        )

        print("\nSaved files:")
        for path in paths:
            print(path)

        if not result["converged"]:
            print(
                "\nWARNING: Controller did not satisfy all "
                "settling criteria."
            )

    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
