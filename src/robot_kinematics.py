
"""Robot FK and position-only IK for Panda reaching."""

import json

import numpy as np
import pybullet as p

from src.robot_scene import (
    OUTPUT_DIR,
    create_robot_scene,
    close_robot_scene,
)
from src.robot_frames import estimate_target_frames


IK_TOLERANCE_M = 0.010


def get_ee_world_position(scene):
    """Read the actual EE link-frame position in World."""

    state = p.getLinkState(
        scene.robot_id,
        scene.ee_link_index,
        computeForwardKinematics=True,
        physicsClientId=scene.client_id,
    )

    return np.asarray(state[4], dtype=float)


def get_arm_joint_angles(scene):
    """Return the seven current arm angles in radians."""

    return np.array([
        p.getJointState(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )[0]
        for index in scene.arm_joint_indices
    ], dtype=float)


def forward_kinematics(scene, arm_angles_rad):
    """Evaluate FK without permanently changing joint states."""

    angles = np.asarray(arm_angles_rad, dtype=float)

    if (
        angles.shape != (7,)
        or not np.all(np.isfinite(angles))
    ):
        raise ValueError("Expected seven finite arm angles")

    previous = [
        p.getJointState(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )[:2]
        for index in scene.arm_joint_indices
    ]

    try:
        # Temporary state assignment for kinematic calculation
        for index, angle in zip(
            scene.arm_joint_indices, angles
        ):
            p.resetJointState(
                scene.robot_id,
                index,
                targetValue=float(angle),
                physicsClientId=scene.client_id,
            )

        return get_ee_world_position(scene)

    finally:
        # Restore original positions and velocities
        for index, (position, velocity) in zip(
            scene.arm_joint_indices, previous
        ):
            p.resetJointState(
                scene.robot_id,
                index,
                targetValue=position,
                targetVelocity=velocity,
                physicsClientId=scene.client_id,
            )


def solve_inverse_kinematics(
    scene,
    target_world,
    tolerance_m=IK_TOLERANCE_M,
    max_rounds=10,
):
    """Solve position-only IK and verify using FK.

    The target is in World coordinates.
    The function does not execute a robot motion.
    """

    target = np.asarray(target_world, dtype=float)

    if (
        target.shape != (3,)
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("Invalid 3D World target")

    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Invalid IK tolerance")

    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_rounds must be positive")

    # Collect every movable joint before the EE link.
    # Panda has seven arm joints plus two finger joints.
    movable = []
    lower = []
    upper = []
    rest = []
    saved_states = {}

    for index in range(scene.ee_link_index + 1):
        info = p.getJointInfo(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )

        if info[2] not in (
            p.JOINT_REVOLUTE,
            p.JOINT_PRISMATIC,
        ):
            continue

        state = p.getJointState(
            scene.robot_id,
            index,
            physicsClientId=scene.client_id,
        )

        movable.append(index)
        lower.append(float(info[8]))
        upper.append(float(info[9]))
        rest.append(float(state[0]))
        saved_states[index] = (state[0], state[1])

    joint_ranges = [
        hi - lo for lo, hi in zip(lower, upper)
    ]

    positions_in_result = {
        index: i for i, index in enumerate(movable)
    }

    arm_angles = get_arm_joint_angles(scene)
    achieved = get_ee_world_position(scene)
    error_m = float(np.linalg.norm(achieved - target))
    rounds_used = 0

    try:
        for round_index in range(1, max_rounds + 1):
            # PyBullet expects the target in World coordinates
            solution = p.calculateInverseKinematics(
                scene.robot_id,
                scene.ee_link_index,
                targetPosition=target.tolist(),
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=joint_ranges,
                restPoses=rest,
                maxNumIterations=200,
                residualThreshold=1e-5,
                physicsClientId=scene.client_id,
            )

            if len(solution) != len(movable):
                raise RuntimeError(
                    "Unexpected IK result length: "
                    f"{len(solution)} vs {len(movable)}"
                )

            # Map IK outputs to the seven arm joints.
            # Fixed joints do not have IK position entries.
            for arm_index in scene.arm_joint_indices:
                result_index = positions_in_result[arm_index]

                # Enforce the URDF joint limits before FK
                angle = float(np.clip(
                    solution[result_index],
                    lower[result_index],
                    upper[result_index],
                ))

                p.resetJointState(
                    scene.robot_id,
                    arm_index,
                    targetValue=angle,
                    physicsClientId=scene.client_id,
                )

            arm_angles = get_arm_joint_angles(scene)
            achieved = get_ee_world_position(scene)
            error_m = float(np.linalg.norm(
                achieved - target
            ))
            rounds_used = round_index

            if error_m <= tolerance_m:
                break

    finally:
        # Never leave the robot in its temporary IK pose
        for index, (position, velocity) in saved_states.items():
            p.resetJointState(
                scene.robot_id,
                index,
                targetValue=position,
                targetVelocity=velocity,
                physicsClientId=scene.client_id,
            )

    return {
        "target_world_m": target,
        "joint_angles_rad": arm_angles,
        "fk_world_m": achieved,
        "ik_error_m": error_m,
        "converged": error_m <= tolerance_m,
        "rounds_used": rounds_used,
    }


def main():
    scene = create_robot_scene()

    try:
        initial_ee = get_ee_world_position(scene)

        # The target comes ONLY from RGB-D Method R
        frames = estimate_target_frames(scene)
        target_world = frames["approach_world_m"]

        result = solve_inverse_kinematics(
            scene, target_world
        )

        # Independent FK check on the returned arm angles
        checked_fk = forward_kinematics(
            scene, result["joint_angles_rad"]
        )

        check_error_mm = 1000 * np.linalg.norm(
            checked_fk - target_world
        )

        print("Initial EE World (m):")
        print(np.round(initial_ee, 6))

        print("\nPerception-driven approach target (m):")
        print(np.round(target_world, 6))

        print("\nIK arm angles (radians):")
        print(np.round(result["joint_angles_rad"], 4))

        print("\nFK of IK solution (World, m):")
        print(np.round(checked_fk, 6))

        print(
            "\nIK-FK position error (mm):",
            f"{check_error_mm:.3f}",
        )
        print("IK converged:", result["converged"])
        print("IK rounds:", result["rounds_used"])

        # The original robot state must be unchanged
        print("\nEE after IK verification (World, m):")
        print(np.round(get_ee_world_position(scene), 6))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        record = {
            "method": "R",
            "target_frame": "world",
            "target_world_m": target_world.tolist(),
            "ik_joint_angles_rad": (
                result["joint_angles_rad"].tolist()
            ),
            "fk_world_m": checked_fk.tolist(),
            "ik_position_error_mm": float(check_error_mm),
            "ik_converged": bool(result["converged"]),
            "ik_rounds": int(result["rounds_used"]),
        }

        path = OUTPUT_DIR / "ik_solution.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print("\nSaved:", path)

        if not result["converged"]:
            print(
                "WARNING: IK target not validated within "
                "the current tolerance."
            )

    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
