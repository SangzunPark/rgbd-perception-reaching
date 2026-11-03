
"""Day 9-2: coordinate frames and perception-based target definition."""

import json

import numpy as np
import pybullet as p

from src.robot_scene import (
    OUTPUT_DIR,
    create_robot_scene,
    close_robot_scene,
    capture_sensor_rgbd,
)
from src.robust_localization import estimate_robust_target


def make_transform(position, orientation):
    """Create a 4x4 pose matrix from position and quaternion."""

    position = np.asarray(position, dtype=float)
    orientation = np.asarray(orientation, dtype=float)

    if position.shape != (3,) or orientation.shape != (4,):
        raise ValueError("Invalid pose dimensions")

    if not np.all(np.isfinite(position)):
        raise ValueError("Non-finite position")

    if not np.all(np.isfinite(orientation)):
        raise ValueError("Non-finite orientation")

    if not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-4):
        raise ValueError("Quaternion must be normalized")

    rotation = np.asarray(
        p.getMatrixFromQuaternion(orientation.tolist())
    ).reshape(3, 3)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position

    return transform


def get_robot_base_transform(scene):
    """Return T_world_from_base_link, accounting for inertial offset."""

    com_position, com_orientation = (
        p.getBasePositionAndOrientation(
            scene.robot_id,
            physicsClientId=scene.client_id,
        )
    )

    dynamics = p.getDynamicsInfo(
        scene.robot_id,
        -1,
        physicsClientId=scene.client_id,
    )

    local_inertial_position = dynamics[3]
    local_inertial_orientation = dynamics[4]

    inverse_position, inverse_orientation = p.invertTransform(
        local_inertial_position,
        local_inertial_orientation,
    )

    base_position, base_orientation = p.multiplyTransforms(
        com_position,
        com_orientation,
        inverse_position,
        inverse_orientation,
    )

    return make_transform(base_position, base_orientation)


def base_to_world(point_base, T_world_from_base):
    """Convert a 3D point from Robot Base to World coordinates."""

    point = np.asarray(point_base, dtype=float)
    T = np.asarray(T_world_from_base, dtype=float)

    if point.shape != (3,) or T.shape != (4, 4):
        raise ValueError("Invalid point or transform dimensions")

    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(T)):
        raise ValueError("Non-finite point or transform")

    return T[:3, :3] @ point + T[:3, 3]


def world_to_base(point_world, T_world_from_base):
    """Convert a 3D point from World to Robot Base coordinates."""

    point = np.asarray(point_world, dtype=float)
    T = np.asarray(T_world_from_base, dtype=float)

    if point.shape != (3,) or T.shape != (4, 4):
        raise ValueError("Invalid point or transform dimensions")

    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(T)):
        raise ValueError("Non-finite point or transform")

    rotation = T[:3, :3]
    translation = T[:3, 3]

    return rotation.T @ (point - translation)


def define_approach_point(target_world, height_m=0.10):
    """Place an approach point above a World-frame target."""

    target = np.asarray(target_world, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Invalid target position")

    if not np.isfinite(height_m) or height_m <= 0:
        raise ValueError("Approach height must be positive")

    return target + np.array([0.0, 0.0, height_m])


def estimate_target_frames(scene, approach_height_m=0.10):
    """Estimate target using R, then express it in both frames.

    No ground-truth input is used.
    No joint motion is commanded.
    """

    rgb, depth_m, view, K = capture_sensor_rgbd(scene)

    # Existing frozen Method R
    target_world, diagnostics = estimate_robust_target(
        rgb,
        depth_m,
        K,
        view,
    )

    # Use the estimated coordinate, never simulator GT
    approach_world = define_approach_point(
        target_world,
        height_m=approach_height_m,
    )

    T_world_from_base = get_robot_base_transform(scene)

    target_base = world_to_base(
        target_world, T_world_from_base
    )
    approach_base = world_to_base(
        approach_world, T_world_from_base
    )

    return {
        "target_world_m": target_world,
        "target_base_m": target_base,
        "approach_world_m": approach_world,
        "approach_base_m": approach_base,
        "T_world_from_base": T_world_from_base,
        "diagnostics": diagnostics,
        "approach_height_m": float(approach_height_m),
    }


def main():
    scene = create_robot_scene()

    try:
        result = estimate_target_frames(scene)

        T = result["T_world_from_base"]

        # Check round-trip transform integrity
        reconstructed = base_to_world(
            result["target_base_m"], T
        )

        round_trip_error_mm = (
            np.linalg.norm(
                reconstructed - result["target_world_m"]
            ) * 1000
        )

        print("Robot Base World position (m):")
        print(np.round(T[:3, 3], 6))

        print("\nMethod R target World (m):")
        print(np.round(result["target_world_m"], 6))

        print("\nMethod R target Robot Base (m):")
        print(np.round(result["target_base_m"], 6))

        print("\nApproach target World (m):")
        print(np.round(result["approach_world_m"], 6))

        print("\nApproach target Robot Base (m):")
        print(np.round(result["approach_base_m"], 6))

        print(
            "\nRound-trip error (mm):",
            f"{round_trip_error_mm:.9f}",
        )

        print("\nMethod R diagnostics:")
        print(result["diagnostics"])

        # Save estimated coordinates only, not ground truth
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        record = {
            "method": "R",
            "target_frame_for_ik": "world",
            "target_world_m": result["target_world_m"].tolist(),
            "target_base_m": result["target_base_m"].tolist(),
            "approach_world_m": result["approach_world_m"].tolist(),
            "approach_base_m": result["approach_base_m"].tolist(),
            "approach_height_m": result["approach_height_m"],
            "diagnostics": result["diagnostics"],
        }

        output_path = OUTPUT_DIR / "target_definition.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print("\nSaved:", output_path)

    finally:
        close_robot_scene(scene)


if __name__ == "__main__":
    main()
