
"""Generate position and viewpoint generalization benchmarks."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from src.camera_geometry import compute_intrinsics
from src.diagnose_projection import project_world_to_pixel
from src.generate_trials import POSITIONS
from src.object_perception import detect_red_object


ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_ROOT = ROOT / "outputs" / "trials"

BENCHMARK_ROOT = (
    ROOT
    / "outputs"
    / "generalization"
    / "position_viewpoint_benchmark"
)

WIDTH = 320
HEIGHT = 240
FOV_DEG = 60.0

NEAR = 0.05
FAR = 2.0

CUBE_HALF = 0.025

# Previously used scenes: development reference only
DEVELOPMENT_POSITIONS = tuple(POSITIONS)

# New positions inside the previous spatial bounds
X_INNER = (-0.075, -0.025, 0.025, 0.075)
Y_INNER = (-0.0375, -0.0125, 0.0125, 0.0375)

INTERPOLATION_POSITIONS = tuple(
    (x, y)
    for y in Y_INNER
    for x in X_INNER
)

# New positions outside the previous spatial bounds
EXTRAPOLATION_POSITIONS = (
    (-0.125, -0.0625),
    (-0.125,  0.0000),
    (-0.125,  0.0625),
    ( 0.125, -0.0625),
    ( 0.125,  0.0000),
    ( 0.125,  0.0625),
    ( 0.000, -0.0750),
    ( 0.000,  0.0750),
)

POSITION_GROUPS = {
    "development": DEVELOPMENT_POSITIONS,
    "interpolation": INTERPOLATION_POSITIONS,
    "extrapolation": EXTRAPOLATION_POSITIONS,
}

# Directions are normalized so camera distance to origin
# remains 1 meter for every viewpoint.
CAMERA_DIRECTIONS = {
    "overhead": (0.0, 0.0, 1.0),
    "oblique_x": (0.25, 0.0, 1.0),
    "oblique_y": (0.0, -0.25, 1.0),
    "oblique_diagonal": (0.20, 0.20, 1.0),
}

CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)

SCENE_FILES = (
    "rgb.png",
    "segmentation.png",
    "depth_m.npy",
    "view_matrix.npy",
    "gt_top_position.npy",
)


def camera_eye(camera_name):
    """Return camera position 1 m from World origin."""

    if camera_name not in CAMERA_DIRECTIONS:
        raise ValueError(f"Unknown camera: {camera_name}")

    direction = np.asarray(
        CAMERA_DIRECTIONS[camera_name], dtype=float
    )

    eye = direction / np.linalg.norm(direction)

    return eye.tolist()


def validate_position_groups():
    """Validate group counts and ensure no position overlap."""

    expected_counts = {
        "development": 9,
        "interpolation": 16,
        "extrapolation": 8,
    }

    for group, positions in POSITION_GROUPS.items():
        assert len(positions) == expected_counts[group]
        assert len(set(positions)) == len(positions)

    dev = set(DEVELOPMENT_POSITIONS)
    inner = set(INTERPOLATION_POSITIONS)
    outer = set(EXTRAPOLATION_POSITIONS)

    assert dev.isdisjoint(inner)
    assert dev.isdisjoint(outer)
    assert inner.isdisjoint(outer)

    # Inner positions must remain inside original bounds
    assert all(
        -0.10 < x < 0.10 and -0.05 < y < 0.05
        for x, y in inner
    )

    # Every outer point must exceed an original bound
    assert all(
        abs(x) > 0.10 or abs(y) > 0.05
        for x, y in outer
    )


def scene_directory(group, index, camera_name):
    """Choose a scene directory without replacing old trials."""

    if group == "development" and camera_name == "overhead":
        return ORIGINAL_ROOT / f"trial_{index:02d}"

    return (
        BENCHMARK_ROOT
        / group
        / camera_name
        / f"scene_{index:02d}"
    )


def render_scene(position, camera_name, output_dir):
    """Render a new RGB-D scene; never overwrite existing data."""

    output_dir = Path(output_dir)

    # Safe re-execution: keep a completed generated scene.
    if output_dir.exists() and any(output_dir.iterdir()):
        if all(
            (output_dir / name).is_file()
            for name in SCENE_FILES
        ):
            return False

        raise RuntimeError(
            f"Incomplete scene directory: {output_dir}. "
            "Inspect it before regenerating."
        )

    x, y = position
    eye = camera_eye(camera_name)

    client = p.connect(p.DIRECT)

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )
        p.setGravity(0, 0, -9.81)

        # 1. Environment
        p.loadURDF("plane.urdf")

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CUBE_HALF] * 3,
            rgbaColor=[1, 0, 0, 1],
        )

        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, CUBE_HALF],
        )

        for _ in range(240):
            p.stepSimulation()

        # 2. Ground truth
        gt_position, gt_orientation = (
            p.getBasePositionAndOrientation(cube_id)
        )

        rotation = np.asarray(
            p.getMatrixFromQuaternion(gt_orientation)
        ).reshape(3, 3)

        gt_top = np.asarray(gt_position) + (
            rotation @ np.array([0, 0, CUBE_HALF])
        )

        # 3. Camera pose and projection
        view = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=CAMERA_TARGET,
            cameraUpVector=CAMERA_UP,
        )

        projection = p.computeProjectionMatrixFOV(
            fov=FOV_DEG,
            aspect=WIDTH / HEIGHT,
            nearVal=NEAR,
            farVal=FAR,
        )

        # 4. RGB-D rendering
        image = p.getCameraImage(
            WIDTH,
            HEIGHT,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
        )

        rgb = np.asarray(
            image[2], dtype=np.uint8
        ).reshape(HEIGHT, WIDTH, 4)[:, :, :3]

        depth_buffer = np.asarray(
            image[3]
        ).reshape(HEIGHT, WIDTH)

        segmentation = np.asarray(
            image[4]
        ).reshape(HEIGHT, WIDTH)

        depth_m = (
            NEAR * FAR
            / (FAR - (FAR - NEAR) * depth_buffer)
        )

        cube_mask = segmentation == cube_id

        if not np.any(cube_mask):
            raise ValueError("Cube is not visible")

        # 5. Save new data
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.imsave(output_dir / "rgb.png", rgb)

        plt.imsave(
            output_dir / "segmentation.png",
            cube_mask.astype(float),
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        np.save(
            output_dir / "depth_m.npy",
            depth_m,
        )
        np.save(
            output_dir / "view_matrix.npy",
            np.asarray(view),
        )
        np.save(
            output_dir / "gt_top_position.npy",
            gt_top,
        )

        return True

    finally:
        p.disconnect(client)


def inspect_scene(scene_dir, position, camera_name):
    """Validate rendered measurements and GT projection."""

    scene_dir = Path(scene_dir)

    for name in SCENE_FILES:
        if not (scene_dir / name).is_file():
            raise FileNotFoundError(scene_dir / name)

    rgb = plt.imread(scene_dir / "rgb.png")[:, :, :3]
    segmentation = plt.imread(
        scene_dir / "segmentation.png"
    )[:, :, 0] > 0.5

    depth = np.load(scene_dir / "depth_m.npy")
    view = np.load(scene_dir / "view_matrix.npy")
    gt = np.load(scene_dir / "gt_top_position.npy")

    if rgb.shape != (HEIGHT, WIDTH, 3):
        raise ValueError("Invalid RGB shape")

    if depth.shape != (HEIGHT, WIDTH):
        raise ValueError("Invalid depth shape")

    if view.shape != (16,) or gt.shape != (3,):
        raise ValueError("Invalid geometry shape")

    if not np.all(np.isfinite(depth) & (depth > 0)):
        raise ValueError("Invalid rendered depth")

    if not np.all(np.isfinite(gt)):
        raise ValueError("Non-finite ground truth")

    # Check that actual cube location matches requested XY.
    if not np.allclose(
        gt[:2], np.asarray(position),
        atol=0.002,
    ):
        raise ValueError("GT position mismatch")

    if not np.isclose(gt[2], 0.05, atol=0.002):
        raise ValueError("Unexpected cube top height")

    # Verify the view matrix matches the requested viewpoint.
    expected_view = p.computeViewMatrix(
        cameraEyePosition=camera_eye(camera_name),
        cameraTargetPosition=CAMERA_TARGET,
        cameraUpVector=CAMERA_UP,
    )

    if not np.allclose(view, expected_view, atol=1e-5):
        raise ValueError("Camera view matrix mismatch")

    rgb_mask = detect_red_object(rgb)

    if not np.any(segmentation):
        raise ValueError("Empty cube segmentation")

    if not np.any(rgb_mask & segmentation):
        raise ValueError("RGB object detection failed")

    K = compute_intrinsics(WIDTH, HEIGHT, FOV_DEG)

    u_gt, v_gt = project_world_to_pixel(
        gt, K, view
    )

    if not (0 <= u_gt < WIDTH and 0 <= v_gt < HEIGHT):
        raise ValueError("GT projected outside image")

    return {
        "gt_top_world_m": gt.tolist(),
        "gt_pixel_uv": [u_gt, v_gt],
        "rgb_detected_pixels": int(
            np.count_nonzero(rgb_mask)
        ),
        "segmentation_pixels": int(
            np.count_nonzero(segmentation)
        ),
    }


def main():
    validate_position_groups()

    manifest = []
    newly_rendered = 0
    reused_original = 0

    for group, positions in POSITION_GROUPS.items():

        print(f"\nPosition group: {group}")

        for index, position in enumerate(
            positions, start=1
        ):
            position_id = f"{group}_{index:02d}"

            for camera_name in CAMERA_DIRECTIONS:
                scene_dir = scene_directory(
                    group, index, camera_name
                )

                is_original = (
                    group == "development"
                    and camera_name == "overhead"
                )

                if is_original:
                    # Never regenerate or modify original trials.
                    reused_original += 1

                else:
                    created = render_scene(
                        position,
                        camera_name,
                        scene_dir,
                    )

                    newly_rendered += int(created)

                details = inspect_scene(
                    scene_dir,
                    position,
                    camera_name,
                )

                record = {
                    "position_id": position_id,
                    "group": group,
                    "camera_name": camera_name,
                    "position_xy_m": list(position),
                    "camera_eye_m": camera_eye(camera_name),
                    "relative_path": str(
                        scene_dir.relative_to(ROOT)
                    ),
                    "reused_original": is_original,
                    **details,
                }

                manifest.append(record)

                print(
                    f"  {position_id:18s} "
                    f"{camera_name:17s} "
                    f"RGB pixels="
                    f"{details['rgb_detected_pixels']:3d}"
                )

    expected_total = (
        sum(len(p) for p in POSITION_GROUPS.values())
        * len(CAMERA_DIRECTIONS)
    )

    assert expected_total == 132
    assert len(manifest) == 132
    assert reused_original == 9

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)

    with (BENCHMARK_ROOT / "manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\nGeneralization dataset validation: PASSED")
    print("Development reference scenes: 9")
    print("Held-out interpolation positions: 16")
    print("Held-out extrapolation positions: 8")
    print("Camera viewpoints: 4")
    print("Total benchmark scenes: 132")
    print(f"New scenes rendered this run: {newly_rendered}")
    print("Original scenes reused: 9")
    print(f"Manifest: {BENCHMARK_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
