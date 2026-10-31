
"""Diagnose GT projection, mask centroid, and pixel selection."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import compute_intrinsics
from src.object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)
from src.generate_trials import POSITIONS


ROOT_DIR = Path(__file__).resolve().parent.parent
TRIAL_ROOT = ROOT_DIR / "outputs" / "trials"

# 검증은 기존 순서의 역순
def project_world_to_pixel(point_world, K, view_matrix):
    """Project a World 3D point to continuous image coordinates."""

    # 1. PyBullet view matrix (column-major)
    V = np.asarray(view_matrix, dtype=float).reshape(
        (4, 4), order="F"
    )

    # 2. World -> OpenGL camera coordinates
    point_h = np.append(point_world, 1.0)
    point_gl = (V @ point_h)[:3]

    # 3. OpenGL -> Optical camera coordinates
    point_opt = np.array([
        point_gl[0],
        -point_gl[1],
        -point_gl[2],
    ])

    x, y, z = point_opt

    if z <= 0:
        raise ValueError("Point is behind the camera")

    # 4. Camera 3D -> continuous image pixel
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    return float(u), float(v)

# 9개의 실험을 진단하기
def main():
    """Compare GT pixel, mask centroid and selected pixel."""

    print(
        "Trial | GT (u,v)       | Centroid (u,v) "
        "| Selected (u,v)"
    )
    print("-" * 75)

    residuals = []

    for index, _ in enumerate(POSITIONS, start=1):

        trial_dir = TRIAL_ROOT / f"trial_{index:02d}"

        # 5. Load data
        rgb = plt.imread(
            trial_dir / "rgb.png"
        )[:, :, :3]

        depth = np.load(trial_dir / "depth_m.npy")
        view = np.load(trial_dir / "view_matrix.npy")
        gt_top = np.load(
            trial_dir / "gt_top_position.npy"
        )

        # 6. Camera parameters
        height, width = depth.shape
        K = compute_intrinsics(width, height, 60)

        # 7. Reproject GT into image
        u_gt, v_gt = project_world_to_pixel(
            gt_top, K, view
        )

        # 8. Compute RGB mask centroid
        mask = detect_red_object(rgb)
        rows, cols = np.where(mask)

        if rows.size == 0:
            raise ValueError(
                f"No object detected in trial_{index:02d}"
            )

        u_centroid = float(cols.mean())
        v_centroid = float(rows.mean())

        # 9. Current integer-pixel selection
        u_sel, v_sel = select_mask_center_pixel(mask)

        # 10. Separate error components
        centroid_minus_gt = np.array([
            u_centroid - u_gt,
            v_centroid - v_gt,
        ])

        selected_minus_centroid = np.array([
            u_sel - u_centroid,
            v_sel - v_centroid,
        ])

        residuals.append((
            centroid_minus_gt,
            selected_minus_centroid,
        ))

        print(
            f"{index:02d}    | "
            f"({u_gt:6.2f},{v_gt:6.2f}) | "
            f"({u_centroid:6.2f},{v_centroid:6.2f}) | "
            f"({u_sel:3d},{v_sel:3d})"
        )

    # 11. Summarize average(of 9 trials) coordinate differences
    centroid_errors = np.array([
        item[0] for item in residuals
    ])
    selection_errors = np.array([
        item[1] for item in residuals
    ])

    print("\n--- Average coordinate differences (pixels) ---")
    print(
        "Centroid - GT (du, dv):",
        np.round(centroid_errors.mean(axis=0), 3),
    )
    print(
        "Selected - Centroid (du, dv):",
        np.round(selection_errors.mean(axis=0), 3),
    )
    print(
        "Selected - GT (du, dv):",
        np.round(
            (centroid_errors + selection_errors).mean(axis=0),
            3,
        ),
    )


if __name__ == "__main__":
    main()
