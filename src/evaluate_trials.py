
"""Evaluate RGB-D localization across multiple trials."""

import csv
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
from src.object_localization import estimate_target
from src.generate_trials import POSITIONS


# 1. Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
TRIAL_ROOT = ROOT_DIR / "outputs" / "trials"
CSV_PATH = ROOT_DIR / "outputs" / "trial_metrics.csv"


def main():
    """Evaluate all trials against Ground Truth."""

    # Store evaluation results
    results = []

    for index, (x, y) in enumerate(POSITIONS, start=1):

        # 2. Locate trial data
        trial_name = f"trial_{index:02d}"
        trial_dir = TRIAL_ROOT / trial_name

        # 3. Load RGB-D and camera data
        rgb = plt.imread(
            trial_dir / "rgb.png"
        )[:, :, :3]

        depth_m = np.load(
            trial_dir / "depth_m.npy"
        )

        view = np.load(
            trial_dir / "view_matrix.npy"
        )

        # 4. Load Ground Truth (evaluation only)
        gt_top = np.load(
            trial_dir / "gt_top_position.npy"
        )

        # 5. Camera intrinsics
        height, width = depth_m.shape
        K = compute_intrinsics(width, height, 60)

        # 6. Detect object
        mask = detect_red_object(rgb)
        u, v = select_mask_center_pixel(mask)

        # 7. Single-depth estimate
        p_single = estimate_target(
            rgb, depth_m, K, view,
            depth_method="single",
        )

        # 8. Median-depth estimate
        p_median = estimate_target(
            rgb, depth_m, K, view,
            depth_method="median",
        )

        # 9. Coordinate errors (mm)
        diff_single = (p_single - gt_top) * 1000
        diff_median = (p_median - gt_top) * 1000

        # 10. XY errors (mm)
        xy_error_single = np.linalg.norm(
            diff_single[:2]
        )

        xy_error_median = np.linalg.norm(
            diff_median[:2]
        )

        # 11. 3D errors (mm)
        error_single = np.linalg.norm(diff_single)
        error_median = np.linalg.norm(diff_median)

        # 12. Difference between methods (mm)
        method_difference = (
            np.linalg.norm(p_single - p_median) * 1000
        )

        # 13. Save one trial's results in memory
        result = {
            "trial": trial_name,
            "configured_x_m": x,
            "configured_y_m": y,
            "detected_pixels": int(np.count_nonzero(mask)),
            "pixel_u": u,
            "pixel_v": v,

            "gt_x_m": float(gt_top[0]),
            "gt_y_m": float(gt_top[1]),
            "gt_z_m": float(gt_top[2]),

            "single_x_m": float(p_single[0]),
            "single_y_m": float(p_single[1]),
            "single_z_m": float(p_single[2]),

            "median_x_m": float(p_median[0]),
            "median_y_m": float(p_median[1]),
            "median_z_m": float(p_median[2]),

            "single_dx_mm": float(diff_single[0]),
            "single_dy_mm": float(diff_single[1]),
            "single_dz_mm": float(diff_single[2]),

            "median_dx_mm": float(diff_median[0]),
            "median_dy_mm": float(diff_median[1]),
            "median_dz_mm": float(diff_median[2]),

            "single_xy_error_mm": float(xy_error_single),
            "median_xy_error_mm": float(xy_error_median),

            "single_error_mm": float(error_single),
            "median_error_mm": float(error_median),

            "method_difference_mm": float(method_difference),
        }

        results.append(result)

        # 14. Print trial evaluation
        print(f"\n--- {trial_name} ---")
        print("Selected pixel:", (u, v))
        print("GT top:", np.round(gt_top, 6))
        print("Single:", np.round(p_single, 6))
        print("Median:", np.round(p_median, 6))

        print(
            f"Single 3D error: {error_single:.3f} mm"
        )
        print(
            f"Median 3D error: {error_median:.3f} mm"
        )
        print(
            f"Method difference: {method_difference:.4f} mm"
        )

    # 15. Save all trial results as CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted evaluation: {len(results)} trials.")
    print(f"Saved CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()

