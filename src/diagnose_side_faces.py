
"""Analyze side-face effects on horizontal localization."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import compute_intrinsics
from src.diagnose_projection import project_world_to_pixel
from src.object_perception import detect_red_object
from src.generate_trials import POSITIONS


ROOT_DIR = Path(__file__).resolve().parent.parent
TRIAL_ROOT = ROOT_DIR / "outputs" / "trials"

# Diagnostic threshold: 1 mm behind nearest visible surface
DEPTH_THRESHOLD_M = 0.001


def centroid(mask):
    """Return continuous (u, v) coordinates."""

    rows, cols = np.where(mask)

    if rows.size == 0:
        raise ValueError("Empty mask")

    return np.array([
        float(cols.mean()),
        float(rows.mean()),
    ])


def main():
    print(
        "Trial | GT u   | RGB du | Seg du | Top du "
        "| Missing RGB | Deeper seg px"
    )
    print("-" * 82)

    for index, _ in enumerate(POSITIONS, start=1):
        trial_dir = TRIAL_ROOT / f"trial_{index:02d}"

        # 1. Load trial data
        rgb = plt.imread(
            trial_dir / "rgb.png"
        )[:, :, :3]

        seg_image = plt.imread(
            trial_dir / "segmentation.png"
        )

        depth = np.load(trial_dir / "depth_m.npy")
        view = np.load(trial_dir / "view_matrix.npy")
        gt = np.load(trial_dir / "gt_top_position.npy")

        # 2. Build masks
        rgb_mask = detect_red_object(rgb)
        seg_mask = seg_image[:, :, 0] > 0.5

        # Pixels present in segmentation but missing in RGB
        missing_rgb_mask = seg_mask & ~rgb_mask

        # 3. Find nearest visible cube depth
        valid_seg = (
            seg_mask
            & np.isfinite(depth)
            & (depth > 0)
        )

        if not np.any(valid_seg):
            raise ValueError("No valid segmented depth")

        nearest_depth = float(np.min(depth[valid_seg]))

        # 4. Separate depth-near top candidates
        top_mask = (
            valid_seg
            & (depth <= nearest_depth + DEPTH_THRESHOLD_M)
        )

        # Pixels deeper than the near-surface threshold
        deeper_mask = valid_seg & ~top_mask

        # 5. Compute mask centers
        rgb_center = centroid(rgb_mask)
        seg_center = centroid(seg_mask)
        top_center = centroid(top_mask)

        # 6. Reproject GT
        h, w = depth.shape
        K = compute_intrinsics(w, h, 60)

        u_gt, v_gt = project_world_to_pixel(
            gt, K, view
        )

        # 7. Horizontal residuals
        rgb_du = rgb_center[0] - u_gt
        seg_du = seg_center[0] - u_gt
        top_du = top_center[0] - u_gt

        n_missing = int(np.count_nonzero(missing_rgb_mask))
        n_deeper = int(np.count_nonzero(deeper_mask))

        print(
            f"{index:02d}    | "
            f"{u_gt:6.2f} | "
            f"{rgb_du:+6.2f} | "
            f"{seg_du:+6.2f} | "
            f"{top_du:+6.2f} | "
            f"{n_missing:11d} | "
            f"{n_deeper:13d}"
        )

        # 8. Investigate pixels excluded by RGB detection
        if n_missing > 0:
            missing_rows, missing_cols = np.where(
                missing_rgb_mask
            )

            depth_delta_mm = (
                depth[missing_rgb_mask] - nearest_depth
            ) * 1000

            deeper_fraction = float(np.mean(
                depth_delta_mm > DEPTH_THRESHOLD_M * 1000
            ))

            print(
                "      Missing RGB pixels:"
                f" mean_u={missing_cols.mean():.2f},"
                f" median_depth_delta={np.median(depth_delta_mm):.2f} mm,"
                f" deeper_fraction={deeper_fraction:.2f}"
            )

    print("\nSide-face diagnostic completed.")


if __name__ == "__main__":
    main()
