
"""Compare RGB detection and PyBullet segmentation alignment."""

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


def mask_centroid(mask):
    """Compute the continuous centroid of a binary mask."""

    rows, cols = np.where(mask)

    if rows.size == 0:
        raise ValueError("Empty mask")

    return np.array([
        cols.mean(),
        rows.mean(),
    ])


def main():
    rgb_minus_gt = []
    seg_minus_gt = []
    rgb_minus_seg = []

    print(
        "Trial | RGB pixels | Seg pixels | "
        "RGB dv | Seg dv | RGB-Seg dv | IoU"
    )
    print("-" * 79)

    for index, _ in enumerate(POSITIONS, start=1):
        trial_dir = TRIAL_ROOT / f"trial_{index:02d}"

        # 1. Load RGB image
        rgb = plt.imread(
            trial_dir / "rgb.png"
        )[:, :, :3]

        # 2. Load PyBullet segmentation PNG
        segmentation_image = plt.imread(
            trial_dir / "segmentation.png"
        )

        # Saved as black and white
        segmentation_mask = (
            segmentation_image[:, :, 0] > 0.5
        )

        # 3. Detect red object independently
        rgb_mask = detect_red_object(rgb)

        # 4. Calculate both mask centroids
        rgb_center = mask_centroid(rgb_mask)
        seg_center = mask_centroid(segmentation_mask)

        # 5. Reproject GT top-face center
        depth = np.load(trial_dir / "depth_m.npy")
        view = np.load(trial_dir / "view_matrix.npy")
        gt_top = np.load(
            trial_dir / "gt_top_position.npy"
        )

        h, w = depth.shape
        K = compute_intrinsics(w, h, 60)

        u_gt, v_gt = project_world_to_pixel(
            gt_top, K, view
        )

        gt_pixel = np.array([u_gt, v_gt])

        # 6. Compute coordinate differences
        delta_rgb_gt = rgb_center - gt_pixel
        delta_seg_gt = seg_center - gt_pixel
        delta_rgb_seg = rgb_center - seg_center

        rgb_minus_gt.append(delta_rgb_gt)
        seg_minus_gt.append(delta_seg_gt)
        rgb_minus_seg.append(delta_rgb_seg)

        # 7. Compare mask overlap
        intersection = np.count_nonzero(
            rgb_mask & segmentation_mask
        )
        union = np.count_nonzero(
            rgb_mask | segmentation_mask
        )

        iou = intersection / union

        print(
            f"{index:02d}    | "
            f"{np.count_nonzero(rgb_mask):10d} | "
            f"{np.count_nonzero(segmentation_mask):10d} | "
            f"{delta_rgb_gt[1]:+6.2f} | "
            f"{delta_seg_gt[1]:+6.2f} | "
            f"{delta_rgb_seg[1]:+10.2f} | "
            f"{iou:.3f}"
        )

    # 8. Average differences over all trials
    rgb_minus_gt = np.array(rgb_minus_gt)
    seg_minus_gt = np.array(seg_minus_gt)
    rgb_minus_seg = np.array(rgb_minus_seg)

    print("\n--- Mean differences (du, dv) in pixels ---")
    print(
        "RGB - GT:",
        np.round(rgb_minus_gt.mean(axis=0), 3)
    )
    print(
        "Seg - GT:",
        np.round(seg_minus_gt.mean(axis=0), 3)
    )
    print(
        "RGB - Seg:",
        np.round(rgb_minus_seg.mean(axis=0), 3)
    )


if __name__ == "__main__":
    main()
