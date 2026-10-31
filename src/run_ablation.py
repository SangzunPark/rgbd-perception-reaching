
"""Compare RGB-D localization improvements independently."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import (
    compute_intrinsics,
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
)
from src.object_localization import estimate_target
from src.object_perception import (
    detect_red_object,
    select_mask_center_pixel,
)
from src.generate_trials import POSITIONS


ROOT_DIR = Path(__file__).resolve().parent.parent
TRIAL_ROOT = ROOT_DIR / "outputs" / "trials"
CSV_PATH = ROOT_DIR / "outputs" / "day6_ablation.csv"

DEPTH_TOLERANCE_M = 0.001

# ID, mask method, pixel method, vertical offset
VARIANTS = [
    ("A", "rgb", "integer", 0.0),
    ("B", "rgb", "subpixel", 0.0),
    ("C", "top", "integer", 0.0),
    ("D", "top", "subpixel", 0.0),
    ("E", "rgb", "integer", 1.0),
    ("F", "rgb", "subpixel", 1.0),
    ("G", "top", "integer", 1.0),
    ("H", "top", "subpixel", 1.0),
]


def detect_top_face(rgb_mask, depth_m):
    """Retain pixels near the closest detected surface."""

    valid = (
        rgb_mask
        & np.isfinite(depth_m)
        & (depth_m > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid RGB-D object pixels")

    closest_depth = np.min(depth_m[valid])

    return (
        valid
        & (depth_m <= closest_depth + DEPTH_TOLERANCE_M)
    )


def estimate_with_variant(
    mask,
    depth_m,
    K,
    view,
    pixel_method,
    vertical_offset,
):
    """Estimate World 3D position with selected configuration."""

    valid = (
        mask
        & np.isfinite(depth_m)
        & (depth_m > 0)
    )

    rows, cols = np.where(valid)

    if rows.size == 0:
        raise ValueError("No valid pixels")

    # 1. Select 2D position
    if pixel_method == "integer":
        u, v = select_mask_center_pixel(valid)

    elif pixel_method == "subpixel":
        u = float(cols.mean())
        v = float(rows.mean())

    else:
        raise ValueError("Invalid pixel method")

    # 2. Estimate surface depth
    z = float(np.median(depth_m[valid]))

    # 3. Apply experimental vertical alignment
    v_adjusted = v + vertical_offset

    # 4. Back-project to Camera 3D
    p_optical = pixel_to_camera(
        u, v_adjusted, z, K
    )

    # 5. Camera -> World
    p_gl = optical_to_opengl(p_optical)
    p_world = opengl_to_world(p_gl, view)

    return p_world


def main():
    results = []

    # Evaluate exactly the same nine scenes
    for index, _ in enumerate(POSITIONS, start=1):
        trial_name = f"trial_{index:02d}"
        trial_dir = TRIAL_ROOT / trial_name

        rgb = plt.imread(
            trial_dir / "rgb.png"
        )[:, :, :3]

        depth = np.load(trial_dir / "depth_m.npy")
        view = np.load(trial_dir / "view_matrix.npy")
        gt = np.load(trial_dir / "gt_top_position.npy")

        h, w = depth.shape
        K = compute_intrinsics(w, h, 60)

        # Detection from RGB-D only, without GT segmentation
        rgb_mask = detect_red_object(rgb)
        top_mask = detect_top_face(rgb_mask, depth)

        masks = {
            "rgb": rgb_mask,
            "top": top_mask,
        }

        for variant_id, mask_type, pixel_method, v_offset in VARIANTS:

            estimate = estimate_with_variant(
                mask=masks[mask_type],
                depth_m=depth,
                K=K,
                view=view,
                pixel_method=pixel_method,
                vertical_offset=v_offset,
            )

            # GT is used only for evaluation
            diff_mm = (estimate - gt) * 1000
            error_mm = float(np.linalg.norm(diff_mm))

            # Ensure A reproduces the original baseline
            if variant_id == "A":
                original = estimate_target(
                    rgb, depth, K, view,
                    depth_method="median",
                )

                baseline_gap_mm = (
                    np.linalg.norm(estimate - original) * 1000
                )

                if baseline_gap_mm > 0.01:
                    raise AssertionError(
                        f"{trial_name}: baseline mismatch "
                        f"({baseline_gap_mm:.4f} mm)"
                    )

            results.append({
                "trial": trial_name,
                "variant": variant_id,
                "mask_type": mask_type,
                "pixel_method": pixel_method,
                "vertical_offset_px": v_offset,
                "estimated_x_m": float(estimate[0]),
                "estimated_y_m": float(estimate[1]),
                "estimated_z_m": float(estimate[2]),
                "dx_mm": float(diff_mm[0]),
                "dy_mm": float(diff_mm[1]),
                "dz_mm": float(diff_mm[2]),
                "error_3d_mm": error_mm,
            })

    # Save all 9 x 8 results
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("Variant | Mask | Pixel    | v Offset | Mean | P90 | Max")
    print("-" * 72)

    for variant_id, mask_type, pixel_method, v_offset in VARIANTS:
        errors = np.array([
            row["error_3d_mm"]
            for row in results
            if row["variant"] == variant_id
        ])

        print(
            f"{variant_id:<7} | "
            f"{mask_type:<4} | "
            f"{pixel_method:<8} | "
            f"{v_offset:+5.1f}px  | "
            f"{np.mean(errors):5.3f} | "
            f"{np.percentile(errors, 90):5.3f} | "
            f"{np.max(errors):5.3f}"
        )

    print(f"\nEvaluated: {len(results)} estimates")
    print(f"Saved CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
