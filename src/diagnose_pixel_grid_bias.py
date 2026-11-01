"""Day 8 Step 5-D: frozen R's clean overhead pixel-grid/centroid diagnostic.

This is an evaluation-only script. It never tunes or changes A/G/R.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import (
    pixel_to_camera,
    optical_to_opengl,
    opengl_to_world,
)
from src.diagnose_projection import project_world_to_pixel
from src.evaluate_position_generalization import (
    ROOT,
    load_overhead_records,
    load_scene,
    save_csv,
)
from src.robust_localization import (
    estimate_robust_target,
    select_robust_surface,
)

OUTPUT = ROOT / "outputs"
RESULT_CSV = OUTPUT / "generalization_pixel_grid_diagnostics.csv"
REFERENCE_CSV = OUTPUT / "position_generalization_results.csv"
ERROR_FIG = OUTPUT / "generalization_overhead_error_components.png"
PHASE_FIG = OUTPUT / "generalization_overhead_pixel_phase.png"
OFFSET_PX = 1.0  # Frozen TinyRenderer-specific vertical alignment
GROUPS = ("development", "interpolation", "extrapolation")


def world_from_pixel_depth(u, v, z_optical, K, view):
    """Back-project a continuous pixel + optical-axis depth to World XYZ."""
    camera = pixel_to_camera(u, v, z_optical, K)
    return opengl_to_world(optical_to_opengl(camera), view)


def gt_optical_depth(gt_world, view):
    """Get the positive optical-axis Z of a ground-truth World point."""
    V = np.asarray(view, dtype=float).reshape((4, 4), order="F")
    p_gl = V @ np.append(np.asarray(gt_world, dtype=float), 1.0)
    z = -float(p_gl[2])
    if not np.isfinite(z) or z <= 0:
        raise ValueError("GT must be in front of the camera")
    return z


def nearest_grid_reference(u_gt, v_gt, offset_px=OFFSET_PX):
    """Nearest *raw* pixel grid to GT after accounting for the fixed v offset.

    This is a hypothetical reference, NOT the pixel used by Method R.
    """
    return (
        float(np.rint(u_gt)),
        float(np.rint(v_gt - offset_px) + offset_px),
    )


def pixel_phase(u_gt, v_gt, offset_px=OFFSET_PX):
    """GT position within its nearest raw-pixel cell, in pixels."""
    return (
        float(u_gt - np.rint(u_gt)),
        float((v_gt - offset_px) - np.rint(v_gt - offset_px)),
    )


def world_error_mm(world, gt_world):
    return float(np.linalg.norm(np.asarray(world) - gt_world) * 1000)


def diagnose_record(record):
    """Compute factual R residuals and non-causal reference measurements."""
    if record["camera_name"] != "overhead":
        raise ValueError("This experiment is for overhead scenes only")

    scene = load_scene(record)
    gt = scene["gt"]
    K, view = scene["K"], scene["view"]
    u_gt, v_gt = project_world_to_pixel(gt, K, view)
    z_gt = gt_optical_depth(gt, view)

    # Check that GT reprojection + backprojection are inverse operations.
    gt_roundtrip = world_from_pixel_depth(u_gt, v_gt, z_gt, K, view)
    if world_error_mm(gt_roundtrip, gt) > 1e-6:
        raise AssertionError("GT projection/backprojection mismatch")

    # Frozen Method R, with NO parameter changes.
    estimated, info = estimate_robust_target(
        scene["rgb"], scene["depth"], K, view,
        vertical_offset_px=OFFSET_PX,
    )
    u_raw, v_raw, z_r, direct_info = select_robust_surface(
        scene["rgb_mask"], scene["depth"]
    )
    if info["inlier_pixels"] != direct_info["inlier_pixels"]:
        raise AssertionError("R's selected inliers are inconsistent")

    u_r, v_r = u_raw, v_raw + OFFSET_PX
    reconstructed_r = world_from_pixel_depth(u_r, v_r, z_r, K, view)
    if np.linalg.norm(reconstructed_r - estimated) > 1e-10:
        raise AssertionError("R's geometry reconstruction does not match")

    mask_rows, mask_cols = np.where(scene["rgb_mask"])
    if not mask_rows.size:
        raise ValueError("Empty RGB mask")
    u_mask = float(mask_cols.mean())
    v_mask = float(mask_rows.mean() + OFFSET_PX)

    # Counterfactual measurements: isolate image point or optical depth.
    image_only = world_from_pixel_depth(u_r, v_r, z_gt, K, view)
    depth_only = world_from_pixel_depth(u_gt, v_gt, z_r, K, view)
    mask_image_only = world_from_pixel_depth(u_mask, v_mask, z_gt, K, view)

    # Illustrative nearest-integer pixel reference, not a prediction of R.
    u_grid, v_grid = nearest_grid_reference(u_gt, v_gt)
    grid_only = world_from_pixel_depth(u_grid, v_grid, z_gt, K, view)
    phase_u, phase_v = pixel_phase(u_gt, v_gt)

    x, y = record["position_xy_m"]
    return {
        "group": record["group"],
        "position_id": record["position_id"],
        "x_m": float(x),
        "y_m": float(y),
        "gt_u": u_gt,
        "gt_v": v_gt,
        "r_u_corrected": u_r,
        "r_v_corrected": v_r,
        "rgb_mask_u_corrected": u_mask,
        "rgb_mask_v_corrected": v_mask,
        "pixel_phase_u": phase_u,
        "pixel_phase_v": phase_v,
        "r_du_px": u_r - u_gt,
        "r_dv_px": v_r - v_gt,
        "rgb_mask_du_px": u_mask - u_gt,
        "rgb_mask_dv_px": v_mask - v_gt,
        "r_center_error_px": float(np.hypot(u_r - u_gt, v_r - v_gt)),
        "r_depth_residual_mm": float((z_r - z_gt) * 1000),
        "r_inlier_pixels": int(info["inlier_pixels"]),
        "actual_error_mm": world_error_mm(estimated, gt),
        "r_centroid_only_error_mm": world_error_mm(image_only, gt),
        "depth_only_error_mm": world_error_mm(depth_only, gt),
        "rgb_mask_centroid_only_error_mm": world_error_mm(mask_image_only, gt),
        "nearest_grid_reference_error_mm": world_error_mm(grid_only, gt),
    }


def validate_reference(rows, reference_path=REFERENCE_CSV):
    """Make sure all 33 R errors reproduce the frozen Step 2 output."""
    with Path(reference_path).open(newline="", encoding="utf-8") as f:
        reference = list(csv.DictReader(f))

    expected = [
        r for r in reference
        if r["method"] == "R" and r["camera_name"] == "overhead"
    ]
    if len(rows) != 33 or len(expected) != 33:
        raise ValueError("Expected 33 overhead R reference records")

    keys = [(r["group"], r["position_id"]) for r in rows]
    ref_keys = [(r["group"], r["position_id"]) for r in expected]
    if len(set(keys)) != 33 or len(set(ref_keys)) != 33:
        raise ValueError("Duplicate overhead position")
    if set(keys) != set(ref_keys):
        raise ValueError("Overhead reference keys mismatch")

    expected_by_key = {
        (r["group"], r["position_id"]): float(r["error_3d_mm"])
        for r in expected
    }
    for r in rows:
        key = (r["group"], r["position_id"])
        if not np.isclose(
            r["actual_error_mm"], expected_by_key[key], atol=1e-6, rtol=0,
        ):
            raise AssertionError(f"Frozen R result mismatch: {key}")
    return len(rows)


def summarize(rows):
    """Mean diagnostics by group; no claim of independent causal effects."""
    measurements = (
        "actual_error_mm", "r_centroid_only_error_mm",
        "depth_only_error_mm", "rgb_mask_centroid_only_error_mm",
        "nearest_grid_reference_error_mm", "r_center_error_px",
        "r_depth_residual_mm", "r_inlier_pixels",
    )
    counts = Counter(r["group"] for r in rows)
    if counts != {"development": 9, "interpolation": 16, "extrapolation": 8}:
        raise ValueError(f"Unexpected position group counts: {counts}")

    result = []
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        item = {"group": group, "cases": len(subset)}
        for key in measurements:
            item[f"mean_{key}"] = float(np.mean([r[key] for r in subset]))
        result.append(item)
    return result


def plot_error_components(summary):
    """Group means: actual R, two controlled counterfactuals, grid reference."""
    series = [
        ("actual_error_mm", "R actual", "#27885D"),
        ("r_centroid_only_error_mm", "R centroid only", "#4E80B7"),
        ("depth_only_error_mm", "R depth only", "#BB6D62"),
        ("nearest_grid_reference_error_mm", "Nearest-pixel reference", "#C99445"),
    ]
    x = np.arange(len(GROUPS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for index, (metric, label, color) in enumerate(series):
        vals = [next(r for r in summary if r["group"] == g)[f"mean_{metric}"]
                for g in GROUPS]
        bars = ax.bar(x + (index - 1.5) * width, vals, width,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, ("Development (9)", "Interpolation (16)", "Extrapolation (8)"))
    ax.set_ylabel("Mean 3D error (mm)")
    ax.set_title("Overhead Camera: R Pixel-Centroid and Depth Diagnostics")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ERROR_FIG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ERROR_FIG}")


def plot_pixel_phase(rows):
    """Compare GT subpixel phase to R's measured centroid residual."""
    colors = {"development": "#4477AA", "interpolation": "#CC8844",
              "extrapolation": "#27885D"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, phase_key, residual_key, name in zip(
        axes,
        ("pixel_phase_u", "pixel_phase_v"),
        ("r_du_px", "r_dv_px"),
        ("U / horizontal", "V / vertical"),
    ):
        for group in GROUPS:
            subset = [r for r in rows if r["group"] == group]
            ax.scatter(
                [r[phase_key] for r in subset],
                [r[residual_key] for r in subset],
                color=colors[group], s=48, alpha=0.85, label=group,
            )
        ax.plot([-0.5, 0.5], [0.5, -0.5], color="#888888", ls="--",
                lw=1, label="Nearest-pixel residual")
        ax.axhline(0, color="black", alpha=0.25, lw=1)
        ax.axvline(0, color="black", alpha=0.25, lw=1)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xlabel("GT subpixel phase (px)")
        ax.set_ylabel("R corrected centroid - GT projection (px)")
        ax.set_title(name)
        ax.grid(alpha=0.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False)
    fig.suptitle("GT Pixel Phase vs Frozen R Centroid Residual", y=1.11)
    fig.tight_layout()
    fig.savefig(PHASE_FIG, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PHASE_FIG}")


def main():
    records = load_overhead_records()
    rows = [diagnose_record(record) for record in records]
    matched = validate_reference(rows)
    summary = summarize(rows)

    save_csv(RESULT_CSV, rows)
    plot_error_components(summary)
    plot_pixel_phase(rows)

    print("\nFrozen overhead R reproduction: PASSED")
    print(f"Overhead scenes validated: {matched}")
    print("Group | N | Actual | Centroid-only | Depth-only | Nearest-grid | Center (px) | Signed depth (mm)")
    print("-" * 114)
    for r in summary:
        print(
            f"{r['group']:<13} {r['cases']:>2} "
            f"{r['mean_actual_error_mm']:>7.3f} "
            f"{r['mean_r_centroid_only_error_mm']:>13.3f} "
            f"{r['mean_depth_only_error_mm']:>11.3f} "
            f"{r['mean_nearest_grid_reference_error_mm']:>12.3f} "
            f"{r['mean_r_center_error_px']:>11.3f} "
            f"{r['mean_r_depth_residual_mm']:>17.3f}"
        )
    print(f"\nSaved: {RESULT_CSV}")
    print("Figures generated: 2")
    print("Note: nearest-grid and counterfactual errors are diagnostic references, not causal attributions.")


if __name__ == "__main__":
    main()
