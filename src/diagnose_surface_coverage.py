"""Diagnose rejected vs inaccurate RGB-D localization under 11x11 patch dropout."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.analyze_generalization import (
    ROBUSTNESS_CSV,
    gt_image_geometry,
    read_csv,
    robust_inlier_mask,
)
from src.depth_corruption import apply_depth_patch_dropout
from src.evaluate_position_generalization import ROOT, load_scene, save_csv
from src.evaluate_viewpoint_generalization import load_manifest
from src.object_perception import select_mask_center_pixel
from src.robust_localization import estimate_robust_target

OUTPUT = ROOT / "outputs"
CSV_PATH = OUTPUT / "generalization_patch_coverage_diagnostics.csv"
FIGURE_PATH = OUTPUT / "generalization_patch_coverage_comparison.png"


def support_metrics(object_mask, inliers):
    """Measure shape of support without treating it as a new confidence rule."""
    obj = np.asarray(object_mask, dtype=bool)
    sel = np.asarray(inliers, dtype=bool)
    if obj.shape != sel.shape or obj.ndim != 2:
        raise ValueError("Incompatible support masks")
    if np.any(sel & ~obj) or not np.any(obj):
        raise ValueError("Support must be inside nonempty object mask")

    obj_r, obj_c = np.where(obj)
    rows, cols = np.where(sel)
    result = {
        "inlier_pixels": int(rows.size),
        "span_u_fraction": 0.0,
        "span_v_fraction": 0.0,
        "bbox_fill_fraction": 0.0,
        "occupied_quadrants": 0,
        "centroid_shift_from_rgb_px": "",
        "candidate_u": "",
        "candidate_v": "",
    }
    if rows.size == 0:
        return result

    obj_w = int(obj_c.max() - obj_c.min() + 1)
    obj_h = int(obj_r.max() - obj_r.min() + 1)
    span_u = int(cols.max() - cols.min() + 1)
    span_v = int(rows.max() - rows.min() + 1)
    mask_u, mask_v = float(obj_c.mean()), float(obj_r.mean())
    u, v = float(cols.mean()), float(rows.mean())
    quadrants = {
        (int(col >= mask_u), int(row >= mask_v))
        for row, col in zip(rows, cols)
    }
    result.update({
        "span_u_fraction": span_u / obj_w,
        "span_v_fraction": span_v / obj_h,
        "bbox_fill_fraction": rows.size / (span_u * span_v),
        "occupied_quadrants": len(quadrants),
        "centroid_shift_from_rgb_px": float(np.hypot(u-mask_u,v-mask_v)),
        "candidate_u": u,
        "candidate_v": v,
    })
    return result


def inspect_patch_case(record, saved):
    """Reconstruct the exact frozen R input and verify saved outcome."""
    scene = load_scene(record)
    center_u, center_v = select_mask_center_pixel(scene["rgb_mask"])
    corrupted = apply_depth_patch_dropout(
        scene["depth"], center_u=center_u, center_v=center_v, radius=5
    )
    object_mask = scene["rgb_mask"]
    valid = object_mask & np.isfinite(corrupted) & (corrupted > 0)
    selected = robust_inlier_mask(object_mask, corrupted) if np.any(valid) else np.zeros_like(object_mask)
    metrics = support_metrics(object_mask, selected)
    u_gt, v_gt, gt_z = gt_image_geometry(scene)
    success = False
    reason = ""
    error_mm = ""
    try:
        estimate, info = estimate_robust_target(
            scene["rgb"], corrupted, scene["K"], scene["view"],
            vertical_offset_px=1.0,
        )
        success = True
        error_mm = float(np.linalg.norm(estimate-scene["gt"])*1000)
        if info["inlier_pixels"] != metrics["inlier_pixels"]:
            raise AssertionError("Inlier count changed")
    except ValueError as exc:
        reason = str(exc)

    if success != (int(saved["success"]) == 1):
        raise AssertionError("Saved R success not reproduced")
    if success:
        if saved["outcome"] != "inaccurate" and saved["outcome"] != "accurate":
            raise AssertionError("Saved R outcome inconsistent")
        if not np.isclose(error_mm, float(saved["error_3d_mm"]), atol=1e-6, rtol=0):
            raise AssertionError("Saved R error not reproduced")
        if (error_mm <= 5) != (saved["outcome"] == "accurate"):
            raise AssertionError("Saved accuracy label not reproduced")
    elif saved["outcome"] != "rejected" or saved["failure_reason"] != reason:
        raise AssertionError("Saved R rejection reason not reproduced")

    signed_depth_residual = (
        float((np.median(corrupted[selected])-gt_z)*1000)
        if np.any(selected) else ""
    )
    candidate_gt_dist = (
        float(np.hypot(metrics["candidate_u"]-u_gt, metrics["candidate_v"]+1.0-v_gt))
        if np.any(selected) else ""
    )
    result = {
        "group": record["group"],
        "position_id": record["position_id"],
        "camera_name": record["camera_name"],
        "outcome": saved["outcome"],
        "error_3d_mm": error_mm,
        "failure_reason": reason,
        "valid_object_pixels": int(valid.sum()),
        **metrics,
        "candidate_gt_center_error_px": candidate_gt_dist,
        "signed_depth_residual_mm": signed_depth_residual,
    }
    return result, {"rgb_mask": object_mask, "valid_mask": valid, "inlier_mask": selected, "gt_uv": (u_gt,v_gt)}


def summarize_cases(rows):
    """Tabulate camera/outcome subgroups without hiding rejected estimates."""
    blocks = defaultdict(list)
    for row in rows:
        blocks[(row["camera_name"], row["outcome"])].append(row)
    summary = []
    for (camera, outcome), block in sorted(blocks.items()):
        def avg(field):
            vals = [float(r[field]) for r in block if r[field] != ""]
            return float(np.mean(vals)) if vals else ""
        summary.append({
            "camera_name": camera, "outcome": outcome, "cases": len(block),
            "mean_valid_pixels": avg("valid_object_pixels"),
            "mean_inlier_pixels": avg("inlier_pixels"),
            "mean_span_u": avg("span_u_fraction"),
            "mean_span_v": avg("span_v_fraction"),
            "mean_bbox_fill": avg("bbox_fill_fraction"),
            "mean_quadrants": avg("occupied_quadrants"),
            "mean_center_shift_px": avg("centroid_shift_from_rgb_px"),
            "mean_gt_center_error_px": avg("candidate_gt_center_error_px"),
        })
    return summary


def draw_binary_panel(ax, mask, rgb_mask, gt_uv, title, candidate=None):
    """Use discrete colors to expose individual pixels without RGB blending."""
    ax.imshow(mask.astype(int), cmap=ListedColormap(["#F5F5F5", "#288763"]),
              vmin=0, vmax=1, interpolation="nearest")
    rr, cc = np.where(rgb_mask)
    ax.set_xlim(max(0,cc.min()-3), min(mask.shape[1]-1,cc.max()+3))
    ax.set_ylim(min(mask.shape[0]-1,rr.max()+3),max(0,rr.min()-3))
    ax.scatter([gt_uv[0]],[gt_uv[1]], s=90, marker="*", color="#028FD1", edgecolor="black",linewidth=0.8)
    if candidate is not None:
        ax.scatter([candidate[0]], [candidate[1]+1], s=100, marker="x", linewidth=2.5, color="#E6A13D")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def plot_paired_case(rows, masks_by_key):
    """Compare inaccurate diagonal return with rejected X view at same position."""
    inaccurate = [r for r in rows if r["camera_name"] == "oblique_diagonal" and r["outcome"] == "inaccurate"]
    if not inaccurate:
        raise ValueError("No diagonal inaccurate case")
    worst = max(inaccurate, key=lambda r: float(r["error_3d_mm"]))
    partner = next(r for r in rows if r["position_id"] == worst["position_id"] and r["camera_name"] == "oblique_x")
    if partner["outcome"] != "rejected":
        raise ValueError("Expected rejected partner in oblique_x")
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for row_index, case in enumerate((worst,partner)):
        key=(case["position_id"],case["camera_name"])
        masks=masks_by_key[key]
        title_prefix=f"{case['camera_name']}: {case['outcome']}"
        for column,(field,label) in enumerate((
            ("rgb_mask","Full RGB Mask"),
            ("valid_mask","Depth Remaining"),
            ("inlier_mask","R Candidate Inliers"),
        )):
            candidate = (
                (case["candidate_u"],case["candidate_v"])
                if column==2 and case["outcome"] != "rejected"
                else None
            )
            draw_binary_panel(
                axes[row_index,column], masks[field], masks["rgb_mask"], masks["gt_uv"],
                f"{title_prefix}\n{label}", candidate
            )
    fig.suptitle(
        f"11x11 Patch at {worst['position_id']}: Same Position, Two Cameras\n"
        "Blue star = projected GT; orange X = returned R center (+1px)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0,0,1,0.89))
    fig.savefig(FIGURE_PATH,dpi=170,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURE_PATH}")
    return worst, partner


def main():
    original = read_csv(ROBUSTNESS_CSV)
    saved_rows = [r for r in original if r["method"] == "R" and r["condition"] == "patch_11x11"]
    if len(saved_rows) != 96 or len({(r["position_id"],r["camera_name"]) for r in saved_rows}) != 96:
        raise ValueError("Expected 96 unique R 11x11 cases")
    manifest = [r for r in load_manifest() if r["group"] in ("interpolation","extrapolation")]
    scenes = {(r["position_id"],r["camera_name"]):r for r in manifest}
    if set(scenes) != {(r["position_id"],r["camera_name"]) for r in saved_rows}:
        raise ValueError("Manifest and stored cases disagree")
    diagnostic_rows=[]
    masks_by_key={}
    for saved in saved_rows:
        key=(saved["position_id"],saved["camera_name"])
        row, masks=inspect_patch_case(scenes[key],saved)
        diagnostic_rows.append(row)
        masks_by_key[key]=masks
    if Counter(r["outcome"] for r in diagnostic_rows) != {"rejected":83,"inaccurate":13}:
        raise AssertionError("Unexpected frozen Patch 11x11 outcomes")
    save_csv(CSV_PATH,diagnostic_rows)
    print(f"Saved: {CSV_PATH}")
    print("\nCamera | Outcome | N | Valid | Inliers | Span U | Span V | Fill | Quadrants | Center shift")
    for row in summarize_cases(diagnostic_rows):
        shift = row["mean_center_shift_px"]
        shift_label = f"{shift:.2f}" if shift != "" else "N/A"
        print(
            f"{row['camera_name']:<18} {row['outcome']:<11} {row['cases']:>3} "
            f"{row['mean_valid_pixels']:>6.1f} {row['mean_inlier_pixels']:>7.1f} "
            f"{row['mean_span_u']:>6.3f} {row['mean_span_v']:>6.3f} "
            f"{row['mean_bbox_fill']:>5.3f} {row['mean_quadrants']:>9.1f} "
            f"{shift_label}"
        )
    bad, rejected=plot_paired_case(diagnostic_rows,masks_by_key)
    print(f"\nExample inaccurate: {bad['position_id']}, {bad['camera_name']}, error={bad['error_3d_mm']:.3f} mm")
    print(f"Matched rejected: {rejected['position_id']}, {rejected['camera_name']}, reason={rejected['failure_reason']}")
    print("Patch coverage diagnosis: PASSED (96/96 frozen results reproduced)")


if __name__ == "__main__":
    main()
