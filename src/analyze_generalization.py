
"""Analyze viewpoint generalization and depth-corruption failures."""

import csv
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from src.depth_corruption import apply_depth_patch_dropout
from src.diagnose_projection import project_world_to_pixel
from src.evaluate_position_generalization import (
    ROOT,
    load_scene,
    save_csv,
)
from src.evaluate_viewpoint_generalization import load_manifest
from src.generate_generalization_scenes import CAMERA_DIRECTIONS
from src.object_perception import select_mask_center_pixel
from src.robust_localization import select_robust_surface
from src.run_ablation import detect_top_face


OUTPUT = ROOT / "outputs"

CLEAN_CSV = OUTPUT / "viewpoint_generalization_results.csv"
ROBUSTNESS_CSV = OUTPUT / "generalization_robustness_results.csv"

DIAGNOSTIC_CSV = OUTPUT / "generalization_geometry_diagnostics.csv"

CAMERAS = tuple(CAMERA_DIRECTIONS)

CONDITIONS = (
    ("Clean", "clean"),
    ("Gaussian 3mm", "gaussian_3mm"),
    ("Gaussian 5mm", "gaussian_5mm"),
    ("Dropout 40%", "dropout_40"),
    ("Dropout 80%", "dropout_80"),
    ("Patch 5x5", "patch_5x5"),
    ("Patch 11x11", "patch_11x11"),
)

METHOD_COLORS = {
    "A": "#4477AA",
    "G": "#CC8844",
    "R": "#339988",
}

OUTCOME_COLORS = {
    "accurate": "#27885D",
    "inaccurate": "#E6A13D",
    "rejected": "#858C98",
}

OUTCOMES = ("accurate", "inaccurate", "rejected")


def read_csv(path):
    """Read existing evaluation results."""

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def outcome_percentages(rows):
    """Calculate all three outcomes, including rejection."""

    if not rows:
        raise ValueError("Empty evaluation group")

    counts = Counter(row["outcome"] for row in rows)

    if set(counts) - set(OUTCOMES):
        raise ValueError("Unknown localization outcome")

    total = len(rows)

    return {
        outcome: 100 * counts[outcome] / total
        for outcome in OUTCOMES
    }


def validate_inputs(clean, robustness):
    """Check that Step 3 and Step 4 outputs are complete."""

    if len(clean) != 396:
        raise ValueError("Expected 396 clean evaluations")

    if len(robustness) != 12384:
        raise ValueError("Expected 12384 robustness evaluations")

    for rows in (clean, robustness):
        if any(row["outcome"] not in OUTCOMES for row in rows):
            raise ValueError("Invalid outcome value")


def save_plot(fig, filename):
    path = OUTPUT / filename
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_clean_viewpoint_errors(rows):
    """Compare A/G/R under clean held-out camera conditions."""

    heldout = [
        row for row in rows
        if row["group"] in ("interpolation", "extrapolation")
    ]

    x = np.arange(len(CAMERAS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, method in enumerate(("A", "G", "R")):
        means = []

        for camera in CAMERAS:
            group = [
                row for row in heldout
                if row["camera_name"] == camera
                and row["method"] == method
            ]

            if len(group) != 24:
                raise ValueError("Incomplete clean camera group")

            returned = [
                float(row["error_3d_mm"])
                for row in group
                if int(row["success"]) == 1
            ]

            if len(returned) != 24:
                raise ValueError(
                    "Unexpected clean rejection; inspect separately"
                )

            means.append(float(np.mean(returned)))

        bars = ax.bar(
            x + (i - 1) * width,
            means,
            width,
            color=METHOD_COLORS[method],
            label=f"Method {method}",
        )

        ax.bar_label(
            bars,
            fmt="%.1f",
            padding=3,
            fontsize=9,
        )

    ax.set_xticks(
        x,
        ("Overhead", "Oblique X", "Oblique Y", "Diagonal"),
    )
    ax.set_ylabel("Mean 3D error (mm)")
    ax.set_title(
        "Clean Generalization: 24 Held-out Positions",
        fontsize=14,
    )
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(title="Algorithm")

    fig.tight_layout()

    save_plot(
        fig,
        "generalization_clean_viewpoint_errors.png",
    )


def plot_robustness_outcomes(rows):
    """Show accurate, inaccurate and rejected returns for R."""

    r_rows = [
        row for row in rows
        if row["method"] == "R"
    ]

    labels = [label for label, _ in CONDITIONS]
    y = np.arange(len(CONDITIONS))

    fig, axes = plt.subplots(
        1, 4,
        figsize=(19, 7),
        sharey=True,
    )

    for ax, camera in zip(axes, CAMERAS):
        accurate = []
        inaccurate = []
        rejected = []

        for _, condition in CONDITIONS:
            group = [
                row for row in r_rows
                if row["camera_name"] == camera
                and row["condition"] == condition
            ]

            expected = (
                240
                if condition.startswith(("gaussian_", "dropout_"))
                else 24
            )

            if len(group) != expected:
                raise ValueError(
                    f"Incomplete outcome data: {camera}, {condition}"
                )

            pct = outcome_percentages(group)

            accurate.append(pct["accurate"])
            inaccurate.append(pct["inaccurate"])
            rejected.append(pct["rejected"])

        accurate = np.array(accurate)
        inaccurate = np.array(inaccurate)
        rejected = np.array(rejected)

        layers = (
            (accurate, np.zeros_like(accurate), "accurate"),
            (inaccurate, accurate, "inaccurate"),
            (
                rejected,
                accurate + inaccurate,
                "rejected",
            ),
        )

        for values, left, outcome in layers:
            ax.barh(
                y,
                values,
                left=left,
                height=0.65,
                color=OUTCOME_COLORS[outcome],
            )

            for j, value in enumerate(values):
                if value >= 15:
                    color = (
                        "black"
                        if outcome == "inaccurate"
                        else "white"
                    )
                    ax.text(
                        left[j] + value / 2,
                        y[j],
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

        ax.set_title(
            camera.replace("_", " ").title(),
            fontsize=12,
        )
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Share (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()

    legend = [
        Patch(
            facecolor=OUTCOME_COLORS[outcome],
            label=outcome.title(),
        )
        for outcome in OUTCOMES
    ]

    fig.suptitle(
        "Method R: Localization Outcomes Across Camera Views",
        fontsize=16,
        y=0.99,
    )

    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.93),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.82,
        bottom=0.12,
        wspace=0.14,
    )

    save_plot(
        fig,
        "generalization_robustness_outcomes.png",
    )


def robust_inlier_mask(object_mask, depth):
    """Reconstruct R's inlier mask for visualization only."""

    valid = (
        object_mask
        & np.isfinite(depth)
        & (depth > 0)
    )

    if not np.any(valid):
        raise ValueError("No valid object depth")

    values = depth[valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    # Same frozen defaults as select_robust_surface().
    tolerance = max(0.001, 3.0 * 1.4826 * mad)

    return (
        valid
        & (np.abs(depth - median) <= tolerance)
    )


def surface_details(scene):
    """Collect G/R surface pixels without changing the estimators."""

    object_mask = scene["rgb_mask"]
    depth = scene["depth"]

    g_mask = detect_top_face(object_mask, depth)
    g_u, g_v = select_mask_center_pixel(g_mask)
    g_z = float(np.median(depth[g_mask]))

    r_u, r_v, r_z, diagnostics = select_robust_surface(
        object_mask, depth
    )

    r_mask = robust_inlier_mask(object_mask, depth)

    if int(r_mask.sum()) != diagnostics["inlier_pixels"]:
        raise AssertionError("R inlier-mask mismatch")

    return {
        "G": {
            "mask": g_mask,
            "u": float(g_u),
            "v": float(g_v),
            "z": g_z,
        },
        "R": {
            "mask": r_mask,
            "u": float(r_u),
            "v": float(r_v),
            "z": float(r_z),
        },
    }


def gt_image_geometry(scene):
    """Compute GT pixel and optical-axis depth."""

    u_gt, v_gt = project_world_to_pixel(
        scene["gt"],
        scene["K"],
        scene["view"],
    )

    view = np.asarray(scene["view"]).reshape(
        (4, 4),
        order="F",
    )

    gt_gl = view @ np.append(scene["gt"], 1.0)
    gt_optical_z = -float(gt_gl[2])

    return u_gt, v_gt, gt_optical_z


def build_geometry_diagnostics(manifest):
    """Measure image-center and optical-depth residuals."""

    rows = []

    for record in manifest:
        if record["group"] == "development":
            continue

        scene = load_scene(record)
        u_gt, v_gt, z_gt = gt_image_geometry(scene)
        surfaces = surface_details(scene)

        for method in ("G", "R"):
            item = surfaces[method]

            # Both G and R use the frozen +1 vertical offset.
            u = item["u"]
            v = item["v"] + 1.0

            rows.append({
                "group": record["group"],
                "position_id": record["position_id"],
                "camera_name": record["camera_name"],
                "method": method,
                "selected_pixels": int(item["mask"].sum()),
                "selected_u": u,
                "selected_v_corrected": v,
                "gt_u": u_gt,
                "gt_v": v_gt,
                "center_error_px": float(
                    np.hypot(u - u_gt, v - v_gt)
                ),
                "selected_depth_m": item["z"],
                "gt_optical_depth_m": z_gt,
                "depth_residual_mm": float(
                    (item["z"] - z_gt) * 1000
                ),
            })

    if len(rows) != 192:
        raise AssertionError(
            "Expected 96 scenes x 2 surface methods"
        )

    return rows


def display_mask(ax, rgb, mask, color, title, gt_uv, center=None):
    """Overlay selected pixels in a zoomed RGB image."""

    ax.imshow(rgb, interpolation="nearest")

    overlay = np.ma.masked_where(
        ~mask,
        np.ones(mask.shape, dtype=float),
    )

    ax.imshow(
        overlay,
        cmap=ListedColormap([color]),
        vmin=0,
        vmax=1,
        alpha=0.65,
        interpolation="nearest",
    )

    ax.scatter(
        [gt_uv[0]],
        [gt_uv[1]],
        marker="*",
        s=180,
        color="#00E5FF",
        edgecolor="black",
        linewidth=0.8,
        label="GT projection",
        zorder=5,
    )

    if center is not None:
        ax.scatter(
            [center[0]],
            [center[1]],
            marker="x",
            s=110,
            linewidth=2.5,
            color="yellow",
            label="Estimated pixel",
            zorder=6,
        )

    object_rows, object_cols = np.where(
        np.any(rgb > 0, axis=2)
        if not np.any(mask)
        else mask
    )

    if object_rows.size:
        pad = 12
        ax.set_xlim(
            max(0, object_cols.min() - pad),
            min(rgb.shape[1], object_cols.max() + pad),
        )
        ax.set_ylim(
            min(rgb.shape[0], object_rows.max() + pad),
            max(0, object_rows.min() - pad),
        )

    ax.set_title(title)
    ax.set_xlabel("Pixel u")
    ax.set_ylabel("Pixel v")


def plot_surface_selection(manifest):
    """Inspect selected G/R pixels in an oblique clean scene."""

    record = next(
        row for row in manifest
        if row["group"] == "interpolation"
        and row["position_id"] == "interpolation_01"
        and row["camera_name"] == "oblique_diagonal"
    )

    scene = load_scene(record)
    surfaces = surface_details(scene)
    u_gt, v_gt, _ = gt_image_geometry(scene)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5.5),
    )

    object_mask = scene["rgb_mask"]

    panels = [
        (
            object_mask,
            "#E6A13D",
            "RGB Object Mask",
            None,
        ),
        (
            surfaces["G"]["mask"],
            "#E65A9D",
            "G: Nearest-Depth Surface",
            (
                surfaces["G"]["u"],
                surfaces["G"]["v"] + 1,
            ),
        ),
        (
            surfaces["R"]["mask"],
            "#27885D",
            "R: Median/MAD Surface",
            (
                surfaces["R"]["u"],
                surfaces["R"]["v"] + 1,
            ),
        ),
    ]

    # All panels use the same crop around the full RGB object.
    rows, cols = np.where(object_mask)
    x0 = max(0, cols.min() - 12)
    x1 = min(scene["rgb"].shape[1], cols.max() + 13)
    y0 = max(0, rows.min() - 12)
    y1 = min(scene["rgb"].shape[0], rows.max() + 13)

    for ax, (mask, color, title, center) in zip(
        axes, panels
    ):
        display_mask(
            ax,
            scene["rgb"],
            mask,
            color,
            title,
            (u_gt, v_gt),
            center,
        )

        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    fig.suptitle(
        "Clean Oblique Diagonal: G vs R Surface Selection\n"
        "Cyan star = GT, yellow X = estimated pixel",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))

    save_plot(
        fig,
        "generalization_surface_selection.png",
    )


def select_patch_failure_case(rows):
    """Select the largest inaccurate R patch return on diagonal view."""

    candidates = [
        row for row in rows
        if row["method"] == "R"
        and row["camera_name"] == "oblique_diagonal"
        and row["condition"] == "patch_11x11"
        and row["outcome"] == "inaccurate"
    ]

    if not candidates:
        raise ValueError(
            "No inaccurate R diagonal patch case found"
        )

    return max(
        candidates,
        key=lambda row: float(row["error_3d_mm"]),
    )


def plot_patch_failure_case(manifest, robustness_rows):
    """Inspect an inaccurate R estimate under an 11x11 missing patch."""

    chosen = select_patch_failure_case(robustness_rows)

    record = next(
        row for row in manifest
        if row["group"] == chosen["group"]
        and row["position_id"] == chosen["position_id"]
        and row["camera_name"] == chosen["camera_name"]
    )

    clean_scene = load_scene(record)

    center_u, center_v = select_mask_center_pixel(
        clean_scene["rgb_mask"]
    )

    damaged = apply_depth_patch_dropout(
        clean_scene["depth"],
        center_u=center_u,
        center_v=center_v,
        radius=5,
    )

    damaged_scene = {
        **clean_scene,
        "depth": damaged,
    }

    # The selected case returned a coordinate, so this should succeed.
    r_u, r_v, _, info = select_robust_surface(
        damaged_scene["rgb_mask"],
        damaged,
    )

    r_mask = robust_inlier_mask(
        damaged_scene["rgb_mask"],
        damaged,
    )

    if int(r_mask.sum()) != info["inlier_pixels"]:
        raise AssertionError("Patch case inlier mismatch")

    valid_object = (
        clean_scene["rgb_mask"]
        & np.isfinite(damaged)
        & (damaged > 0)
    )

    u_gt, v_gt, _ = gt_image_geometry(clean_scene)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5.5),
    )

    panels = [
        (
            clean_scene["rgb_mask"],
            "#E6A13D",
            "Original RGB Mask",
            None,
        ),
        (
            valid_object,
            "#4477AA",
            "Depth Remaining After Patch",
            None,
        ),
        (
            r_mask,
            "#27885D",
            "R: Accepted Depth Inliers",
            (r_u, r_v + 1.0),
        ),
    ]

    rows, cols = np.where(clean_scene["rgb_mask"])
    x0 = max(0, cols.min() - 12)
    x1 = min(clean_scene["rgb"].shape[1], cols.max() + 13)
    y0 = max(0, rows.min() - 12)
    y1 = min(clean_scene["rgb"].shape[0], rows.max() + 13)

    for ax, (mask, color, title, center) in zip(
        axes, panels
    ):
        display_mask(
            ax,
            clean_scene["rgb"],
            mask,
            color,
            title,
            (u_gt, v_gt),
            center,
        )

        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)

    fig.suptitle(
        f"R Inaccurate Return: "
        f"{chosen['position_id']} / Oblique Diagonal\n"
        f"3D error={float(chosen['error_3d_mm']):.2f} mm, "
        f"valid={info['valid_pixels']}, "
        f"inliers={info['inlier_pixels']}",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))

    save_plot(
        fig,
        "generalization_patch_failure_case.png",
    )


def main():
    clean = read_csv(CLEAN_CSV)
    robustness = read_csv(ROBUSTNESS_CSV)

    validate_inputs(clean, robustness)

    manifest = load_manifest()

    if len(manifest) != 132:
        raise ValueError("Incomplete scene manifest")

    plot_clean_viewpoint_errors(clean)
    plot_robustness_outcomes(robustness)
    plot_surface_selection(manifest)
    plot_patch_failure_case(manifest, robustness)

    diagnostics = build_geometry_diagnostics(manifest)
    save_csv(DIAGNOSTIC_CSV, diagnostics)

    print("\nGeneralization analysis validation: PASSED")
    print(f"Clean predictions: {len(clean)}")
    print(f"Robustness predictions: {len(robustness)}")
    print(f"Geometry diagnostics: {len(diagnostics)}")
    print("Generated figures: 4")
    print(f"Saved: {DIAGNOSTIC_CSV}")


if __name__ == "__main__":
    main()
