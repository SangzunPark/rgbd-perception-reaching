
"""Visualize spatial localization errors across trials."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
CSV_PATH = OUTPUT_DIR / "trial_metrics.csv"


def build_error_grid(rows, metric="single_error_mm"):
    """Arrange errors according to World XY positions."""

    xs = sorted({
        float(row["configured_x_m"]) for row in rows
    })

    ys = sorted({
        float(row["configured_y_m"]) for row in rows
    }, reverse=True)

    grid = np.full((len(ys), len(xs)), np.nan)

    for row in rows:
        x = float(row["configured_x_m"])
        y = float(row["configured_y_m"])
        error = float(row[metric])

        col = xs.index(x)
        row_idx = ys.index(y)

        if np.isfinite(grid[row_idx, col]):
            raise ValueError("Duplicate trial position")

        grid[row_idx, col] = error

    if not np.all(np.isfinite(grid)):
        raise ValueError("Incomplete or invalid error grid")

    return xs, ys, grid


def summarize_errors(rows, metric="single_error_mm"):
    """Calculate descriptive error statistics."""

    errors = np.array([
        float(row[metric]) for row in rows
    ])

    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("No valid error values")

    return {
        "mean": float(np.mean(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }


def main():
    # 1. Read evaluation CSV
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("No trial results found")

    trial_names = [row["trial"] for row in rows]
    indices = np.arange(len(rows))

    single = np.array([
        float(row["single_error_mm"]) for row in rows
    ])

    median = np.array([
        float(row["median_error_mm"]) for row in rows
    ])

    # 2. Spatial error heatmap
    xs, ys, error_grid = build_error_grid(rows)

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(
        error_grid,
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:+.2f}" for x in xs])

    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:+.2f}" for y in ys])

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("Spatial 3D Localization Error")

    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.text(
                j, i,
                f"{error_grid[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, label="3D error (mm)")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "spatial_error_heatmap.png",
        dpi=150,
    )
    plt.close(fig)

    # 3. Single vs Median comparison
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.36

    ax.bar(
        indices - width / 2, single, width,
        label="Single",
    )

    ax.bar(
        indices + width / 2, median, width,
        label="Median",
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(trial_names, rotation=45)
    ax.set_ylabel("3D error (mm)")
    ax.set_title("Single vs Median Depth")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "depth_method_comparison.png",
        dpi=150,
    )
    plt.close(fig)

    # 4. XYZ signed error comparison
    dx = np.array([
        float(row["single_dx_mm"]) for row in rows
    ])

    dy = np.array([
        float(row["single_dy_mm"]) for row in rows
    ])

    dz = np.array([
        float(row["single_dz_mm"]) for row in rows
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25

    ax.bar(indices - width, dx, width, label="dX")
    ax.bar(indices, dy, width, label="dY")
    ax.bar(indices + width, dz, width, label="dZ")

    ax.axhline(0, linewidth=1)
    ax.set_xticks(indices)
    ax.set_xticklabels(trial_names, rotation=45)
    ax.set_ylabel("Signed error (mm)")
    ax.set_title("XYZ Localization Error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "xyz_error_components.png",
        dpi=150,
    )
    plt.close(fig)

    # 5. Print summary
    summary = summarize_errors(rows)

    print("Trials:", len(rows))
    print(f"Mean error: {summary['mean']:.3f} mm")
    print(f"Minimum error: {summary['min']:.3f} mm")
    print(f"Maximum error: {summary['max']:.3f} mm")

    print("\nSaved:")
    print("outputs/spatial_error_heatmap.png")
    print("outputs/depth_method_comparison.png")
    print("outputs/xyz_error_components.png")


if __name__ == "__main__":
    main()
