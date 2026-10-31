
"""Visualize paired RGB-D localization robustness results."""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

SUMMARY_PATH = OUTPUT / "robust_localization_summary.csv"
RESULT_PATH = OUTPUT / "robust_localization_benchmark.csv"

METHODS = ("A", "G", "R")

GAUSSIAN = [
    (0, "sigma_0mm"),
    (1, "sigma_1mm"),
    (3, "sigma_3mm"),
    (5, "sigma_5mm"),
]

RANDOM_DROPOUT = [
    (0, "clean"),
    (5, "dropout_05"),
    (20, "dropout_20"),
    (40, "dropout_40"),
    (80, "dropout_80"),
    (100, "dropout_100"),
]


def load_summary(path=SUMMARY_PATH):
    """Load and validate the complete A/G/R summary."""

    table = {}
    conditions = {}

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["experiment"],
                row["condition"],
                row["method"],
            )

            if key in table:
                raise ValueError(f"Duplicate summary: {key}")

            table[key] = row

            group_key = key[:2]
            conditions.setdefault(group_key, set()).add(
                row["method"]
            )

    if not table:
        raise ValueError("Summary is empty")

    for condition, methods in conditions.items():
        if methods != set(METHODS):
            raise ValueError(
                f"Incomplete methods: {condition}"
            )

    return table


def get_metric(table, experiment, condition, method, field):
    """Return a numeric metric or NaN when undefined."""

    row = table[(experiment, condition, method)]
    value = row[field]

    if value == "":
        return np.nan

    return float(value)


def save_figure(fig, filename):
    """Save a single figure and release resources."""

    path = OUTPUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Saved: {path}")


def plot_gaussian_error(table):
    """Mean error at each Gaussian noise level."""

    x = [sigma for sigma, _ in GAUSSIAN]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        y = [
            get_metric(
                table, "gaussian", condition,
                method, "mean_error_mm"
            )
            for _, condition in GAUSSIAN
        ]

        ax.plot(x, y, marker="o", label=method)

    ax.set(
        title="Gaussian Noise: Mean 3D Localization Error",
        xlabel="Gaussian depth noise sigma (mm)",
        ylabel="Mean 3D error (mm)",
    )
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="Method")

    save_figure(fig, "robust_gaussian_error.png")


def plot_gaussian_selected_pixels(table):
    """Inspect how noise changes selected surface support."""

    x = [sigma for sigma, _ in GAUSSIAN]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        y = [
            get_metric(
                table, "gaussian", condition,
                method, "mean_selected_pixels"
            )
            for _, condition in GAUSSIAN
        ]

        ax.plot(x, y, marker="o", label=method)

    ax.set(
        title="Gaussian Noise: Selected Pixel Count",
        xlabel="Gaussian depth noise sigma (mm)",
        ylabel="Mean selected pixels",
    )
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="Method")

    save_figure(
        fig, "robust_gaussian_selected_pixels.png"
    )


def plot_random_dropout_error(table):
    """Compare mean error under random missing depth."""

    x = [rate for rate, _ in RANDOM_DROPOUT]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        y = [
            get_metric(
                table, "missing", condition,
                method, "mean_error_mm"
            )
            for _, condition in RANDOM_DROPOUT
        ]

        # 100% dropout has no successful estimates:
        # missing mean errors remain NaN, not zero.
        ax.plot(x, y, marker="o", label=method)

    ax.set(
        title="Random Depth Dropout: Mean 3D Error",
        xlabel="Random depth dropout (%)",
        ylabel="Mean 3D error of successful estimates (mm)",
    )
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="Method")

    save_figure(
        fig, "robust_random_dropout_error.png"
    )


def plot_missing_depth_accuracy(table):
    """Compare the fraction of all cases within 5 mm."""

    conditions = [
        ("Clean", "clean"),
        ("Random 5%", "dropout_05"),
        ("Random 20%", "dropout_20"),
        ("Random 40%", "dropout_40"),
        ("Random 80%", "dropout_80"),
        ("Random 100%", "dropout_100"),
        ("Patch 3x3", "patch_3x3"),
        ("Patch 5x5", "patch_5x5"),
        ("Patch 11x11", "patch_11x11"),
    ]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    for index, method in enumerate(METHODS):
        y = [
            get_metric(
                table, "missing", condition,
                method, "within_5mm_pct"
            )
            for _, condition in conditions
        ]

        ax.bar(
            x + (index - 1) * width,
            y,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x,
        [label for label, _ in conditions],
        rotation=30,
        ha="right",
    )
    ax.set(
        title="Missing Depth: Estimates Within 5 mm",
        ylabel="Within 5 mm (% of all cases)",
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    save_figure(
        fig, "robust_missing_depth_accuracy.png"
    )


def plot_patch_failure_rate(table):
    """Visualize explicit rejection under centered patch dropout."""

    conditions = [
        ("3x3", "patch_3x3"),
        ("5x5", "patch_5x5"),
        ("11x11", "patch_11x11"),
    ]

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, method in enumerate(METHODS):
        y = [
            get_metric(
                table, "missing", condition,
                method, "failure_rate_pct"
            )
            for _, condition in conditions
        ]

        ax.bar(
            x + (index - 1) * width,
            y,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x,
        [label for label, _ in conditions],
    )
    ax.set(
        title="Centered Patch Dropout: Localization Failure Rate",
        xlabel="Missing depth patch size",
        ylabel="Failure rate (%)",
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method")

    save_figure(fig, "robust_patch_failure_rate.png")


def plot_localization_outcomes(table):
    """Visualize accurate, inaccurate, and rejected outcomes."""

    from matplotlib.patches import Patch

    conditions = [
        ("Clean", "clean"),
        ("Random 5%", "dropout_05"),
        ("Random 20%", "dropout_20"),
        ("Random 40%", "dropout_40"),
        ("Random 80%", "dropout_80"),
        ("Random 100%", "dropout_100"),
        ("Patch 3x3", "patch_3x3"),
        ("Patch 5x5", "patch_5x5"),
        ("Patch 11x11", "patch_11x11"),
    ]

    # One consistent color per outcome
    colors = {
        "accurate": "#27885D",
        "inaccurate": "#E6A13D",
        "rejected": "#858C98",
    }

    methods = ("A", "G", "R")
    method_titles = {
        "A": "A — Baseline",
        "G": "G — Min Depth",
        "R": "R — Robust",
    }

    labels = [name for name, _ in conditions]
    y = np.arange(len(conditions))

    # Three aligned panels, one per method
    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 7),
        sharey=True,
    )

    for ax, method in zip(axes, methods):

        accurate = []
        inaccurate = []
        rejected = []

        for _, condition in conditions:

            acc = get_metric(
                table, "missing", condition,
                method, "within_5mm_pct"
            )

            rej = get_metric(
                table, "missing", condition,
                method, "failure_rate_pct"
            )

            if not np.isfinite(acc) or not np.isfinite(rej):
                raise ValueError(
                    f"Missing outcome percentages: "
                    f"{condition}, {method}"
                )

            wrong = 100.0 - acc - rej

            if wrong < -1e-6:
                raise ValueError(
                    f"Invalid outcome percentages: "
                    f"{condition}, {method}"
                )

            accurate.append(acc)
            inaccurate.append(max(0.0, wrong))
            rejected.append(rej)

        accurate = np.array(accurate)
        inaccurate = np.array(inaccurate)
        rejected = np.array(rejected)

        # Horizontal 100% stacked bars
        ax.barh(
            y, accurate,
            color=colors["accurate"],
            height=0.65,
        )

        ax.barh(
            y, inaccurate,
            left=accurate,
            color=colors["inaccurate"],
            height=0.65,
        )

        ax.barh(
            y, rejected,
            left=accurate + inaccurate,
            color=colors["rejected"],
            height=0.65,
        )

        # Add percentage labels inside sufficiently wide segments
        for i in range(len(conditions)):
            segments = [
                (accurate[i], 0, "white"),
                (inaccurate[i], accurate[i], "black"),
                (
                    rejected[i],
                    accurate[i] + inaccurate[i],
                    "white",
                ),
            ]

            for value, left, text_color in segments:
                if value >= 13:
                    ax.text(
                        left + value / 2,
                        y[i],
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="medium",
                        color=text_color,
                    )

        ax.set_title(method_titles[method], fontsize=13)

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Share of all cases (%)")

        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

    # The first panel owns shared condition labels
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()

    # Single legend shared by all three panels
    legend_handles = [
        Patch(
            facecolor=colors["accurate"],
            label="Accurate (<= 5 mm)",
        ),
        Patch(
            facecolor=colors["inaccurate"],
            label="Inaccurate (> 5 mm)",
        ),
        Patch(
            facecolor=colors["rejected"],
            label="Rejected",
        ),
    ]

    fig.suptitle(
        "RGB-D Localization Outcomes Under Missing Depth",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.92),
        ncol=3,
        frameon=False,
    )

    # Space for title, legend, and left-side labels
    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        top=0.81,
        bottom=0.12,
        wspace=0.12,
    )

    path = OUTPUT / "robust_localization_outcomes.png"

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


def inspect_patch_failures(path=RESULT_PATH):
    """Identify why R rejected the largest patch condition."""

    reasons = Counter()

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["experiment"] == "missing"
                and row["condition"] == "patch_11x11"
                and row["method"] == "R"
                and int(row["success"]) == 0
            ):
                reasons[row["failure_reason"]] += 1

    print("\nR failure reasons for Patch 11x11:")

    if not reasons:
        print("No failures recorded")
    else:
        for reason, count in sorted(reasons.items()):
            print(f"  {reason}: {count}")

    return reasons


def main():
    table = load_summary()

    print("A/G/R summary validation: PASSED")

    plot_gaussian_error(table)
    plot_gaussian_selected_pixels(table)
    plot_random_dropout_error(table)
    plot_missing_depth_accuracy(table)
    plot_patch_failure_rate(table)
    plot_localization_outcomes(table)

    reasons = inspect_patch_failures()

    print("\nGenerated 6 robustness figures.")
    print(
        "Patch 11x11 R rejections:",
        sum(reasons.values()),
    )


if __name__ == "__main__":
    main()
