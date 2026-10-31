
"""Summarize and visualize Day 6 ablation results."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
INPUT_CSV = OUTPUT_DIR / "day6_ablation.csv"
SUMMARY_CSV = OUTPUT_DIR / "day6_ablation_summary.csv"

VARIANTS = "ABCDEFGH"


def group_trials(rows):
    """Group records by variant and trial, validating coverage."""

    groups = {variant: {} for variant in VARIANTS}

    for row in rows:
        variant = row["variant"]
        trial = row["trial"]

        if variant not in groups:
            raise ValueError(f"Unknown variant: {variant}")

        if trial in groups[variant]:
            raise ValueError("Duplicate trial result")

        groups[variant][trial] = row

    reference = set(groups["A"])

    if not reference:
        raise ValueError("No baseline results")

    for variant in VARIANTS:
        if set(groups[variant]) != reference:
            raise ValueError(
                f"Incomplete trials for variant {variant}"
            )

    return groups


def summarize_variant(trials):
    """Compute 3D error and signed XYZ bias statistics."""

    records = list(trials.values())

    errors = np.array([
        float(row["error_3d_mm"]) for row in records
    ])

    components = np.array([
        [
            float(row["dx_mm"]),
            float(row["dy_mm"]),
            float(row["dz_mm"]),
        ]
        for row in records
    ])

    if not np.all(np.isfinite(errors)):
        raise ValueError("Non-finite errors")

    if np.any(errors < 0):
        raise ValueError("Negative distance error")

    if not np.all(np.isfinite(components)):
        raise ValueError("Non-finite XYZ components")

    mean_bias = components.mean(axis=0)

    return {
        "count": len(errors),
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "p90_mm": float(np.percentile(errors, 90)),
        "max_mm": float(np.max(errors)),
        "mean_dx_mm": float(mean_bias[0]),
        "mean_dy_mm": float(mean_bias[1]),
        "mean_dz_mm": float(mean_bias[2]),
    }


def main():
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = group_trials(rows)

    summaries = {
        variant: summarize_variant(groups[variant])
        for variant in VARIANTS
    }

    # 1. Verify A against original Day 5 measurements
    baseline_path = OUTPUT_DIR / "trial_metrics.csv"

    with baseline_path.open(
        newline="", encoding="utf-8"
    ) as f:
        original = {
            row["trial"]: float(row["median_error_mm"])
            for row in csv.DictReader(f)
        }

    assert set(groups["A"]) == set(original)

    for trial, row in groups["A"].items():
        current = float(row["error_3d_mm"])
        assert np.isclose(
            current, original[trial], atol=0.01
        ), f"Baseline mismatch: {trial}"

    # 2. Save variant summaries
    with SUMMARY_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = ["variant"] + list(
            summaries["A"].keys()
        )

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for variant in VARIANTS:
            writer.writerow({
                "variant": variant,
                **summaries[variant],
            })

    # 3. Plot mean error of all variants
    labels = list(VARIANTS)
    means = [
        summaries[v]["mean_mm"] for v in labels
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(labels, means)
    ax.set_ylabel("Mean 3D error (mm)")
    ax.set_xlabel("Ablation variant")
    ax.set_title("Day 6: Ablation Mean Error")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "day6_ablation_means.png",
        dpi=150,
    )
    plt.close(fig)

    # 4. Paired per-trial comparison: A vs G
    trial_names = sorted(groups["A"])

    baseline_errors = [
        float(groups["A"][trial]["error_3d_mm"])
        for trial in trial_names
    ]

    improved_errors = [
        float(groups["G"][trial]["error_3d_mm"])
        for trial in trial_names
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        trial_names, baseline_errors,
        marker="o", label="A: Baseline",
    )
    ax.plot(
        trial_names, improved_errors,
        marker="o", label="G: Improved",
    )

    ax.set_ylabel("3D error (mm)")
    ax.set_title("Paired Localization Error")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "day6_paired_errors.png",
        dpi=150,
    )
    plt.close(fig)

    # 5. Signed XYZ bias comparison
    axes = ["dX", "dY", "dZ"]

    baseline_bias = [
        summaries["A"][f"mean_{axis.lower()}_mm"]
        for axis in axes
    ]

    improved_bias = [
        summaries["G"][f"mean_{axis.lower()}_mm"]
        for axis in axes
    ]

    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        x - width / 2,
        baseline_bias,
        width,
        label="A: Baseline",
    )

    ax.bar(
        x + width / 2,
        improved_bias,
        width,
        label="G: Improved",
    )

    ax.set_xticks(x, axes)
    ax.set_ylabel("Mean signed error (mm)")
    ax.set_title("XYZ Bias: Before vs After")
    ax.axhline(0, linewidth=1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "day6_xyz_bias.png",
        dpi=150,
    )
    plt.close(fig)

    # 6. Print final comparison
    a = summaries["A"]["mean_mm"]
    g = summaries["G"]["mean_mm"]

    reduction = (a - g) / a * 100

    print("Baseline consistency: PASSED")
    print(f"Evaluated: {len(rows)} records")
    print(f"Baseline mean: {a:.3f} mm")
    print(f"Improved mean: {g:.3f} mm")
    print(f"Mean error reduction: {reduction:.2f}%")

    print("\nSaved:")
    print("outputs/day6_ablation_summary.csv")
    print("outputs/day6_ablation_means.png")
    print("outputs/day6_paired_errors.png")
    print("outputs/day6_xyz_bias.png")


if __name__ == "__main__":
    main()
