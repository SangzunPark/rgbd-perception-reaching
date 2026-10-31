
"""Evaluate localization under Gaussian depth noise."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import compute_intrinsics
from src.depth_corruption import add_gaussian_depth_noise
from src.generate_trials import POSITIONS
from src.object_perception import detect_red_object
from src.run_ablation import (
    detect_top_face,
    estimate_with_variant,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
TRIAL_ROOT = ROOT_DIR / "outputs" / "trials"
OUTPUT_DIR = ROOT_DIR / "outputs"

RESULT_CSV = OUTPUT_DIR / "gaussian_depth_robustness.csv"
SUMMARY_CSV = OUTPUT_DIR / "gaussian_depth_robustness_summary.csv"

SIGMA_LEVELS_MM = (0.0, 1.0, 3.0, 5.0)
NOISY_REPEATS = 20

RESULT_FIELDS = [
    "sigma_mm",
    "repeat",
    "trial",
    "seed",
    "method",
    "success",
    "error_3d_mm",
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "valid_object_pixels",
    "selected_pixels",
    "failure_reason",
]


def load_trial(index):
    """Load one existing simulation scene."""

    trial_name = f"trial_{index:02d}"
    directory = TRIAL_ROOT / trial_name

    rgb = plt.imread(
        directory / "rgb.png"
    )[:, :, :3]

    depth = np.load(directory / "depth_m.npy")
    view = np.load(directory / "view_matrix.npy")
    gt = np.load(directory / "gt_top_position.npy")

    height, width = depth.shape
    K = compute_intrinsics(width, height, 60)

    rgb_mask = detect_red_object(rgb)

    return {
        "name": trial_name,
        "depth": depth,
        "view": view,
        "gt": gt,
        "K": K,
        "rgb_mask": rgb_mask,
    }


def evaluate_method(scene, corrupted_depth, method):
    """Evaluate Day 6 configuration A or G."""

    rgb_mask = scene["rgb_mask"]

    valid_object_pixels = int(np.count_nonzero(
        rgb_mask
        & np.isfinite(corrupted_depth)
        & (corrupted_depth > 0)
    ))

    selected_pixels = 0

    try:
        if method == "A":
            # Day 6 baseline:
            # RGB mask, integer pixel, no vertical offset
            mask = rgb_mask
            vertical_offset = 0.0

        elif method == "G":
            # Day 6 improved:
            # Top-face mask, integer pixel, +1 pixel
            mask = detect_top_face(
                rgb_mask, corrupted_depth
            )
            vertical_offset = 1.0

        else:
            raise ValueError(f"Unknown method: {method}")

        selected_pixels = int(np.count_nonzero(
            mask
            & np.isfinite(corrupted_depth)
            & (corrupted_depth > 0)
        ))

        estimated = estimate_with_variant(
            mask=mask,
            depth_m=corrupted_depth,
            K=scene["K"],
            view=scene["view"],
            pixel_method="integer",
            vertical_offset=vertical_offset,
        )

        if not np.all(np.isfinite(estimated)):
            raise ValueError("Non-finite estimated position")

        difference_mm = (estimated - scene["gt"]) * 1000

        return {
            "success": 1,
            "error_3d_mm": float(
                np.linalg.norm(difference_mm)
            ),
            "dx_mm": float(difference_mm[0]),
            "dy_mm": float(difference_mm[1]),
            "dz_mm": float(difference_mm[2]),
            "valid_object_pixels": valid_object_pixels,
            "selected_pixels": selected_pixels,
            "failure_reason": "",
        }

    except ValueError as exc:
        # Expected localization failures are recorded,
        # rather than silently omitted.
        return {
            "success": 0,
            "error_3d_mm": "",
            "dx_mm": "",
            "dy_mm": "",
            "dz_mm": "",
            "valid_object_pixels": valid_object_pixels,
            "selected_pixels": selected_pixels,
            "failure_reason": str(exc),
        }


def validate_pairs(rows):
    """Check that every corrupted input has both A and G."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            float(row["sigma_mm"]),
            int(row["repeat"]),
            row["trial"],
        )
        groups[key].append(row["method"])

    for key, methods in groups.items():
        if sorted(methods) != ["A", "G"]:
            raise ValueError(
                f"Missing or duplicate method: {key}"
            )

    return len(groups)


def summarize_results(rows):
    """Summarize accuracy and failure rate per condition."""

    groups = defaultdict(list)

    for row in rows:
        key = (float(row["sigma_mm"]), row["method"])
        groups[key].append(row)

    summaries = []

    for (sigma_mm, method), group in sorted(groups.items()):
        successful = [
            row for row in group
            if row["success"] == 1
        ]

        errors = np.array([
            row["error_3d_mm"] for row in successful
        ], dtype=float)

        total = len(group)
        success_count = len(successful)

        summaries.append({
            "sigma_mm": sigma_mm,
            "method": method,
            "cases": total,
            "successes": success_count,
            "failures": total - success_count,
            "success_rate_pct": 100 * success_count / total,
            "mean_error_mm": (
                float(np.mean(errors))
                if errors.size else ""
            ),
            "p90_error_mm": (
                float(np.percentile(errors, 90))
                if errors.size else ""
            ),
            "max_error_mm": (
                float(np.max(errors))
                if errors.size else ""
            ),
        })

    return summaries


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    scenes = [
        load_trial(index)
        for index in range(1, len(POSITIONS) + 1)
    ]

    results = []

    for sigma_mm in SIGMA_LEVELS_MM:
        repeats = (
            1 if sigma_mm == 0 else NOISY_REPEATS
        )

        for repeat in range(repeats):
            for index, scene in enumerate(scenes, start=1):

                # Reproducible seed per condition/repeat/scene
                seed = (
                    42
                    + int(sigma_mm) * 100000
                    + repeat * 100
                    + index
                )

                rng = np.random.default_rng(seed)

                # Generate corruption ONCE for this scenario
                corrupted = add_gaussian_depth_noise(
                    scene["depth"],
                    sigma_m=sigma_mm / 1000,
                    rng=rng,
                )

                # Reuse exactly the same array for A and G
                for method in ("A", "G"):
                    measurement = evaluate_method(
                        scene, corrupted, method
                    )

                    results.append({
                        "sigma_mm": sigma_mm,
                        "repeat": repeat,
                        "trial": scene["name"],
                        "seed": seed,
                        "method": method,
                        **measurement,
                    })

    # Verify paired evaluation integrity
    n_scenarios = validate_pairs(results)

    summaries = summarize_results(results)

    # Confirm the clean condition reproduces Day 6
    for method, expected in (("A", 4.903), ("G", 0.463)):
        clean = next(
            row for row in summaries
            if row["sigma_mm"] == 0
            and row["method"] == method
        )

        measured = clean["mean_error_mm"]

        if (
            clean["failures"] != 0
            or not np.isclose(measured, expected, atol=0.02)
        ):
            raise AssertionError(
                f"Clean baseline mismatch: {method}"
            )

    save_csv(RESULT_CSV, results, RESULT_FIELDS)

    save_csv(
        SUMMARY_CSV,
        summaries,
        list(summaries[0].keys()),
    )

    print("Clean baseline reproduction: PASSED")
    print("Paired evaluation validation: PASSED")
    print(f"Corrupted-input scenarios: {n_scenarios}")
    print(f"Localization evaluations: {len(results)}")

    print(
        "\nSigma | Method | Cases | Failures | "
        "Success | Mean | P90 | Max"
    )
    print("-" * 84)

    for row in summaries:
        def fmt(value):
            return (
                f"{value:.3f}"
                if isinstance(value, (float, np.floating))
                else "N/A"
            )

        print(
            f"{row['sigma_mm']:4.0f}mm | "
            f"{row['method']:^6} | "
            f"{row['cases']:5d} | "
            f"{row['failures']:8d} | "
            f"{row['success_rate_pct']:6.1f}% | "
            f"{fmt(row['mean_error_mm']):>5} | "
            f"{fmt(row['p90_error_mm']):>5} | "
            f"{fmt(row['max_error_mm']):>5}"
        )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
