
"""Paired robustness benchmark for localization methods A, G, R."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.depth_corruption import add_gaussian_depth_noise
from src.evaluate_gaussian_robustness import (
    SIGMA_LEVELS_MM,
    NOISY_REPEATS,
    evaluate_method,
    load_trial,
)
from src.evaluate_missing_depth import CONDITIONS, corrupt_depth
from src.generate_trials import POSITIONS
from src.robust_localization import estimate_robust_target


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

RESULT_CSV = OUTPUT / "robust_localization_benchmark.csv"
SUMMARY_CSV = OUTPUT / "robust_localization_summary.csv"

ACCURACY_THRESHOLD_MM = 5.0
METHODS = ("A", "G", "R")

RESULT_FIELDS = [
    "experiment",
    "condition",
    "repeat",
    "trial",
    "method",
    "success",
    "error_3d_mm",
    "within_5mm",
    "selected_pixels",
    "inlier_pixels",
    "estimated_sigma_mm",
    "failure_reason",
]


def evaluate_r(scene, corrupted_depth):
    """Evaluate the new robust estimator on one depth image."""

    try:
        estimated, info = estimate_robust_target(
            scene["rgb"],
            corrupted_depth,
            scene["K"],
            scene["view"],
        )

        if not np.all(np.isfinite(estimated)):
            raise ValueError("Non-finite position")

        error_mm = float(
            np.linalg.norm(estimated - scene["gt"]) * 1000
        )

        return {
            "success": 1,
            "error_3d_mm": error_mm,
            "selected_pixels": info["inlier_pixels"],
            "inlier_pixels": info["inlier_pixels"],
            "estimated_sigma_mm": info["estimated_sigma_mm"],
            "failure_reason": "",
        }

    except ValueError as exc:
        return {
            "success": 0,
            "error_3d_mm": "",
            "selected_pixels": 0,
            "inlier_pixels": "",
            "estimated_sigma_mm": "",
            "failure_reason": str(exc),
        }


def evaluate_scenario(
    scene, corrupted_depth, experiment, condition, repeat
):
    """Evaluate all methods on exactly the same input."""

    rows = []

    for method in METHODS:
        if method == "R":
            result = evaluate_r(scene, corrupted_depth)
        else:
            result = evaluate_method(
                scene, corrupted_depth, method
            )

        success = int(result["success"])
        error = result["error_3d_mm"]

        within_5mm = int(
            success == 1
            and float(error) <= ACCURACY_THRESHOLD_MM
        )

        rows.append({
            "experiment": experiment,
            "condition": condition,
            "repeat": repeat,
            "trial": scene["name"],
            "method": method,
            "success": success,
            "error_3d_mm": error,
            "within_5mm": within_5mm,
            "selected_pixels": result["selected_pixels"],
            "inlier_pixels": result.get("inlier_pixels", ""),
            "estimated_sigma_mm": result.get(
                "estimated_sigma_mm", ""
            ),
            "failure_reason": result["failure_reason"],
        })

    return rows


def validate_triplets(rows):
    """Require exactly one A, G, R for every input scenario."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["experiment"],
            row["condition"],
            int(row["repeat"]),
            row["trial"],
        )

        groups[key].append(row["method"])

    for key, methods in groups.items():
        if Counter(methods) != Counter(METHODS):
            raise ValueError(f"Invalid method triplet: {key}")

    return len(groups)


def summarize(rows):
    """Calculate accuracy and failure metrics per condition."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["experiment"],
            row["condition"],
            row["method"],
        )
        groups[key].append(row)

    summaries = []

    for key, group in sorted(groups.items()):
        experiment, condition, method = key

        successful = [
            row for row in group
            if int(row["success"]) == 1
        ]

        errors = np.array([
            float(row["error_3d_mm"])
            for row in successful
        ])

        total = len(group)
        failures = total - len(successful)

        qualified = sum(
            int(row["within_5mm"])
            for row in group
        )

        summaries.append({
            "experiment": experiment,
            "condition": condition,
            "method": method,
            "cases": total,
            "failures": failures,
            "failure_rate_pct": 100 * failures / total,
            "within_5mm_pct": 100 * qualified / total,
            "mean_error_mm": (
                float(errors.mean()) if errors.size else ""
            ),
            "p90_error_mm": (
                float(np.percentile(errors, 90))
                if errors.size else ""
            ),
            "max_error_mm": (
                float(errors.max()) if errors.size else ""
            ),
            "mean_selected_pixels": float(np.mean([
                int(row["selected_pixels"])
                for row in group
            ])),
        })

    return summaries


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save empty results")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def verify_previous_results(rows):
    """Confirm A/G reproduce the existing Step 2 and 3 CSVs."""

    old_gaussian = OUTPUT / "gaussian_depth_robustness.csv"
    old_missing = OUTPUT / "missing_depth_robustness.csv"

    def read_index(path, experiment):
        with path.open(newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))

        indexed = {}

        for row in records:
            condition = (
                f"sigma_{float(row['sigma_mm']):g}mm"
                if experiment == "gaussian"
                else row["condition"]
            )

            key = (
                experiment,
                condition,
                int(row["repeat"]),
                row["trial"],
                row["method"],
            )

            if key in indexed:
                raise ValueError(f"Duplicate original: {key}")

            indexed[key] = row

        return indexed

    original = {
        **read_index(old_gaussian, "gaussian"),
        **read_index(old_missing, "missing"),
    }

    matched = 0

    for row in rows:
        if row["method"] == "R":
            continue

        key = (
            row["experiment"],
            row["condition"],
            int(row["repeat"]),
            row["trial"],
            row["method"],
        )

        if key not in original:
            raise AssertionError(
                f"Missing original evaluation: {key}"
            )

        previous = original[key]

        if int(row["success"]) != int(previous["success"]):
            raise AssertionError(
                f"Success mismatch: {key}"
            )

        if int(row["success"]) == 1:
            if not np.isclose(
                float(row["error_3d_mm"]),
                float(previous["error_3d_mm"]),
                rtol=0,
                atol=1e-8,
            ):
                raise AssertionError(
                    f"Error mismatch: {key}"
                )

        matched += 1

    if matched != len(original):
        raise AssertionError("Original result count mismatch")

    return matched


def main():
    scenes = []

    for index in range(1, len(POSITIONS) + 1):
        scene = load_trial(index)

        image_path = (
            OUTPUT / "trials" / scene["name"] / "rgb.png"
        )

        scene["rgb"] = plt.imread(
            image_path
        )[:, :, :3]

        scenes.append(scene)

    results = []

    # 1. Gaussian conditions: reproduce Step 2 exactly
    for sigma_mm in SIGMA_LEVELS_MM:
        repeats = (
            1 if sigma_mm == 0 else NOISY_REPEATS
        )

        condition = f"sigma_{sigma_mm:g}mm"

        for repeat in range(repeats):
            for index, scene in enumerate(scenes, start=1):

                seed = (
                    42
                    + int(sigma_mm) * 100000
                    + repeat * 100
                    + index
                )

                rng = np.random.default_rng(seed)

                corrupted = add_gaussian_depth_noise(
                    scene["depth"],
                    sigma_m=sigma_mm / 1000,
                    rng=rng,
                )

                results.extend(evaluate_scenario(
                    scene,
                    corrupted,
                    experiment="gaussian",
                    condition=condition,
                    repeat=repeat,
                ))

    # 2. Missing depth conditions: reproduce Step 3 exactly
    for condition, kind, rate, radius, repeats in CONDITIONS:
        for repeat in range(repeats):
            for index, scene in enumerate(scenes, start=1):

                corrupted = corrupt_depth(
                    scene,
                    kind,
                    rate,
                    radius,
                    repeat,
                    index,
                )

                results.extend(evaluate_scenario(
                    scene,
                    corrupted,
                    experiment="missing",
                    condition=condition,
                    repeat=repeat,
                ))

    # 3. Validate all paired comparisons
    scenarios = validate_triplets(results)

    # 4. Reproduce all previously measured A/G results
    # 중요한 실험 설계, 조건을 기존과 동일하게 기존 csv 파일을 읽음
    reproduced = verify_previous_results(results)

    # 5. Summarize and save
    summaries = summarize(results)

    save_csv(RESULT_CSV, results)
    save_csv(SUMMARY_CSV, summaries)

    print("A/G previous-result reproduction: PASSED")
    print(f"Previous A/G evaluations verified: {reproduced}")
    print("A/G/R triplet validation: PASSED")
    print(f"Input scenarios: {scenarios}")
    print(f"Localization evaluations: {len(results)}")

    def fmt(value):
        if value == "":
            return "N/A"
        return f"{float(value):.3f}"

    print(
        "\nExperiment | Condition     | Method | "
        "Fail % | <=5mm % | Mean | P90 | Max"
    )
    print("-" * 88)

    for row in summaries:
        print(
            f"{row['experiment']:<10} | "
            f"{row['condition']:<13} | "
            f"{row['method']:^6} | "
            f"{row['failure_rate_pct']:6.1f} | "
            f"{row['within_5mm_pct']:7.1f} | "
            f"{fmt(row['mean_error_mm']):>6} | "
            f"{fmt(row['p90_error_mm']):>6} | "
            f"{fmt(row['max_error_mm']):>6}"
        )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
