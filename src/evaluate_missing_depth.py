
"""Evaluate localization robustness to missing depth values."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.depth_corruption import (
    apply_random_depth_dropout,
    apply_depth_patch_dropout,
)
from src.evaluate_gaussian_robustness import (
    load_trial,
    evaluate_method,
)
from src.generate_trials import POSITIONS
from src.object_perception import select_mask_center_pixel


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"

RESULT_CSV = OUTPUT_DIR / "missing_depth_robustness.csv"
SUMMARY_CSV = OUTPUT_DIR / "missing_depth_robustness_summary.csv"

CONDITIONS = [
    ("clean", "clean", 0.0, 0, 1),
    ("dropout_05", "random", 0.05, 0, 20),
    ("dropout_20", "random", 0.20, 0, 20),
    ("dropout_40", "random", 0.40, 0, 20),
    ("dropout_80", "random", 0.80, 0, 20),
    ("dropout_100", "random", 1.00, 0, 1),
    ("patch_3x3", "patch", 0.0, 1, 1),
    ("patch_5x5", "patch", 0.0, 2, 1),
    ("patch_11x11", "patch", 0.0, 5, 1),
]


def validate_pairs(rows):
    """Verify A and G exist once per corrupted input."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["condition"],
            int(row["repeat"]),
            row["trial"],
        )
        groups[key].append(row["method"])

    for key, methods in groups.items():
        if sorted(methods) != ["A", "G"]:
            raise ValueError(f"Invalid method pair: {key}")

    return len(groups)


def summarize_results(rows):
    """Summarize failures and errors for every condition."""

    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["condition"], row["method"])
        ].append(row)

    summaries = []

    for condition, method in sorted(groups):
        group = groups[(condition, method)]

        successful = [
            row for row in group
            if int(row["success"]) == 1
        ]

        errors = np.array(
            [float(r["error_3d_mm"]) for r in successful],
            dtype=float,
        )

        n = len(group)
        failures = n - len(successful)

        summaries.append({
            "condition": condition,
            "method": method,
            "cases": n,
            "failures": failures,
            "success_rate_pct": 100 * (n - failures) / n,
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
            "mean_missing_object_pct": float(np.mean([
                float(r["missing_object_pct"]) for r in group
            ])),
            "mean_selected_pixels": float(np.mean([
                int(r["selected_pixels"]) for r in group
            ])),
        })

    return summaries


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def corrupt_depth(scene, kind, rate, radius, repeat, index):
    """Apply corruption once for both algorithms."""

    clean = scene["depth"]

    if kind == "clean":
        return clean.copy()

    if kind == "random":
        seed = (
            83000
            + int(rate * 100) * 1000
            + repeat * 20
            + index
        )
        rng = np.random.default_rng(seed)

        return apply_random_depth_dropout(
            clean, probability=rate, rng=rng
        )

    if kind == "patch":
        # Pixel selected from RGB only, without GT
        u, v = select_mask_center_pixel(
            scene["rgb_mask"]
        )

        return apply_depth_patch_dropout(
            clean,
            center_u=int(u),
            center_v=int(v),
            radius=radius,
        )

    raise ValueError(f"Unknown corruption type: {kind}")


def main():
    scenes = [
        load_trial(i)
        for i in range(1, len(POSITIONS) + 1)
    ]

    rows = []

    for condition, kind, rate, radius, repeats in CONDITIONS:
        for repeat in range(repeats):
            for index, scene in enumerate(scenes, start=1):

                corrupted = corrupt_depth(
                    scene, kind, rate, radius, repeat, index
                )

                # Measure missing depth only within the RGB object.
                clean_object = (
                    scene["rgb_mask"]
                    & np.isfinite(scene["depth"])
                    & (scene["depth"] > 0)
                )

                n_clean = np.count_nonzero(clean_object)

                if n_clean == 0:
                    raise ValueError("No clean object depth")

                n_remaining = np.count_nonzero(
                    clean_object
                    & np.isfinite(corrupted)
                    & (corrupted > 0)
                )

                missing_pct = (
                    100 * (n_clean - n_remaining) / n_clean
                )

                # Same corrupted image is passed to A and G.
                for method in ("A", "G"):
                    result = evaluate_method(
                        scene, corrupted, method
                    )

                    rows.append({
                        "condition": condition,
                        "corruption_type": kind,
                        "dropout_probability": rate,
                        "patch_radius": radius,
                        "repeat": repeat,
                        "trial": scene["name"],
                        "method": method,
                        "missing_object_pct": missing_pct,
                        **result,
                    })

    scenario_count = validate_pairs(rows)
    summaries = summarize_results(rows)

    # Check that the clean experiment matches Day 6.
    for method, expected in (
        ("A", 4.903),
        ("G", 0.463),
    ):
        item = next(
            s for s in summaries
            if s["condition"] == "clean"
            and s["method"] == method
        )

        if (
            item["failures"] != 0
            or not np.isclose(
                item["mean_error_mm"], expected, atol=0.02
            )
        ):
            raise AssertionError(
                f"Clean result mismatch: {method}"
            )

    save_csv(
        RESULT_CSV, rows, list(rows[0].keys())
    )
    save_csv(
        SUMMARY_CSV, summaries, list(summaries[0].keys())
    )

    def fmt(value):
        if value == "":
            return "N/A"
        return f"{value:.3f}"

    print("Clean baseline reproduction: PASSED")
    print("Paired evaluation validation: PASSED")
    print(f"Corrupted-input scenarios: {scenario_count}")
    print(f"Localization evaluations: {len(rows)}")

    print(
        "\nCondition    | Method | Cases | Fail | "
        "Mean | P90 | Max | Missing % | Selected"
    )
    print("-" * 91)

    for s in summaries:
        print(
            f"{s['condition']:<12} | "
            f"{s['method']:^6} | "
            f"{s['cases']:5d} | "
            f"{s['failures']:4d} | "
            f"{fmt(s['mean_error_mm']):>6} | "
            f"{fmt(s['p90_error_mm']):>6} | "
            f"{fmt(s['max_error_mm']):>6} | "
            f"{s['mean_missing_object_pct']:8.2f} | "
            f"{s['mean_selected_pixels']:8.1f}"
        )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
