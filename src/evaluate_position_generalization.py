
"""Evaluate frozen localization methods at new object positions."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.camera_geometry import compute_intrinsics
from src.evaluate_gaussian_robustness import evaluate_method
from src.generate_generalization_scenes import (
    BENCHMARK_ROOT,
    POSITION_GROUPS,
)
from src.object_perception import detect_red_object
from src.robust_localization import estimate_robust_target


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"

RESULT_CSV = OUTPUT / "position_generalization_results.csv"
SUMMARY_CSV = OUTPUT / "position_generalization_summary.csv"

METHODS = ("A", "G", "R")
ACCURACY_THRESHOLD_MM = 5.0

FIELDS = [
    "group",
    "position_id",
    "camera_name",
    "x_m",
    "y_m",
    "distance_from_origin_m",
    "method",
    "success",
    "outcome",
    "error_3d_mm",
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "selected_pixels",
    "failure_reason",
]


def load_overhead_records(path=MANIFEST_PATH):
    """Select all 33 overhead scenes from the benchmark."""

    with Path(path).open(encoding="utf-8") as f:
        manifest = json.load(f)

    overhead = [
        item for item in manifest
        if item["camera_name"] == "overhead"
    ]

    expected = {
        group: len(positions)
        for group, positions in POSITION_GROUPS.items()
    }

    observed = Counter(
        item["group"] for item in overhead
    )

    if observed != expected:
        raise ValueError(
            f"Unexpected group counts: {observed}"
        )

    position_ids = [
        item["position_id"] for item in overhead
    ]

    if len(position_ids) != len(set(position_ids)):
        raise ValueError("Duplicate position ID")

    return overhead


def load_scene(record):
    """Load RGB-D measurements and GT for one scene."""

    directory = ROOT / record["relative_path"]

    rgb = plt.imread(
        directory / "rgb.png"
    )[:, :, :3]

    depth = np.load(directory / "depth_m.npy")
    view = np.load(directory / "view_matrix.npy")
    gt = np.load(directory / "gt_top_position.npy")

    height, width = depth.shape

    if rgb.shape != (height, width, 3):
        raise ValueError("RGB/Depth shape mismatch")

    K = compute_intrinsics(
        width, height, vertical_fov_deg=60
    )

    return {
        "rgb": rgb,
        "depth": depth,
        "view": view,
        "gt": gt,
        "K": K,
        "rgb_mask": detect_red_object(rgb),
    }


def classify_outcome(success, error_mm, threshold_mm=5.0):
    """Classify a localization outcome."""

    if int(success) == 0:
        return "rejected"

    if not np.isfinite(error_mm):
        raise ValueError("Successful estimate has invalid error")

    if error_mm <= threshold_mm:
        return "accurate"

    return "inaccurate"


def evaluate_r(scene):
    """Evaluate the frozen robust estimator R."""

    try:
        estimated, diagnostics = estimate_robust_target(
            scene["rgb"],
            scene["depth"],
            scene["K"],
            scene["view"],
            vertical_offset_px=1.0,
        )

        if not np.all(np.isfinite(estimated)):
            raise ValueError("Non-finite estimated position")

        delta_mm = (estimated - scene["gt"]) * 1000

        return {
            "success": 1,
            "error_3d_mm": float(np.linalg.norm(delta_mm)),
            "dx_mm": float(delta_mm[0]),
            "dy_mm": float(delta_mm[1]),
            "dz_mm": float(delta_mm[2]),
            "selected_pixels": diagnostics["inlier_pixels"],
            "failure_reason": "",
        }

    except ValueError as exc:
        return {
            "success": 0,
            "error_3d_mm": "",
            "dx_mm": "",
            "dy_mm": "",
            "dz_mm": "",
            "selected_pixels": 0,
            "failure_reason": str(exc),
        }


def evaluate_scene(record, scene):
    """Evaluate A, G and R without changing their parameters."""

    rows = []

    x, y = record["position_xy_m"]

    for method in METHODS:

        if method == "R":
            result = evaluate_r(scene)
        else:
            result = evaluate_method(
                scene,
                scene["depth"],
                method,
            )

        success = int(result["success"])

        outcome = classify_outcome(
            success,
            float(result["error_3d_mm"])
            if success else np.nan,
            ACCURACY_THRESHOLD_MM,
        )

        rows.append({
            "group": record["group"],
            "position_id": record["position_id"],
            "camera_name": record["camera_name"],
            "x_m": float(x),
            "y_m": float(y),
            "distance_from_origin_m": float(
                np.hypot(x, y)
            ),
            "method": method,
            "success": success,
            "outcome": outcome,
            "error_3d_mm": result["error_3d_mm"],
            "dx_mm": result["dx_mm"],
            "dy_mm": result["dy_mm"],
            "dz_mm": result["dz_mm"],
            "selected_pixels": result["selected_pixels"],
            "failure_reason": result["failure_reason"],
        })

    return rows


def validate_triplets(rows):
    """Require one A, G and R per position."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["group"],
            row["position_id"],
            row["camera_name"],
        )
        groups[key].append(row["method"])

    for key, methods in groups.items():
        if Counter(methods) != Counter(METHODS):
            raise ValueError(f"Invalid method triplet: {key}")

    return len(groups)


def summarize_results(rows):
    """Summarize error and outcome rates by group and method."""

    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["group"], row["method"])
        ].append(row)

    summaries = []

    for (group_name, method), records in sorted(groups.items()):

        successful = [
            row for row in records
            if int(row["success"]) == 1
        ]

        errors = np.array([
            float(row["error_3d_mm"])
            for row in successful
        ], dtype=float)

        counts = Counter(
            row["outcome"] for row in records
        )

        n = len(records)

        summaries.append({
            "group": group_name,
            "method": method,
            "cases": n,
            "accurate": counts["accurate"],
            "inaccurate": counts["inaccurate"],
            "rejected": counts["rejected"],
            "accurate_pct": 100 * counts["accurate"] / n,
            "inaccurate_pct": 100 * counts["inaccurate"] / n,
            "rejected_pct": 100 * counts["rejected"] / n,
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
        })

    return summaries


def save_csv(path, records):
    """Save rows with a header."""

    if not records:
        raise ValueError("No records to save")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    overhead_records = load_overhead_records()

    results = []

    for record in overhead_records:
        scene = load_scene(record)

        results.extend(
            evaluate_scene(record, scene)
        )

    scenarios = validate_triplets(results)

    if scenarios != 33 or len(results) != 99:
        raise AssertionError("Unexpected evaluation count")

    summaries = summarize_results(results)

    # Verify development reference against previous clean results.
    for method, expected in (
        ("A", 4.903),
        ("G", 0.463),
        ("R", 0.463),
    ):
        summary = next(
            row for row in summaries
            if row["group"] == "development"
            and row["method"] == method
        )

        if (
            summary["rejected"] != 0
            or not np.isclose(
                summary["mean_error_mm"],
                expected,
                atol=0.02,
            )
        ):
            raise AssertionError(
                f"Development reference mismatch: {method}"
            )

    save_csv(RESULT_CSV, results)
    save_csv(SUMMARY_CSV, summaries)

    print("Development reference reproduction: PASSED")
    print("A/G/R paired evaluation: PASSED")
    print(f"Overhead scenes: {scenarios}")
    print(f"Localization evaluations: {len(results)}")

    print(
        "\nGroup         | Method | Cases | "
        "Mean | P90 | Max | Accurate % | "
        "Inaccurate % | Rejected %"
    )
    print("-" * 105)

    def fmt(value):
        return (
            f"{value:.3f}"
            if isinstance(value, (int, float, np.number))
            else "N/A"
        )

    for row in summaries:
        print(
            f"{row['group']:<13} | "
            f"{row['method']:^6} | "
            f"{row['cases']:5d} | "
            f"{fmt(row['mean_error_mm']):>6} | "
            f"{fmt(row['p90_error_mm']):>6} | "
            f"{fmt(row['max_error_mm']):>6} | "
            f"{row['accurate_pct']:10.1f} | "
            f"{row['inaccurate_pct']:12.1f} | "
            f"{row['rejected_pct']:10.1f}"
        )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
