
"""Evaluate frozen RGB-D localization under viewpoint changes."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.evaluate_position_generalization import (
    METHODS,
    ROOT,
    evaluate_scene,
    load_scene,
    save_csv,
    validate_triplets,
)
from src.generate_generalization_scenes import (
    BENCHMARK_ROOT,
    CAMERA_DIRECTIONS,
    POSITION_GROUPS,
)


MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"

RESULT_CSV = ROOT / "outputs" / "viewpoint_generalization_results.csv"
SUMMARY_CSV = ROOT / "outputs" / "viewpoint_generalization_summary.csv"
PAIRED_CSV = ROOT / "outputs" / "viewpoint_paired_deltas.csv"

POSITION_CSV = ROOT / "outputs" / "position_generalization_results.csv"


def load_manifest(path=MANIFEST_PATH):
    """Require the complete 33-position by 4-viewpoint design."""

    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)

    expected = {
        (group, f"{group}_{i:02d}", camera)
        for group, positions in POSITION_GROUPS.items()
        for i in range(1, len(positions) + 1)
        for camera in CAMERA_DIRECTIONS
    }

    actual = [
        (r["group"], r["position_id"], r["camera_name"])
        for r in records
    ]

    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "Incomplete or duplicate viewpoint manifest"
        )

    return records


def validate_viewpoint_pairs(rows):
    """Require A/G/R for every position and camera."""

    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["group"], row["position_id"])
        ].append(
            (row["camera_name"], row["method"])
        )

    expected = {
        (camera, method)
        for camera in CAMERA_DIRECTIONS
        for method in METHODS
    }

    for key, combinations in groups.items():
        if (
            len(combinations) != len(expected)
            or set(combinations) != expected
        ):
            raise ValueError(
                f"Missing or duplicate camera-method pair: {key}"
            )

    return len(groups)


def summarize_viewpoints(rows):
    """Summarize outcomes by position group, camera, method."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["group"],
            row["camera_name"],
            row["method"],
        )
        groups[key].append(row)

    summaries = []

    for group in POSITION_GROUPS:
        for camera in CAMERA_DIRECTIONS:
            for method in METHODS:

                block = groups[(group, camera, method)]

                if not block:
                    continue

                counts = Counter(
                    row["outcome"] for row in block
                )

                n = len(block)

                errors = np.array([
                    float(row["error_3d_mm"])
                    for row in block
                    if int(row["success"]) == 1
                ], dtype=float)

                if (
                    counts["accurate"]
                    + counts["inaccurate"]
                    + counts["rejected"]
                    != n
                ):
                    raise ValueError(
                        "Invalid outcome classification"
                    )

                summaries.append({
                    "group": group,
                    "camera_name": camera,
                    "method": method,
                    "cases": n,
                    "accurate": counts["accurate"],
                    "inaccurate": counts["inaccurate"],
                    "rejected": counts["rejected"],
                    "accurate_pct": 100 * counts["accurate"] / n,
                    "inaccurate_pct": 100 * counts["inaccurate"] / n,
                    "rejected_pct": 100 * counts["rejected"] / n,
                    "mean_error_mm": (
                        float(errors.mean())
                        if errors.size else ""
                    ),
                    "p90_error_mm": (
                        float(np.percentile(errors, 90))
                        if errors.size else ""
                    ),
                    "max_error_mm": (
                        float(errors.max())
                        if errors.size else ""
                    ),
                })

    return summaries


def build_paired_deltas(rows):
    """Compare each oblique camera to overhead at the same position."""

    by_position = defaultdict(dict)

    for row in rows:
        key = (
            row["group"],
            row["position_id"],
            row["method"],
        )

        camera = row["camera_name"]

        if camera in by_position[key]:
            raise ValueError(
                f"Duplicate viewpoint: {key}, {camera}"
            )

        by_position[key][camera] = row

    paired = []
    expected_views = set(CAMERA_DIRECTIONS)

    for (group, position_id, method), views in by_position.items():

        if set(views) != expected_views:
            raise ValueError(
                f"Incomplete viewpoint pairing: "
                f"{position_id}, {method}"
            )

        baseline = views["overhead"]
        baseline_returned = int(baseline["success"]) == 1

        for camera in CAMERA_DIRECTIONS:

            if camera == "overhead":
                continue

            sample = views[camera]
            view_returned = int(sample["success"]) == 1

            if baseline_returned and view_returned:
                status = "both_returned"

                delta = (
                    float(sample["error_3d_mm"])
                    - float(baseline["error_3d_mm"])
                )

            elif baseline_returned:
                status = "view_rejected"
                delta = ""

            elif view_returned:
                status = "overhead_rejected"
                delta = ""

            else:
                status = "both_rejected"
                delta = ""

            paired.append({
                "group": group,
                "position_id": position_id,
                "method": method,
                "camera_name": camera,
                "overhead_outcome": baseline["outcome"],
                "view_outcome": sample["outcome"],
                "overhead_error_mm": baseline["error_3d_mm"],
                "view_error_mm": sample["error_3d_mm"],
                "delta_error_mm": delta,
                "pair_status": status,
            })

    return paired


def validate_overhead_reproduction(rows, reference_rows):
    """Verify overhead predictions match frozen Step 2 results."""

    overhead = [
        row for row in rows
        if row["camera_name"] == "overhead"
    ]

    def keyed(records):
        result = {}

        for row in records:
            key = (
                row["group"],
                row["position_id"],
                row["method"],
            )

            if key in result:
                raise ValueError(
                    f"Duplicate overhead result: {key}"
                )

            result[key] = row

        return result

    actual = keyed(overhead)
    expected = keyed(reference_rows)

    if actual.keys() != expected.keys():
        raise ValueError(
            "Overhead keys differ from Step 2"
        )

    for key in actual:
        a = actual[key]
        b = expected[key]

        if (
            int(a["success"]) != int(b["success"])
            or a["outcome"] != b["outcome"]
        ):
            raise ValueError(
                f"Overhead outcome changed: {key}"
            )

        if int(a["success"]) == 1:
            if not np.isclose(
                float(a["error_3d_mm"]),
                float(b["error_3d_mm"]),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"Overhead error changed: {key}"
                )

    return len(actual)


def main():
    manifest = load_manifest()

    results = []

    # Evaluate A/G/R on the exact same RGB-D scene.
    for record in manifest:
        scene = load_scene(record)
        results.extend(
            evaluate_scene(record, scene)
        )

    # 132 scenes, 3 methods each
    if validate_triplets(results) != 132:
        raise AssertionError(
            "Expected 132 complete A/G/R triplets"
        )

    if (
        validate_viewpoint_pairs(results) != 33
        or len(results) != 396
    ):
        raise AssertionError(
            "Expected 33 positions and 396 evaluations"
        )

    # Confirm overhead results did not change.
    with POSITION_CSV.open(
        newline="", encoding="utf-8"
    ) as f:
        reference = list(csv.DictReader(f))

    reproduced = validate_overhead_reproduction(
        results, reference
    )

    if reproduced != 99:
        raise AssertionError(
            "Expected 99 reproduced overhead results"
        )

    summaries = summarize_viewpoints(results)
    paired = build_paired_deltas(results)

    if len(summaries) != 36:
        raise AssertionError("Expected 36 summary rows")

    if len(paired) != 297:
        raise AssertionError("Expected 297 viewpoint pairs")

    save_csv(RESULT_CSV, results)
    save_csv(SUMMARY_CSV, summaries)
    save_csv(PAIRED_CSV, paired)

    print(
        "Overhead Step 2 reproduction: "
        "PASSED (99 evaluations)"
    )
    print(
        "Viewpoint pairing: PASSED "
        "(33 positions, 4 views)"
    )
    print(f"Scene evaluations: {len(manifest)}")
    print(f"Localization predictions: {len(results)}")
    print(f"Oblique-vs-overhead pairs: {len(paired)}")

    print(
        "\nGroup | Camera | Method | N | Mean | P90 | "
        "Max | Accurate% | Inaccurate% | Rejected%"
    )
    print("-" * 104)

    def display(value):
        return (
            f"{float(value):.3f}"
            if value != ""
            else "N/A"
        )

    for row in summaries:
        print(
            f"{row['group']:<13} "
            f"{row['camera_name']:<18} "
            f"{row['method']:^3} "
            f"{row['cases']:>2} "
            f"{display(row['mean_error_mm']):>7} "
            f"{display(row['p90_error_mm']):>7} "
            f"{display(row['max_error_mm']):>7} "
            f"{row['accurate_pct']:>9.1f} "
            f"{row['inaccurate_pct']:>11.1f} "
            f"{row['rejected_pct']:>9.1f}"
        )

    print(
        "\nMean paired error change: "
        "Oblique - Overhead (mm)"
    )
    print(
        "Positive = higher error. "
        "Rejected pairs excluded from the mean."
    )

    for group in POSITION_GROUPS:
        for camera in CAMERA_DIRECTIONS:

            if camera == "overhead":
                continue

            for method in METHODS:
                items = [
                    row for row in paired
                    if row["group"] == group
                    and row["camera_name"] == camera
                    and row["method"] == method
                ]

                valid = [
                    float(row["delta_error_mm"])
                    for row in items
                    if row["pair_status"] == "both_returned"
                ]

                mean = (
                    f"{np.mean(valid):+.3f}"
                    if valid else "N/A"
                )

                print(
                    f"{group:<13} {camera:<18} "
                    f"{method} "
                    f"delta={mean:>7} "
                    f"paired={len(valid):>2}/{len(items)}"
                )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {PAIRED_CSV}")


if __name__ == "__main__":
    main()
