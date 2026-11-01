
"""Evaluate depth-corruption robustness on held-out camera scenes."""

import csv
from collections import Counter, defaultdict

import numpy as np

from src.depth_corruption import (
    add_gaussian_depth_noise,
    apply_random_depth_dropout,
    apply_depth_patch_dropout,
)
from src.evaluate_position_generalization import (
    METHODS,
    ROOT,
    evaluate_scene,
    load_scene,
    save_csv,
)
from src.evaluate_viewpoint_generalization import (
    load_manifest,
)
from src.object_perception import select_mask_center_pixel


OUTPUT = ROOT / "outputs"

RESULT_CSV = OUTPUT / "generalization_robustness_results.csv"
SUMMARY_CSV = OUTPUT / "generalization_robustness_summary.csv"
CAMERA_SUMMARY_CSV = (
    OUTPUT / "generalization_robustness_camera_summary.csv"
)

VIEWPOINT_RESULT_CSV = (
    OUTPUT / "viewpoint_generalization_results.csv"
)

# (name, corruption kind, strength, repetitions)
CONDITIONS = (
    ("clean", "clean", 0.0, 1),
    ("gaussian_3mm", "gaussian", 0.003, 10),
    ("gaussian_5mm", "gaussian", 0.005, 10),
    ("dropout_40", "dropout", 0.40, 10),
    ("dropout_80", "dropout", 0.80, 10),
    ("patch_5x5", "patch", 2, 1),
    ("patch_11x11", "patch", 5, 1),
)

HELDOUT_GROUPS = ("interpolation", "extrapolation")


def corrupt_scene_depth(scene, kind, strength, seed=None):
    """Corrupt a copy of the depth map, without using GT."""

    depth = scene["depth"]

    if kind == "clean":
        return depth.copy()

    if kind in ("gaussian", "dropout"):
        if seed is None:
            raise ValueError("Random corruption requires a seed")

        rng = np.random.default_rng(seed)

        if kind == "gaussian":
            return add_gaussian_depth_noise(
                depth,
                sigma_m=float(strength),
                rng=rng,
            )

        return apply_random_depth_dropout(
            depth,
            probability=float(strength),
            rng=rng,
        )

    if kind == "patch":
        u, v = select_mask_center_pixel(
            scene["rgb_mask"]
        )

        return apply_depth_patch_dropout(
            depth,
            center_u=u,
            center_v=v,
            radius=int(strength),
        )

    raise ValueError(f"Unknown corruption type: {kind}")


def validate_corruption_triplets(rows):
    """Require one A/G/R result per corrupted input."""

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["group"],
            row["position_id"],
            row["camera_name"],
            row["condition"],
            int(row["repeat"]),
        )

        groups[key].append(row["method"])

    for key, methods in groups.items():
        if Counter(methods) != Counter(METHODS):
            raise ValueError(
                f"Incomplete or duplicate method triplet: {key}"
            )

    return len(groups)


def summarize(rows, group_fields):
    """Summarize all outcomes, including explicit rejection."""

    grouped = defaultdict(list)

    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summaries = []

    for key, block in sorted(grouped.items()):
        counts = Counter(
            row["outcome"] for row in block
        )

        errors = np.array([
            float(row["error_3d_mm"])
            for row in block
            if int(row["success"]) == 1
        ], dtype=float)

        total = len(block)

        if (
            counts["accurate"]
            + counts["inaccurate"]
            + counts["rejected"]
            != total
        ):
            raise ValueError("Invalid outcome counts")

        item = dict(zip(group_fields, key))

        item.update({
            "cases": total,
            "accurate": counts["accurate"],
            "inaccurate": counts["inaccurate"],
            "rejected": counts["rejected"],
            "accurate_pct": 100 * counts["accurate"] / total,
            "inaccurate_pct": 100 * counts["inaccurate"] / total,
            "rejected_pct": 100 * counts["rejected"] / total,
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
                for row in block
            ])),
        })

        summaries.append(item)

    return summaries


def validate_clean_reproduction(rows, reference_path):
    """Verify clean held-out results reproduce Step 3."""

    with reference_path.open(
        newline="", encoding="utf-8"
    ) as f:
        reference = list(csv.DictReader(f))

    def key(row):
        return (
            row["group"],
            row["position_id"],
            row["camera_name"],
            row["method"],
        )

    expected = {
        key(row): row
        for row in reference
        if row["group"] in HELDOUT_GROUPS
    }

    clean = [
        row for row in rows
        if row["condition"] == "clean"
    ]

    actual = {key(row): row for row in clean}

    if (
        len(expected) != 288
        or len(actual) != 288
        or len(clean) != 288
        or expected.keys() != actual.keys()
    ):
        raise ValueError("Clean evaluation keys mismatch")

    for item_key in expected:
        a = actual[item_key]
        b = expected[item_key]

        if (
            int(a["success"]) != int(b["success"])
            or a["outcome"] != b["outcome"]
        ):
            raise ValueError(
                f"Clean outcome mismatch: {item_key}"
            )

        if int(a["success"]) == 1:
            if not np.isclose(
                float(a["error_3d_mm"]),
                float(b["error_3d_mm"]),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"Clean error mismatch: {item_key}"
                )

    return len(actual)


def main():
    manifest = load_manifest()

    heldout = [
        record for record in manifest
        if record["group"] in HELDOUT_GROUPS
    ]

    if len(heldout) != 96:
        raise AssertionError("Expected 96 held-out scenes")

    results = []

    for scene_index, record in enumerate(heldout):
        scene = load_scene(record)

        # Work through conditions while keeping this scene loaded.
        for condition_index, (
            condition, kind, strength, repeats
        ) in enumerate(CONDITIONS):

            for repeat in range(repeats):
                # A distinct deterministic seed per input scenario.
                seed = (
                    20261101
                    + condition_index * 1_000_000
                    + repeat * 1_000
                    + scene_index
                )

                corrupted = corrupt_scene_depth(
                    scene,
                    kind=kind,
                    strength=strength,
                    seed=seed,
                )

                # Never modify the original scene in place.
                damaged_scene = {
                    **scene,
                    "depth": corrupted,
                }

                # All three methods receive the SAME corrupted depth.
                evaluations = evaluate_scene(
                    record,
                    damaged_scene,
                )

                for result in evaluations:
                    results.append({
                        "condition": condition,
                        "repeat": repeat,
                        "seed": (
                            seed
                            if kind in ("gaussian", "dropout")
                            else ""
                        ),
                        **result,
                    })

        if (scene_index + 1) % 16 == 0:
            print(
                f"Processed held-out scenes: "
                f"{scene_index + 1}/96",
                flush=True,
            )

    scenarios = validate_corruption_triplets(results)

    if scenarios != 4128 or len(results) != 12384:
        raise AssertionError(
            "Unexpected robustness evaluation count"
        )

    reproduced = validate_clean_reproduction(
        results,
        VIEWPOINT_RESULT_CSV,
    )

    grouped_summary = summarize(
        results,
        ("group", "camera_name", "condition", "method"),
    )

    camera_summary = summarize(
        results,
        ("camera_name", "condition", "method"),
    )

    if len(grouped_summary) != 168:
        raise AssertionError("Expected 168 grouped summaries")

    if len(camera_summary) != 84:
        raise AssertionError("Expected 84 camera summaries")

    save_csv(RESULT_CSV, results)
    save_csv(SUMMARY_CSV, grouped_summary)
    save_csv(CAMERA_SUMMARY_CSV, camera_summary)

    print("\nClean Step 3 reproduction: PASSED")
    print(f"Clean A/G/R results verified: {reproduced}")
    print("Corrupted A/G/R triplets: PASSED")
    print(f"Input scenarios: {scenarios}")
    print(f"Localization evaluations: {len(results)}")

    print(
        "\nMethod R by camera and condition"
        "\nCamera | Condition | Cases | Mean | P90 | "
        "Accurate% | Inaccurate% | Rejected%"
    )
    print("-" * 96)

    def fmt(value):
        return (
            f"{float(value):.3f}"
            if value != ""
            else "N/A"
        )

    for row in camera_summary:
        if row["method"] != "R":
            continue

        print(
            f"{row['camera_name']:<18} "
            f"{row['condition']:<14} "
            f"{row['cases']:>4} "
            f"{fmt(row['mean_error_mm']):>7} "
            f"{fmt(row['p90_error_mm']):>7} "
            f"{row['accurate_pct']:>9.1f} "
            f"{row['inaccurate_pct']:>11.1f} "
            f"{row['rejected_pct']:>9.1f}"
        )

    print(f"\nSaved: {RESULT_CSV}")
    print(f"Saved: {SUMMARY_CSV}")
    print(f"Saved: {CAMERA_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
