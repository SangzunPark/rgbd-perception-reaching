
"""Tests for held-out depth robustness evaluation."""

import numpy as np
import pytest

from src.evaluate_generalization_robustness import (
    CONDITIONS,
    corrupt_scene_depth,
    summarize,
    validate_corruption_triplets,
)


def make_scene():
    return {
        "depth": np.ones((9, 9), dtype=float),
        "rgb_mask": np.ones((9, 9), dtype=bool),
    }


def make_row(method, outcome, error=""):
    return {
        "group": "interpolation",
        "position_id": "interpolation_01",
        "camera_name": "oblique_x",
        "condition": "gaussian_3mm",
        "repeat": 0,
        "method": method,
        "success": int(outcome != "rejected"),
        "outcome": outcome,
        "error_3d_mm": error,
        "selected_pixels": 10,
    }


def test_expected_conditions():
    assert len(CONDITIONS) == 7

    repeats = sum(item[3] for item in CONDITIONS)
    assert repeats == 43


def test_clean_depth_unchanged():
    scene = make_scene()

    result = corrupt_scene_depth(
        scene, "clean", 0.0
    )

    assert np.array_equal(result, scene["depth"])
    assert result is not scene["depth"]


def test_gaussian_corruption_reproducible():
    scene = make_scene()

    a = corrupt_scene_depth(
        scene, "gaussian", 0.003, seed=123
    )

    b = corrupt_scene_depth(
        scene, "gaussian", 0.003, seed=123
    )

    assert np.array_equal(a, b)


def test_random_dropout_reproducible():
    scene = make_scene()

    a = corrupt_scene_depth(
        scene, "dropout", 0.8, seed=321
    )

    b = corrupt_scene_depth(
        scene, "dropout", 0.8, seed=321
    )

    assert np.array_equal(a, b, equal_nan=True)
    assert np.isnan(a).any()


def test_centered_patch_dropout():
    scene = make_scene()

    result = corrupt_scene_depth(
        scene, "patch", 2
    )

    assert np.isnan(result).sum() == 25
    assert np.isfinite(scene["depth"]).all()


def test_complete_method_triplets():
    rows = [
        make_row("A", "accurate", 1.0),
        make_row("G", "inaccurate", 10.0),
        make_row("R", "rejected"),
    ]

    assert validate_corruption_triplets(rows) == 1

    with pytest.raises(ValueError):
        validate_corruption_triplets(rows[:2])

    with pytest.raises(ValueError):
        validate_corruption_triplets(rows + [rows[0]])


def test_summary_distinguishes_inaccurate_and_rejected():
    rows = [
        make_row("R", "accurate", 2.0),
        make_row("R", "inaccurate", 8.0),
        make_row("R", "rejected"),
    ]

    result = summarize(
        rows,
        ("group", "camera_name", "method"),
    )[0]

    assert result["cases"] == 3
    assert result["accurate"] == 1
    assert result["inaccurate"] == 1
    assert result["rejected"] == 1

    assert result["mean_error_mm"] == pytest.approx(5.0)
    assert result["accurate_pct"] == pytest.approx(100 / 3)
    assert result["inaccurate_pct"] == pytest.approx(100 / 3)
    assert result["rejected_pct"] == pytest.approx(100 / 3)
