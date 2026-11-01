
"""Tests for viewpoint comparison logic."""

from copy import deepcopy

import pytest

from src.evaluate_viewpoint_generalization import (
    build_paired_deltas,
    summarize_viewpoints,
    validate_overhead_reproduction,
    validate_viewpoint_pairs,
)
from src.generate_generalization_scenes import (
    CAMERA_DIRECTIONS,
)


def example_rows():
    """Synthetic results for one position and four cameras."""

    rows = []

    for camera in CAMERA_DIRECTIONS:
        for method in ("A", "G", "R"):

            error = (
                1.0 if camera == "overhead"
                else 3.0
            )

            rows.append({
                "group": "interpolation",
                "position_id": "interpolation_01",
                "camera_name": camera,
                "method": method,
                "success": 1,
                "outcome": "accurate",
                "error_3d_mm": error,
            })

    return rows


def test_valid_camera_method_grid():
    rows = example_rows()

    assert validate_viewpoint_pairs(rows) == 1

    with pytest.raises(ValueError):
        validate_viewpoint_pairs(rows[:-1])

    with pytest.raises(ValueError):
        validate_viewpoint_pairs(rows + [rows[0]])


def test_viewpoint_paired_error_differences():
    paired = build_paired_deltas(example_rows())

    assert len(paired) == 9

    assert all(
        row["pair_status"] == "both_returned"
        for row in paired
    )

    assert all(
        row["delta_error_mm"] == pytest.approx(2.0)
        for row in paired
    )


def test_rejections_not_treated_as_numeric_error():
    rows = example_rows()

    row = next(
        item for item in rows
        if item["camera_name"] == "oblique_x"
        and item["method"] == "R"
    )

    row.update(
        success=0,
        outcome="rejected",
        error_3d_mm="",
    )

    pair = next(
        item for item in build_paired_deltas(rows)
        if item["camera_name"] == "oblique_x"
        and item["method"] == "R"
    )

    assert pair["pair_status"] == "view_rejected"
    assert pair["delta_error_mm"] == ""


def test_viewpoint_summary_counts_rejections():
    rows = example_rows()

    row = next(
        item for item in rows
        if item["camera_name"] == "oblique_x"
        and item["method"] == "R"
    )

    row.update(
        success=0,
        outcome="rejected",
        error_3d_mm="",
    )

    summary = next(
        item for item in summarize_viewpoints(rows)
        if item["camera_name"] == "oblique_x"
        and item["method"] == "R"
    )

    assert summary["cases"] == 1
    assert summary["rejected_pct"] == pytest.approx(100.0)
    assert summary["mean_error_mm"] == ""


def test_overhead_reproduction_detects_changes():
    full = example_rows()

    reference = deepcopy([
        row for row in full
        if row["camera_name"] == "overhead"
    ])

    assert validate_overhead_reproduction(
        full, reference
    ) == 3

    reference[0]["error_3d_mm"] = 9.0

    with pytest.raises(ValueError):
        validate_overhead_reproduction(
            full, reference
        )
