"""Unit tests for diagnostic support geometry (no dataset required)."""

import numpy as np
import pytest

from src.diagnose_surface_coverage import support_metrics, summarize_cases


def test_compact_surface_support():
    obj = np.ones((9, 9), dtype=bool)
    inliers = np.zeros((9, 9), dtype=bool)
    inliers[2:7, 2:7] = True

    metrics = support_metrics(obj, inliers)
    assert metrics["inlier_pixels"] == 25
    assert metrics["span_u_fraction"] == pytest.approx(5 / 9)
    assert metrics["span_v_fraction"] == pytest.approx(5 / 9)
    assert metrics["bbox_fill_fraction"] == pytest.approx(1.0)
    assert metrics["occupied_quadrants"] == 4
    assert metrics["centroid_shift_from_rgb_px"] == pytest.approx(0)


def test_l_shaped_pixels_have_large_spans_but_low_fill():
    obj = np.ones((9, 9), dtype=bool)
    inliers = np.zeros((9, 9), dtype=bool)
    inliers[0, :] = True
    inliers[:, 8] = True

    metrics = support_metrics(obj, inliers)
    assert metrics["inlier_pixels"] == 17
    assert metrics["span_u_fraction"] == pytest.approx(1.0)
    assert metrics["span_v_fraction"] == pytest.approx(1.0)
    assert metrics["bbox_fill_fraction"] == pytest.approx(17 / 81)
    assert metrics["occupied_quadrants"] == 3
    assert metrics["centroid_shift_from_rgb_px"] > 0


def test_empty_support_has_no_centroid():
    obj = np.ones((4, 4), dtype=bool)
    inliers = np.zeros((4, 4), dtype=bool)
    metrics = support_metrics(obj, inliers)
    assert metrics["inlier_pixels"] == 0
    assert metrics["span_u_fraction"] == 0.0
    assert metrics["candidate_u"] == ""
    assert metrics["candidate_v"] == ""


def test_support_cannot_extend_outside_object():
    obj = np.zeros((4, 4), dtype=bool)
    obj[1:3, 1:3] = True
    inliers = np.zeros((4, 4), dtype=bool)
    inliers[0, 0] = True
    with pytest.raises(ValueError):
        support_metrics(obj, inliers)


def test_rejections_have_separate_summary_counts():
    common = {
        "camera_name": "oblique_diagonal",
        "valid_object_pixels": 12,
        "inlier_pixels": 10,
        "span_u_fraction": 0.6,
        "span_v_fraction": 0.6,
        "bbox_fill_fraction": 0.3,
        "occupied_quadrants": 2,
        "centroid_shift_from_rgb_px": 3.0,
        "candidate_gt_center_error_px": 4.0,
    }
    rows = [
        {**common, "outcome": "inaccurate"},
        {**common, "outcome": "rejected"},
        {**common, "outcome": "rejected"},
    ]
    summary = summarize_cases(rows)
    counts = {r["outcome"]: r["cases"] for r in summary}
    assert counts == {"inaccurate": 1, "rejected": 2}
