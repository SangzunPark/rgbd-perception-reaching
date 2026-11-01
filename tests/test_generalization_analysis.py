
"""Tests for generalization analysis and failure diagnostics."""

import numpy as np
import pytest

from src.analyze_generalization import (
    outcome_percentages,
    robust_inlier_mask,
    select_patch_failure_case,
    surface_details,
    gt_image_geometry,
)
from src.camera_geometry import compute_intrinsics
from src.robust_localization import select_robust_surface


def test_outcome_percentages():
    rows = [
        {"outcome": "accurate"},
        {"outcome": "accurate"},
        {"outcome": "inaccurate"},
        {"outcome": "rejected"},
    ]

    result = outcome_percentages(rows)

    assert result["accurate"] == pytest.approx(50.0)
    assert result["inaccurate"] == pytest.approx(25.0)
    assert result["rejected"] == pytest.approx(25.0)
    assert sum(result.values()) == pytest.approx(100.0)


def test_empty_or_invalid_outcomes():
    with pytest.raises(ValueError):
        outcome_percentages([])

    with pytest.raises(ValueError):
        outcome_percentages([{"outcome": "unknown"}])


def test_robust_inlier_mask_matches_estimator():
    mask = np.ones((9, 9), dtype=bool)
    depth = np.ones((9, 9), dtype=float)

    # A closer-depth outlier should not dominate R.
    depth[0, 0] = 0.90

    inliers = robust_inlier_mask(mask, depth)

    u, v, z, diagnostics = select_robust_surface(
        mask, depth
    )

    assert inliers.sum() == 80
    assert diagnostics["inlier_pixels"] == 80
    assert z == pytest.approx(1.0)
    assert u == pytest.approx(4.05)
    assert v == pytest.approx(4.05)


def test_surface_details_uses_frozen_estimators():
    depth = np.ones((9, 9), dtype=float)
    mask = np.ones((9, 9), dtype=bool)

    scene = {
        "depth": depth,
        "rgb_mask": mask,
    }

    details = surface_details(scene)

    assert details["G"]["mask"].sum() == 81
    assert details["R"]["mask"].sum() == 81
    assert details["G"]["z"] == pytest.approx(1.0)
    assert details["R"]["z"] == pytest.approx(1.0)


def test_gt_optical_depth():
    # Identity OpenGL view; World z=-1 means optical depth=1.
    scene = {
        "gt": np.array([0.0, 0.0, -1.0]),
        "K": compute_intrinsics(320, 240, 60),
        "view": np.eye(4).reshape(-1, order="F"),
    }

    u, v, depth = gt_image_geometry(scene)

    assert u == pytest.approx(160.0)
    assert v == pytest.approx(120.0)
    assert depth == pytest.approx(1.0)


def test_select_largest_inaccurate_patch_case():
    rows = [
        {
            "method": "R",
            "camera_name": "oblique_diagonal",
            "condition": "patch_11x11",
            "outcome": "inaccurate",
            "error_3d_mm": "12.0",
        },
        {
            "method": "R",
            "camera_name": "oblique_diagonal",
            "condition": "patch_11x11",
            "outcome": "inaccurate",
            "error_3d_mm": "19.0",
        },
        {
            "method": "R",
            "camera_name": "oblique_x",
            "condition": "patch_11x11",
            "outcome": "inaccurate",
            "error_3d_mm": "30.0",
        },
    ]

    chosen = select_patch_failure_case(rows)

    assert float(chosen["error_3d_mm"]) == pytest.approx(19.0)
    assert chosen["camera_name"] == "oblique_diagonal"
