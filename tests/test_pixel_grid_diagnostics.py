"""Day 8 Step 5-D unit tests for pixel-grid and centroid diagnostics."""

import csv

import numpy as np
import pytest

from src.camera_geometry import compute_intrinsics
from src.diagnose_pixel_grid_bias import (
    gt_optical_depth,
    nearest_grid_reference,
    pixel_phase,
    summarize,
    validate_reference,
    world_error_mm,
    world_from_pixel_depth,
)


def test_backprojection_of_center_pixel():
    K = compute_intrinsics(320, 240, 60)
    view = np.eye(4).reshape(-1, order="F")
    result = world_from_pixel_depth(160, 120, 1.0, K, view)
    np.testing.assert_allclose(result, [0, 0, -1], atol=1e-12)
    assert gt_optical_depth(np.array([0, 0, -1]), view) == pytest.approx(1)


def test_nearest_grid_reference_respects_v_offset():
    assert nearest_grid_reference(160.49, 120.49) == (160, 120)
    assert nearest_grid_reference(160.51, 120.51) == (161, 121)
    assert nearest_grid_reference(160.49, 120.49, offset_px=0) == (160, 120)


def test_pixel_phases_are_in_half_pixel_range():
    for u in np.linspace(100.02, 220.98, 31):
        for v in (101.1, 120.25, 157.9):
            pu, pv = pixel_phase(u, v)
            assert -0.5 <= pu <= 0.5
            assert -0.5 <= pv <= 0.5


def test_counterfactual_image_and_depth_errors():
    K = compute_intrinsics(320, 240, 60)
    view = np.eye(4).reshape(-1, order="F")
    gt = np.array([0.0, 0.0, -1.0])

    image_only = world_from_pixel_depth(160.5, 120.5, 1.0, K, view)
    depth_only = world_from_pixel_depth(160, 120, 1.01, K, view)

    expected_image_mm = np.hypot(0.5 / K[0, 0], 0.5 / K[1, 1]) * 1000
    assert world_error_mm(image_only, gt) == pytest.approx(expected_image_mm)
    assert world_error_mm(depth_only, gt) == pytest.approx(10.0)


def test_reference_reproduction_rejects_mismatch(tmp_path):
    # 33 distinct R reference rows, exactly as the frozen evaluation expects.
    groups = (("development", 9), ("interpolation", 16), ("extrapolation", 8))
    rows = []
    reference_rows = []
    for group, n in groups:
        for i in range(1, n + 1):
            name = f"{group}_{i:02d}"
            rows.append({"group": group, "position_id": name, "actual_error_mm": 2.0})
            reference_rows.append({
                "group": group, "position_id": name, "method": "R",
                "camera_name": "overhead", "error_3d_mm": "2.0",
            })

    path = tmp_path / "reference.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)

    assert validate_reference(rows, path) == 33
    rows[0]["actual_error_mm"] = 2.1
    with pytest.raises(AssertionError):
        validate_reference(rows, path)


def test_summary_rejects_missing_group():
    with pytest.raises(ValueError):
        summarize([{"group": "development"}])
