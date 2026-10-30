
import numpy as np
import pytest

from src.visualize_trials import (
    build_error_grid,
    summarize_errors,
)


def make_test_rows():
    """Synthetic 2x2 experiment with known errors."""
    return [
        {
            "configured_x_m": "0.10",
            "configured_y_m": "-0.05",
            "single_error_mm": "4.0",
        },
        {
            "configured_x_m": "-0.10",
            "configured_y_m": "0.05",
            "single_error_mm": "1.0",
        },
        {
            "configured_x_m": "0.10",
            "configured_y_m": "0.05",
            "single_error_mm": "2.0",
        },
        {
            "configured_x_m": "-0.10",
            "configured_y_m": "-0.05",
            "single_error_mm": "3.0",
        },
    ]


def test_error_grid_position_order():
    rows = make_test_rows()

    xs, ys, grid = build_error_grid(rows)

    assert xs == [-0.10, 0.10]
    assert ys == [0.05, -0.05]

    expected = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    np.testing.assert_allclose(grid, expected)


def test_summary_statistics():
    rows = make_test_rows()

    summary = summarize_errors(rows)

    assert summary["mean"] == pytest.approx(2.5)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)


def test_incomplete_grid():
    rows = make_test_rows()[:-1]

    with pytest.raises(ValueError):
        build_error_grid(rows)


def test_duplicate_position():
    rows = make_test_rows()
    rows.append(rows[0].copy())

    with pytest.raises(ValueError):
        build_error_grid(rows)
