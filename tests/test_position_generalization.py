
import numpy as np
import pytest

from src.evaluate_position_generalization import (
    classify_outcome,
    summarize_results,
    validate_triplets,
)


def make_row(method, outcome, error="", position_id="p01"):
    return {
        "group": "interpolation",
        "position_id": position_id,
        "camera_name": "overhead",
        "method": method,
        "success": int(outcome != "rejected"),
        "outcome": outcome,
        "error_3d_mm": error,
    }


def test_outcome_classification():
    assert classify_outcome(1, 2.0) == "accurate"
    assert classify_outcome(1, 5.0) == "accurate"
    assert classify_outcome(1, 5.1) == "inaccurate"
    assert classify_outcome(0, np.nan) == "rejected"


def test_nonfinite_success_error_rejected():
    with pytest.raises(ValueError):
        classify_outcome(1, np.nan)


def test_complete_triplets():
    rows = [
        make_row("A", "accurate", 2.0),
        make_row("G", "inaccurate", 8.0),
        make_row("R", "rejected"),
    ]

    assert validate_triplets(rows) == 1

    with pytest.raises(ValueError):
        validate_triplets(rows[:2])


def test_summary_counts_all_outcomes():
    rows = [
        make_row("R", "accurate", 2.0, "p01"),
        make_row("R", "inaccurate", 8.0, "p02"),
        make_row("R", "rejected", "", "p03"),
    ]

    summary = summarize_results(rows)[0]

    assert summary["cases"] == 3
    assert summary["accurate"] == 1
    assert summary["inaccurate"] == 1
    assert summary["rejected"] == 1

    assert summary["accurate_pct"] == pytest.approx(100 / 3)
    assert summary["mean_error_mm"] == pytest.approx(5.0)
    assert summary["p90_error_mm"] == pytest.approx(7.4)
    assert summary["max_error_mm"] == pytest.approx(8.0)
