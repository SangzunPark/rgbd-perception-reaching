
import pytest

from src.evaluate_missing_depth import (
    validate_pairs,
    summarize_results,
)


def make_row(method, repeat, success, error, missing):
    return {
        "condition": "dropout_20",
        "repeat": repeat,
        "trial": "trial_01",
        "method": method,
        "success": success,
        "error_3d_mm": error if success else "",
        "missing_object_pct": missing,
        "selected_pixels": 10 if success else 0,
    }


def test_valid_pairs():
    rows = [
        make_row("A", 0, 1, 2.0, 20.0),
        make_row("G", 0, 1, 1.0, 20.0),
        make_row("A", 1, 1, 3.0, 25.0),
        make_row("G", 1, 0, "", 25.0),
    ]

    assert validate_pairs(rows) == 2


def test_missing_pair_rejected():
    rows = [
        make_row("A", 0, 1, 2.0, 20.0),
    ]

    with pytest.raises(ValueError):
        validate_pairs(rows)


def test_summary_counts_failures():
    rows = [
        make_row("A", 0, 1, 2.0, 20.0),
        make_row("G", 0, 1, 1.0, 20.0),
        make_row("A", 1, 1, 4.0, 25.0),
        make_row("G", 1, 0, "", 25.0),
    ]

    summary = summarize_results(rows)

    a = next(s for s in summary if s["method"] == "A")
    g = next(s for s in summary if s["method"] == "G")

    assert a["mean_error_mm"] == pytest.approx(3.0)
    assert a["failures"] == 0

    assert g["failures"] == 1
    assert g["success_rate_pct"] == pytest.approx(50.0)
    assert g["mean_error_mm"] == pytest.approx(1.0)

    assert g["mean_missing_object_pct"] == pytest.approx(22.5)


def test_all_failures_have_no_mean_error():
    rows = [
        make_row("G", 0, 0, "", 100.0),
        make_row("G", 1, 0, "", 100.0),
    ]

    summary = summarize_results(rows)

    assert summary[0]["failures"] == 2
    assert summary[0]["mean_error_mm"] == ""
    assert summary[0]["success_rate_pct"] == 0
