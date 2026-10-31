
import pytest

from src.evaluate_gaussian_robustness import (
    summarize_results,
    validate_pairs,
)


def make_row(repeat, method, success, error):
    return {
        "sigma_mm": 3.0,
        "repeat": repeat,
        "trial": "trial_01",
        "method": method,
        "success": success,
        "error_3d_mm": error if success else "",
    }


def test_valid_paired_evaluation():
    rows = [
        make_row(0, "A", 1, 2.0),
        make_row(0, "G", 1, 1.0),
        make_row(1, "A", 1, 4.0),
        make_row(1, "G", 0, ""),
    ]

    assert validate_pairs(rows) == 2


def test_missing_method_rejected():
    rows = [
        make_row(0, "A", 1, 2.0),
    ]

    with pytest.raises(ValueError):
        validate_pairs(rows)


def test_summary_includes_failures():
    rows = [
        make_row(0, "A", 1, 2.0),
        make_row(0, "G", 1, 1.0),
        make_row(1, "A", 1, 4.0),
        make_row(1, "G", 0, ""),
    ]

    summary = summarize_results(rows)

    a = next(r for r in summary if r["method"] == "A")
    g = next(r for r in summary if r["method"] == "G")

    assert a["cases"] == 2
    assert a["mean_error_mm"] == pytest.approx(3.0)
    assert a["success_rate_pct"] == pytest.approx(100.0)

    assert g["cases"] == 2
    assert g["failures"] == 1
    assert g["success_rate_pct"] == pytest.approx(50.0)
    assert g["mean_error_mm"] == pytest.approx(1.0)
