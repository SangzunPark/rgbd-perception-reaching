
import pytest

from src.evaluate_robust_methods import (
    summarize,
    validate_triplets,
)


def make_row(method, success, error, repeat=0):
    return {
        "experiment": "gaussian",
        "condition": "sigma_3mm",
        "repeat": repeat,
        "trial": "trial_01",
        "method": method,
        "success": success,
        "error_3d_mm": error if success else "",
        "within_5mm": int(
            bool(success) and error <= 5.0
        ),
        "selected_pixels": 100 if success else 0,
    }


def test_complete_method_triplet():
    rows = [
        make_row("A", 1, 4.0),
        make_row("G", 1, 8.0),
        make_row("R", 1, 1.0),
    ]

    assert validate_triplets(rows) == 1


def test_missing_method_rejected():
    rows = [
        make_row("A", 1, 4.0),
        make_row("G", 1, 8.0),
    ]

    with pytest.raises(ValueError):
        validate_triplets(rows)


def test_duplicate_method_rejected():
    rows = [
        make_row("A", 1, 4.0),
        make_row("A", 1, 4.0),
        make_row("G", 1, 8.0),
        make_row("R", 1, 1.0),
    ]

    with pytest.raises(ValueError):
        validate_triplets(rows)


def test_accuracy_rate_counts_failures():
    rows = [
        make_row("R", 1, 2.0, repeat=0),
        make_row("R", 1, 7.0, repeat=1),
        make_row("R", 0, 0.0, repeat=2),
    ]

    summary = summarize(rows)[0]

    assert summary["cases"] == 3
    assert summary["failures"] == 1

    assert summary["failure_rate_pct"] == pytest.approx(
        100 / 3
    )

    assert summary["within_5mm_pct"] == pytest.approx(
        100 / 3
    )

    assert summary["mean_error_mm"] == pytest.approx(
        4.5
    )
