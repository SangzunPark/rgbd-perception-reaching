
import pytest

from src.analyze_ablation import (
    group_trials,
    summarize_variant,
)


def make_rows():
    """Create two synthetic trials for all eight variants."""

    rows = []

    for variant in "ABCDEFGH":
        for trial, error in [
            ("trial_01", 2.0),
            ("trial_02", 4.0),
        ]:
            rows.append({
                "variant": variant,
                "trial": trial,
                "error_3d_mm": error,
                "dx_mm": 0.0,
                "dy_mm": error,
                "dz_mm": 0.0,
            })

    return rows


def test_group_trials():
    groups = group_trials(make_rows())

    assert len(groups) == 8

    for variant in "ABCDEFGH":
        assert len(groups[variant]) == 2


def test_summary_statistics():
    groups = group_trials(make_rows())
    result = summarize_variant(groups["A"])

    assert result["count"] == 2
    assert result["mean_mm"] == pytest.approx(3.0)
    assert result["median_mm"] == pytest.approx(3.0)
    assert result["p90_mm"] == pytest.approx(3.8)
    assert result["max_mm"] == pytest.approx(4.0)

    assert result["mean_dx_mm"] == pytest.approx(0.0)
    assert result["mean_dy_mm"] == pytest.approx(3.0)
    assert result["mean_dz_mm"] == pytest.approx(0.0)


def test_duplicate_trial_rejected():
    rows = make_rows()
    rows.append(rows[0].copy())

    with pytest.raises(ValueError):
        group_trials(rows)


def test_missing_trial_rejected():
    rows = make_rows()

    rows = [
        row for row in rows
        if not (
            row["variant"] == "H"
            and row["trial"] == "trial_02"
        )
    ]

    with pytest.raises(ValueError):
        group_trials(rows)
