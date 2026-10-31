
import csv

import numpy as np
import pytest

from src.visualize_robustness import (
    get_metric,
    load_summary,
)


def write_summary(tmp_path, rows):
    path = tmp_path / "summary.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "condition",
                "method",
                "mean_error_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def make_rows():
    return [
        {
            "experiment": "gaussian",
            "condition": "sigma_3mm",
            "method": method,
            "mean_error_mm": value,
        }
        for method, value in [
            ("A", "4.9"),
            ("G", "19.6"),
            ("R", "0.6"),
        ]
    ]


def test_load_complete_summary(tmp_path):
    path = write_summary(tmp_path, make_rows())

    table = load_summary(path)

    assert len(table) == 3

    assert get_metric(
        table,
        "gaussian",
        "sigma_3mm",
        "R",
        "mean_error_mm",
    ) == pytest.approx(0.6)


def test_missing_mean_returns_nan(tmp_path):
    rows = make_rows()

    rows[2]["mean_error_mm"] = ""

    path = write_summary(tmp_path, rows)
    table = load_summary(path)

    result = get_metric(
        table,
        "gaussian",
        "sigma_3mm",
        "R",
        "mean_error_mm",
    )

    assert np.isnan(result)


def test_incomplete_methods_rejected(tmp_path):
    rows = make_rows()[:2]

    path = write_summary(tmp_path, rows)

    with pytest.raises(
        ValueError,
        match="Incomplete methods",
    ):
        load_summary(path)


def test_duplicate_method_rejected(tmp_path):
    rows = make_rows()
    rows.append(rows[0].copy())

    path = write_summary(tmp_path, rows)

    with pytest.raises(
        ValueError,
        match="Duplicate summary",
    ):
        load_summary(path)
