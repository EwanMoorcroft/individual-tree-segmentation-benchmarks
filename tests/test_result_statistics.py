from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.result_statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_plot_confidence_intervals,
    summarise_matched_iou,
    summarise_plot_distribution,
    summarise_sites,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "plot_id": "plot-a",
            "site": "site-b",
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 1,
            "f1": 0.0,
        },
        {
            "plot_id": "plot-b",
            "site": "site-a",
            "true_positives": 1,
            "false_positives": 3,
            "false_negatives": 3,
            "f1": 0.25,
        },
        {
            "plot_id": "plot-c",
            "site": "site-a",
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 1,
            "f1": 0.5,
        },
        {
            "plot_id": "plot-d",
            "site": "site-b",
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "f1": 1.0,
        },
    ]


def test_plot_distribution_reports_median_iqr_zero_count_and_micro() -> None:
    summary = summarise_plot_distribution(_rows())

    assert summary == {
        "plot_count": 4,
        "true_positives": 3,
        "false_positives": 5,
        "false_negatives": 5,
        "micro_precision": 3 / 8,
        "micro_recall": 3 / 8,
        "micro_f1": 3 / 8,
        "mean_plot_f1": 0.4375,
        "median_plot_f1": 0.375,
        "plot_f1_q1": 0.1875,
        "plot_f1_q3": 0.625,
        "plot_f1_iqr": 0.4375,
        "zero_f1_plot_count": 1,
        "zero_f1_plot_fraction": 0.25,
        "quantile_method": "linear",
    }


def test_plot_distribution_accepts_exact_csv_numeric_strings() -> None:
    rows = [
        {
            "plot_id": "p1",
            "true_positives": "1",
            "false_positives": "0",
            "false_negatives": "0",
            "f1": "1.0",
        }
    ]

    summary = summarise_plot_distribution(rows)

    assert summary["micro_f1"] == 1.0
    assert summary["median_plot_f1"] == 1.0


def test_site_summaries_are_sorted_and_include_macro_and_micro_metrics() -> None:
    summaries = summarise_sites(list(reversed(_rows())))

    assert [row["site"] for row in summaries] == ["site-a", "site-b"]
    site_a, site_b = summaries
    assert site_a["plot_count"] == 2
    assert site_a["mean_plot_f1"] == 0.375
    assert site_a["median_plot_f1"] == 0.375
    assert site_a["micro_f1"] == pytest.approx(1 / 3)
    assert site_b["mean_plot_f1"] == 0.5
    assert site_b["micro_f1"] == 0.5
    assert site_b["zero_f1_plot_count"] == 1


def test_matched_iou_summary_uses_nulls_for_no_matches() -> None:
    assert summarise_matched_iou([]) == {
        "matched_pair_count": 0,
        "mean_matched_iou": None,
        "median_matched_iou": None,
    }
    assert summarise_matched_iou([{"iou": 0.25}, {"iou": "0.75"}]) == {
        "matched_pair_count": 2,
        "mean_matched_iou": 0.5,
        "median_matched_iou": 0.5,
    }


def test_bootstrap_is_deterministic_and_row_order_invariant() -> None:
    expected = bootstrap_plot_confidence_intervals(
        _rows(), iterations=80, seed=41
    )
    reordered = bootstrap_plot_confidence_intervals(
        list(reversed(_rows())), iterations=80, seed=41
    )

    assert expected == reordered
    assert [row["metric"] for row in expected] == [
        "mean_plot_f1",
        "median_plot_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ]
    assert all(row["resampling_scheme"] == "plot" for row in expected)
    assert all(row["plot_count"] == 4 for row in expected)
    assert all(row["site_count"] is None for row in expected)
    assert all(row["quantile_method"] == "linear" for row in expected)


def test_bootstrap_estimates_recompute_micro_from_summed_plot_counts() -> None:
    rows = [
        {
            "plot_id": "large-perfect",
            "true_positives": 100,
            "false_positives": 0,
            "false_negatives": 0,
            "f1": 1.0,
        },
        {
            "plot_id": "small-failure",
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 1,
            "f1": 0.0,
        },
    ]

    results = bootstrap_plot_confidence_intervals(
        rows,
        metrics=("mean_plot_f1", "micro_f1"),
        iterations=20,
        seed=7,
    )
    by_metric = {row["metric"]: row for row in results}

    assert by_metric["mean_plot_f1"]["estimate"] == 0.5
    assert by_metric["micro_f1"]["estimate"] == pytest.approx(200 / 202)
    assert by_metric["micro_f1"]["estimate"] != by_metric["mean_plot_f1"][
        "estimate"
    ]


def test_site_stratified_bootstrap_preserves_singleton_strata() -> None:
    rows = [
        {
            "plot_id": "p2",
            "site": "second",
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 1,
            "f1": 0.0,
        },
        {
            "plot_id": "p1",
            "site": "first",
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "f1": 1.0,
        },
    ]

    results = bootstrap_plot_confidence_intervals(
        rows,
        iterations=25,
        seed=5,
        stratify_by="site",
    )

    for row in results:
        assert row["resampling_scheme"] == "site_stratified_plot"
        assert row["stratify_by"] == "site"
        assert row["site_count"] == 2
        assert row["ci_lower"] == row["estimate"] == row["ci_upper"]
    assert results == bootstrap_plot_confidence_intervals(
        list(reversed(rows)),
        iterations=25,
        seed=5,
        stratify_by="site",
    )


def test_bootstrap_handles_zero_micro_denominators() -> None:
    rows = [
        {
            "plot_id": "empty",
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "f1": 0.0,
        }
    ]

    results = bootstrap_plot_confidence_intervals(rows, iterations=2)

    for row in results:
        assert row["estimate"] == 0.0
        assert row["ci_lower"] == 0.0
        assert row["ci_upper"] == 0.0
    assert results[0]["bootstrap_seed"] == DEFAULT_BOOTSTRAP_SEED
    assert DEFAULT_BOOTSTRAP_ITERATIONS == 10_000


@pytest.mark.parametrize(
    ("rows", "exception", "message"),
    [
        ([], ValueError, "must not be empty"),
        (
            [{"plot_id": "p"}],
            ValueError,
            "missing fields",
        ),
        (
            [
                {
                    "plot_id": "p",
                    "true_positives": -1,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "f1": 0,
                }
            ],
            ValueError,
            "cannot be negative",
        ),
        (
            [
                {
                    "plot_id": "p",
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "f1": float("nan"),
                }
            ],
            ValueError,
            "finite",
        ),
    ],
)
def test_plot_summary_rejects_invalid_rows(
    rows: list[dict[str, object]],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        summarise_plot_distribution(rows)


def test_statistics_reject_duplicate_plots_missing_sites_and_invalid_iou() -> None:
    duplicate = [_rows()[0], {**_rows()[1], "plot_id": "plot-a"}]
    with pytest.raises(ValueError, match="duplicate plot_id"):
        summarise_plot_distribution(duplicate)
    without_site = [{key: value for key, value in _rows()[0].items() if key != "site"}]
    with pytest.raises(ValueError, match="site"):
        summarise_sites(without_site)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        summarise_matched_iou([{"iou": 1.1}])
    with pytest.raises(ValueError, match="unsupported quantile_method"):
        summarise_plot_distribution(_rows(), quantile_method="not-a-method")


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"iterations": 0}, ValueError, "at least 1"),
        ({"iterations": True}, TypeError, "integer"),
        ({"seed": -1}, ValueError, "non-negative"),
        ({"confidence_level": 1.0}, ValueError, r"in \(0, 1\)"),
        ({"stratify_by": "collection"}, ValueError, "None or 'site'"),
        ({"metrics": ()}, ValueError, "must not be empty"),
        ({"metrics": ("unknown",)}, ValueError, "unsupported"),
        ({"metrics": ("micro_f1", "micro_f1")}, ValueError, "duplicates"),
    ],
)
def test_bootstrap_rejects_invalid_parameters(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    arguments: dict[str, object] = {"iterations": 2}
    arguments.update(kwargs)
    with pytest.raises(exception, match=message):
        bootstrap_plot_confidence_intervals(_rows(), **arguments)


def test_site_stratified_bootstrap_requires_site_on_every_row() -> None:
    rows = [{key: value for key, value in _rows()[0].items() if key != "site"}]
    with pytest.raises(ValueError, match="site"):
        bootstrap_plot_confidence_intervals(
            rows,
            iterations=2,
            stratify_by="site",
        )
