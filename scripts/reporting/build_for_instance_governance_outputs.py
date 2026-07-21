#!/usr/bin/env python3
"""Build deterministic FOR-instance site, distribution, and bootstrap CSVs.

Only committed per-plot metric CSVs and the canonical headline table are read.
No prediction artefact is opened and no accepted score is replaced.  Every
source aggregate must agree exactly with its canonical headline before any
output payload is returned or written.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.instance_metrics import precision_recall_f1
from benchmark.result_statistics import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE_LEVEL,
    SUPPORTED_BOOTSTRAP_METRICS,
    bootstrap_plot_confidence_intervals,
    summarise_plot_distribution,
    summarise_sites,
)


OUTPUT_ROOT = ROOT / "outputs" / "for_instance_benchmark_metrics"
HEADLINE_CSV = OUTPUT_ROOT / "for_instance_method_benchmark_results.csv"

SITE_OUTPUT = "for_instance_method_site_results.csv"
DISTRIBUTION_OUTPUT = "for_instance_plot_distribution_diagnostics.csv"
BOOTSTRAP_OUTPUT = "for_instance_bootstrap_confidence_intervals.csv"

EXPECTED_SITES = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")
RESULT_KEYS = (
    ("treex", "unsupervised_parameterised"),
    ("segmentanytree", "published_pretrained"),
    ("segmentanytree", "fine_tuned_on_dev"),
    ("treelearn", "published_pretrained"),
    ("treelearn", "fine_tuned_on_dev"),
    ("tls2trees", "development_tuned"),
    ("tls2trees", "published_default"),
)


@dataclass(frozen=True)
class PlotSource:
    relative_path: str
    kind: str
    filters: tuple[tuple[str, str], ...] = ()


PLOT_SOURCES = {
    ("treex", "unsupervised_parameterised"): PlotSource(
        "methods/treex/examples/treex_combined_dev_test_summary.csv",
        "treex_harmonized_union_mask",
        (("split", "test"),),
    ),
    ("segmentanytree", "published_pretrained"): PlotSource(
        "methods/segmentanytree/examples/"
        "sat_completed_target_plot_results_20260711.csv",
        "aligned",
        (("variant", "published_pretrained"),),
    ),
    ("segmentanytree", "fine_tuned_on_dev"): PlotSource(
        "methods/segmentanytree/examples/"
        "sat_completed_target_plot_results_20260711.csv",
        "aligned",
        (("variant", "fine_tuned_on_dev"),),
    ),
    ("treelearn", "published_pretrained"): PlotSource(
        "methods/treelearn/examples/"
        "treelearn_pretrained_test_plot_results_20260714.csv",
        "aligned",
        (("variant", "published_pretrained"),),
    ),
    ("treelearn", "fine_tuned_on_dev"): PlotSource(
        "methods/treelearn/examples/"
        "treelearn_finetuned_test_plot_results_20260713.csv",
        "aligned",
        (("variant", "fine_tuned_on_dev"),),
    ),
    ("tls2trees", "development_tuned"): PlotSource(
        "methods/tls2trees/examples/"
        "tls2trees_development_tuned_test_plot_results.csv",
        "aligned",
        (("variant", "development_tuned"), ("target", "leaf_on")),
    ),
    ("tls2trees", "published_default"): PlotSource(
        "methods/tls2trees/examples/"
        "tls2trees_published_default_test_plot_results.csv",
        "aligned",
        (("variant", "published_default"), ("target", "leaf_on")),
    ),
}

IDENTITY_FIELDS = (
    "dataset_slug",
    "method_slug",
    "variant",
    "run_id",
    "training_mode",
    "evaluation_protocol",
    "matching_policy",
    "evaluation_mask",
    "evaluation_split",
    "comparable_group",
)
SITE_FIELDS = (
    *IDENTITY_FIELDS,
    "site",
    "plots",
    "predicted_instances",
    "reference_instances",
    "true_positives",
    "false_positives",
    "false_negatives",
    "mean_plot_f1",
    "median_plot_f1",
    "plot_f1_q1",
    "plot_f1_q3",
    "plot_f1_iqr",
    "zero_f1_plot_count",
    "zero_f1_plot_fraction",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "quantile_method",
    "result_status",
    "ranking_eligible",
    "selection_eligible",
    "source_plot_metrics",
)
DISTRIBUTION_FIELDS = (
    *IDENTITY_FIELDS,
    "plots",
    "predicted_instances",
    "reference_instances",
    "true_positives",
    "false_positives",
    "false_negatives",
    "mean_plot_f1",
    "median_plot_f1",
    "plot_f1_q1",
    "plot_f1_q3",
    "plot_f1_iqr",
    "zero_f1_plot_count",
    "zero_f1_plot_fraction",
    "site_count",
    "mean_site_macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "quantile_method",
    "result_status",
    "ranking_eligible",
    "selection_eligible",
    "source_plot_metrics",
)
BOOTSTRAP_FIELDS = (
    *IDENTITY_FIELDS,
    "metric",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "bootstrap_iterations",
    "bootstrap_seed",
    "resampling_scheme",
    "stratify_by",
    "plot_count",
    "site_count",
    "quantile_method",
    "result_status",
    "ranking_eligible",
    "selection_eligible",
    "source_plot_metrics",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_headlines() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(HEADLINE_CSV)
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["method_slug"], row["variant"])
        if key in by_key:
            raise ValueError(f"Duplicate canonical headline identity: {key}")
        by_key[key] = row
    if set(by_key) != set(RESULT_KEYS) or len(rows) != len(RESULT_KEYS):
        raise ValueError(
            "Canonical headline must contain exactly the seven accepted results"
        )
    return by_key


def _required_integer(row: Mapping[str, str], field: str, *, label: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid integer field {field!r}") from exc
    if value < 0:
        raise ValueError(f"{label} field {field!r} cannot be negative")
    return value


def _required_unit_float(
    row: Mapping[str, str], field: str, *, label: str
) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid numeric field {field!r}") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} field {field!r} must be finite and in [0, 1]")
    return value


def _normalize_aligned_plot(
    row: Mapping[str, str], *, label: str
) -> dict[str, Any]:
    relative_path = row.get("relative_path", "").strip()
    site = row.get("collection", "").strip()
    if not relative_path or not site:
        raise ValueError(f"{label} requires relative_path and collection")
    return {
        "plot_id": PurePosixPath(relative_path).with_suffix("").as_posix(),
        "site": site,
        "predicted_instances": _required_integer(
            row, "predicted_instances", label=label
        ),
        "reference_instances": _required_integer(
            row, "reference_instances", label=label
        ),
        "true_positives": _required_integer(row, "true_positives", label=label),
        "false_positives": _required_integer(row, "false_positives", label=label),
        "false_negatives": _required_integer(row, "false_negatives", label=label),
        "f1": _required_unit_float(row, "f1", label=label),
    }


def _normalize_treex_plot(
    row: Mapping[str, str], *, label: str
) -> dict[str, Any]:
    plot_id = row.get("plot_id", "").strip()
    site = row.get("site", "").strip()
    if not plot_id or not site:
        raise ValueError(f"{label} requires plot_id and site")
    return {
        "plot_id": plot_id,
        "site": site,
        "predicted_instances": _required_integer(
            row, "predicted_trees_harmonized_union_mask", label=label
        ),
        "reference_instances": _required_integer(
            row, "reference_trees", label=label
        ),
        "true_positives": _required_integer(
            row, "true_positives_harmonized", label=label
        ),
        "false_positives": _required_integer(
            row, "false_positives_harmonized", label=label
        ),
        "false_negatives": _required_integer(
            row, "false_negatives_harmonized", label=label
        ),
        "f1": _required_unit_float(row, "f1_harmonized", label=label),
    }


def _verify_plot_rows(
    rows: Sequence[Mapping[str, Any]], headline: Mapping[str, str]
) -> None:
    expected_plots = int(headline["plots"])
    if len(rows) != expected_plots:
        raise ValueError(
            f"{headline['method_slug']}/{headline['variant']} has {len(rows)} "
            f"plot rows; expected {expected_plots}"
        )
    plot_ids = [str(row["plot_id"]) for row in rows]
    if len(plot_ids) != len(set(plot_ids)):
        raise ValueError("Per-plot source contains duplicate plot IDs")
    site_counts = Counter(str(row["site"]) for row in rows)
    if set(site_counts) != set(EXPECTED_SITES):
        raise ValueError(f"Per-plot source has unexpected sites: {site_counts}")

    for row in rows:
        predicted = int(row["predicted_instances"])
        references = int(row["reference_instances"])
        true_positives = int(row["true_positives"])
        false_positives = int(row["false_positives"])
        false_negatives = int(row["false_negatives"])
        if predicted != true_positives + false_positives:
            raise ValueError(f"Prediction-count identity failed for {row['plot_id']}")
        if references != true_positives + false_negatives:
            raise ValueError(f"Reference-count identity failed for {row['plot_id']}")
        _, _, calculated_f1 = precision_recall_f1(
            true_positives, false_positives, false_negatives
        )
        if not math.isclose(
            float(row["f1"]), calculated_f1, rel_tol=0.0, abs_tol=2e-16
        ):
            raise ValueError(f"F1 count identity failed for {row['plot_id']}")


def load_plot_rows(
    key: tuple[str, str], headline: Mapping[str, str]
) -> list[dict[str, Any]]:
    source = PLOT_SOURCES[key]
    raw_rows = read_csv(ROOT / source.relative_path)
    rows = [
        row
        for row in raw_rows
        if all(row.get(field) == value for field, value in source.filters)
    ]
    if source.kind == "aligned":
        expected_fields = {
            "method_slug": headline["method_slug"],
            "variant": headline["variant"],
            "run_id": headline["run_id"],
            "dataset_split": headline["evaluation_split"],
            "evaluation_protocol": headline["evaluation_protocol"],
            "matching_policy": headline["matching_policy"],
            "evaluation_mask": headline["evaluation_mask"],
            "iou_threshold": "0.5",
        }
        for row_number, row in enumerate(rows, start=1):
            for field, expected in expected_fields.items():
                actual = row.get(field)
                if field == "iou_threshold":
                    if actual is None or float(actual) != float(expected):
                        raise ValueError(
                            f"{source.relative_path} row {row_number} field "
                            f"{field!r} does not match its canonical headline"
                        )
                elif actual != expected:
                    raise ValueError(
                        f"{source.relative_path} row {row_number} field "
                        f"{field!r} does not match its canonical headline"
                    )
        normalized = [
            _normalize_aligned_plot(
                row, label=f"{source.relative_path} row {row_number}"
            )
            for row_number, row in enumerate(rows, start=1)
        ]
    elif source.kind == "treex_harmonized_union_mask":
        normalized = [
            _normalize_treex_plot(
                row, label=f"{source.relative_path} row {row_number}"
            )
            for row_number, row in enumerate(rows, start=1)
        ]
    else:
        raise ValueError(f"Unknown plot source kind: {source.kind}")

    normalized.sort(key=lambda row: str(row["plot_id"]))
    _verify_plot_rows(normalized, headline)
    return normalized


def _identity(headline: Mapping[str, str]) -> dict[str, str]:
    return {field: headline[field] for field in IDENTITY_FIELDS}


def _diagnostic_metadata(source: PlotSource) -> dict[str, str]:
    return {
        "result_status": "diagnostic",
        "ranking_eligible": "false",
        "selection_eligible": "false",
        "source_plot_metrics": source.relative_path,
    }


def _instance_totals(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    return (
        sum(int(row["predicted_instances"]) for row in rows),
        sum(int(row["reference_instances"]) for row in rows),
    )


def verify_headline_aggregate(
    headline: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    predicted_instances, reference_instances = _instance_totals(rows)
    calculated: dict[str, int | float] = {
        "plots": len(rows),
        "predicted_instances": predicted_instances,
        "reference_instances": reference_instances,
        "true_positives": int(summary["true_positives"]),
        "false_positives": int(summary["false_positives"]),
        "false_negatives": int(summary["false_negatives"]),
        "mean_plot_f1": float(summary["mean_plot_f1"]),
        "micro_precision": float(summary["micro_precision"]),
        "micro_recall": float(summary["micro_recall"]),
        "micro_f1": float(summary["micro_f1"]),
    }
    integer_fields = (
        "plots",
        "predicted_instances",
        "reference_instances",
        "true_positives",
        "false_positives",
        "false_negatives",
    )
    float_fields = (
        "mean_plot_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    )
    for field in integer_fields:
        if calculated[field] != int(headline[field]):
            raise ValueError(
                f"Per-plot {field} disagrees with canonical headline for "
                f"{headline['method_slug']}/{headline['variant']}"
            )
    for field in float_fields:
        if calculated[field] != float(headline[field]):
            raise ValueError(
                f"Per-plot {field} disagrees exactly with canonical headline "
                f"for {headline['method_slug']}/{headline['variant']}"
            )


def _site_output_rows(
    headline: Mapping[str, str],
    plots: Sequence[Mapping[str, Any]],
    source: PlotSource,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = summarise_sites(plots)
    by_site_plots: dict[str, list[Mapping[str, Any]]] = {
        site: [row for row in plots if row["site"] == site]
        for site in EXPECTED_SITES
    }
    by_site_summary = {str(row["site"]): row for row in summaries}
    if set(by_site_summary) != set(EXPECTED_SITES):
        raise ValueError("Expected exactly five site summaries")

    output: list[dict[str, Any]] = []
    for site in EXPECTED_SITES:
        summary = by_site_summary[site]
        predicted_instances, reference_instances = _instance_totals(
            by_site_plots[site]
        )
        output.append(
            {
                **_identity(headline),
                "site": site,
                "plots": summary["plot_count"],
                "predicted_instances": predicted_instances,
                "reference_instances": reference_instances,
                **{
                    field: summary[field]
                    for field in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "mean_plot_f1",
                        "median_plot_f1",
                        "plot_f1_q1",
                        "plot_f1_q3",
                        "plot_f1_iqr",
                        "zero_f1_plot_count",
                        "zero_f1_plot_fraction",
                        "micro_precision",
                        "micro_recall",
                        "micro_f1",
                        "quantile_method",
                    )
                },
                **_diagnostic_metadata(source),
            }
        )
    return output, summaries


def build_governance_rows(
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, list[dict[str, Any]]]:
    """Build and verify all three output row sets without writing files."""

    if isinstance(bootstrap_iterations, bool) or not isinstance(
        bootstrap_iterations, int
    ):
        raise TypeError("bootstrap_iterations must be an integer")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")

    headlines = load_headlines()
    site_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for key in RESULT_KEYS:
        headline = headlines[key]
        source = PLOT_SOURCES[key]
        plots = load_plot_rows(key, headline)
        distribution = summarise_plot_distribution(plots)
        verify_headline_aggregate(headline, plots, distribution)
        current_site_rows, site_summaries = _site_output_rows(
            headline, plots, source
        )
        site_rows.extend(current_site_rows)

        predicted_instances, reference_instances = _instance_totals(plots)
        distribution_rows.append(
            {
                **_identity(headline),
                "plots": headline["plots"],
                "predicted_instances": headline["predicted_instances"],
                "reference_instances": headline["reference_instances"],
                "true_positives": headline["true_positives"],
                "false_positives": headline["false_positives"],
                "false_negatives": headline["false_negatives"],
                "mean_plot_f1": headline["mean_plot_f1"],
                "median_plot_f1": distribution["median_plot_f1"],
                "plot_f1_q1": distribution["plot_f1_q1"],
                "plot_f1_q3": distribution["plot_f1_q3"],
                "plot_f1_iqr": distribution["plot_f1_iqr"],
                "zero_f1_plot_count": distribution["zero_f1_plot_count"],
                "zero_f1_plot_fraction": distribution[
                    "zero_f1_plot_fraction"
                ],
                "site_count": len(site_summaries),
                "mean_site_macro_f1": statistics.fmean(
                    float(row["mean_plot_f1"]) for row in site_summaries
                ),
                "micro_precision": headline["micro_precision"],
                "micro_recall": headline["micro_recall"],
                "micro_f1": headline["micro_f1"],
                "quantile_method": distribution["quantile_method"],
                **_diagnostic_metadata(source),
            }
        )
        if (
            predicted_instances != int(headline["predicted_instances"])
            or reference_instances != int(headline["reference_instances"])
        ):
            raise AssertionError("Verified instance totals changed unexpectedly")

        for stratify_by in (None, "site"):
            intervals = bootstrap_plot_confidence_intervals(
                plots,
                metrics=SUPPORTED_BOOTSTRAP_METRICS,
                iterations=bootstrap_iterations,
                seed=DEFAULT_BOOTSTRAP_SEED,
                confidence_level=DEFAULT_CONFIDENCE_LEVEL,
                stratify_by=stratify_by,
                quantile_method="linear",
            )
            for interval in intervals:
                bootstrap_rows.append(
                    {
                        **_identity(headline),
                        **interval,
                        **_diagnostic_metadata(source),
                    }
                )

    if len(site_rows) != 35:
        raise AssertionError("Expected 35 method-site rows")
    if len(distribution_rows) != 7:
        raise AssertionError("Expected seven distribution rows")
    expected_bootstrap_rows = (
        len(RESULT_KEYS) * 2 * len(SUPPORTED_BOOTSTRAP_METRICS)
    )
    if len(bootstrap_rows) != expected_bootstrap_rows:
        raise AssertionError(
            f"Expected {expected_bootstrap_rows} bootstrap interval rows"
        )
    return {
        SITE_OUTPUT: site_rows,
        DISTRIBUTION_OUTPUT: distribution_rows,
        BOOTSTRAP_OUTPUT: bootstrap_rows,
    }


def _csv_bytes(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_output_payloads(
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> dict[str, bytes]:
    rows = build_governance_rows(bootstrap_iterations=bootstrap_iterations)
    return {
        SITE_OUTPUT: _csv_bytes(rows[SITE_OUTPUT], SITE_FIELDS),
        DISTRIBUTION_OUTPUT: _csv_bytes(
            rows[DISTRIBUTION_OUTPUT], DISTRIBUTION_FIELDS
        ),
        BOOTSTRAP_OUTPUT: _csv_bytes(
            rows[BOOTSTRAP_OUTPUT], BOOTSTRAP_FIELDS
        ),
    }


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-iterations",
        type=_positive_integer,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any generated CSV is missing or not byte-identical.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    payloads = build_output_payloads(
        bootstrap_iterations=args.bootstrap_iterations
    )
    if args.check:
        stale = [
            name
            for name, payload in payloads.items()
            if not (output_dir / name).is_file()
            or (output_dir / name).read_bytes() != payload
        ]
        if stale:
            raise SystemExit(
                "Governance outputs are stale or missing: " + ", ".join(stale)
            )
        print(f"Governance outputs are current: {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)
    print(f"Wrote {len(payloads)} governance CSVs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
