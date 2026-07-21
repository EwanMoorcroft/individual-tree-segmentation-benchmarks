"""Distribution and plot-bootstrap statistics for accepted per-plot rows."""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .instance_metrics import precision_recall_f1


DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260721
DEFAULT_CONFIDENCE_LEVEL = 0.95
SUPPORTED_BOOTSTRAP_METRICS = (
    "mean_plot_f1",
    "median_plot_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
)

_NONNEGATIVE_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _required_text(value: Any, *, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_number} field {field!r} must be non-empty text")
    return value.strip()


def _nonnegative_integer(value: Any, *, field: str, row_number: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"row {row_number} field {field!r} must be an integer")
    if isinstance(value, (int, np.integer)):
        normalized = int(value)
    elif isinstance(value, str) and _NONNEGATIVE_INTEGER.fullmatch(value.strip()):
        normalized = int(value)
    else:
        raise TypeError(
            f"row {row_number} field {field!r} must be a non-negative integer"
        )
    if normalized < 0:
        raise ValueError(f"row {row_number} field {field!r} cannot be negative")
    return normalized


def _unit_float(value: Any, *, field: str, row_number: int) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"row {row_number} field {field!r} must be numeric")
    if isinstance(value, str):
        try:
            normalized = float(value.strip())
        except ValueError as exc:
            raise TypeError(
                f"row {row_number} field {field!r} must be numeric"
            ) from exc
    elif isinstance(value, (int, float, np.integer, np.floating)):
        normalized = float(value)
    else:
        raise TypeError(f"row {row_number} field {field!r} must be numeric")
    if not np.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(
            f"row {row_number} field {field!r} must be finite and in [0, 1]"
        )
    return normalized


def _normalise_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_site: bool,
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    if not rows:
        raise ValueError("rows must not be empty")

    normalized: list[dict[str, Any]] = []
    seen_plot_ids: set[str] = set()
    required = ("plot_id", "true_positives", "false_positives", "false_negatives", "f1")
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {row_number} must be a mapping")
        missing = [field for field in required if field not in row]
        if require_site and "site" not in row:
            missing.append("site")
        if missing:
            raise ValueError(
                f"row {row_number} is missing fields: {', '.join(missing)}"
            )

        plot_id = _required_text(row["plot_id"], field="plot_id", row_number=row_number)
        if plot_id in seen_plot_ids:
            raise ValueError(f"duplicate plot_id: {plot_id}")
        seen_plot_ids.add(plot_id)
        item = {
            "plot_id": plot_id,
            "true_positives": _nonnegative_integer(
                row["true_positives"], field="true_positives", row_number=row_number
            ),
            "false_positives": _nonnegative_integer(
                row["false_positives"], field="false_positives", row_number=row_number
            ),
            "false_negatives": _nonnegative_integer(
                row["false_negatives"], field="false_negatives", row_number=row_number
            ),
            "f1": _unit_float(row["f1"], field="f1", row_number=row_number),
        }
        if require_site:
            item["site"] = _required_text(
                row["site"], field="site", row_number=row_number
            )
        normalized.append(item)
    return sorted(normalized, key=lambda row: row["plot_id"])


def _quantiles(
    values: np.ndarray,
    probabilities: tuple[float, ...],
    *,
    method: str,
) -> np.ndarray:
    if not isinstance(method, str) or not method:
        raise TypeError("quantile_method must be non-empty text")
    try:
        return np.asarray(np.quantile(values, probabilities, method=method))
    except ValueError as exc:
        raise ValueError(f"unsupported quantile_method: {method}") from exc


def _summarise_normalized_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantile_method: str,
) -> dict[str, Any]:
    f1_values = np.asarray([row["f1"] for row in rows], dtype=np.float64)
    q1, median, q3 = _quantiles(
        f1_values, (0.25, 0.50, 0.75), method=quantile_method
    )
    true_positives = sum(int(row["true_positives"]) for row in rows)
    false_positives = sum(int(row["false_positives"]) for row in rows)
    false_negatives = sum(int(row["false_negatives"]) for row in rows)
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    zero_count = int(np.count_nonzero(f1_values == 0.0))
    return {
        "plot_count": len(rows),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_plot_f1": float(statistics.fmean(f1_values)),
        "median_plot_f1": float(median),
        "plot_f1_q1": float(q1),
        "plot_f1_q3": float(q3),
        "plot_f1_iqr": float(q3 - q1),
        "zero_f1_plot_count": zero_count,
        "zero_f1_plot_fraction": float(zero_count / len(rows)),
        "quantile_method": quantile_method,
    }


def summarise_plot_distribution(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantile_method: str = "linear",
) -> dict[str, Any]:
    """Summarise plot F1 distribution and micro counts for one result row."""

    normalized = _normalise_rows(rows, require_site=False)
    return _summarise_normalized_rows(
        normalized, quantile_method=quantile_method
    )


def summarise_sites(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantile_method: str = "linear",
) -> list[dict[str, Any]]:
    """Return plot-distribution and micro summaries for each explicit site."""

    normalized = _normalise_rows(rows, require_site=True)
    sites: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        sites.setdefault(str(row["site"]), []).append(row)
    return [
        {
            "site": site,
            **_summarise_normalized_rows(
                sites[site], quantile_method=quantile_method
            ),
        }
        for site in sorted(sites)
    ]


def summarise_matched_iou(
    match_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    """Summarise matched-pair IoUs; empty input yields explicit nulls."""

    if isinstance(match_rows, (str, bytes)) or not isinstance(match_rows, Sequence):
        raise TypeError("match_rows must be a sequence of mappings")
    ious: list[float] = []
    for row_number, row in enumerate(match_rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"match row {row_number} must be a mapping")
        if "iou" not in row:
            raise ValueError(f"match row {row_number} is missing field: iou")
        ious.append(_unit_float(row["iou"], field="iou", row_number=row_number))
    return {
        "matched_pair_count": len(ious),
        "mean_matched_iou": float(np.mean(ious)) if ious else None,
        "median_matched_iou": float(np.median(ious)) if ious else None,
    }


def _validated_bootstrap_metrics(metrics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence):
        raise TypeError("metrics must be a sequence of metric names")
    if not metrics:
        raise ValueError("metrics must not be empty")
    normalized: list[str] = []
    for metric in metrics:
        if not isinstance(metric, str):
            raise TypeError("metrics must contain only strings")
        if metric not in SUPPORTED_BOOTSTRAP_METRICS:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
        normalized.append(metric)
    if len(set(normalized)) != len(normalized):
        raise ValueError("metrics must not contain duplicates")
    return tuple(normalized)


def _metrics_for_indices(
    rows: Sequence[Mapping[str, Any]], indices: np.ndarray
) -> dict[str, float]:
    f1_values = np.asarray([rows[int(index)]["f1"] for index in indices])
    true_positives = sum(
        int(rows[int(index)]["true_positives"]) for index in indices
    )
    false_positives = sum(
        int(rows[int(index)]["false_positives"]) for index in indices
    )
    false_negatives = sum(
        int(rows[int(index)]["false_negatives"]) for index in indices
    )
    precision, recall, f1 = precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    return {
        "mean_plot_f1": float(statistics.fmean(f1_values)),
        "median_plot_f1": float(np.median(f1_values)),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def bootstrap_plot_confidence_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = SUPPORTED_BOOTSTRAP_METRICS,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    stratify_by: str | None = None,
    quantile_method: str = "linear",
) -> list[dict[str, Any]]:
    """Return percentile CIs from plot-level resampling.

    Ordinary resampling draws ``n`` plots from all ``n`` rows.  With
    ``stratify_by='site'``, each site draws its original number of plots with
    replacement, preserving observed site composition.  Micro metrics are
    recomputed from the summed counts of every resample.
    """

    selected_metrics = _validated_bootstrap_metrics(metrics)
    if isinstance(iterations, bool) or not isinstance(
        iterations, (int, np.integer)
    ):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, (int, float, np.integer, np.floating)
    ):
        raise TypeError("confidence_level must be numeric")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if stratify_by not in (None, "site"):
        raise ValueError("stratify_by must be None or 'site'")

    normalized = _normalise_rows(rows, require_site=stratify_by == "site")
    rng = np.random.default_rng(int(seed))
    sample_count = len(normalized)
    all_indices = np.arange(sample_count, dtype=np.int64)
    strata: list[np.ndarray] | None = None
    if stratify_by == "site":
        site_indices: dict[str, list[int]] = {}
        for index, row in enumerate(normalized):
            site_indices.setdefault(str(row["site"]), []).append(index)
        strata = [
            np.asarray(site_indices[site], dtype=np.int64)
            for site in sorted(site_indices)
        ]

    distributions = {
        metric: np.empty(int(iterations), dtype=np.float64)
        for metric in selected_metrics
    }
    for iteration in range(int(iterations)):
        if strata is None:
            sampled = rng.choice(all_indices, size=sample_count, replace=True)
        else:
            sampled = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in strata
                ]
            )
        iteration_metrics = _metrics_for_indices(normalized, sampled)
        for metric in selected_metrics:
            distributions[metric][iteration] = iteration_metrics[metric]

    estimates = _metrics_for_indices(normalized, all_indices)
    tail = (1.0 - confidence) / 2.0
    scheme = "site_stratified_plot" if strata is not None else "plot"
    results: list[dict[str, Any]] = []
    for metric in selected_metrics:
        lower, upper = _quantiles(
            distributions[metric],
            (tail, 1.0 - tail),
            method=quantile_method,
        )
        results.append(
            {
                "metric": metric,
                "estimate": estimates[metric],
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "confidence_level": confidence,
                "bootstrap_iterations": int(iterations),
                "bootstrap_seed": int(seed),
                "resampling_scheme": scheme,
                "stratify_by": stratify_by,
                "plot_count": sample_count,
                "site_count": len(strata) if strata is not None else None,
                "quantile_method": quantile_method,
            }
        )
    return results
