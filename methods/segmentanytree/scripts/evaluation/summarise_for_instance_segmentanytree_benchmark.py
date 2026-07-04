"""Aggregate per-plot SegmentAnyTree/FOR-instance evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
COMPLETED_STATUSES = {
    "completed",
    "completed_with_postprocess_repair",
    "evaluated",
    "complete",
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_plot_metrics(metrics_root: Path) -> list[dict[str, str]]:
    if not metrics_root.is_dir():
        raise FileNotFoundError(f"Metrics directory does not exist: {metrics_root}")
    metric_files = sorted(
        path
        for path in metrics_root.rglob("*.csv")
        if not path.name.endswith(
            ("_matches.csv", "_unmatched_predictions.csv", "_unmatched_references.csv")
        )
    )
    if not metric_files:
        raise ValueError(f"No per-plot metric CSV files found in: {metrics_root}")

    rows: list[dict[str, str]] = []
    for metric_file in metric_files:
        for row in read_csv_rows(metric_file):
            row["metrics_source"] = str(metric_file)
            rows.append(row)
    if not rows:
        raise ValueError(f"Per-plot metric files contain no rows: {metrics_root}")
    return rows


def inventory_indexes(
    inventory_rows: Iterable[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    by_relative_path: dict[str, dict[str, str]] = {}
    by_collection_plot: dict[tuple[str, str], dict[str, str]] = {}
    for row in inventory_rows:
        relative_path = row.get("relative_path", "").strip()
        collection = row.get("collection", "").strip()
        plot_name = Path(relative_path).stem if relative_path else Path(
            row.get("filename", "")
        ).stem
        if relative_path:
            by_relative_path[relative_path] = row
        if collection and plot_name:
            by_collection_plot[(collection, plot_name)] = row
    return by_relative_path, by_collection_plot


def join_inventory(
    metric_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_relative_path, by_collection_plot = inventory_indexes(inventory_rows)
    joined: list[dict[str, str]] = []
    for metric in metric_rows:
        row = dict(metric)
        relative_path = row.get("relative_path", "").strip()
        collection = row.get("collection", "").strip()
        plot_name = row.get("plot_name", "").strip()
        inventory = by_relative_path.get(relative_path)
        if inventory is None and collection and plot_name:
            inventory = by_collection_plot.get((collection, plot_name))
        if inventory:
            row["relative_path"] = relative_path or inventory.get("relative_path", "")
            row["collection"] = collection or inventory.get("collection", "")
            row["split"] = row.get("split", "").strip() or inventory.get("split", "")
            row["inventory_reference_tree_count"] = inventory.get(
                "reference_tree_count", ""
            )
        else:
            row.setdefault("inventory_reference_tree_count", "")
        joined.append(row)
    return joined


def add_missing_inventory_rows(
    plot_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    evaluated_paths = {
        row.get("relative_path", "").strip()
        for row in plot_rows
        if row.get("relative_path", "").strip()
    }
    evaluated_collection_plots = {
        (row.get("collection", "").strip(), row.get("plot_name", "").strip())
        for row in plot_rows
    }
    combined = list(plot_rows)
    for inventory in inventory_rows:
        relative_path = inventory.get("relative_path", "").strip()
        collection = inventory.get("collection", "").strip()
        plot_name = Path(relative_path).stem if relative_path else Path(
            inventory.get("filename", "")
        ).stem
        if relative_path in evaluated_paths or (
            collection,
            plot_name,
        ) in evaluated_collection_plots:
            continue
        combined.append(
            {
                "relative_path": relative_path,
                "collection": collection,
                "plot_name": plot_name,
                "split": inventory.get("split", "").strip(),
                "inventory_reference_tree_count": inventory.get(
                    "reference_tree_count", ""
                ),
                "status": "missing_metrics",
                "metrics_source": "",
            }
        )
    return sorted(
        combined,
        key=lambda row: (
            row.get("collection", ""),
            row.get("relative_path", ""),
            row.get("plot_name", ""),
        ),
    )


def number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field!r}: {value!r}") from exc


def integer_total(rows: list[dict[str, Any]], field: str) -> int:
    return int(sum(number(row, field) or 0.0 for row in rows))


def mean_available(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := number(row, field)) is not None]
    return sum(values) / len(values) if values else None


def aggregate_rows(rows: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    true_positives = integer_total(rows, "true_positives")
    false_positives = integer_total(rows, "false_positives")
    false_negatives = integer_total(rows, "false_negatives")
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    micro_precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    micro_recall = true_positives / recall_denominator if recall_denominator else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    statuses = [str(row.get("status", "")).strip().lower() for row in rows]
    completed = sum(status in COMPLETED_STATUSES for status in statuses)
    runtime_values = [
        value for row in rows if (value := number(row, "runtime_seconds")) is not None
    ]
    memory_values = [
        value for row in rows if (value := number(row, "peak_memory_gb")) is not None
    ]
    return {
        "group": group_name,
        "plot_count": len(rows),
        "evaluated_plot_count": sum(bool(row.get("metrics_source")) for row in rows),
        "total_reference_trees": integer_total(rows, "reference_tree_count"),
        "total_predicted_trees": integer_total(rows, "predicted_tree_count"),
        "total_true_positives": true_positives,
        "total_false_positives": false_positives,
        "total_false_negatives": false_negatives,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_plot_f1": mean_available(rows, "f1"),
        "mean_matched_iou": mean_available(rows, "mean_matched_iou"),
        "mean_median_matched_iou": mean_available(rows, "median_matched_iou"),
        "total_runtime_seconds": sum(runtime_values) if runtime_values else None,
        "mean_runtime_seconds": (
            sum(runtime_values) / len(runtime_values) if runtime_values else None
        ),
        "peak_memory_max_gb": max(memory_values) if memory_values else None,
        "completed_count": completed,
        "failed_count": len(rows) - completed,
    }


def grouped_summaries(
    rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get(field, "")).strip() or "unassigned"
        groups[label].append(row)
    return [aggregate_rows(groups[label], label) for label in sorted(groups)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty summary CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise full FOR-instance SegmentAnyTree per-plot metrics."
    )
    parser.add_argument(
        "--metrics-root",
        default="results/tables/segmentanytree_for_instance/per_plot",
    )
    parser.add_argument(
        "--inventory-csv",
        default="results/metadata/segmentanytree_for_instance/inventory.csv",
    )
    parser.add_argument(
        "--plot-output",
        default="results/tables/segmentanytree_for_instance_plot_metrics.csv",
    )
    parser.add_argument(
        "--collection-output",
        default=(
            "results/tables/"
            "segmentanytree_for_instance_summary_by_collection.csv"
        ),
    )
    parser.add_argument(
        "--split-output",
        default="results/tables/segmentanytree_for_instance_summary_by_split.csv",
    )
    parser.add_argument(
        "--json-output",
        default=(
            "results/metadata/segmentanytree_for_instance/"
            "benchmark_summary.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_root = resolve_path(args.metrics_root)
    inventory_path = resolve_path(args.inventory_csv)
    metric_rows = read_plot_metrics(metrics_root)
    inventory_rows = read_csv_rows(inventory_path)
    plot_rows = join_inventory(metric_rows, inventory_rows)
    plot_rows = add_missing_inventory_rows(plot_rows, inventory_rows)
    collection_rows = grouped_summaries(plot_rows, "collection")
    split_rows = grouped_summaries(plot_rows, "split")
    overall = aggregate_rows(plot_rows, "all")

    plot_output = resolve_path(args.plot_output)
    collection_output = resolve_path(args.collection_output)
    split_output = resolve_path(args.split_output)
    json_output = resolve_path(args.json_output)
    write_csv(plot_output, plot_rows)
    write_csv(collection_output, collection_rows)
    write_csv(split_output, split_rows)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "for_instance_segmentanytree",
        "metrics_root": str(metrics_root),
        "inventory_csv": str(inventory_path),
        "overall": overall,
        "by_collection": collection_rows,
        "by_split": split_rows,
        "outputs": {
            "plot_metrics_csv": str(plot_output),
            "collection_summary_csv": str(collection_output),
            "split_summary_csv": str(split_output),
            "metadata_json": str(json_output),
        },
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Plots: {overall['plot_count']}")
    print(f"Micro F1: {overall['micro_f1']:.6f}")
    print(f"Plot metrics: {plot_output}")
    print(f"Summary metadata: {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
