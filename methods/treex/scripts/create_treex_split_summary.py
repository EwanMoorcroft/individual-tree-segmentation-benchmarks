"""Combine per-plot TreeX JSON outputs into one split summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "plot_id",
    "total_points",
    "tree_class_points",
    "reference_tree_count_tree_classes",
    "predicted_instances",
    "elapsed_seconds",
    "reference_trees",
    "predicted_trees_harmonized_union_mask",
    "predicted_trees_on_reference_labelled_mask",
    "true_positives_labelled_mask",
    "false_positives_labelled_mask",
    "false_negatives_labelled_mask",
    "precision_labelled_mask",
    "recall_labelled_mask",
    "f1_labelled_mask",
    "mean_matched_iou_labelled_mask",
    "median_matched_iou_labelled_mask",
    "true_positives_harmonized",
    "false_positives_harmonized",
    "false_negatives_harmonized",
    "precision_harmonized",
    "recall_harmonized",
    "f1_harmonized",
    "mean_matched_iou_harmonized",
    "median_matched_iou_harmonized",
]
RUN_FIELDS = OUTPUT_FIELDS[1:6]
METRIC_FIELDS = OUTPUT_FIELDS[6:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one TreeX development or test summary CSV."
    )
    parser.add_argument("--plot-list", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_fields(
    value: dict[str, Any],
    fields: list[str],
    path: Path,
) -> None:
    missing = set(fields) - set(value)
    if missing:
        raise ValueError(f"{path} is missing fields {sorted(missing)}")


def main() -> int:
    args = parse_args()
    plot_list_path = Path(args.plot_list).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    with plot_list_path.open("r", encoding="utf-8", newline="") as handle:
        plot_rows = list(csv.DictReader(handle))
    if not plot_rows:
        raise ValueError(f"Plot list contains no rows: {plot_list_path}")
    if "plot_id" not in plot_rows[0]:
        raise ValueError(f"Plot list is missing plot_id: {plot_list_path}")

    plot_ids = [row["plot_id"].strip() for row in plot_rows]
    if any(not plot_id for plot_id in plot_ids):
        raise ValueError(f"Plot list contains an empty plot_id: {plot_list_path}")
    if len(set(plot_ids)) != len(plot_ids):
        raise ValueError(f"Plot list contains duplicate plot IDs: {plot_list_path}")

    file_suffix = "treex_test" if args.split == "test" else "treex"
    output_rows: list[dict[str, Any]] = []
    for plot_id in sorted(plot_ids):
        safe_plot_id = plot_id.replace("/", "_")
        summary_path = (
            results_root / f"{safe_plot_id}_{file_suffix}_summary.json"
        )
        metrics_path = (
            results_root / f"{safe_plot_id}_{file_suffix}_metrics.json"
        )
        summary = load_json(summary_path)
        metrics = load_json(metrics_path)
        require_fields(summary, ["plot_id", *RUN_FIELDS], summary_path)
        require_fields(metrics, ["plot_id", *METRIC_FIELDS], metrics_path)
        if summary["plot_id"] != plot_id or metrics["plot_id"] != plot_id:
            raise ValueError(f"Plot ID mismatch for {plot_id}")
        if int(summary["reference_tree_count_tree_classes"]) != int(
            metrics["reference_trees"]
        ):
            raise ValueError(f"Reference tree count mismatch for {plot_id}")
        output_rows.append(
            {
                "plot_id": plot_id,
                **{field: summary[field] for field in RUN_FIELDS},
                **{field: metrics[field] for field in METRIC_FIELDS},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
