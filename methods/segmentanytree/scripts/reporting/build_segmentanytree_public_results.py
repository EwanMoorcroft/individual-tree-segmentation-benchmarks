"""Build public-safe SegmentAnyTree/FOR-instance benchmark result tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PLOT_FIELDS = [
    "relative_path",
    "collection",
    "plot_name",
    "split",
    "point_count",
    "reference_tree_count",
    "predicted_tree_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "median_matched_iou",
    "iou_threshold",
    "coordinate_tolerance",
    "runtime_seconds",
    "peak_memory_gb",
    "evaluation_runtime_seconds",
    "status",
]

SUMMARY_FIELDS = [
    "group",
    "plot_count",
    "evaluated_plot_count",
    "completed_count",
    "failed_count",
    "total_reference_trees",
    "total_predicted_trees",
    "total_true_positives",
    "total_false_positives",
    "total_false_negatives",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_plot_f1",
    "mean_of_plot_mean_matched_iou",
    "mean_of_plot_median_matched_iou",
    "pooled_match_count",
    "pooled_mean_matched_iou",
    "pooled_median_matched_iou",
    "total_runtime_seconds",
    "mean_runtime_seconds",
    "peak_memory_max_gb",
]

INVENTORY_FIELDS = [
    "relative_path",
    "collection",
    "plot_name",
    "split",
    "point_count",
    "reference_tree_count",
    "classification_values",
    "has_treeID",
    "has_treeSP",
    "positive_treeID_point_count",
    "zero_treeID_point_count",
]

MATCH_FIELDS = [
    "collection",
    "plot_name",
    "split",
    "prediction",
    "reference",
    "iou",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    materialised = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialised)


def as_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field, "")
    return int(float(value)) if str(value).strip() else 0


def as_float(row: dict[str, Any], field: str) -> float:
    value = row.get(field, "")
    return float(value) if str(value).strip() else 0.0


def aggregate(
    rows: list[dict[str, Any]],
    group: str,
    match_ious: list[float],
) -> dict[str, Any]:
    true_positives = sum(as_int(row, "true_positives") for row in rows)
    false_positives = sum(as_int(row, "false_positives") for row in rows)
    false_negatives = sum(as_int(row, "false_negatives") for row in rows)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    micro_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    runtimes = [as_float(row, "runtime_seconds") for row in rows]
    memories = [as_float(row, "peak_memory_gb") for row in rows]
    statuses = [str(row.get("status", "")).strip().lower() for row in rows]
    completed = sum(status == "completed" for status in statuses)
    return {
        "group": group,
        "plot_count": len(rows),
        "evaluated_plot_count": len(rows),
        "completed_count": completed,
        "failed_count": len(rows) - completed,
        "total_reference_trees": sum(
            as_int(row, "reference_tree_count") for row in rows
        ),
        "total_predicted_trees": sum(
            as_int(row, "predicted_tree_count") for row in rows
        ),
        "total_true_positives": true_positives,
        "total_false_positives": false_positives,
        "total_false_negatives": false_negatives,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "mean_plot_f1": statistics.fmean(as_float(row, "f1") for row in rows),
        "mean_of_plot_mean_matched_iou": statistics.fmean(
            as_float(row, "mean_matched_iou") for row in rows
        ),
        "mean_of_plot_median_matched_iou": statistics.fmean(
            as_float(row, "median_matched_iou") for row in rows
        ),
        "pooled_match_count": len(match_ious),
        "pooled_mean_matched_iou": (
            statistics.fmean(match_ious) if match_ious else 0.0
        ),
        "pooled_median_matched_iou": (
            statistics.median(match_ious) if match_ious else 0.0
        ),
        "total_runtime_seconds": sum(runtimes),
        "mean_runtime_seconds": statistics.fmean(runtimes),
        "peak_memory_max_gb": max(memories),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize provisional released-checkpoint SegmentAnyTree/FOR-instance "
            "diagnostics."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing transferred results/tables and results/metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("methods/segmentanytree/examples"),
    )
    parser.add_argument(
        "--publication-status",
        default="provisional_coordinate_evaluation_revalidation_required",
        help=(
            "Status written to the public manifest. Use a final status only "
            "after the point-aligned evaluation gates pass."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tables = source_root / "results/tables"
    metadata = source_root / "results/metadata/segmentanytree_for_instance"
    per_plot_root = tables / "segmentanytree_for_instance/per_plot"

    source_plots = read_csv(
        tables / "segmentanytree_for_instance_plot_metrics.csv"
    )
    source_inventory = read_csv(metadata / "inventory.csv")
    if len(source_plots) != 32 or len(source_inventory) != 32:
        raise ValueError("Expected 32 plot metrics and 32 inventory rows")

    inventory_by_path = {
        row["relative_path"]: row for row in source_inventory
    }
    public_plots: list[dict[str, Any]] = []
    plot_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_plots:
        relative_path = row["relative_path"]
        inventory = inventory_by_path[relative_path]
        public = {
            field: row.get(field, "")
            for field in PLOT_FIELDS
        }
        public["point_count"] = inventory["point_count"]
        public_plots.append(public)
        plot_lookup[(row["collection"], row["plot_name"])] = public

    public_plots.sort(
        key=lambda row: (row["collection"], row["relative_path"])
    )

    public_matches: list[dict[str, Any]] = []
    match_ious_by_collection: dict[str, list[float]] = defaultdict(list)
    match_ious_by_split: dict[str, list[float]] = defaultdict(list)
    for path in sorted(per_plot_root.glob("*_matches.csv")):
        stem = path.stem.removesuffix("_matches")
        try:
            collection, plot_name = stem.split("_", 1)
        except ValueError as exc:
            raise ValueError(f"Unexpected match filename: {path.name}") from exc
        plot = plot_lookup[(collection, plot_name)]
        for match in read_csv(path):
            iou = float(match["iou"])
            public_matches.append(
                {
                    "collection": collection,
                    "plot_name": plot_name,
                    "split": plot["split"],
                    "prediction": match["prediction"],
                    "reference": match["reference"],
                    "iou": match["iou"],
                }
            )
            match_ious_by_collection[collection].append(iou)
            match_ious_by_split[str(plot["split"])].append(iou)

    if len(public_matches) != sum(
        as_int(row, "true_positives") for row in public_plots
    ):
        raise ValueError("Match row count does not equal total true positives")

    grouped_collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_plots:
        grouped_collection[str(row["collection"])].append(row)
        grouped_split[str(row["split"])].append(row)

    all_ious = [float(row["iou"]) for row in public_matches]
    overall = [aggregate(public_plots, "all", all_ious)]
    by_collection = [
        aggregate(
            grouped_collection[group],
            group,
            match_ious_by_collection[group],
        )
        for group in sorted(grouped_collection)
    ]
    by_split = [
        aggregate(grouped_split[group], group, match_ious_by_split[group])
        for group in sorted(grouped_split)
    ]

    public_inventory = [
        {
            "relative_path": row["relative_path"],
            "collection": row["collection"],
            "plot_name": Path(row["relative_path"]).stem,
            "split": row["split"],
            "point_count": row["point_count"],
            "reference_tree_count": row["reference_tree_count"],
            "classification_values": row["classification_values"],
            "has_treeID": row["has_treeID"],
            "has_treeSP": row["has_treeSP"],
            "positive_treeID_point_count": row[
                "positive_treeID_point_count"
            ],
            "zero_treeID_point_count": row["zero_treeID_point_count"],
        }
        for row in source_inventory
    ]
    public_inventory.sort(
        key=lambda row: (row["collection"], row["relative_path"])
    )

    prefix = "provisional_released_checkpoint"
    write_csv(output_dir / f"{prefix}_plot_metrics.csv", public_plots, PLOT_FIELDS)
    write_csv(output_dir / f"{prefix}_summary.csv", overall, SUMMARY_FIELDS)
    write_csv(
        output_dir / f"{prefix}_summary_by_collection.csv",
        by_collection,
        SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / f"{prefix}_summary_by_split.csv",
        by_split,
        SUMMARY_FIELDS,
    )
    write_csv(output_dir / f"{prefix}_matches.csv", public_matches, MATCH_FIELDS)
    write_csv(
        output_dir / f"{prefix}_inventory.csv",
        public_inventory,
        INVENTORY_FIELDS,
    )

    manifest = {
        "benchmark": "for_instance_segmentanytree",
        "status": args.publication_status,
        "evaluation_input": "provisional_coordinate_rematched_final_export",
        "historical_aligned_result_status": "completed_retained_historical",
        "historical_aligned_run_id": "sat_for_quicktune_to49_20260706_140730",
        "current_target_status": "completed_target_comparison",
        "plot_count": len(public_plots),
        "match_count": len(public_matches),
        "source_file_count": len(source_inventory),
        "public_tables": [
            f"{prefix}_plot_metrics.csv",
            f"{prefix}_summary.csv",
            f"{prefix}_summary_by_collection.csv",
            f"{prefix}_summary_by_split.csv",
            f"{prefix}_matches.csv",
            f"{prefix}_inventory.csv",
        ],
    }
    (output_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
