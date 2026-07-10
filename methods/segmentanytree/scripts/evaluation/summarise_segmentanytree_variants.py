"""Validate and summarise aligned SegmentAnyTree variant metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=METRICS_ROOT for --variant.")
    label, root = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty.")
    return label, resolve_path(root)


def metric_payloads(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Metrics root does not exist: {root}")
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluator") != "pointwise_instance_metrics":
            continue
        payload["_source"] = str(path)
        payloads.append(payload)
    return payloads


def summarise_variant(
    label: str,
    root: Path,
    expected_plots: int,
    expected_split: str | None,
    require_any_predictions: bool,
) -> dict[str, Any]:
    payloads = metric_payloads(root)
    if len(payloads) != expected_plots:
        raise ValueError(
            f"{label}: expected {expected_plots} metric files, found {len(payloads)}"
        )
    plot_keys = [(item.get("collection"), item.get("plot_name")) for item in payloads]
    if len(set(plot_keys)) != len(plot_keys):
        raise ValueError(f"{label}: duplicate collection/plot metric records")
    if expected_split:
        wrong = [
            item.get("relative_path")
            for item in payloads
            if item.get("split") != expected_split
        ]
        if wrong:
            raise ValueError(
                f"{label}: metrics contain plots outside split "
                f"{expected_split}: {wrong}"
            )
    if any(int(item.get("point_count", 0)) <= 0 for item in payloads):
        raise ValueError(f"{label}: one or more metric records has no aligned points")
    if any(int(item.get("reference_instance_count", 0)) <= 0 for item in payloads):
        raise ValueError(f"{label}: one or more metric records has no reference trees")

    predicted_instances = sum(
        int(item.get("prediction_instance_count", 0)) for item in payloads
    )
    if require_any_predictions and predicted_instances <= 0:
        raise ValueError(f"{label}: aligned outputs contain zero predicted instances")

    harmonized = [item["harmonized"] for item in payloads]
    paper = [item["paper_compatible"] for item in payloads]
    true_positives = sum(int(item["true_positives"]) for item in harmonized)
    false_positives = sum(int(item["false_positives"]) for item in harmonized)
    false_negatives = sum(int(item["false_negatives"]) for item in harmonized)
    micro_precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    micro_recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    micro_f1 = (
        2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        if 2 * true_positives + false_positives + false_negatives
        else 0.0
    )
    return {
        "variant": label,
        "result_status": "completed_aligned_pointwise_test",
        "dataset_split": expected_split or "mixed",
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "metrics_root": str(root),
        "plots": len(payloads),
        "predicted_instances": predicted_instances,
        "reference_instances": sum(
            int(item["reference_instance_count"]) for item in payloads
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_plot_f1": statistics.fmean(float(item["f1"]) for item in harmonized),
        "mean_plot_precision": statistics.fmean(
            float(item["precision"]) for item in harmonized
        ),
        "mean_plot_recall": statistics.fmean(
            float(item["recall"]) for item in harmonized
        ),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_matched_iou": statistics.fmean(
            float(item["mean_matched_iou"]) for item in harmonized
        ),
        "mean_unweighted_coverage": statistics.fmean(
            float(item["mean_unweighted_coverage"]) for item in payloads
        ),
        "mean_weighted_coverage": statistics.fmean(
            float(item["mean_weighted_coverage"]) for item in payloads
        ),
        "paper_mean_f1": statistics.fmean(float(item["f1"]) for item in paper),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate complete aligned metric sets and write one row per SAT variant."
        )
    )
    parser.add_argument("--variant", action="append", required=True, type=parse_variant)
    parser.add_argument("--expected-plots", type=int, required=True)
    parser.add_argument("--expected-split")
    parser.add_argument("--require-any-predictions", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_plots <= 0:
        raise ValueError("--expected-plots must be positive")
    rows = [
        summarise_variant(
            label,
            root,
            args.expected_plots,
            args.expected_split,
            args.require_any_predictions,
        )
        for label, root in args.variant
    ]
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['variant']}: plots={row['plots']} "
            f"mean_plot_f1={row['mean_plot_f1']:.4f} "
            f"micro_f1={row['micro_f1']:.4f}"
        )
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
