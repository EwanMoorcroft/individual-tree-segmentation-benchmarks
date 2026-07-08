"""Sweep validation-only SegmentAnyTree post-processing settings."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics as st
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ID = "sat_for_quicktune_to49_20260706_140730"
DEFAULT_SPLIT_ROOT = "results/metadata/segmentanytree_for_instance"
DEFAULT_OUTPUT_DIR = "results/tables/method_audit"
DEFAULT_MIN_POINTS = "0,500,1000,2500,5000,10000"
DEFAULT_TREE_FRACTIONS = "0,0.6,0.7,0.8,0.9,0.95"

FIELDS = [
    "run_id",
    "selected_using_split",
    "min_predicted_instance_points",
    "min_predicted_tree_fraction",
    "n_plots",
    "mean_f1",
    "median_f1",
    "min_f1",
    "max_f1",
    "mean_precision",
    "mean_recall",
    "total_predictions",
    "total_references",
    "total_tp",
    "total_fp",
    "total_fn",
    "protocol_note",
]


def resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    if any(value < 0 for value in values):
        raise ValueError("Minimum instance point values cannot be negative")
    return sorted(set(values))


def parse_floats(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one floating point value is required")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Tree fraction values must be in [0, 1]")
    return sorted(set(values))


def mean(values: list[float]) -> float:
    return float(st.mean(values)) if values else 0.0


def summarize_candidate(
    run_id: str,
    split: str,
    min_points: int,
    min_tree_fraction: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    f1 = [float(result["harmonized"]["f1"]) for result in results]
    precision = [float(result["harmonized"]["precision"]) for result in results]
    recall = [float(result["harmonized"]["recall"]) for result in results]
    return {
        "run_id": run_id,
        "selected_using_split": split,
        "min_predicted_instance_points": min_points,
        "min_predicted_tree_fraction": min_tree_fraction,
        "n_plots": len(results),
        "mean_f1": mean(f1),
        "median_f1": float(st.median(f1)) if f1 else 0.0,
        "min_f1": min(f1) if f1 else 0.0,
        "max_f1": max(f1) if f1 else 0.0,
        "mean_precision": mean(precision),
        "mean_recall": mean(recall),
        "total_predictions": sum(
            int(result["prediction_instance_count"]) for result in results
        ),
        "total_references": sum(
            int(result["reference_instance_count"]) for result in results
        ),
        "total_tp": sum(
            int(result["harmonized"]["true_positives"]) for result in results
        ),
        "total_fp": sum(
            int(result["harmonized"]["false_positives"]) for result in results
        ),
        "total_fn": sum(
            int(result["harmonized"]["false_negatives"]) for result in results
        ),
        "protocol_note": "validation_only_selection_no_test_access",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> dict[str, Path]:
    audit = load_script(
        ROOT / "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py",
        "sat_failure_audit_for_validation_sweep",
    )
    pointwise = audit.load_pointwise_module()

    split_root = resolve(args.split_root)
    rows = [
        payload
        for split, _, payload in audit.load_metric_rows(split_root, args.run_id)
        if split == args.split
    ]
    if not rows:
        raise RuntimeError(
            f"No metric JSON files found for {args.split}/{args.run_id}"
        )

    reference_tree_classes = audit.parse_number_set(args.reference_tree_classes)
    prediction_tree_classes = audit.parse_number_set(args.prediction_tree_classes)
    ignored_reference_labels = audit.parse_number_set(args.ignored_reference_labels)
    ignored_prediction_labels = audit.parse_number_set(args.ignored_prediction_labels)
    reference_background_labels = audit.parse_number_set(
        args.reference_background_instance_labels
    )

    labels_by_plot = []
    for index, payload in enumerate(rows, start=1):
        print(
            "loading",
            index,
            "of",
            len(rows),
            payload.get("collection"),
            payload.get("plot_name"),
            flush=True,
        )
        labels = audit.load_labels_from_payload(
            pointwise,
            payload,
            args.semantic_offset,
            reference_background_labels,
            ignored_reference_labels,
            reference_tree_classes,
        )
        labels_by_plot.append(labels)

    candidates: list[dict[str, Any]] = []
    for min_points in parse_ints(args.min_predicted_instance_points):
        for min_tree_fraction in parse_floats(args.min_predicted_tree_fraction):
            results = [
                pointwise.evaluate_pointwise(
                    labels,
                    reference_tree_classes=reference_tree_classes,
                    prediction_tree_classes=prediction_tree_classes,
                    ignored_reference_labels=ignored_reference_labels,
                    ignored_prediction_labels=ignored_prediction_labels,
                    iou_threshold=args.iou_threshold,
                    min_predicted_instance_points=min_points,
                    min_predicted_tree_fraction=min_tree_fraction,
                )
                for labels in labels_by_plot
            ]
            row = summarize_candidate(
                args.run_id,
                args.split,
                min_points,
                min_tree_fraction,
                results,
            )
            candidates.append(row)
            print(
                "candidate",
                "min_points",
                min_points,
                "tree_fraction",
                min_tree_fraction,
                "mean_f1",
                round(float(row["mean_f1"]), 6),
                "fp",
                row["total_fp"],
                "fn",
                row["total_fn"],
                flush=True,
            )

    candidates.sort(
        key=lambda row: (
            float(row["mean_f1"]),
            float(row["min_f1"]),
            float(row["mean_precision"]),
            -int(row["total_predictions"]),
        ),
        reverse=True,
    )
    best = [candidates[0]]

    output_dir = resolve(args.output_dir)
    outputs = {
        "sweep": output_dir / f"sat_validation_postprocess_sweep_{args.run_id}.csv",
        "best": output_dir / f"sat_validation_postprocess_best_{args.run_id}.csv",
    }
    write_csv(outputs["sweep"], candidates)
    write_csv(outputs["best"], best)
    print("BEST=", best[0], flush=True)
    print("SWEEP_CSV=", outputs["sweep"], flush=True)
    print("BEST_CSV=", outputs["best"], flush=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select SegmentAnyTree post-processing settings from validation "
            "outputs only. The script does not read held-out test results."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--split", default="trained_validation")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-predicted-instance-points",
        default=DEFAULT_MIN_POINTS,
        help="Comma-separated grid, e.g. 0,500,1000,2500,5000.",
    )
    parser.add_argument(
        "--min-predicted-tree-fraction",
        default=DEFAULT_TREE_FRACTIONS,
        help="Comma-separated grid in [0,1], e.g. 0,0.7,0.8,0.9.",
    )
    parser.add_argument("--semantic-offset", type=float, default=1.0)
    parser.add_argument("--reference-tree-classes", default="2")
    parser.add_argument("--prediction-tree-classes", default="2")
    parser.add_argument("--ignored-reference-labels", default="-1")
    parser.add_argument("--ignored-prediction-labels", default="-1,0")
    parser.add_argument("--reference-background-instance-labels", default="1")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    run_sweep(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
