"""Audit SegmentAnyTree FOR-instance failure modes from aligned outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_ID = "sat_for_quicktune_to49_20260706_140730"
DEFAULT_SPLIT_ROOT = "results/metadata/segmentanytree_for_instance"
DEFAULT_TRAINING_MANIFEST = (
    "results/metadata/segmentanytree_for_instance/training_splits/"
    "full_split_manifest.json"
)
DEFAULT_OUTPUT_DIR = "results/tables/method_audit"

PLOT_FIELDS = [
    "run_id",
    "split",
    "site",
    "plot_name",
    "relative_path",
    "point_count",
    "evaluated_point_count",
    "reference_count",
    "prediction_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "mean_unweighted_coverage",
    "mean_weighted_coverage",
]
SITE_FIELDS = [
    "run_id",
    "split",
    "site",
    "n_plots",
    "mean_f1",
    "median_f1",
    "min_f1",
    "max_f1",
    "mean_precision",
    "mean_recall",
    "mean_matched_iou",
    "total_predictions",
    "total_references",
    "total_tp",
    "total_fp",
    "total_fn",
]
PREDICTION_FIELDS = [
    "run_id",
    "split",
    "site",
    "plot_name",
    "prediction_id",
    "point_count",
    "modal_predicted_semantic",
    "best_reference_id",
    "best_iou",
    "reference_tree_point_fraction",
    "failure_mode",
]
REFERENCE_FIELDS = [
    "run_id",
    "split",
    "site",
    "plot_name",
    "reference_id",
    "point_count",
    "best_prediction_id",
    "best_iou",
    "overlapping_prediction_count",
    "failure_mode",
]
DOMAIN_FIELDS = [
    "split",
    "training_role",
    "site",
    "plot_name",
    "relative_path",
    "selected_for_profile",
    "point_count",
    "reference_tree_count",
    "class_4_points",
    "class_5_points",
    "class_6_points",
    "tree_class_points",
    "ignored_class_points",
    "tree_class_fraction",
    "tree_size_min",
    "tree_size_median",
    "tree_size_p90",
    "tree_size_max",
]


def resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_pointwise_module() -> ModuleType:
    path = ROOT / "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py"
    spec = importlib.util.spec_from_file_location("pointwise_instance_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pointwise evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_number_set(text: str) -> set[float]:
    return {float(value.strip()) for value in text.split(",") if value.strip()}


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def as_float(value: Any) -> float:
    return float(value) if value not in ("", None) else 0.0


def quantile(values: list[int], fraction: float) -> int | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return int(ordered[index])


def modal_value(values: np.ndarray) -> str:
    if not len(values):
        return ""
    unique, counts = np.unique(values, return_counts=True)
    return str(scalar(unique[int(np.argmax(counts))]))


def metric_json_paths(split_root: Path, run_id: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for split in ("trained_validation", "trained_test"):
        root = split_root / split / run_id
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            paths.append((split, path))
    return paths


def load_metric_rows(split_root: Path, run_id: str) -> list[tuple[str, Path, dict[str, Any]]]:
    rows: list[tuple[str, Path, dict[str, Any]]] = []
    for split, path in metric_json_paths(split_root, run_id):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("evaluator") != "pointwise_instance_metrics":
            continue
        rows.append((split, path, payload))
    return rows


def classify_prediction(
    best_iou: float,
    point_count: int,
    reference_tree_fraction: float,
    near_miss_iou: float,
    large_instance_points: int,
) -> str:
    if best_iou >= near_miss_iou:
        return "near_miss"
    if reference_tree_fraction < 0.5:
        return "background_confusion"
    if point_count >= large_instance_points:
        return "large_extra_instance"
    return "small_extra_instance"


def classify_reference(
    best_iou: float,
    overlapping_prediction_count: int,
    near_miss_iou: float,
) -> str:
    if best_iou >= near_miss_iou:
        return "near_miss"
    if overlapping_prediction_count >= 2:
        return "fragmented"
    return "missed_tree"


def build_matrix_context(
    pointwise: ModuleType,
    labels: Any,
    reference_tree_classes: set[float],
    prediction_tree_classes: set[float],
    ignored_reference_labels: set[float],
    ignored_prediction_labels: set[float],
) -> dict[str, Any]:
    reference_tree = np.isin(
        labels.reference_semantic.astype(np.float64),
        list(reference_tree_classes),
    )
    predicted_tree = np.isin(
        labels.predicted_semantic.astype(np.float64),
        list(prediction_tree_classes),
    )
    evaluation_mask = reference_tree | predicted_tree
    predicted_instance = labels.predicted_instance[evaluation_mask]
    reference_instance = labels.reference_instance[evaluation_mask]
    predicted_semantic = labels.predicted_semantic[evaluation_mask]
    reference_semantic = labels.reference_semantic[evaluation_mask]
    predicted_ids = pointwise.allowed_instance_ids(
        predicted_instance,
        predicted_semantic,
        ignored_prediction_labels,
        prediction_tree_classes,
    )
    reference_ids = pointwise.allowed_instance_ids(
        reference_instance,
        reference_semantic,
        ignored_reference_labels,
        reference_tree_classes,
    )
    matrix = pointwise.contingency_iou(
        predicted_instance,
        reference_instance,
        predicted_ids,
        reference_ids,
    )
    return {
        "predicted_instance": predicted_instance,
        "reference_instance": reference_instance,
        "predicted_semantic": predicted_semantic,
        "reference_semantic": reference_semantic,
        "predicted_ids": predicted_ids,
        "reference_ids": reference_ids,
        "matrix": matrix,
    }


def analyse_labels(
    pointwise: ModuleType,
    labels: Any,
    payload: dict[str, Any],
    split: str,
    run_id: str,
    iou_threshold: float,
    near_miss_iou: float,
    fragmentation_iou: float,
    large_instance_points: int,
    reference_tree_classes: set[float],
    prediction_tree_classes: set[float],
    ignored_reference_labels: set[float],
    ignored_prediction_labels: set[float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result = pointwise.evaluate_pointwise(
        labels,
        reference_tree_classes=reference_tree_classes,
        prediction_tree_classes=prediction_tree_classes,
        ignored_reference_labels=ignored_reference_labels,
        ignored_prediction_labels=ignored_prediction_labels,
        iou_threshold=iou_threshold,
    )
    context = build_matrix_context(
        pointwise,
        labels,
        reference_tree_classes,
        prediction_tree_classes,
        ignored_reference_labels,
        ignored_prediction_labels,
    )
    matrix = context["matrix"]
    predicted_ids = context["predicted_ids"]
    reference_ids = context["reference_ids"]
    predicted_instance = context["predicted_instance"]
    reference_instance = context["reference_instance"]
    predicted_semantic = context["predicted_semantic"]
    reference_semantic = context["reference_semantic"]
    matches = pointwise.maximum_threshold_matching(matrix, iou_threshold)
    matched_predictions = {prediction for prediction, _ in matches}
    matched_references = {reference for _, reference in matches}

    harmonized = result["harmonized"]
    plot_row = {
        "run_id": run_id,
        "split": split,
        "site": payload.get("collection", ""),
        "plot_name": payload.get("plot_name", ""),
        "relative_path": payload.get("relative_path", ""),
        "point_count": result["point_count"],
        "evaluated_point_count": result["evaluated_point_count"],
        "reference_count": result["reference_instance_count"],
        "prediction_count": result["prediction_instance_count"],
        "true_positives": harmonized["true_positives"],
        "false_positives": harmonized["false_positives"],
        "false_negatives": harmonized["false_negatives"],
        "precision": harmonized["precision"],
        "recall": harmonized["recall"],
        "f1": harmonized["f1"],
        "mean_matched_iou": harmonized["mean_matched_iou"],
        "mean_unweighted_coverage": result["mean_unweighted_coverage"],
        "mean_weighted_coverage": result["mean_weighted_coverage"],
    }

    prediction_rows: list[dict[str, Any]] = []
    for index, prediction_id in enumerate(predicted_ids):
        if index in matched_predictions:
            continue
        instance_mask = predicted_instance == prediction_id
        best_index = int(np.argmax(matrix[index])) if matrix.shape[1] else -1
        best_iou = float(matrix[index, best_index]) if best_index >= 0 else 0.0
        reference_tree_points = np.isin(
            reference_semantic[instance_mask].astype(np.float64),
            list(reference_tree_classes),
        )
        reference_tree_fraction = (
            float(np.mean(reference_tree_points)) if np.any(instance_mask) else 0.0
        )
        point_count = int(np.count_nonzero(instance_mask))
        prediction_rows.append(
            {
                "run_id": run_id,
                "split": split,
                "site": payload.get("collection", ""),
                "plot_name": payload.get("plot_name", ""),
                "prediction_id": scalar(prediction_id),
                "point_count": point_count,
                "modal_predicted_semantic": modal_value(predicted_semantic[instance_mask]),
                "best_reference_id": (
                    scalar(reference_ids[best_index]) if best_index >= 0 else ""
                ),
                "best_iou": best_iou,
                "reference_tree_point_fraction": reference_tree_fraction,
                "failure_mode": classify_prediction(
                    best_iou,
                    point_count,
                    reference_tree_fraction,
                    near_miss_iou,
                    large_instance_points,
                ),
            }
        )

    reference_rows: list[dict[str, Any]] = []
    for index, reference_id in enumerate(reference_ids):
        if index in matched_references:
            continue
        instance_mask = reference_instance == reference_id
        best_index = int(np.argmax(matrix[:, index])) if matrix.shape[0] else -1
        best_iou = float(matrix[best_index, index]) if best_index >= 0 else 0.0
        overlapping_count = (
            int(np.count_nonzero(matrix[:, index] >= fragmentation_iou))
            if matrix.shape[0]
            else 0
        )
        reference_rows.append(
            {
                "run_id": run_id,
                "split": split,
                "site": payload.get("collection", ""),
                "plot_name": payload.get("plot_name", ""),
                "reference_id": scalar(reference_id),
                "point_count": int(np.count_nonzero(instance_mask)),
                "best_prediction_id": (
                    scalar(predicted_ids[best_index]) if best_index >= 0 else ""
                ),
                "best_iou": best_iou,
                "overlapping_prediction_count": overlapping_count,
                "failure_mode": classify_reference(
                    best_iou,
                    overlapping_count,
                    near_miss_iou,
                ),
            }
        )

    return plot_row, prediction_rows, reference_rows


def load_labels_from_payload(
    pointwise: ModuleType,
    payload: dict[str, Any],
    semantic_offset: float,
    reference_background_labels: set[float],
    ignored_reference_labels: set[float],
    reference_tree_classes: set[float],
) -> Any:
    inputs = payload.get("inputs", {})
    labels = pointwise.load_internal_evaluation(
        Path(inputs["instance_evaluation_ply"]),
        Path(inputs["semantic_evaluation_ply"]),
        "preds",
        "gt",
        "preds",
        "gt",
        semantic_offset,
    )
    if pointwise.reference_semantic_requires_instance_fallback(
        labels,
        reference_background_labels,
        ignored_reference_labels,
        reference_tree_classes,
    ):
        labels = pointwise.derive_reference_semantic_from_instance(
            labels,
            reference_background_labels,
            reference_tree_classes,
            ignored_reference_labels,
        )
    return labels


def site_rows(run_id: str, plot_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        grouped[(str(row["split"]), str(row["site"]))].append(row)

    rows = []
    for (split, site), values in sorted(grouped.items()):
        rows.append(
            {
                "run_id": run_id,
                "split": split,
                "site": site,
                "n_plots": len(values),
                "mean_f1": st.fmean(as_float(row["f1"]) for row in values),
                "median_f1": st.median(as_float(row["f1"]) for row in values),
                "min_f1": min(as_float(row["f1"]) for row in values),
                "max_f1": max(as_float(row["f1"]) for row in values),
                "mean_precision": st.fmean(
                    as_float(row["precision"]) for row in values
                ),
                "mean_recall": st.fmean(as_float(row["recall"]) for row in values),
                "mean_matched_iou": st.fmean(
                    as_float(row["mean_matched_iou"]) for row in values
                ),
                "total_predictions": sum(
                    int(row["prediction_count"]) for row in values
                ),
                "total_references": sum(int(row["reference_count"]) for row in values),
                "total_tp": sum(int(row["true_positives"]) for row in values),
                "total_fp": sum(int(row["false_positives"]) for row in values),
                "total_fn": sum(int(row["false_negatives"]) for row in values),
            }
        )
    return rows


def domain_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "split": record.get("dataset_split", ""),
        "training_role": record.get("training_role", ""),
        "site": record.get("collection", ""),
        "plot_name": record.get("plot_name", ""),
        "relative_path": record.get("relative_path", ""),
        "selected_for_profile": record.get("selected_for_profile", ""),
        "point_count": record.get("point_count", ""),
        "reference_tree_count": record.get("reference_tree_count", ""),
        "class_4_points": "",
        "class_5_points": "",
        "class_6_points": "",
        "tree_class_points": "",
        "ignored_class_points": "",
        "tree_class_fraction": "",
        "tree_size_min": "",
        "tree_size_median": "",
        "tree_size_p90": "",
        "tree_size_max": "",
    }
    source_path = Path(str(record.get("source_path", ""))).expanduser()
    if not source_path.is_file():
        return row

    import laspy

    cloud = laspy.read(source_path)
    classification = np.asarray(cloud.classification, dtype=np.int64)
    tree_ids = np.asarray(cloud["treeID"], dtype=np.int64)
    tree_sizes = [
        int(np.count_nonzero(tree_ids == tree_id))
        for tree_id in np.unique(tree_ids)
        if int(tree_id) > 0
    ]
    class_counts = {
        class_id: int(np.count_nonzero(classification == class_id))
        for class_id in (4, 5, 6)
    }
    tree_class_points = sum(class_counts.values())
    ignored_class_points = int(np.count_nonzero(np.isin(classification, [0, 1, 2, 3])))
    point_count = int(len(classification))
    row.update(
        {
            "point_count": point_count,
            "reference_tree_count": len(tree_sizes),
            "class_4_points": class_counts[4],
            "class_5_points": class_counts[5],
            "class_6_points": class_counts[6],
            "tree_class_points": tree_class_points,
            "ignored_class_points": ignored_class_points,
            "tree_class_fraction": (
                float(tree_class_points) / point_count if point_count else 0.0
            ),
            "tree_size_min": min(tree_sizes) if tree_sizes else "",
            "tree_size_median": quantile(tree_sizes, 0.5),
            "tree_size_p90": quantile(tree_sizes, 0.9),
            "tree_size_max": max(tree_sizes) if tree_sizes else "",
        }
    )
    return row


def domain_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [domain_row(record) for record in manifest.get("records", [])]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args: argparse.Namespace) -> dict[str, Path]:
    pointwise = load_pointwise_module()
    split_root = resolve(args.split_root)
    output_dir = resolve(args.output_dir)
    training_manifest = resolve(args.training_manifest)
    reference_tree_classes = parse_number_set(args.reference_tree_classes)
    prediction_tree_classes = parse_number_set(args.prediction_tree_classes)
    ignored_reference_labels = parse_number_set(args.ignored_reference_labels)
    ignored_prediction_labels = parse_number_set(args.ignored_prediction_labels)
    reference_background_labels = parse_number_set(
        args.reference_background_instance_labels
    )

    plot_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for split, _, payload in load_metric_rows(split_root, args.run_id):
        labels = load_labels_from_payload(
            pointwise,
            payload,
            args.semantic_offset,
            reference_background_labels,
            ignored_reference_labels,
            reference_tree_classes,
        )
        plot_row, plot_prediction_rows, plot_reference_rows = analyse_labels(
            pointwise,
            labels,
            payload,
            split,
            args.run_id,
            args.iou_threshold,
            args.near_miss_iou,
            args.fragmentation_iou,
            args.large_instance_points,
            reference_tree_classes,
            prediction_tree_classes,
            ignored_reference_labels,
            ignored_prediction_labels,
        )
        plot_rows.append(plot_row)
        prediction_rows.extend(plot_prediction_rows)
        reference_rows.extend(plot_reference_rows)

    outputs = {
        "plot": output_dir / f"sat_plot_failure_modes_{args.run_id}.csv",
        "site": output_dir / f"sat_site_failure_modes_{args.run_id}.csv",
        "prediction": output_dir
        / f"sat_unmatched_prediction_audit_{args.run_id}.csv",
        "reference": output_dir / f"sat_unmatched_reference_audit_{args.run_id}.csv",
        "domain": output_dir
        / f"sat_training_vs_validation_domain_audit_{args.run_id}.csv",
    }
    write_csv(outputs["plot"], PLOT_FIELDS, plot_rows)
    write_csv(outputs["site"], SITE_FIELDS, site_rows(args.run_id, plot_rows))
    write_csv(outputs["prediction"], PREDICTION_FIELDS, prediction_rows)
    write_csv(outputs["reference"], REFERENCE_FIELDS, reference_rows)
    write_csv(outputs["domain"], DOMAIN_FIELDS, domain_rows(training_manifest))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SegmentAnyTree aligned FOR-instance failure modes."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--training-manifest", default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--semantic-offset", type=float, default=1.0)
    parser.add_argument("--reference-tree-classes", default="2")
    parser.add_argument("--prediction-tree-classes", default="2")
    parser.add_argument("--ignored-reference-labels", default="-1")
    parser.add_argument("--ignored-prediction-labels", default="-1,0")
    parser.add_argument("--reference-background-instance-labels", default="1")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--near-miss-iou", type=float, default=0.25)
    parser.add_argument("--fragmentation-iou", type=float, default=0.1)
    parser.add_argument("--large-instance-points", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    outputs = run_audit(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
