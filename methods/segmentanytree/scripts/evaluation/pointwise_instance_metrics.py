"""Evaluate aligned point-wise instance labels.

This evaluator is intended for method outputs that retain one prediction and
one reference label for every evaluated point. It avoids coordinate joins and
reports both the matching policy used by the published SegmentAnyTree
implementation and a strict one-to-one assignment for cross-method comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class PointLabels:
    predicted_instance: np.ndarray
    reference_instance: np.ndarray
    predicted_semantic: np.ndarray
    reference_semantic: np.ndarray


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_ply_fields(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    from plyfile import PlyData

    data = PlyData.read(str(path))
    if not data.elements:
        raise ValueError(f"PLY file contains no elements: {path}")
    vertices = data.elements[0].data
    available = set(vertices.dtype.names or ())
    missing = set(fields) - available
    if missing:
        raise ValueError(
            f"PLY file is missing fields {sorted(missing)}: {path}; "
            f"available fields: {sorted(available)}"
        )
    return {field: np.asarray(vertices[field]) for field in fields}


def load_internal_evaluation(
    instance_path: Path,
    semantic_path: Path,
    predicted_instance_field: str,
    reference_instance_field: str,
    predicted_semantic_field: str,
    reference_semantic_field: str,
    semantic_offset: float,
) -> PointLabels:
    instances = read_ply_fields(
        instance_path,
        [predicted_instance_field, reference_instance_field],
    )
    semantics = read_ply_fields(
        semantic_path,
        [predicted_semantic_field, reference_semantic_field],
    )
    lengths = {
        len(instances[predicted_instance_field]),
        len(instances[reference_instance_field]),
        len(semantics[predicted_semantic_field]),
        len(semantics[reference_semantic_field]),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Internal instance and semantic evaluation files are not point-aligned; "
            f"observed lengths: {sorted(lengths)}"
        )
    return PointLabels(
        predicted_instance=instances[predicted_instance_field],
        reference_instance=instances[reference_instance_field],
        predicted_semantic=(
            semantics[predicted_semantic_field].astype(np.float64)
            + semantic_offset
        ),
        reference_semantic=(
            semantics[reference_semantic_field].astype(np.float64)
            + semantic_offset
        ),
    )


def load_combined_labelled_cloud(
    path: Path,
    predicted_instance_field: str,
    reference_instance_field: str,
    predicted_semantic_field: str,
    reference_semantic_field: str,
) -> PointLabels:
    fields = [
        predicted_instance_field,
        reference_instance_field,
        predicted_semantic_field,
        reference_semantic_field,
    ]
    if path.suffix.lower() == ".ply":
        values = read_ply_fields(path, fields)
    elif path.suffix.lower() in {".las", ".laz"}:
        import laspy

        cloud = laspy.read(path)
        available = set(cloud.point_format.dimension_names)
        missing = set(fields) - available
        if missing:
            raise ValueError(
                f"Point cloud is missing fields {sorted(missing)}: {path}; "
                f"available fields: {sorted(available)}"
            )
        values = {field: np.asarray(cloud[field]) for field in fields}
    else:
        raise ValueError(f"Unsupported labelled point cloud: {path}")
    return PointLabels(
        predicted_instance=values[predicted_instance_field],
        reference_instance=values[reference_instance_field],
        predicted_semantic=values[predicted_semantic_field],
        reference_semantic=values[reference_semantic_field],
    )


def derive_reference_semantic_from_instance(
    labels: PointLabels,
    background_labels: set[float],
    tree_classes: set[float],
    ignored_labels: set[float] | None = None,
) -> PointLabels:
    """Recover a binary reference mask from aligned instance labels."""

    if len(tree_classes) != 1:
        raise ValueError(
            "Instance-derived reference semantics require one tree class."
        )
    excluded_labels = background_labels | (ignored_labels or set())
    return PointLabels(
        predicted_instance=labels.predicted_instance,
        reference_instance=labels.reference_instance,
        predicted_semantic=labels.predicted_semantic,
        reference_semantic=np.where(
            np.isin(
                labels.reference_instance,
                list(excluded_labels),
            ),
            np.nan,
            next(iter(tree_classes)),
        ),
    )


def reference_semantic_requires_instance_fallback(
    labels: PointLabels,
    background_labels: set[float],
    ignored_labels: set[float],
    tree_classes: set[float],
) -> bool:
    """Return whether aligned instances contain trees missing from semantics."""

    has_semantic_trees = bool(
        np.any(
            np.isin(
                labels.reference_semantic.astype(np.float64),
                list(tree_classes),
            )
        )
    )
    excluded_labels = background_labels | ignored_labels
    has_reference_instances = bool(
        np.any(
            ~np.isin(
                labels.reference_instance.astype(np.float64),
                list(excluded_labels),
            )
        )
    )
    return has_reference_instances and not has_semantic_trees


def label_is_ignored(values: np.ndarray, ignored: set[float]) -> np.ndarray:
    return np.isin(values.astype(np.float64), list(ignored))


def allowed_instance_ids(
    instances: np.ndarray,
    semantics: np.ndarray,
    ignored_instance_labels: set[float],
    tree_semantic_classes: set[float],
) -> np.ndarray:
    allowed: list[Any] = []
    for instance_id in np.unique(instances):
        if float(instance_id) in ignored_instance_labels:
            continue
        points = instances == instance_id
        semantic_values, counts = np.unique(semantics[points], return_counts=True)
        modal_semantic = semantic_values[int(np.argmax(counts))]
        if float(modal_semantic) in tree_semantic_classes:
            allowed.append(instance_id)
    return np.asarray(allowed, dtype=instances.dtype)


def contingency_iou(
    predicted: np.ndarray,
    reference: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(predicted_ids), len(reference_ids)), dtype=np.float64)
    if not len(predicted_ids) or not len(reference_ids):
        return matrix

    predicted_index = {value.item(): index for index, value in enumerate(predicted_ids)}
    reference_index = {value.item(): index for index, value in enumerate(reference_ids)}
    predicted_counts = {
        value.item(): int(np.count_nonzero(predicted == value))
        for value in predicted_ids
    }
    reference_counts = {
        value.item(): int(np.count_nonzero(reference == value))
        for value in reference_ids
    }

    valid = np.isin(predicted, predicted_ids) & np.isin(reference, reference_ids)
    if not np.any(valid):
        return matrix
    pairs = np.column_stack([predicted[valid], reference[valid]])
    unique_pairs, intersections = np.unique(pairs, axis=0, return_counts=True)
    for pair, intersection in zip(unique_pairs, intersections):
        prediction_id = pair[0].item()
        reference_id = pair[1].item()
        union = (
            predicted_counts[prediction_id]
            + reference_counts[reference_id]
            - int(intersection)
        )
        matrix[
            predicted_index[prediction_id],
            reference_index[reference_id],
        ] = float(intersection) / union if union else 0.0
    return matrix


def maximum_threshold_matching(
    matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    candidates = [
        [
            int(index)
            for index in np.argsort(matrix[row])[::-1]
            if matrix[row, index] >= threshold
        ]
        for row in range(matrix.shape[0])
    ]
    reference_owner: dict[int, int] = {}

    def assign(prediction: int, seen: set[int]) -> bool:
        for reference in candidates[prediction]:
            if reference in seen:
                continue
            seen.add(reference)
            owner = reference_owner.get(reference)
            if owner is None or assign(owner, seen):
                reference_owner[reference] = prediction
                return True
        return False

    order = sorted(
        range(matrix.shape[0]),
        key=lambda row: float(np.max(matrix[row])) if matrix.shape[1] else 0.0,
        reverse=True,
    )
    for prediction in order:
        assign(prediction, set())
    return sorted(
        (prediction, reference)
        for reference, prediction in reference_owner.items()
    )


def metric_values(
    true_positives: int,
    prediction_count: int,
    reference_count: int,
) -> dict[str, float | int]:
    false_positives = prediction_count - true_positives
    false_negatives = max(reference_count - true_positives, 0)
    precision = true_positives / prediction_count if prediction_count else 0.0
    recall = true_positives / reference_count if reference_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_pointwise(
    labels: PointLabels,
    reference_tree_classes: set[float],
    prediction_tree_classes: set[float],
    ignored_reference_labels: set[float],
    ignored_prediction_labels: set[float],
    iou_threshold: float,
    min_predicted_instance_points: int = 0,
    min_predicted_tree_fraction: float = 0.0,
) -> dict[str, Any]:
    """Evaluate aligned labels with paper-compatible and one-to-one matching."""

    lengths = {
        len(labels.predicted_instance),
        len(labels.reference_instance),
        len(labels.predicted_semantic),
        len(labels.reference_semantic),
    }
    if len(lengths) != 1:
        raise ValueError(f"Point label arrays are not aligned: {sorted(lengths)}")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    if min_predicted_instance_points < 0:
        raise ValueError("Minimum predicted instance points cannot be negative")
    if not 0 <= min_predicted_tree_fraction <= 1:
        raise ValueError(
            "Minimum predicted tree fraction must be in the interval [0, 1]"
        )

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

    predicted_ids = allowed_instance_ids(
        predicted_instance,
        predicted_semantic,
        ignored_prediction_labels,
        prediction_tree_classes,
    )
    if min_predicted_instance_points:
        predicted_ids = np.asarray(
            [
                instance_id
                for instance_id in predicted_ids
                if np.count_nonzero(predicted_instance == instance_id)
                >= min_predicted_instance_points
            ],
            dtype=predicted_ids.dtype,
        )
    if min_predicted_tree_fraction:
        predicted_ids = np.asarray(
            [
                instance_id
                for instance_id in predicted_ids
                if (
                    np.count_nonzero(
                        np.isin(
                            labels.predicted_semantic[
                                labels.predicted_instance == instance_id
                            ],
                            list(prediction_tree_classes),
                        )
                    )
                    / np.count_nonzero(labels.predicted_instance == instance_id)
                )
                >= min_predicted_tree_fraction
            ],
            dtype=predicted_ids.dtype,
        )
    reference_ids = allowed_instance_ids(
        reference_instance,
        reference_semantic,
        ignored_reference_labels,
        reference_tree_classes,
    )
    matrix = contingency_iou(
        predicted_instance,
        reference_instance,
        predicted_ids,
        reference_ids,
    )

    paper_best_reference = (
        np.argmax(matrix, axis=1)
        if matrix.shape[1]
        else np.zeros(matrix.shape[0], dtype=int)
    )
    paper_best_iou = (
        np.max(matrix, axis=1)
        if matrix.shape[1]
        else np.zeros(matrix.shape[0], dtype=float)
    )
    paper_matches = [
        {
            "prediction": str(predicted_ids[index].item()),
            "reference": str(reference_ids[paper_best_reference[index]].item()),
            "iou": float(paper_best_iou[index]),
        }
        for index in range(len(predicted_ids))
        if paper_best_iou[index] >= iou_threshold
    ]
    paper_metrics = metric_values(
        len(paper_matches),
        len(predicted_ids),
        len(reference_ids),
    )
    paper_metrics.update(
        {
            "matching_policy": "paper_per_prediction_best_iou",
            "mean_matched_iou": (
                float(np.mean([row["iou"] for row in paper_matches]))
                if paper_matches
                else 0.0
            ),
            "matches": paper_matches,
        }
    )

    one_to_one_indices = maximum_threshold_matching(matrix, iou_threshold)
    one_to_one_matches = [
        {
            "prediction": str(predicted_ids[prediction].item()),
            "reference": str(reference_ids[reference].item()),
            "iou": float(matrix[prediction, reference]),
        }
        for prediction, reference in one_to_one_indices
    ]
    one_to_one_metrics = metric_values(
        len(one_to_one_matches),
        len(predicted_ids),
        len(reference_ids),
    )
    one_to_one_metrics.update(
        {
            "matching_policy": "one_to_one",
            "mean_matched_iou": (
                float(np.mean([row["iou"] for row in one_to_one_matches]))
                if one_to_one_matches
                else 0.0
            ),
            "matches": one_to_one_matches,
        }
    )

    reference_best_iou = (
        np.max(matrix, axis=0)
        if matrix.shape[0]
        else np.zeros(matrix.shape[1], dtype=float)
    )
    reference_sizes = np.asarray(
        [
            np.count_nonzero(reference_instance == reference_id)
            for reference_id in reference_ids
        ],
        dtype=np.float64,
    )
    mean_coverage = (
        float(np.mean(reference_best_iou)) if len(reference_best_iou) else 0.0
    )
    weighted_coverage = (
        float(np.average(reference_best_iou, weights=reference_sizes))
        if len(reference_best_iou) and np.sum(reference_sizes)
        else 0.0
    )

    return {
        "point_count": len(labels.predicted_instance),
        "evaluated_point_count": int(np.count_nonzero(evaluation_mask)),
        "prediction_instance_count": len(predicted_ids),
        "reference_instance_count": len(reference_ids),
        "iou_threshold": iou_threshold,
        "min_predicted_instance_points": min_predicted_instance_points,
        "min_predicted_tree_fraction": min_predicted_tree_fraction,
        "mean_unweighted_coverage": mean_coverage,
        "mean_weighted_coverage": weighted_coverage,
        "paper_compatible": paper_metrics,
        "harmonized": one_to_one_metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_number_set(text: str) -> set[float]:
    return {
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate point-aligned instance predictions using paper-compatible "
            "and harmonized one-to-one metrics."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--combined-labelled-point-cloud")
    source.add_argument("--instance-evaluation-ply")
    parser.add_argument(
        "--semantic-evaluation-ply",
        help="Required with --instance-evaluation-ply.",
    )
    parser.add_argument("--predicted-instance-field", default="preds")
    parser.add_argument("--reference-instance-field", default="gt")
    parser.add_argument("--predicted-semantic-field", default="preds")
    parser.add_argument("--reference-semantic-field", default="gt")
    parser.add_argument(
        "--semantic-offset",
        type=float,
        default=1.0,
        help="Offset used by the internal SegmentAnyTree evaluation files.",
    )
    parser.add_argument("--reference-tree-classes", default="2")
    parser.add_argument("--prediction-tree-classes", default="2")
    parser.add_argument(
        "--reference-background-instance-labels",
        help=(
            "Derive reference tree semantics by excluding these instance "
            "labels. Use this for aligned outputs whose reference semantic "
            "field is degenerate."
        ),
    )
    parser.add_argument("--ignored-reference-labels", default="-1")
    parser.add_argument("--ignored-prediction-labels", default="-1")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-predicted-instance-points", type=int, default=0)
    parser.add_argument("--min-predicted-tree-fraction", type=float, default=0.0)
    parser.add_argument("--plot-name")
    parser.add_argument("--collection")
    parser.add_argument("--split")
    parser.add_argument("--relative-path")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--paper-matches-csv")
    parser.add_argument("--harmonized-matches-csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.instance_evaluation_ply and not args.semantic_evaluation_ply:
        raise SystemExit(
            "--semantic-evaluation-ply is required with "
            "--instance-evaluation-ply"
        )
    if args.combined_labelled_point_cloud:
        input_mode = "combined_labelled_point_cloud"
        input_path = resolve_path(args.combined_labelled_point_cloud)
        labels = load_combined_labelled_cloud(
            input_path,
            args.predicted_instance_field,
            args.reference_instance_field,
            args.predicted_semantic_field,
            args.reference_semantic_field,
        )
        inputs = {"combined_labelled_point_cloud": str(input_path)}
    else:
        input_mode = "internal_aligned_ply"
        instance_path = resolve_path(args.instance_evaluation_ply)
        semantic_path = resolve_path(args.semantic_evaluation_ply)
        labels = load_internal_evaluation(
            instance_path,
            semantic_path,
            args.predicted_instance_field,
            args.reference_instance_field,
            args.predicted_semantic_field,
            args.reference_semantic_field,
            args.semantic_offset,
        )
        inputs = {
            "instance_evaluation_ply": str(instance_path),
            "semantic_evaluation_ply": str(semantic_path),
        }

    reference_tree_classes = parse_number_set(args.reference_tree_classes)
    ignored_reference_labels = parse_number_set(args.ignored_reference_labels)
    reference_semantic_source = "semantic_field"
    if args.reference_background_instance_labels:
        background_labels = parse_number_set(
            args.reference_background_instance_labels
        )
        if len(reference_tree_classes) != 1:
            raise SystemExit(
                "--reference-background-instance-labels requires one "
                "--reference-tree-classes value"
            )
        if reference_semantic_requires_instance_fallback(
            labels,
            background_labels,
            ignored_reference_labels,
            reference_tree_classes,
        ):
            labels = derive_reference_semantic_from_instance(
                labels,
                background_labels,
                reference_tree_classes,
                ignored_reference_labels,
            )
            reference_semantic_source = "instance_background_labels"

    result = evaluate_pointwise(
        labels,
        reference_tree_classes=reference_tree_classes,
        prediction_tree_classes=parse_number_set(args.prediction_tree_classes),
        ignored_reference_labels=ignored_reference_labels,
        ignored_prediction_labels=parse_number_set(args.ignored_prediction_labels),
        iou_threshold=args.iou_threshold,
        min_predicted_instance_points=args.min_predicted_instance_points,
        min_predicted_tree_fraction=args.min_predicted_tree_fraction,
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "pointwise_instance_metrics",
        "input_mode": input_mode,
        "inputs": inputs,
        "plot_name": args.plot_name,
        "collection": args.collection,
        "split": args.split,
        "relative_path": args.relative_path,
        "reference_semantic_source": reference_semantic_source,
        "reference_background_instance_labels": sorted(
            parse_number_set(args.reference_background_instance_labels or "")
        ),
        "reference_tree_classes": sorted(
            reference_tree_classes
        ),
        "prediction_tree_classes": sorted(
            parse_number_set(args.prediction_tree_classes)
        ),
        "ignored_reference_labels": sorted(
            parse_number_set(args.ignored_reference_labels)
        ),
        "ignored_prediction_labels": sorted(
            parse_number_set(args.ignored_prediction_labels)
        ),
        "min_predicted_instance_points": args.min_predicted_instance_points,
        "min_predicted_tree_fraction": args.min_predicted_tree_fraction,
        **result,
    }
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.paper_matches_csv:
        write_csv(
            resolve_path(args.paper_matches_csv),
            result["paper_compatible"]["matches"],
        )
    if args.harmonized_matches_csv:
        write_csv(
            resolve_path(args.harmonized_matches_csv),
            result["harmonized"]["matches"],
        )
    print(
        "Paper-compatible F1: "
        f"{result['paper_compatible']['f1']:.6f}"
    )
    print(f"Harmonized F1: {result['harmonized']['f1']:.6f}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
