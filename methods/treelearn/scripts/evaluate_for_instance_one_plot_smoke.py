"""Evaluate one point-aligned TreeLearn FOR-instance development smoke output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SHARED_EVALUATOR = (
    ROOT / "methods" / "segmentanytree" / "scripts" / "evaluation"
)
if str(SHARED_EVALUATOR) not in sys.path:
    sys.path.insert(0, str(SHARED_EVALUATOR))

from pointwise_instance_metrics import (  # noqa: E402
    PointLabels,
    allowed_instance_ids,
    contingency_iou,
    evaluate_pointwise,
)


REFERENCE_TREE_CLASSES = {4.0, 5.0, 6.0}
PREDICTION_TREE_CLASSES = {4.0}
IGNORED_INSTANCE_LABELS = {-1.0, 0.0}
IOU_THRESHOLD = 0.5
EXPECTED_UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
EXPECTED_CHECKPOINT_MD5 = "56a3d78f689ae7f1190906b975700311"

MATCH_FIELDS = [
    "plot_id",
    "pred_tree_id",
    "target_tree_id",
    "intersection_points",
    "predicted_points",
    "reference_points",
    "union_points",
    "iou",
]
UNMATCHED_PREDICTION_FIELDS = [
    "plot_id",
    "pred_tree_id",
    "predicted_points",
    "best_target_tree_id",
    "best_iou",
]
UNMATCHED_REFERENCE_FIELDS = [
    "plot_id",
    "target_tree_id",
    "reference_points",
    "best_pred_tree_id",
    "best_iou",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_arrays(
    pred_tree_id: np.ndarray,
    target_tree_id: np.ndarray,
    classification: np.ndarray,
    pred_classification: np.ndarray,
    source_row_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate the row-preserving adapter contract and normalise integer IDs."""

    arrays = [
        np.asarray(value)
        for value in (
            pred_tree_id,
            target_tree_id,
            classification,
            pred_classification,
            source_row_index,
        )
    ]
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("Adapted prediction arrays must be one-dimensional")
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Adapted prediction arrays are not aligned: {sorted(lengths)}")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("Adapted prediction arrays must contain at least one point")

    raw_row_index = arrays[4]
    expected_row_index = np.arange(len(arrays[0]), dtype=np.int64)
    if not np.issubdtype(raw_row_index.dtype, np.integer) or not np.array_equal(
        raw_row_index,
        expected_row_index,
    ):
        raise ValueError("source_row_index must equal np.arange(point_count)")
    pred, target, classes, pred_classes, row_index = (
        value.astype(np.int64, copy=False) for value in arrays
    )
    expected_pred_classes = np.where(pred > 0, 4, 0)
    if not np.array_equal(pred_classes, expected_pred_classes):
        raise ValueError(
            "pred_classification must map positive pred_tree_id values to 4 "
            "and background or unassigned values to 0"
        )
    return pred, target, classes, pred_classes, row_index


def _evaluation_ids(
    labels: PointLabels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact IDs and IoU matrix used by the shared evaluator."""

    reference_tree = np.isin(
        labels.reference_semantic.astype(np.float64),
        sorted(REFERENCE_TREE_CLASSES),
    )
    predicted_tree = np.isin(
        labels.predicted_semantic.astype(np.float64),
        sorted(PREDICTION_TREE_CLASSES),
    )
    evaluation_mask = reference_tree | predicted_tree
    predicted_instance = labels.predicted_instance[evaluation_mask]
    reference_instance = labels.reference_instance[evaluation_mask]
    predicted_semantic = labels.predicted_semantic[evaluation_mask]
    reference_semantic = labels.reference_semantic[evaluation_mask]

    predicted_ids = allowed_instance_ids(
        predicted_instance,
        predicted_semantic,
        IGNORED_INSTANCE_LABELS,
        PREDICTION_TREE_CLASSES,
    )
    reference_ids = allowed_instance_ids(
        reference_instance,
        reference_semantic,
        IGNORED_INSTANCE_LABELS,
        REFERENCE_TREE_CLASSES,
    )
    matrix = contingency_iou(
        predicted_instance,
        reference_instance,
        predicted_ids,
        reference_ids,
    )
    return (
        predicted_ids,
        reference_ids,
        matrix,
        predicted_instance,
        reference_instance,
    )


def _match_rows(
    plot_id: str,
    matches: list[dict[str, Any]],
    predicted_instance: np.ndarray,
    reference_instance: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        pred_tree_id = int(match["prediction"])
        target_tree_id = int(match["reference"])
        predicted_points = int(np.count_nonzero(predicted_instance == pred_tree_id))
        reference_points = int(np.count_nonzero(reference_instance == target_tree_id))
        intersection = int(
            np.count_nonzero(
                (predicted_instance == pred_tree_id)
                & (reference_instance == target_tree_id)
            )
        )
        rows.append(
            {
                "plot_id": plot_id,
                "pred_tree_id": pred_tree_id,
                "target_tree_id": target_tree_id,
                "intersection_points": intersection,
                "predicted_points": predicted_points,
                "reference_points": reference_points,
                "union_points": predicted_points + reference_points - intersection,
                "iou": float(match["iou"]),
            }
        )
    return rows


def _unmatched_prediction_rows(
    plot_id: str,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
    matrix: np.ndarray,
    predicted_instance: np.ndarray,
    matched_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, value in enumerate(predicted_ids):
        pred_tree_id = int(value)
        if pred_tree_id in matched_ids:
            continue
        if matrix.shape[1]:
            best_column = int(np.argmax(matrix[row_index]))
            best_iou = float(matrix[row_index, best_column])
            best_target: int | None = (
                int(reference_ids[best_column]) if best_iou > 0 else None
            )
        else:
            best_target = None
            best_iou = 0.0
        rows.append(
            {
                "plot_id": plot_id,
                "pred_tree_id": pred_tree_id,
                "predicted_points": int(
                    np.count_nonzero(predicted_instance == pred_tree_id)
                ),
                "best_target_tree_id": best_target,
                "best_iou": best_iou,
            }
        )
    return rows


def _unmatched_reference_rows(
    plot_id: str,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
    matrix: np.ndarray,
    reference_instance: np.ndarray,
    matched_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column_index, value in enumerate(reference_ids):
        target_tree_id = int(value)
        if target_tree_id in matched_ids:
            continue
        if matrix.shape[0]:
            best_row = int(np.argmax(matrix[:, column_index]))
            best_iou = float(matrix[best_row, column_index])
            best_prediction: int | None = (
                int(predicted_ids[best_row]) if best_iou > 0 else None
            )
        else:
            best_prediction = None
            best_iou = 0.0
        rows.append(
            {
                "plot_id": plot_id,
                "target_tree_id": target_tree_id,
                "reference_points": int(
                    np.count_nonzero(reference_instance == target_tree_id)
                ),
                "best_pred_tree_id": best_prediction,
                "best_iou": best_iou,
            }
        )
    return rows


def evaluate_arrays(
    pred_tree_id: np.ndarray,
    target_tree_id: np.ndarray,
    classification: np.ndarray,
    pred_classification: np.ndarray,
    source_row_index: np.ndarray,
    plot_id: str,
    split: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate a TreeLearn adapter output with the frozen shared protocol."""

    if split != "dev":
        raise ValueError("The one-plot TreeLearn smoke evaluator requires --split dev")
    pred, target, classes, pred_classes, _ = validate_arrays(
        pred_tree_id,
        target_tree_id,
        classification,
        pred_classification,
        source_row_index,
    )

    # TreeLearn positive instance IDs are tree predictions. Reference semantics
    # follow the FOR-instance definition: LAS classes 4, 5 and 6 are tree points.
    labels = PointLabels(
        predicted_instance=pred,
        reference_instance=target,
        predicted_semantic=pred_classes,
        reference_semantic=classes,
    )
    shared_result = evaluate_pointwise(
        labels,
        reference_tree_classes=REFERENCE_TREE_CLASSES,
        prediction_tree_classes=PREDICTION_TREE_CLASSES,
        ignored_reference_labels=IGNORED_INSTANCE_LABELS,
        ignored_prediction_labels=IGNORED_INSTANCE_LABELS,
        iou_threshold=IOU_THRESHOLD,
        min_predicted_instance_points=0,
        min_predicted_tree_fraction=0.0,
    )
    predicted_ids, reference_ids, matrix, predicted_eval, reference_eval = (
        _evaluation_ids(labels)
    )
    harmonized = shared_result["harmonized"]
    matches = _match_rows(
        plot_id,
        harmonized["matches"],
        predicted_eval,
        reference_eval,
    )
    matched_predictions = {int(row["pred_tree_id"]) for row in matches}
    matched_references = {int(row["target_tree_id"]) for row in matches}
    unmatched_predictions = _unmatched_prediction_rows(
        plot_id,
        predicted_ids,
        reference_ids,
        matrix,
        predicted_eval,
        matched_predictions,
    )
    unmatched_references = _unmatched_reference_rows(
        plot_id,
        predicted_ids,
        reference_ids,
        matrix,
        reference_eval,
        matched_references,
    )

    summary: dict[str, Any] = {
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "plot_id": plot_id,
        "split": split,
        "dataset_split": split,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": IOU_THRESHOLD,
        "iou_threshold_operator": ">=",
        "point_correspondence": "source_row_index",
        "prediction_semantic_mapping": "pred_tree_id > 0 -> class 4; else 0",
        "reference_tree_classes": sorted(int(value) for value in REFERENCE_TREE_CLASSES),
        "ignored_instance_labels": sorted(
            int(value) for value in IGNORED_INSTANCE_LABELS
        ),
        "tuned_prediction_filtering": False,
        "min_predicted_instance_points": 0,
        "min_predicted_tree_fraction": 0.0,
        "point_count": int(shared_result["point_count"]),
        "evaluated_point_count": int(shared_result["evaluated_point_count"]),
        "prediction_instance_count": int(
            shared_result["prediction_instance_count"]
        ),
        "reference_instance_count": int(shared_result["reference_instance_count"]),
        "true_positives": int(harmonized["true_positives"]),
        "false_positives": int(harmonized["false_positives"]),
        "false_negatives": int(harmonized["false_negatives"]),
        "precision": float(harmonized["precision"]),
        "recall": float(harmonized["recall"]),
        "f1": float(harmonized["f1"]),
        "mean_matched_iou": float(harmonized["mean_matched_iou"]),
        "mean_unweighted_coverage": float(
            shared_result["mean_unweighted_coverage"]
        ),
        "mean_weighted_coverage": float(shared_result["mean_weighted_coverage"]),
        "harmonized": harmonized,
    }
    return summary, matches, unmatched_predictions, unmatched_references


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a stable CSV schema, including when there are no data rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one row-aligned TreeLearn FOR-instance development smoke "
            "prediction with the fixed harmonized protocol."
        )
    )
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--inference-metadata", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plot-id", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--split", "--expected-split", dest="split", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument(
        "--harmonized-matches-csv",
        "--harmonised-matches-csv",
        dest="harmonized_matches_csv",
        required=True,
    )
    parser.add_argument("--unmatched-predictions-csv", required=True)
    parser.add_argument("--unmatched-references-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.split != "dev":
        raise ValueError("The one-plot TreeLearn smoke evaluator requires --split dev")

    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Adapted prediction NPZ does not exist: {prediction_path}")
    with np.load(prediction_path) as data:
        required = {
            "pred_tree_id",
            "target_tree_id",
            "classification",
            "pred_classification",
            "source_row_index",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Adapted prediction NPZ is missing arrays {sorted(missing)}")
        summary, matches, unmatched_predictions, unmatched_references = (
            evaluate_arrays(
                data["pred_tree_id"],
                data["target_tree_id"],
                data["classification"],
                data["pred_classification"],
                data["source_row_index"],
                args.plot_id,
                args.split,
            )
        )

    metadata_path = Path(args.inference_metadata).expanduser().resolve()
    inference_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if inference_metadata.get("status") != "completed":
        raise ValueError("TreeLearn inference metadata is not completed")
    if inference_metadata.get("run_id") != args.run_id:
        raise ValueError("TreeLearn inference metadata run ID does not match")
    if inference_metadata.get("plot", {}).get("relative_path") != args.relative_path:
        raise ValueError("TreeLearn inference metadata plot path does not match")
    recorded_prediction = Path(
        inference_metadata.get("outputs", {}).get("adapted_npz", "")
    ).resolve()
    if recorded_prediction != prediction_path:
        raise ValueError("TreeLearn prediction path does not match inference metadata")
    if inference_metadata.get("plot", {}).get("split") != "dev":
        raise ValueError("TreeLearn inference metadata is not development-only")
    repository = inference_metadata.get("environment", {}).get(
        "treelearn_repository", {}
    )
    if repository.get("commit") != EXPECTED_UPSTREAM_COMMIT or repository.get(
        "dirty"
    ) is not False:
        raise ValueError("TreeLearn inference metadata does not freeze a clean upstream")
    benchmark_repository = inference_metadata.get("environment", {}).get(
        "benchmark_repository", {}
    )
    if (
        len(str(benchmark_repository.get("commit", ""))) != 40
        or benchmark_repository.get("dirty") is not False
    ):
        raise ValueError("TreeLearn inference metadata does not freeze benchmark code")
    checkpoint = inference_metadata.get("checkpoint", {})
    if (
        checkpoint.get("md5") != EXPECTED_CHECKPOINT_MD5
        or checkpoint.get("source_md5") != EXPECTED_CHECKPOINT_MD5
    ):
        raise ValueError("TreeLearn inference metadata checkpoint identity does not match")
    retained_entries = inference_metadata.get("retention", {}).get("files", [])
    matching_entries = [
        entry
        for entry in retained_entries
        if entry.get("path")
        and Path(entry["path"]).expanduser().resolve() == prediction_path
    ]
    if len(matching_entries) != 1 or matching_entries[0].get("exists") is not True:
        raise ValueError("TreeLearn prediction is missing from the retention inventory")
    expected_prediction_sha256 = matching_entries[0].get("sha256")
    actual_prediction_sha256 = sha256(prediction_path)
    if actual_prediction_sha256 != expected_prediction_sha256:
        raise ValueError("TreeLearn prediction SHA-256 does not match inference metadata")

    summary.update(
        {
            "status": "completed_development_smoke_evaluation",
            "run_id": args.run_id,
            "relative_path": args.relative_path,
            "prediction_npz": str(prediction_path),
            "prediction_npz_sha256": actual_prediction_sha256,
            "prediction_npz_size_bytes": prediction_path.stat().st_size,
            "inference_metadata": str(metadata_path),
            "next_gate": "manual_alignment_review_before_full_development_evaluation",
        }
    )
    metrics_path = Path(args.metrics_json).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        Path(args.harmonized_matches_csv).expanduser().resolve(),
        matches,
        MATCH_FIELDS,
    )
    write_csv(
        Path(args.unmatched_predictions_csv).expanduser().resolve(),
        unmatched_predictions,
        UNMATCHED_PREDICTION_FIELDS,
    )
    write_csv(
        Path(args.unmatched_references_csv).expanduser().resolve(),
        unmatched_references,
        UNMATCHED_REFERENCE_FIELDS,
    )
    # Publish the completion marker only after every companion table exists.
    metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
