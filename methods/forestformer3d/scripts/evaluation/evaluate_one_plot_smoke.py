"""Evaluate the aligned ForestFormer3D development smoke prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EVALUATOR_ROOT = (
    ROOT / "methods" / "segmentanytree" / "scripts" / "evaluation"
)
if str(EVALUATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_ROOT))

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
EXPECTED_RELATIVE_PATH = "CULS/plot_1_annotated.las"
EXPECTED_POINT_COUNT = 1_816_672

MATCH_FIELDS = (
    "plot_id",
    "pred_tree_id",
    "target_tree_id",
    "intersection_points",
    "predicted_points",
    "reference_points",
    "union_points",
    "iou",
)
UNMATCHED_PREDICTION_FIELDS = (
    "plot_id",
    "pred_tree_id",
    "predicted_points",
    "best_target_tree_id",
    "best_iou",
)
UNMATCHED_REFERENCE_FIELDS = (
    "plot_id",
    "target_tree_id",
    "reference_points",
    "best_pred_tree_id",
    "best_iou",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(
    path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_arrays(
    labels: PointLabels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reference_tree = np.isin(
        labels.reference_semantic.astype(np.float64),
        sorted(REFERENCE_TREE_CLASSES),
    )
    predicted_tree = np.isin(
        labels.predicted_semantic.astype(np.float64),
        sorted(PREDICTION_TREE_CLASSES),
    )
    mask = reference_tree | predicted_tree
    predicted_instance = labels.predicted_instance[mask]
    reference_instance = labels.reference_instance[mask]
    predicted_semantic = labels.predicted_semantic[mask]
    reference_semantic = labels.reference_semantic[mask]
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
        predicted_instance, reference_instance, predicted_ids, reference_ids
    )
    return (
        predicted_ids,
        reference_ids,
        matrix,
        predicted_instance,
        reference_instance,
    )


def evaluate(
    prediction_npz: Path,
    validation_json: Path,
    reference_resource_json: Path,
    dummy_resource_json: Path,
    output_root: Path,
    *,
    run_id: str,
    inference_benchmark_commit: str,
    evaluation_benchmark_commit: str,
    expected_point_count: int = EXPECTED_POINT_COUNT,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Refusing existing evaluation root: {output_root}")
    validation = json.loads(validation_json.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "passed"
        or validation.get("split") != "development"
        or validation.get("relative_path") != EXPECTED_RELATIVE_PATH
        or validation.get("held_out_access") is not False
        or validation.get("exact_row_alignment") is not True
    ):
        raise ValueError("Smoke validation record does not pass the development gate")
    expected_npz_sha = (
        validation.get("artifacts", {}).get("harmonised_npz", {}).get("sha256")
    )
    observed_npz_sha = sha256_file(prediction_npz)
    if expected_npz_sha != observed_npz_sha:
        raise ValueError("Harmonised prediction SHA-256 differs from validation")

    with np.load(prediction_npz) as values:
        required = {
            "pred_tree_id",
            "target_tree_id",
            "classification",
            "pred_classification",
            "source_row_index",
        }
        if set(values.files) != required:
            raise ValueError("Unexpected harmonised prediction fields")
        arrays = {name: np.asarray(values[name]) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if lengths != {expected_point_count}:
        raise ValueError(f"Unexpected aligned array lengths: {sorted(lengths)}")
    if not np.array_equal(
        arrays["source_row_index"], np.arange(expected_point_count)
    ):
        raise ValueError("source_row_index is not the exact identity map")
    expected_pred_class = np.where(arrays["pred_tree_id"] > 0, 4, 0)
    if not np.array_equal(arrays["pred_classification"], expected_pred_class):
        raise ValueError("Predicted semantic mapping differs from the contract")

    labels = PointLabels(
        predicted_instance=arrays["pred_tree_id"],
        reference_instance=arrays["target_tree_id"],
        predicted_semantic=arrays["pred_classification"],
        reference_semantic=arrays["classification"],
    )
    result = evaluate_pointwise(
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
        _evaluation_arrays(labels)
    )
    harmonized = result["harmonized"]
    match_rows: list[dict[str, object]] = []
    for match in harmonized["matches"]:
        pred_id = int(match["prediction"])
        ref_id = int(match["reference"])
        predicted_points = int(np.count_nonzero(predicted_eval == pred_id))
        reference_points = int(np.count_nonzero(reference_eval == ref_id))
        intersection = int(
            np.count_nonzero(
                (predicted_eval == pred_id) & (reference_eval == ref_id)
            )
        )
        match_rows.append(
            {
                "plot_id": EXPECTED_RELATIVE_PATH,
                "pred_tree_id": pred_id,
                "target_tree_id": ref_id,
                "intersection_points": intersection,
                "predicted_points": predicted_points,
                "reference_points": reference_points,
                "union_points": predicted_points + reference_points - intersection,
                "iou": float(match["iou"]),
            }
        )
    matched_pred = {int(row["pred_tree_id"]) for row in match_rows}
    matched_ref = {int(row["target_tree_id"]) for row in match_rows}
    unmatched_predictions: list[dict[str, object]] = []
    for row_index, value in enumerate(predicted_ids):
        pred_id = int(value)
        if pred_id in matched_pred:
            continue
        best_column = int(np.argmax(matrix[row_index])) if matrix.shape[1] else 0
        best_iou = float(matrix[row_index, best_column]) if matrix.shape[1] else 0.0
        unmatched_predictions.append(
            {
                "plot_id": EXPECTED_RELATIVE_PATH,
                "pred_tree_id": pred_id,
                "predicted_points": int(np.count_nonzero(predicted_eval == pred_id)),
                "best_target_tree_id": (
                    int(reference_ids[best_column]) if best_iou > 0 else None
                ),
                "best_iou": best_iou,
            }
        )
    unmatched_references: list[dict[str, object]] = []
    for column_index, value in enumerate(reference_ids):
        ref_id = int(value)
        if ref_id in matched_ref:
            continue
        best_row = int(np.argmax(matrix[:, column_index])) if matrix.shape[0] else 0
        best_iou = float(matrix[best_row, column_index]) if matrix.shape[0] else 0.0
        unmatched_references.append(
            {
                "plot_id": EXPECTED_RELATIVE_PATH,
                "target_tree_id": ref_id,
                "reference_points": int(np.count_nonzero(reference_eval == ref_id)),
                "best_pred_tree_id": (
                    int(predicted_ids[best_row]) if best_iou > 0 else None
                ),
                "best_iou": best_iou,
            }
        )

    resources = {
        "reference": json.loads(reference_resource_json.read_text(encoding="utf-8")),
        "dummy": json.loads(dummy_resource_json.read_text(encoding="utf-8")),
    }
    output_root.mkdir(parents=True)
    metrics: dict[str, object] = {
        "schema": "forestformer3d_one_plot_smoke_metrics_v1",
        "status": "completed_development_smoke_evaluation",
        "method": "ForestFormer3D",
        "run_id": run_id,
        "training_mode": "published_pretrained",
        "split": "development",
        "relative_path": EXPECTED_RELATIVE_PATH,
        "held_out_access": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": IOU_THRESHOLD,
        "point_count": int(result["point_count"]),
        "evaluated_point_count": int(result["evaluated_point_count"]),
        "prediction_instance_count": int(result["prediction_instance_count"]),
        "reference_instance_count": int(result["reference_instance_count"]),
        "true_positives": int(harmonized["true_positives"]),
        "false_positives": int(harmonized["false_positives"]),
        "false_negatives": int(harmonized["false_negatives"]),
        "precision": float(harmonized["precision"]),
        "recall": float(harmonized["recall"]),
        "f1": float(harmonized["f1"]),
        "mean_matched_iou": float(harmonized["mean_matched_iou"]),
        "mean_unweighted_coverage": float(result["mean_unweighted_coverage"]),
        "mean_weighted_coverage": float(result["mean_weighted_coverage"]),
        "inference_benchmark_commit": inference_benchmark_commit,
        "evaluation_benchmark_commit": evaluation_benchmark_commit,
        "prediction_npz_sha256": observed_npz_sha,
        "resource_usage": resources,
        "next_gate": "manual_alignment_confirmation_before_full_development",
    }
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_root / "matches.csv", match_rows, MATCH_FIELDS)
    _write_csv(
        output_root / "unmatched_predictions.csv",
        unmatched_predictions,
        UNMATCHED_PREDICTION_FIELDS,
    )
    _write_csv(
        output_root / "unmatched_references.csv",
        unmatched_references,
        UNMATCHED_REFERENCE_FIELDS,
    )
    report = {
        "schema": "forestformer3d_manual_alignment_report_v1",
        "status": "awaiting_human_confirmation",
        "public_safe": True,
        "contains_coordinates": False,
        "run_id": run_id,
        "split": "development",
        "relative_path": EXPECTED_RELATIVE_PATH,
        "point_count": expected_point_count,
        "prediction_npz_sha256": observed_npz_sha,
        "raw_reference_ply_sha256": validation["artifacts"]["reference_ply"]["sha256"],
        "exact_source_row_identity": True,
        "predicted_tree_points": int(
            np.count_nonzero(arrays["pred_tree_id"] > 0)
        ),
        "predicted_instances": int(len(predicted_ids)),
        "reference_tree_points": int(
            np.count_nonzero(arrays["target_tree_id"] > 0)
        ),
        "reference_instances": int(len(reference_ids)),
        "matched_instances": len(match_rows),
        "review_artifact": "raw/reference/forestformer3d_smoke_test.ply",
        "review_fields": [
            "semantic_pred",
            "instance_pred",
            "semantic_gt",
            "instance_gt",
        ],
        "review_instruction": (
            "Open the retained development PLY in a point-cloud viewer, colour "
            "by instance_pred and compare against instance_gt. Confirm that "
            "predicted crowns occupy the same forest geometry and that no "
            "coordinate or row displacement is visible."
        ),
    }
    (output_root / "manual_alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    retained = (
        "metrics.json",
        "matches.csv",
        "unmatched_predictions.csv",
        "unmatched_references.csv",
        "manual_alignment_report.json",
    )
    with (output_root / "artifact_sha256.txt").open("w", encoding="utf-8") as handle:
        for name in retained:
            path = output_root / name
            handle.write(f"{sha256_file(path)}  {name}\n")
    (output_root / "evaluation.complete").touch(exist_ok=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True, type=Path)
    parser.add_argument("--validation-json", required=True, type=Path)
    parser.add_argument("--reference-resource-json", required=True, type=Path)
    parser.add_argument("--dummy-resource-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inference-benchmark-commit", required=True)
    parser.add_argument("--evaluation-benchmark-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(
        args.prediction_npz,
        args.validation_json,
        args.reference_resource_json,
        args.dummy_resource_json,
        args.output_root,
        run_id=args.run_id,
        inference_benchmark_commit=args.inference_benchmark_commit,
        evaluation_benchmark_commit=args.evaluation_benchmark_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
