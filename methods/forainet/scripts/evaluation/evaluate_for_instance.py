"""Evaluate a source-row-aligned ForAINet prediction under the shared protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
RUNTIME = ROOT / "methods" / "forainet" / "scripts" / "runtime"
for directory in (SRC, RUNTIME):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmark.instance_metrics import (  # noqa: E402
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)
from forainet_contract import as_int64  # noqa: E402


TREE_CLASSES = (4, 5, 6)
IGNORED_INSTANCE_IDS = (-1, 0)
IOU_THRESHOLD = 0.5


def validate_aligned_arrays(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "pred_tree_id",
        "target_tree_id",
        "classification",
        "pred_classification",
        "source_row_index",
    }
    if missing := required - set(payload):
        raise ValueError(f"aligned prediction is missing fields: {sorted(missing)}")
    arrays = {name: as_int64(name, payload[name]) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("aligned prediction fields are empty or mismatched")
    expected = np.arange(next(iter(lengths)), dtype=np.int64)
    if not np.array_equal(arrays["source_row_index"], expected):
        raise ValueError("source_row_index must equal the original source order")
    if unknown := sorted(
        set(np.unique(arrays["pred_classification"])) - {0, 4, 5, 6}
    ):
        raise ValueError(f"unknown predicted benchmark classes: {unknown}")
    return arrays


def evaluate(payload: dict[str, np.ndarray]) -> tuple[dict[str, Any], ...]:
    arrays = validate_aligned_arrays(payload)
    pred_ids_all = arrays["pred_tree_id"]
    ref_ids_all = arrays["target_tree_id"]
    pred_semantic_all = arrays["pred_classification"]
    ref_semantic_all = arrays["classification"]

    mask = np.isin(pred_semantic_all, TREE_CLASSES) | np.isin(
        ref_semantic_all, TREE_CLASSES
    )
    pred = pred_ids_all[mask]
    ref = ref_ids_all[mask]
    pred_semantic = pred_semantic_all[mask]
    ref_semantic = ref_semantic_all[mask]
    pred_ids = np.asarray(
        sorted(
            int(value)
            for value in np.unique(pred)
            if value not in IGNORED_INSTANCE_IDS
            and np.any((pred == value) & np.isin(pred_semantic, TREE_CLASSES))
        ),
        dtype=np.int64,
    )
    ref_ids = np.asarray(
        sorted(
            int(value)
            for value in np.unique(ref)
            if value not in IGNORED_INSTANCE_IDS
            and np.any((ref == value) & np.isin(ref_semantic, TREE_CLASSES))
        ),
        dtype=np.int64,
    )
    matrix = np.zeros((len(pred_ids), len(ref_ids)), dtype=np.float64)
    intersections = np.zeros_like(matrix, dtype=np.int64)
    unions = np.zeros_like(matrix, dtype=np.int64)
    for row, pred_id in enumerate(pred_ids):
        pred_mask = (pred == pred_id) & np.isin(pred_semantic, TREE_CLASSES)
        for column, ref_id in enumerate(ref_ids):
            ref_mask = (ref == ref_id) & np.isin(ref_semantic, TREE_CLASSES)
            intersection = int(np.count_nonzero(pred_mask & ref_mask))
            union = int(np.count_nonzero(pred_mask | ref_mask))
            intersections[row, column] = intersection
            unions[row, column] = union
            matrix[row, column] = intersection / union if union else 0.0

    matched_indices = maximum_cardinality_threshold_matching(matrix, IOU_THRESHOLD)
    matched_pred = {row for row, _ in matched_indices}
    matched_ref = {column for _, column in matched_indices}
    matches = [
        {
            "pred_tree_id": int(pred_ids[row]),
            "target_tree_id": int(ref_ids[column]),
            "intersection_points": int(intersections[row, column]),
            "union_points": int(unions[row, column]),
            "iou": float(matrix[row, column]),
        }
        for row, column in matched_indices
    ]
    unmatched_predictions = [
        {
            "pred_tree_id": int(pred_ids[row]),
            "predicted_points": int(np.count_nonzero(pred == pred_ids[row])),
            "best_iou": float(np.max(matrix[row])) if len(ref_ids) else 0.0,
        }
        for row in range(len(pred_ids))
        if row not in matched_pred
    ]
    unmatched_references = [
        {
            "target_tree_id": int(ref_ids[column]),
            "reference_points": int(np.count_nonzero(ref == ref_ids[column])),
            "best_iou": float(np.max(matrix[:, column])) if len(pred_ids) else 0.0,
        }
        for column in range(len(ref_ids))
        if column not in matched_ref
    ]
    tp = len(matches)
    fp = len(unmatched_predictions)
    fn = len(unmatched_references)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    summary = {
        "protocol_id": "for_instance_pointwise_v1",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "point_correspondence": "exact_source_row_index",
        "coordinate_matching": False,
        "reference_tree_classes": list(TREE_CLASSES),
        "ignored_reference_classes": [0, 1, 2, 3],
        "ignored_instance_ids": list(IGNORED_INSTANCE_IDS),
        "iou_threshold": IOU_THRESHOLD,
        "iou_threshold_operator": ">=",
        "matching": "maximum_cardinality_one_to_one",
        "evaluated_point_count": int(np.count_nonzero(mask)),
        "prediction_instance_count": len(pred_ids),
        "reference_instance_count": len(ref_ids),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return summary, matches, unmatched_predictions, unmatched_references


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--matches-csv", required=True, type=Path)
    parser.add_argument("--unmatched-predictions-csv", required=True, type=Path)
    parser.add_argument("--unmatched-references-csv", required=True, type=Path)
    parser.add_argument("--split", choices=("dev",), required=True)
    args = parser.parse_args()
    outputs = (
        args.metrics_json,
        args.matches_csv,
        args.unmatched_predictions_csv,
        args.unmatched_references_csv,
    )
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    with np.load(args.prediction_npz, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    summary, matches, unmatched_predictions, unmatched_references = evaluate(payload)
    summary["split"] = args.split
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        args.matches_csv,
        matches,
        ["pred_tree_id", "target_tree_id", "intersection_points", "union_points", "iou"],
    )
    write_csv(
        args.unmatched_predictions_csv,
        unmatched_predictions,
        ["pred_tree_id", "predicted_points", "best_iou"],
    )
    write_csv(
        args.unmatched_references_csv,
        unmatched_references,
        ["target_tree_id", "reference_points", "best_iou"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
