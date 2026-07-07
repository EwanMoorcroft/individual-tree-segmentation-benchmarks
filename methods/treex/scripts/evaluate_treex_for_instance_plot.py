"""Evaluate one TreeX FOR-instance prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]


def load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one TreeX prediction using labelled-mask and strict "
            "whole-prediction metrics."
        )
    )
    parser.add_argument(
        "--config",
        default="methods/treex/configs/for_instance_benchmark.yml",
    )
    parser.add_argument("--prediction-npz", required=True)
    parser.add_argument("--plot-id", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--matches-csv", required=True)
    parser.add_argument("--diagnostics-csv", required=True)
    return parser.parse_args()


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
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
    denominator = (2 * true_positives) + false_positives + false_negatives
    f1 = 0.0 if denominator == 0 else (2 * true_positives) / denominator
    return precision, recall, f1


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    metrics_json_path = Path(args.metrics_json).expanduser().resolve()
    metrics_csv_path = Path(args.metrics_csv).expanduser().resolve()
    matches_path = Path(args.matches_csv).expanduser().resolve()
    diagnostics_path = Path(args.diagnostics_csv).expanduser().resolve()

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction NPZ does not exist: {prediction_path}"
        )

    with np.load(prediction_path) as data:
        required_arrays = {"pred_tree_id", "target_tree_id", "classification"}
        missing_arrays = required_arrays - set(data.files)
        if missing_arrays:
            raise ValueError(
                f"Prediction NPZ is missing arrays {sorted(missing_arrays)}"
            )
        predicted = data["pred_tree_id"].astype(np.int64)
        target = data["target_tree_id"].astype(np.int64)
        classification = data["classification"].astype(np.int64)

    lengths = {len(predicted), len(target), len(classification)}
    if len(lengths) != 1:
        raise ValueError(f"Prediction arrays are not aligned: {sorted(lengths)}")

    tree_classes = set(config["dataset"]["tree_classes"])
    ignored_tree_ids = set(config["dataset"]["ignored_tree_ids"])
    invalid_tree_id = int(config["method"]["params"]["invalid_tree_id"])
    iou_threshold = float(config["evaluation"]["iou_threshold"])

    tree_class_mask = np.isin(classification, sorted(tree_classes))
    labelled_tree_mask = tree_class_mask & ~np.isin(
        target,
        sorted(ignored_tree_ids),
    )
    target_eval = target[labelled_tree_mask]
    predicted_eval = predicted[labelled_tree_mask]

    target_ids = np.array(
        sorted(
            int(value)
            for value in np.unique(target_eval)
            if int(value) not in ignored_tree_ids
        ),
        dtype=np.int64,
    )
    predicted_ids_labelled = np.array(
        sorted(
            int(value)
            for value in np.unique(predicted_eval)
            if int(value) != invalid_tree_id
        ),
        dtype=np.int64,
    )
    predicted_ids_all = np.array(
        sorted(
            int(value)
            for value in np.unique(predicted)
            if int(value) != invalid_tree_id
        ),
        dtype=np.int64,
    )

    candidates: list[dict[str, Any]] = []
    for target_id in target_ids:
        target_mask = target_eval == target_id
        target_points = int(target_mask.sum())
        for predicted_id in predicted_ids_labelled:
            predicted_mask = predicted_eval == predicted_id
            predicted_points = int(predicted_mask.sum())
            intersection = int(np.count_nonzero(target_mask & predicted_mask))
            if intersection == 0:
                continue
            union = target_points + predicted_points - intersection
            candidates.append(
                {
                    "target_tree_id": int(target_id),
                    "pred_tree_id": int(predicted_id),
                    "intersection": intersection,
                    "target_points": target_points,
                    "pred_points_on_labelled_mask": predicted_points,
                    "union": union,
                    "iou": float(intersection / union),
                    "precision": float(intersection / predicted_points),
                    "recall": float(intersection / target_points),
                }
            )

    candidates.sort(
        key=lambda row: (
            -row["iou"],
            row["target_tree_id"],
            row["pred_tree_id"],
        )
    )
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    matches: list[dict[str, Any]] = []
    for row in candidates:
        if row["iou"] < iou_threshold:
            continue
        if row["target_tree_id"] in matched_targets:
            continue
        if row["pred_tree_id"] in matched_predictions:
            continue
        matched_targets.add(row["target_tree_id"])
        matched_predictions.add(row["pred_tree_id"])
        matches.append(row)

    true_positives = len(matches)
    false_negatives = len(target_ids) - true_positives
    false_positives_labelled = (
        len(predicted_ids_labelled) - true_positives
    )
    false_positives_strict = len(predicted_ids_all) - true_positives
    precision_labelled, recall_labelled, f1_labelled = precision_recall_f1(
        true_positives,
        false_positives_labelled,
        false_negatives,
    )
    precision_strict, recall_strict, f1_strict = precision_recall_f1(
        true_positives,
        false_positives_strict,
        false_negatives,
    )

    diagnostics: list[dict[str, Any]] = []
    for predicted_id in predicted_ids_all:
        prediction_mask = predicted == predicted_id
        labelled_overlap = prediction_mask & labelled_tree_mask
        tree_class_overlap = prediction_mask & tree_class_mask
        best_target: int | None = None
        best_iou = 0.0
        for target_id in target_ids:
            target_mask = (target == target_id) & labelled_tree_mask
            intersection = int(
                np.count_nonzero(prediction_mask & target_mask)
            )
            union = int(
                np.count_nonzero(labelled_overlap | target_mask)
            )
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_target = int(target_id)
        prediction_points = int(prediction_mask.sum())
        diagnostics.append(
            {
                "plot_id": args.plot_id,
                "pred_tree_id": int(predicted_id),
                "points_total": prediction_points,
                "points_on_tree_classes": int(tree_class_overlap.sum()),
                "points_on_labelled_tree_mask": int(labelled_overlap.sum()),
                "labelled_overlap_fraction": (
                    float(labelled_overlap.sum() / prediction_points)
                    if prediction_points
                    else 0.0
                ),
                "best_target_tree_id": best_target,
                "best_labelled_mask_iou": float(best_iou),
                "strict_false_positive": (
                    int(predicted_id) not in matched_predictions
                ),
            }
        )

    matched_ious = [row["iou"] for row in matches]
    summary = {
        "method": config["method"]["algorithm"],
        "profile": config["method"]["run_profile"],
        "plot_id": args.plot_id,
        "iou_threshold": iou_threshold,
        "tree_classes": sorted(tree_classes),
        "eval_points_labelled_tree_mask": int(labelled_tree_mask.sum()),
        "reference_trees": int(len(target_ids)),
        "predicted_trees_all": int(len(predicted_ids_all)),
        "predicted_trees_on_labelled_mask": int(
            len(predicted_ids_labelled)
        ),
        "true_positives": int(true_positives),
        "false_positives_labelled_mask": int(false_positives_labelled),
        "false_positives_strict": int(false_positives_strict),
        "false_negatives": int(false_negatives),
        "precision_labelled_mask": float(precision_labelled),
        "recall_labelled_mask": float(recall_labelled),
        "f1_labelled_mask": float(f1_labelled),
        "precision_strict": float(precision_strict),
        "recall_strict": float(recall_strict),
        "f1_strict": float(f1_strict),
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_ious else 0.0
        ),
        "prediction_npz": str(prediction_path),
    }
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(metrics_csv_path, [summary], list(summary))
    write_csv(
        matches_path,
        matches,
        [
            "target_tree_id",
            "pred_tree_id",
            "intersection",
            "target_points",
            "pred_points_on_labelled_mask",
            "union",
            "iou",
            "precision",
            "recall",
        ],
    )
    write_csv(
        diagnostics_path,
        diagnostics,
        [
            "plot_id",
            "pred_tree_id",
            "points_total",
            "points_on_tree_classes",
            "points_on_labelled_tree_mask",
            "labelled_overlap_fraction",
            "best_target_tree_id",
            "best_labelled_mask_iou",
            "strict_false_positive",
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
