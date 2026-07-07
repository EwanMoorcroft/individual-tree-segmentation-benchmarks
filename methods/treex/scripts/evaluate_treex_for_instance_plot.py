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


def encode_ids(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if ids.size == 0:
        return np.full(values.shape, -1, dtype=np.int64)
    positions = np.searchsorted(ids, values)
    in_vocab = (
        (positions >= 0)
        & (positions < len(ids))
        & (ids[positions] == values)
    )
    indices = np.full(values.shape, -1, dtype=np.int64)
    indices[in_vocab] = positions[in_vocab]
    return indices


def build_intersection_matrix(
    predicted: np.ndarray,
    reference: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_index = encode_ids(predicted, predicted_ids)
    ref_index = encode_ids(reference, reference_ids)
    pred_counts = np.bincount(
        pred_index[pred_index >= 0],
        minlength=len(predicted_ids),
    ).astype(np.int64)
    ref_counts = np.bincount(
        ref_index[ref_index >= 0],
        minlength=len(reference_ids),
    ).astype(np.int64)

    intersections = np.zeros(
        (len(predicted_ids), len(reference_ids)),
        dtype=np.int64,
    )
    active = (pred_index >= 0) & (ref_index >= 0)
    if np.any(active) and len(reference_ids):
        flat_pairs = pred_index[active] * len(reference_ids) + ref_index[active]
        flat_unique, flat_counts = np.unique(flat_pairs, return_counts=True)
        pred_bins = flat_unique // len(reference_ids)
        ref_bins = flat_unique % len(reference_ids)
        intersections[pred_bins, ref_bins] = flat_counts.astype(np.int64)

    union = (
        pred_counts[:, None].astype(np.float64)
        + ref_counts[None, :].astype(np.float64)
        - intersections.astype(np.float64)
    )
    iou = np.zeros_like(intersections, dtype=np.float64)
    np.divide(
        intersections.astype(np.float64),
        union,
        out=iou,
        where=union > 0,
    )
    return iou, intersections, pred_counts, ref_counts


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

    (
        labelled_iou,
        labelled_intersections,
        labelled_pred_counts,
        labelled_ref_counts,
    ) = build_intersection_matrix(
        predicted_eval,
        target_eval,
        predicted_ids_labelled,
        target_ids,
    )
    candidates: list[dict[str, Any]] = []
    labelled_indices = np.argwhere(labelled_iou > 0)
    for pred_bin, target_bin in labelled_indices:
        intersection = int(labelled_intersections[pred_bin, target_bin])
        target_points = int(labelled_ref_counts[target_bin])
        pred_points = int(labelled_pred_counts[pred_bin])
        union = target_points + pred_points - intersection
        candidates.append(
            {
                "target_tree_id": int(target_ids[target_bin]),
                "pred_tree_id": int(predicted_ids_labelled[pred_bin]),
                "intersection": intersection,
                "target_points": target_points,
                "pred_points_on_labelled_mask": pred_points,
                "union": union,
                "iou": float(labelled_iou[pred_bin, target_bin]),
                "precision": (
                    float(intersection / pred_points)
                    if pred_points
                    else 0.0
                ),
                "recall": (
                    float(intersection / target_points)
                    if target_points
                    else 0.0
                ),
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

    full_iou, _, full_pred_counts, _ = build_intersection_matrix(
        predicted[labelled_tree_mask],
        target[labelled_tree_mask],
        predicted_ids_all,
        target_ids,
    )
    tree_class_indices = encode_ids(predicted[tree_class_mask], predicted_ids_all)
    full_points_on_tree_classes = np.zeros(len(predicted_ids_all), dtype=np.int64)
    if tree_class_indices.size:
        valid_tree_class_indices = tree_class_indices[tree_class_indices >= 0]
        if valid_tree_class_indices.size:
            full_points_on_tree_classes = np.bincount(
                valid_tree_class_indices,
                minlength=len(predicted_ids_all),
            )
    labelled_mask_indices = encode_ids(
        predicted[labelled_tree_mask],
        predicted_ids_all,
    )
    full_points_on_labelled_mask = np.zeros(len(predicted_ids_all), dtype=np.int64)
    if labelled_mask_indices.size:
        valid_labelled_mask_indices = labelled_mask_indices[
            labelled_mask_indices >= 0
        ]
        if valid_labelled_mask_indices.size:
            full_points_on_labelled_mask = np.bincount(
                valid_labelled_mask_indices,
                minlength=len(predicted_ids_all),
            )
    diagnostics: list[dict[str, Any]] = []
    if target_ids.size:
        best_columns = np.argmax(full_iou, axis=1)
        best_ious = full_iou[np.arange(len(predicted_ids_all)), best_columns]
    else:
        best_columns = np.zeros(len(predicted_ids_all), dtype=np.int64)
        best_ious = np.zeros(len(predicted_ids_all), dtype=np.float64)
    for pred_bin, predicted_id in enumerate(predicted_ids_all):
        prediction_points = int(full_pred_counts[pred_bin])  # points on all mask
        best_target = (
            int(target_ids[best_columns[pred_bin]])
            if prediction_points
            and best_ious[pred_bin] > 0
            and len(target_ids)
            else None
        )
        if best_target is None:
            best_score = 0.0
        else:
            best_score = float(best_ious[pred_bin])
        diagnostics.append(
            {
                "plot_id": args.plot_id,
                "pred_tree_id": int(predicted_id),
                "points_total": prediction_points,
                "points_on_tree_classes": int(
                    full_points_on_tree_classes[pred_bin]
                ),
                "points_on_labelled_tree_mask": int(
                    full_points_on_labelled_mask[pred_bin]
                ),
                "labelled_overlap_fraction": (
                    float(
                        full_points_on_labelled_mask[pred_bin]
                        / prediction_points
                    )
                    if prediction_points
                    else 0.0
                ),
                "best_target_tree_id": best_target,
                "best_labelled_mask_iou": best_score,
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
