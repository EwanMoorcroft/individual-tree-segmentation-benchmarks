"""Evaluate one point-aligned TreeX FOR-instance prediction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.instance_metrics import (
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)


def load_config(path_text: str) -> dict[str, Any]:
    """Load a repository-relative or absolute YAML configuration."""

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one TreeX prediction using the harmonised union mask "
            "and a reference-labelled-mask diagnostic."
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


def encode_ids(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Encode known instance IDs as dense zero-based indices."""

    if ids.size == 0:
        return np.full(values.shape, -1, dtype=np.int64)
    positions = np.searchsorted(ids, values)
    in_bounds = (positions >= 0) & (positions < len(ids))
    in_vocab = np.zeros(values.shape, dtype=bool)
    in_vocab[in_bounds] = ids[positions[in_bounds]] == values[in_bounds]
    indices = np.full(values.shape, -1, dtype=np.int64)
    indices[in_vocab] = positions[in_vocab]
    return indices


def build_intersection_matrix(
    predicted: np.ndarray,
    reference: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pairwise IoU and supporting count matrices for aligned labels."""

    pred_index = encode_ids(predicted, predicted_ids)
    ref_index = encode_ids(reference, reference_ids)
    pred_counts = np.bincount(
        pred_index[pred_index >= 0], minlength=len(predicted_ids)
    ).astype(np.int64)
    ref_counts = np.bincount(
        ref_index[ref_index >= 0], minlength=len(reference_ids)
    ).astype(np.int64)

    intersections = np.zeros(
        (len(predicted_ids), len(reference_ids)), dtype=np.int64
    )
    active = (pred_index >= 0) & (ref_index >= 0)
    if np.any(active) and len(reference_ids):
        flat_pairs = pred_index[active] * len(reference_ids) + ref_index[active]
        flat_unique, flat_counts = np.unique(flat_pairs, return_counts=True)
        intersections[
            flat_unique // len(reference_ids),
            flat_unique % len(reference_ids),
        ] = flat_counts.astype(np.int64)

    union = (
        pred_counts[:, None].astype(np.float64)
        + ref_counts[None, :].astype(np.float64)
        - intersections.astype(np.float64)
    )
    iou = np.zeros_like(intersections, dtype=np.float64)
    np.divide(intersections, union, out=iou, where=union > 0)
    return iou, intersections, pred_counts, ref_counts


def build_matches(
    iou: np.ndarray,
    intersections: np.ndarray,
    predicted_counts: np.ndarray,
    reference_counts: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
    threshold: float,
    protocol: str,
) -> list[dict[str, Any]]:
    """Create match rows from a maximum-cardinality threshold assignment."""

    rows: list[dict[str, Any]] = []
    for pred_bin, ref_bin in maximum_cardinality_threshold_matching(
        iou, threshold
    ):
        intersection = int(intersections[pred_bin, ref_bin])
        pred_points = int(predicted_counts[pred_bin])
        ref_points = int(reference_counts[ref_bin])
        rows.append(
            {
                "protocol": protocol,
                "target_tree_id": int(reference_ids[ref_bin]),
                "pred_tree_id": int(predicted_ids[pred_bin]),
                "intersection": intersection,
                "target_points": ref_points,
                "pred_points": pred_points,
                "union": pred_points + ref_points - intersection,
                "iou": float(iou[pred_bin, ref_bin]),
                "precision": intersection / pred_points if pred_points else 0.0,
                "recall": intersection / ref_points if ref_points else 0.0,
            }
        )
    return rows


def metric_fields(
    matches: list[dict[str, Any]],
    prediction_count: int,
    reference_count: int,
    suffix: str,
) -> dict[str, float | int]:
    """Return explicitly suffixed count and score fields for one protocol."""

    true_positives = len(matches)
    false_positives = prediction_count - true_positives
    false_negatives = reference_count - true_positives
    precision, recall, f1 = precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    matched_ious = [float(row["iou"]) for row in matches]
    return {
        f"true_positives_{suffix}": true_positives,
        f"false_positives_{suffix}": false_positives,
        f"false_negatives_{suffix}": false_negatives,
        f"precision_{suffix}": precision,
        f"recall_{suffix}": recall,
        f"f1_{suffix}": f1,
        f"mean_matched_iou_{suffix}": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        f"median_matched_iou_{suffix}": (
            float(np.median(matched_ious)) if matched_ious else 0.0
        ),
    }


def _best_overlap(
    matrix: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> dict[int, tuple[int | None, float]]:
    """Return the best reference and IoU for each prediction row."""

    result: dict[int, tuple[int | None, float]] = {}
    for row, predicted_id in enumerate(predicted_ids):
        if matrix.shape[1] == 0:
            result[int(predicted_id)] = (None, 0.0)
            continue
        column = int(np.argmax(matrix[row]))
        score = float(matrix[row, column])
        target_id = int(reference_ids[column]) if score > 0 else None
        result[int(predicted_id)] = (target_id, score)
    return result


def evaluate_arrays(
    predicted: np.ndarray,
    target: np.ndarray,
    classification: np.ndarray,
    config: dict[str, Any],
    plot_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate aligned TreeX arrays under primary and diagnostic protocols."""

    arrays = [np.asarray(value) for value in (predicted, target, classification)]
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("Prediction arrays must be one-dimensional")
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Prediction arrays are not aligned: {sorted(lengths)}")
    predicted, target, classification = (
        value.astype(np.int64, copy=False) for value in arrays
    )

    tree_classes = set(config["dataset"]["tree_classes"])
    ignored_tree_ids = set(config["dataset"]["ignored_tree_ids"])
    invalid_tree_id = int(config["method"]["params"]["invalid_tree_id"])
    threshold = float(config["evaluation"]["iou_threshold"])

    tree_class_mask = np.isin(classification, sorted(tree_classes))
    reference_tree_mask = tree_class_mask & ~np.isin(
        target, sorted(ignored_tree_ids)
    )
    predicted_tree_mask = predicted != invalid_tree_id
    union_mask = reference_tree_mask | predicted_tree_mask

    reference_ids = np.array(
        sorted(int(value) for value in np.unique(target[reference_tree_mask])),
        dtype=np.int64,
    )
    labelled_predicted_ids = np.array(
        sorted(
            int(value)
            for value in np.unique(predicted[reference_tree_mask])
            if int(value) != invalid_tree_id
        ),
        dtype=np.int64,
    )
    union_predicted_ids = np.array(
        sorted(
            int(value)
            for value in np.unique(predicted[union_mask])
            if int(value) != invalid_tree_id
        ),
        dtype=np.int64,
    )

    labelled_data = build_intersection_matrix(
        predicted[reference_tree_mask],
        target[reference_tree_mask],
        labelled_predicted_ids,
        reference_ids,
    )
    union_data = build_intersection_matrix(
        predicted[union_mask],
        target[union_mask],
        union_predicted_ids,
        reference_ids,
    )
    labelled_matches = build_matches(
        *labelled_data,
        labelled_predicted_ids,
        reference_ids,
        threshold,
        "reference_labelled_mask_diagnostic",
    )
    union_matches = build_matches(
        *union_data,
        union_predicted_ids,
        reference_ids,
        threshold,
        "harmonized_union_mask",
    )

    summary: dict[str, Any] = {
        "method": config["method"]["algorithm"],
        "profile": config["method"]["run_profile"],
        "plot_id": plot_id,
        "iou_threshold": threshold,
        "matching_policy": "maximum_cardinality_one_to_one",
        "primary_reporting_protocol": "harmonized_union_mask",
        "primary_evaluation_mask": (
            "union_of_reference_tree_and_predicted_tree_points"
        ),
        "tree_classes": sorted(tree_classes),
        "total_points": int(len(predicted)),
        "eval_points_reference_labelled_mask": int(reference_tree_mask.sum()),
        "eval_points_harmonized_union_mask": int(union_mask.sum()),
        "reference_trees": int(len(reference_ids)),
        "predicted_trees_on_reference_labelled_mask": int(
            len(labelled_predicted_ids)
        ),
        "predicted_trees_harmonized_union_mask": int(len(union_predicted_ids)),
        **metric_fields(
            labelled_matches,
            len(labelled_predicted_ids),
            len(reference_ids),
            "labelled_mask",
        ),
        **metric_fields(
            union_matches,
            len(union_predicted_ids),
            len(reference_ids),
            "harmonized",
        ),
    }

    union_best = _best_overlap(union_data[0], union_predicted_ids, reference_ids)
    labelled_best = _best_overlap(
        labelled_data[0], labelled_predicted_ids, reference_ids
    )
    matched_predictions = {int(row["pred_tree_id"]) for row in union_matches}
    diagnostics: list[dict[str, Any]] = []
    for pred_bin, predicted_id_value in enumerate(union_predicted_ids):
        predicted_id = int(predicted_id_value)
        best_target, best_union_iou = union_best[predicted_id]
        _, best_labelled_iou = labelled_best.get(predicted_id, (None, 0.0))
        points_total = int(union_data[2][pred_bin])
        points_on_reference = int(
            np.count_nonzero(
                (predicted == predicted_id) & reference_tree_mask
            )
        )
        diagnostics.append(
            {
                "plot_id": plot_id,
                "pred_tree_id": predicted_id,
                "points_total": points_total,
                "points_on_reference_tree_classes": int(
                    np.count_nonzero(
                        (predicted == predicted_id) & tree_class_mask
                    )
                ),
                "points_on_reference_labelled_mask": points_on_reference,
                "reference_labelled_fraction": (
                    points_on_reference / points_total if points_total else 0.0
                ),
                "best_target_tree_id": best_target,
                "best_harmonized_union_iou": best_union_iou,
                "best_labelled_mask_iou": best_labelled_iou,
                "harmonized_false_positive": predicted_id not in matched_predictions,
            }
        )
    return summary, union_matches, diagnostics


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write rows with a stable header, including empty result sets."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Load one retained prediction, evaluate it, and write public tables."""

    args = parse_args()
    config = load_config(args.config)
    prediction_path = Path(args.prediction_npz).expanduser().resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction NPZ does not exist: {prediction_path}")

    with np.load(prediction_path) as data:
        required_arrays = {"pred_tree_id", "target_tree_id", "classification"}
        missing_arrays = required_arrays - set(data.files)
        if missing_arrays:
            raise ValueError(
                f"Prediction NPZ is missing arrays {sorted(missing_arrays)}"
            )
        summary, matches, diagnostics = evaluate_arrays(
            data["pred_tree_id"],
            data["target_tree_id"],
            data["classification"],
            config,
            args.plot_id,
        )

    summary["prediction_npz"] = str(prediction_path)
    metrics_json_path = Path(args.metrics_json).expanduser().resolve()
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        Path(args.metrics_csv).expanduser().resolve(), [summary], list(summary)
    )
    write_csv(
        Path(args.matches_csv).expanduser().resolve(),
        matches,
        [
            "protocol",
            "target_tree_id",
            "pred_tree_id",
            "intersection",
            "target_points",
            "pred_points",
            "union",
            "iou",
            "precision",
            "recall",
        ],
    )
    write_csv(
        Path(args.diagnostics_csv).expanduser().resolve(),
        diagnostics,
        [
            "plot_id",
            "pred_tree_id",
            "points_total",
            "points_on_reference_tree_classes",
            "points_on_reference_labelled_mask",
            "reference_labelled_fraction",
            "best_target_tree_id",
            "best_harmonized_union_iou",
            "best_labelled_mask_iou",
            "harmonized_false_positive",
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
