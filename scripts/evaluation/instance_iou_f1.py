"""Evaluate predicted tree instances when external reference instances exist.

Prediction directories must contain one point-cloud file per predicted tree.
References may use the same one-file-per-tree layout, or one LAS/LAZ/PLY point
cloud with an instance-ID field. Coordinates are quantised by
--coordinate-tolerance before point-set IoU is calculated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import read_ply_vertices


SUPPORTED_EXTENSIONS = {".las", ".laz", ".ply"}
NO_REFERENCE_MESSAGE = "No reference instance labels supplied; IoU/F1 cannot be computed."


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_coordinates(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".ply":
        _, points = read_ply_vertices(path, columns=["x", "y", "z"])
        return np.column_stack([points["x"], points["y"], points["z"]])
    if path.suffix.lower() in {".las", ".laz"}:
        import laspy

        cloud = laspy.read(path)
        return np.column_stack([cloud.x, cloud.y, cloud.z])
    raise ValueError(f"Unsupported point-cloud extension: {path.suffix}")


def quantise_coordinates(coordinates: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("--coordinate-tolerance must be greater than zero")
    quantised = np.rint(np.asarray(coordinates, dtype=np.float64) / tolerance).astype(np.int64)
    return np.unique(quantised, axis=0)


def instance_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Instance directory does not exist: {directory}")
    leafoff = sorted(directory.rglob("*.leafoff.ply"))
    if leafoff:
        return leafoff
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_instance_directory(directory: Path, tolerance: float) -> dict[str, np.ndarray]:
    files = instance_files(directory)
    if not files:
        raise ValueError(f"No supported instance point clouds found in: {directory}")
    return {
        str(path.relative_to(directory)): quantise_coordinates(load_coordinates(path), tolerance)
        for path in files
    }


def normalise_label(value: Any) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item)


def load_labelled_reference(
    path: Path,
    label_field: str,
    ignored_labels: set[str],
    tolerance: float,
) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        header, points = read_ply_vertices(path)
        if label_field not in header.columns:
            raise ValueError(f"Reference PLY is missing label field {label_field!r}")
        coordinates = np.column_stack([points["x"], points["y"], points["z"]])
        labels = points[label_field]
    elif suffix in {".las", ".laz"}:
        import laspy

        cloud = laspy.read(path)
        dimensions = list(cloud.point_format.dimension_names)
        if label_field not in dimensions:
            raise ValueError(
                f"Reference point cloud is missing label field {label_field!r}; "
                f"dimensions: {dimensions}"
            )
        coordinates = np.column_stack([cloud.x, cloud.y, cloud.z])
        labels = np.asarray(cloud[label_field])
    else:
        raise ValueError(f"Unsupported reference point-cloud extension: {suffix}")

    instances: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        key = normalise_label(label)
        if key in ignored_labels:
            continue
        instances[key] = quantise_coordinates(coordinates[labels == label], tolerance)
    if not instances:
        raise ValueError("Reference labelled point cloud contains no usable instance labels")
    return instances


def rows_as_void(points: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(points)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def point_set_iou(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 and len(right) == 0:
        return 1.0
    intersection = len(np.intersect1d(rows_as_void(left), rows_as_void(right), assume_unique=True))
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def iou_matrix(
    predictions: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> tuple[list[str], list[str], np.ndarray]:
    prediction_ids = list(predictions)
    reference_ids = list(references)
    matrix = np.zeros((len(prediction_ids), len(reference_ids)), dtype=np.float64)
    reference_bounds = {
        key: (values.min(axis=0), values.max(axis=0)) for key, values in references.items()
    }
    for prediction_index, prediction_id in enumerate(prediction_ids):
        predicted = predictions[prediction_id]
        predicted_min = predicted.min(axis=0)
        predicted_max = predicted.max(axis=0)
        for reference_index, reference_id in enumerate(reference_ids):
            reference_min, reference_max = reference_bounds[reference_id]
            if np.any(predicted_max < reference_min) or np.any(reference_max < predicted_min):
                continue
            matrix[prediction_index, reference_index] = point_set_iou(
                predicted, references[reference_id]
            )
    return prediction_ids, reference_ids, matrix


def maximum_threshold_matching(matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    candidates = [
        [int(index) for index in np.argsort(matrix[row])[::-1] if matrix[row, index] >= threshold]
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
    return sorted((prediction, reference) for reference, prediction in reference_owner.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute point-set instance IoU matching, precision, recall, and F1.",
        epilog=(
            "Predictions/references may be directories with one LAS/LAZ/PLY per tree. "
            "Alternatively use --reference-labelled-point-cloud with --reference-label-field."
        ),
    )
    parser.add_argument("--predicted-instance-dir")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-instance-dir")
    reference.add_argument("--reference-labelled-point-cloud")
    parser.add_argument("--reference-label-field", default="tree_id")
    parser.add_argument("--ignore-reference-labels", default="0,-1")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coordinate-tolerance", type=float, default=0.01)
    parser.add_argument("--output-json", default="results/metadata/instance_iou_f1.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reference_instance_dir and not args.reference_labelled_point_cloud:
        print(NO_REFERENCE_MESSAGE, file=sys.stderr)
        return 2
    if not args.predicted_instance_dir:
        print("No predicted instance directory supplied; IoU/F1 cannot be computed.", file=sys.stderr)
        return 2
    if not 0 < args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be in the interval (0, 1]")

    prediction_dir = resolve_path(args.predicted_instance_dir)
    predictions = load_instance_directory(prediction_dir, args.coordinate_tolerance)
    if args.reference_instance_dir:
        reference_source = resolve_path(args.reference_instance_dir)
        references = load_instance_directory(reference_source, args.coordinate_tolerance)
        reference_mode = "per_instance_directory"
    else:
        reference_source = resolve_path(args.reference_labelled_point_cloud)
        ignored_labels = {
            label.strip() for label in args.ignore_reference_labels.split(",") if label.strip()
        }
        references = load_labelled_reference(
            reference_source,
            args.reference_label_field,
            ignored_labels,
            args.coordinate_tolerance,
        )
        reference_mode = "labelled_point_cloud"

    prediction_ids, reference_ids, matrix = iou_matrix(predictions, references)
    matched_indices = maximum_threshold_matching(matrix, args.iou_threshold)
    matches = [
        {
            "prediction": prediction_ids[prediction],
            "reference": reference_ids[reference],
            "iou": float(matrix[prediction, reference]),
        }
        for prediction, reference in matched_indices
    ]
    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(references) - true_positives
    precision = true_positives / len(predictions) if predictions else 0.0
    recall = true_positives / len(references) if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_matched_iou = (
        float(np.mean([match["iou"] for match in matches])) if matches else 0.0
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_directory": str(prediction_dir),
        "reference_source": str(reference_source),
        "reference_mode": reference_mode,
        "coordinate_tolerance": args.coordinate_tolerance,
        "iou_threshold": args.iou_threshold,
        "prediction_instance_count": len(predictions),
        "reference_instance_count": len(references),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        "matches": matches,
    }
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Mean matched IoU: {mean_matched_iou:.6f}")
    print(f"Metadata: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
