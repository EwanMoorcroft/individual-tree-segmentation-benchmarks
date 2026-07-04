"""Evaluate predicted tree instances when external reference instances exist.

Prediction directories must contain one point-cloud file per predicted tree.
References may use the same one-file-per-tree layout, or one LAS/LAZ/PLY point
cloud with an instance-ID field. Coordinates are quantised by
--coordinate-tolerance before point-set IoU is calculated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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


def is_nonpositive_label(value: Any) -> bool:
    item = value.item() if hasattr(value, "item") else value
    try:
        return float(item) <= 0
    except (TypeError, ValueError):
        return False


def load_labelled_reference(
    path: Path,
    label_field: str,
    ignored_labels: set[str],
    tolerance: float,
    semantic_field: str | None = None,
    reference_classes: set[float] | None = None,
    ignore_nonpositive_labels: bool = True,
) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        header, points = read_ply_vertices(path)
        if label_field not in header.columns:
            raise ValueError(f"Reference PLY is missing label field {label_field!r}")
        coordinates = np.column_stack([points["x"], points["y"], points["z"]])
        labels = points[label_field]
        if semantic_field is not None:
            if semantic_field not in header.columns:
                raise ValueError(
                    f"Reference PLY is missing semantic field {semantic_field!r}"
                )
            semantic_values = points[semantic_field]
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
        if semantic_field is not None:
            if semantic_field not in dimensions:
                raise ValueError(
                    f"Reference point cloud is missing semantic field "
                    f"{semantic_field!r}; dimensions: {dimensions}"
                )
            semantic_values = np.asarray(cloud[semantic_field])
    else:
        raise ValueError(f"Unsupported reference point-cloud extension: {suffix}")

    if semantic_field is not None:
        if not reference_classes:
            raise ValueError("Reference classes are required with a semantic field")
        semantic_mask = np.isin(semantic_values, list(reference_classes))
        coordinates = coordinates[semantic_mask]
        labels = labels[semantic_mask]

    instances: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        key = normalise_label(label)
        if key in ignored_labels or (
            ignore_nonpositive_labels and is_nonpositive_label(label)
        ):
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--reference-semantic-field")
    parser.add_argument("--reference-classes", nargs="+", type=float)
    parser.add_argument("--ignored-reference-classes", nargs="+", type=float)
    parser.add_argument("--ignore-reference-labels", default="0,-1")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--coordinate-tolerance", type=float, default=0.01)
    parser.add_argument("--plot-name")
    parser.add_argument("--collection")
    parser.add_argument("--split")
    parser.add_argument("--relative-path")
    parser.add_argument(
        "--run-metadata-json",
        help="Optional method-run metadata providing runtime, memory, and status.",
    )
    parser.add_argument("--output-json", default="results/metadata/instance_iou_f1.json")
    parser.add_argument(
        "--output-metrics-csv",
        help="Defaults beside --output-json.",
    )
    parser.add_argument(
        "--output-matches-csv",
        help="Defaults beside --output-json.",
    )
    parser.add_argument(
        "--output-unmatched-predictions-csv",
        help="Defaults beside --output-json.",
    )
    parser.add_argument(
        "--output-unmatched-references-csv",
        help="Defaults beside --output-json.",
    )
    return parser.parse_args()


def main() -> int:
    """Run coordinate-set instance evaluation for a configured reference."""

    args = parse_args()
    if not args.reference_instance_dir and not args.reference_labelled_point_cloud:
        print(NO_REFERENCE_MESSAGE, file=sys.stderr)
        return 2
    if not args.predicted_instance_dir:
        print("No predicted instance directory supplied; IoU/F1 cannot be computed.", file=sys.stderr)
        return 2
    if not 0 < args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be in the interval (0, 1]")
    if args.coordinate_tolerance <= 0:
        raise SystemExit("--coordinate-tolerance must be greater than zero")
    if args.reference_semantic_field and not args.reference_classes:
        raise SystemExit(
            "--reference-classes is required with --reference-semantic-field"
        )
    if args.reference_classes and not args.reference_semantic_field:
        raise SystemExit(
            "--reference-semantic-field is required with --reference-classes"
        )
    if args.ignored_reference_classes and not args.reference_semantic_field:
        raise SystemExit(
            "--reference-semantic-field is required with "
            "--ignored-reference-classes"
        )
    if set(args.reference_classes or []) & set(args.ignored_reference_classes or []):
        raise SystemExit("Reference and ignored semantic classes must not overlap")

    started = time.perf_counter()
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
            semantic_field=args.reference_semantic_field,
            reference_classes=(
                set(args.reference_classes) if args.reference_classes else None
            ),
            ignore_nonpositive_labels=True,
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
    median_matched_iou = (
        float(np.median([match["iou"] for match in matches])) if matches else 0.0
    )
    matched_prediction_ids = {match["prediction"] for match in matches}
    matched_reference_ids = {match["reference"] for match in matches}
    unmatched_predictions = [
        prediction_id
        for prediction_id in prediction_ids
        if prediction_id not in matched_prediction_ids
    ]
    unmatched_references = [
        reference_id
        for reference_id in reference_ids
        if reference_id not in matched_reference_ids
    ]
    evaluation_runtime_seconds = round(time.perf_counter() - started, 6)
    plot_name = args.plot_name or prediction_dir.name
    run_metadata: dict[str, Any] = {}
    run_metadata_path: Path | None = None
    if args.run_metadata_json:
        run_metadata_path = resolve_path(args.run_metadata_json)
        if not run_metadata_path.is_file():
            raise FileNotFoundError(f"Run metadata does not exist: {run_metadata_path}")
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(run_metadata, dict):
            raise ValueError(f"Run metadata must contain a JSON object: {run_metadata_path}")
    method_runtime_seconds = run_metadata.get("runtime_seconds")
    peak_memory_gb = run_metadata.get("peak_memory_gb")
    run_status = run_metadata.get("status", "evaluated")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_directory": str(prediction_dir),
        "reference_source": str(reference_source),
        "reference_mode": reference_mode,
        "reference_label_field": (
            args.reference_label_field if reference_mode == "labelled_point_cloud" else None
        ),
        "reference_semantic_field": args.reference_semantic_field,
        "reference_classes": args.reference_classes,
        "ignored_reference_classes": args.ignored_reference_classes or [],
        "ignored_reference_labels": sorted(ignored_labels)
        if reference_mode == "labelled_point_cloud"
        else [],
        "ignore_nonpositive_reference_labels": True,
        "coordinate_tolerance": args.coordinate_tolerance,
        "iou_threshold": args.iou_threshold,
        "plot_name": plot_name,
        "collection": args.collection,
        "split": args.split,
        "relative_path": args.relative_path,
        "run_metadata_json": str(run_metadata_path) if run_metadata_path else None,
        "runtime_seconds": method_runtime_seconds,
        "peak_memory_gb": peak_memory_gb,
        "status": run_status,
        "prediction_instance_count": len(predictions),
        "reference_instance_count": len(references),
        "predicted_tree_count": len(predictions),
        "reference_tree_count": len(references),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        "median_matched_iou": median_matched_iou,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_references": unmatched_references,
    }
    output_path = resolve_path(args.output_json)
    metrics_path = (
        resolve_path(args.output_metrics_csv)
        if args.output_metrics_csv
        else output_path.with_name(f"{output_path.stem}_metrics.csv")
    )
    matches_path = (
        resolve_path(args.output_matches_csv)
        if args.output_matches_csv
        else output_path.with_name(f"{output_path.stem}_matches.csv")
    )
    unmatched_predictions_path = (
        resolve_path(args.output_unmatched_predictions_csv)
        if args.output_unmatched_predictions_csv
        else output_path.with_name(f"{output_path.stem}_unmatched_predictions.csv")
    )
    unmatched_references_path = (
        resolve_path(args.output_unmatched_references_csv)
        if args.output_unmatched_references_csv
        else output_path.with_name(f"{output_path.stem}_unmatched_references.csv")
    )
    metrics_row = {
        "plot_name": plot_name,
        "collection": args.collection or "",
        "split": args.split or "",
        "relative_path": args.relative_path or "",
        "prediction_directory": str(prediction_dir),
        "reference_source": str(reference_source),
        "evaluation_mode": "leaf_off"
        if set(args.reference_classes or []) == {4.0, 6.0}
        else "configured",
        "reference_classes": ";".join(
            normalise_label(value) for value in (args.reference_classes or [])
        ),
        "ignored_reference_classes": ";".join(
            normalise_label(value)
            for value in (args.ignored_reference_classes or [])
        ),
        "ignored_reference_labels": ";".join(
            sorted(ignored_labels)
            if reference_mode == "labelled_point_cloud"
            else []
        ),
        "predicted_tree_count": len(predictions),
        "reference_tree_count": len(references),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_matched_iou,
        "median_matched_iou": median_matched_iou,
        "iou_threshold": args.iou_threshold,
        "coordinate_tolerance": args.coordinate_tolerance,
        "runtime_seconds": method_runtime_seconds,
        "peak_memory_gb": peak_memory_gb,
        "status": run_status,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
    }
    write_csv(metrics_path, list(metrics_row), [metrics_row])
    write_csv(matches_path, ["prediction", "reference", "iou"], matches)
    write_csv(
        unmatched_predictions_path,
        ["prediction"],
        [{"prediction": prediction} for prediction in unmatched_predictions],
    )
    write_csv(
        unmatched_references_path,
        ["reference"],
        [{"reference": reference} for reference in unmatched_references],
    )
    payload["outputs"] = {
        "metadata_json": str(output_path),
        "metrics_csv": str(metrics_path),
        "matched_pairs_csv": str(matches_path),
        "unmatched_predictions_csv": str(unmatched_predictions_path),
        "unmatched_references_csv": str(unmatched_references_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"Mean matched IoU: {mean_matched_iou:.6f}")
    print(f"Median matched IoU: {median_matched_iou:.6f}")
    print(f"Metadata: {output_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Matches: {matches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
