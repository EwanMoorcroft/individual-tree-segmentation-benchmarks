"""Validate whether a SegmentAnyTree export preserves the source point cloud."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def quantised_coordinates(cloud: Any, tolerance: float) -> np.ndarray:
    if tolerance <= 0:
        raise ValueError("Coordinate tolerance must be greater than zero")
    matrix = np.column_stack([cloud.x, cloud.y, cloud.z])
    return np.rint(matrix / tolerance).astype(np.int64)


def structured_coordinates(coordinates: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(coordinates)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).ravel()


def coordinate_summary(coordinates: np.ndarray) -> dict[str, int]:
    keys = structured_coordinates(coordinates)
    unique_keys, counts = np.unique(keys, return_counts=True)
    return {
        "point_count": len(keys),
        "unique_coordinate_count": len(unique_keys),
        "duplicate_coordinate_row_count": int(np.sum(counts - 1)),
        "duplicate_coordinate_group_count": int(np.count_nonzero(counts > 1)),
    }


def coordinate_label_conflicts(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> int:
    if len(coordinates) != len(labels):
        raise ValueError("Coordinates and labels are not aligned")
    if not len(coordinates):
        return 0
    order = np.lexsort(
        (
            labels.astype(np.float64),
            coordinates[:, 2],
            coordinates[:, 1],
            coordinates[:, 0],
        )
    )
    sorted_coordinates = coordinates[order]
    sorted_labels = labels[order].astype(np.float64)
    starts = np.concatenate(
        (
            np.array([0]),
            np.flatnonzero(np.any(np.diff(sorted_coordinates, axis=0), axis=1))
            + 1,
        )
    )
    minimum = np.minimum.reduceat(sorted_labels, starts)
    maximum = np.maximum.reduceat(sorted_labels, starts)
    return int(np.count_nonzero(minimum != maximum))


def coordinate_multisets_equal(
    source_coordinates: np.ndarray,
    output_coordinates: np.ndarray,
) -> bool:
    if len(source_coordinates) != len(output_coordinates):
        return False
    source_keys, source_counts = np.unique(
        structured_coordinates(source_coordinates),
        return_counts=True,
    )
    output_keys, output_counts = np.unique(
        structured_coordinates(output_coordinates),
        return_counts=True,
    )
    return bool(
        np.array_equal(source_keys, output_keys)
        and np.array_equal(source_counts, output_counts)
    )


def validate_export(
    source_path: Path,
    output_path: Path,
    coordinate_tolerance: float,
    reference_instance_field: str,
    prediction_instance_field: str,
) -> dict[str, Any]:
    import laspy

    source = laspy.read(source_path)
    output = laspy.read(output_path)
    source_dimensions = set(source.point_format.dimension_names)
    output_dimensions = set(output.point_format.dimension_names)
    if reference_instance_field not in source_dimensions:
        raise ValueError(
            f"Source is missing {reference_instance_field!r}: {source_path}"
        )
    missing_output = {
        reference_instance_field,
        prediction_instance_field,
    } - output_dimensions
    if missing_output:
        raise ValueError(
            f"Output is missing fields {sorted(missing_output)}: {output_path}"
        )

    source_coordinates = quantised_coordinates(source, coordinate_tolerance)
    output_coordinates = quantised_coordinates(output, coordinate_tolerance)
    source_summary = coordinate_summary(source_coordinates)
    output_summary = coordinate_summary(output_coordinates)
    same_multiset = coordinate_multisets_equal(
        source_coordinates,
        output_coordinates,
    )
    source_reference_conflicts = coordinate_label_conflicts(
        source_coordinates,
        np.asarray(source[reference_instance_field]),
    )
    output_reference_conflicts = coordinate_label_conflicts(
        output_coordinates,
        np.asarray(output[reference_instance_field]),
    )
    output_prediction_conflicts = coordinate_label_conflicts(
        output_coordinates,
        np.asarray(output[prediction_instance_field]),
    )
    point_count_equal = len(source.points) == len(output.points)
    passed = (
        point_count_equal
        and same_multiset
        and source_reference_conflicts == output_reference_conflicts
    )
    return {
        "status": "passed" if passed else "failed",
        "safe_for_final_accuracy_evaluation": passed,
        "coordinate_tolerance": coordinate_tolerance,
        "point_count_equal": point_count_equal,
        "coordinate_multiset_equal": same_multiset,
        "point_count_delta": len(output.points) - len(source.points),
        "source": source_summary,
        "output": output_summary,
        "source_reference_conflicting_coordinate_count": (
            source_reference_conflicts
        ),
        "output_reference_conflicting_coordinate_count": (
            output_reference_conflicts
        ),
        "output_prediction_conflicting_coordinate_count": (
            output_prediction_conflicts
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a SegmentAnyTree labelled export preserves source "
            "point rows and coordinate multiplicity."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coordinate-tolerance", type=float, default=0.001)
    parser.add_argument("--reference-instance-field", default="treeID")
    parser.add_argument("--prediction-instance-field", default="PredInstance")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = resolve_path(args.source)
    output_path = resolve_path(args.output)
    result = validate_export(
        source_path,
        output_path,
        args.coordinate_tolerance,
        args.reference_instance_field,
        args.prediction_instance_field,
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "output": str(output_path),
        **result,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Status: {result['status']}")
    print(f"Point-count delta: {result['point_count_delta']}")
    print(f"Coordinate multiset equal: {result['coordinate_multiset_equal']}")
    print(f"Output: {output_json}")
    return 0 if result["safe_for_final_accuracy_evaluation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
