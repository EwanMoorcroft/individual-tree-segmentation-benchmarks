"""Strict row-alignment and semantic contracts for ForAINet outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FORAINET_CLASS_NAMES = {
    0: "low_vegetation",
    1: "ground",
    2: "stem_points",
    3: "live_branches",
    4: "branches",
}
FORAINET_TO_BENCHMARK_CLASS = {0: 0, 1: 0, 2: 4, 3: 5, 4: 6}
UNCOVERED_PREDICTION_SENTINEL = -1
REFERENCE_TREE_CLASSES = (4, 5, 6)
IGNORED_REFERENCE_CLASSES = (0, 1, 2, 3)
IGNORED_INSTANCE_IDS = (-1, 0)


@dataclass(frozen=True)
class AlignedPrediction:
    pred_tree_id: np.ndarray
    pred_classification: np.ndarray
    source_row_index: np.ndarray


def as_int64(name: str, values: np.ndarray) -> np.ndarray:
    """Return losslessly converted one-dimensional integer data."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if np.any(array > np.iinfo(np.int64).max):
            raise ValueError(f"{name} is outside the int64 range")
    elif np.issubdtype(array.dtype, np.floating):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        if not np.all(array == np.trunc(array)):
            raise ValueError(f"{name} must contain integers")
        limits = np.iinfo(np.int64)
        if np.any(array < limits.min) or np.any(array > limits.max):
            raise ValueError(f"{name} is outside the int64 range")
    converted = array.astype(np.int64, copy=False)
    if not np.array_equal(array, converted):
        raise ValueError(f"{name} cannot be converted without loss")
    return converted


def align_full_resolution_prediction(
    *,
    source_row_index: np.ndarray,
    pred_semantic_internal: np.ndarray,
    pred_instance_id: np.ndarray,
    expected_point_count: int,
) -> AlignedPrediction:
    """Align a complete official post-merge output by stable source row.

    Input rows may be reordered. They must nevertheless contain every source
    row exactly once. Duplicate overlap rows must be resolved by the official
    merger before this function is called.
    """

    if expected_point_count <= 0:
        raise ValueError("expected_point_count must be positive")

    rows = as_int64("source_row_index", source_row_index)
    semantics = as_int64("pred_semantic_internal", pred_semantic_internal)
    instances = as_int64("pred_instance_id", pred_instance_id)
    if not (len(rows) == len(semantics) == len(instances)):
        raise ValueError("prediction arrays have mismatched lengths")
    if len(rows) != expected_point_count:
        raise ValueError(
            "prediction does not contain exactly one row per source point"
        )
    if np.any(rows < 0) or np.any(rows >= expected_point_count):
        raise ValueError("source_row_index contains out-of-range values")
    if len(np.unique(rows)) != expected_point_count:
        raise ValueError("source_row_index contains duplicated or missing rows")

    unknown = sorted(
        set(np.unique(semantics))
        - set(FORAINET_CLASS_NAMES)
        - {UNCOVERED_PREDICTION_SENTINEL}
    )
    if unknown:
        raise ValueError(f"unknown ForAINet semantic class IDs: {unknown}")
    uncovered = semantics == UNCOVERED_PREDICTION_SENTINEL
    if np.any(uncovered & (instances != UNCOVERED_PREDICTION_SENTINEL)):
        raise ValueError(
            "uncovered semantic sentinel must be paired with instance sentinel -1"
        )

    order = np.argsort(rows, kind="stable")
    aligned_rows = rows[order]
    expected_rows = np.arange(expected_point_count, dtype=np.int64)
    if not np.array_equal(aligned_rows, expected_rows):
        raise ValueError("source_row_index is not a complete source-row mapping")
    aligned_semantics = semantics[order]
    aligned_instances = instances[order]

    pred_classification = np.zeros(expected_point_count, dtype=np.uint8)
    covered = aligned_semantics != UNCOVERED_PREDICTION_SENTINEL
    pred_classification[covered] = np.asarray(
        [
            FORAINET_TO_BENCHMARK_CLASS[int(value)]
            for value in aligned_semantics[covered]
        ],
        dtype=np.uint8,
    )
    if np.any(aligned_instances < -1):
        raise ValueError("ForAINet instance IDs must be -1 or non-negative")
    predicted_tree = np.isin(aligned_semantics, (2, 3, 4)) & (
        aligned_instances >= 0
    )
    pred_tree_id = np.zeros(expected_point_count, dtype=np.int64)
    if np.any(predicted_tree):
        maximum = int(np.max(aligned_instances[predicted_tree]))
        if maximum == np.iinfo(np.int64).max:
            raise ValueError("ForAINet instance ID cannot be shifted into int64")
        pred_tree_id[predicted_tree] = aligned_instances[predicted_tree] + 1
    return AlignedPrediction(
        pred_tree_id=pred_tree_id,
        pred_classification=pred_classification,
        source_row_index=expected_rows,
    )


def collapse_identical_overlap_rows(
    *,
    source_row_index: np.ndarray,
    pred_semantic_internal: np.ndarray,
    pred_instance_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse only overlap rows with identical official merged predictions.

    Conflicting tile-level predictions require the official merger and are
    rejected here; the benchmark adapter is not allowed to choose between them.
    """

    rows = as_int64("source_row_index", source_row_index)
    semantics = as_int64("pred_semantic_internal", pred_semantic_internal)
    instances = as_int64("pred_instance_id", pred_instance_id)
    if not (len(rows) == len(semantics) == len(instances)):
        raise ValueError("overlap arrays have mismatched lengths")

    retained: dict[int, tuple[int, int]] = {}
    for row, semantic, instance in zip(rows, semantics, instances):
        candidate = (int(semantic), int(instance))
        previous = retained.get(int(row))
        if previous is not None and previous != candidate:
            raise ValueError(
                f"conflicting overlap predictions for source row {int(row)}"
            )
        retained[int(row)] = candidate

    ordered_rows = np.asarray(sorted(retained), dtype=np.int64)
    ordered_semantics = np.asarray(
        [retained[int(row)][0] for row in ordered_rows], dtype=np.int64
    )
    ordered_instances = np.asarray(
        [retained[int(row)][1] for row in ordered_rows], dtype=np.int64
    )
    return ordered_rows, ordered_semantics, ordered_instances
