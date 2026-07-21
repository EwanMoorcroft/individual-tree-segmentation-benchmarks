"""Additive diagnostics for retained instance-segmentation predictions.

The functions in this module are deliberately separate from the canonical
evaluators.  They reuse the frozen maximum-cardinality matcher, but do not
write result files or alter accepted benchmark scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .instance_metrics import (
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)


DEFAULT_IOU_THRESHOLDS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class InstanceEvidence:
    """Point-count evidence for predicted-by-reference instance pairs."""

    predicted_ids: np.ndarray
    reference_ids: np.ndarray
    intersections: np.ndarray
    predicted_point_counts: np.ndarray
    reference_point_counts: np.ndarray
    iou: np.ndarray


def _integer_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers")
    if array.dtype.kind == "u" and array.size:
        if int(np.max(array)) > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    return np.asarray(array, dtype=np.int64)


def _instance_ids(values: np.ndarray, *, name: str) -> np.ndarray:
    ids = _integer_vector(values, name=name)
    if np.any(ids <= 0):
        raise ValueError(f"{name} must contain only positive instance IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must not contain duplicate instance IDs")
    return ids


def _boolean_vector(
    values: np.ndarray,
    *,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind != "b":
        raise TypeError(f"{name} must contain booleans")
    if expected_length is not None and len(array) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}")
    return np.asarray(array, dtype=bool)


def _indices_for_ids(labels: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map labels to caller-ordered ID indices without allocating N by K."""

    indices = np.full(len(labels), -1, dtype=np.int64)
    if not len(ids) or not len(labels):
        return indices, np.zeros(len(labels), dtype=bool)

    sorted_order = np.argsort(ids, kind="stable")
    sorted_ids = ids[sorted_order]
    insertion = np.searchsorted(sorted_ids, labels)
    in_range = insertion < len(sorted_ids)
    matched = np.zeros(len(labels), dtype=bool)
    matched[in_range] = sorted_ids[insertion[in_range]] == labels[in_range]
    indices[matched] = sorted_order[insertion[matched]]
    return indices, matched


def build_instance_evidence(
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    predicted_ids: np.ndarray,
    reference_ids: np.ndarray,
    *,
    scoring_mask: np.ndarray | None = None,
) -> InstanceEvidence:
    """Build intersection, marginal-count, and IoU arrays on scored points.

    ``predicted_ids`` and ``reference_ids`` define the valid positive
    instances and their row/column ordering.  Other label values are treated
    as background or ignored.  Every supplied ID must occur at least once
    after applying ``scoring_mask``.
    """

    predicted = _integer_vector(predicted_labels, name="predicted_labels")
    reference = _integer_vector(reference_labels, name="reference_labels")
    if len(predicted) != len(reference):
        raise ValueError("predicted_labels and reference_labels must align")

    prediction_ids = _instance_ids(predicted_ids, name="predicted_ids")
    target_ids = _instance_ids(reference_ids, name="reference_ids")
    mask = (
        np.ones(len(predicted), dtype=bool)
        if scoring_mask is None
        else _boolean_vector(
            scoring_mask,
            name="scoring_mask",
            expected_length=len(predicted),
        )
    )
    predicted = predicted[mask]
    reference = reference[mask]

    prediction_index, valid_prediction = _indices_for_ids(
        predicted, prediction_ids
    )
    reference_index, valid_reference = _indices_for_ids(reference, target_ids)

    predicted_counts = np.bincount(
        prediction_index[valid_prediction], minlength=len(prediction_ids)
    ).astype(np.int64, copy=False)
    reference_counts = np.bincount(
        reference_index[valid_reference], minlength=len(target_ids)
    ).astype(np.int64, copy=False)
    if np.any(predicted_counts == 0):
        missing = prediction_ids[predicted_counts == 0].tolist()
        raise ValueError(
            "predicted_ids absent from scored points: "
            + ", ".join(str(value) for value in missing)
        )
    if np.any(reference_counts == 0):
        missing = target_ids[reference_counts == 0].tolist()
        raise ValueError(
            "reference_ids absent from scored points: "
            + ", ".join(str(value) for value in missing)
        )

    intersections = np.zeros(
        (len(prediction_ids), len(target_ids)), dtype=np.int64
    )
    paired = valid_prediction & valid_reference
    if np.any(paired):
        flat_pairs = (
            prediction_index[paired] * len(target_ids) + reference_index[paired]
        )
        intersections = np.bincount(
            flat_pairs,
            minlength=len(prediction_ids) * len(target_ids),
        ).reshape(len(prediction_ids), len(target_ids))

    unions = (
        predicted_counts[:, None]
        + reference_counts[None, :]
        - intersections
    )
    iou = np.zeros(intersections.shape, dtype=np.float64)
    np.divide(intersections, unions, out=iou, where=unions > 0)

    return InstanceEvidence(
        predicted_ids=prediction_ids.copy(),
        reference_ids=target_ids.copy(),
        intersections=intersections,
        predicted_point_counts=predicted_counts,
        reference_point_counts=reference_counts,
        iou=iou,
    )


def _validated_evidence(evidence: InstanceEvidence) -> InstanceEvidence:
    if not isinstance(evidence, InstanceEvidence):
        raise TypeError("evidence must be an InstanceEvidence")

    predicted_ids = _instance_ids(evidence.predicted_ids, name="predicted_ids")
    reference_ids = _instance_ids(evidence.reference_ids, name="reference_ids")
    intersections = np.asarray(evidence.intersections)
    predicted_counts = _integer_vector(
        evidence.predicted_point_counts, name="predicted_point_counts"
    )
    reference_counts = _integer_vector(
        evidence.reference_point_counts, name="reference_point_counts"
    )
    iou = np.asarray(evidence.iou, dtype=np.float64)

    expected_shape = (len(predicted_ids), len(reference_ids))
    if intersections.shape != expected_shape or intersections.ndim != 2:
        raise ValueError(f"intersections must have shape {expected_shape}")
    if intersections.dtype.kind not in "iu" or intersections.dtype.kind == "b":
        raise TypeError("intersections must contain integers")
    intersections = np.asarray(intersections, dtype=np.int64)
    if len(predicted_counts) != len(predicted_ids):
        raise ValueError("predicted_point_counts must align with predicted_ids")
    if len(reference_counts) != len(reference_ids):
        raise ValueError("reference_point_counts must align with reference_ids")
    if np.any(intersections < 0) or np.any(predicted_counts < 0) or np.any(
        reference_counts < 0
    ):
        raise ValueError("evidence counts cannot be negative")
    if np.any(predicted_counts == 0) or np.any(reference_counts == 0):
        raise ValueError("every instance ID must have a positive point count")
    if np.any(np.sum(intersections, axis=1) > predicted_counts) or np.any(
        np.sum(intersections, axis=0) > reference_counts
    ):
        raise ValueError("intersections cannot exceed marginal point counts")
    if iou.shape != expected_shape or iou.ndim != 2:
        raise ValueError(f"iou must have shape {expected_shape}")
    if np.any(~np.isfinite(iou)) or np.any((iou < 0) | (iou > 1)):
        raise ValueError("iou values must be finite and in the interval [0, 1]")

    unions = (
        predicted_counts[:, None]
        + reference_counts[None, :]
        - intersections
    )
    expected_iou = np.zeros(expected_shape, dtype=np.float64)
    np.divide(intersections, unions, out=expected_iou, where=unions > 0)
    if not np.allclose(iou, expected_iou, rtol=1e-12, atol=1e-12):
        raise ValueError("iou is inconsistent with the supplied point counts")

    return InstanceEvidence(
        predicted_ids=predicted_ids,
        reference_ids=reference_ids,
        intersections=intersections,
        predicted_point_counts=predicted_counts,
        reference_point_counts=reference_counts,
        iou=iou,
    )


def _validated_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    if isinstance(thresholds, (str, bytes)) or not isinstance(
        thresholds, Sequence
    ):
        raise TypeError("thresholds must be a sequence of numbers")
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    normalized: list[float] = []
    for threshold in thresholds:
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise TypeError("thresholds must contain only numbers")
        value = float(threshold)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError("thresholds must be finite and in the interval (0, 1]")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must not contain duplicates")
    return tuple(normalized)


def evaluate_iou_thresholds(
    evidence: InstanceEvidence,
    thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate the frozen one-to-one matcher independently per threshold."""

    values = _validated_evidence(evidence)
    results: list[dict[str, Any]] = []
    for threshold in _validated_thresholds(thresholds):
        matches = maximum_cardinality_threshold_matching(values.iou, threshold)
        prediction_indices = tuple(row for row, _ in matches)
        reference_indices = tuple(column for _, column in matches)
        matched_ious = tuple(
            float(values.iou[row, column]) for row, column in matches
        )
        matched_prediction_indices = set(prediction_indices)
        matched_reference_indices = set(reference_indices)
        unmatched_prediction_indices = tuple(
            index
            for index in range(len(values.predicted_ids))
            if index not in matched_prediction_indices
        )
        unmatched_reference_indices = tuple(
            index
            for index in range(len(values.reference_ids))
            if index not in matched_reference_indices
        )

        true_positives = len(matches)
        false_positives = len(values.predicted_ids) - true_positives
        false_negatives = len(values.reference_ids) - true_positives
        precision, recall, f1 = precision_recall_f1(
            true_positives, false_positives, false_negatives
        )
        results.append(
            {
                "iou_threshold": threshold,
                "prediction_instance_count": len(values.predicted_ids),
                "reference_instance_count": len(values.reference_ids),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_pair_count": true_positives,
                "mean_matched_iou": (
                    float(np.mean(matched_ious)) if matched_ious else None
                ),
                "median_matched_iou": (
                    float(np.median(matched_ious)) if matched_ious else None
                ),
                "matched_prediction_indices": prediction_indices,
                "matched_reference_indices": reference_indices,
                "matched_prediction_ids": tuple(
                    int(values.predicted_ids[index])
                    for index in prediction_indices
                ),
                "matched_reference_ids": tuple(
                    int(values.reference_ids[index])
                    for index in reference_indices
                ),
                "matched_ious": matched_ious,
                "unmatched_prediction_count": len(unmatched_prediction_indices),
                "unmatched_reference_count": len(unmatched_reference_indices),
                "unmatched_prediction_indices": unmatched_prediction_indices,
                "unmatched_reference_indices": unmatched_reference_indices,
                "unmatched_prediction_ids": tuple(
                    int(values.predicted_ids[index])
                    for index in unmatched_prediction_indices
                ),
                "unmatched_reference_ids": tuple(
                    int(values.reference_ids[index])
                    for index in unmatched_reference_indices
                ),
            }
        )
    return results


def split_merge_indicators(
    evidence: InstanceEvidence,
    *,
    min_intersection_points: int = 1,
) -> dict[str, int | float | None]:
    """Return threshold-free split/merge indicators from the overlap graph.

    A split (over-segmentation) is one reference touching two or more
    predictions.  A merge (under-segmentation) is one prediction touching two
    or more references.  Matches and IoU thresholds are intentionally not used.
    """

    values = _validated_evidence(evidence)
    if (
        isinstance(min_intersection_points, bool)
        or not isinstance(min_intersection_points, (int, np.integer))
    ):
        raise TypeError("min_intersection_points must be an integer")
    if min_intersection_points < 1:
        raise ValueError("min_intersection_points must be at least 1")

    overlap = values.intersections >= int(min_intersection_points)
    fragments_per_reference = np.sum(overlap, axis=0, dtype=np.int64)
    references_per_prediction = np.sum(overlap, axis=1, dtype=np.int64)
    prediction_count = len(values.predicted_ids)
    reference_count = len(values.reference_ids)
    return {
        "min_intersection_points": int(min_intersection_points),
        "prediction_instance_count": prediction_count,
        "reference_instance_count": reference_count,
        "overlap_edge_count": int(np.count_nonzero(overlap)),
        "oversegmented_reference_count": int(
            np.count_nonzero(fragments_per_reference > 1)
        ),
        "oversegmentation_extra_fragment_count": int(
            np.sum(np.maximum(fragments_per_reference - 1, 0))
        ),
        "undersegmented_prediction_count": int(
            np.count_nonzero(references_per_prediction > 1)
        ),
        "undersegmentation_extra_reference_count": int(
            np.sum(np.maximum(references_per_prediction - 1, 0))
        ),
        "zero_overlap_prediction_count": int(
            np.count_nonzero(references_per_prediction == 0)
        ),
        "zero_overlap_reference_count": int(
            np.count_nonzero(fragments_per_reference == 0)
        ),
        "prediction_to_reference_count_ratio": (
            float(prediction_count / reference_count)
            if reference_count
            else None
        ),
    }


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def point_error_decomposition(
    *,
    reference_tree_mask: np.ndarray,
    predicted_tree_mask: np.ndarray,
    predicted_instance_assigned_mask: np.ndarray,
    scoring_mask: np.ndarray | None = None,
) -> dict[str, int | float | None]:
    """Count semantic and instance-assignment point errors on one alignment.

    Semantic omission rate uses reference-tree points as its denominator;
    semantic commission rate uses predicted-tree points.  Instance-assignment
    rates use the analogous reference-tree or assigned-instance denominators.
    A null rate means that its denominator is zero.
    """

    reference_tree = _boolean_vector(
        reference_tree_mask, name="reference_tree_mask"
    )
    predicted_tree = _boolean_vector(
        predicted_tree_mask,
        name="predicted_tree_mask",
        expected_length=len(reference_tree),
    )
    predicted_instance = _boolean_vector(
        predicted_instance_assigned_mask,
        name="predicted_instance_assigned_mask",
        expected_length=len(reference_tree),
    )
    evaluated = (
        np.ones(len(reference_tree), dtype=bool)
        if scoring_mask is None
        else _boolean_vector(
            scoring_mask,
            name="scoring_mask",
            expected_length=len(reference_tree),
        )
    )

    reference_tree = reference_tree[evaluated]
    predicted_tree = predicted_tree[evaluated]
    predicted_instance = predicted_instance[evaluated]

    evaluated_count = len(reference_tree)
    reference_tree_count = int(np.count_nonzero(reference_tree))
    reference_background_count = evaluated_count - reference_tree_count
    predicted_tree_count = int(np.count_nonzero(predicted_tree))
    predicted_instance_count = int(np.count_nonzero(predicted_instance))

    semantic_omission = int(np.count_nonzero(reference_tree & ~predicted_tree))
    semantic_commission = int(
        np.count_nonzero(~reference_tree & predicted_tree)
    )
    reference_tree_unassigned = int(
        np.count_nonzero(reference_tree & ~predicted_instance)
    )
    detected_reference_tree_unassigned = int(
        np.count_nonzero(reference_tree & predicted_tree & ~predicted_instance)
    )
    instance_on_background = int(
        np.count_nonzero(~reference_tree & predicted_instance)
    )
    semantic_tree_without_instance = int(
        np.count_nonzero(predicted_tree & ~predicted_instance)
    )
    instance_without_semantic_tree = int(
        np.count_nonzero(predicted_instance & ~predicted_tree)
    )

    return {
        "evaluated_point_count": evaluated_count,
        "reference_tree_point_count": reference_tree_count,
        "reference_background_point_count": reference_background_count,
        "predicted_tree_point_count": predicted_tree_count,
        "predicted_instance_assigned_point_count": predicted_instance_count,
        "semantic_omission_point_count": semantic_omission,
        "semantic_omission_rate": _optional_rate(
            semantic_omission, reference_tree_count
        ),
        "semantic_commission_point_count": semantic_commission,
        "semantic_commission_rate": _optional_rate(
            semantic_commission, predicted_tree_count
        ),
        "reference_tree_unassigned_point_count": reference_tree_unassigned,
        "reference_tree_unassigned_rate": _optional_rate(
            reference_tree_unassigned, reference_tree_count
        ),
        "semantically_detected_reference_tree_unassigned_point_count": (
            detected_reference_tree_unassigned
        ),
        "predicted_instance_on_reference_background_point_count": (
            instance_on_background
        ),
        "predicted_instance_on_reference_background_rate": _optional_rate(
            instance_on_background, predicted_instance_count
        ),
        "predicted_tree_without_instance_point_count": (
            semantic_tree_without_instance
        ),
        "predicted_tree_without_instance_rate": _optional_rate(
            semantic_tree_without_instance, predicted_tree_count
        ),
        "predicted_instance_without_semantic_tree_point_count": (
            instance_without_semantic_tree
        ),
        "predicted_instance_without_semantic_tree_rate": _optional_rate(
            instance_without_semantic_tree, predicted_instance_count
        ),
    }
