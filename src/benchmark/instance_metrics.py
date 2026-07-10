"""Shared metrics for one-to-one instance-segmentation evaluation."""

from __future__ import annotations

import numpy as np


def maximum_cardinality_threshold_matching(
    matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """Return a maximum-cardinality one-to-one match above an IoU threshold.

    Rows represent predicted instances and columns represent references. IoU is
    used only to order candidate edges; the objective is the largest valid
    matching, which is the protocol quantity needed to derive TP, FP, and FN.
    """

    scores = np.asarray(matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("IoU matrix must be two-dimensional")
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    if np.any(~np.isfinite(scores)):
        raise ValueError("IoU matrix must contain only finite values")

    candidates = [
        [
            int(index)
            for index in np.argsort(scores[row], kind="stable")[::-1]
            if scores[row, index] >= threshold
        ]
        for row in range(scores.shape[0])
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
        range(scores.shape[0]),
        key=lambda row: (
            float(np.max(scores[row])) if scores.shape[1] else 0.0,
            -row,
        ),
        reverse=True,
    )
    for prediction in order:
        assign(prediction, set())

    return sorted(
        (prediction, reference)
        for reference, prediction in reference_owner.items()
    )


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from non-negative counts."""

    counts = (true_positives, false_positives, false_negatives)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        for value in counts
    ):
        raise TypeError("TP, FP, and FN must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("TP, FP, and FN cannot be negative")

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        true_positives / recall_denominator if recall_denominator else 0.0
    )
    f1_denominator = (2 * true_positives) + false_positives + false_negatives
    f1 = 2 * true_positives / f1_denominator if f1_denominator else 0.0
    return float(precision), float(recall), float(f1)
