from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.diagnostic_metrics import (
    DEFAULT_IOU_THRESHOLDS,
    InstanceEvidence,
    build_instance_evidence,
    evaluate_iou_thresholds,
    point_error_decomposition,
    split_merge_indicators,
)


def _threshold_evidence() -> InstanceEvidence:
    return build_instance_evidence(
        np.array([1, 1, 2, 2, 0]),
        np.array([10, 10, 10, 20, 20]),
        np.array([1, 2]),
        np.array([10, 20]),
    )


def test_build_instance_evidence_counts_intersections_and_iou() -> None:
    evidence = _threshold_evidence()

    np.testing.assert_array_equal(evidence.predicted_point_counts, [2, 2])
    np.testing.assert_array_equal(evidence.reference_point_counts, [3, 2])
    np.testing.assert_array_equal(evidence.intersections, [[2, 0], [1, 1]])
    np.testing.assert_allclose(evidence.iou, [[2 / 3, 0], [1 / 4, 1 / 3]])


def test_build_instance_evidence_applies_boolean_scoring_mask() -> None:
    evidence = build_instance_evidence(
        np.array([1, 1, 2, 2]),
        np.array([10, 10, 20, 20]),
        np.array([1, 2]),
        np.array([10, 20]),
        scoring_mask=np.array([True, False, True, False]),
    )

    np.testing.assert_array_equal(evidence.intersections, [[1, 0], [0, 1]])
    np.testing.assert_allclose(evidence.iou, np.eye(2))


def test_multithreshold_metrics_use_inclusive_boundary_and_rematch() -> None:
    rows = evaluate_iou_thresholds(_threshold_evidence())

    assert tuple(row["iou_threshold"] for row in rows) == DEFAULT_IOU_THRESHOLDS
    assert [row["true_positives"] for row in rows] == [2, 1, 0]
    assert [row["false_positives"] for row in rows] == [0, 1, 2]
    assert [row["false_negatives"] for row in rows] == [0, 1, 2]
    assert rows[0]["matched_prediction_ids"] == (1, 2)
    assert rows[0]["matched_reference_ids"] == (10, 20)
    assert rows[0]["matched_ious"] == pytest.approx((2 / 3, 1 / 3))
    assert rows[0]["mean_matched_iou"] == pytest.approx(0.5)
    assert rows[0]["median_matched_iou"] == pytest.approx(0.5)
    assert rows[1]["f1"] == 0.5
    assert rows[1]["unmatched_prediction_ids"] == (2,)
    assert rows[1]["unmatched_reference_ids"] == (20,)
    assert rows[2]["matched_pair_count"] == 0
    assert rows[2]["mean_matched_iou"] is None
    assert rows[2]["median_matched_iou"] is None


def test_threshold_metrics_handle_empty_predictions_and_empty_both() -> None:
    reference_only = build_instance_evidence(
        np.array([0, 0]),
        np.array([10, 10]),
        np.array([], dtype=int),
        np.array([10]),
    )
    row = evaluate_iou_thresholds(reference_only, (0.5,))[0]
    assert (row["true_positives"], row["false_positives"], row["false_negatives"]) == (
        0,
        0,
        1,
    )
    assert (row["precision"], row["recall"], row["f1"]) == (0.0, 0.0, 0.0)
    assert row["unmatched_reference_ids"] == (10,)

    empty = build_instance_evidence(
        np.array([0]),
        np.array([0]),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    empty_row = evaluate_iou_thresholds(empty, (0.5,))[0]
    assert empty_row["prediction_instance_count"] == 0
    assert empty_row["reference_instance_count"] == 0
    assert empty_row["f1"] == 0.0
    assert empty_row["mean_matched_iou"] is None


def test_split_merge_indicators_cover_split_merge_and_zero_overlap() -> None:
    evidence = build_instance_evidence(
        np.array([1, 2, 3, 3, 4, 0]),
        np.array([10, 10, 20, 30, 0, 40]),
        np.array([1, 2, 3, 4]),
        np.array([10, 20, 30, 40]),
    )

    values = split_merge_indicators(evidence)

    assert values == {
        "min_intersection_points": 1,
        "prediction_instance_count": 4,
        "reference_instance_count": 4,
        "overlap_edge_count": 4,
        "oversegmented_reference_count": 1,
        "oversegmentation_extra_fragment_count": 1,
        "undersegmented_prediction_count": 1,
        "undersegmentation_extra_reference_count": 1,
        "zero_overlap_prediction_count": 1,
        "zero_overlap_reference_count": 1,
        "prediction_to_reference_count_ratio": 1.0,
    }
    stricter = split_merge_indicators(evidence, min_intersection_points=2)
    assert stricter["overlap_edge_count"] == 0
    assert stricter["zero_overlap_prediction_count"] == 4
    assert stricter["zero_overlap_reference_count"] == 4


def test_split_merge_ratio_is_null_without_references() -> None:
    evidence = build_instance_evidence(
        np.array([1]),
        np.array([0]),
        np.array([1]),
        np.array([], dtype=int),
    )

    values = split_merge_indicators(evidence)

    assert values["prediction_to_reference_count_ratio"] is None
    assert values["zero_overlap_prediction_count"] == 1


def test_point_error_decomposition_counts_semantic_and_assignment_errors() -> None:
    values = point_error_decomposition(
        reference_tree_mask=np.array([True, True, True, False, False, False]),
        predicted_tree_mask=np.array([True, False, True, True, False, False]),
        predicted_instance_assigned_mask=np.array(
            [True, False, False, True, True, False]
        ),
    )

    assert values["evaluated_point_count"] == 6
    assert values["reference_tree_point_count"] == 3
    assert values["reference_background_point_count"] == 3
    assert values["semantic_omission_point_count"] == 1
    assert values["semantic_omission_rate"] == pytest.approx(1 / 3)
    assert values["semantic_commission_point_count"] == 1
    assert values["semantic_commission_rate"] == pytest.approx(1 / 3)
    assert values["reference_tree_unassigned_point_count"] == 2
    assert values["reference_tree_unassigned_rate"] == pytest.approx(2 / 3)
    assert (
        values["semantically_detected_reference_tree_unassigned_point_count"]
        == 1
    )
    assert values["predicted_instance_on_reference_background_point_count"] == 2
    assert values["predicted_instance_on_reference_background_rate"] == pytest.approx(
        2 / 3
    )
    assert values["predicted_tree_without_instance_point_count"] == 1
    assert values["predicted_instance_without_semantic_tree_point_count"] == 1


def test_point_error_decomposition_applies_mask_and_nulls_zero_denominators() -> None:
    values = point_error_decomposition(
        reference_tree_mask=np.array([False, True]),
        predicted_tree_mask=np.array([False, True]),
        predicted_instance_assigned_mask=np.array([False, True]),
        scoring_mask=np.array([True, False]),
    )

    assert values["evaluated_point_count"] == 1
    assert values["reference_tree_point_count"] == 0
    assert values["predicted_tree_point_count"] == 0
    assert values["semantic_omission_rate"] is None
    assert values["semantic_commission_rate"] is None
    assert values["reference_tree_unassigned_rate"] is None
    assert values["predicted_instance_on_reference_background_rate"] is None


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        (
            (np.zeros((1, 1), dtype=int), np.zeros(1, dtype=int), np.array([], dtype=int), np.array([], dtype=int)),
            ValueError,
            "one-dimensional",
        ),
        (
            (np.zeros(2, dtype=int), np.zeros(1, dtype=int), np.array([], dtype=int), np.array([], dtype=int)),
            ValueError,
            "must align",
        ),
        (
            (np.zeros(1, dtype=float), np.zeros(1, dtype=int), np.array([], dtype=int), np.array([], dtype=int)),
            TypeError,
            "integers",
        ),
        (
            (np.array([1]), np.array([10]), np.array([1, 1]), np.array([10])),
            ValueError,
            "duplicate",
        ),
        (
            (np.array([0]), np.array([10]), np.array([0]), np.array([10])),
            ValueError,
            "positive",
        ),
        (
            (np.array([1]), np.array([10]), np.array([2]), np.array([10])),
            ValueError,
            "absent",
        ),
    ],
)
def test_build_instance_evidence_rejects_invalid_arrays(
    arguments: tuple[np.ndarray, ...],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        build_instance_evidence(*arguments)

    with pytest.raises(TypeError, match="booleans"):
        build_instance_evidence(
            np.array([1]),
            np.array([10]),
            np.array([1]),
            np.array([10]),
            scoring_mask=np.array([1]),
        )


def test_diagnostic_functions_reject_invalid_parameters_and_evidence() -> None:
    evidence = _threshold_evidence()
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_iou_thresholds(evidence, ())
    with pytest.raises(ValueError, match="duplicates"):
        evaluate_iou_thresholds(evidence, (0.5, 0.5))
    with pytest.raises(ValueError, match="interval"):
        evaluate_iou_thresholds(evidence, (0.0,))
    with pytest.raises(TypeError, match="numbers"):
        evaluate_iou_thresholds(evidence, ("0.5",))
    with pytest.raises(ValueError, match="at least 1"):
        split_merge_indicators(evidence, min_intersection_points=0)
    with pytest.raises(TypeError, match="integer"):
        split_merge_indicators(evidence, min_intersection_points=True)

    inconsistent = InstanceEvidence(
        predicted_ids=np.array([1]),
        reference_ids=np.array([10]),
        intersections=np.array([[1]]),
        predicted_point_counts=np.array([1]),
        reference_point_counts=np.array([1]),
        iou=np.array([[0.5]]),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        evaluate_iou_thresholds(inconsistent, (0.5,))

    impossible_intersections = InstanceEvidence(
        predicted_ids=np.array([1]),
        reference_ids=np.array([10, 20]),
        intersections=np.array([[1, 1]]),
        predicted_point_counts=np.array([1]),
        reference_point_counts=np.array([1, 1]),
        iou=np.array([[1.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="marginal"):
        split_merge_indicators(impossible_intersections)


def test_point_error_decomposition_requires_aligned_boolean_vectors() -> None:
    with pytest.raises(TypeError, match="booleans"):
        point_error_decomposition(
            reference_tree_mask=np.array([1]),
            predicted_tree_mask=np.array([True]),
            predicted_instance_assigned_mask=np.array([True]),
        )
    with pytest.raises(ValueError, match="length 1"):
        point_error_decomposition(
            reference_tree_mask=np.array([True]),
            predicted_tree_mask=np.array([True, False]),
            predicted_instance_assigned_mask=np.array([True]),
        )
