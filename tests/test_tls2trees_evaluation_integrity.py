from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import laspy
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import write_xyz_ply


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_script(
    "methods/tls2trees/scripts/evaluation/"
    "evaluate_for_instance_tls2trees_plot.py",
    "tls2trees_integrity_evaluator",
)


def alignment_metadata(
    *,
    scale: tuple[float, float, float] = (0.001, 0.001, 0.001),
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance: float = 0.001,
) -> dict[str, object]:
    return {
        "schema_version": "tls2trees_for_instance_alignment",
        "coordinate_frame": {
            "source": "source_crs",
            "predictions": "restored_source_crs",
            "units": "metres",
            "local_shift_m": [100.0, 200.0, 0.0],
            "predictions_restored_to_source": True,
        },
        "source_las": {
            "scale_m": list(scale),
            "offset_m": list(offset),
        },
        "matching": {
            "coordinate_tolerance_m": tolerance,
            "distance_metric": "euclidean",
        },
    }


def source_row_alignment_metadata() -> dict[str, object]:
    return {
        "schema_version": "tls2trees_for_instance_alignment",
        "point_correspondence": "source_row_via_voxel_representative",
        "raw_coordinate_evaluation_permitted": False,
        "coordinate_frame": {
            "source": "source_crs",
            "aligned_predictions": "source_row_indices_no_coordinates",
            "raw_predictions": "grid_aligned_local_shift",
            "units": "metres",
            "local_shift_m": [100.0, 200.0, 0.0],
            "predictions_restored_to_source": False,
        },
        "source_las": {
            "scale_m": [0.001, 0.001, 0.001],
            "offset_m": [0.0, 0.0, 0.0],
        },
        "matching": {
            "coordinate_tolerance_m": 0.001,
            "distance_metric": "euclidean",
        },
    }


def evaluate(
    *,
    target: str = "leaf_off",
    prediction_names: list[str],
    predicted: list[tuple[float, float, float]],
    owners: list[int],
    source: list[tuple[float, float, float]],
    tree_ids: list[int],
    classes: list[int],
    tolerance: float = 0.001,
) -> dict[str, object]:
    return EVALUATOR.evaluate_coordinate_arrays(
        target=target,
        prediction_names=prediction_names,
        prediction_coordinates=np.asarray(predicted, dtype=np.float64).reshape(-1, 3),
        prediction_instance_index=np.asarray(owners, dtype=np.int64),
        source_coordinates=np.asarray(source, dtype=np.float64).reshape(-1, 3),
        reference_tree_ids=np.asarray(tree_ids, dtype=np.int64),
        classification=np.asarray(classes, dtype=np.int64),
        coordinate_tolerance_m=tolerance,
    )


def write_reference_las(
    path: Path,
    coordinates: np.ndarray,
    tree_ids: np.ndarray,
    classes: np.ndarray,
) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    cloud = laspy.LasData(header)
    cloud.x = coordinates[:, 0]
    cloud.y = coordinates[:, 1]
    cloud.z = coordinates[:, 2]
    cloud.classification = classes.astype(np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = tree_ids.astype(np.int32)
    cloud.write(path)


def test_target_file_selection_is_explicit_and_has_no_fallback(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    write_xyz_ply(
        prediction_dir / "a.leafoff.ply",
        np.array([[0.0, 0.0, 0.0]]),
    )
    write_xyz_ply(
        prediction_dir / "b.leafon.ply",
        np.array([[1.0, 0.0, 0.0]]),
    )
    write_xyz_ply(
        prediction_dir / "unscoped.ply",
        np.array([[2.0, 0.0, 0.0]]),
    )

    assert [path.name for path in EVALUATOR.prediction_files(prediction_dir, "leaf_off")] == [
        "a.leafoff.ply"
    ]
    assert [path.name for path in EVALUATOR.prediction_files(prediction_dir, "leaf_on")] == [
        "b.leafon.ply"
    ]
    with pytest.raises(ValueError, match="Target must be one of"):
        EVALUATOR.prediction_files(prediction_dir, "configured")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert EVALUATOR.prediction_files(empty_dir, "leaf_off") == []


def test_alignment_metadata_declares_frame_tolerance_and_header_scale() -> None:
    canonical = EVALUATOR.validate_alignment_metadata(
        alignment_metadata(),
        actual_source_las_scale_m=(0.001, 0.001, 0.001),
    )

    assert canonical["coordinate_frame"]["predictions"] == "restored_source_crs"
    assert canonical["coordinate_frame"]["units"] == "metres"
    assert canonical["matching"] == {
        "coordinate_tolerance_m": 0.001,
        "distance_metric": "euclidean",
        "source_assignment": "maximum_cardinality_one_to_one",
    }

    not_restored = alignment_metadata()
    not_restored["coordinate_frame"]["predictions_restored_to_source"] = False
    with pytest.raises(ValueError, match="restored"):
        EVALUATOR.validate_alignment_metadata(not_restored)

    with pytest.raises(ValueError, match="scale does not match"):
        EVALUATOR.validate_alignment_metadata(
            alignment_metadata(), actual_source_las_scale_m=(0.01, 0.01, 0.01)
        )


def test_coordinate_multiplicity_is_preserved_in_matching_and_iou() -> None:
    result = evaluate(
        prediction_names=["tree_1.leafoff.ply", "tree_2.leafoff.ply"],
        predicted=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        owners=[0, 0, 1],
        source=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        tree_ids=[1, 1, 2],
        classes=[4, 4, 6],
    )

    assert result["safe_for_scoring"] is True
    assert result["true_positives"] == 2
    assert result["mean_matched_iou"] == pytest.approx(1.0)
    assert result["mean_unweighted_coverage"] == pytest.approx(1.0)
    assert result["mean_weighted_coverage"] == pytest.approx(1.0)
    assert result["evaluated_point_count"] == 3
    assert result["coordinate_alignment"][
        "duplicate_coordinate_row_count_within_instances"
    ] == 1
    assert result["coordinate_alignment"]["source_coordinate_multiplicity"] == {
        "coordinate_row_count": 3,
        "unique_coordinate_count": 2,
        "duplicate_coordinate_group_count": 1,
        "duplicate_coordinate_row_count": 1,
    }
    assert result["matches"][0]["intersection"] == 2
    assert result["coordinate_alignment"]["reference_coordinate_retention"] == 1.0


def test_within_tree_duplicate_reports_unassignable_many_to_one_conflict() -> None:
    result = evaluate(
        prediction_names=["tree_1.leafoff.ply"],
        predicted=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        owners=[0, 0],
        source=[(0.0, 0.0, 0.0)],
        tree_ids=[1],
        classes=[4],
    )

    assert result["safe_for_scoring"] is False
    assert result["status"] == "invalid_coordinate_alignment"
    assert result["coordinate_alignment"][
        "duplicate_coordinate_row_count_within_instances"
    ] == 1
    assert result["coordinate_alignment"][
        "conflicting_many_to_one_prediction_count"
    ] == 1
    assert result["true_positives"] is None


def test_across_tree_duplicate_is_reported_and_cannot_share_a_source_row() -> None:
    result = evaluate(
        prediction_names=["tree_1.leafoff.ply", "tree_2.leafoff.ply"],
        predicted=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        owners=[0, 1],
        source=[(0.0, 0.0, 0.0)],
        tree_ids=[1],
        classes=[4],
    )

    alignment = result["coordinate_alignment"]
    assert alignment["shared_coordinate_group_count_across_instances"] == 1
    assert alignment["prediction_row_count_in_shared_coordinate_groups"] == 2
    assert alignment["contended_source_coordinate_count"] == 1
    assert alignment["conflicting_many_to_one_prediction_count"] == 1
    assert result["safe_for_scoring"] is False


def test_unmatched_and_ambiguous_prediction_coordinates_block_scoring() -> None:
    unmatched = evaluate(
        prediction_names=["tree.leafoff.ply"],
        predicted=[(10.0, 0.0, 0.0)],
        owners=[0],
        source=[(0.0, 0.0, 0.0)],
        tree_ids=[1],
        classes=[4],
    )
    assert unmatched["coordinate_alignment"][
        "unmatched_prediction_coordinate_count_no_candidate"
    ] == 1
    assert unmatched["safe_for_scoring"] is False

    ambiguous = evaluate(
        prediction_names=["tree.leafoff.ply"],
        predicted=[(0.0005, 0.0, 0.0)],
        owners=[0],
        source=[(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)],
        tree_ids=[1, 2],
        classes=[4, 4],
        tolerance=0.001,
    )
    assert ambiguous["coordinate_alignment"][
        "ambiguous_source_identity_prediction_count"
    ] == 1
    assert ambiguous["safe_for_scoring"] is False

    nearest_is_unambiguous = evaluate(
        prediction_names=["tree.leafoff.ply"],
        predicted=[(0.0, 0.0, 0.0)],
        owners=[0],
        source=[(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)],
        tree_ids=[1, 2],
        classes=[4, 4],
        tolerance=0.001,
    )
    assert nearest_is_unambiguous["safe_for_scoring"] is True
    assert nearest_is_unambiguous["coordinate_alignment"][
        "within_tolerance_candidate_edge_count"
    ] == 2
    assert nearest_is_unambiguous["coordinate_alignment"]["candidate_edge_count"] == 1


def test_target_classes_and_nonpositive_tree_ids_are_contractual() -> None:
    common = {
        "prediction_names": [],
        "predicted": [],
        "owners": [],
        "source": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        "tree_ids": [1, 0, -1],
        "classes": [5, 4, 6],
    }
    leaf_off = evaluate(target="leaf_off", **common)
    leaf_on = evaluate(target="leaf_on", **common)

    assert leaf_off["reference_instance_count"] == 0
    assert leaf_off["false_negatives"] == 0
    assert leaf_on["reference_instance_count"] == 1
    assert leaf_on["false_negatives"] == 1
    assert leaf_on["target_contract"]["valid_reference_instance_rule"] == "treeID > 0"
    assert leaf_on["target_contract"]["invalid_reference_instance_rule"] == (
        "treeID <= 0 is background"
    )
    assert leaf_on["reference_contract_audit"]["nonpositive_tree_id_row_count"] == 2

    with pytest.raises(ValueError, match="outside the target contract"):
        evaluate(
            prediction_names=[],
            predicted=[],
            owners=[],
            source=[(0.0, 0.0, 0.0)],
            tree_ids=[1],
            classes=[7],
        )


def test_empty_prediction_and_reference_sets_have_defined_zero_metrics() -> None:
    empty_predictions = evaluate(
        prediction_names=[],
        predicted=[],
        owners=[],
        source=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        tree_ids=[1, 1],
        classes=[4, 4],
    )
    assert empty_predictions["safe_for_scoring"] is True
    assert (
        empty_predictions["true_positives"],
        empty_predictions["false_positives"],
        empty_predictions["false_negatives"],
    ) == (0, 0, 1)
    assert empty_predictions["precision"] == 0.0
    assert empty_predictions["recall"] == 0.0
    assert empty_predictions["f1"] == 0.0

    empty_references = evaluate(
        prediction_names=["background.leafoff.ply"],
        predicted=[(0.0, 0.0, 0.0)],
        owners=[0],
        source=[(0.0, 0.0, 0.0)],
        tree_ids=[0],
        classes=[4],
    )
    assert empty_references["safe_for_scoring"] is True
    assert (
        empty_references["true_positives"],
        empty_references["false_positives"],
        empty_references["false_negatives"],
    ) == (0, 1, 0)
    assert empty_references["coordinate_alignment"][
        "reference_coordinate_retention"
    ] == 1.0


def test_per_plot_metrics_use_shared_matcher_and_expose_split_merge_indicators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, int], float]] = []
    shared_matcher = EVALUATOR.maximum_cardinality_threshold_matching

    def recording_matcher(matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        calls.append((matrix.shape, threshold))
        return shared_matcher(matrix, threshold)

    monkeypatch.setattr(
        EVALUATOR, "maximum_cardinality_threshold_matching", recording_matcher
    )
    result = evaluate(
        prediction_names=[
            "merged.leafoff.ply",
            "fragment_1.leafoff.ply",
            "fragment_2.leafoff.ply",
        ],
        predicted=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ],
        owners=[0, 0, 1, 2],
        source=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
        ],
        tree_ids=[1, 1, 2, 2],
        classes=[4, 4, 6, 6],
    )

    assert calls == [((3, 2), 0.5)]
    assert (
        result["true_positives"],
        result["false_positives"],
        result["false_negatives"],
    ) == (2, 1, 0)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    assert result["f1"] == pytest.approx(0.8)
    assert result["mean_matched_iou"] == pytest.approx(0.5)
    assert result["mean_unweighted_coverage"] == pytest.approx(0.5)
    assert result["mean_weighted_coverage"] == pytest.approx(0.5)
    assert result["evaluated_point_count"] == 4
    assert result["oversegmented_reference_count"] == 2
    assert result["oversegmentation_extra_fragment_count"] == 2
    assert result["undersegmented_prediction_count"] == 1
    assert result["undersegmentation_extra_reference_count"] == 1


def test_loaded_plot_verifies_las_frame_and_ignores_opposite_target(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "plot.las"
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    write_reference_las(
        reference,
        coordinates,
        np.array([1, 1]),
        np.array([4, 6]),
    )
    predictions = tmp_path / "predictions"
    write_xyz_ply(predictions / "tree.leafoff.ply", coordinates)
    write_xyz_ply(
        predictions / "opposite.leafon.ply", np.array([[50.0, 0.0, 0.0]])
    )

    result = EVALUATOR.evaluate_plot(
        target="leaf_off",
        prediction_directory=predictions,
        reference_source=reference,
        alignment_metadata=alignment_metadata(),
    )

    assert result["safe_for_scoring"] is True
    assert result["prediction_pattern"] == "*.leafoff.ply"
    assert result["prediction_instance_count"] == 1
    assert result["true_positives"] == 1
    assert result["coordinate_frame_metadata"]["source_las"]["scale_m"] == [
        0.001,
        0.001,
        0.001,
    ]


def test_source_row_aligned_npz_is_the_preferred_pointwise_route(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "plot.las"
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    write_reference_las(
        reference,
        coordinates,
        np.array([1, 1, 0]),
        np.array([4, 6, 2]),
    )
    aligned = tmp_path / "source_row_predictions.npz"
    np.savez_compressed(
        aligned,
        source_row_index=np.arange(3),
        predicted_instance_id=np.array([1, 1, 0]),
        prediction_names=np.array(["tree.leafoff.ply"]),
        source_las_sha256=np.asarray(EVALUATOR.sha256_file(reference)),
    )

    result = EVALUATOR.evaluate_aligned_plot(
        target="leaf_off",
        aligned_predictions=aligned,
        reference_source=reference,
        alignment_metadata=source_row_alignment_metadata(),
    )

    assert result["safe_for_scoring"] is True
    assert result["evaluator"] == (
        "for_instance_tls2trees_source_row_class3_ignore"
    )
    assert result["point_correspondence"]["mode"] == "source_row_index"
    assert result["coordinate_tolerance_m"] is None
    assert result["raw_alignment_coordinate_tolerance_m"] == 0.001
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["f1"] == 1.0

    with pytest.raises(ValueError, match="Raw coordinate evaluation is disabled"):
        EVALUATOR.evaluate_plot(
            target="leaf_off",
            prediction_directory=tmp_path,
            reference_source=reference,
            alignment_metadata=source_row_alignment_metadata(),
        )


def test_source_row_aligned_route_rejects_row_order_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="np.arange"):
        EVALUATOR.evaluate_source_row_arrays(
            target="leaf_off",
            prediction_names=[],
            predicted_instance_ids=np.array([0, 0]),
            source_row_index=np.array([1, 0]),
            reference_tree_ids=np.array([1, 1]),
            classification=np.array([4, 6]),
        )


def test_source_row_scoring_ignores_class3_outpoints_and_compacts_instances() -> None:
    result = EVALUATOR.evaluate_source_row_arrays(
        target="leaf_on",
        prediction_names=["outside.leafon.ply", "tree.leafon.ply"],
        predicted_instance_ids=np.array([2, 2, 1]),
        source_row_index=np.arange(3),
        reference_tree_ids=np.array([7, 0, 0]),
        classification=np.array([5, 3, 3]),
    )

    assert result["prediction_instance_count"] == 1
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["f1"] == 1.0
    assert result["matches"][0]["prediction"] == "tree.leafon.ply"
    assert result["matches"][0]["predicted_points"] == 1
    assert result["semantic_ignore"] == {
        "ignored_semantic_classes": [3],
        "ignored_source_row_count": 2,
        "raw_prediction_instance_count": 2,
        "raw_predicted_positive_row_count": 3,
        "ignored_predicted_point_count": 2,
        "ignored_prediction_instance_count": 1,
        "evaluated_prediction_instance_count": 1,
        "evaluated_predicted_positive_row_count": 1,
    }
    assert result["point_correspondence"]["raw_predicted_positive_row_count"] == 3
    assert result["point_correspondence"]["predicted_positive_row_count"] == 1


def test_nonignored_background_prediction_still_reduces_iou() -> None:
    result = EVALUATOR.evaluate_source_row_arrays(
        target="leaf_on",
        prediction_names=["tree.leafon.ply"],
        predicted_instance_ids=np.array([1, 1, 1]),
        source_row_index=np.arange(3),
        reference_tree_ids=np.array([7, 0, 0]),
        classification=np.array([5, 2, 2]),
    )

    assert result["prediction_instance_count"] == 1
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["mean_unweighted_coverage"] == pytest.approx(1 / 3)
    assert result["semantic_ignore"]["ignored_predicted_point_count"] == 0


def test_coordinate_scoring_ignores_class3_without_alignment_conflicts() -> None:
    result = evaluate(
        target="leaf_on",
        prediction_names=["tree.leafon.ply", "outside.leafon.ply"],
        predicted=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        owners=[0, 0, 1],
        source=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        tree_ids=[7, 0],
        classes=[5, 3],
    )

    assert result["safe_for_scoring"] is True
    assert result["prediction_instance_count"] == 1
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["matches"][0]["prediction"] == "tree.leafon.ply"
    assert result["coordinate_alignment"]["matched_prediction_coordinate_count"] == 1
    assert result["coordinate_alignment"]["ignored_prediction_coordinate_count"] == 2
    assert result["coordinate_alignment"]["conflicting_many_to_one_prediction_count"] == 0
    assert result["semantic_ignore"]["ignored_prediction_instance_count"] == 1


def test_coordinate_tie_between_ignored_and_scored_rows_is_invalid() -> None:
    result = evaluate(
        target="leaf_on",
        prediction_names=["tree.leafon.ply"],
        predicted=[(0.0005, 0.0, 0.0)],
        owners=[0],
        source=[(0.0, 0.0, 0.0), (0.001, 0.0, 0.0)],
        tree_ids=[7, 0],
        classes=[5, 3],
        tolerance=0.001,
    )

    assert result["safe_for_scoring"] is False
    assert result["coordinate_alignment"]["mixed_ignored_scored_tie_count"] == 1
    assert result["coordinate_alignment"][
        "ambiguous_source_identity_prediction_count"
    ] == 1
