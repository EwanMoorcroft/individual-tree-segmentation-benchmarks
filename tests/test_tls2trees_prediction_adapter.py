from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import write_xyz_ply


def load_adapter() -> ModuleType:
    path = (
        ROOT
        / "methods/tls2trees/scripts/evaluation/"
        "adapt_for_instance_tls2trees_predictions.py"
    )
    spec = importlib.util.spec_from_file_location("tls2trees_prediction_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source_map(path: Path) -> None:
    np.savez_compressed(
        path,
        source_row_index=np.arange(4),
        source_to_representative_index=np.array([0, 0, 1, 1]),
        representative_source_row_index=np.array([0, 2]),
        representative_local_xyz=np.array([[0.001, 0.0, 1.0], [1.001, 0.0, 1.0]]),
    )


def test_raw_tree_points_project_to_every_source_row_in_their_voxel(
    tmp_path: Path,
) -> None:
    adapter = load_adapter()
    raw = tmp_path / "raw"
    source_map = tmp_path / "source_map.npz"
    write_source_map(source_map)
    write_xyz_ply(raw / "tree_a.leafoff.ply", np.array([[0.0011, 0.0, 1.0]]))
    write_xyz_ply(raw / "tree_b.leafoff.ply", np.array([[1.0009, 0.0, 1.0]]))

    result = adapter.adapt_target(
        target="leaf_off",
        raw_root=raw,
        aligned_root=tmp_path / "aligned",
        source_map_path=source_map,
        input_las=tmp_path / "source.las",
        input_las_sha256="synthetic-source-sha",
        local_origin_xyz=np.array([100.0, 200.0, 0.0]),
        las_scales=np.array([0.001, 0.001, 0.001]),
        las_offsets=np.array([0.0, 0.0, 0.0]),
        tolerance_m=0.001,
    )

    with np.load(result["aligned_prediction_npz"]) as aligned:
        assert aligned["source_row_index"].tolist() == [0, 1, 2, 3]
        assert aligned["predicted_instance_id"].tolist() == [1, 1, 2, 2]
        assert aligned["predicted_semantic_label"].tolist() == [1, 1, 1, 1]
        assert aligned["prediction_names"].tolist() == [
            "tree_a.leafoff.ply",
            "tree_b.leafoff.ply",
        ]
    assert result["point_correspondence"] == "source_row_via_voxel_representative"
    assert result["raw_coordinate_evaluation_permitted"] is False
    assert result["coordinate_frame"]["predictions_restored_to_source"] is False
    assert result["coordinate_frame"]["aligned_predictions"] == (
        "source_row_indices_no_coordinates"
    )
    assert result["raw_alignment_diagnostics"]["status"] == "passed"
    assert result["predicted_source_row_count"] == 4


def test_raw_cross_tree_representative_conflict_is_rejected(tmp_path: Path) -> None:
    adapter = load_adapter()
    representative = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="across_tree_conflicts=1"):
        adapter.assign_raw_to_representatives(
            prediction_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            prediction_owner=np.array([1, 2]),
            representative_xyz=representative,
            tolerance_m=0.001,
        )


def test_shared_leaf_off_representatives_merge_cross_tile_tree_copies() -> None:
    adapter = load_adapter()
    representative = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    raw_to_rep, resolved_owner, diagnostics = adapter.assign_raw_to_representatives(
        prediction_xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        ),
        prediction_owner=np.array([1, 1, 2, 2]),
        representative_xyz=representative,
        tolerance_m=0.001,
        merge_cross_tree_conflicts=True,
    )

    assert raw_to_rep.tolist() == [0, 1, 1, 2]
    assert resolved_owner.tolist() == [1, 1, 1, 1]
    assert diagnostics["owner_components"] == [[1, 2]]
    assert diagnostics["across_tree_conflicting_representative_count"] == 1
    assert diagnostics["post_reconciliation_duplicate_representative_count"] == 1
    assert diagnostics["post_reconciliation_conflicting_representative_count"] == 0
    assert diagnostics["raw_prediction_tree_count"] == 2
    assert diagnostics["resolved_prediction_tree_count"] == 1
    assert diagnostics["merged_owner_component_count"] == 1
    assert diagnostics["status"] == "passed"


def test_leaf_on_must_obey_leaf_off_owner_components() -> None:
    adapter = load_adapter()
    representative = np.array([[0.0, 0.0, 0.0]])
    _, resolved_owner, diagnostics = adapter.assign_raw_to_representatives(
        prediction_xyz=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        prediction_owner=np.array([1, 2]),
        representative_xyz=representative,
        tolerance_m=0.001,
        owner_groups=[[1, 2]],
    )

    assert resolved_owner.tolist() == [1, 1]
    assert diagnostics["owner_reconciliation"] == (
        "provided_leaf_off_owner_components"
    )
    assert diagnostics["post_reconciliation_conflicting_representative_count"] == 0


def test_overlapping_leaf_points_use_nearest_leaf_off_stem_support() -> None:
    adapter = load_adapter()
    representative = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    )
    raw_to_rep, resolved_owner, diagnostics = adapter.assign_raw_to_representatives(
        prediction_xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ),
        prediction_owner=np.array([1, 1, 2, 2]),
        representative_xyz=representative,
        tolerance_m=0.001,
        owner_groups=[[1], [2]],
        conflict_support_by_owner={1: np.array([0]), 2: np.array([2])},
    )

    assert raw_to_rep.tolist() == [0, 1, 1, 2]
    assert resolved_owner.tolist() == [1, 1, 1, 2]
    assert diagnostics["owner_components"] == [[1], [2]]
    assert diagnostics["across_tree_conflicting_representative_count"] == 1
    assert diagnostics["leaf_on_conflicting_representative_count_resolved"] == 1
    assert diagnostics["post_reconciliation_conflicting_representative_count"] == 0
    assert diagnostics["cross_tree_conflict_resolution"] == (
        "nearest_leaf_off_representative_then_lowest_owner"
    )


def test_adapter_records_merged_cross_tile_prediction_sources(tmp_path: Path) -> None:
    adapter = load_adapter()
    raw = tmp_path / "raw"
    source_map = tmp_path / "source_map.npz"
    np.savez_compressed(
        source_map,
        source_row_index=np.arange(3),
        source_to_representative_index=np.arange(3),
        representative_local_xyz=np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
        ),
    )
    write_xyz_ply(
        raw / "tile_0_T0.leafoff.ply",
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
    )
    write_xyz_ply(
        raw / "tile_1_T0.leafoff.ply",
        np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
    )

    result = adapter.adapt_target(
        target="leaf_off",
        raw_root=raw,
        aligned_root=tmp_path / "aligned",
        source_map_path=source_map,
        input_las=tmp_path / "source.las",
        input_las_sha256="synthetic-source-sha",
        local_origin_xyz=np.zeros(3),
        las_scales=np.full(3, 0.001),
        las_offsets=np.zeros(3),
        tolerance_m=0.001,
        merge_cross_tree_conflicts=True,
    )

    assert result["prediction_instance_count"] == 1
    assert result["prediction_source_files"] == [
        ["tile_0_T0.leafoff.ply", "tile_1_T0.leafoff.ply"]
    ]
    with np.load(result["aligned_prediction_npz"]) as aligned:
        assert aligned["predicted_instance_id"].tolist() == [1, 1, 1]
        assert aligned["prediction_names"].tolist() == [
            "merged::tile_0_T0.leafoff.ply|tile_1_T0.leafoff.ply"
        ]


def test_empty_raw_target_produces_aligned_background_rows(tmp_path: Path) -> None:
    adapter = load_adapter()
    raw = tmp_path / "raw"
    raw.mkdir()
    source_map = tmp_path / "source_map.npz"
    write_source_map(source_map)

    result = adapter.adapt_target(
        target="leaf_on",
        raw_root=raw,
        aligned_root=tmp_path / "aligned",
        source_map_path=source_map,
        input_las=tmp_path / "source.las",
        input_las_sha256="synthetic-source-sha",
        local_origin_xyz=np.zeros(3),
        las_scales=np.full(3, 0.001),
        las_offsets=np.zeros(3),
        tolerance_m=0.001,
    )

    with np.load(result["aligned_prediction_npz"]) as aligned:
        assert aligned["predicted_instance_id"].tolist() == [0, 0, 0, 0]
        assert aligned["prediction_names"].size == 0
    assert result["prediction_instance_count"] == 0
    assert result["background_source_row_count"] == 4
