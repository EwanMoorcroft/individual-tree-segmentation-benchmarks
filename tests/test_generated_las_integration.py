from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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

from benchmark.ply_io import read_ply_vertices, write_xyz_ply
from benchmark.result_statistics import summarise_plot_distribution


PREPARER_PATH = (
    "methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
)
ADAPTER_PATH = (
    "methods/tls2trees/scripts/evaluation/"
    "adapt_for_instance_tls2trees_predictions.py"
)
EVALUATOR_PATH = (
    "methods/tls2trees/scripts/evaluation/evaluate_for_instance_tls2trees_plot.py"
)


def _load_script(relative_path: str, module_name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_generated_las(path: Path) -> None:
    """Write a small, deliberately non-spatially ordered labelled cloud."""

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x = np.array(
        [
            100.401,
            100.001,
            100.601,
            100.101,
            100.901,
            100.501,
            100.201,
            100.701,
            101.001,
            100.301,
            101.101,
            100.801,
        ]
    )
    cloud.y = np.array(
        [
            200.041,
            200.001,
            200.061,
            200.011,
            200.091,
            200.051,
            200.021,
            200.071,
            200.101,
            200.031,
            200.111,
            200.081,
        ]
    )
    cloud.z = np.array(
        [
            5.041,
            5.001,
            5.061,
            5.011,
            5.091,
            5.051,
            5.021,
            5.071,
            5.101,
            5.031,
            5.111,
            5.081,
        ]
    )
    cloud.classification = np.array(
        [4, 4, 4, 6, 2, 6, 4, 6, 2, 4, 3, 4], dtype=np.uint8
    )
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.array(
        [20, 10, 30, 10, 0, 20, 10, 30, 0, 20, 0, 30], dtype=np.int32
    )
    cloud.add_extra_dim(
        laspy.ExtraBytesParams(name="source_index", type=np.uint32)
    )
    cloud["source_index"] = np.arange(12, dtype=np.uint32)
    cloud.write(path)


def _write_synthetic_manifest(path: Path, input_las: Path) -> dict[str, object]:
    digest = hashlib.sha256(input_las.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": "FOR-instance synthetic integration fixture",
        "dataset_split": "development",
        "split_metadata_sha256": "synthetic-fixture",
        "plots": [
            {
                "task_index": 0,
                "split": "development",
                "relative_path": "SYNTHETIC/generated_plot.las",
                "collection": "SYNTHETIC",
                "safe_plot_id": "SYNTHETIC_generated_plot",
                "input_las": str(input_las),
                "point_count": 12,
                "reference_tree_count": 3,
                "input_sha256": digest,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_generated_las_runs_through_tls2trees_source_row_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = _load_script(PREPARER_PATH, "generated_las_tls2trees_preparer")
    adapter = _load_script(ADAPTER_PATH, "generated_las_tls2trees_adapter")
    evaluator = _load_script(EVALUATOR_PATH, "generated_las_tls2trees_evaluator")

    input_las = tmp_path / "generated_plot.las"
    manifest_path = tmp_path / "synthetic_manifest.json"
    output_root = tmp_path / "pipeline_outputs"
    _write_generated_las(input_las)
    manifest = _write_synthetic_manifest(manifest_path, input_las)

    source = laspy.read(input_las)
    dimensions = set(source.point_format.dimension_names)
    assert {"X", "Y", "Z", "classification", "treeID", "source_index"} <= dimensions
    assert np.asarray(source["source_index"]).tolist() == list(range(12))
    assert np.asarray(source["treeID"]).tolist() == [
        20,
        10,
        30,
        10,
        0,
        20,
        10,
        30,
        0,
        20,
        0,
        30,
    ]

    # The production loader intentionally admits only the frozen real split.
    # Replace that identity gate with the generated manifest row; every adapter,
    # geometry, source-row, and checksum operation below remains production code.
    monkeypatch.setattr(
        preparer,
        "load_manifest_plot",
        lambda manifest_path, task_index: (
            manifest,
            dict(manifest["plots"][task_index]),  # type: ignore[index]
        ),
    )
    preparation = preparer.prepare_plot(
        manifest_path=manifest_path,
        task_index=0,
        output_root=output_root,
        run_id="synthetic-integration-001",
        tile_size_m=10.0,
        voxel_size_m=0.02,
    )

    plot_root = Path(preparation["plot_root"])
    converted_root = plot_root / "converted"
    source_map_path = converted_root / "source_map.npz"
    tile_paths = sorted((converted_root / "tiles").glob("*.ply"))
    assert preparation["source_point_count"] == 12
    assert preparation["representative_point_count"] == 12
    assert preparation["labels_stripped"] is True
    assert preparation["reference_fields_passed_to_method"] == []
    assert dimensions <= set(preparation["input_dimensions"])
    assert len(tile_paths) == 1
    tile_header, tile_points = read_ply_vertices(tile_paths[0])
    assert tile_header.columns == ["x", "y", "z"]
    assert tile_header.vertex_count == 12
    assert np.column_stack(
        (tile_points["x"], tile_points["y"], tile_points["z"])
    ).shape == (12, 3)

    with np.load(source_map_path) as source_map:
        source_rows = np.asarray(source_map["source_row_index"])
        source_to_representative = np.asarray(
            source_map["source_to_representative_index"]
        )
        representative_rows = np.asarray(
            source_map["representative_source_row_index"]
        )
        representative_xyz = np.asarray(source_map["representative_local_xyz"])
        local_origin = np.asarray(source_map["local_origin_xyz"])
        las_scales = np.asarray(source_map["las_scales"])
        las_offsets = np.asarray(source_map["las_offsets"])
    assert source_rows.tolist() == list(range(12))
    assert source_to_representative.shape == (12,)
    assert representative_xyz.shape == (12, 3)
    assert sorted(representative_rows.tolist()) == list(range(12))
    assert representative_rows.tolist() != list(range(12))

    raw_predictions = plot_root / "raw_predictions"
    prediction_source_rows = {
        "tree-10.leafoff.ply": [1, 3, 6],
        "tree-20.leafoff.ply": [0, 5, 9],
        "tree-extra.leafoff.ply": [4, 8],
    }
    for filename, selected_source_rows in prediction_source_rows.items():
        representative_indices = source_to_representative[selected_source_rows]
        write_xyz_ply(
            raw_predictions / filename,
            representative_xyz[representative_indices],
        )

    source_digest = hashlib.sha256(input_las.read_bytes()).hexdigest()
    alignment = adapter.adapt_target(
        target="leaf_off",
        raw_root=raw_predictions,
        aligned_root=plot_root / "aligned_predictions",
        source_map_path=source_map_path,
        input_las=input_las,
        input_las_sha256=source_digest,
        local_origin_xyz=local_origin,
        las_scales=las_scales,
        las_offsets=las_offsets,
        tolerance_m=0.001,
    )
    aligned_path = Path(alignment["aligned_prediction_npz"])
    with np.load(aligned_path, allow_pickle=False) as aligned:
        assert aligned["source_row_index"].tolist() == list(range(12))
        assert aligned["predicted_instance_id"].tolist() == [
            2,
            1,
            0,
            1,
            3,
            2,
            1,
            0,
            3,
            2,
            0,
            0,
        ]
        assert aligned["prediction_names"].tolist() == [
            "tree-10.leafoff.ply",
            "tree-20.leafoff.ply",
            "tree-extra.leafoff.ply",
        ]
        assert str(np.asarray(aligned["source_las_sha256"]).item()) == source_digest
    assert alignment["point_correspondence"] == (
        "source_row_via_voxel_representative"
    )
    assert alignment["raw_alignment_diagnostics"]["status"] == "passed"

    plot_metrics_path = tmp_path / "evaluation" / "plot_metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_for_instance_tls2trees_plot.py",
            "--target",
            "leaf_off",
            "--aligned-predictions-npz",
            str(aligned_path),
            "--reference-labelled-point-cloud",
            str(input_las),
            "--alignment-metadata-json",
            str(alignment["alignment_metadata"]),
            "--plot-id",
            "SYNTHETIC/generated_plot",
            "--relative-path",
            "SYNTHETIC/generated_plot.las",
            "--split",
            "dev",
            "--iou-threshold",
            "0.5",
            "--output-json",
            str(plot_metrics_path),
        ],
    )
    assert evaluator.main() == 0
    metrics = json.loads(plot_metrics_path.read_text(encoding="utf-8"))
    assert metrics["point_correspondence"]["mode"] == "source_row_index"
    assert metrics["point_correspondence"]["source_row_order_complete"] is True
    assert metrics["reference_instance_count"] == 3
    assert metrics["prediction_instance_count"] == 3
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["mean_matched_iou"] == pytest.approx(1.0)
    assert metrics["unmatched_predictions"] == ["tree-extra.leafoff.ply"]
    assert metrics["unmatched_references"] == [30]

    aggregate = summarise_plot_distribution(
        [
            {
                "plot_id": metrics["plot_id"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "f1": metrics["f1"],
            }
        ]
    )
    aggregate_path = tmp_path / "evaluation" / "aggregate_metrics.csv"
    evaluator.write_csv(aggregate_path, list(aggregate), [aggregate])
    with aggregate_path.open(encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["true_positives"] == "2"
    assert aggregate_rows[0]["false_positives"] == "1"
    assert aggregate_rows[0]["false_negatives"] == "1"
    assert float(aggregate_rows[0]["micro_f1"]) == pytest.approx(2 / 3)
    assert float(aggregate_rows[0]["median_plot_f1"]) == pytest.approx(2 / 3)
    binary_artifacts = [
        *tmp_path.rglob("*.las"),
        *tmp_path.rglob("*.ply"),
        *tmp_path.rglob("*.npz"),
    ]
    assert binary_artifacts
    assert set(tmp_path.rglob("*.las")) == {input_las}
    assert all(path.is_relative_to(tmp_path) for path in binary_artifacts)
    assert aggregate_path.is_relative_to(tmp_path)
