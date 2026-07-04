from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import laspy
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import read_ply_vertices, write_tls2trees_ply


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_annotated_las(path: Path, *, include_tree_sp: bool = False) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cloud.y = np.zeros(6)
    cloud.z = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    cloud.classification = np.array([2, 3, 4, 5, 6, 4], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    if include_tree_sp:
        cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeSP", type=np.uint16))
        cloud["treeSP"] = np.array([0, 0, 10, 10, 20, 20], dtype=np.uint16)
    cloud.write(path)


def write_prediction_ply(
    path: Path, coordinates: list[tuple[float, float, float]]
) -> None:
    points = np.asarray(coordinates, dtype=np.float64)
    write_tls2trees_ply(
        path,
        len(points),
        [
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "n_z": points[:, 2] - np.min(points[:, 2]),
                "label": np.full(len(points), 3.0),
            }
        ],
    )


def test_for_instance_tls2trees_config_parses() -> None:
    path = ROOT / "methods/tls2trees/configs/for_instance_accuracy.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["dataset"]["name"] == "FOR-instance"
    assert config["dataset"]["pilot"]["relative_path"] == (
        "CULS/plot_1_annotated.las"
    )
    assert config["conversion"]["evaluation_mode"] == "leaf_off"
    assert config["conversion"]["reference_classes"] == [4, 6]
    assert config["conversion"]["ignored_classes"] == [0, 1, 2, 3, 5]
    assert config["evaluation"]["iou_threshold"] == 0.5
    assert config["evaluation"]["coordinate_tolerance"] == 0.02


def test_for_instance_runner_uses_conservative_instance_parameters(
    tmp_path: Path,
) -> None:
    runner = load_script(
        "methods/tls2trees/scripts/runtime/run_tls2trees_for_instance_plot.py",
        "for_instance_runner",
    )
    config, _, _ = runner.load_config(
        str(ROOT / "methods/tls2trees/configs/for_instance_accuracy.yml")
    )
    command = runner.build_command(
        tmp_path / "instance_patched.py",
        tmp_path / "001.downsample.segmented.ply",
        tmp_path / "tile_index.dat",
        tmp_path / "predictions",
        config["method"]["instance_parameters"],
    )

    assert command[command.index("--n-tiles") + 1] == "1"
    assert command[command.index("--slice-thickness") + 1] == "0.5"
    boundary = command.index("--find-stems-boundary")
    assert command[boundary + 1 : boundary + 3] == ["1.5", "2.0"]
    assert command[command.index("--min-points-per-tree") + 1] == "100"
    assert "--ignore-missing-tiles" in command
    assert "--verbose" in command
    assert "--add-leaves" not in command


def test_for_instance_inventory_reads_reference_fields(tmp_path: Path) -> None:
    inventory = load_script(
        "methods/segmentanytree/scripts/data/inspect_for_instance_inventory.py", "for_inventory"
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    collection = dataset_root / "CULS"
    collection.mkdir(parents=True)
    input_path = collection / "plot_1_annotated.las"
    write_annotated_las(input_path, include_tree_sp=True)

    record = inventory.inspect_las(input_path, dataset_root, chunk_size=2)

    assert record["relative_path"] == "CULS/plot_1_annotated.las"
    assert record["collection"] == "CULS"
    assert record["point_count"] == 6
    assert record["has_treeID"] is True
    assert record["has_treeSP"] is True
    assert record["classification_values"] == [2, 3, 4, 5, 6]
    assert record["positive_treeID_point_count"] == 4
    assert record["zero_treeID_point_count"] == 2
    assert record["reference_tree_count"] == 2


def test_converter_writes_leaf_off_mapping(tmp_path: Path) -> None:
    converter = load_script(
        "methods/tls2trees/scripts/data/convert_for_instance_to_tls2trees_ply.py",
        "for_converter",
    )
    input_path = tmp_path / "plot_1_annotated.las"
    output_dir = tmp_path / "converted"
    write_annotated_las(input_path)

    metadata = converter.convert(
        input_path=input_path,
        output_dir=output_dir,
        tile_name="001",
        reference_classes=(4, 6),
        chunk_size=2,
    )

    output_path = output_dir / "001.downsample.segmented.ply"
    header, points = read_ply_vertices(output_path)
    assert header.vertex_count == 3
    assert points["label"].tolist() == [3.0, 3.0, 3.0]
    assert points["n_z"].tolist() == [0.0, 2.0, 3.0]
    assert metadata["retained_classes"] == [4, 6]
    assert metadata["ignored_classes"] == [2, 3, 5]
    assert metadata["positive_reference_tree_count"] == 2
    assert metadata["label_counts"] == {"1": 0, "3": 3}
    assert (output_dir / "tile_index.dat").is_file()


def test_converter_can_retain_live_branches_as_nonwood(tmp_path: Path) -> None:
    converter = load_script(
        "methods/tls2trees/scripts/data/convert_for_instance_to_tls2trees_ply.py",
        "for_converter_live",
    )
    input_path = tmp_path / "plot_1_annotated.las"
    output_dir = tmp_path / "converted"
    write_annotated_las(input_path)

    metadata = converter.convert(
        input_path=input_path,
        output_dir=output_dir,
        reference_classes=(4, 6),
        retain_live_branches=True,
        chunk_size=10,
    )

    _, points = read_ply_vertices(output_dir / "001.downsample.segmented.ply")
    assert points["label"].tolist() == [3.0, 1.0, 3.0, 3.0]
    assert metadata["retained_classes"] == [4, 5, 6]
    assert metadata["reference_classes"] == [4, 6]
    assert metadata["label_counts"] == {"1": 1, "3": 3}


def test_evaluator_filters_leaf_off_classes_and_writes_tables(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = load_script(
        "shared/evaluation/instance_iou_f1.py",
        "for_instance_evaluator",
    )
    reference_path = tmp_path / "plot_1_annotated.las"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    write_annotated_las(reference_path)
    write_prediction_ply(predictions / "tree_1.leafoff.ply", [(2.0, 0.0, 12.0)])
    write_prediction_ply(
        predictions / "tree_2.leafoff.ply",
        [(4.0, 0.0, 14.0), (5.0, 0.0, 15.0)],
    )
    write_prediction_ply(
        predictions / "tree_extra.leafoff.ply", [(20.0, 0.0, 20.0)]
    )
    metadata_path = tmp_path / "metrics.json"
    metrics_path = tmp_path / "metrics.csv"
    matches_path = tmp_path / "matches.csv"
    unmatched_predictions_path = tmp_path / "unmatched_predictions.csv"
    unmatched_references_path = tmp_path / "unmatched_references.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "instance_iou_f1.py",
            "--plot-name",
            "CULS_plot_1_annotated",
            "--predicted-instance-dir",
            str(predictions),
            "--reference-labelled-point-cloud",
            str(reference_path),
            "--reference-label-field",
            "treeID",
            "--reference-semantic-field",
            "classification",
            "--reference-classes",
            "4",
            "6",
            "--ignored-reference-classes",
            "0",
            "1",
            "2",
            "3",
            "5",
            "--coordinate-tolerance",
            "0.02",
            "--iou-threshold",
            "0.5",
            "--output-json",
            str(metadata_path),
            "--output-metrics-csv",
            str(metrics_path),
            "--output-matches-csv",
            str(matches_path),
            "--output-unmatched-predictions-csv",
            str(unmatched_predictions_path),
            "--output-unmatched-references-csv",
            str(unmatched_references_path),
        ],
    )

    assert evaluator.main() == 0
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["true_positives"] == 2
    assert payload["false_positives"] == 1
    assert payload["false_negatives"] == 0
    assert payload["precision"] == pytest.approx(2 / 3)
    assert payload["recall"] == pytest.approx(1.0)
    assert payload["f1"] == pytest.approx(0.8)
    assert payload["mean_matched_iou"] == pytest.approx(1.0)
    assert payload["reference_classes"] == [4.0, 6.0]
    assert payload["ignored_reference_classes"] == [0.0, 1.0, 2.0, 3.0, 5.0]

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    with matches_path.open(encoding="utf-8", newline="") as handle:
        match_rows = list(csv.DictReader(handle))
    with unmatched_predictions_path.open(encoding="utf-8", newline="") as handle:
        unmatched_prediction_rows = list(csv.DictReader(handle))
    with unmatched_references_path.open(encoding="utf-8", newline="") as handle:
        unmatched_reference_rows = list(csv.DictReader(handle))

    assert metric_rows[0]["true_positives"] == "2"
    assert metric_rows[0]["reference_classes"] == "4;6"
    assert metric_rows[0]["ignored_reference_classes"] == "0;1;2;3;5"
    assert len(match_rows) == 2
    assert len(unmatched_prediction_rows) == 1
    assert unmatched_prediction_rows[0]["prediction"] == "tree_extra.leafoff.ply"
    assert unmatched_reference_rows == []
