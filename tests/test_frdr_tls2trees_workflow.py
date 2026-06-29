from __future__ import annotations

import importlib.util
import shutil
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

from benchmark.ply_io import read_ply_vertices


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_woods_las(path: Path, woods: list[int] | None = None) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([100.0, 101.0, 102.0])
    las.y = np.array([200.0, 201.0, 202.0])
    las.z = np.array([10.0, 12.0, 11.0])
    las.add_extra_dim(laspy.ExtraBytesParams(name="woods", type=np.uint8))
    las["woods"] = np.array(woods or [1, 2, 1], dtype=np.uint8)
    las.write(path)


def test_frdr_conversion_writes_tls2trees_schema(tmp_path: Path) -> None:
    converter = load_script(
        "scripts/data/convert_frdr_woods_to_tls2trees_ply.py", "frdr_converter"
    )
    input_path = tmp_path / "plot.las"
    output_dir = tmp_path / "converted"
    write_woods_las(input_path)

    metadata = converter.convert(
        input_path=input_path,
        output_dir=output_dir,
        tile_name="001",
        wood_field="woods",
        wood_value=1,
        nonwood_value=2,
        chunk_size=2,
        overwrite=False,
    )

    output_ply = output_dir / "001.downsample.segmented.ply"
    header, points = read_ply_vertices(output_ply)
    assert header.columns == ["x", "y", "z", "n_z", "label"]
    assert header.vertex_count == 3
    assert points["n_z"].tolist() == [0.0, 2.0, 1.0]
    assert points["label"].tolist() == [3.0, 1.0, 3.0]
    tile_index_fields = (output_dir / "tile_index.dat").read_text(encoding="utf-8").split()
    assert tile_index_fields[0] == "001"
    assert len(tile_index_fields) == 5
    assert tile_index_fields[-1] == str(output_ply)
    assert metadata["woods_counts"] == {"1": 2, "2": 1}
    assert metadata["original_point_count"] == 3
    assert metadata["retained_point_count"] == 3
    assert metadata["dropped_unknown_count"] == 0


def test_frdr_conversion_unknown_policy_drop(tmp_path: Path) -> None:
    converter = load_script(
        "scripts/data/convert_frdr_woods_to_tls2trees_ply.py", "frdr_converter_unknown"
    )
    input_path = tmp_path / "unknown.las"
    write_woods_las(input_path, woods=[0, 1, 2])

    with pytest.raises(ValueError, match="Unexpected woods value"):
        converter.convert(
            input_path,
            tmp_path / "fail",
            "001",
            "woods",
            1,
            2,
            2,
            False,
        )

    metadata = converter.convert(
        input_path,
        tmp_path / "drop",
        "001",
        "woods",
        1,
        2,
        2,
        False,
        unknown_policy="drop",
    )
    header, points = read_ply_vertices(
        tmp_path / "drop" / "001.downsample.segmented.ply"
    )

    assert header.vertex_count == 2
    assert points["label"].tolist() == [3.0, 1.0]
    assert points["n_z"].tolist() == [1.0, 0.0]
    assert metadata["original_point_count"] == 3
    assert metadata["retained_point_count"] == 2
    assert metadata["dropped_unknown_count"] == 1
    assert metadata["unknown_woods_values"] == ["0"]
    assert metadata["label_counts"] == {"1": 1, "3": 1}


def test_inventory_reports_unknown_woods_values(tmp_path: Path) -> None:
    inventory = load_script(
        "scripts/data/inspect_frdr_dataset_inventory.py", "frdr_inventory"
    )
    input_path = tmp_path / "NSpruce_plot2.las"
    write_woods_las(input_path, woods=[0, 1, 2])

    record = inventory.inspect_laz(input_path, "woods", 1, 2, 2)

    assert record["filename"] == "NSpruce_plot2.las"
    assert record["point_count"] == 3
    assert record["has_woods"] is True
    assert record["unknown_woods_values"] == ["0"]


def test_tls2trees_output_summary_reads_tree_ply(tmp_path: Path) -> None:
    converter = load_script(
        "scripts/data/convert_frdr_woods_to_tls2trees_ply.py", "frdr_converter_summary"
    )
    summariser = load_script(
        "scripts/methods/summarise_tls2trees_outputs.py", "tls2trees_summariser"
    )
    input_path = tmp_path / "plot.las"
    converted_dir = tmp_path / "converted"
    prediction_dir = tmp_path / "predictions"
    write_woods_las(input_path)
    converter.convert(input_path, converted_dir, "001", "woods", 1, 2, 10, False)
    prediction_dir.mkdir()
    shutil.copy2(
        converted_dir / "001.downsample.segmented.ply",
        prediction_dir / "001_T0.leafoff.ply",
    )

    summary = summariser.summarise_plot(
        "Test_plot",
        prediction_dir,
        tmp_path / "summary.json",
        tmp_path / "summary.csv",
    )

    assert summary["status"] == "complete"
    assert summary["leafoff_file_count"] == 1
    assert summary["total_predicted_tree_points"] == 3
    assert summary["predicted_tree_count"] == 1
    assert summary["min_tree_points"] == 3
    assert summary["max_tree_points"] == 3
    assert summary["mean_tree_points"] == 3.0
    assert summary["bounding_box"]["x_min"] == 100.0


def test_combined_summary_merges_conversion_and_run_metadata() -> None:
    summariser = load_script(
        "scripts/methods/summarise_tls2trees_outputs.py",
        "tls2trees_combined_summariser",
    )
    summary = {
        "plot_name": "Plot",
        "status": "complete",
        "predicted_tree_count": 2,
        "predicted_tree_points": 30,
        "min_tree_points": 10,
        "max_tree_points": 20,
        "mean_tree_points": 15.0,
        "output_directory": "/predictions/Plot",
    }
    conversion = {
        "original_point_count": 100,
        "retained_point_count": 99,
        "dropped_unknown_count": 1,
        "wood_point_count": 40,
        "nonwood_point_count": 59,
    }
    run = {
        "runtime_seconds": 158.0,
        "peak_memory_gb": 2.26,
        "return_code": 0,
        "status": "completed",
    }

    row = summariser.combined_row(summary, conversion, run)

    assert row["input_point_count"] == 100
    assert row["dropped_unknown_count"] == 1
    assert row["predicted_tree_count"] == 2
    assert row["runtime_seconds"] == 158.0
    assert row["peak_memory_gb"] == 2.26
    assert row["status"] == "complete"


def test_instance_command_uses_patched_cli_arguments(tmp_path: Path) -> None:
    runner = load_script(
        "scripts/methods/run_tls2trees_instance_for_plot.py", "tls2trees_plot_runner"
    )
    command = runner.build_command(
        tmp_path / "instance_patched.py",
        tmp_path / "001.downsample.segmented.ply",
        tmp_path / "tile_index.dat",
        tmp_path / "predictions",
        {
            "n_tiles": 1,
            "n_zeros": 3,
            "overlap": False,
            "slice_thickness": 0.5,
            "find_stems_boundary": [1.5, 2.0],
            "ignore_missing_tiles": True,
            "add_leaves": False,
        },
    )

    assert "--find-stems-boundary" in command
    boundary_index = command.index("--find-stems-boundary")
    assert command[boundary_index + 1 : boundary_index + 3] == ["1.5", "2.0"]
    assert "--n-zeros" in command
    assert "--overlap" not in command
    assert "--add-leaves" not in command
    assert "--ignore-missing-tiles" in command


def test_pandas_patch_restores_clstr_column() -> None:
    patch = load_script(
        "scripts/methods/tls2trees_patched/instance_patched.py",
        "tls2trees_instance_patch",
    )
    source = "before\n    chull = chull.reset_index(drop=True)\nafter\n"

    transformed = patch.patched_source(source)

    assert "chull.insert(0, 'clstr'" in transformed
    assert transformed.count("chull = chull.reset_index(drop=True)") == 1


def test_evaluator_refuses_missing_reference(monkeypatch, capsys) -> None:
    evaluator = load_script("scripts/evaluation/instance_iou_f1.py", "instance_iou_f1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["instance_iou_f1.py", "--predicted-instance-dir", "predictions"],
    )

    assert evaluator.main() == 2
    assert evaluator.NO_REFERENCE_MESSAGE in capsys.readouterr().err
