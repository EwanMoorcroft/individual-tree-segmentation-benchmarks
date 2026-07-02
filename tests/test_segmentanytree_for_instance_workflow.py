from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
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

from benchmark.ply_io import read_ply_vertices, write_xyz_ply


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_annotated_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cloud.y = np.zeros(5)
    cloud.z = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    cloud.classification = np.array([2, 4, 5, 6, 4], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.array([0, 1, 1, 2, 2], dtype=np.int32)
    cloud.write(path)


def write_labelled_prediction_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.array([0.0, 1.0, 2.0, 3.0])
    cloud.y = np.zeros(4)
    cloud.z = np.array([10.0, 11.0, 12.0, 13.0])
    cloud.add_extra_dim(
        laspy.ExtraBytesParams(name="predicted_instance", type=np.int32)
    )
    cloud["predicted_instance"] = np.array([0, 7, 7, 9], dtype=np.int32)
    cloud.write(path)


def minimal_runner_config(project_root: Path, dataset_root: Path) -> dict:
    return {
        "project": {
            "benchmark_name": "for_instance_segmentanytree",
            "barkla_root": str(project_root),
        },
        "dataset": {
            "name": "FOR-instance",
            "root": str(dataset_root),
            "split_metadata_file": "data_split_metadata.csv",
        },
        "method": {
            "repo_path": "external/SegmentAnyTree",
            "command_template": [
                "python",
                "{repo_path}/never_run.py",
                "--input",
                "{staged_input_dir}",
                "--output",
                "{output_dir}",
            ],
        },
        "paths": {
            "predictions_root": "data/predictions/segmentanytree/for_instance",
            "staged_inputs_root": (
                "data/interim/segmentanytree/for_instance/staged_inputs"
            ),
            "run_metadata_root": (
                "results/metadata/segmentanytree_for_instance/runs"
            ),
            "logs_root": "logs/segmentanytree_for_instance",
        },
        "runtime": {"overwrite": False},
    }


def test_segmentanytree_config_has_required_for_instance_fields() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/for_instance_segmentanytree_benchmark.yml").read_text(
            encoding="utf-8"
        )
    )

    assert config["project"]["benchmark_name"] == "for_instance_segmentanytree"
    assert config["dataset"]["name"] == "FOR-instance"
    assert config["dataset"]["root"] == (
        "~/data/datasets/for_instance/FORinstance_dataset"
    )
    assert config["dataset"]["reference_instance_field"] == "treeID"
    assert config["dataset"]["semantic_field"] == "classification"
    assert config["dataset"]["reference_classes"] == [4, 5, 6]
    assert config["dataset"]["ignored_classes"] == [0, 1, 2, 3]
    assert config["dataset"]["ignored_tree_ids"] == [0]
    assert config["dataset"]["pilot"]["relative_path"] == (
        "CULS/plot_1_annotated.las"
    )
    assert config["benchmark"]["array_size"] == 32
    assert config["method"]["execution_mode"] == "apptainer_slurm"
    assert config["method"]["apptainer_image"] == (
        "~/scratch/containers/segment-any-tree_latest.sif"
    )
    assert config["method"]["python_userbase"] == (
        "~/fastscratch/segmentanytree_pyuser_v1"
    )
    assert config["method"]["output_format"] == "labelled_point_cloud"
    assert config["method"]["prediction_instance_field"] == "PredInstance"
    assert config["method"]["gpu_required"] is True
    assert config["method"]["command_template"] is None
    assert config["evaluation"]["iou_threshold"] == 0.5
    assert config["evaluation"]["coordinate_tolerance"] == 0.02


def test_required_for_instance_scripts_are_present_and_tracked() -> None:
    relative_paths = [
        "scripts/data/inspect_for_instance_inventory.py",
        "scripts/data/convert_for_instance_to_tls2trees_ply.py",
        "scripts/data/select_for_instance_plot.py",
    ]
    assert all((ROOT / relative_path).is_file() for relative_path in relative_paths)

    completed = subprocess.run(
        ["git", "ls-files", "--", *relative_paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(completed.stdout.splitlines()) == set(relative_paths)


def test_split_aware_plot_selection_is_deterministic(tmp_path: Path) -> None:
    selector = load_script(
        "scripts/data/select_for_instance_plot.py", "segmentanytree_selector"
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    for relative_path in (
        "NIBIO/plot_2_annotated.las",
        "CULS/plot_1_annotated.las",
    ):
        path = dataset_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_annotated_las(path)
    (dataset_root / "data_split_metadata.csv").write_text(
        "relative_path,split\n"
        "CULS/plot_1_annotated.las,development\n"
        "NIBIO/plot_2_annotated.las,test\n",
        encoding="utf-8",
    )

    records = selector.discover_plots(dataset_root)

    assert [record["relative_path"] for record in records] == [
        "CULS/plot_1_annotated.las",
        "NIBIO/plot_2_annotated.las",
    ]
    assert [record["split"] for record in records] == ["development", "test"]
    assert selector.discover_plots(dataset_root, selected_split="test")[0][
        "relative_path"
    ] == "NIBIO/plot_2_annotated.las"


def test_inventory_records_split_and_required_fields(tmp_path: Path) -> None:
    inventory = load_script(
        "scripts/data/inspect_for_instance_inventory.py",
        "segmentanytree_inventory",
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    input_path = dataset_root / "CULS/plot_1_annotated.las"
    input_path.parent.mkdir(parents=True)
    write_annotated_las(input_path)

    record = inventory.inspect_las(
        input_path, dataset_root, chunk_size=2, split="development"
    )

    assert record["split"] == "development"
    assert record["point_count"] == 5
    assert record["classification_values"] == [2, 4, 5, 6]
    assert record["treeID_positive_count"] == 4
    assert record["treeID_zero_count"] == 1
    assert record["positive_treeID_point_count"] == 4
    assert record["zero_treeID_point_count"] == 1
    assert record["reference_tree_count"] == 2


def test_inventory_cli_accepts_public_output_argument_names(monkeypatch) -> None:
    inventory = load_script(
        "scripts/data/inspect_for_instance_inventory.py",
        "segmentanytree_inventory_cli",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_for_instance_inventory.py",
            "--dataset-root",
            "dataset",
            "--output-csv",
            "inventory.csv",
            "--output-json",
            "inventory.json",
        ],
    )

    args = inventory.parse_args()

    assert args.output_csv == "inventory.csv"
    assert args.output_json == "inventory.json"


def test_segmentanytree_runner_help_loads_selector() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/methods/run_segmentanytree_for_instance.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--dry-run" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


def test_wrapper_dry_run_does_not_execute_method(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runner = load_script(
        "scripts/methods/run_segmentanytree_for_instance.py",
        "segmentanytree_runner_dry",
    )
    dataset_root = tmp_path / "dataset"
    input_path = dataset_root / "CULS/plot_1_annotated.las"
    input_path.parent.mkdir(parents=True)
    write_annotated_las(input_path)
    repo_path = tmp_path / "external/SegmentAnyTree"
    repo_path.mkdir(parents=True)
    config = minimal_runner_config(tmp_path, dataset_root)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segmentanytree_for_instance.py",
            "--config",
            str(config_path),
            "--plot-path",
            "CULS/plot_1_annotated.las",
            "--dry-run",
        ],
    )

    assert runner.main() == 0
    assert not (repo_path / "executed").exists()
    output = capsys.readouterr().out
    assert "Dry run; command not executed" in output
    metadata_path = (
        tmp_path
        / "results/metadata/segmentanytree_for_instance/runs/CULS"
        / "plot_1_annotated_run.json"
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert payload["input_file"] == str(input_path.resolve())
    assert payload["command"][0] == "python"


def test_configured_wrapper_delegates_to_apptainer_slurm(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runner = load_script(
        "scripts/methods/run_segmentanytree_for_instance.py",
        "segmentanytree_runner_slurm",
    )
    dataset_root = tmp_path / "dataset"
    input_path = dataset_root / "CULS/plot_1_annotated.las"
    input_path.parent.mkdir(parents=True)
    write_annotated_las(input_path)
    (dataset_root / "data_split_metadata.csv").write_text(
        "relative_path,split\nCULS/plot_1_annotated.las,dev\n",
        encoding="utf-8",
    )
    (tmp_path / "external/SegmentAnyTree").mkdir(parents=True)
    config = minimal_runner_config(tmp_path, dataset_root)
    config["dataset"]["pilot"] = {
        "relative_path": "CULS/plot_1_annotated.las"
    }
    config["method"]["execution_mode"] = "apptainer_slurm"
    config["method"]["command_template"] = None
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segmentanytree_for_instance.py",
            "--config",
            str(config_path),
            "--plot-path",
            "CULS/plot_1_annotated.las",
            "--dry-run",
        ],
    )

    assert runner.main() == 0
    output = capsys.readouterr().out
    assert "run_segmentanytree_for_instance_pilot_apptainer.sbatch" in output
    payload = json.loads(
        (
            tmp_path
            / "results/metadata/segmentanytree_for_instance/runs/CULS"
            / "plot_1_annotated_run.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["execution_mode"] == "apptainer_slurm"
    assert payload["status"] == "dry_run"


def test_wrapper_parses_peak_memory_only_with_units(tmp_path: Path) -> None:
    runner = load_script(
        "scripts/methods/run_segmentanytree_for_instance.py",
        "segmentanytree_runner_memory",
    )
    log_path = tmp_path / "method.log"
    log_path.write_text(
        "peak memory: 2048 MiB\npeak memory: 3 GB\n", encoding="utf-8"
    )

    assert runner.parse_peak_memory_gb(log_path) == pytest.approx(3.0)


@pytest.mark.parametrize("missing", ["repo", "command"])
def test_wrapper_fails_clearly_when_preflight_is_unresolved(
    tmp_path: Path, monkeypatch, capsys, missing: str
) -> None:
    runner = load_script(
        "scripts/methods/run_segmentanytree_for_instance.py",
        f"segmentanytree_runner_missing_{missing}",
    )
    dataset_root = tmp_path / "dataset"
    input_path = dataset_root / "CULS/plot_1_annotated.las"
    input_path.parent.mkdir(parents=True)
    write_annotated_las(input_path)
    config = minimal_runner_config(tmp_path, dataset_root)
    if missing == "command":
        (tmp_path / "external/SegmentAnyTree").mkdir(parents=True)
        config["method"]["command_template"] = None
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_segmentanytree_for_instance.py",
            "--config",
            str(config_path),
            "--plot-path",
            "CULS/plot_1_annotated.las",
            "--dry-run",
        ],
    )

    assert runner.main() == 2
    error = capsys.readouterr().err
    if missing == "repo":
        assert "SegmentAnyTree checkout does not exist" in error
    else:
        assert runner.COMMAND_REQUIRED_MESSAGE in error


def test_normaliser_splits_tiny_labelled_prediction(tmp_path: Path) -> None:
    normaliser = load_script(
        "scripts/methods/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser",
    )
    input_path = tmp_path / "prediction.las"
    output_dir = tmp_path / "normalised"
    write_labelled_prediction_las(input_path)

    payload = normaliser.normalise(
        input_path=input_path.resolve(),
        output_dir=output_dir.resolve(),
        requested_format="labelled_point_cloud",
        instance_field="predicted_instance",
        ignored_labels={"0", "-1"},
        overwrite=False,
    )

    assert payload["predicted_instance_count"] == 2
    assert payload["ignored_point_count"] == 1
    assert payload["output_point_count"] == 3
    outputs = sorted(output_dir.glob("instance_*.ply"))
    assert [path.name for path in outputs] == [
        "instance_7.ply",
        "instance_9.ply",
    ]
    assert read_ply_vertices(outputs[0])[0].vertex_count == 2


def test_export_patch_is_narrow_and_requires_expected_source() -> None:
    patcher = load_script(
        (
            "scripts/methods/segmentanytree_runtime_patches/"
            "prepare_pandas_to_las_patch.py"
        ),
        "segmentanytree_export_patch",
    )
    source = """
standard_columns_with_data_types = {
    'scan_angle': 'uint16',
}
for column in standard_columns_with_data_types:
    if column in df.columns:
        las_file[column] = df[column].astype(standard_columns_with_data_types[column])
"""

    patched = patcher.patch_source(source)

    assert "'scan_angle': 'int16'" in patched
    assert 'df[column].round() if column == "scan_angle"' in patched
    with pytest.raises(ValueError, match="unsigned scan_angle"):
        patcher.patch_source(patched)


def test_serial_pool_preserves_map_order() -> None:
    serial = load_script(
        "scripts/methods/segmentanytree_runtime_patches/sitecustomize.py",
        "segmentanytree_serial_pool",
    )

    with serial.SerialPool(processes=1) as pool:
        assert pool.map(abs, [-2, -1, 0]) == [2, 1, 0]


def test_evaluator_uses_for_instance_tree_classes(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = load_script(
        "scripts/evaluation/instance_iou_f1.py",
        "segmentanytree_evaluator",
    )
    reference_path = tmp_path / "reference.las"
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    write_annotated_las(reference_path)
    write_xyz_ply(
        predictions / "tree_1.ply",
        np.array([[1.0, 0.0, 11.0], [2.0, 0.0, 12.0]]),
    )
    write_xyz_ply(
        predictions / "tree_2.ply",
        np.array([[3.0, 0.0, 13.0], [4.0, 0.0, 14.0]]),
    )
    write_xyz_ply(
        predictions / "tree_extra.ply",
        np.array([[20.0, 0.0, 20.0]]),
    )
    output_json = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "instance_iou_f1.py",
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
            "5",
            "6",
            "--ignored-reference-classes",
            "0",
            "1",
            "2",
            "3",
            "--ignore-reference-labels",
            "0",
            "--coordinate-tolerance",
            "0.02",
            "--iou-threshold",
            "0.5",
            "--output-json",
            str(output_json),
        ],
    )

    assert evaluator.main() == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["true_positives"] == 2
    assert payload["false_positives"] == 1
    assert payload["false_negatives"] == 0
    assert payload["precision"] == pytest.approx(2 / 3)
    assert payload["recall"] == pytest.approx(1.0)
    assert payload["f1"] == pytest.approx(0.8)
    assert payload["mean_matched_iou"] == pytest.approx(1.0)
    assert payload["median_matched_iou"] == pytest.approx(1.0)


def test_summariser_computes_micro_metrics() -> None:
    summariser = load_script(
        "scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py",
        "segmentanytree_summariser",
    )
    rows = [
        {
            "reference_tree_count": "3",
            "predicted_tree_count": "4",
            "true_positives": "2",
            "false_positives": "2",
            "false_negatives": "1",
            "f1": "0.5714285714",
            "mean_matched_iou": "0.75",
            "median_matched_iou": "0.75",
            "runtime_seconds": "10",
            "peak_memory_gb": "2",
            "status": "completed",
            "metrics_source": "plot_1_metrics.csv",
        },
        {
            "reference_tree_count": "2",
            "predicted_tree_count": "1",
            "true_positives": "1",
            "false_positives": "0",
            "false_negatives": "1",
            "f1": "0.6666666667",
            "mean_matched_iou": "0.6",
            "median_matched_iou": "0.6",
            "runtime_seconds": "20",
            "peak_memory_gb": "3",
            "status": "failed",
            "metrics_source": "plot_2_metrics.csv",
        },
    ]

    summary = summariser.aggregate_rows(rows, "all")

    assert summary["total_true_positives"] == 3
    assert summary["evaluated_plot_count"] == 2
    assert summary["total_false_positives"] == 2
    assert summary["total_false_negatives"] == 2
    assert summary["micro_precision"] == pytest.approx(3 / 5)
    assert summary["micro_recall"] == pytest.approx(3 / 5)
    assert summary["micro_f1"] == pytest.approx(3 / 5)
    assert summary["total_runtime_seconds"] == 30
    assert summary["peak_memory_max_gb"] == 3
    assert summary["completed_count"] == 1
    assert summary["failed_count"] == 1


def test_public_inventory_example_has_required_columns() -> None:
    with (ROOT / "examples/for_instance_inventory_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    assert {
        "relative_path",
        "collection",
        "split",
        "point_count",
        "reference_tree_count",
        "classification_values",
        "has_treeID",
        "has_treeSP",
    } <= columns
    assert rows


def test_public_segmentanytree_pilot_record_matches_evaluation() -> None:
    with (
        ROOT / "examples/segmentanytree_for_instance_pilot_metrics.csv"
    ).open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["split"] == "dev"
    assert int(row["predicted_tree_count"]) == 21
    assert int(row["reference_tree_count"]) == 6
    assert int(row["true_positives"]) == 6
    assert int(row["false_positives"]) == 15
    assert int(row["false_negatives"]) == 0
    assert float(row["f1"]) == pytest.approx(4 / 9)
    assert float(row["mean_matched_iou"]) == pytest.approx(
        0.8507642594155439
    )
    assert row["status"] == "completed_with_postprocess_repair"


def test_no_raw_point_cloud_or_archive_is_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = {".las", ".laz", ".ply", ".zip", ".npy", ".npz"}
    tracked = [Path(line) for line in completed.stdout.splitlines()]

    assert not [path for path in tracked if path.suffix.lower() in forbidden]
