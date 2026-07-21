from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        (ROOT / "methods/segmentanytree/configs/for_instance_benchmark.yml").read_text(
            encoding="utf-8"
        )
    )

    assert config["project"]["benchmark_name"] == "for_instance_segmentanytree"
    assert config["project"]["status"] == "completed"
    assert config["project"]["protocol_id"] == "for_instance_pointwise_v1"
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
    assert config["benchmark"]["provisional_coordinate_evaluation_count"] == 32
    assert config["benchmark"]["validated_pointwise_evaluation_count"] == 33
    assert config["benchmark"]["validated_pointwise_evaluation_scope"] == (
        "published_pretrained_fine_tuned_and_historical_retrained_held_out_test"
    )
    assert config["benchmark"]["development_validation_pointwise_evaluation_count"] == 5
    assert config["benchmark"]["primary_reporting_split"] == "test"
    assert config["method"]["execution_mode"] == "apptainer_slurm"
    assert config["method"]["apptainer_image"] == (
        "~/scratch/containers/segment-any-tree_latest.sif"
    )
    assert config["method"]["python_userbase"] == (
        "~/fastscratch/segmentanytree_pyuser_v1"
    )
    assert config["method"]["output_format"] == "labelled_point_cloud"
    assert config["method"]["prediction_instance_field"] == "PredInstance"
    assert config["method"]["checkpoint"]["filename"] == "PointGroup-PAPER.pt"
    assert config["method"]["checkpoint"]["sha256"] == (
        "0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
    )
    assert config["training"]["current_mode"] == (
        "published_pretrained_then_fine_tuned_on_dev"
    )
    assert config["training"]["target_comparison"]["from_scratch_training_permitted"] is False
    assert config["training"]["target_comparison"]["status"] == "completed"
    assert config["training"]["target_comparison"]["public_site_results"] == (
        "methods/segmentanytree/examples/"
        "sat_completed_target_site_results_20260711.csv"
    )
    fine_tuned = config["training"]["target_comparison"]["fine_tuned_result"]
    assert fine_tuned["run_id"] == (
        "segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931"
    )
    assert fine_tuned["final_test_mean_plot_f1"] == pytest.approx(
        0.5446789009405125
    )
    assert fine_tuned["final_test_micro_f1"] == pytest.approx(0.531986531986532)
    assert (
        config["training"]["historical_retrained_experiment"]["latest_completed_run_id"]
        == "sat_for_quicktune_to49_20260706_140730"
    )
    assert (
        config["training"]["historical_retrained_experiment"]["rejected_run_id"]
        == "sat_for_quicktune_to55_20260707_214305"
    )
    assert (
        config["training"]["historical_retrained_experiment"]["final_test_status"]
        == "completed_retained_historical"
    )
    assert config["training"]["forbidden_training_split"] == ["test"]
    assert config["method"]["gpu_required"] is True
    assert config["method"]["command_template"] is None
    assert config["evaluation"]["iou_threshold"] == 0.5
    assert config["evaluation"]["threshold_operator"] == "greater_than_or_equal"
    assert config["evaluation"]["primary_input"] == (
        "aligned_internal_prediction_arrays"
    )
    assert config["evaluation"]["coordinate_rematching"]["status"] == (
        "provisional_only"
    )
    assert config["evaluation"]["coordinate_rematching"]["tolerance"] == 0.02
    assert config["runtime"]["gpu_partition"] == "gpu-l40s-low"


def test_required_for_instance_scripts_are_present_and_tracked() -> None:
    relative_paths = [
        "methods/segmentanytree/scripts/data/inspect_for_instance_inventory.py",
        "methods/tls2trees/scripts/data/convert_for_instance_to_tls2trees_ply.py",
        "methods/segmentanytree/scripts/data/select_for_instance_plot.py",
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
        "methods/segmentanytree/scripts/data/select_for_instance_plot.py", "segmentanytree_selector"
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
        "methods/segmentanytree/scripts/data/inspect_for_instance_inventory.py",
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
        "methods/segmentanytree/scripts/data/inspect_for_instance_inventory.py",
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
            str(ROOT / "methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py"),
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
        "methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py",
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
        "methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py",
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
        "methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py",
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
        "methods/segmentanytree/scripts/runtime/run_segmentanytree_for_instance.py",
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
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
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


def test_normaliser_refuses_symlinked_output_without_touching_target(
    tmp_path: Path,
) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_symlink",
    )
    input_path = tmp_path / "prediction.las"
    protected_dir = tmp_path / "protected"
    protected_dir.mkdir()
    marker = protected_dir / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    output_link = tmp_path / "normalised"
    output_link.symlink_to(protected_dir, target_is_directory=True)
    write_labelled_prediction_las(input_path)

    with pytest.raises(ValueError, match="symlinked destination"):
        normaliser.normalise(
            input_path=input_path,
            output_dir=output_link,
            requested_format="labelled_point_cloud",
            instance_field="predicted_instance",
            ignored_labels={"0", "-1"},
            overwrite=True,
        )

    assert output_link.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_normaliser_allows_canonical_platform_style_parent_alias(
    tmp_path: Path,
) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_parent_alias",
    )
    input_path = tmp_path / "prediction.las"
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias_parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    write_labelled_prediction_las(input_path)

    payload = normaliser.normalise(
        input_path=input_path,
        output_dir=alias_parent / "normalised",
        requested_format="labelled_point_cloud",
        instance_field="predicted_instance",
        ignored_labels={"0", "-1"},
        overwrite=False,
    )

    canonical_output = real_parent / "normalised"
    assert Path(payload["output_path"]) == canonical_output
    assert (canonical_output / "instance_7.ply").is_file()


def test_normaliser_refuses_output_nested_below_input(tmp_path: Path) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_overlap",
    )
    input_dir = tmp_path / "per_tree"
    input_dir.mkdir()
    source = input_dir / "tree.ply"
    write_xyz_ply(source, np.array([[1.0, 2.0, 3.0]]))

    with pytest.raises(ValueError, match="must not be equal, nested"):
        normaliser.normalise(
            input_path=input_dir,
            output_dir=input_dir / "normalised",
            requested_format="per_tree_directory",
            instance_field=None,
            ignored_labels=set(),
            overwrite=True,
        )

    assert source.is_file()
    assert not (input_dir / "normalised").exists()


def test_normaliser_failed_overwrite_preserves_previous_output(
    tmp_path: Path,
) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_failed_overwrite",
    )
    input_path = tmp_path / "prediction.las"
    output_dir = tmp_path / "normalised"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("old output\n", encoding="utf-8")
    write_labelled_prediction_las(input_path)

    with pytest.raises(ValueError, match="missing instance field"):
        normaliser.normalise(
            input_path=input_path,
            output_dir=output_dir,
            requested_format="labelled_point_cloud",
            instance_field="missing_field",
            ignored_labels={"0", "-1"},
            overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "old output\n"
    assert not list(tmp_path.glob(".normalised.partial.*"))


def test_normaliser_invalid_external_metadata_preserves_previous_output(
    tmp_path: Path,
) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_invalid_metadata",
    )
    input_path = tmp_path / "prediction.las"
    output_dir = tmp_path / "normalised"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("old output\n", encoding="utf-8")
    invalid_metadata = tmp_path / "metadata_directory"
    invalid_metadata.mkdir()
    write_labelled_prediction_las(input_path)

    with pytest.raises(IsADirectoryError, match="Metadata output is a directory"):
        normaliser.normalise(
            input_path=input_path,
            output_dir=output_dir,
            requested_format="labelled_point_cloud",
            instance_field="predicted_instance",
            ignored_labels={"0", "-1"},
            overwrite=True,
            metadata_path=invalid_metadata,
        )

    assert marker.read_text(encoding="utf-8") == "old output\n"
    assert not list(tmp_path.glob(".normalised.partial.*"))


def test_normaliser_successful_overwrite_swaps_complete_directory(
    tmp_path: Path,
) -> None:
    normaliser = load_script(
        "methods/segmentanytree/scripts/runtime/normalise_segmentanytree_predictions.py",
        "segmentanytree_normaliser_successful_overwrite",
    )
    input_path = tmp_path / "prediction.las"
    output_dir = tmp_path / "normalised"
    output_dir.mkdir()
    (output_dir / "obsolete.txt").write_text("old\n", encoding="utf-8")
    write_labelled_prediction_las(input_path)

    payload = normaliser.normalise(
        input_path=input_path,
        output_dir=output_dir,
        requested_format="labelled_point_cloud",
        instance_field="predicted_instance",
        ignored_labels={"0", "-1"},
        overwrite=True,
    )

    assert payload["predicted_instance_count"] == 2
    assert not (output_dir / "obsolete.txt").exists()
    assert (output_dir / "normalisation_metadata.json").is_file()
    assert not list(tmp_path.glob(".normalised.partial.*"))
    assert not list(tmp_path.glob(".normalised.backup.*"))


def test_export_patch_is_narrow_and_requires_expected_source() -> None:
    patcher = load_script(
        (
            "methods/segmentanytree/scripts/runtime/patches/"
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
    assert (
        "\n        values = df[column].round() if column == \"scan_angle\" "
        "else df[column]\n"
        "        las_file[column] = values.astype("
        "standard_columns_with_data_types[column])\n"
    ) in patched
    compile(patched, "pandas_to_las.py", "exec")
    with pytest.raises(ValueError, match="unsigned scan_angle"):
        patcher.patch_source(patched)


def test_serial_pool_preserves_map_order() -> None:
    serial = load_script(
        "methods/segmentanytree/scripts/runtime/patches/sitecustomize.py",
        "segmentanytree_serial_pool",
    )

    with serial.SerialPool(processes=1) as pool:
        assert pool.map(abs, [-2, -1, 0]) == [2, 1, 0]


def test_evaluator_uses_for_instance_tree_classes(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = load_script(
        "shared/evaluation/instance_iou_f1.py",
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
        "methods/segmentanytree/scripts/evaluation/summarise_for_instance_segmentanytree_benchmark.py",
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


def test_pointwise_evaluator_reports_paper_and_harmonized_metrics() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_evaluator",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([10, 10, 20, 20, 30, 30]),
        reference_instance=np.array([1, 1, 2, 2, 0, 0]),
        predicted_semantic=np.array([2, 2, 2, 2, 2, 2]),
        reference_semantic=np.array([2, 2, 2, 2, 1, 1]),
    )

    result = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
    )

    assert result["reference_instance_count"] == 2
    assert result["prediction_instance_count"] == 3
    assert result["paper_compatible"]["true_positives"] == 2
    assert result["paper_compatible"]["f1"] == pytest.approx(0.8)
    assert result["harmonized"]["true_positives"] == 2
    assert result["harmonized"]["f1"] == pytest.approx(0.8)
    assert result["mean_unweighted_coverage"] == pytest.approx(1.0)
    assert result["mean_weighted_coverage"] == pytest.approx(1.0)


def test_pointwise_evaluator_can_filter_small_predicted_instances() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_min_prediction_size",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([10, 10, 20, 20, 30]),
        reference_instance=np.array([1, 1, 2, 2, 0]),
        predicted_semantic=np.array([2, 2, 2, 2, 2]),
        reference_semantic=np.array([2, 2, 2, 2, 1]),
    )

    unfiltered = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
    )
    filtered = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
        min_predicted_instance_points=2,
    )

    assert unfiltered["prediction_instance_count"] == 3
    assert unfiltered["harmonized"]["false_positives"] == 1
    assert filtered["prediction_instance_count"] == 2
    assert filtered["min_predicted_instance_points"] == 2
    assert filtered["harmonized"]["false_positives"] == 0
    assert filtered["harmonized"]["f1"] == pytest.approx(1.0)


def test_pointwise_evaluator_can_filter_low_tree_fraction_predictions() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_tree_fraction_filter",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([10, 10, 20, 20, 20, 20, 20]),
        reference_instance=np.array([1, 1, 2, 2, 0, 0, 0]),
        predicted_semantic=np.array([2, 2, 2, 2, 2, 1, 1]),
        reference_semantic=np.array([2, 2, 2, 2, 1, 1, 1]),
    )

    unfiltered = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
    )
    filtered = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
        min_predicted_tree_fraction=0.75,
    )

    assert unfiltered["prediction_instance_count"] == 2
    assert filtered["prediction_instance_count"] == 1
    assert filtered["min_predicted_tree_fraction"] == 0.75
    assert filtered["harmonized"]["false_positives"] == 0


def test_sat_failure_audit_classifies_large_false_positive() -> None:
    audit = load_script(
        "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py",
        "segmentanytree_failure_audit_large_fp",
    )
    pointwise = audit.load_pointwise_module()
    labels = pointwise.PointLabels(
        predicted_instance=np.array([10, 10, 20, 20, 20, 20, 20, 20]),
        reference_instance=np.array([1, 1, 2, 3, 4, 5, 6, 7]),
        predicted_semantic=np.array([2, 2, 2, 2, 2, 2, 2, 2]),
        reference_semantic=np.array([2, 2, 2, 2, 2, 2, 2, 2]),
    )
    _, prediction_rows, _ = audit.analyse_labels(
        pointwise,
        labels,
        {"collection": "NIBIO", "plot_name": "synthetic"},
        "trained_test",
        "synthetic_run",
        0.5,
        0.25,
        0.1,
        5,
        {2},
        {2},
        {-1},
        {-1, 0},
    )

    row = next(row for row in prediction_rows if row["prediction_id"] == 20)
    assert row["point_count"] == 6
    assert row["best_iou"] < 0.25
    assert row["failure_mode"] == "large_extra_instance"


def test_sat_failure_audit_classifies_missed_reference() -> None:
    audit = load_script(
        "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py",
        "segmentanytree_failure_audit_missed_ref",
    )
    pointwise = audit.load_pointwise_module()
    labels = pointwise.PointLabels(
        predicted_instance=np.array([10, 10, -1, -1, -1]),
        reference_instance=np.array([1, 1, 2, 2, 2]),
        predicted_semantic=np.array([2, 2, 1, 1, 1]),
        reference_semantic=np.array([2, 2, 2, 2, 2]),
    )
    _, _, reference_rows = audit.analyse_labels(
        pointwise,
        labels,
        {"collection": "RMIT", "plot_name": "synthetic"},
        "trained_test",
        "synthetic_run",
        0.5,
        0.25,
        0.1,
        5,
        {2},
        {2},
        {-1},
        {-1, 0},
    )

    row = next(row for row in reference_rows if row["reference_id"] == 2)
    assert row["point_count"] == 3
    assert row["best_iou"] == 0.0
    assert row["failure_mode"] == "missed_tree"


def test_sat_failure_audit_outputs_are_public_safe() -> None:
    audit = load_script(
        "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py",
        "segmentanytree_failure_audit_public_safe",
    )
    fields = (
        audit.PLOT_FIELDS
        + audit.SITE_FIELDS
        + audit.PREDICTION_FIELDS
        + audit.REFERENCE_FIELDS
        + audit.DOMAIN_FIELDS
    )

    assert "x" not in fields
    assert "y" not in fields
    assert "z" not in fields
    assert all("coordinate" not in field for field in fields)


def test_sat_failure_audit_cli_help_parses() -> None:
    script = (
        ROOT
        / "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--run-id" in completed.stdout
    assert "--training-manifest" in completed.stdout


def test_sat_postprocess_sweep_cli_help_parses() -> None:
    script = (
        ROOT
        / "methods/segmentanytree/scripts/diagnostics/"
        "sweep_sat_validation_postprocessing.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--min-predicted-instance-points" in completed.stdout
    assert "--min-predicted-tree-fraction" in completed.stdout


def test_sat_postprocess_sweep_uses_validation_rows_only(tmp_path: Path) -> None:
    sweep = load_script(
        (
            "methods/segmentanytree/scripts/diagnostics/"
            "sweep_sat_validation_postprocessing.py"
        ),
        "segmentanytree_postprocess_sweep_run",
    )

    def parse_number_set(text: str) -> set[float]:
        return {float(value) for value in text.split(",") if value}

    def load_metric_rows(split_root: Path, run_id: str):
        return [
            ("trained_validation", tmp_path / "val.json", {"score": 0.6}),
            ("trained_test", tmp_path / "test.json", {"score": 0.9}),
        ]

    def evaluate_pointwise(labels, **kwargs):
        return {
            "prediction_instance_count": 3,
            "reference_instance_count": 2,
            "harmonized": {
                "f1": labels["score"],
                "precision": labels["score"],
                "recall": labels["score"],
                "true_positives": 1,
                "false_positives": 2,
                "false_negatives": 1,
            },
        }

    fake_audit = SimpleNamespace(
        load_pointwise_module=lambda: SimpleNamespace(evaluate_pointwise=evaluate_pointwise),
        load_metric_rows=load_metric_rows,
        parse_number_set=parse_number_set,
        load_labels_from_payload=lambda *args: args[1],
    )
    sweep.load_script = lambda *args: fake_audit

    outputs = sweep.run_sweep(
        SimpleNamespace(
            run_id="synthetic_run",
            split_root=str(tmp_path),
            split="trained_validation",
            output_dir=str(tmp_path / "out"),
            min_predicted_instance_points="0",
            min_predicted_tree_fraction="0",
            semantic_offset=1.0,
            reference_tree_classes="2",
            prediction_tree_classes="2",
            ignored_reference_labels="-1",
            ignored_prediction_labels="-1,0",
            reference_background_instance_labels="1",
            iou_threshold=0.5,
        )
    )

    best = list(csv.DictReader(outputs["best"].open(encoding="utf-8")))
    assert len(best) == 1
    assert best[0]["n_plots"] == "1"
    assert best[0]["mean_f1"] == "0.6"


def test_sat_failure_audit_writes_expected_csvs(tmp_path: Path) -> None:
    audit = load_script(
        "methods/segmentanytree/scripts/diagnostics/audit_sat_failure_modes.py",
        "segmentanytree_failure_audit_outputs",
    )
    pointwise = audit.load_pointwise_module()

    def synthetic_labels(*args, **kwargs):
        return pointwise.PointLabels(
            predicted_instance=np.array([10, 10, -1, -1, -1]),
            reference_instance=np.array([1, 1, 2, 2, 2]),
            predicted_semantic=np.array([2, 2, 1, 1, 1]),
            reference_semantic=np.array([2, 2, 2, 2, 2]),
        )

    audit.load_labels_from_payload = synthetic_labels

    run_id = "synthetic_run"
    split_root = tmp_path / "metadata"
    metric_root = split_root / "trained_test" / run_id / "RMIT"
    metric_root.mkdir(parents=True)
    (metric_root / "test.json").write_text(
        json.dumps(
            {
                "evaluator": "pointwise_instance_metrics",
                "collection": "RMIT",
                "plot_name": "test",
                "relative_path": "RMIT/test.las",
                "inputs": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}\n', encoding="utf-8")
    output_dir = tmp_path / "audit"

    outputs = audit.run_audit(
        SimpleNamespace(
            run_id=run_id,
            split_root=str(split_root),
            training_manifest=str(manifest),
            output_dir=str(output_dir),
            semantic_offset=1.0,
            reference_tree_classes="2",
            prediction_tree_classes="2",
            ignored_reference_labels="-1",
            ignored_prediction_labels="-1,0",
            reference_background_instance_labels="1",
            iou_threshold=0.5,
            near_miss_iou=0.25,
            fragmentation_iou=0.1,
            large_instance_points=5,
        )
    )

    assert set(outputs) == {"plot", "site", "prediction", "reference", "domain"}
    assert all(path.is_file() for path in outputs.values())
    for path in outputs.values():
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "x" not in header
        assert "y" not in header
        assert "z" not in header


def test_pointwise_evaluator_recovers_degenerate_reference_semantics() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_reference_recovery",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([-1, -1, 10, 10, 20, 20]),
        reference_instance=np.array([1, -1, 1102, 1102, 1103, 1103]),
        predicted_semantic=np.array([1, 1, 2, 2, 2, 2]),
        reference_semantic=np.array([1, 1, 1, 1, 1, 1]),
    )

    assert evaluator.reference_semantic_requires_instance_fallback(
        labels,
        background_labels={1},
        ignored_labels={-1},
        tree_classes={2},
    )
    recovered = evaluator.derive_reference_semantic_from_instance(
        labels,
        background_labels={1},
        tree_classes={2},
        ignored_labels={-1},
    )
    result = evaluator.evaluate_pointwise(
        recovered,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={-1},
        ignored_prediction_labels={-1},
        iou_threshold=0.5,
    )

    assert np.isnan(recovered.reference_semantic[0])
    assert np.isnan(recovered.reference_semantic[1])
    assert recovered.reference_semantic[2:].tolist() == [2, 2, 2, 2]
    assert result["reference_instance_count"] == 2
    assert result["harmonized"]["true_positives"] == 2
    assert result["harmonized"]["f1"] == 1.0

    valid_semantics = evaluator.PointLabels(
        predicted_instance=labels.predicted_instance,
        reference_instance=labels.reference_instance,
        predicted_semantic=labels.predicted_semantic,
        reference_semantic=np.array([1, 1, 2, 2, 2, 2]),
    )
    assert not evaluator.reference_semantic_requires_instance_fallback(
        valid_semantics,
        background_labels={1},
        ignored_labels={-1},
        tree_classes={2},
    )


def test_pointwise_evaluator_exposes_matching_policy_difference() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_matching",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([10, 10, 20, 20]),
        reference_instance=np.array([1, 1, 1, 1]),
        predicted_semantic=np.array([2, 2, 2, 2]),
        reference_semantic=np.array([2, 2, 2, 2]),
    )

    result = evaluator.evaluate_pointwise(
        labels,
        reference_tree_classes={2},
        prediction_tree_classes={2},
        ignored_reference_labels={0, -1},
        ignored_prediction_labels={0, -1},
        iou_threshold=0.5,
    )

    assert result["paper_compatible"]["true_positives"] == 2
    assert result["harmonized"]["true_positives"] == 1
    assert result["harmonized"]["false_positives"] == 1
    assert result["harmonized"]["false_negatives"] == 0


def test_pointwise_evaluator_rejects_unaligned_arrays() -> None:
    evaluator = load_script(
        "methods/segmentanytree/scripts/evaluation/pointwise_instance_metrics.py",
        "segmentanytree_pointwise_alignment",
    )
    labels = evaluator.PointLabels(
        predicted_instance=np.array([1, 1]),
        reference_instance=np.array([1]),
        predicted_semantic=np.array([2, 2]),
        reference_semantic=np.array([2, 2]),
    )

    with pytest.raises(ValueError, match="not aligned"):
        evaluator.evaluate_pointwise(
            labels,
            reference_tree_classes={2},
            prediction_tree_classes={2},
            ignored_reference_labels={0, -1},
            ignored_prediction_labels={0, -1},
            iou_threshold=0.5,
        )


def test_run_metadata_hashes_checkpoint(tmp_path: Path) -> None:
    recorder = load_script(
        "methods/segmentanytree/scripts/runtime/record_segmentanytree_run.py",
        "segmentanytree_run_recorder",
    )
    checkpoint = tmp_path / "PointGroup-PAPER.pt"
    checkpoint.write_bytes(b"fixed checkpoint fixture")

    assert recorder.sha256(checkpoint) == (
        "cae5c917d62fb2c0f9f2a62ffce50fdab7f8fdd54f1aaff4da5353dde7fade14"
    )
    assert recorder.sha256(tmp_path / "missing.pt") is None


def test_tracker_patch_enables_aligned_instance_output() -> None:
    patcher = load_script(
        (
            "methods/segmentanytree/scripts/runtime/patches/"
            "prepare_pointgroup_tracker_patch.py"
        ),
        "segmentanytree_tracker_patcher",
    )
    source = (
        "class Tracker:\n"
        "    def finalise(self):\n"
        "        if True:\n"
        "            if True:\n"
        "                for i in range(1):\n"
        "                    has_prediction = test_area_i.ins_pre != -1\n"
        '                    print("writing evaluation txt")\n'
    )

    patched = patcher.patch_source(source)

    assert patched.count("Instance_results_forEval_{}.ply") == 2
    assert 'SEGMENTANYTREE_ALIGNED_OUTPUT_DIR", "."' in patched
    assert "if not bool(has_prediction.any())" in patched
    assert "_sat_full_ins_pred" in patched
    assert "test_area_i.instance_labels" in patched


def test_training_split_is_seeded_and_excludes_test_data(tmp_path: Path) -> None:
    preparer = load_script(
        "methods/segmentanytree/scripts/data/prepare_segmentanytree_for_instance_training.py",
        "segmentanytree_training_preparer_split",
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    rows = []
    for index in range(8):
        path = dataset_root / "NIBIO" / f"plot_{index}_annotated.las"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_annotated_las(path)
        rows.append(f"NIBIO/{path.name},NIBIO,dev")
    for index in range(2):
        path = dataset_root / "CULS" / f"test_{index}.las"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_annotated_las(path)
        rows.append(f"CULS/{path.name},CULS,test")
    metadata = dataset_root / "data_split_metadata.csv"
    metadata.write_text(
        "relative_path,collection,split\n"
        + "\n".join(rows)
        + "\nNIBIO2/missing_plot.las,NIBIO2,dev\n",
        encoding="utf-8",
    )

    records = preparer.read_split_rows(dataset_root, metadata)
    first = preparer.assign_training_roles(records, seed=42, validation_fraction=0.25)
    second = preparer.assign_training_roles(records, seed=42, validation_fraction=0.25)

    assert [row["training_role"] for row in first] == [
        row["training_role"] for row in second
    ]
    assert sum(row["training_role"] == "train" for row in first) == 6
    assert sum(row["training_role"] == "val" for row in first) == 2
    assert sum(row["training_role"] == "held_out_test" for row in first) == 2


def test_training_split_rejects_unassigned_local_las(tmp_path: Path) -> None:
    preparer = load_script(
        "methods/segmentanytree/scripts/data/prepare_segmentanytree_for_instance_training.py",
        "segmentanytree_training_preparer_unassigned",
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    for name in ("listed.las", "unlisted.las"):
        path = dataset_root / "CULS" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_annotated_las(path)
    metadata = dataset_root / "data_split_metadata.csv"
    metadata.write_text(
        "relative_path,collection,split\n"
        "CULS/listed.las,CULS,dev\n"
        "NIBIO2/not_downloaded.las,NIBIO2,dev\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no split metadata"):
        preparer.read_split_rows(dataset_root, metadata)


def test_training_preparation_writes_expected_ply_without_test(
    tmp_path: Path,
) -> None:
    preparer = load_script(
        "methods/segmentanytree/scripts/data/prepare_segmentanytree_for_instance_training.py",
        "segmentanytree_training_preparer_convert",
    )
    dataset_root = tmp_path / "FORinstance_dataset"
    metadata_rows = []
    for collection, name, split in (
        ("CULS", "plot_1.las", "dev"),
        ("CULS", "plot_2.las", "dev"),
        ("NIBIO", "plot_3.las", "dev"),
        ("NIBIO", "plot_4.las", "dev"),
        ("SCION", "plot_5.las", "test"),
    ):
        path = dataset_root / collection / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_annotated_las(path)
        metadata_rows.append(f"{collection}/{name},{collection},{split}")
    metadata = dataset_root / "data_split_metadata.csv"
    metadata.write_text(
        "relative_path,collection,split\n"
        + "\n".join(metadata_rows)
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "prepared"
    manifest_path = tmp_path / "manifest.json"

    manifest = preparer.prepare(
        dataset_root=dataset_root,
        metadata_path=metadata,
        output_root=output_root,
        manifest_path=manifest_path,
        profile="pilot",
        seed=42,
        validation_fraction=0.25,
        pilot_train_count=2,
        pilot_val_count=1,
        overwrite=False,
    )

    assert manifest["selected_role_counts"] == {
        "train": 2,
        "val": 1,
        "held_out_test": 0,
    }
    assert manifest["dataset_role_counts"] == {
        "train": 3,
        "val": 1,
        "held_out_test": 1,
    }
    assert manifest["test_data_converted"] is False
    converted = list((output_root / "treeinsfused/raw").rglob("*.ply"))
    assert len(converted) == 3
    assert not [path for path in converted if "_test" in path.name]
    header, vertices = read_ply_vertices(converted[0])
    assert set(header.columns) == {
        "x",
        "y",
        "z",
        "intensity",
        "semantic_seg",
        "treeID",
    }
    assert vertices["semantic_seg"].tolist() == [
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]
    assert vertices["treeID"].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]


def test_training_config_and_slurm_gates_are_explicit() -> None:
    config = yaml.safe_load(
        (
            ROOT / "methods/segmentanytree/configs/for_instance_training.yml"
        ).read_text(encoding="utf-8")
    )
    assert config["method"]["primary_training_mode"] == "fine_tuned_on_dev"
    assert config["method"]["comparison_baseline_mode"] == "published_pretrained"
    assert config["method"]["paper_scenario"] == "scenario_1_uls_only"
    assert config["dataset"]["internal_validation"]["seed"] == 42
    assert config["dataset"]["internal_validation"]["expected_training_plots"] == 16
    assert config["dataset"]["internal_validation"]["expected_validation_plots"] == 5
    assert config["preparation"]["convert_test_for_training"] is False
    assert config["evaluation"]["use_test_for_model_selection"] is False
    assert config["project"]["status"] == "completed"
    primary = config["training"]["replacement_fine_tuned_protocol"]
    baseline = config["training"]["published_pretrained_protocol"]
    assert primary["status"] == "completed_primary_result"
    assert primary["held_out_test"]["mean_plot_f1"] == pytest.approx(
        0.5446789009405125
    )
    assert baseline["status"] == "completed_target_baseline"
    assert baseline["mean_plot_f1"] == pytest.approx(0.45340897930934665)
    assert config["training"]["held_out_test_status"] == (
        "completed_no_repeat_for_setting_selection"
    )
    assert config["training"]["result_retention"]["public_site_results"] == (
        "methods/segmentanytree/examples/"
        "sat_completed_target_site_results_20260711.csv"
    )

    training_task = (
        ROOT / "methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_task.sh"
    ).read_text(encoding="utf-8")
    final_test = (
        ROOT
        / "methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch"
    ).read_text(encoding="utf-8")
    validation_eval = (
        ROOT
        / "methods/segmentanytree/slurm/evaluation/"
        "evaluate_segmentanytree_for_instance_trained_validation.sbatch"
    ).read_text(encoding="utf-8")
    test_eval = (
        ROOT
        / "methods/segmentanytree/slurm/evaluation/"
        "evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch"
    ).read_text(encoding="utf-8")
    assert "SEGMENTANYTREE_EXECUTE" in training_task
    assert "wandb.log=false" in training_task
    assert "tensorboard.log=false" in training_task
    assert '"epochs=$HYDRA_STOP_EPOCH"' in training_task
    assert '"batch_size=$BATCH_SIZE"' in training_task
    assert "training.epochs=" not in training_task
    assert "prepare_dev_only_trainer_patch.py" in training_task
    assert "trainer.py:ro" in training_task
    assert "data.dataroot=/sat_data" in training_task
    assert "SEGMENTANYTREE_RESUME_CHECKPOINT" in training_task
    assert "SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256" in training_task
    assert "SEGMENTANYTREE_PRETRAINED_CHECKPOINT" in training_task
    assert "SEGMENTANYTREE_PRETRAINED_CHECKPOINT_SHA256" in training_task
    assert 'TRAINING_MODE="fine_tuned_on_dev"' in training_task
    assert "SEGMENTANYTREE_REQUIRE_PRETRAINED_LOAD=1" in training_task
    assert "SEGMENTANYTREE_PRETRAINED_PATH=/sat_pretrained/PointGroup-PAPER.pt" in training_task
    assert 'TRAIN_COMMAND+=("optim.base_lr=$BASE_LR")' in training_task
    assert 'TRAIN_COMMAND+=("+training.optim.base_lr=$BASE_LR")' not in training_task
    assert 'TRAIN_COMMAND+=("training.optim.base_lr=$BASE_LR")' not in training_task
    assert "checkpoint_dir=/sat_resume" in training_task
    assert "weight_name=latest" in training_task
    assert "SEGMENTANYTREE_STALL_TIMEOUT_SECONDS" in training_task
    assert "run_stall_watchdog" in training_task
    assert "setsid /usr/bin/time" in training_task
    assert "prepare_spawn_meanshift_patch.py" in training_task
    assert "meanshift_cluster.py:ro" in training_task
    assert "SEGMENTANYTREE_MEANSHIFT_JOBS" in training_task
    assert "SEGMENTANYTREE_OMP_NUM_THREADS" in training_task
    assert "SEGMENTANYTREE_FINAL_TEST_CONFIRMED" in final_test
    assert "SEGMENTANYTREE_EVALUATION_RUN_ID" in validation_eval
    assert "SEGMENTANYTREE_EVALUATION_RUN_ID" in test_eval
    evaluation_task = (
        ROOT
        / "methods/segmentanytree/slurm/evaluation/"
        "evaluate_segmentanytree_pointwise_task.sh"
    ).read_text(encoding="utf-8")
    assert "--reference-background-instance-labels 1" in evaluation_task
    assert "--ignored-prediction-labels=-1" in evaluation_task
    assert "SEGMENTANYTREE_MIN_PREDICTED_INSTANCE_POINTS" in evaluation_task
    assert "--min-predicted-instance-points" in evaluation_task
    assert "SEGMENTANYTREE_MIN_PREDICTED_TREE_FRACTION" in evaluation_task
    assert "--min-predicted-tree-fraction" in evaluation_task


def test_dev_only_trainer_patch_preserves_validation() -> None:
    patcher = load_script(
        (
            "methods/segmentanytree/scripts/runtime/patches/"
            "prepare_dev_only_trainer_patch.py"
        ),
        "segmentanytree_dev_only_trainer_patcher",
    )
    block = patcher.TEST_EVALUATION_BLOCK
    source = (
        patcher.IMPORT_BLOCK
        + "def train(self):\n"
        + patcher.INCOMPLETE_RESUME_BLOCK
        + patcher.PRETRAINED_LOAD_ANCHOR
        + block
        + "    if self._dataset.has_val_loader:\n"
        + '        self._test_epoch(epoch, "val")\n'
        + block
    )

    patched = patcher.patch_source(source)

    assert patched.count("if False:") == 2
    assert 'self._test_epoch(epoch, "val")' in patched
    assert 'self._test_epoch(epoch, "test")' in patched
    assert "torch.cuda.amp.GradScaler" in patched
    assert patcher.INCOMPLETE_RESUME_BLOCK not in patched
    assert "faulthandler.dump_traceback_later" in patched
    assert "SEGMENTANYTREE_REQUIRE_PRETRAINED_LOAD" in patched
    assert "compatible_fraction" in patched
    assert "SEGMENTANYTREE_PRETRAINED_VALIDATION_OUTPUT" in patched


def test_training_metadata_records_finetune_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = load_script(
        (
            "methods/segmentanytree/scripts/runtime/"
            "record_segmentanytree_training_run.py"
        ),
        "segmentanytree_finetune_recorder",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_role_counts": {"dev": 21, "test": 11},
                "selected_role_counts": {
                    "train": 16,
                    "val": 5,
                    "held_out_test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    pretrained = tmp_path / "pretrained" / "PointGroup-PAPER.pt"
    pretrained.parent.mkdir()
    pretrained.write_bytes(b"released checkpoint")
    checkpoint = tmp_path / "output" / "run" / "PointGroup-PAPER.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fine-tuned checkpoint")
    validation = tmp_path / "output" / "pretrained_load_validation.json"
    validation.write_text(
        json.dumps({"compatible_fraction": 0.999}), encoding="utf-8"
    )
    command = tmp_path / "output" / "training_command.txt"
    command.write_text("apptainer exec training\n", encoding="utf-8")
    image = tmp_path / "segment-any-tree.sif"
    image.write_bytes(b"image")
    external_repo = tmp_path / "SegmentAnyTree"
    external_repo.mkdir()
    output = tmp_path / "metadata.json"
    pretrained_sha256 = recorder.sha256(pretrained)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_segmentanytree_training_run.py",
            "--output",
            str(output),
            "--run-id",
            "finetune-test",
            "--run-type",
            "full_training",
            "--training-mode",
            "fine_tuned_on_dev",
            "--profile",
            "full",
            "--split-manifest",
            str(manifest),
            "--training-data-root",
            str(tmp_path / "training-data"),
            "--training-output-root",
            str(tmp_path / "output"),
            "--external-repo",
            str(external_repo),
            "--image",
            str(image),
            "--python-userbase",
            str(tmp_path / "python-userbase"),
            "--command-file",
            str(command),
            "--checkpoint",
            str(checkpoint),
            "--pretrained-checkpoint",
            str(pretrained),
            "--pretrained-checkpoint-sha256",
            pretrained_sha256,
            "--pretrained-weight-name",
            "latest",
            "--pretrained-validation-json",
            str(validation),
            "--base-lr",
            "0.0001",
            "--requested-epochs",
            "35",
            "--hydra-stop-epoch",
            "36",
            "--batch-size",
            "8",
            "--status",
            "completed",
            "--return-code",
            "0",
        ],
    )

    assert recorder.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_mode"] == "fine_tuned_on_dev"
    assert payload["pretrained_checkpoint"] == str(pretrained)
    assert payload["pretrained_checkpoint_sha256"] == pretrained_sha256
    assert payload["pretrained_weight_name"] == "latest"
    assert payload["pretrained_load_validation"]["compatible_fraction"] == 0.999
    assert payload["base_lr"] == 0.0001


def test_meanshift_patch_uses_persistent_spawn_pool() -> None:
    patcher = load_script(
        (
            "methods/segmentanytree/scripts/runtime/patches/"
            "prepare_spawn_meanshift_patch.py"
        ),
        "segmentanytree_meanshift_patcher",
    )
    source = (
        patcher.IMPORT_ANCHOR
        + "ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)\n"
        + "with multiprocessing.Pool(processes=processes) as pool:\n"
        + "    first = pool.map(function, values)\n"
        + "with multiprocessing.Pool(processes=processes) as pool:\n"
        + "    second = pool.map(function, values)\n"
    )

    patched = patcher.patch_source(source)

    assert patched.count("with _PersistentSpawnPool(processes) as pool:") == 2
    assert "multiprocessing.Pool(processes=processes)" not in patched
    assert 'multiprocessing.get_context("spawn")' in patched
    assert "maxtasksperchild=1000" in patched
    assert "SEGMENTANYTREE_MEANSHIFT_JOBS" in patched


def test_training_metadata_records_resume_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    recorder = load_script(
        (
            "methods/segmentanytree/scripts/runtime/"
            "record_segmentanytree_training_run.py"
        ),
        "segmentanytree_training_recorder",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_role_counts": {"dev": 21, "test": 11},
                "selected_role_counts": {
                    "train": 16,
                    "val": 5,
                    "held_out_test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    source_checkpoint = tmp_path / "source" / "PointGroup-PAPER.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"reviewed checkpoint")
    output_checkpoint = tmp_path / "output" / "run" / "PointGroup-PAPER.pt"
    output_checkpoint.parent.mkdir(parents=True)
    output_checkpoint.write_bytes(b"resumed checkpoint")
    command_file = tmp_path / "output" / "training_command.txt"
    command_file.write_text("apptainer exec training\n", encoding="utf-8")
    image = tmp_path / "segment-any-tree.sif"
    image.write_bytes(b"image")
    external_repo = tmp_path / "SegmentAnyTree"
    external_repo.mkdir()
    stall_marker = tmp_path / "output" / "stall_watchdog.txt"
    stall_marker.write_text(
        "No training log progress for 1200 seconds.\n",
        encoding="utf-8",
    )
    output = tmp_path / "metadata.json"
    source_sha256 = recorder.sha256(source_checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_segmentanytree_training_run.py",
            "--output",
            str(output),
            "--run-id",
            "resume-test",
            "--run-type",
            "full_training",
            "--training-mode",
            "resumed_from_dev_checkpoint",
            "--profile",
            "full",
            "--split-manifest",
            str(manifest),
            "--training-data-root",
            str(tmp_path / "training-data"),
            "--training-output-root",
            str(tmp_path / "output"),
            "--external-repo",
            str(external_repo),
            "--image",
            str(image),
            "--python-userbase",
            str(tmp_path / "python-userbase"),
            "--command-file",
            str(command_file),
            "--checkpoint",
            str(output_checkpoint),
            "--resume-checkpoint",
            str(source_checkpoint),
            "--resume-checkpoint-sha256",
            source_sha256,
            "--resume-start-epoch",
            "31",
            "--requested-epochs",
            "150",
            "--hydra-stop-epoch",
            "151",
            "--batch-size",
            "4",
            "--status",
            "failed",
            "--return-code",
            "143",
            "--stall-timeout-seconds",
            "1200",
            "--stall-marker",
            str(stall_marker),
        ],
    )

    assert recorder.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["training_mode"] == "resumed_from_dev_checkpoint"
    assert payload["resume_checkpoint"] == str(source_checkpoint)
    assert payload["resume_checkpoint_sha256"] == source_sha256
    assert payload["resume_start_epoch"] == 31
    assert payload["stall_timeout_seconds"] == 1200
    assert payload["stall_watchdog"].startswith("No training log progress")


def test_segmentanytree_training_shell_scripts_parse() -> None:
    scripts = [
        "methods/segmentanytree/slurm/training/prepare_for_instance_segmentanytree_splits.sbatch",
        "methods/segmentanytree/slurm/training/prepare_segmentanytree_released_checkpoint.sbatch",
        "methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_task.sh",
        "methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_pilot.sbatch",
        "methods/segmentanytree/slurm/training/train_segmentanytree_for_instance_full.sbatch",
        "methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_pointwise_task.sh",
        "methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_trained_validation.sbatch",
        "methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_trained_validation.sbatch",
        "methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_test_from_checkpoint.sbatch",
        "methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_for_instance_test_from_checkpoint.sbatch",
        "methods/segmentanytree/slurm/evaluation/audit_segmentanytree_failure_modes.sbatch",
        "methods/segmentanytree/slurm/evaluation/sweep_segmentanytree_validation_postprocessing.sbatch",
        "methods/segmentanytree/slurm/evaluation/validate_segmentanytree_variant.sbatch",
        "methods/segmentanytree/slurm/evaluation/summarise_segmentanytree_three_variations.sbatch",
        "methods/segmentanytree/slurm/evaluation/summarise_segmentanytree_pretrained_finetune.sbatch",
        "methods/segmentanytree/slurm/inference/run_published_pretrained_dev_smoke.sbatch",
        "methods/segmentanytree/slurm/evaluation/evaluate_published_pretrained_dev_smoke.sbatch",
        "methods/segmentanytree/slurm/evaluation/validate_published_pretrained_dev_smoke.sbatch",
        "methods/segmentanytree/slurm/submit_full_training_chain.sh",
        "methods/segmentanytree/slurm/submit_published_pretrained_dev_smoke.sh",
        "methods/segmentanytree/slurm/submit_published_pretrained_test.sh",
        "methods/segmentanytree/slurm/monitor_published_pretrained_test.sh",
        "methods/segmentanytree/slurm/submit_finetuned_dev_validation.sh",
        "methods/segmentanytree/slurm/monitor_finetuned_dev_validation.sh",
        "methods/segmentanytree/slurm/submit_finetuned_test.sh",
        "methods/segmentanytree/slurm/monitor_finetuned_test.sh",
        "methods/segmentanytree/slurm/submit_three_variation_overnight.sh",
        "methods/segmentanytree/slurm/recover_three_variation_pretrained.sh",
        "methods/segmentanytree/slurm/monitor_three_variation_overnight.sh",
        "methods/segmentanytree/slurm/submit_pretrained_finetune_comparison.sh",
        "methods/segmentanytree/slurm/recover_pretrained_finetune_pretrained.sh",
        "methods/segmentanytree/slurm/monitor_pretrained_finetune_comparison.sh",
    ]
    for relative_path in scripts:
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / relative_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            relative_path,
            completed.stderr,
        )

    full_training = (
        ROOT
        / "methods/segmentanytree/slurm/training/"
        "train_segmentanytree_for_instance_full.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --time=23:59:00" in full_training


def test_published_pretrained_development_smoke_is_isolated() -> None:
    submitter = (
        ROOT
        / "methods/segmentanytree/slurm/submit_published_pretrained_dev_smoke.sh"
    ).read_text(encoding="utf-8")
    inference = (
        ROOT
        / "methods/segmentanytree/slurm/inference/"
        "run_published_pretrained_dev_smoke.sbatch"
    ).read_text(encoding="utf-8")
    evaluation = (
        ROOT
        / "methods/segmentanytree/slurm/evaluation/"
        "evaluate_published_pretrained_dev_smoke.sbatch"
    ).read_text(encoding="utf-8")
    preparation = (
        ROOT
        / "methods/segmentanytree/slurm/training/"
        "prepare_segmentanytree_released_checkpoint.sbatch"
    ).read_text(encoding="utf-8")

    assert "SEGMENTANYTREE_PRETRAINED_DEV_SMOKE_CONFIRMED" in submitter
    assert "segmentanytree_for-instance_published_pretrained_${STAMP}" in submitter
    assert "CULS/plot_1_annotated.las" in submitter
    assert "--split dev" in submitter
    assert "afterok:$checkpoint_job" in submitter
    assert "afterok:$inference_job" in submitter
    assert "afterok:$evaluation_job" in submitter
    assert "No held-out test or fine-tuning job was submitted" in submitter
    assert "fine_tuned_on_dev" not in submitter
    assert "--split test" not in submitter
    assert "test_from_checkpoint" not in submitter
    assert "SEGMENTANYTREE_REQUIRED_SPLIT=dev" in inference
    assert "SEGMENTANYTREE_RUN_TYPE=published_pretrained" in inference
    assert "--split dev" in evaluation
    assert "SOURCE_DIR=" in preparation
    assert 'test -s "$partial/.hydra/overrides.yaml"' in preparation
    assert 'cp -a "$SOURCE_DIR/." /sat_export/' in preparation


def test_published_pretrained_development_smoke_gate(tmp_path: Path) -> None:
    validator = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "validate_published_pretrained_dev_smoke.py"
        ),
        "segmentanytree_pretrained_dev_smoke_validator",
    )
    bundle = tmp_path / "released_model_bundle"
    (bundle / ".hydra").mkdir(parents=True)
    checkpoint = bundle / "PointGroup-PAPER.pt"
    checkpoint.write_bytes(b"released weights")
    (bundle / ".hydra" / "overrides.yaml").write_text(
        "model_name: PointGroup-PAPER\n", encoding="utf-8"
    )
    checkpoint_sha256 = validator.sha256(checkpoint)
    input_file = tmp_path / "plot_1_annotated.las"
    input_file.write_bytes(b"synthetic input")
    aligned_instance = tmp_path / "Instance_results_forEval_0.ply"
    aligned_semantic = tmp_path / "semantic_segmentation_0.ply"
    aligned_instance.write_bytes(b"aligned instance")
    aligned_semantic.write_bytes(b"aligned semantic")
    run_metadata = tmp_path / "run.json"
    run_metadata.write_text(
        json.dumps(
            {
                "status": "completed",
                "return_code": 0,
                "run_type": "published_pretrained",
                "split": "dev",
                "relative_path": "CULS/plot_1_annotated.las",
                "external_commit": validator.EXPECTED_EXTERNAL_COMMIT,
                "checkpoint_sha256": checkpoint_sha256,
                "input_file": str(input_file),
                "input_sha256": validator.sha256(input_file),
                "aligned_instance_evaluation_exists": True,
                "aligned_semantic_evaluation_exists": True,
                "aligned_instance_evaluation": str(aligned_instance),
                "aligned_instance_evaluation_sha256": validator.sha256(
                    aligned_instance
                ),
                "aligned_semantic_evaluation": str(aligned_semantic),
                "aligned_semantic_evaluation_sha256": validator.sha256(
                    aligned_semantic
                ),
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.json"
    metrics_payload = {
        "evaluator": "pointwise_instance_metrics",
        "input_mode": "internal_aligned_ply",
        "split": "dev",
        "relative_path": "CULS/plot_1_annotated.las",
        "point_count": 100,
        "reference_instance_count": 2,
        "prediction_instance_count": 3,
    }
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")

    payload = validator.validate_smoke(
        run_metadata,
        metrics,
        bundle,
        "CULS/plot_1_annotated.las",
        checkpoint_sha256,
        validator.EXPECTED_EXTERNAL_COMMIT,
    )
    assert payload["status"] == "smoke-tested"
    assert payload["training_mode"] == "published_pretrained"
    assert payload["held_out_test_accessed"] is False
    assert payload["accuracy_benchmark_completed"] is False
    assert payload["prediction_instance_count"] == 3
    assert payload["released_weight_test_overlap_status"] == (
        "unresolved_do_not_claim_leakage_free"
    )

    metrics_payload["prediction_instance_count"] = 0
    metrics.write_text(json.dumps(metrics_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="zero predicted instances"):
        validator.validate_smoke(
            run_metadata,
            metrics,
            bundle,
            "CULS/plot_1_annotated.las",
            checkpoint_sha256,
            validator.EXPECTED_EXTERNAL_COMMIT,
        )


def test_published_pretrained_test_freeze_and_submission_are_guarded(
    tmp_path: Path,
) -> None:
    freezer = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "prepare_published_pretrained_test_freeze.py"
        ),
        "segmentanytree_pretrained_test_freezer",
    )
    bundle = tmp_path / "released_model_bundle"
    (bundle / ".hydra").mkdir(parents=True)
    checkpoint = bundle / "PointGroup-PAPER.pt"
    checkpoint.write_bytes(b"released weights")
    overrides = bundle / ".hydra" / "overrides.yaml"
    overrides.write_text(
        "\n".join(
            [
                "- task=panoptic",
                "- data=panoptic/treeins_rad8",
                "- models=panoptic/area4_ablation_3heads_5",
                "- model_name=PointGroup-PAPER",
                "- training=treeins",
                "- job_name=mls_data_run",
                "- batch_size=6",
                "- epochs=100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_sha256 = freezer.sha256(checkpoint)
    freezer.EXPECTED_CHECKPOINT_SHA256 = checkpoint_sha256
    smoke_path = tmp_path / (
        "segmentanytree_for-instance-published-pretrained-placeholder.json"
    )
    smoke_path.write_text(
        json.dumps(
            {
                "status": "smoke-tested",
                "training_mode": "published_pretrained",
                "dataset_split": "dev",
                "held_out_test_accessed": False,
                "next_gate": "manual_review_before_held_out_test_submission",
                "prediction_instance_count": 20,
                "reference_instance_count": 6,
                "external_commit": freezer.EXPECTED_EXTERNAL_COMMIT,
                "checkpoint_sha256": checkpoint_sha256,
                "released_weight_test_overlap_status": (
                    "unresolved_do_not_claim_leakage_free"
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Explicit acceptance"):
        freezer.freeze_test_evaluation(
            smoke_path, bundle, "run", False
        )
    payload = freezer.freeze_test_evaluation(smoke_path, bundle, "run", True)
    assert payload["status"] == "frozen_for_one_time_held_out_evaluation"
    assert payload["released_checkpoint_job_name"] == "mls_data_run"
    assert payload["expected_test_plots"] == 11
    assert payload["weight_updates"] is False
    assert payload["repeat_test_for_setting_selection_permitted"] is False

    submitter = (
        ROOT / "methods/segmentanytree/slurm/submit_published_pretrained_test.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        ROOT / "methods/segmentanytree/slurm/monitor_published_pretrained_test.sh"
    ).read_text(encoding="utf-8")
    assert "SEGMENTANYTREE_PUBLISHED_PRETRAINED_TEST_CONFIRMED" in submitter
    assert "SEGMENTANYTREE_ACCEPT_UNRESOLVED_TRAINING_MANIFEST" in submitter
    assert "SEGMENTANYTREE_DEV_SMOKE_EVIDENCE" in submitter
    assert "SEGMENTANYTREE_CHECKPOINT_DIR=$CHECKPOINT_BUNDLE" in submitter
    assert "SEGMENTANYTREE_RUN_TYPE=published_pretrained" in submitter
    assert "--array=0-10%2" in submitter
    assert "--array=0-10%4" in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11" in submitter
    assert "train_segmentanytree" not in submitter
    assert "fine_tuned_on_dev" not in submitter
    assert "time remaining" not in monitor.lower()
    assert "%.9L" in monitor
    assert "%.19e" in monitor
    assert "tail " not in monitor


def test_finetuned_dev_freeze_and_submission_are_development_only(
    tmp_path: Path,
) -> None:
    freezer = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "prepare_finetuned_dev_training_freeze.py"
        ),
        "segmentanytree_finetuned_dev_freezer",
    )
    bundle = tmp_path / "released_model_bundle"
    (bundle / ".hydra").mkdir(parents=True)
    checkpoint = bundle / "PointGroup-PAPER.pt"
    checkpoint.write_bytes(b"released weights")
    (bundle / ".hydra" / "overrides.yaml").write_text(
        "- job_name=mls_data_run\n", encoding="utf-8"
    )
    freezer.EXPECTED_CHECKPOINT_SHA256 = freezer.sha256(checkpoint)

    split_manifest = tmp_path / "full_split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "selected_role_counts": {
                    "train": 16,
                    "val": 5,
                    "held_out_test": 0,
                },
                "test_data_converted": False,
            }
        ),
        encoding="utf-8",
    )
    stage1_run_id = "segmentanytree_for-instance_published_pretrained_20260710_231601"
    stage1_freeze = tmp_path / f"{stage1_run_id}.json"
    stage1_freeze.write_text(
        json.dumps(
            {
                "status": "frozen_for_one_time_held_out_evaluation",
                "run_id": stage1_run_id,
                "training_mode": "published_pretrained",
                "checkpoint_sha256": freezer.EXPECTED_CHECKPOINT_SHA256,
                "expected_test_plots": 11,
                "repeat_test_for_setting_selection_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    stage1_summary = tmp_path / "final_summary.csv"
    stage1_summary.write_text(
        "variant,result_status,dataset_split,plots,metrics_root\n"
        "published_pretrained,completed_aligned_pointwise_test,test,11,"
        f"/results/{stage1_run_id}/held_out_test\n",
        encoding="utf-8",
    )
    run_id = "segmentanytree_for-instance_fine_tuned_on_dev_20260711_010203"

    payload = freezer.freeze_finetuned_dev_training(
        split_manifest,
        bundle,
        stage1_freeze,
        stage1_summary,
        run_id,
    )
    assert payload["status"] == "frozen_for_development_only_fine_tuning"
    assert payload["training_mode"] == "fine_tuned_on_dev"
    assert payload["initialisation"] == "released_checkpoint_weights_only"
    assert payload["optimizer_state"] == "fresh"
    assert payload["smoke_epochs"] == 1
    assert payload["training_epochs"] == 35
    assert payload["held_out_test_plots"] == 0
    assert payload["held_out_test_jobs_permitted"] is False
    assert payload["stage1_test_run_id"] == stage1_run_id

    contaminated_manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    contaminated_manifest["selected_role_counts"]["held_out_test"] = 1
    split_manifest.write_text(json.dumps(contaminated_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected development split counts"):
        freezer.freeze_finetuned_dev_training(
            split_manifest,
            bundle,
            stage1_freeze,
            stage1_summary,
            run_id,
        )

    submitter = (
        ROOT / "methods/segmentanytree/slurm/submit_finetuned_dev_validation.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        ROOT / "methods/segmentanytree/slurm/monitor_finetuned_dev_validation.sh"
    ).read_text(encoding="utf-8")
    assert "SEGMENTANYTREE_FINETUNE_DEV_CONFIRMED" in submitter
    assert "SEGMENTANYTREE_PRETRAINED_CHECKPOINT=$RELEASED_CHECKPOINT" in submitter
    assert "SEGMENTANYTREE_TRAIN_EPOCHS=1" in submitter
    assert "SEGMENTANYTREE_TRAIN_EPOCHS=35" in submitter
    assert "SEGMENTANYTREE_TRAIN_BASE_LR=0.0001" in submitter
    assert "--array=0-4%2" in submitter
    assert "--array=0-4%4" in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=5" in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=dev" in submitter
    assert "SEGMENTANYTREE_VARIANT_REQUIRE_PREDICTIONS=1" in submitter
    assert "SEGMENTANYTREE_FINAL_TEST_CONFIRMED" not in submitter
    assert "test_from_checkpoint" not in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test" not in submitter
    assert "%.9L" in monitor
    assert "%.19e" in monitor
    assert "tail " not in monitor
    assert "held_out_test_jobs=0" in monitor


def test_finetuned_test_freeze_and_submission_are_one_time(
    tmp_path: Path,
) -> None:
    freezer = load_script(
        "methods/segmentanytree/scripts/evaluation/prepare_finetuned_test_freeze.py",
        "segmentanytree_finetuned_test_freezer",
    )
    run_id = "segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931"
    checkpoint = tmp_path / "PointGroup-PAPER.pt"
    checkpoint.write_bytes(b"fine tuned weights")
    checkpoint_sha256 = freezer.sha256(checkpoint)

    development_freeze = tmp_path / "development_freeze.json"
    development_freeze.write_text(
        json.dumps(
            {
                "status": "frozen_for_development_only_fine_tuning",
                "run_id": run_id,
                "training_mode": "fine_tuned_on_dev",
                "held_out_test_jobs_permitted": False,
                "next_gate": (
                    "five_plot_development_validation_with_nonzero_instances"
                ),
            }
        ),
        encoding="utf-8",
    )
    development_summary = tmp_path / "validation_summary.csv"
    development_summary.write_text(
        "variant,result_status,dataset_split,plots,predicted_instances,"
        "metrics_root,mean_plot_f1\n"
        "fine_tuned_on_dev_validation,completed_aligned_pointwise_test,dev,5,"
        f"270,/results/{run_id},0.5666797905966935\n",
        encoding="utf-8",
    )
    training_metadata = tmp_path / "training.json"
    training_metadata.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "training_mode": "fine_tuned_on_dev",
                "profile": "full",
                "status": "completed",
                "return_code": 0,
                "requested_epochs": 35,
                "batch_size": 8,
                "external_commit": freezer.EXPECTED_EXTERNAL_COMMIT,
                "pretrained_checkpoint_sha256": freezer.EXPECTED_RELEASED_SHA256,
                "pretrained_weight_name": "latest",
                "base_lr": 0.0001,
                "pretrained_load_validation": {"compatible_fraction": 0.99},
                "checkpoint_sha256": checkpoint_sha256,
            }
        ),
        encoding="utf-8",
    )

    payload = freezer.freeze_finetuned_test(
        run_id,
        development_freeze,
        development_summary,
        training_metadata,
        checkpoint,
    )
    assert payload["status"] == "frozen_for_one_time_held_out_evaluation"
    assert payload["training_mode"] == "fine_tuned_on_dev"
    assert payload["expected_test_plots"] == 11
    assert payload["weight_updates"] is False
    assert payload["repeat_test_for_setting_selection_permitted"] is False
    assert payload["development_predicted_instances"] == 270

    training = json.loads(training_metadata.read_text(encoding="utf-8"))
    training["requested_epochs"] = 34
    training_metadata.write_text(json.dumps(training), encoding="utf-8")
    with pytest.raises(ValueError, match="requested_epochs"):
        freezer.freeze_finetuned_test(
            run_id,
            development_freeze,
            development_summary,
            training_metadata,
            checkpoint,
        )

    submitter = (
        ROOT / "methods/segmentanytree/slurm/submit_finetuned_test.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        ROOT / "methods/segmentanytree/slurm/monitor_finetuned_test.sh"
    ).read_text(encoding="utf-8")
    assert "SEGMENTANYTREE_FINETUNED_TEST_CONFIRMED" in submitter
    assert "SEGMENTANYTREE_FINAL_TEST_CONFIRMED=1" in submitter
    assert "--array=0-10%2" in submitter
    assert "--array=0-10%4" in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_PLOTS=11" in submitter
    assert "SEGMENTANYTREE_VARIANT_EXPECTED_SPLIT=test" in submitter
    assert "Repeated test submission is refused" in submitter
    assert "train_segmentanytree" not in submitter
    assert "%.9L" in monitor
    assert "%.19e" in monitor
    assert "tail " not in monitor


def test_full_training_submission_wrapper_derives_validation_array() -> None:
    wrapper = (
        ROOT
        / "methods/segmentanytree/slurm/submit_full_training_chain.sh"
    ).read_text(encoding="utf-8")

    assert 'mkdir -p "$LOG_ROOT"' in wrapper
    assert 'VALIDATION_LAST=$((VALIDATION_COUNT - 1))' in wrapper
    assert '--array="0-${VALIDATION_LAST}%1"' in wrapper
    assert '--array="0-${VALIDATION_LAST}%4"' in wrapper
    assert "held_out_test" in wrapper
    assert "test_from_checkpoint" not in wrapper
    assert "SEGMENTANYTREE_RESUME_CHECKPOINT" in wrapper
    assert "SEGMENTANYTREE_RESUME_CHECKPOINT_SHA256" in wrapper
    assert "SEGMENTANYTREE_TRAIN_PARTITION" in wrapper
    assert "SEGMENTANYTREE_TRAIN_TIME" in wrapper
    assert "SEGMENTANYTREE_TRAIN_CPUS" in wrapper
    assert "SEGMENTANYTREE_TRAIN_MEMORY" in wrapper
    assert '--partition="$TRAIN_PARTITION"' in wrapper
    assert '--cpus-per-task="$TRAIN_CPUS"' in wrapper
    assert 'printf \'FULL_RESUME_CHECKPOINT=%q\\n\'' in wrapper


def test_pretrained_finetune_workflow_is_guarded_and_non_destructive() -> None:
    submitter = (
        ROOT / "methods/segmentanytree/slurm/submit_three_variation_overnight.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        ROOT / "methods/segmentanytree/slurm/monitor_three_variation_overnight.sh"
    ).read_text(encoding="utf-8")
    canonical = (
        ROOT / "methods/segmentanytree/slurm/submit_pretrained_finetune_comparison.sh"
    ).read_text(encoding="utf-8")
    recovery = (
        ROOT / "methods/segmentanytree/slurm/recover_three_variation_pretrained.sh"
    ).read_text(encoding="utf-8")

    assert "SEGMENTANYTREE_THREE_VARIATION_CONFIRMED" in submitter
    assert "SEGMENTANYTREE_PRETRAINED_CHECKPOINT" in submitter
    assert "SEGMENTANYTREE_CHECKPOINT_DIR=$RELEASED_DIR" not in submitter
    assert "SEGMENTANYTREE_TRAIN_BASE_LR=0.0001" in submitter
    assert "SEGMENTANYTREE_TRAIN_EPOCHS=$FINETUNE_EPOCHS" in submitter
    assert 'dependency="afterok:$pretrained_final_gate"' in submitter
    assert "afterok:$finetune_validation_gate" in submitter
    assert "SEGMENTANYTREE_FINAL_TEST_CONFIRMED=1" in submitter
    assert "pretrained_finetune_$STAMP.csv" in submitter
    assert "SEGMENTANYTREE_RETRAINED_METRICS_ROOT" not in submitter
    assert "summarise_segmentanytree_pretrained_finetune.sbatch" in submitter
    assert "SEGMENTANYTREE_PRETRAINED_FINETUNE_CONFIRMED" in canonical
    assert "segmentanytree_recovery_archive" in recovery
    assert "rm -rf" not in recovery
    assert "--follow" in monitor
    assert "tail -n 20" not in monitor
    assert "ETA:" in monitor


def test_segmentanytree_variant_summary_rejects_zero_predictions(
    tmp_path: Path,
) -> None:
    summariser = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "summarise_segmentanytree_variants.py"
        ),
        "segmentanytree_variant_summariser",
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    payload = {
        "evaluator": "pointwise_instance_metrics",
        "collection": "CULS",
        "plot_name": "plot_2_annotated",
        "relative_path": "CULS/plot_2_annotated.las",
        "split": "test",
        "point_count": 100,
        "prediction_instance_count": 0,
        "reference_instance_count": 2,
        "mean_unweighted_coverage": 0.0,
        "mean_weighted_coverage": 0.0,
        "harmonized": {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 2,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
        },
        "paper_compatible": {"f1": 0.0},
    }
    (metrics / "plot.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="zero predicted instances"):
        summariser.summarise_variant("fine_tuned", metrics, 1, "test", True)

    payload["prediction_instance_count"] = 3
    payload["harmonized"].update(
        {
            "true_positives": 1,
            "false_positives": 2,
            "false_negatives": 1,
            "precision": 1 / 3,
            "recall": 0.5,
            "f1": 0.4,
            "mean_matched_iou": 0.75,
        }
    )
    payload["paper_compatible"]["f1"] = 0.4
    (metrics / "plot.json").write_text(json.dumps(payload), encoding="utf-8")
    row = summariser.summarise_variant("fine_tuned", metrics, 1, "test", True)
    assert row["mean_plot_f1"] == pytest.approx(0.4)
    assert row["micro_f1"] == pytest.approx(0.4)
    assert row["matching_policy"] == "maximum_cardinality_one_to_one"
    assert row["predicted_instances"] == 3


def test_paper_aligned_inference_stops_before_export_merge() -> None:
    script = (
        ROOT
        / "methods/segmentanytree/scripts/runtime/patches/"
        "run_inference_for_pointwise_evaluation.sh"
    ).read_text(encoding="utf-8")

    assert "python3 eval.py" in script
    assert "eval_status=$?" in script
    assert "eval_status\" -ne 139" in script
    assert "Continuing after eval.py exit 139" in script
    assert "rename_result_files_instance.py" in script
    assert "Instance_results_forEval_" in script
    assert "merge_pt_ss_is" not in script
    assert "SEGMENTANYTREE_CHECKPOINT_DATA_CACHE" in script

    task = (
        ROOT
        / "methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_paper_test_task.sh"
    ).read_text(encoding="utf-8")
    assert 'CUSTOM_DATA_ROOT="$RUNTIME_DIR/checkpoint_data"' in task
    assert '--bind "$CUSTOM_DATA_ROOT:/sat_data"' in task
    assert "SEGMENTANYTREE_ALIGNED_OUTPUT_DIR=/sat_output" in task


def test_revalidation_slurm_scripts_support_test_only_selection() -> None:
    scripts = [
        "methods/segmentanytree/slurm/inference/run_segmentanytree_for_instance_array.sbatch",
        "methods/segmentanytree/slurm/evaluation/inspect_segmentanytree_internal_outputs.sbatch",
        "methods/segmentanytree/slurm/evaluation/audit_segmentanytree_for_instance_export_array.sbatch",
        "methods/segmentanytree/slurm/evaluation/evaluate_segmentanytree_pointwise_array.sbatch",
    ]

    for relative_path in scripts:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "FOR_INSTANCE_SPLIT" in text
        assert "--split" in text


def test_export_validator_detects_coordinate_and_label_conflicts() -> None:
    validator = load_script(
        "methods/segmentanytree/scripts/evaluation/validate_segmentanytree_export.py",
        "segmentanytree_export_validator",
    )
    source = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int64)
    reordered = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    expanded = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
        dtype=np.int64,
    )

    assert validator.coordinate_multisets_equal(source, reordered)
    assert not validator.coordinate_multisets_equal(source, expanded)
    assert (
        validator.coordinate_label_conflicts(
            source,
            np.array([1, 2, 3]),
        )
        == 1
    )


def test_export_validator_accepts_row_preserving_labelled_las(
    tmp_path: Path,
) -> None:
    validator = load_script(
        "methods/segmentanytree/scripts/evaluation/validate_segmentanytree_export.py",
        "segmentanytree_export_validator_las",
    )
    source_path = tmp_path / "source.las"
    output_path = tmp_path / "output.las"
    write_annotated_las(source_path)
    output = laspy.read(source_path)
    output.add_extra_dim(
        laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
    )
    output["PredInstance"] = np.array([0, 1, 1, 2, 2], dtype=np.int32)
    output.write(output_path)

    result = validator.validate_export(
        source_path,
        output_path,
        coordinate_tolerance=0.001,
        reference_instance_field="treeID",
        prediction_instance_field="PredInstance",
    )

    assert result["status"] == "passed"
    assert result["point_count_delta"] == 0
    assert result["coordinate_multiset_equal"] is True


def test_internal_output_inspector_identifies_evaluation_candidates(
    tmp_path: Path,
) -> None:
    inspector = load_script(
        "methods/segmentanytree/scripts/evaluation/inspect_segmentanytree_outputs.py",
        "segmentanytree_output_inspector",
    )
    (tmp_path / "semantic.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property int preds\n"
        "property int gt\n"
        "end_header\n"
        "0 0\n"
        "1 1\n"
        "1 1\n",
        encoding="utf-8",
    )
    (tmp_path / "instance.ply").write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 12\n"
        "property int preds\n"
        "property int gt\n"
        "end_header\n"
        + "\n".join(f"{index} {index}" for index in range(12))
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "PointGroup-PAPER.pt").write_bytes(b"checkpoint")

    payload = inspector.inspect_outputs(tmp_path)

    assert payload["semantic_candidates"] == ["semantic.ply"]
    assert payload["instance_candidates"] == ["instance.ply"]
    assert len(payload["checkpoint_files"]) == 1
    assert len(payload["checkpoint_files"][0]["sha256"]) == 64


def test_revalidation_summary_combines_audit_and_inventory(
    tmp_path: Path,
) -> None:
    summariser = load_script(
        "methods/segmentanytree/scripts/evaluation/summarise_segmentanytree_revalidation.py",
        "segmentanytree_revalidation_summary",
    )
    metadata = tmp_path / "metadata"
    audit = metadata / "export_validation/NIBIO/plot_1.json"
    inventory = metadata / "internal_output_inventory/NIBIO/plot_1.json"
    audit.parent.mkdir(parents=True)
    inventory.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "status": "failed",
                "safe_for_final_accuracy_evaluation": False,
                "point_count_delta": 56,
                "coordinate_multiset_equal": False,
            }
        ),
        encoding="utf-8",
    )
    inventory.write_text(
        json.dumps(
            {
                "instance_candidates": ["Instance_results_forEval0.ply"],
                "semantic_candidates": ["Semantic_results_forEval0.ply"],
                "checkpoint_files": [
                    {
                        "relative_path": "PointGroup-PAPER.pt",
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = summariser.collect(metadata)

    assert len(rows) == 1
    assert rows[0]["collection"] == "NIBIO"
    assert rows[0]["plot_name"] == "plot_1"
    assert rows[0]["point_count_delta"] == 56
    assert rows[0]["instance_candidate_count"] == 1
    assert rows[0]["semantic_candidate_count"] == 1
    assert rows[0]["checkpoint_sha256"] == "a" * 64


def test_public_inventory_example_has_required_columns() -> None:
    with (ROOT / "methods/tls2trees/examples/for_instance_inventory_summary.csv").open(
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
        ROOT / "methods/segmentanytree/examples/pilot_metrics.csv"
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
    status = json.loads(
        (
            ROOT / "methods/segmentanytree/examples/pilot_status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "provisional_coordinate_evaluation"
    assert status["historical_aligned_result_status"] == (
        "completed_retained_historical"
    )
    assert status["current_target_status"] == "completed_target_comparison"


def test_completed_segmentanytree_target_results_are_consistent() -> None:
    path = (
        ROOT
        / "methods/segmentanytree/examples/"
        "sat_completed_target_results_20260711.csv"
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["training_mode"] for row in rows] == [
        "published_pretrained",
        "fine_tuned_on_dev",
    ]
    assert {row["dataset_split"] for row in rows} == {"test"}
    assert {int(row["plots"]) for row in rows} == {11}
    assert {int(row["reference_instances"]) for row in rows} == {323}
    for row in rows:
        tp = int(row["true_positives"])
        fp = int(row["false_positives"])
        fn = int(row["false_negatives"])
        assert int(row["predicted_instances"]) == tp + fp
        assert int(row["reference_instances"]) == tp + fn
        assert float(row["micro_precision"]) == pytest.approx(tp / (tp + fp))
        assert float(row["micro_recall"]) == pytest.approx(tp / (tp + fn))
        assert float(row["micro_f1"]) == pytest.approx(
            2 * tp / (2 * tp + fp + fn)
        )

    baseline, primary = rows
    assert float(baseline["mean_plot_f1"]) == pytest.approx(
        0.45340897930934665
    )
    assert float(primary["mean_plot_f1"]) == pytest.approx(
        0.5446789009405125
    )
    assert int(primary["false_positives"]) == 331
    assert float(primary["micro_f1"]) > float(baseline["micro_f1"])

    provenance = json.loads(
        (
            ROOT
            / "methods/segmentanytree/examples/"
            "sat_completed_target_provenance_20260711.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["fine_tuned"]["run_id"] == primary["run_id"]
    assert provenance["published_pretrained"]["run_id"] == baseline["run_id"]
    assert provenance["repeat_test_for_setting_selection_permitted"] is False
    assert provenance["site_result_table"].endswith(
        "sat_completed_target_site_results_20260711.csv"
    )
    assert provenance["retention_verification"]["status"] == "retention-verified"


def test_segmentanytree_retention_verifier_requires_aligned_outputs(
    tmp_path: Path,
) -> None:
    verifier = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "verify_completed_sat_retention.py"
        ),
        "segmentanytree_retention_verifier",
    )
    predictions = tmp_path / "predictions"
    metadata = tmp_path / "run_metadata"
    metrics = tmp_path / "metrics"
    output_dir = predictions / "CULS/plot_1"
    output_dir.mkdir(parents=True)
    metadata_dir = metadata / "CULS"
    metadata_dir.mkdir(parents=True)
    metric_dir = metrics / "CULS"
    metric_dir.mkdir(parents=True)
    instance = output_dir / "Instance_results_forEval_0.ply"
    semantic = output_dir / "semantic_segmentation_0.ply"
    instance.write_bytes(b"instance labels")
    semantic.write_bytes(b"semantic labels")
    (metadata_dir / "plot_1_run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "split": "test",
                "aligned_instance_evaluation": str(instance),
                "aligned_instance_evaluation_sha256": "instance-sha",
                "aligned_semantic_evaluation": str(semantic),
                "aligned_semantic_evaluation_sha256": "semantic-sha",
            }
        ),
        encoding="utf-8",
    )
    (metric_dir / "plot_1.json").write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "result_status,dataset_split,plots,predicted_instances,"
        "reference_instances,true_positives,false_positives,false_negatives,"
        "mean_plot_f1,micro_f1\n"
        "completed_aligned_pointwise_test,test,1,3,2,1,2,1,0.4,0.4\n",
        encoding="utf-8",
    )

    payload = verifier.validate_aligned_run(
        tmp_path,
        "synthetic",
        predictions,
        metadata,
        metrics,
        summary,
        "test",
        1,
    )
    assert len(payload["prediction_instance_files"]) == 1
    assert len(payload["prediction_semantic_files"]) == 1
    assert payload["summary_metrics"]["micro_f1"] == "0.4"

    semantic.unlink()
    with pytest.raises(ValueError, match="retained counts"):
        verifier.validate_aligned_run(
            tmp_path,
            "synthetic",
            predictions,
            metadata,
            metrics,
            summary,
            "test",
            1,
        )


def test_completed_segmentanytree_site_summary_groups_all_five_sites(
    tmp_path: Path,
) -> None:
    summariser = load_script(
        (
            "methods/segmentanytree/scripts/evaluation/"
            "summarise_completed_sat_sites.py"
        ),
        "segmentanytree_completed_site_summariser",
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for index, site in enumerate(summariser.EXPECTED_SITES, start=1):
        payload = {
            "evaluator": "pointwise_instance_metrics",
            "collection": site,
            "plot_name": f"plot_{index}",
            "relative_path": f"{site}/plot_{index}.las",
            "split": "test",
            "prediction_instance_count": index + 2,
            "reference_instance_count": index + 1,
            "mean_unweighted_coverage": 0.6 + index / 100,
            "mean_weighted_coverage": 0.7 + index / 100,
            "harmonized": {
                "true_positives": index,
                "false_positives": 2,
                "false_negatives": 1,
                "precision": index / (index + 2),
                "recall": index / (index + 1),
                "f1": 2 * index / (2 * index + 3),
                "mean_matched_iou": 0.8 + index / 100,
            },
            "paper_compatible": {"f1": 2 * index / (2 * index + 3)},
        }
        (metrics / f"{site}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    rows = summariser.summarise_sites("fine_tuned_on_dev", metrics, 5, "test")
    assert [row["site"] for row in rows] == list(summariser.EXPECTED_SITES)
    assert sum(int(row["plots"]) for row in rows) == 5
    assert sum(int(row["true_positives"]) for row in rows) == 15
    assert rows[0]["micro_f1"] == pytest.approx(2 / 5)
    assert rows[-1]["micro_f1"] == pytest.approx(10 / 13)
    assert {row["result_status"] for row in rows} == {
        "completed_aligned_pointwise_site_summary"
    }

    (metrics / "TUWIEN.json").unlink()
    with pytest.raises(ValueError, match="expected 5 metrics"):
        summariser.summarise_sites("fine_tuned_on_dev", metrics, 5, "test")


def test_completed_segmentanytree_public_site_results_reconcile() -> None:
    site_path = (
        ROOT
        / "methods/segmentanytree/examples/"
        "sat_completed_target_site_results_20260711.csv"
    )
    aggregate_path = (
        ROOT
        / "methods/segmentanytree/examples/"
        "sat_completed_target_results_20260711.csv"
    )
    with site_path.open(encoding="utf-8", newline="") as handle:
        site_rows = list(csv.DictReader(handle))
    with aggregate_path.open(encoding="utf-8", newline="") as handle:
        aggregate_rows = {
            row["training_mode"]: row for row in csv.DictReader(handle)
        }

    expected_sites = {"CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"}
    assert len(site_rows) == 10
    assert {row["variant"] for row in site_rows} == set(aggregate_rows)
    assert {row["dataset_split"] for row in site_rows} == {"test"}
    assert {row["result_status"] for row in site_rows} == {
        "completed_aligned_pointwise_site_summary"
    }

    for variant, aggregate in aggregate_rows.items():
        rows = [row for row in site_rows if row["variant"] == variant]
        assert {row["site"] for row in rows} == expected_sites
        for field in (
            "plots",
            "predicted_instances",
            "reference_instances",
            "true_positives",
            "false_positives",
            "false_negatives",
        ):
            assert sum(int(row[field]) for row in rows) == int(aggregate[field])
        for row in rows:
            tp = int(row["true_positives"])
            fp = int(row["false_positives"])
            fn = int(row["false_negatives"])
            assert float(row["micro_precision"]) == pytest.approx(tp / (tp + fp))
            assert float(row["micro_recall"]) == pytest.approx(tp / (tp + fn))
            assert float(row["micro_f1"]) == pytest.approx(
                2 * tp / (2 * tp + fp + fn)
            )

    fine_tuned = {
        row["site"]: row
        for row in site_rows
        if row["variant"] == "fine_tuned_on_dev"
    }
    assert float(fine_tuned["SCION"]["mean_plot_f1"]) == pytest.approx(
        0.7206349206349205
    )
    assert float(fine_tuned["TUWIEN"]["mean_plot_f1"]) == pytest.approx(
        0.3661971830985915
    )
    assert float(fine_tuned["RMIT"]["mean_plot_f1"]) == pytest.approx(
        0.4768211920529801
    )


def test_cross_method_workbook_has_comparable_target_values() -> None:
    workbook = (
        ROOT
        / "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_benchmark_tracker.xlsx"
    )
    with zipfile.ZipFile(workbook) as archive:
        text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith(".xml")
        )
    assert "FOR-instance Shared-Protocol Leaderboard" in text
    assert "Differently Scoped Held-out Results" in text
    assert "FOR-instance All Held-out Results" in text
    assert "SegmentAnyTree" in text
    assert "TreeX" in text
    assert "TreeLearn" in text
    assert "Published pretrained" in text
    assert "Fine-tuned" in text
    assert "Only ranking-eligible rows using the classes 4/5/6 shared pointwise protocol are shown." in text
    assert "These accepted TLS2trees results use the class-3-ignore scoring mask and are not ranked with the shared protocol." in text
    assert "TLS2trees" in text
    assert "Development tuned" in text
    assert "Prediction Retention" in text
    assert "Leaf-on result excludes class-3 out-points" in text
    assert "Shared-protocol primary" in text
    assert "Shared-protocol baseline" in text
    assert "Primary; separate scoring domain" in text
    assert "Baseline; separate scoring domain" in text
    assert "Legacy-v1 reported" not in text
    assert "for_instance_pointwise_class3_ignore" in text
    assert "treelearn_for-instance_published_pretrained_20260714_134109" in text
    assert "trees above 10 m" in text
    assert "Not evaluated comparably" not in text
    assert "published pretrained matched validation baseline" not in text
    assert "fine-tuned on development epoch 70" not in text
    assert "treelearn_matched_internal_validation" not in text
    assert "Pending aligned evaluation" not in text
    assert "Pending released-weight fine-tuning" not in text


def test_public_provisional_segmentanytree_diagnostics_are_sanitised() -> None:
    prefix = ROOT / "methods/segmentanytree/examples/provisional_released_checkpoint"
    with Path(f"{prefix}_plot_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plots = list(csv.DictReader(handle))
    with Path(f"{prefix}_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        overall = next(csv.DictReader(handle))
    with Path(f"{prefix}_matches.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        matches = list(csv.DictReader(handle))
    with Path(f"{prefix}_inventory.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        inventory = list(csv.DictReader(handle))
    manifest = json.loads(Path(f"{prefix}_manifest.json").read_text())

    assert len(plots) == 32
    assert len(inventory) == 32
    assert len(matches) == 376
    assert {row["status"] for row in plots} == {"completed"}
    assert int(overall["total_reference_trees"]) == 1130
    assert int(overall["total_predicted_trees"]) == 2532
    assert int(overall["total_true_positives"]) == 376
    assert int(overall["total_false_positives"]) == 2156
    assert int(overall["total_false_negatives"]) == 754
    assert float(overall["micro_f1"]) == pytest.approx(0.20535226652102676)
    assert float(overall["pooled_mean_matched_iou"]) == pytest.approx(
        0.7263750375278218
    )
    assert manifest["status"] == (
        "provisional_coordinate_evaluation_revalidation_required"
    )
    assert manifest["historical_aligned_result_status"] == (
        "completed_retained_historical"
    )
    assert manifest["current_target_status"] == "completed_target_comparison"

    published_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.glob("methods/segmentanytree/examples/provisional_released_checkpoint_*.*")
        if path.suffix in {".csv", ".json"}
    )
    assert "/" + "Users/" not in published_text
    assert "/mnt/" not in published_text
    assert "sge" + "moorc" not in published_text


def test_provisional_workbook_cannot_claim_final_accuracy() -> None:
    workbook = (
        ROOT
        / "methods/segmentanytree/examples/"
        "provisional_released_checkpoint_results.xlsx"
    )
    with zipfile.ZipFile(workbook) as archive:
        workbook_text = "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith(".xml")
        )

    assert "Provisional Released-Checkpoint Diagnostic" in workbook_text
    assert "not accepted accuracy results" in workbook_text
    assert "Completed SegmentAnyTree accuracy benchmark" not in workbook_text
    assert "All 32 completed SegmentAnyTree evaluations" not in workbook_text


def test_no_raw_point_cloud_or_archive_is_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    forbidden = {
        ".las",
        ".laz",
        ".ply",
        ".zip",
        ".npy",
        ".npz",
        ".sif",
        ".pt",
        ".pth",
        ".ckpt",
        ".h5",
        ".hdf5",
        ".pkl",
        ".joblib",
    }
    tracked = [Path(line) for line in completed.stdout.splitlines()]

    assert not [path for path in tracked if path.suffix.lower() in forbidden]
