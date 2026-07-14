from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "methods/treelearn/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


common = load("for_instance_test_common")
prepare = load("prepare_for_instance_pretrained_test")
guard = load("run_treelearn_pipeline_empty_group_guard")
recovery = load("prepare_for_instance_pretrained_test_recovery")


def test_prepare_freezes_clean_checkpoint_and_exact_test_subset(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "treelearn_for-instance_published_pretrained_20260714_120000"
    checkpoint = tmp_path / "model_weights_finetuned.pth"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "FORinstance_dataset"
    metadata = dataset / "data_split_metadata.csv"
    metadata.parent.mkdir()
    metadata.write_text(
        "path,folder,split\n"
        + "".join(
            f"{path},{Path(path).parts[0]},test\n"
            for path in common.EXPECTED_TEST_PATHS
        )
        + "CULS/plot_1_annotated.las,CULS,dev\n"
    )
    for relative in common.EXPECTED_TEST_PATHS:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(prepare, "md5", lambda path: prepare.EXPECTED_CHECKPOINT_MD5)
    monkeypatch.setattr(prepare, "sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        prepare,
        "inspect_las",
        lambda path: common.EXPECTED_TEST_INVENTORY[
            path.relative_to(dataset).as_posix()
        ],
    )
    payload = prepare.prepare(checkpoint, dataset, metadata, run_id)
    assert payload["status"] == "frozen_for_one_time_held_out_test"
    assert payload["run_id"] == run_id
    assert payload["variant"] == "published_pretrained"
    assert payload["training_mode"] == "published_pretrained"
    assert payload["dataset_split"] == "test"
    assert payload["held_out_test_accessed"] is True
    assert payload["repeat_test_for_setting_selection_permitted"] is False
    assert payload["checkpoint_md5"] == prepare.EXPECTED_CHECKPOINT_MD5
    assert payload["checkpoint_sha256"] == "a" * 64
    assert len(payload["plots"]) == 11
    assert all(row["split"] == "test" for row in payload["plots"])
    manifest = tmp_path / "test_freeze.json"
    manifest.write_text(json.dumps(payload))
    rows, loaded = common.load_test_manifest(manifest)
    assert loaded["run_id"] == run_id
    assert len(rows) == 11


def test_pretrained_route_is_one_time_separate_and_retains_predictions() -> None:
    submit = (
        ROOT / "methods/treelearn/slurm/submit_for_instance_pretrained_test.sh"
    ).read_text()
    task = (
        ROOT / "methods/treelearn/slurm/run_for_instance_pretrained_test.sbatch"
    ).read_text()
    monitor = (
        ROOT / "methods/treelearn/slurm/monitor_for_instance_pretrained_test.sh"
    ).read_text()
    assert "TREELEARN_PRETRAINED_TEST_CONFIRMED" in submit
    assert "treelearn_published_pretrained_test_once" in submit
    assert "Refusing repeated" in submit
    assert "--array=0-10%2" in submit
    assert "No training job was submitted" in submit
    assert "model_weights_finetuned.pth" in submit
    assert "106a80de2991c5f23484a3f9d03e3b16" in submit
    assert "variant=published_pretrained" in task
    assert "held_out_test_accessed=true" in task
    assert "tail " not in monitor
    assert "squeue" in monitor and "sacct" in monitor


def test_empty_group_guard_preserves_unassigned_background() -> None:
    coords = np.zeros((4, 3), dtype=np.float32)
    predictions = np.full(4, -1, dtype=np.int64)
    calls = []
    result = guard.guarded_assignment(
        coords,
        predictions,
        -1,
        original=lambda *args: (_ for _ in ()).throw(AssertionError("called")),
        on_empty=calls.append,
    )
    assert np.array_equal(result, np.zeros(4, dtype=np.int64))
    assert result is not predictions
    assert calls == [4]


def test_empty_group_guard_delegates_when_reference_clusters_exist() -> None:
    coords = np.zeros((3, 3), dtype=np.float32)
    predictions = np.array([2, -1, -1], dtype=np.int64)
    expected = np.array([2, 2, 2], dtype=np.int64)
    result = guard.guarded_assignment(
        coords,
        predictions,
        -1,
        original=lambda *args: expected,
    )
    assert result is expected


def test_recovery_archives_only_failed_task_and_partial_aggregates(tmp_path: Path) -> None:
    run_id = "treelearn_for-instance_published_pretrained_20260714_134109"
    split = tmp_path / "data_split_metadata.csv"
    split.write_text("path,split\n")
    plots = []
    for task_index, relative in enumerate(common.EXPECTED_TEST_PATHS):
        point_count, reference_count = common.EXPECTED_TEST_INVENTORY[relative]
        plots.append(
            {
                "task_index": task_index,
                "plot_id": common.plot_id(relative),
                "safe_plot_id": common.safe_plot_id(common.plot_id(relative)),
                "relative_path": relative,
                "collection": Path(relative).parts[0],
                "split": "test",
                "input_las": str((tmp_path / relative).resolve()),
                "point_count": point_count,
                "reference_tree_count": reference_count,
                "input_sha256": "a" * 64,
                "split_metadata": str(split.resolve()),
                "split_metadata_sha256": "b" * 64,
            }
        )
    manifest = tmp_path / "test_freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "frozen_for_one_time_held_out_test",
                "method": "TreeLearn",
                "dataset": "FOR-instance",
                "run_id": run_id,
                "variant": "published_pretrained",
                "training_mode": "published_pretrained",
                "dataset_split": "test",
                "held_out_test_accessed": True,
                "repeat_test_for_setting_selection_permitted": False,
                "expected_test_plot_count": 11,
                "checkpoint_sha256": "c" * 64,
                "plots": plots,
            }
        )
    )
    runtime = tmp_path / "runtime" / recovery.SAFE_PLOT_ID
    runtime.mkdir(parents=True)
    (runtime / "partial_tile.npz").write_bytes(b"partial")
    predictions = tmp_path / "predictions"
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    inference = metadata / f"{recovery.SAFE_PLOT_ID}_inference.json"
    inference.write_text(
        json.dumps(
            {
                "status": "failed",
                "error": {"message": "non-zero exit status 1"},
            }
        )
    )
    tables = tmp_path / "tables"
    for row in plots:
        root = tables / "per_plot" / row["safe_plot_id"]
        root.mkdir(parents=True)
        if row["task_index"] == recovery.TASK_INDEX:
            (root / "status.json").write_text("{}")
        else:
            (root / "metrics.json").write_text("{}")
    (tables / "failures.csv").write_text(
        "task_index,relative_path,status\n"
        f"8,{recovery.RELATIVE_PATH},documented_inference_failure\n"
    )
    original = "1" * 40
    current = "2" * 40
    (tables / "run_summary.json").write_text(
        json.dumps(
            {
                "benchmark_commit": original,
                "completed_plots": 10,
                "documented_failures": 1,
            }
        )
    )
    for name in recovery.AGGREGATES:
        path = tables / name
        if not path.exists():
            path.write_text(name)
    archive = tmp_path / "archive"
    output = metadata / "task_8_execution_recovery.json"
    payload = recovery.prepare(
        manifest,
        tmp_path / "runtime",
        predictions,
        metadata,
        tables,
        archive,
        output,
        original,
        current,
    )
    assert payload["status"] == "prepared_single_execution_recovery"
    assert payload["task_index"] == 8
    assert not runtime.exists()
    assert not (tables / "per_plot" / recovery.SAFE_PLOT_ID).exists()
    assert (tables / "per_plot" / plots[0]["safe_plot_id"] / "metrics.json").is_file()
    assert (archive / "failure/inference.json").is_file()
    assert (archive / "aggregate/final_summary.csv").is_file()
    assert output.is_file()


def test_pretrained_recovery_route_is_exact_and_does_not_retrain() -> None:
    submit = (
        ROOT / "methods/treelearn/slurm/submit_for_instance_pretrained_test_recovery.sh"
    ).read_text()
    task = (
        ROOT / "methods/treelearn/slurm/run_for_instance_pretrained_test_recovery.sbatch"
    ).read_text()
    summary = (
        ROOT / "methods/treelearn/slurm/summarise_for_instance_pretrained_test_recovery.sbatch"
    ).read_text()
    assert "treelearn_for-instance_published_pretrained_20260714_134109" in submit
    assert "treelearn_pretrained_test_recovery_once" in submit
    assert "--task-index 8" in task
    assert "--allow-empty-group-recovery" in task
    assert "model_or_parameter_selection_performed=false" in task
    assert "--recovery-manifest" in summary
    assert "train.py" not in submit + task + summary
