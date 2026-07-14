from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
