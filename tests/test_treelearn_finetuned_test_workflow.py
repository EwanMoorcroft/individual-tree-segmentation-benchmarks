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
prepare = load("prepare_for_instance_finetuned_test")


def test_test_contract_matches_completed_cross_method_subset() -> None:
    assert common.EXPECTED_TEST_PLOTS == 11
    assert common.EXPECTED_TEST_SITE_COUNTS == {
        "CULS": 1,
        "NIBIO": 6,
        "RMIT": 1,
        "SCION": 2,
        "TUWIEN": 1,
    }
    assert common.EXPECTED_TEST_POINTS == 49_709_922
    assert common.EXPECTED_TEST_REFERENCE_TREES == 323
    assert common.EXPECTED_TEST_PATHS == (
        "CULS/plot_2_annotated.las",
        "NIBIO/plot_17_annotated.las",
        "NIBIO/plot_18_annotated.las",
        "NIBIO/plot_1_annotated.las",
        "NIBIO/plot_22_annotated.las",
        "NIBIO/plot_23_annotated.las",
        "NIBIO/plot_5_annotated.las",
        "RMIT/test.las",
        "SCION/plot_31_annotated.las",
        "SCION/plot_61_annotated.las",
        "TUWIEN/test.las",
    )


def test_prepare_freezes_selected_epoch_35_and_exact_test_subset(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "treelearn_for-instance_fine_tuned_on_dev_long_20260712_233227"
    checkpoint = tmp_path / "epoch_35.pth"
    checkpoint.write_bytes(b"checkpoint")
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "status": "frozen_selected_checkpoint_pending_manual_held_out_test_authorisation",
                "method": "TreeLearn",
                "training_mode": "fine_tuned_on_dev",
                "source_long_run_id": run_id,
                "held_out_test_accessed": False,
                "test_jobs_submitted": 0,
                "selected_config_id": "full_lr_1e-5",
                "selected_epoch": 35,
                "selected_seed": 42,
                "training_plots": 16,
                "initial_checkpoint_md5": "106a80de2991c5f23484a3f9d03e3b16",
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": "a" * 64,
                "next_gate": "manual_review_before_any_held_out_test_submission",
            }
        )
    )
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
    inventory = common.EXPECTED_TEST_INVENTORY
    for relative in common.EXPECTED_TEST_PATHS:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"las")

    monkeypatch.setattr(prepare, "sha256", lambda path: "a" * 64)
    monkeypatch.setattr(
        prepare,
        "inspect_las",
        lambda path: inventory[path.relative_to(dataset).as_posix()],
    )
    payload = prepare.prepare(selected, dataset, metadata, run_id)
    assert payload["status"] == "frozen_for_one_time_held_out_test"
    assert payload["held_out_test_accessed"] is True
    assert payload["repeat_test_for_setting_selection_permitted"] is False
    assert payload["checkpoint_sha256"] == "a" * 64
    assert len(payload["plots"]) == 11
    assert all(row["split"] == "test" for row in payload["plots"])


def test_test_route_is_explicit_one_time_and_retains_predictions() -> None:
    submit = (
        ROOT / "methods/treelearn/slurm/submit_for_instance_finetuned_test.sh"
    ).read_text()
    task = (
        ROOT / "methods/treelearn/slurm/run_for_instance_finetuned_test.sbatch"
    ).read_text()
    monitor = (
        ROOT / "methods/treelearn/slurm/monitor_for_instance_finetuned_test.sh"
    ).read_text()
    runner = (SCRIPTS / "run_for_instance_one_plot_smoke.py").read_text()
    evaluator = (SCRIPTS / "evaluate_for_instance_one_plot_smoke.py").read_text()
    summary = (SCRIPTS / "summarise_for_instance_finetuned_test.py").read_text()
    gate = (SCRIPTS / "validate_for_instance_finetuned_test.py").read_text()
    assert "TREELEARN_FINETUNED_TEST_CONFIRMED" in submit
    assert "--array=0-10%2" in submit
    assert "Refusing repeated or colliding" in submit
    assert "No training job was submitted" in submit
    assert "--held-out-test-authorized" in runner
    assert '"held_out_test"' in runner
    assert '"held_out_test"' in evaluator
    assert "held_out_test_accessed=true" in task
    assert "verified_prediction_file_count" in summary
    assert "complete_test_prediction_set_retained" in summary
    assert "EXPECTED_TEST_PLOTS * 5" in gate
    assert "tail " not in monitor
    assert "squeue" in monitor and "sacct" in monitor
