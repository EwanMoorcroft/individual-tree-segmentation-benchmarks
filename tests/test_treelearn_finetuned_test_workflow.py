from __future__ import annotations

import csv
import importlib.util
import json
import math
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


def test_completed_test_public_evidence_reconciles_and_is_retained() -> None:
    examples = ROOT / "methods/treelearn/examples"
    with (
        examples / "treelearn_finetuned_test_results_20260713.csv"
    ).open(encoding="utf-8", newline="") as handle:
        overall_rows = list(csv.DictReader(handle))
    with (
        examples / "treelearn_finetuned_test_site_results_20260713.csv"
    ).open(encoding="utf-8", newline="") as handle:
        sites = list(csv.DictReader(handle))
    provenance = json.loads(
        (
            examples / "treelearn_finetuned_test_provenance_20260713.json"
        ).read_text(encoding="utf-8")
    )

    assert len(overall_rows) == 1
    overall = overall_rows[0]
    assert overall["run_id"] == (
        "treelearn_for-instance_fine_tuned_on_dev_long_20260712_233227"
    )
    assert overall["dataset_split"] == "test"
    assert overall["evaluation_protocol"] == "for_instance_pointwise_v1"
    assert overall["matching_policy"] == "maximum_cardinality_one_to_one"
    assert overall["evaluation_mask"] == (
        "union_of_reference_tree_and_predicted_tree_points"
    )
    assert overall["held_out_test_accessed"] == "true"
    assert overall["retention_status"] == "retention_verified"
    assert int(overall["retained_prediction_files"]) == 55
    assert int(overall["plots"]) == 11
    assert int(overall["reference_instances"]) == 323
    assert {row["site"] for row in sites} == {
        "CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"
    }
    assert all(row["dataset_split"] == "test" for row in sites)
    assert all(row["held_out_test_accessed"] == "true" for row in sites)

    for overall_field, site_field in (
        ("plots", "completed_plots"),
        ("point_count", "point_count"),
        ("evaluated_point_count", "evaluated_point_count"),
        ("predicted_instances", "predicted_instances"),
        ("reference_instances", "reference_instances"),
        ("true_positives", "true_positives"),
        ("false_positives", "false_positives"),
        ("false_negatives", "false_negatives"),
    ):
        assert int(overall[overall_field]) == sum(
            int(row[site_field]) for row in sites
        )

    tp = int(overall["true_positives"])
    fp = int(overall["false_positives"])
    fn = int(overall["false_negatives"])
    assert math.isclose(float(overall["micro_precision"]), tp / (tp + fp))
    assert math.isclose(float(overall["micro_recall"]), tp / (tp + fn))
    assert math.isclose(float(overall["micro_f1"]), 2 * tp / (2 * tp + fp + fn))
    assert math.isclose(
        float(overall["mean_plot_f1"]),
        sum(
            float(row["mean_plot_f1"]) * int(row["completed_plots"])
            for row in sites
        ) / 11,
    )

    assert provenance["run_id"] == overall["run_id"]
    assert provenance["result_status"] == overall["result_status"]
    assert provenance["held_out_test_accessed"] is True
    assert provenance["repeat_test_for_setting_selection_permitted"] is False
    assert provenance["checkpoint"]["sha256"] == (
        "dcc02bb9fdd81cfbdb94454bb7a744c17eee7fa2c4a53096d529b21eb64fc590"
    )
    assert provenance["retention"]["verified_prediction_file_count"] == 55
    assert provenance["retention"]["manifest_sha256"] == (
        "972ad17ba103b151095d2925862e76e7186594b549d659cea2ca781d62600b0b"
    )
    assert provenance["run_summary_sha256"] == (
        "a612e3f1b8a51aaf3b86ba977262ee45941721a50f693573cec01324dbdfce8b"
    )
