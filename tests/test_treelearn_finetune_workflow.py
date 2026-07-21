from __future__ import annotations

import importlib.util
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "methods/treelearn/scripts/prepare_for_instance_finetune.py"


def load_module():
    spec = importlib.util.spec_from_file_location("treelearn_finetune_prepare", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finetune_preparation_is_seeded_and_development_only(tmp_path: Path) -> None:
    module = load_module()
    dataset = tmp_path / "dataset"
    plots = []
    for index in range(21):
        site = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")[index % 5]
        source = dataset / site / f"plot_{index}.las"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic")
        plots.append({"task_index": index, "split": "dev", "collection": site,
                      "relative_path": f"{site}/plot_{index}.las", "input_las": str(source)})
    manifest = tmp_path / "development.json"
    manifest.write_text(json.dumps({"held_out_test_accessed": False, "plots": plots}))
    repo = tmp_path / "TreeLearn"
    (repo / "configs/_modular").mkdir(parents=True)
    checkpoint = tmp_path / "published.pth"
    checkpoint.write_bytes(b"weights")

    assert module.assign_roles(plots, 42) == module.assign_roles(plots, 42)
    roles = module.assign_roles(plots, 42)
    assert sum(row["training_role"] == "train" for row in roles) == 16
    assert sum(row["training_role"] == "validation" for row in roles) == 5

    run_root = tmp_path / "run"
    frozen = module.prepare(manifest, run_root, repo, checkpoint, 42, 64)
    assert frozen["training_mode"] == "fine_tuned_on_dev"
    assert frozen["held_out_test_accessed"] is False
    assert (frozen["training_plots"], frozen["validation_plots"]) == (16, 5)
    assert len(list((run_root / "data/train/forests").iterdir())) == 16
    full = json.loads((run_root / "configs/finetune_full.yaml").read_text())
    assert full["pretrain"] == str(checkpoint.resolve())
    assert full["epochs"] == 100
    assert full["optimizer"]["lr"] == 0.0003
    assert isinstance(full["scheduler"]["lr_min"], float)
    assert isinstance(full["scheduler"]["warmup_lr_init"], float)


def test_finetune_submission_has_smoke_gate_and_no_test_route() -> None:
    submitter = (ROOT / "methods/treelearn/slurm/submit_for_instance_finetune.sh").read_text()
    monitor = (ROOT / "methods/treelearn/slurm/monitor_for_instance_finetune.sh").read_text()
    assert 'afterok:$PREP' in submitter
    assert 'afterok:$SMOKE' in submitter
    assert "TREELEARN_FINETUNE_DEV_CONFIRMED" in submitter
    assert "No held-out test job was submitted" in submitter
    assert "checkpoint_ready" in monitor


def test_finetune_validation_is_five_plot_test_locked() -> None:
    submitter = (ROOT / "methods/treelearn/slurm/submit_for_instance_finetune_validation.sh").read_text()
    task = (ROOT / "methods/treelearn/slurm/run_for_instance_finetune_validation.sbatch").read_text()
    summary = (ROOT / "methods/treelearn/scripts/summarise_for_instance_finetune_validation.py").read_text()
    assert "--array=0,3,7,8,20%2" in submitter
    assert 'VALIDATION_RUN_ID="${TRAINING_RUN_ID}_${VALIDATION_TAG}_validation_$STAMP"' in submitter
    assert "TREELEARN_FINETUNE_CHECKPOINT_OVERRIDE" in submitter
    assert "TREELEARN_FINETUNE_VALIDATION_AFTEROK_JOB" in submitter
    assert "--training-mode fine_tuned_on_dev" in task
    assert "held_out_test_accessed=false" in task
    assert 'len(rows) != 5' in summary
    assert '"retention_status": "retention_verified"' in summary
    assert "No held-out test job was submitted" in submitter


def test_completed_finetune_result_is_rejected_and_retained() -> None:
    result_path = ROOT / "methods/treelearn/examples/treelearn_finetune_validation_results_20260712.csv"
    with result_path.open(encoding="utf-8", newline="") as handle:
        result = next(csv.DictReader(handle))
    tp = int(result["true_positives"])
    fp = int(result["false_positives"])
    fn = int(result["false_negatives"])
    assert result["evaluation_protocol"] == "for_instance_pointwise_v1"
    assert result["matching_policy"] == "maximum_cardinality_one_to_one"
    assert result["dataset_split"] == "dev_validation"
    assert result["held_out_test_accessed"] == "false"
    assert result["retention_status"] == "retention_verified"
    assert result["result_status"] == "rejected_validation_regression"
    assert int(result["retained_prediction_files"]) == 25
    assert int(result["retained_prediction_bytes"]) == 2_704_552_488
    assert math.isclose(float(result["micro_precision"]), tp / (tp + fp))
    assert math.isclose(float(result["micro_recall"]), tp / (tp + fn))
    assert math.isclose(float(result["micro_f1"]), 2 * tp / (2 * tp + fp + fn))
    assert float(result["mean_plot_f1_delta"]) < 0
    assert float(result["micro_f1_delta"]) < 0


def test_cross_method_results_separate_comparable_groups_and_retain_predictions() -> None:
    results_path = (
        ROOT
        / "outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv"
    )
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    published_default_tls2trees = (
        ROOT
        / "methods/tls2trees/examples/"
        "tls2trees_published_default_test_results.csv"
    ).is_file()
    legacy_rows = [row for row in rows if row["method_slug"] != "tls2trees"]
    assert len(rows) == 6 + int(published_default_tls2trees)
    assert {row["comparable_group"] for row in legacy_rows} == {
        "held_out_test_primary"
    }
    assert {(row["method_slug"], row["training_mode"]) for row in legacy_rows} == {
        ("segmentanytree", "published_pretrained"),
        ("segmentanytree", "fine_tuned_on_dev"),
        ("treex", "external_training_only"),
        ("treelearn", "published_pretrained"),
        ("treelearn", "fine_tuned_on_dev"),
    }
    assert {row["evaluation_split"] for row in rows} == {"test"}
    assert {int(row["plots"]) for row in rows} == {11}
    assert {int(row["reference_instances"]) for row in rows} == {323}
    assert all(
        row["evaluation_protocol"] == "for_instance_pointwise_v1"
        for row in legacy_rows
    )
    clean_pretrained = next(
        row
        for row in rows
        if row["method_slug"] == "treelearn"
        and row["training_mode"] == "published_pretrained"
    )
    assert clean_pretrained["run_id"] == (
        "treelearn_for-instance_published_pretrained_20260714_134109"
    )
    assert int(clean_pretrained["true_positives"]) == 34
    assert int(clean_pretrained["false_positives"]) == 332
    assert int(clean_pretrained["false_negatives"]) == 289
    assert math.isclose(float(clean_pretrained["mean_plot_f1"]), 0.07894350692791946)
    assert math.isclose(float(clean_pretrained["micro_f1"]), 0.09869375907111756)
    tls2trees_rows = [row for row in rows if row["method_slug"] == "tls2trees"]
    assert len(tls2trees_rows) == 1 + int(published_default_tls2trees)
    assert {row["variant"] for row in tls2trees_rows} == {
        "development_tuned",
        *({"published_default"} if published_default_tls2trees else set()),
    }
    assert {row["evaluation_protocol"] for row in tls2trees_rows} == {
        "for_instance_pointwise_class3_ignore"
    }
    assert {row["comparable_group"] for row in tls2trees_rows} == {
        "held_out_test_tls2trees_class3_ignore"
    }

    diagnostics_path = (
        ROOT
        / "outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv"
    )
    with diagnostics_path.open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    assert len(diagnostics) == 4 + int(published_default_tls2trees)
    assert {row["method_slug"] for row in diagnostics} == {
        "treelearn",
        "tls2trees",
    }
    assert {row["variant"] for row in diagnostics} == {
        "published_pretrained",
        "fine_tuned_on_dev",
        "development_tuned",
        *({"published_default"} if published_default_tls2trees else set()),
    }
    assert all(
        row["variant"] == row["training_mode"]
        or (
            row["method_slug"] == "tls2trees"
            and row["variant"] in {"development_tuned", "published_default"}
            and row["training_mode"] == "external_training_only"
        )
        for row in diagnostics
    )
    assert {row["evaluation_split"] for row in diagnostics} == {
        "dev",
        "dev_validation",
        "test",
    }
    assert {int(row["plots"]) for row in diagnostics} == {5, 11, 21}
    tls2trees_diagnostics = [
        row for row in diagnostics if row["method_slug"] == "tls2trees"
    ]
    assert len(tls2trees_diagnostics) == 1 + int(published_default_tls2trees)
    assert {row["evaluation_protocol"] for row in tls2trees_diagnostics} == {
        "for_instance_pointwise_class3_ignore"
    }
    assert {row["comparable_group"] for row in tls2trees_diagnostics} == {
        "tls2trees_leaf_off_class3_ignore_diagnostic"
    }
    assert {row["result_status"] for row in tls2trees_diagnostics} == {
        "completed_aligned_pointwise_test"
    }
    assert {row["held_out_test_accessed"] for row in tls2trees_diagnostics} == {
        "true"
    }
    assert not any(row["comparable_group"] == "held_out_test_primary" for row in diagnostics)

    retention_path = (
        ROOT
        / "outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv"
    )
    with retention_path.open(encoding="utf-8", newline="") as handle:
        retention = list(csv.DictReader(handle))
    retained = {
        (row["method_slug"], row["variant"], row["retention_profile"]): row
        for row in retention
    }
    for key in (
        ("treex", "unsupervised_parameterised", "held_out_test"),
        ("segmentanytree", "published_pretrained", "held_out_test"),
        ("segmentanytree", "fine_tuned_on_dev", "held_out_test"),
        ("treelearn", "published_pretrained", "held_out_test"),
        ("treelearn", "fine_tuned_on_dev", "checkpoint_sweep"),
        ("treelearn", "fine_tuned_on_dev", "held_out_test"),
        (
            "tls2trees",
            "development_tuned",
            "held_out_test_tls2trees_class3_ignore",
        ),
    ):
        assert retained[key]["future_metrics_without_inference"] == "true"
    if published_default_tls2trees:
        assert retained[
            (
                "tls2trees",
                "published_default",
                "held_out_test_tls2trees_class3_ignore",
            )
        ]["future_metrics_without_inference"] == "true"
    assert (
        "treelearn",
        "published_pretrained",
        "development_diagnostic",
    ) in retained
    assert ("treelearn", "fine_tuned_on_dev", "epoch_70_validation") in retained
    assert int(
        retained[
            (
                "tls2trees",
                "development_tuned",
                "held_out_test_tls2trees_class3_ignore",
            )
        ][
            "retained_file_count"
        ]
    ) == 22
    published = retained[("treelearn", "published_pretrained", "held_out_test")]
    assert published["run_id"] == (
        "treelearn_for-instance_published_pretrained_20260714_134109"
    )
    assert published["evaluation_split"] == "test"
    assert int(published["retained_file_count"]) == 55
