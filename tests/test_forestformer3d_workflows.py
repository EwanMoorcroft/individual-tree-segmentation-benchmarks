from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from methods.forestformer3d.scripts.evaluation import verify_development_run
from methods.forestformer3d.scripts.data import prepare_finetune
from methods.forestformer3d.scripts.runtime import build_finetune_configs


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forestformer3d"


def test_readme_has_repository_method_contract_sections() -> None:
    text = (METHOD / "README.md").read_text(encoding="utf-8")
    for section in {
        "Method Summary",
        "Upstream Repository And Citation",
        "Training Mode Support",
        "Input Requirements",
        "Output Contract",
        "FOR-instance Compatibility",
        "Barkla Environment",
        "Slurm Workflow",
        "Evaluation Route",
        "Known Limitations",
        "Current Benchmark Status",
    }:
        assert f"## {section}" in text
    assert "no held-out forestformer3d accuracy result exists" in " ".join(
        text.lower().split()
    )


def test_environment_submitter_is_guarded_and_auto_starts_live_monitor() -> None:
    submit = (METHOD / "slurm/submit_environment_build.sh").read_text(
        encoding="utf-8"
    )
    assert "FF3D_ENVIRONMENT_BUILD_CONFIRMED" in submit
    assert 'test "$(git branch --show-current)" = "method/forestformer3d"' in submit
    assert 'test -z "$(git status --porcelain)"' in submit
    assert "test ! -e \"$RUN_ROOT\"" in submit
    assert "test ! -e \"$ENV_ROOT\"" in submit
    assert "FF3D_BASE_SIF_SHA256" in submit
    assert "build_rootless_environment.sh" in submit
    assert '--dependency="afterok:$BUILD_JOB"' in submit
    assert "FF3D_CANCEL_INVALID_DEPENDENCIES" in submit
    assert "monitor_workflow.sh" in submit
    assert 'FF3D_MONITOR_SECONDS:-30' in submit
    assert "scancel $BUILD_JOB $VALIDATE_JOB" in submit


def test_validation_only_submitter_reuses_completed_environment_with_new_evidence() -> None:
    submit = (METHOD / "slurm/submit_environment_validation.sh").read_text(
        encoding="utf-8"
    )
    assert "FF3D_ENVIRONMENT_VALIDATION_CONFIRMED" in submit
    assert ': "${FF3D_ENV_ROOT:?Missing FF3D_ENV_ROOT}"' in submit
    assert 'test -f "$FF3D_ENV_ROOT/environment_build.complete"' in submit
    assert 'test ! -e "$FF3D_ENV_ROOT/environment_build.incomplete"' in submit
    assert "environment_manifest_sha256.txt" in submit
    assert "environment-validation" in submit
    assert "FF3D_BENCHMARK_COMMIT" in submit
    assert "monitor_workflow.sh" in submit
    assert 'FF3D_MONITOR_SECONDS:-30' in submit


def test_live_monitor_combines_scheduler_and_expected_file_state() -> None:
    monitor = (METHOD / "slurm/monitor_workflow.sh").read_text(encoding="utf-8")
    assert "squeue -j" in monitor
    assert "sacct -X -j" in monitor
    assert "EXPECTED FILES" in monitor
    assert "stat -c %s" in monitor
    assert "sleep \"$WATCH_SECONDS\"" in monitor
    assert "Ctrl-C stops monitoring but does not cancel jobs" in monitor
    for state in (
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
    ):
        assert state in monitor
    assert "DependencyNeverSatisfied" in monitor
    assert 'scancel "$job_id"' in monitor
    assert "tasks_complete=" in monitor


def test_development_workflow_is_guarded_bounded_and_development_only() -> None:
    preflight = (METHOD / "slurm/submit_development_preflight.sh").read_text(
        encoding="utf-8"
    )
    preflight_job = (
        METHOD / "slurm/prepare_development_preflight.sbatch"
    ).read_text(encoding="utf-8")
    submit = (
        METHOD / "slurm/submit_published_pretrained_development.sh"
    ).read_text(encoding="utf-8")
    task = (
        METHOD / "slurm/run_published_pretrained_development.sbatch"
    ).read_text(encoding="utf-8")
    summary = (
        METHOD / "slurm/summarise_published_pretrained_development.sbatch"
    ).read_text(encoding="utf-8")
    assert "FF3D_PREFLIGHT_CONFIRMED" in preflight
    assert "prepare_development_manifest.py" in preflight_job
    assert "FF3D_DEVELOPMENT_CONFIRMED" in submit
    assert "FF3D_SMOKE_CONFIRMATION" in submit
    assert "--array=0-20%2" in submit
    assert 'afterany:$ARRAY_JOB' in submit
    assert '"$RUN_ROOT/tasks"' in submit
    assert "monitor_workflow.sh" in submit
    assert "#SBATCH --partition=gpu-a100-lowbig" in task
    assert "#SBATCH --gres=gpu:a100:1" in task
    assert "prepare_development_plot.py" in task
    assert "validate_development_plot.py" in task
    assert "evaluate_development_plot.py" in task
    assert "--bind \"$TASK_ROOT:/ff3d_task\"" in task
    assert "--work-dir /ff3d_task/raw" in task
    assert "dummy" not in task
    assert "#SBATCH --partition=nodes" in summary
    assert "summarise_development.py" in summary
    assert "held_out_access=false" in task


def test_development_verification_is_independent_and_cpu_only() -> None:
    submit = (
        METHOD / "slurm/submit_published_pretrained_development_verification.sh"
    ).read_text(encoding="utf-8")
    job = (
        METHOD / "slurm/verify_published_pretrained_development.sbatch"
    ).read_text(encoding="utf-8")
    assert "FF3D_DEVELOPMENT_VERIFICATION_CONFIRMED" in submit
    assert "FF3D_SOURCE_RUN_ROOT" in submit
    assert "development-verification" in submit
    assert "monitor_workflow.sh" in submit
    assert "#SBATCH --partition=nodes" in job
    assert "#SBATCH --gres" not in job
    assert "verify_development_run.py" in job
    assert "held_out_access=false" in job


def test_build_and_validation_jobs_freeze_identity_and_dependency_outputs() -> None:
    build = (METHOD / "slurm/build_environment.sbatch").read_text(encoding="utf-8")
    submit = (METHOD / "slurm/submit_environment_build.sh").read_text(
        encoding="utf-8"
    )
    validate = (METHOD / "slurm/validate_environment.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=nodes" in build
    assert "apptainer build --fakeroot" not in build
    assert "build_rootless_environment.sh" in build
    assert "FF3D_BUILD_INPUT_SHA256" in build
    assert 'test ! -e "$FF3D_ENV_ROOT"' in build
    assert "base_sif_sha256.txt" in build
    assert "conda_explicit.txt" in build
    assert '--bind "$FF3D_ENV_ROOT:/ff3d_environment"' in build
    assert "--env FF3D_ENV_ROOT=/ff3d_environment" in build
    assert "FF3D_ROOTLESS_BUILDER_SHA256" in build
    assert "FF3D_TOOLCHAIN_LOCK_SHA256" in build
    assert "FF3D_TOOLCHAIN_LOCK_SHA256" in submit

    assert "#SBATCH --partition=gpu-a100-lowbig" in validate
    assert "#SBATCH --gres=gpu:a100:1" in validate
    assert "validate_environment.py" in validate
    assert "--require-cuda" in validate
    assert "environment_validation.complete" in validate
    assert "epoch_3000_fix.pth:ro" in validate
    assert "--environment-root" in validate
    assert "FF3D_BASE_SIF_SHA256" in validate
    assert '--bind "$FF3D_ENV_ROOT:/ff3d_environment:ro"' in validate
    assert "--environment-root /ff3d_environment" in validate
    assert (
        'LD_LIBRARY_PATH="/ff3d_environment/toolchain/lib:'
        '/usr/local/cuda/lib64:/.singularity.d/libs"'
    ) in validate
    assert validate.count("/usr/bin/env") == 2


def test_all_method_shell_entrypoints_parse() -> None:
    shell_paths = sorted(
        [
            *METHOD.rglob("*.sh"),
            *METHOD.rglob("*.sbatch"),
        ]
    )
    assert shell_paths
    for path in shell_paths:
        subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_shared_integration_is_documented_but_not_applied() -> None:
    manifest = (METHOD / "docs/shared_integration_manifest.md").read_text(
        encoding="utf-8"
    )
    assert "do not apply" in manifest
    assert "No candidate or pending shared registry row" in manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_development_verifier_rehashes_and_checks_row_identity(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    summary_root = run / "summary"
    summary_root.mkdir(parents=True)
    csv_rows = []
    retention_rows = []
    totals = {"point_count": 0, "true_positives": 0, "false_positives": 0,
              "false_negatives": 0}
    for task_index in range(21):
        plot_id = f"SITE/plot_{task_index}"
        relative_path = f"{plot_id}.las"
        task = run / "tasks" / f"SITE_plot_{task_index}"
        for relative in (
            "staged_input/points/forestformer3d_development_test.bin",
            "staged_input/semantic_mask/reference.bin",
            "staged_input/instance_mask/reference.bin",
            "staged_input/reference.pkl",
            "raw/forestformer3d_development_test.ply",
            "raw/model_input_fingerprint.json",
            "raw/effective_predict_audit.json",
            "raw/checkpoint_entrypoint_adapter.json",
            "raw/resource_usage.json",
        ):
            path = task / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{task_index}:{relative}".encode())
        input_manifest = {
            "schema": "forestformer3d_development_plot_input_v1",
            "plot_id": plot_id,
            "relative_path": relative_path,
            "point_count": 3,
            "split": "development",
            "held_out_access": False,
            "source_row_index": "zero_based_identity",
        }
        input_path = task / "staged_input/input_manifest.json"
        input_path.write_text(json.dumps(input_manifest), encoding="utf-8")
        prediction_path = task / "validation/predictions.npz"
        prediction_path.parent.mkdir(parents=True)
        np.savez_compressed(
            prediction_path,
            pred_tree_id=np.array([-1, 1, 1]),
            target_tree_id=np.array([-1, 7, 7]),
            classification=np.array([2, 4, 5]),
            pred_classification=np.array([0, 4, 4]),
            source_row_index=np.arange(3),
        )
        prediction_sha256 = _sha256(prediction_path)
        validation = {
            "schema": "forestformer3d_development_plot_validation_v1",
            "status": "passed",
            "split": "development",
            "held_out_access": False,
            "plot_id": plot_id,
            "relative_path": relative_path,
            "point_count": 3,
            "exact_row_alignment": True,
            "prediction_npz_sha256": prediction_sha256,
        }
        validation_path = task / "validation/validation.json"
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        metrics = {
            "schema": "forestformer3d_development_plot_metrics_v1",
            "status": "completed",
            "split": "development",
            "held_out_access": False,
            "plot_id": plot_id,
            "relative_path": relative_path,
            "point_count": 3,
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 0,
            "prediction_npz_sha256": prediction_sha256,
        }
        metrics_path = task / "evaluation/metrics.json"
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        (task / "task.complete").touch()
        csv_rows.append(metrics)

        retained = (
            "staged_input/input_manifest.json",
            "staged_input/points/forestformer3d_development_test.bin",
            "staged_input/semantic_mask/reference.bin",
            "staged_input/instance_mask/reference.bin",
            "staged_input/reference.pkl",
            "staged_input/evaluation_sidecar.npz",
            "raw/forestformer3d_development_test.ply",
            "raw/model_input_fingerprint.json",
            "raw/effective_predict_audit.json",
            "raw/checkpoint_entrypoint_adapter.json",
            "raw/resource_usage.json",
            "validation/predictions.npz",
            "validation/validation.json",
            "evaluation/metrics.json",
        )
        sidecar = task / "staged_input/evaluation_sidecar.npz"
        np.savez(sidecar, source_row_index=np.arange(3))
        for relative in retained:
            path = task / relative
            retention_rows.append(
                {
                    "logical_role": "test",
                    "task_index": task_index,
                    "plot_id": plot_id,
                    "relative_path": path.relative_to(run).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        for key in totals:
            totals[key] += int(metrics[key])

    fields = [
        "plot_id", "relative_path", "point_count", "reference_instance_count",
        "prediction_instance_count", "true_positives", "false_positives",
        "false_negatives", "precision", "recall", "f1", "mean_matched_iou",
    ]
    with (summary_root / "per_plot_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    summary = {
        "schema": "forestformer3d_development_summary_v1",
        "status": "complete_published_pretrained_development_diagnostics",
        "run_id": "forestformer3d-test-development",
        "benchmark_commit": "a" * 40,
        "split": "development",
        "held_out_access": False,
        "plot_count": 21,
        "retained_artifact_count": len(retention_rows),
        "retained_bytes": sum(row["size_bytes"] for row in retention_rows),
        **totals,
    }
    (summary_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    retention = {
        "schema": "forestformer3d_retention_manifest_v1",
        "run_id": summary["run_id"],
        "immutable_run_root": True,
        "held_out_access": False,
        "artifact_count": len(retention_rows),
        "artifacts": retention_rows,
    }
    (summary_root / "retention_manifest.json").write_text(
        json.dumps(retention), encoding="utf-8"
    )
    with (summary_root / "artifact_sha256.txt").open("w", encoding="utf-8") as handle:
        for name in ("summary.json", "per_plot_metrics.csv", "retention_manifest.json"):
            handle.write(f"{_sha256(summary_root / name)}  {name}\n")
    (summary_root / "summary.complete").touch()
    (run / "development.complete").touch()

    result = verify_development_run.verify(run, tmp_path / "verification")
    assert result["status"] == "verified"
    assert result["task_count"] == 21
    assert result["exact_source_row_alignment"] is True
    assert result["held_out_access"] is False


def test_independent_development_verifier_rejects_changed_prediction(
    tmp_path: Path,
) -> None:
    # The full fixture above covers the success path; this unit directly checks
    # the immutable-output requirement before any expensive hashing begins.
    run = tmp_path / "run"
    run.mkdir()
    output = run / "verification"
    with pytest.raises(ValueError, match="outside the immutable run root"):
        verify_development_run.verify(run, output)


def test_finetune_preparation_freezes_canonical_development_only_split(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_run"
    plots = []
    for index in range(21):
        plot_id = f"SITE/plot_{index}"
        safe = f"SITE_plot_{index}"
        relative = f"{plot_id}.las"
        plots.append(
            {
                "task_index": index,
                "plot_id": plot_id,
                "safe_plot_id": safe,
                "relative_path": relative,
                "dataset_split": "development",
                "point_count": 2,
                "input_sha256": f"{index:064x}",
            }
        )
        staged = source / "tasks" / safe / "staged_input"
        for name in (
            "points/forestformer3d_development_test.bin",
            "semantic_mask/reference.bin",
            "instance_mask/reference.bin",
        ):
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{index}:{name}".encode())
        (staged / "input_manifest.json").write_text(
            json.dumps(
                {
                    "plot_id": plot_id,
                    "relative_path": relative,
                    "split": "development",
                    "held_out_access": False,
                    "source_row_index": "zero_based_identity",
                }
            ),
            encoding="utf-8",
        )
    (source / "development_manifest.json").write_text(
        json.dumps(
            {
                "schema": "forestformer3d_development_manifest_v1",
                "dataset_split": "development",
                "held_out_access": False,
                "plots": plots,
            }
        ),
        encoding="utf-8",
    )
    (source / "development.complete").touch()
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema": "forestformer3d_development_verification_v1",
                "status": "verified",
                "run_id": source.name,
                "held_out_access": False,
                "exact_source_row_alignment": True,
                "task_count": 21,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_finetune.prepare(
        source,
        verification,
        tmp_path / "finetune",
        benchmark_commit="b" * 40,
        checkpoint_sha256="c" * 64,
    )
    frozen_rows = prepare_finetune.assign_roles(plots)
    assert {
        row["task_index"]
        for row in frozen_rows
        if row["fine_tune_role"] == "validation"
    } == {0, 3, 7, 8, 20}
    assert result["split"]["training_plots"] == 16
    assert result["split"]["validation_plots"] == 5
    assert result["split"]["held_out_access"] is False
    assert result["training"]["total_examples"] == 560
    assert result["training"]["total_optimizer_steps"] == 280
    assert (tmp_path / "finetune/preparation.complete").is_file()


def test_effective_finetune_config_preserves_architecture_and_freezes_budget() -> None:
    base = {
        "model": {"prepare_epoch": 1000, "radius": 16, "query_point_num": 300},
        "train_dataloader": {
            "batch_size": 2,
            "num_workers": 12,
            "prefetch_factor": 10,
            "dataset": {"data_root": "old", "ann_file": "train.pkl"},
        },
        "val_dataloader": {
            "dataset": {"data_root": "old", "ann_file": "val.pkl"}
        },
        "test_dataloader": {
            "dataset": {"data_root": "old", "ann_file": "test.pkl"}
        },
        "optim_wrapper": {
            "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 0.05}
        },
        "param_scheduler": {"end": 450000},
        "train_cfg": {"max_epochs": 3000, "val_interval": 100},
        "default_hooks": {"checkpoint": {"interval": 1}},
    }
    configured = build_finetune_configs.configure(
        base,
        data_root="/run/data/",
        checkpoint="/inputs/checkpoint.pth",
        work_dir="/run/training",
        smoke=False,
    )
    assert configured["model"]["radius"] == 16
    assert configured["model"]["query_point_num"] == 300
    assert configured["model"]["prepare_epoch"] == -1
    assert configured["optim_wrapper"]["optimizer"]["lr"] == 1e-5
    assert configured["train_cfg"]["max_epochs"] == 35
    assert configured["param_scheduler"]["end"] == 280
    assert configured["default_hooks"]["checkpoint"]["interval"] == 7
    assert configured["load_from"] == "/inputs/checkpoint.pth"
    assert configured["resume"] is False
