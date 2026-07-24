from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest
import yaml
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forainet"


def load_script(relative_path: str, name: str):
    path = METHOD / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_script("scripts/runtime/forainet_contract.py", "forainet_contract")
evaluator = load_script(
    "scripts/evaluation/evaluate_for_instance.py", "forainet_evaluator"
)
sidecar = load_script(
    "scripts/data/prepare_alignment_sidecar.py", "forainet_sidecar"
)
input_adapter = load_script(
    "scripts/data/prepare_label_isolated_input.py", "forainet_input_adapter"
)
merge_extractor = load_script(
    "scripts/runtime/extract_official_merge.py", "forainet_merge_extractor"
)
exposure = load_script(
    "scripts/provenance/validate_exposure_audit.py", "forainet_exposure"
)
retention = load_script(
    "scripts/provenance/build_retention_manifest.py", "forainet_retention"
)
development_manifest = load_script(
    "scripts/provenance/prepare_development_manifest.py",
    "forainet_development_manifest",
)
development_task = load_script(
    "scripts/provenance/resolve_development_task.py",
    "forainet_development_task",
)
development_summary = load_script(
    "scripts/provenance/summarise_development_run.py",
    "forainet_development_summary",
)
development_export = load_script(
    "scripts/provenance/export_development_results.py",
    "forainet_development_export",
)
finetune_data = load_script(
    "scripts/data/prepare_finetune_data.py",
    "forainet_finetune_data",
)
finetune_validation_task = load_script(
    "scripts/provenance/resolve_finetune_validation_task.py",
    "forainet_finetune_validation_task",
)
finetune_training_config = load_script(
    "scripts/provenance/stage_finetune_training_config.py",
    "forainet_finetune_training_config",
)
finetune_snapshots = load_script(
    "scripts/provenance/snapshot_finetune_checkpoints.py",
    "forainet_finetune_snapshots",
)
finetune_validation_summary = load_script(
    "scripts/provenance/summarise_finetune_validation.py",
    "forainet_finetune_validation_summary",
)


def test_scaffold_and_configs_are_method_local() -> None:
    assert {
        "configs",
        "docs",
        "examples",
        "scripts",
        "slurm",
    } <= {path.name for path in METHOD.iterdir() if path.is_dir()}
    assert {
        "data",
        "runtime",
        "evaluation",
        "provenance",
    } <= {path.name for path in (METHOD / "scripts").iterdir() if path.is_dir()}
    for path in sorted((METHOD / "configs").glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["project"]["dataset_slug"] == "for-instance"
        assert payload["project"]["method_slug"] == "forainet"
        assert payload["dataset"]["slug"] == "for-instance"
        assert payload["method"]["slug"] == "forainet"


def test_qualification_identity_is_frozen() -> None:
    config = yaml.safe_load(
        (METHOD / "configs/qualification.yml").read_text(encoding="utf-8")
    )
    assert config["method"]["selected_release"] == "original"
    assert config["method"]["selection_reason"] == (
        "no_complete_official_forainetv2_release"
    )
    assert config["method"]["upstream_commit"] == (
        "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
    )
    assert config["method"]["checkpoint"]["sha256"] == (
        "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
    )
    assert config["method"]["checkpoint"]["provider_checksum"] is None
    container = config["method"]["container"]
    assert container["base_image_digest"] == (
        "sha256:83e4b2841034cdf45ea5b9a5b472eb2c07b1b23d4836d32666a881db29a8dceb"
    )
    assert container["cuda_arch"] == "8.0"
    assert container["qualification_gpu"] == "a100"
    assert container["barkla_build_probe"]["sha256"] == (
        "2a111b22871288abe8eb205fe4a14424290bc4e2376e6c4c170f82260b3052db"
    )
    assert container["build_toolchain"]["installer_sha256"] == (
        "41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1"
    )
    assert container["build_toolchain"]["release_rpm_sha256"] == (
        "1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605"
    )
    assert config["gates"]["barkla_root_mapped_fakeroot_probe_passed"] is True
    assert config["gates"]["barkla_root_mapped_apt_build_blocked"] is True
    assert config["gates"]["barkla_userlocal_fakeroot_toolchain_verified"] is True
    assert config["gates"]["barkla_image_verified"] is True
    assert config["gates"]["checkpoint_full_load_verified"] is True
    assert container["qualified_image"]["sha256"] == (
        "ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d"
    )
    assert container["qualified_image"]["numpy"] == "1.23.5"
    assert container["checkpoint_load_evidence"]["numpy"] == "1.23.5"
    assert container["checkpoint_load_evidence"]["compatible_fraction"] == 1.0
    assert config["gates"]["held_out_authorised"] is False


def test_container_definition_pins_mutable_upstream_inputs() -> None:
    definition = (
        METHOD / "containers/forainet-cuda111-a100.def"
    ).read_text(encoding="utf-8")
    assert "From: nvidia/cuda@sha256:" in definition
    assert "11.1.1-cudnn8-devel-ubuntu20.04@sha256:" not in definition
    for commit in (
        "9f81ae66b33b883cd08ee4f64d08cf633608b118",
        "74099d10a51c71c14318bce63d6421f698b24f24",
        "ec3b205fbd7da9f1e41b9d83cdf3f6236e2ef1c4",
    ):
        assert commit in definition
    assert "TORCH_CUDA_ARCH_LIST=8.0" in definition
    assert "--requirement /opt/forainet/requirements.lock" in definition
    assert "numpy==1.23.5" in definition
    assert 'assert numpy.__version__ == "1.23.5"' in definition
    assert "numpy==1.24.4" not in definition
    assert "hdbscan/archive/master" not in definition
    assert "git+https://github.com/NVIDIA/MinkowskiEngine.git" not in definition
    probe = (METHOD / "containers/fakeroot-apt-probe.def").read_text(
        encoding="utf-8"
    )
    assert "From: nvidia/cuda@sha256:" in probe
    assert "apt-get install -y --no-install-recommends less" in probe
    lock = (METHOD / "containers/requirements.lock").read_text(encoding="utf-8")
    assert "pylidar" not in lock
    assert "rios" not in lock


def test_image_build_is_cpu_only_and_qualification_targets_a100() -> None:
    build = (METHOD / "slurm/build_forainet_image.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=nodes" in build
    assert "#SBATCH --gres" not in build
    assert '"$apptainer" build --fakeroot' in build
    assert 'mktemp -d "/tmp/forai-build-${SLURM_JOB_ID}-' in build
    assert "${FORAINET_TOOLCHAIN_ROOT:?set FORAINET_TOOLCHAIN_ROOT}" in build

    installer = (
        METHOD / "slurm/install_forainet_apptainer_toolchain.sbatch"
    ).read_text(encoding="utf-8")
    assert '-d el8 -v "$release_rpm"' in installer
    assert "41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1" in installer
    assert "1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605" in installer
    assert '"apptainer version 1.3.6-1"' in installer
    assert '"$apptainer" build --fakeroot' in installer
    assert '"apptainer version 1.3.6-1"' in build
    qualification = (METHOD / "slurm/qualify_forainet_assets.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu-a-lowsmall" in qualification
    assert "#SBATCH --gres=gpu:a100:1" in qualification
    assert '"$image" \\\n  python3.8 ' in qualification
    assert 'benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"' in qualification
    assert '--bind "$benchmark_root:$benchmark_root:ro"' in qualification
    checkpoint_probe = (
        METHOD / "scripts/provenance/probe_checkpoint_load.py"
    ).read_text(encoding="utf-8")
    assert "ModelCheckpoint(" in checkpoint_probe
    assert "run_config.data.fold = []" in checkpoint_probe
    assert "checkpoint.dataset_properties" in checkpoint_probe
    assert '"numpy": numpy.__version__' in checkpoint_probe
    assert "PretainedRegistry" not in checkpoint_probe
    smoke = (METHOD / "slurm/run_forainet_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu-a-lowsmall" in smoke
    assert "#SBATCH --gres=gpu:a100:1" in smoke
    assert "CULS/plot_1_annotated.las" in smoke
    assert "FORAINET_SMOKE_CONFIRMED" in smoke
    assert "data_split_metadata.csv" in smoke
    assert 'sub(/\\r$/, "", header)' in smoke
    assert 'sub(/\\r$/, "", value)' in smoke
    assert "ForAINet smoke failed at line" in smoke
    assert 'mkdir -p "$(dirname "$FORAINET_RUN_ROOT")"' in smoke
    assert 'mkdir "$FORAINET_RUN_ROOT"' in smoke
    assert "rev-parse --git-common-dir" in smoke
    assert '--bind "$benchmark_git_common:$benchmark_git_common:ro"' in smoke
    runtime = (
        METHOD / "scripts/runtime/run_for_instance_smoke.py"
    ).read_text(encoding="utf-8")
    assert 'cpu_open3d_env["LD_LIBRARY_PATH"] = ""' in runtime
    assert runtime.count("env=cpu_open3d_env") == 2
    assert '"cpu_open3d_uses_container_glx": True' in runtime
    assert '"treeinsfused" / "raw"' in runtime
    assert "os.link(inference_ply, raw_catalogue_input)" in runtime
    assert '"dataset_raw_catalogue": "hardlink_to_label_isolated_input"' in runtime
    assert "accepted_error_markers" in runtime
    assert 'treeins_set1.py\\", line 204, in final_eval' in runtime
    assert "ZeroDivisionError: float division by zero" in runtime
    assert '"official_eval_exit_code": label_probe_exit_code' in runtime
    assert '"schema": "forainet_label_independence_probe_v2"' in runtime
    assert '"primary_file_sha256": sha256(primary)' in runtime
    assert '"probe_file_sha256": sha256(probe)' in runtime
    assert '"primary_prediction_values_sha256"' in runtime
    assert '"probe_prediction_values_sha256"' in runtime
    assert runtime.index("label_probe_exit_code = run_checked") > runtime.index(
        "label_probe_output"
    )


def test_exposure_table_is_exact_and_test_only() -> None:
    rows = exposure.validate(METHOD / "examples/checkpoint_exposure_32_plots.csv")
    assert len(rows) == 32
    assert sum(row["benchmark_split"] == "dev" for row in rows) == 21
    assert sum(row["benchmark_split"] == "test" for row in rows) == 11
    assert {
        row["relative_path"]
        for row in rows
        if row["benchmark_split"] == "test"
    } == exposure.EXPECTED_TEST_PATHS

    evidence = json.loads(
        (METHOD / "examples/exposure_evidence_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["retrieved_date"] == "2026-07-22"
    by_id = {row["identifier"]: row for row in evidence["sources"]}
    assert by_id["official_repository"]["git_commit"] == (
        "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
    )
    assert by_id["official_original_test_fold"]["git_blob_sha1"] == (
        "f9886421d90261f1bf80319ddfe7f5218665ddbf"
    )
    assert by_id["official_checkpoint"]["locally_computed_sha256"] == (
        "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
    )


def test_full_development_route_is_guarded_and_development_only() -> None:
    accepted = json.loads(
        (
            METHOD / "examples/accepted_development_smoke_20260723.json"
        ).read_text(encoding="utf-8")
    )
    assert accepted["status"] == "accepted"
    assert accepted["split"] == "dev"
    assert accepted["held_out_access"] is False
    assert accepted["source_row_index_exact"] is True
    assert accepted["coordinate_matching"] is False

    runtime = (
        METHOD / "scripts/runtime/run_for_instance_smoke.py"
    ).read_text(encoding="utf-8")
    assert '"finetune_validation"' in runtime
    assert "non-smoke route requires frozen task identity" in runtime
    assert "runtime route permits development plots only" in runtime
    assert "verified_by_accepted_development_smoke" in runtime
    assert (
        "args.expected_point_count\n"
        '                if args.route in {"development", "finetune_validation"}'
        in runtime
    )

    task = (METHOD / "slurm/run_forainet_development.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu-a-lowsmall" in task
    assert "#SBATCH --gres=gpu:a100:1" in task
    assert "#SBATCH --cpus-per-task=12" in task
    assert "#SBATCH --mem=128G" in task
    assert "#SBATCH --time=04:00:00" in task
    assert "FORAINET_DEVELOPMENT_CONFIRMED" in task
    assert "--route development" in task
    assert "development task index is outside 0..20" in task

    submit = (METHOD / "slurm/submit_forainet_development.sh").read_text(
        encoding="utf-8"
    )
    assert '--array="0-20%2"' in submit
    assert '--dependency="afterok:$prepare_job"' in submit
    assert '--dependency="afterany:$array_job"' in submit
    recovery = (
        METHOD / "slurm/submit_forainet_development_recovery.sh"
    ).read_text(encoding="utf-8")
    assert '--array="16,20%2"' in recovery
    assert '--dependency="afterok:$array_job"' in recovery
    assert ')" = "19"' in recovery
    monitor = (METHOD / "slurm/monitor_forainet_development.sh").read_text(
        encoding="utf-8"
    )
    for status in (
        "PENDING",
        "RUNNING",
        "COMPLETED_WAITING_GATE",
        "COMPLETE",
        "FAILED",
        "BLOCKED",
        "STALE",
        "UNKNOWN",
    ):
        assert status in monitor
    assert "held_out_access=forbidden" in monitor


def test_finetune_split_and_official_setting_one_mapping() -> None:
    plots = [
        {"task_index": index, "relative_path": f"SITE/plot_{index}.las", "split": "dev"}
        for index in range(21)
    ]
    assigned = finetune_data.assign_roles(plots, 42)
    validation = [
        row["task_index"]
        for row in assigned
        if row["training_role"] == "validation"
    ]
    assert validation == [0, 3, 7, 8, 20]
    assert sum(row["training_role"] == "train" for row in assigned) == 16

    classification = np.asarray([0, 1, 2, 3, 4, 5, 6, 4, 5, 6])
    tree_id = np.asarray([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    keep, stuff_ids = finetune_data.official_keep_mask(
        classification, tree_id
    )
    assert stuff_ids.tolist() == [0]
    assert keep.tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert finetune_data.OFFICIAL_SEMANTIC_MAPPING == {
        0: 0,
        1: 1,
        2: 2,
        4: 3,
        5: 4,
        6: 5,
    }


def test_finetune_plan_is_checkpoint_initialised_and_test_locked() -> None:
    config = yaml.safe_load(
        (METHOD / "configs/for_instance_finetune.yml").read_text(
            encoding="utf-8"
        )
    )
    plan = config["training_plan"]
    assert config["dataset"]["training_plot_count"] == 16
    assert config["dataset"]["validation_plot_count"] == 5
    assert config["dataset"]["split_seed"] == 42
    assert config["dataset"]["held_out_access"] == "forbidden"
    assert plan["split_seed"] == 42
    assert plan["upstream_training_seed"] == 2022
    assert plan["configured_epoch_limit_exclusive"] == 150
    assert plan["upstream_epoch_labels"] == [1, 149]
    assert plan["effective_epoch_count"] == 149
    assert plan["checkpoint_epochs"] == [30, 60, 90, 120, 149]
    assert plan["batch_size"] == 4
    assert plan["samples_per_epoch"] == 3000
    assert plan["precision"] == "fp32"
    assert plan["checkpoint_initialisation"]["route"] == (
        "models.PointGroup-PAPER.path_pretrained"
    )
    assert plan["checkpoint_initialisation"]["weight_name"] == "latest"
    assert plan["checkpoint_initialisation"]["resume"] is False
    assert plan["selection_metric"] == "canonical_validation_micro_f1"
    assert plan["evaluation_protocol"] == "for_instance_pointwise_v1"
    assert plan["held_out_access"] == "forbidden"


def test_finetune_smoke_is_development_gated_and_official() -> None:
    prepare = (
        METHOD / "slurm/prepare_forainet_finetune.sbatch"
    ).read_text(encoding="utf-8")
    smoke = (
        METHOD / "slurm/run_forainet_finetune_smoke.sbatch"
    ).read_text(encoding="utf-8")
    submit = (
        METHOD / "slurm/submit_forainet_finetune_smoke.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        METHOD / "slurm/monitor_forainet_finetune_smoke.sh"
    ).read_text(encoding="utf-8")
    assert 'test -f "$FORAINET_DEVELOPMENT_ROOT/final_gate.json"' in prepare
    assert "--development-final-gate" in prepare
    assert "--seed 42" in prepare
    assert "#SBATCH --gres" not in prepare
    assert "#SBATCH --gres=gpu:a100:1" in smoke
    assert "data=panoptic/treeins_set1_treemix3d" in smoke
    assert "models=panoptic/FORpartseg_3heads" in smoke
    assert "training=treeins_set1_mixtree" in smoke
    assert "stage_finetune_model_config.py" in smoke
    assert "stage_finetune_training_config.py" in smoke
    assert (
        '--bind "$staged_model_config:$official_model_config:ro"' in smoke
    )
    assert (
        '--bind "$staged_training_config:$official_training_config:ro"'
        in smoke
    )
    assert "models.PointGroup-PAPER.path_pretrained=" not in smoke
    assert "training.epochs=2" in smoke
    assert "debugging.early_break=true" in smoke
    assert "visualization.activate=false" in smoke
    assert "validate_finetune_smoke.py" in smoke
    assert '--dependency="afterok:$prepare_job"' in submit
    assert "FORAINET_FINETUNE_CONFIRMED" in submit
    assert "held_out_access=forbidden full_training_submitted=no" in monitor
    assert "No held-out" not in smoke

    stage = (
        METHOD / "scripts/provenance/stage_finetune_model_config.py"
    ).read_text(encoding="utf-8")
    assert (
        "cfa698af22a4f545436aaf7285d46bc1d3690044c4d4de9db40bcac8ed90c2f5"
        in stage
    )
    assert '"models.PointGroup-PAPER.path_pretrained_only"' in stage
    assert "differences != [177]" in stage
    training_stage = (
        METHOD / "scripts/provenance/stage_finetune_training_config.py"
    ).read_text(encoding="utf-8")
    assert (
        "7f491d8e4060974fafba1401ac38cec23c3476160decefd64ce60099c230ae96"
        in training_stage
    )
    assert '"training_values_changed": False' in training_stage
    assert "differences != [1, 2]" in training_stage


def test_finetune_training_config_stage_changes_header_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official = tmp_path / "treeins_set1_mixtree.yaml"
    official.write_text(
        f"{finetune_training_config.REFERENCE_LINE}\n"
        f"{finetune_training_config.PACKAGE_LINE}\n"
        "epochs: 150\n"
        "batch_size: 4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        finetune_training_config,
        "EXPECTED_OFFICIAL_CONFIG_SHA256",
        finetune_training_config.sha256(official),
    )
    staged = tmp_path / "staged.yaml"
    metadata = tmp_path / "metadata.json"
    payload = finetune_training_config.stage(official, staged, metadata)
    source_lines = official.read_text(encoding="utf-8").splitlines()
    staged_lines = staged.read_text(encoding="utf-8").splitlines()

    assert staged_lines == [source_lines[1], source_lines[0], *source_lines[2:]]
    assert payload["changed_lines"] == [1, 2]
    assert payload["training_values_changed"] is False
    assert json.loads(metadata.read_text(encoding="utf-8")) == payload


def test_full_finetune_is_smoke_gated_and_retains_frozen_candidates() -> None:
    task = (
        METHOD / "slurm/run_forainet_finetune_full.sbatch"
    ).read_text(encoding="utf-8")
    submit = (
        METHOD / "slurm/submit_forainet_finetune_full.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        METHOD / "slurm/monitor_forainet_finetune_full.sh"
    ).read_text(encoding="utf-8")
    watcher = (
        METHOD / "scripts/provenance/snapshot_finetune_checkpoints.py"
    ).read_text(encoding="utf-8")
    validator = (
        METHOD / "scripts/provenance/validate_finetune_full.py"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in task
    assert "#SBATCH --time=3-00:00:00" in task
    assert 'test -f "$FORAINET_FINETUNE_ROOT/smoke/final_gate.json"' in task
    assert "training.epochs=150" in task
    assert "stage_finetune_training_config.py" in task
    assert (
        '--bind "$staged_training_config:$official_training_config:ro"'
        in task
    )
    assert "debugging.early_break=false" in task
    assert "snapshot_finetune_checkpoints.py" in task
    assert "validate_finetune_full.py" in task
    assert "FORAINET_FINETUNE_FULL_CONFIRMED" in submit
    assert "validation_submitted=no" in monitor
    assert "EXPECTED_EPOCHS = (30, 60, 90, 120, 149)" in watcher
    assert '"held_out_access": False' in watcher
    assert "epoch-149 candidate differs" in validator
    assert '"next_gate": "canonical_five_plot_candidate_validation"' in validator


def test_finetune_checkpoint_watcher_retains_rolling_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "PointGroup-PAPER.pt"
    candidates = tmp_path / "candidates"
    completion = tmp_path / "training.complete"

    def write_checkpoint(epoch: int) -> None:
        checkpoint.write_text(f"{epoch}\n", encoding="utf-8")

    def checkpoint_epoch(path: Path) -> tuple[int, int]:
        epoch = int(path.read_text(encoding="utf-8").strip())
        return epoch, epoch

    write_checkpoint(1)
    monkeypatch.setattr(finetune_snapshots, "EXPECTED_EPOCHS", (1, 2))
    monkeypatch.setattr(finetune_snapshots, "EXPECTED_TENSOR_COUNT", 1)
    monkeypatch.setattr(finetune_snapshots, "checkpoint_epoch", checkpoint_epoch)
    sleep_calls = 0

    def advance(_: int) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            write_checkpoint(2)
        elif sleep_calls == 2:
            completion.write_text("complete\n", encoding="utf-8")

    monkeypatch.setattr(finetune_snapshots.time, "sleep", advance)
    records = finetune_snapshots.snapshot(
        checkpoint,
        candidates,
        completion,
        poll_seconds=5,
        timeout_seconds=30,
    )

    assert [row["epoch"] for row in records] == [1, 2]
    assert all((candidates / row["filename"]).is_file() for row in records)
    index = json.loads((candidates / "index.json").read_text(encoding="utf-8"))
    assert index["status"] == "complete"
    assert index["expected_epochs"] == [1, 2]
    assert index["candidate_count"] == 2
    assert index["held_out_access"] is False


def test_finetune_validation_is_five_by_five_and_test_locked() -> None:
    task = (
        METHOD / "slurm/run_forainet_finetune_validation.sbatch"
    ).read_text(encoding="utf-8")
    submit = (
        METHOD / "slurm/submit_forainet_finetune_validation.sh"
    ).read_text(encoding="utf-8")
    monitor = (
        METHOD / "slurm/monitor_forainet_finetune_validation.sh"
    ).read_text(encoding="utf-8")
    resolver = (
        METHOD / "scripts/provenance/resolve_finetune_validation_task.py"
    ).read_text(encoding="utf-8")
    summary = (
        METHOD / "scripts/provenance/summarise_finetune_validation.py"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in task
    assert "--checkpoint-kind fine_tuned_on_dev" in task
    assert "--route finetune_validation" in task
    assert "--expected-checkpoint-epoch" in task
    assert '--array="0-24%2"' in submit
    assert '--dependency="afterany:$array_job"' in submit
    assert "plot_gates=%s/25" in monitor
    assert "test_submission=no" in monitor
    assert "EXPECTED_EPOCHS = (30, 60, 90, 120, 149)" in resolver
    assert "VALIDATION_TASKS = 25" in resolver
    assert '"selection_metric": "micro_f1"' in summary
    assert '"lower_false_positives", "earlier_epoch"' in summary
    assert '"held_out_access": False' in summary


def test_finetune_validation_task_maps_candidate_major_order(
    tmp_path: Path,
) -> None:
    records = []
    for index in range(21):
        records.append(
            {
                "task_index": index,
                "relative_path": f"SITE/plot_{index}.las",
                "split": "dev",
                "training_role": (
                    "validation" if index in {0, 3, 7, 8, 20} else "train"
                ),
                "source_sha256": f"source-{index}",
                "source_point_count": 1000 + index,
            }
        )
    manifest = tmp_path / "finetune.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "forainet_finetune_data_manifest_v1",
                "status": "complete",
                "held_out_access": False,
                "held_out_paths_included": False,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "candidates.json"
    index.write_text(
        json.dumps(
            {
                "schema": "forainet_finetune_candidate_index_v1",
                "status": "complete",
                "held_out_access": False,
                "candidates": [
                    {
                        "epoch": epoch,
                        "filename": f"epoch_{epoch}.pt",
                        "sha256": f"checkpoint-{epoch}",
                    }
                    for epoch in (30, 60, 90, 120, 149)
                ],
            }
        ),
        encoding="utf-8",
    )
    task = finetune_validation_task.resolve(manifest, index, 7)
    assert task["candidate_epoch"] == 60
    assert task["plot_offset"] == 2
    assert task["development_task_index"] == 7
    assert task["relative_path"] == "SITE/plot_7.las"
    assert task["point_count"] == 1007


def test_finetune_candidate_selection_uses_frozen_tie_breakers() -> None:
    candidates = [
        {"candidate_epoch": 30, "f1": 0.7, "false_positives": 30},
        {"candidate_epoch": 60, "f1": 0.8, "false_positives": 40},
        {"candidate_epoch": 90, "f1": 0.8, "false_positives": 20},
        {"candidate_epoch": 120, "f1": 0.8, "false_positives": 20},
    ]
    selected = finetune_validation_summary.select_candidate(candidates)
    assert selected["candidate_epoch"] == 90
    with pytest.raises(ValueError, match="candidate summary is empty"):
        finetune_validation_summary.select_candidate([])


def test_exposure_validator_rejects_test_training_role(tmp_path: Path) -> None:
    source = METHOD / "examples/checkpoint_exposure_32_plots.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["checkpoint_role"] = "train_or_validation"
    invalid = tmp_path / "invalid.csv"
    with invalid.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="test-only"):
        exposure.validate(invalid)


def test_alignment_reorders_by_exact_source_index() -> None:
    result = contract.align_full_resolution_prediction(
        source_row_index=np.asarray([2, 0, 3, 1]),
        pred_semantic_internal=np.asarray([4, 0, 1, 2]),
        pred_instance_id=np.asarray([22, -1, 0, 11]),
        expected_point_count=4,
    )
    assert result.source_row_index.tolist() == [0, 1, 2, 3]
    assert result.pred_classification.tolist() == [0, 4, 6, 0]
    assert result.pred_tree_id.tolist() == [0, 12, 23, 0]


def test_alignment_maps_verified_uncovered_sentinel_to_background() -> None:
    result = contract.align_full_resolution_prediction(
        source_row_index=np.arange(3),
        pred_semantic_internal=np.asarray([2, -1, 4]),
        pred_instance_id=np.asarray([7, -1, 9]),
        expected_point_count=3,
    )
    assert result.pred_classification.tolist() == [4, 0, 6]
    assert result.pred_tree_id.tolist() == [8, 0, 10]
    with pytest.raises(ValueError, match="must be paired"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([2, -1]),
            pred_instance_id=np.asarray([7, 0]),
            expected_point_count=2,
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([0, 1, 1], "duplicated or missing"),
        ([0, 1], "exactly one row"),
        ([0, 1, 3], "out-of-range"),
    ],
)
def test_alignment_rejects_invalid_row_maps(rows: list[int], message: str) -> None:
    count = 3
    with pytest.raises(ValueError, match=message):
        contract.align_full_resolution_prediction(
            source_row_index=np.asarray(rows),
            pred_semantic_internal=np.zeros(len(rows), dtype=np.int64),
            pred_instance_id=np.zeros(len(rows), dtype=np.int64),
            expected_point_count=count,
        )


def test_alignment_rejects_unknown_semantics_and_stuff_instances() -> None:
    with pytest.raises(ValueError, match="unknown ForAINet"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([0, 9]),
            pred_instance_id=np.asarray([0, 0]),
            expected_point_count=2,
        )


def test_semantics_control_the_prediction_union_and_instance_ids() -> None:
    result = contract.align_full_resolution_prediction(
        source_row_index=np.arange(3),
        pred_semantic_internal=np.asarray([2, 3, 4]),
        pred_instance_id=np.asarray([-1, 0, 99]),
        expected_point_count=3,
    )
    assert result.pred_tree_id.tolist() == [0, 1, 100]
    assert result.pred_classification.tolist() == [4, 5, 6]
    stuff_result = contract.align_full_resolution_prediction(
        source_row_index=np.arange(2),
        pred_semantic_internal=np.asarray([0, 2]),
        pred_instance_id=np.asarray([5, 7]),
        expected_point_count=2,
    )
    assert stuff_result.pred_classification.tolist() == [0, 4]
    assert stuff_result.pred_tree_id.tolist() == [0, 8]
    with pytest.raises(ValueError, match="must be -1 or non-negative"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([2, 3]),
            pred_instance_id=np.asarray([-2, 0]),
            expected_point_count=2,
        )


def test_overlap_rows_collapse_only_when_identical() -> None:
    rows, semantics, instances = contract.collapse_identical_overlap_rows(
        source_row_index=np.asarray([0, 1, 1, 2]),
        pred_semantic_internal=np.asarray([0, 2, 2, 4]),
        pred_instance_id=np.asarray([0, 7, 7, 9]),
    )
    assert rows.tolist() == [0, 1, 2]
    assert semantics.tolist() == [0, 2, 4]
    assert instances.tolist() == [0, 7, 9]
    with pytest.raises(ValueError, match="conflicting overlap"):
        contract.collapse_identical_overlap_rows(
            source_row_index=np.asarray([0, 0]),
            pred_semantic_internal=np.asarray([2, 3]),
            pred_instance_id=np.asarray([7, 8]),
        )


def evaluation_payload() -> dict[str, np.ndarray]:
    return {
        "pred_tree_id": np.asarray([10, 10, 0, 20, 20, 30, 0, 0]),
        "target_tree_id": np.asarray([1, 1, 1, 2, 2, 0, 3, 3]),
        "classification": np.asarray([4, 4, 4, 5, 5, 2, 6, 6]),
        "pred_classification": np.asarray([4, 4, 0, 5, 5, 6, 0, 0]),
        "source_row_index": np.arange(8),
    }


def test_evaluator_uses_union_mask_and_maximum_matching() -> None:
    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate(evaluation_payload())
    )
    assert summary["protocol_id"] == "for_instance_pointwise_v1"
    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["f1"] == pytest.approx(2 / 3)
    assert {(row["pred_tree_id"], row["target_tree_id"]) for row in matches} == {
        (10, 1),
        (20, 2),
    }
    assert [row["pred_tree_id"] for row in unmatched_predictions] == [30]
    assert [row["target_tree_id"] for row in unmatched_references] == [3]


def test_evaluator_handles_all_background_and_noncontiguous_ids() -> None:
    payload = evaluation_payload()
    payload["pred_tree_id"] = np.zeros(8, dtype=np.int64)
    payload["pred_classification"] = np.zeros(8, dtype=np.int64)
    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate(payload)
    )
    assert summary["prediction_instance_count"] == 0
    assert summary["reference_instance_count"] == 3
    assert summary["f1"] == 0.0
    assert matches == []
    assert unmatched_predictions == []
    assert len(unmatched_references) == 3


def test_evaluator_rejects_bad_alignment_and_length() -> None:
    payload = evaluation_payload()
    payload["source_row_index"] = np.asarray([0, 1, 3, 2, 4, 5, 6, 7])
    with pytest.raises(ValueError, match="source_row_index"):
        evaluator.evaluate(payload)
    payload = evaluation_payload()
    payload["classification"] = payload["classification"][:-1]
    with pytest.raises(ValueError, match="mismatched"):
        evaluator.evaluate(payload)
    payload = evaluation_payload()
    del payload["pred_tree_id"]
    with pytest.raises(ValueError, match="missing fields"):
        evaluator.evaluate(payload)


def write_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.0, 1.0, 2.0])
    cloud.y = np.asarray([0.0, 0.0, 0.0])
    cloud.z = np.asarray([1.0, 2.0, 3.0])
    cloud.classification = np.asarray([2, 4, 5], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.asarray([0, 10, 11], dtype=np.int32)
    cloud.write(path)


def test_generated_las_sidecar_preserves_exact_rows(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text(
        "relative_path,split\nSYNTHETIC/plot.las,dev\n", encoding="utf-8"
    )
    metadata, arrays = sidecar.prepare(
        source=source,
        relative_path="SYNTHETIC/plot.las",
        split_metadata=split,
    )
    assert metadata["point_count"] == 3
    assert metadata["reference_tree_count"] == 2
    assert arrays["source_row_index"].tolist() == [0, 1, 2]
    assert arrays["x"].tolist() == [0.0, 1.0, 2.0]
    assert arrays["classification"].tolist() == [2, 4, 5]
    assert arrays["target_tree_id"].tolist() == [0, 10, 11]


def test_label_isolated_input_retains_every_row_and_hides_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text(
        "path,folder,split\nSYNTHETIC/plot.las,SYNTHETIC,dev\n",
        encoding="utf-8",
    )
    metadata, arrays = sidecar.prepare(
        source=source,
        relative_path="SYNTHETIC/plot.las",
        split_metadata=split,
    )
    assert metadata["split"] == "dev"
    assert arrays["source_row_index"].tolist() == [0, 1, 2]

    cloud = laspy.read(source)
    inference_ply = tmp_path / "input.ply"
    conversion = input_adapter.write_label_isolated_ply(inference_ply, cloud)
    vertex = PlyData.read(inference_ply)["vertex"].data
    assert len(vertex) == 3
    assert np.asarray(vertex["semantic_seg"]).tolist() == [1.0, 1.0, 1.0]
    assert np.asarray(vertex["treeID"]).tolist() == [-1.0, -1.0, -1.0]
    assert conversion["reference_classification_supplied_to_model"] is False
    assert conversion["reference_tree_id_supplied_to_model"] is False
    assert conversion["dropped_source_rows"] == 0


def write_prediction_ply(path: Path, source_ply: Path) -> None:
    source = PlyData.read(source_ply)["vertex"].data
    vertices = np.zeros(
        len(source),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("instance_preds", "i2"),
            ("semantic_preds", "i2"),
        ],
    )
    for name in ("x", "y", "z"):
        vertices[name] = source[name]
    vertices["instance_preds"] = np.asarray([-1, 0, 7], dtype=np.int16)
    vertices["semantic_preds"] = np.asarray([0, 2, 4], dtype=np.int16)
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(path)


def test_official_merge_extraction_uses_original_array_order(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    cloud = laspy.read(source)
    inference_ply = tmp_path / "input.ply"
    input_adapter.write_label_isolated_ply(inference_ply, cloud)
    merged = tmp_path / "merged.ply"
    write_prediction_ply(merged, inference_ply)
    arrays, metadata = merge_extractor.extract(merged, inference_ply, 3)
    assert arrays["source_row_index"].tolist() == [0, 1, 2]
    assert arrays["pred_instance_id"].tolist() == [-1, 0, 7]
    assert metadata["coordinate_matching_used"] is False
    assert metadata["coordinate_order_valid"] is True

    reordered = PlyData.read(merged)
    reordered["vertex"].data = reordered["vertex"].data[::-1].copy()
    reordered_path = tmp_path / "reordered.ply"
    reordered.write(reordered_path)
    with pytest.raises(ValueError, match="exact covered source-row order"):
        merge_extractor.extract(reordered_path, inference_ply, 3)


def test_official_merge_extraction_preserves_uncovered_source_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    cloud = laspy.read(source)
    inference_ply = tmp_path / "input.ply"
    input_adapter.write_label_isolated_ply(inference_ply, cloud)
    source_vertices = PlyData.read(inference_ply)["vertex"].data
    merged_vertices = np.zeros(
        2,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("instance_preds", "i2"),
            ("semantic_preds", "i2"),
        ],
    )
    for name in ("x", "y", "z"):
        merged_vertices[name] = source_vertices[name][[0, 2]]
    merged_vertices["instance_preds"] = np.asarray([4, 9], dtype=np.int16)
    merged_vertices["semantic_preds"] = np.asarray([2, 4], dtype=np.int16)
    merged = tmp_path / "merged.ply"
    PlyData(
        [PlyElement.describe(merged_vertices, "vertex")], byte_order="<"
    ).write(merged)
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    np.savetxt(tile_dir / "tile_0_0_indices.txt", [0, 2], fmt="%d")

    arrays, metadata = merge_extractor.extract(
        merged, inference_ply, 3, tile_dir
    )

    assert arrays["source_row_index"].tolist() == [0, 1, 2]
    assert arrays["pred_instance_id"].tolist() == [4, -1, 9]
    assert arrays["pred_semantic_internal"].tolist() == [2, -1, 4]
    assert metadata["covered_source_point_count"] == 2
    assert metadata["uncovered_source_point_count"] == 1
    assert metadata["uncovered_source_row_indices"] == [1]
    assert metadata["coordinate_matching_used"] is False


def test_sidecar_refuses_test_split_and_missing_fields(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text("relative_path,split\nSITE/plot.las,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="development-only"):
        sidecar.prepare(
            source=source,
            relative_path="SITE/plot.las",
            split_metadata=split,
        )


def test_shell_scripts_are_syntactically_valid() -> None:
    for path in sorted((METHOD / "slurm").iterdir()):
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True
        )
        assert completed.returncode == 0, (path, completed.stderr)


def test_retention_manifest_detects_missing_and_changed_files(tmp_path: Path) -> None:
    assert {
        "aligned_prediction_metadata",
        "input_conversion",
        "label_independence_probe",
        "merge_alignment",
        "raw_output_inventory",
    } <= retention.REQUIRED_SMOKE_ROLES
    role_paths = {}
    for index, role in enumerate(sorted(retention.REQUIRED_SMOKE_ROLES)):
        path = tmp_path / f"artifact_{index}.txt"
        path.write_text(f"{role}\n", encoding="utf-8")
        role_paths[role] = path
    manifest = retention.build(tmp_path, role_paths)
    retention.validate(tmp_path, manifest)
    role_paths["aligned_prediction"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        retention.validate(tmp_path, manifest)
    incomplete = dict(role_paths)
    del incomplete["plot_metrics"]
    with pytest.raises(ValueError, match="retention roles differ"):
        retention.build(tmp_path, incomplete)


def test_development_manifest_freezes_only_the_21_development_plots(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    split_metadata = dataset_root / "data_split_metadata.csv"
    rows = []
    for index in range(32):
        split = "dev" if index < 21 else "test"
        relative_path = f"SITE_{index % 3}/plot_{index:02d}.las"
        source = dataset_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        write_las(source)
        rows.append(
            {
                "path": relative_path,
                "folder": f"SITE_{index % 3}",
                "split": split,
            }
        )
    with split_metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "folder", "split"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "path": "ABSENT/not_in_local_catalogue.las",
                "folder": "ABSENT",
                "split": "dev",
            }
        )
    accepted_smoke = tmp_path / "accepted_smoke.json"
    accepted_smoke.write_text(
        json.dumps(
            {
                "schema": "forainet_accepted_development_smoke_v1",
                "status": "accepted",
                "run_id": development_manifest.EXPECTED_SMOKE_RUN_ID,
                "held_out_access": False,
            }
        ),
        encoding="utf-8",
    )
    plots, payload = development_manifest.build(
        dataset_root=dataset_root,
        split_metadata=split_metadata,
        accepted_smoke=accepted_smoke,
    )
    assert len(plots) == 21
    assert all(row["split"] == "dev" for row in plots)
    assert all(row["point_count"] == 3 for row in plots)
    assert payload["held_out_paths_included"] is False
    assert payload["split_metadata_row_count"] == 33
    assert payload["available_catalogue_count"] == 32
    assert payload["total_point_count"] == 63
    output_csv = tmp_path / "development.csv"
    output_json = tmp_path / "development.json"
    development_manifest.write_outputs(
        plots, payload, output_csv, output_json
    )
    resolved = development_task.resolve(output_csv, 20)
    assert resolved["relative_path"] == rows[20]["path"]
    with pytest.raises(ValueError, match="exactly once"):
        development_task.resolve(output_csv, 21)


def test_development_summary_requires_all_plots_and_hash_complete_retention(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest_root = run_root / "manifest"
    manifest_root.mkdir()
    manifest_rows = []
    benchmark_commit = "a" * 40
    recovery_commit = "c" * 40
    recovery_root = tmp_path / "recovery"
    run_id = (
        "forainet__for-instance__published-pretrained__none__development__"
        "20260723T210000"
    )
    for task_index in range(21):
        relative_path = f"SITE_{task_index % 3}/plot_{task_index:02d}.las"
        manifest_rows.append(
            {
                "task_index": task_index,
                "relative_path": relative_path,
                "split": "dev",
                "source_sha256": f"{task_index:064x}",
                "size_bytes": 100,
                "point_count": 10,
            }
        )
        plot_base = recovery_root if task_index in {16, 20} else run_root
        plot_root = plot_base / "plots" / f"task_{task_index:03d}"
        metrics_path = plot_root / "evaluation" / "metrics.json"
        plot_metadata_path = plot_root / "metadata" / "plot.json"
        metrics_path.parent.mkdir(parents=True)
        plot_metadata_path.parent.mkdir(parents=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "protocol_id": "for_instance_pointwise_v1",
                    "split": "dev",
                    "coordinate_matching": False,
                    "evaluated_point_count": 8,
                    "reference_instance_count": 3,
                    "prediction_instance_count": 3,
                    "true_positives": 2,
                    "false_positives": 1,
                    "false_negatives": 1,
                    "precision": 2 / 3,
                    "recall": 2 / 3,
                    "f1": 2 / 3,
                }
            ),
            encoding="utf-8",
        )
        plot_metadata_path.write_text(
            json.dumps(
                {
                    "route": "development",
                    "benchmark_commit": (
                        recovery_commit
                        if task_index in {16, 20}
                        else benchmark_commit
                    ),
                    "relative_path": relative_path,
                    "reference_labels_supplied_to_model": False,
                    "point_count": 10,
                    "wall_runtime_seconds": 5.0,
                    "peak_child_rss_kb": 1000,
                    "aligned_prediction_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        role_paths = {}
        for role in sorted(retention.REQUIRED_SMOKE_ROLES):
            if role == "plot_metrics":
                role_paths[role] = metrics_path
            elif role == "plot_metadata":
                role_paths[role] = plot_metadata_path
            else:
                path = plot_root / "retained" / f"{role}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{role}\n", encoding="utf-8")
                role_paths[role] = path
        retention_payload = retention.build(plot_root, role_paths)
        retention_path = plot_root / "retention" / "manifest.json"
        retention_path.parent.mkdir()
        retention_path.write_text(
            json.dumps(retention_payload), encoding="utf-8"
        )
        (plot_root / "final_gate.json").write_text(
            json.dumps(
                {
                    "schema": "forainet_development_plot_final_gate_v1",
                    "status": "complete",
                    "held_out_access": False,
                    "relative_path": relative_path,
                    "development_task_index": task_index,
                    "retention_manifest_sha256": retention.sha256(
                        retention_path
                    ),
                }
            ),
            encoding="utf-8",
        )
    manifest_csv = manifest_root / "development.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_json = manifest_root / "development.json"
    manifest_json.write_text(
        json.dumps(
            {
                "schema": "forainet_development_manifest_v1",
                "status": "complete",
                "expected_plot_count": 21,
                "held_out_paths_included": False,
            }
        ),
        encoding="utf-8",
    )
    payload = development_summary.summarise(
        run_root,
        run_id,
        manifest_csv,
        manifest_json,
        benchmark_commit,
        recovery_root,
        recovery_commit,
    )
    assert payload["completed_plots"] == 21
    assert payload["overall"]["true_positives"] == 42
    assert payload["overall"]["false_positives"] == 21
    assert payload["overall"]["false_negatives"] == 21
    assert payload["overall"]["f1"] == pytest.approx(2 / 3)
    assert payload["recovered_task_indices"] == [16, 20]
    assert payload["implementation_commits"] == [
        benchmark_commit,
        recovery_commit,
    ]
    final_gate = json.loads((run_root / "final_gate.json").read_text())
    assert final_gate["held_out_access"] is False
    assert final_gate["completed_plots"] == 21
    public_root = tmp_path / "public"
    provenance = development_export.export(run_root, public_root)
    assert provenance["run_id"] == run_id
    assert provenance["held_out_access"] is False
    assert provenance["ranking_eligible"] is False
    assert {
        path.name for path in public_root.iterdir() if path.is_file()
    } == {
        "forainet_development_plot_results.csv",
        "forainet_development_site_results.csv",
        "forainet_development_results.json",
        "forainet_development_provenance.json",
    }


def test_development_export_rejects_private_markers(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        json.dumps({"run_root": "/users/private/fastscratch/result"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="private marker"):
        development_export.ensure_public_text(unsafe)


def test_no_test_submission_route_exists() -> None:
    names = {path.name for path in (METHOD / "slurm").iterdir()}
    assert not any("test" in name for name in names)
    published = yaml.safe_load(
        (METHOD / "configs/for_instance_published_test.yml").read_text(
            encoding="utf-8"
        )
    )
    assert published["gates"]["current_authorisation"] is False


def test_public_files_do_not_contain_private_paths() -> None:
    forbidden = ("/users/", "/cluster/", "/mnt/")
    for path in METHOD.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_cli_help_is_available() -> None:
    scripts = [
        METHOD / "scripts/data/prepare_alignment_sidecar.py",
        METHOD / "scripts/data/prepare_label_isolated_input.py",
        METHOD / "scripts/runtime/extract_official_merge.py",
        METHOD / "scripts/runtime/normalise_forainet_predictions.py",
        METHOD / "scripts/evaluation/evaluate_for_instance.py",
        METHOD / "scripts/provenance/validate_exposure_audit.py",
        METHOD / "scripts/provenance/verify_forainet_assets.py",
        METHOD / "scripts/provenance/probe_checkpoint_load.py",
        METHOD / "scripts/provenance/build_retention_manifest.py",
        METHOD / "scripts/provenance/build_alignment_review.py",
        METHOD / "scripts/provenance/prepare_development_manifest.py",
        METHOD / "scripts/provenance/resolve_development_task.py",
        METHOD / "scripts/provenance/summarise_development_run.py",
        METHOD / "scripts/provenance/stage_finetune_training_config.py",
    ]
    for path in scripts:
        completed = subprocess.run(
            [sys.executable, str(path), "--help"], capture_output=True, text=True
        )
        assert completed.returncode == 0, (path, completed.stderr)


def test_alignment_review_uses_exact_source_rows_and_writes_local_figure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(
        prediction,
        classification=np.asarray([1, 4, 5], dtype=np.int16),
        pred_classification=np.asarray([0, 4, 6], dtype=np.int16),
        pred_tree_id=np.asarray([0, 3, 4], dtype=np.int32),
        source_row_index=np.arange(3, dtype=np.int64),
        target_tree_id=np.asarray([0, 10, 11], dtype=np.int32),
    )
    figure = tmp_path / "review.png"
    report = tmp_path / "review.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(METHOD / "scripts/provenance/build_alignment_review.py"),
            "--source-las",
            str(source),
            "--prediction-npz",
            str(prediction),
            "--relative-path",
            "SYNTHETIC/plot.las",
            "--output-png",
            str(figure),
            "--output-json",
            str(report),
            "--maximum-union-points",
            "3",
            "--maximum-context-points",
            "3",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")},
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "waiting_manual_confirmation"
    assert payload["source_row_index_exact"] is True
    assert payload["coordinate_matching"] is False
    assert payload["point_count"] == 3
    assert payload["reference_tree_count"] == 2
    assert payload["predicted_tree_count"] == 2
    assert figure.stat().st_size > 0


def test_evaluator_cli_writes_required_tables(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, **evaluation_payload())
    outputs = {
        "metrics": tmp_path / "metrics.json",
        "matches": tmp_path / "matches.csv",
        "unmatched_predictions": tmp_path / "unmatched_predictions.csv",
        "unmatched_references": tmp_path / "unmatched_references.csv",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(METHOD / "scripts/evaluation/evaluate_for_instance.py"),
            "--prediction-npz",
            str(prediction),
            "--metrics-json",
            str(outputs["metrics"]),
            "--matches-csv",
            str(outputs["matches"]),
            "--unmatched-predictions-csv",
            str(outputs["unmatched_predictions"]),
            "--unmatched-references-csv",
            str(outputs["unmatched_references"]),
            "--split",
            "dev",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(outputs["metrics"].read_text())["true_positives"] == 2
    assert all(path.is_file() for path in outputs.values())
