from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
RUNBOOK = ROOT / "methods/tls2trees/docs/for_instance_published_default_smoke.md"
PRIVATE_MAC_ROOT = "/" + "Users" + "/"

STAGES = (
    "inventory_published_default_dev_smoke.sbatch",
    "convert_published_default_dev_smoke.sbatch",
    "semantic_published_default_dev_smoke.sbatch",
    "instance_published_default_dev_smoke.sbatch",
    "adapt_published_default_dev_smoke.sbatch",
    "evaluate_published_default_dev_smoke.sbatch",
    "gate_published_default_dev_smoke.sbatch",
    "summarise_published_default_dev_smoke.sbatch",
)


def text(name: str) -> str:
    return (SLURM / name).read_text(encoding="utf-8")


def test_smoke_shell_files_have_strict_syntax_and_public_paths() -> None:
    paths = [
        SLURM / "setup_tls2trees_environment.sbatch",
        SLURM / "published_default_dev_smoke_common.sh",
        SLURM / "submit_published_default_dev_smoke.sh",
        SLURM / "resume_published_default_dev_smoke_from_instance.sh",
        SLURM / "monitor_published_default_dev_smoke.sh",
        *(SLURM / name for name in STAGES),
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in source or path.name.endswith("_common.sh")
        assert PRIVATE_MAC_ROOT not in source
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_submission_is_one_plot_published_default_development_only() -> None:
    submit = text("submit_published_default_dev_smoke.sh")
    assert 'TLS2TREES_DEV_SMOKE_CONFIRMED:-0' in submit
    assert 'TLS2TREES_REQUESTED_VARIANT:-published_default' in submit
    assert 'TLS2TREES_REQUESTED_SPLIT:-development' in submit
    assert "development_tuned" not in submit
    assert "--split test" not in submit
    assert "--allow-held-out-test" not in submit
    assert "--array=" not in submit
    assert 'STAGE0_INDEX=0' in submit
    assert "resolve-stage0" in text("published_default_dev_smoke_common.sh")
    assert "run_gate.runnable" in submit

    assert "ca12cb73b2c736d80b020e8025f8d975d42e6f01" in submit
    assert "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b" in submit
    assert 'git -C "$UPSTREAM_REPO" status --porcelain' in submit
    assert "TLS2TREES_SMOKE_MIN_FREE_BYTES" in submit
    assert '$HOME/fastscratch/venvs/tls2trees' in submit
    assert ".tls2trees_setup_complete.json" in submit
    assert "TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256" in submit
    assert "validate_tls2trees_environment.py" in submit
    assert "--setup-marker-json" in submit
    assert "--skip-model-load" in submit


def test_environment_setup_is_isolated_guarded_and_gpu_validated() -> None:
    setup = text("setup_tls2trees_environment.sbatch")
    assert "TLS2TREES_SETUP_CONFIRMED:-0" in setup
    assert "TLS2TREES_SETUP_RESUME_PARTIAL" in setup
    assert "TLS2TREES_SETUP_VALIDATE_PARTIAL_ONLY" in setup
    assert "validating_existing_partial_tls2trees_env" in setup
    assert "network_install_steps=skipped" in setup
    assert "Validation-only recovery requires an existing partial Conda prefix" in setup
    assert "#SBATCH --partition=gpu-l40s-low" in setup
    assert "#SBATCH --gres=gpu:1" in setup
    assert "#SBATCH --no-requeue" in setup
    assert '$HOME/fastscratch/venvs/tls2trees' in setup
    assert '$HOME/fastscratch/venvs/treebench' not in setup
    assert 'python=3.9' in setup
    assert 'cudatoolkit=11.1.1=h6406543_8' in setup
    assert 'torch==1.9.0+cu111' in setup
    assert 'torch-geometric==1.7.2' in setup
    assert 'torch-cluster==1.5.9' in setup
    assert 'torch-scatter==2.0.8' in setup
    assert 'torch-sparse==0.6.11' in setup
    assert 'torch-spline-conv==1.2.1' in setup
    assert "https://download.pytorch.org/whl/torch_stable.html" in setup
    assert "https://data.pyg.org/whl/torch-1.9.0+cu111.html" in setup
    assert "TLS2TREES_CONDA_CHANNEL" in setup
    assert "https://conda.anaconda.org/conda-forge" in setup
    assert "https://prefix.dev/conda-forge" in setup
    assert "--override-channels" in setup
    assert "CONDA_NUMBER_CHANNEL_NOTICES=0" in setup
    assert "conda_channel_preflight=passed" in setup
    assert 'CONDA_CHANNEL_RECORD="${METHOD_ENV}.tls2trees_conda_channel"' in setup
    assert "tls2trees_load_recorded_conda_channel" in setup
    assert "does not match the prefix record" in setup
    assert setup.index("tls2trees_preflight_conda_channel\n") < setup.index(
        "PREFIX_MODIFIED=1", setup.index('if [[ "$INSTALL_ENV" == "1" ]]')
    )
    assert setup.index(
        'mv "$CONDA_CHANNEL_RECORD.partial" "$CONDA_CHANNEL_RECORD"'
    ) < setup.index(
        "PREFIX_MODIFIED=1", setup.index('if [[ "$INSTALL_ENV" == "1" ]]')
    )
    assert "validate_tls2trees_environment.py" in setup
    assert "--require-cuda" in setup
    assert ".tls2trees_setup_complete.json" in setup
    assert "PREFIX_MODIFIED=0" in setup
    assert '"$PREFIX_MODIFIED" == "1"' in setup
    assert "conda-explicit.txt" in setup
    assert "conda-channel.txt" in setup
    assert "cuda-runtime.txt" in setup
    assert "pip-freeze.txt" in setup
    assert "benchmark-commit.txt" in setup
    assert "No benchmark job was submitted." in setup
    common = text("published_default_dev_smoke_common.sh")
    assert "TLS2TREES_EXPECTED_METHOD_ENV_MARKER_SHA256" in common
    assert "environment marker changed after submission" in common
    assert 'conda activate "$TLS2TREES_METHOD_ENV"' in common
    assert 'LD_LIBRARY_PATH="$TLS2TREES_METHOD_ENV/lib' in common
    assert 'source "$TLS2TREES_METHOD_ENV/bin/activate"' not in common
    submit = text("submit_published_default_dev_smoke.sh")
    assert 'LD_LIBRARY_PATH="$METHOD_ENV/lib' in submit


def test_submission_dependency_chain_includes_alignment_and_both_targets() -> None:
    submit = text("submit_published_default_dev_smoke.sh")
    markers = [
        "INVENTORY_JOB=$(sbatch",
        "CONVERT_JOB=$(sbatch",
        "SEMANTIC_JOB=$(sbatch",
        "INSTANCE_JOB=$(sbatch",
        "ADAPTER_JOB=$(sbatch",
        "LEAF_OFF_EVALUATE_JOB=$(sbatch",
        "LEAF_ON_EVALUATE_JOB=$(sbatch",
        "GATE_JOB=$(sbatch",
        "SUMMARY_JOB=$(sbatch",
    ]
    offsets = [submit.index(marker) for marker in markers]
    assert offsets == sorted(offsets)
    assert 'afterok:$INSTANCE_JOB' in submit
    assert submit.count('afterok:$ADAPTER_JOB') == 2
    assert 'afterok:$LEAF_OFF_EVALUATE_JOB:$LEAF_ON_EVALUATE_JOB' in submit
    assert 'afterok:$GATE_JOB' in submit
    assert submit.count("--kill-on-invalid-dep=yes") == 8
    assert "TLS2TREES_TARGET=leaf_off" in submit
    assert "TLS2TREES_TARGET=leaf_on" in submit
    for variable in (
        "MANIFEST_JSON",
        "STAGE0_INDEX",
        "RUN_ID",
        "OUTPUT_ROOT",
        "TLS2TREES_REPO",
    ):
        assert f"{variable}=" in submit


def test_instance_recovery_reuses_semantic_and_preserves_failed_attempt() -> None:
    recovery = text("resume_published_default_dev_smoke_from_instance.sh")
    instance = text("instance_published_default_dev_smoke.sbatch")
    assert "TLS2TREES_INSTANCE_RECOVERY_CONFIRMED" in recovery
    assert "OLD_INSTANCE_STATE" in recovery
    assert 'if [[ "$OLD_INSTANCE_STATE" != "FAILED" ]]' in recovery
    assert "find \"$FAILED_RAW_ROOT\" -type f" in recovery
    assert "SEMANTIC_JOB=$(sbatch" not in recovery
    assert "INSTANCE_JOB=$(sbatch" in recovery
    assert "afterok:$INSTANCE_JOB" in recovery
    assert "instance_recovery_chain_submitted" in recovery
    assert "TLS2TREES_INSTANCE_RECOVERY_CONFIRMED" in instance
    assert "--resume-failed-empty-output" in instance


def test_stage_resources_and_scientific_gates_are_explicit() -> None:
    semantic = text("semantic_published_default_dev_smoke.sbatch")
    instance = text("instance_published_default_dev_smoke.sbatch")
    adapter = text("adapt_published_default_dev_smoke.sbatch")
    evaluation = text("evaluate_published_default_dev_smoke.sbatch")
    gate = text("gate_published_default_dev_smoke.sbatch")

    assert "#SBATCH --partition=gpu-l40s-low" in semantic
    assert "#SBATCH --gres=gpu:1" in semantic
    assert "#SBATCH --no-requeue" in semantic
    assert "#SBATCH --cpus-per-task=10" in semantic
    assert "--require-cuda" in semantic
    assert "--setup-marker-json" in semantic
    assert "TLS2TREES_ENV_VALIDATOR" in semantic
    assert "#SBATCH --mem=96G" in instance
    assert "TLS2TREES_ENV_VALIDATOR" in instance
    assert '--tls2trees-repo "$TLS2TREES_REPO"' in instance
    assert "--setup-marker-json" in instance
    assert "--coordinate-tolerance-m 0.001" in adapter
    assert "source_row_predictions.npz" in adapter
    assert "alignment_metadata.json" in adapter
    assert "--aligned-predictions-npz" in evaluation
    assert "--split dev" in evaluation
    assert 'TARGET" != "leaf_off"' in evaluation
    assert 'TARGET" != "leaf_on"' in evaluation
    assert "validate_for_instance_tls2trees_smoke.py" in text(
        "published_default_dev_smoke_common.sh"
    )
    assert 'p.get("status") == "passed_automated_gates"' in gate
    assert 'p.get("manual_alignment_review_required") is True' in gate
    assert 'p.get("full_development_authorised") is False' in gate
    assert 'p.get("held_out_test_authorised") is False' in gate
    assert 'prediction_instance_count' in gate


def test_monitor_and_runbook_stop_at_manual_development_gate() -> None:
    monitor = text("monitor_published_default_dev_smoke.sh")
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "squeue -j" in monitor
    assert "sacct -X -j" in monitor
    assert "TLS2TREES_SMOKE_ADAPTER_JOB" in monitor
    assert "No tuning, full-development array or held-out-test job" in monitor
    assert "exit 0" not in monitor
    assert "TLS2TREES_DEV_SMOKE_CONFIRMED=1" in runbook
    assert "submit_published_default_dev_smoke.sh" in runbook
    assert "monitor_published_default_dev_smoke.sh" in runbook
    assert "manual review gate" in runbook
    assert "does not authorise" in runbook
    assert '${TLS2TREES_METHOD_ENV:-$HOME/fastscratch/venvs/tls2trees}/bin/python' in runbook
    assert "setup_tls2trees_environment.sbatch" in runbook
    assert "TLS2TREES_ENVIRONMENT_SETUP_VALIDATED" in runbook
    assert PRIVATE_MAC_ROOT not in runbook
