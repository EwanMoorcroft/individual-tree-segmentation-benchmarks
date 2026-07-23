from __future__ import annotations

import subprocess
from pathlib import Path


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
    assert "no ForestFormer3D accuracy result exists" in text


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
    assert 'sbatch --help 2>&1 | grep -q -- "--kill-on-invalid-dep"' in submit
    assert '--dependency="afterok:$BUILD_JOB"' in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert "monitor_workflow.sh" in submit
    assert 'FF3D_MONITOR_SECONDS:-30' in submit
    assert "scancel $BUILD_JOB $VALIDATE_JOB" in submit


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


def test_build_and_validation_jobs_freeze_identity_and_dependency_outputs() -> None:
    build = (METHOD / "slurm/build_environment.sbatch").read_text(encoding="utf-8")
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

    assert "#SBATCH --partition=gpu-a100-lowbig" in validate
    assert "#SBATCH --gres=gpu:a100:1" in validate
    assert "validate_environment.py" in validate
    assert "--require-cuda" in validate
    assert "environment_validation.complete" in validate
    assert "epoch_3000_fix.pth:ro" in validate
    assert "--environment-root" in validate
    assert "FF3D_BASE_SIF_SHA256" in validate


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
