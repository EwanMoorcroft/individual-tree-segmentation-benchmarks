from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
CONFIG = ROOT / "methods/tls2trees/configs/for_instance_development_tuned_stage1.yml"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage1_candidates_are_exact_probe_promotions_with_both_targets() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    required = payload["probe_promotion"]["required_viable_candidate_ids"]
    candidates = payload["candidates"]
    assert payload["project"]["status"] == "stage1_candidates_frozen_execution_ready"
    assert payload["dataset"]["allowed_split"] == "development"
    assert payload["dataset"]["stage0_plot_count"] == 5
    assert payload["scope"]["targets"] == ["leaf_off", "leaf_on"]
    assert payload["scope"]["development_accuracy_metrics_permitted"] is True
    assert payload["scope"]["selection_uses_held_out_test_metrics"] is False
    assert payload["scope"]["held_out_test_accessed"] is False
    assert payload["scope"]["final_configuration_selected"] is False
    assert [candidate["candidate_id"] for candidate in candidates] == required
    assert [candidate["candidate_index"] for candidate in candidates] == list(range(4))
    assert all(candidate["parameters"]["find_stems_min_points"] == 50 for candidate in candidates)
    assert all(candidate["parameters"]["add_leaves"] is True for candidate in candidates)
    assert payload["run_gate"]["stage1_task_count"] == 20
    assert payload["run_gate"]["full_development_runnable"] is False
    assert payload["run_gate"]["held_out_test_runnable"] is False


def test_generic_development_context_is_explicitly_variant_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = load_script(
        RUNTIME / "for_instance_published_common.py", "stage1_common_context"
    )
    row = {
        "task_index": 4,
        "safe_plot_id": "NIBIO_plot_10_annotated",
        "relative_path": "NIBIO/plot_10_annotated.las",
    }
    monkeypatch.setattr(common, "load_and_verify_manifest_plot", lambda *a, **k: ({}, row))
    plot_root, resolved = common.resolve_development_plot_context(
        manifest_path=tmp_path / "manifest.json",
        task_index=4,
        output_root=tmp_path,
        run_id="stage1_run",
        variant="development_tuned",
        allowed_variants={"development_tuned"},
    )
    assert resolved == row
    assert plot_root == (
        tmp_path
        / "tls2trees/for_instance/development_tuned/development/stage1_run/NIBIO_plot_10_annotated"
    )
    with pytest.raises(ValueError, match="not permitted"):
        common.resolve_development_plot_context(
            manifest_path=tmp_path / "manifest.json",
            task_index=4,
            output_root=tmp_path,
            run_id="stage1_run",
            variant="test_variant",
            allowed_variants={"development_tuned"},
        )


def test_stage1_runner_requires_exact_completed_probe_evidence(tmp_path: Path) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        "tls2trees_stage1_candidate_runner",
    )
    config, resolved = runner.load_stage1_config(str(CONFIG))
    assert resolved == CONFIG
    required = config["probe_promotion"]["required_viable_candidate_ids"]
    summary = tmp_path / "probe.json"
    summary.write_text(
        json.dumps(
            {
                "status": "viable_candidates_found",
                "viable_candidate_ids": required,
                "held_out_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    evidence = runner.verify_probe_evidence(summary, runner.sha256(summary), required)
    assert evidence["viable_candidate_ids"] == required
    with pytest.raises(ValueError, match="differ"):
        runner.verify_probe_evidence(summary, runner.sha256(summary), required[:-1])


def test_small_graph_recovery_archives_failed_attempt_without_deleting_it(
    tmp_path: Path,
) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        "tls2trees_stage1_small_graph_archive",
    )
    plot_root = tmp_path / "RMIT_train"
    raw = plot_root / "predictions/raw"
    logs = plot_root / "logs/instance"
    metadata = plot_root / "metadata/instance_run.json"
    raw.mkdir(parents=True)
    logs.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    (raw / "partial.leafoff.ply").write_text("retained partial output", encoding="utf-8")
    (logs / "tile_000000.stderr.log").write_text(
        "ValueError: Expected n_neighbors <= n_samples, n_samples = 144, n_neighbors = 201\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: Instance tile 0 failed with return code 1",
            }
        ),
        encoding="utf-8",
    )
    evidence = runner.archive_failed_small_graph_attempt(plot_root)
    archive = plot_root / "recovery/instance_small_graph_attempt_1"
    assert evidence["status"] == "failed_small_graph_attempt_archived"
    assert (archive / "instance_run.json").is_file()
    assert (archive / "logs/tile_000000.stderr.log").is_file()
    assert (archive / "raw/partial.leafoff.ply").is_file()
    assert not raw.exists()
    assert not metadata.exists()


def test_repeated_audited_failure_uses_next_archive_attempt(tmp_path: Path) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        "tls2trees_repeated_empty_stem_archive",
    )
    plot_root = tmp_path / "NIBIO_plot_16"
    prior = plot_root / "recovery/instance_empty_in_tile_stems_attempt_1"
    prior.mkdir(parents=True)
    raw = plot_root / "predictions/raw"
    logs = plot_root / "logs/instance"
    metadata = plot_root / "metadata/instance_run.json"
    raw.mkdir(parents=True)
    logs.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    (logs / "tile_000001.stderr.log").write_text(
        "ValueError: cannot set a frame with no defined index and a scalar\n",
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: Instance tile 1 failed with return code 1",
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.archive_failed_audited_instance_attempt(plot_root)

    assert evidence["attempt"] == 2
    assert evidence["archive_root"].endswith(
        "instance_empty_in_tile_stems_attempt_2"
    )
    assert prior.is_dir()
    assert not raw.exists()
    assert not metadata.exists()


@pytest.mark.parametrize(
    ("signature", "kind"),
    (
        (
            "RuntimeError: Cannot restore clstr: groupby.apply did not return a grouped index\n",
            "empty_groupby",
        ),
        (
            "ValueError: cannot set a frame with no defined index and a scalar\n",
            "empty_in_tile_stems",
        ),
    ),
)
def test_empty_prediction_recovery_archives_only_audited_failures(
    tmp_path: Path,
    signature: str,
    kind: str,
) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        f"tls2trees_{kind}_archive",
    )
    plot_root = tmp_path / kind
    raw = plot_root / "predictions/raw"
    logs = plot_root / "logs/instance"
    metadata = plot_root / "metadata/instance_run.json"
    raw.mkdir(parents=True)
    logs.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    (logs / "tile_000000.stderr.log").write_text(signature, encoding="utf-8")
    metadata.write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: Instance tile 0 failed with return code 1",
            }
        ),
        encoding="utf-8",
    )

    evidence = runner.archive_failed_audited_instance_attempt(plot_root)

    archive = plot_root / f"recovery/instance_{kind}_attempt_1"
    assert evidence["failure_kind"] == kind
    assert evidence["status"] == f"failed_{kind}_attempt_archived"
    assert (archive / "instance_run.json").is_file()
    assert (archive / "logs/tile_000000.stderr.log").is_file()


def test_stage1_summary_aggregates_all_five_sites_and_both_targets(tmp_path: Path) -> None:
    summariser = load_script(
        ROOT / "methods/tls2trees/scripts/evaluation/summarise_tls2trees_development_stage1.py",
        "tls2trees_stage1_summary",
    )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    collections = ["CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"]
    plots = []
    selection = []
    for index, collection in enumerate(collections):
        plots.append(
            {
                "task_index": index,
                "safe_plot_id": f"{collection}_plot",
                "relative_path": f"{collection}/plot.las",
                "collection": collection,
            }
        )
        selection.append({"stage0_index": index, "task_index": index})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_split": "development",
                "plots": plots,
                "stage0_selection": selection,
            }
        ),
        encoding="utf-8",
    )
    required = config["probe_promotion"]["required_viable_candidate_ids"]
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "status": "viable_candidates_found",
                "viable_candidate_ids": required,
                "held_out_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "predictions"
    workflow_run_id = "tls2trees_for-instance_development_tuned_stage1_20260718_150000"
    for candidate_index, candidate_id in enumerate(required):
        for plot in plots:
            plot_root = (
                output_root
                / "tls2trees/for_instance/development_tuned/development"
                / f"{workflow_run_id}__{candidate_id}"
                / plot["safe_plot_id"]
            )
            metadata = plot_root / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "instance_run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "candidate_id": candidate_id,
                        "workflow_run_id": workflow_run_id,
                        "stage1_config_sha256": summariser.sha256(CONFIG),
                        "probe_summary_sha256": summariser.sha256(probe_path),
                        "held_out_test_accessed": False,
                        "runtime_seconds": 10,
                        "peak_rss_gb": 1,
                    }
                ),
                encoding="utf-8",
            )
            (metadata / "adapter_run.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "variant": "development_tuned",
                        "split": "development",
                        "held_out_test_accessed": False,
                        "runtime_seconds": 2,
                    }
                ),
                encoding="utf-8",
            )
            for target in ("leaf_off", "leaf_on"):
                metric_path = plot_root / "evaluation" / target / "plot_metrics.json"
                metric_path.parent.mkdir(parents=True)
                tp = candidate_index + 1
                metric_path.write_text(
                    json.dumps(
                        {
                            "status": "evaluated",
                            "safe_for_scoring": True,
                            "split": "dev",
                            "target": target,
                            "prediction_instance_count": tp + 1,
                            "reference_instance_count": tp + 2,
                            "true_positives": tp,
                            "false_positives": 1,
                            "false_negatives": 2,
                            "precision": tp / (tp + 1),
                            "recall": tp / (tp + 2),
                            "f1": 2 * tp / (2 * tp + 3),
                            "mean_matched_iou": 0.6,
                        }
                    ),
                    encoding="utf-8",
                )
    payload = summariser.summarise(
        output_root=output_root,
        workflow_run_id=workflow_run_id,
        manifest_path=manifest_path,
        stage1_config_path=CONFIG,
        probe_summary_path=probe_path,
        probe_summary_sha256=summariser.sha256(probe_path),
    )
    assert payload["status"] == "stage1_completed"
    assert payload["valid_metric_count"] == 40
    assert len(payload["plot_metrics"]) == 40
    assert len(payload["aggregates"]) == 8
    assert payload["held_out_test_accessed"] is False
    assert payload["final_configuration_selected"] is False
    best = required[-1]
    assert payload["candidate_rankings_for_review"]["leaf_off"][0] == best
    aggregate = next(
        row for row in payload["aggregates"] if row["candidate_id"] == best and row["target"] == "leaf_on"
    )
    assert aggregate["evaluated_plot_count"] == 5
    assert aggregate["true_positives"] == 20
    assert aggregate["micro_f1"] == pytest.approx(40 / 55)


def test_stage1_slurm_chain_is_guarded_bounded_and_executable() -> None:
    names = [
        "prepare_semantic_development_stage1.sbatch",
        "evaluate_development_stage1_candidate.sbatch",
        "summarise_development_stage1.sbatch",
        "submit_development_stage1.sh",
        "monitor_development_stage1.sh",
        "resume_development_stage1_small_graph.sh",
    ]
    sources = {}
    for name in names:
        path = SLURM / name
        source = path.read_text(encoding="utf-8")
        sources[name] = source
        assert "set -euo pipefail" in source
        assert "/Users/" not in source
        completed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    for name in (
        "submit_development_stage1.sh",
        "monitor_development_stage1.sh",
        "resume_development_stage1_small_graph.sh",
    ):
        assert (SLURM / name).stat().st_mode & 0o111
    submit = sources["submit_development_stage1.sh"]
    semantic = sources["prepare_semantic_development_stage1.sbatch"]
    candidate = sources["evaluate_development_stage1_candidate.sbatch"]
    monitor = sources["monitor_development_stage1.sh"]
    recovery = sources["resume_development_stage1_small_graph.sh"]
    assert "TLS2TREES_STAGE1_CONFIRMED" in submit
    assert '--array="0-4%2"' in submit
    assert '--array="0-19%4"' in submit
    assert 'afterok:$SEMANTIC_JOB' in submit
    assert 'afterany:$CANDIDATE_JOB' in submit
    assert "#SBATCH --partition=gpu-l40s-low" in semantic
    assert "--require-cuda" in semantic
    assert "--variant development_tuned" in semantic + candidate
    assert candidate.count("--split dev") == 2
    assert "for TARGET in leaf_off leaf_on" in candidate
    assert "full-development" not in submit
    assert "--split test" not in submit + semantic + candidate
    assert "final_configuration_selected=false" in submit + monitor
    assert "held_out_test_accessed=false" in submit + semantic + candidate + monitor
    assert "TLS2TREES_STAGE1_RECOVERY_CONFIRMED" in recovery
    assert "--format=JobID%30,State" in recovery
    assert (
        '-j "$OLD_CANDIDATE_JOB" \\\n  --format=JobIDRaw,State'
        not in recovery
    )
    assert "Expected n_neighbors <= n_samples" in recovery
    assert "Raw prediction ownership is not unique" in recovery
    assert "TLS2TREES_STAGE1_RECOVERY=1" in recovery
    assert 'RECOVERY_MODE="adapter_ownership"' in candidate
    assert 'RECOVERY_MODE="small_graph_instance"' in candidate
    assert "adapter_ownership_attempt_1" in candidate + recovery
    assert "PARTIAL_LEAF_OFF_METADATA" in candidate + recovery
    assert 'afterany:$RECOVERY_JOB' in recovery
    assert "--array=\"$FAILED_LIST%2\"" in recovery
    assert "completed_tasks_and_semantic_cache_reused=true" in recovery
    assert "--split test" not in recovery
