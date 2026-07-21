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
CONFIGS = ROOT / "methods/tls2trees/configs"
EVALUATION = ROOT / "methods/tls2trees/scripts/evaluation"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"


def load_module(name: str) -> ModuleType:
    path = EVALUATION / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def aggregate_record(candidate_id: str, target: str, runtime: float) -> dict:
    leader = candidate_id == "p04_min_points_50_lower_band"
    return {
        "candidate_id": candidate_id,
        "target": target,
        "expected_plot_count": 5,
        "evaluated_plot_count": 5,
        "failed_or_invalid_plot_count": 0,
        "true_positives": 1 if target == "leaf_off" else 3,
        "false_positives": (4 if target == "leaf_off" else 2) if leader else 6,
        "false_negatives": 328 if target == "leaf_off" else 326,
        "micro_precision": 0.2 if leader else 1 / 7,
        "micro_recall": 1 / 329 if target == "leaf_off" else 3 / 329,
        "micro_f1": (0.005988 if target == "leaf_off" else 0.017964)
        if leader
        else (0.005952 if target == "leaf_off" else 0.017857),
        "mean_plot_f1": 0.002484 if target == "leaf_off" else 0.133333,
        "total_instance_runtime_seconds": runtime,
        "maximum_instance_peak_rss_gb": 1.1 if leader else 3.2,
        "per_site_f1": {"CULS": 0.0},
    }


def write_stage1_summary(path: Path) -> None:
    candidate_ids = [
        "p02_min_points_50",
        "p03_min_points_50_radius_015",
        "p04_min_points_50_lower_band",
        "p05_min_points_50_graph_3_gap_5",
    ]
    aggregates = [
        aggregate_record(candidate_id, target, 1583.4 if candidate_id.startswith("p04") else 4988.8)
        for candidate_id in candidate_ids
        for target in ("leaf_off", "leaf_on")
    ]
    plots = [
        {
            "safe_plot_id": f"plot_{index}",
            "relative_path": f"SITE{index}/plot.las",
            "collection": collection,
        }
        for index, collection in enumerate(("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"))
    ]
    path.write_text(
        json.dumps(
            {
                "status": "stage1_completed",
                "valid_metric_count": 40,
                "held_out_test_accessed": False,
                "final_configuration_selected": False,
                "workflow_run_id": "stage1-run",
                "candidate_rankings_for_review": {
                    "leaf_off": [
                        "p04_min_points_50_lower_band",
                        "p02_min_points_50",
                    ],
                    "leaf_on": [
                        "p04_min_points_50_lower_band",
                        "p02_min_points_50",
                    ],
                },
                "aggregates": aggregates,
                "plot_metrics": plots,
            }
        ),
        encoding="utf-8",
    )


def test_stage2_config_freezes_two_development_candidates() -> None:
    payload = yaml.safe_load(
        (CONFIGS / "for_instance_development_tuned_stage2.yml").read_text()
    )
    selected = payload["selection"]["selected_candidates"]
    assert [item["candidate_id"] for item in selected] == [
        "p04_min_points_50_lower_band",
        "p02_min_points_50",
    ]
    assert all(item["targets"] == ["leaf_off", "leaf_on"] for item in selected)
    assert payload["dataset"]["exact_plot_count"] == 21
    assert payload["execution"]["candidate_plot_task_count"] == 42
    assert payload["execution"]["target_metric_count"] == 84
    assert payload["scope"]["held_out_test_accessed"] is False
    assert payload["scope"]["final_configuration_selected"] is False
    assert payload["run_gate"]["held_out_test_runnable"] is False


def test_freeze_manifest_binds_complete_stage1_evidence(tmp_path: Path) -> None:
    module = load_module("freeze_tls2trees_development_stage2_candidates.py")
    summary = tmp_path / "stage1.json"
    write_stage1_summary(summary)
    payload = module.freeze(
        stage1_summary_path=summary,
        stage1_config_path=CONFIGS / "for_instance_development_tuned_stage1.yml",
        stage2_config_path=CONFIGS / "for_instance_development_tuned_stage2.yml",
        benchmark_commit="synthetic-commit",
    )

    assert payload["status"] == "frozen_for_full_development_stage2"
    assert [item["candidate_id"] for item in payload["selected_candidates"]] == [
        "p04_min_points_50_lower_band",
        "p02_min_points_50",
    ]
    assert payload["source_stage1_summary_sha256"] == module.sha256(summary)
    assert len(payload["development_plots_used"]) == 5
    assert payload["expected_stage2_metric_count"] == 84
    assert payload["confirmation_no_test_metrics_used"] is True
    assert payload["held_out_test_accessed"] is False
    assert payload["final_configuration_selected"] is False


def write_stage2_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    output_root = tmp_path / "predictions"
    run_id = "stage2-run"
    plots = [
        {
            "task_index": index,
            "safe_plot_id": f"plot_{index:02d}",
            "relative_path": f"SITE/plot_{index:02d}.las",
            "collection": "CULS" if index < 11 else "RMIT",
        }
        for index in range(21)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"dataset_split": "development", "plots": plots}))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "frozen_for_full_development_stage2",
                "held_out_test_accessed": False,
                "final_configuration_selected": False,
                "confirmation_no_test_metrics_used": True,
                "selected_candidates": [
                    {
                        "stage2_candidate_index": 0,
                        "stage1_candidate_index": 2,
                        "candidate_id": "p04_min_points_50_lower_band",
                    },
                    {
                        "stage2_candidate_index": 1,
                        "stage1_candidate_index": 0,
                        "candidate_id": "p02_min_points_50",
                    },
                ],
            }
        )
    )
    for candidate_id in ("p04_min_points_50_lower_band", "p02_min_points_50"):
        for plot in plots:
            root = (
                output_root
                / "tls2trees/for_instance/development_tuned/development"
                / f"{run_id}__{candidate_id}"
                / plot["safe_plot_id"]
            )
            metadata = root / "metadata"
            metadata.mkdir(parents=True)
            (metadata / "instance_run.json").write_text(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "workflow_run_id": run_id,
                        "held_out_test_accessed": False,
                        "runtime_seconds": 10.0,
                        "peak_rss_gb": 1.0,
                    }
                )
            )
            (metadata / "adapter_run.json").write_text(
                json.dumps(
                    {
                        "variant": "development_tuned",
                        "split": "development",
                        "held_out_test_accessed": False,
                        "runtime_seconds": 1.0,
                    }
                )
            )
            for target in ("leaf_off", "leaf_on"):
                metric_root = root / "evaluation" / target
                metric_root.mkdir(parents=True)
                leader = candidate_id.startswith("p04")
                tp = 1 if leader else 0
                fp = 0 if leader else 1
                fn = 1
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                (metric_root / "plot_metrics.json").write_text(
                    json.dumps(
                        {
                            "status": "evaluated",
                            "safe_for_scoring": True,
                            "split": "dev",
                            "target": target,
                            "plot_id": plot["safe_plot_id"],
                            "prediction_instance_count": tp + fp,
                            "reference_instance_count": tp + fn,
                            "true_positives": tp,
                            "false_positives": fp,
                            "false_negatives": fn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "mean_matched_iou": 0.75 if tp else 0.0,
                            "oversegmented_reference_count": 0,
                            "undersegmented_prediction_count": 0,
                        }
                    )
                )
    return output_root, manifest, selection, run_id


def test_stage2_summary_requires_all_84_metrics(tmp_path: Path) -> None:
    module = load_module("summarise_tls2trees_development_stage2.py")
    output_root, manifest, selection, run_id = write_stage2_fixture(tmp_path)
    payload = module.summarise(
        output_root=output_root,
        workflow_run_id=run_id,
        manifest_path=manifest,
        selection_path=selection,
    )

    assert payload["status"] == "stage2_completed"
    assert payload["expected_metric_count"] == 84
    assert payload["valid_metric_count"] == 84
    assert len(payload["plot_metrics"]) == 84
    assert payload["candidate_rankings_for_review"]["leaf_off"][0] == (
        "p04_min_points_50_lower_band"
    )
    leader = next(
        item
        for item in payload["aggregates"]
        if item["candidate_id"] == "p04_min_points_50_lower_band"
        and item["target"] == "leaf_on"
    )
    assert leader["evaluated_plot_count"] == 21
    assert leader["true_positives"] == 21
    assert leader["micro_f1"] == pytest.approx(2 / 3)
    assert payload["held_out_test_accessed"] is False
    assert payload["final_configuration_selected"] is False


def test_stage2_slurm_chain_is_guarded_and_bounded() -> None:
    names = [
        "prepare_semantic_development_stage2.sbatch",
        "evaluate_development_stage2_candidate.sbatch",
        "summarise_development_stage2.sbatch",
        "submit_development_stage2.sh",
        "monitor_development_stage2.sh",
        "resume_development_stage2_empty_predictions.sh",
    ]
    sources = {}
    for name in names:
        path = SLURM / name
        source = path.read_text(encoding="utf-8")
        sources[name] = source
        assert "set -euo pipefail" in source
        assert "/Users/" not in source
        checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
    submit = sources["submit_development_stage2.sh"]
    semantic = sources["prepare_semantic_development_stage2.sbatch"]
    candidate = sources["evaluate_development_stage2_candidate.sbatch"]
    summary = sources["summarise_development_stage2.sbatch"]
    monitor = sources["monitor_development_stage2.sh"]
    recovery = sources["resume_development_stage2_empty_predictions.sh"]
    assert "TLS2TREES_STAGE2_CONFIRMED" in submit + semantic + candidate + summary
    assert '--array="0-20%2"' in submit
    assert '--array="0-41%4"' in submit
    assert 'MANIFEST_PLOT_COUNT' in submit
    assert '!= "21"' in submit
    assert 'FROZEN_SOURCE_RUN_ID' in submit
    assert 'TLS2TREES_STAGE2_SOURCE_STAGE1_SUMMARY_SHA256' in submit
    assert 'afterok:$SEMANTIC_JOB' in submit
    assert 'afterany:$CANDIDATE_JOB' in submit
    assert "TASK_INDEX=$((SLURM_ARRAY_TASK_ID % 21))" in candidate
    assert "for TARGET in leaf_off leaf_on" in candidate
    assert "expected_metrics=84" in submit
    assert "final_configuration_selected=false" in submit + monitor
    assert "held_out_test_accessed=false" in submit + semantic + candidate + monitor
    assert "--split test" not in submit + semantic + candidate + summary
    assert "TLS2TREES_STAGE2_RECOVERY_CONFIRMED" in recovery
    assert 'EXPECTED_FAILED_LIST="6,18,27,39"' in recovery
    assert 'EXPECTED_FAILED_LIST="6,27"' in recovery
    assert "9838903" in recovery
    assert "cannot set a frame with no defined index and a scalar" in recovery
    assert "Cannot restore clstr: groupby.apply did not return a grouped index" in recovery
    assert "TLS2TREES_STAGE2_RECOVERY=1" in recovery
    assert '--array="$FAILED_LIST%2"' in recovery
    assert "--resume-failed-audited-instance" in candidate
    assert "completed_tasks_and_semantic_cache_reused=true" in recovery
    assert "TLS2TREES_STAGE2_EXPECTED_CANDIDATE_TASKS" in monitor + recovery
    assert "--split test" not in recovery
