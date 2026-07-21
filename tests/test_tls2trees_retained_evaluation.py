from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/"
    "evaluate_retained_tls2trees_predictions.py"
)
EVALUATOR = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/"
    "evaluate_for_instance_tls2trees_plot.py"
)
FINALISER = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/"
    "finalise_tls2trees_held_out_results.py"
)
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "tls2trees_retained_evaluation", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def load_finaliser():
    spec = importlib.util.spec_from_file_location("tls2trees_finaliser", FINALISER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reference(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.0, 1.0, 2.0])
    cloud.y = np.zeros(3)
    cloud.z = np.zeros(3)
    cloud.classification = np.asarray([4, 4, 3], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.asarray([1, 1, 0], dtype=np.int32)
    cloud.write(path)


def write_retained_source(
    output_root: Path,
    *,
    split: str,
    workflow: str,
    candidate: str,
    target: str,
    plot: dict[str, object],
) -> dict[str, object]:
    candidate_run = (
        f"{workflow}__{candidate}"
        if split == "development"
        else f"{workflow}__{target}__{candidate}"
    )
    plot_root = (
        output_root
        / "tls2trees/for_instance/development_tuned"
        / split
        / candidate_run
        / str(plot["safe_plot_id"])
    )
    aligned_root = plot_root / "predictions/aligned" / target
    aligned_root.mkdir(parents=True)
    aligned = aligned_root / "source_row_predictions.npz"
    aligned.write_bytes(b"retained-source-row-prediction")
    source = Path(str(plot["input_las"])).resolve()
    metadata = {
        "schema_version": "tls2trees_for_instance_alignment",
        "status": "passed",
        "target": target,
        "point_correspondence": "source_row_via_voxel_representative",
        "raw_coordinate_evaluation_permitted": False,
        "aligned_prediction_npz": str(aligned.resolve()),
        "aligned_prediction_npz_sha256": digest(aligned),
        "source_las_path": str(source),
        "source_las_sha256": digest(source),
        "source_row_count": int(plot["point_count"]),
    }
    alignment = aligned_root / "alignment_metadata.json"
    alignment.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "candidate_id": candidate,
        "target": target,
        "task_index": int(plot["task_index"]),
        "aligned_predictions_npz": str(aligned.resolve()),
        "aligned_predictions_npz_sha256": digest(aligned),
        "safe_plot_id": str(plot["safe_plot_id"]),
    }


def plan_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    development_output = tmp_path / "development_predictions"
    test_output = tmp_path / "test_predictions"
    development_plots = []
    for index in range(21):
        source = tmp_path / f"development_{index}.las"
        source.write_bytes(f"development-{index}".encode())
        development_plots.append(
            {
                "task_index": index,
                "safe_plot_id": f"dev_{index}",
                "relative_path": f"DEV/plot_{index}.las",
                "collection": "DEV",
                "input_las": str(source),
                "point_count": 1,
            }
        )
    test_plots = []
    for index in range(11):
        source = tmp_path / f"test_{index}.las"
        source.write_bytes(f"test-{index}".encode())
        test_plots.append(
            {
                "task_index": index,
                "safe_plot_id": f"test_{index}",
                "relative_path": f"TEST/plot_{index}.las",
                "collection": "TEST",
                "input_las": str(source),
                "point_count": 49_709_922 if index == 0 else 0,
                "reference_tree_count": 323 if index == 0 else 0,
            }
        )
    development_manifest = tmp_path / "development_manifest.json"
    development_manifest.write_text(
        json.dumps({"dataset_split": "development", "plots": development_plots})
    )
    test_manifest = tmp_path / "test_manifest.json"
    test_manifest.write_text(json.dumps({"dataset_split": "test", "plots": test_plots}))
    selection = tmp_path / "stage2_selection.json"
    candidates = [
        {
            "stage1_candidate_index": 2,
            "candidate_id": "p04_min_points_50_lower_band",
        },
        {"stage1_candidate_index": 0, "candidate_id": "p02_min_points_50"},
    ]
    selection.write_text(
        json.dumps(
            {
                "status": "frozen_for_full_development_stage2",
                "held_out_test_accessed": False,
                "confirmation_no_test_metrics_used": True,
                "selected_candidates": candidates,
            }
        )
    )
    final = tmp_path / "final_selection.json"
    final.write_text(
        json.dumps(
            {
                "status": "development_tuned_configuration_frozen",
                "final_configuration_selected": True,
                "held_out_test_accessed": False,
                "selected_by_target": {
                    "leaf_off": candidates[0],
                    "leaf_on": candidates[1],
                },
            }
        )
    )
    development_metric_rows = []
    for candidate in candidates:
        for plot in development_plots:
            for target in MODULE.TARGETS:
                development_metric_rows.append(
                    write_retained_source(
                        development_output,
                        split="development",
                        workflow="stage2-run",
                        candidate=str(candidate["candidate_id"]),
                        target=target,
                        plot=plot,
                    )
                )
    test_metric_rows = []
    for target in MODULE.TARGETS:
        candidate = json.loads(final.read_text())["selected_by_target"][target]
        for plot in test_plots:
            test_metric_rows.append(
                write_retained_source(
                    test_output,
                    split="test",
                    workflow="test-run",
                    candidate=str(candidate["candidate_id"]),
                    target=target,
                    plot=plot,
                )
            )
    retention = tmp_path / "retention.json"
    retention.write_text(
        json.dumps(
            {
                "status": "retention_verified",
                "run_id": "test-run",
                "dataset_split": "test",
                "verified_prediction_files": 22,
                "files": [
                    {
                        "candidate_id": row["candidate_id"],
                        "target": row["target"],
                        "plot_index": row["task_index"],
                        "plot_id": row["safe_plot_id"],
                        "relative_path": row["aligned_predictions_npz"],
                        "sha256": row["aligned_predictions_npz_sha256"],
                    }
                    for row in test_metric_rows
                ],
            }
        )
    )
    kwargs = {
        "evaluation_run_id": "evaluation-run",
        "benchmark_commit": "abc123",
        "evaluator_path": EVALUATOR,
        "metrics_root": tmp_path / "new_metrics",
        "development_output_root": development_output,
        "development_workflow_run_id": "stage2-run",
        "development_manifest_path": development_manifest,
        "development_selection_path": selection,
        "test_output_root": test_output,
        "test_workflow_run_id": "test-run",
        "test_manifest_path": test_manifest,
        "final_selection_path": final,
        "final_selection_sha256": digest(final),
        "test_retention_manifest_path": retention,
        "test_retention_manifest_sha256": digest(retention),
    }
    return kwargs, final


def test_plan_binds_84_development_and_22_retained_test_predictions(
    tmp_path: Path,
) -> None:
    kwargs, _ = plan_fixture(tmp_path)
    payload = MODULE.build_plan(**kwargs)
    assert payload["status"] == "retained_prediction_evaluation_plan_frozen"
    assert payload["expected_task_count"] == 106
    assert sum(row["split"] == "development" for row in payload["tasks"]) == 84
    assert sum(row["split"] == "test" for row in payload["tasks"]) == 22
    assert payload["inference_rerun"] is False
    assert payload["configuration_selection_performed"] is False
    assert all(row["aligned_predictions_npz_sha256"] for row in payload["tasks"])
    assert all(row["alignment_metadata_sha256"] for row in payload["tasks"])
    assert all(row["source_las_sha256"] for row in payload["tasks"])
    assert all("source_metrics_json" not in row for row in payload["tasks"])


def test_plan_rejects_test_prediction_not_bound_to_retention_manifest(
    tmp_path: Path,
) -> None:
    kwargs, _ = plan_fixture(tmp_path)
    retention = Path(kwargs["test_retention_manifest_path"])
    payload = json.loads(retention.read_text())
    payload["files"][0]["sha256"] = "0" * 64
    retention.write_text(json.dumps(payload))
    kwargs["test_retention_manifest_sha256"] = digest(retention)
    with pytest.raises(ValueError, match="retained prediction evidence"):
        MODULE.build_plan(**kwargs)


def test_plan_rejects_prediction_changed_after_alignment(tmp_path: Path) -> None:
    kwargs, _ = plan_fixture(tmp_path)
    aligned = next(Path(kwargs["development_output_root"]).rglob("*.npz"))
    aligned.write_bytes(b"changed-after-alignment")

    with pytest.raises(ValueError, match="alignment provenance mismatch"):
        MODULE.build_plan(**kwargs)


def test_plan_rejects_source_las_changed_after_alignment(tmp_path: Path) -> None:
    kwargs, _ = plan_fixture(tmp_path)
    manifest = json.loads(Path(kwargs["development_manifest_path"]).read_text())
    source = Path(manifest["plots"][0]["input_las"])
    source.write_bytes(b"changed-after-alignment")

    with pytest.raises(ValueError, match="alignment provenance mismatch"):
        MODULE.build_plan(**kwargs)


def test_task_rejects_prediction_changed_after_plan_freeze(tmp_path: Path) -> None:
    kwargs, _ = plan_fixture(tmp_path)
    plan = MODULE.build_plan(**kwargs)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    aligned = Path(plan["tasks"][0]["aligned_predictions_npz"])
    aligned.write_bytes(b"changed-after-plan-freeze")

    with pytest.raises(
        RuntimeError, match="Retained source-row prediction checksum changed"
    ):
        MODULE.run_task(
            plan_path=plan_path,
            plan_sha256=digest(plan_path),
            task_index=0,
        )


def test_summary_emits_finalisation_compatible_22_row_test_view(tmp_path: Path) -> None:
    kwargs, final = plan_fixture(tmp_path)
    plan = MODULE.build_plan(**kwargs)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = digest(plan_path)
    for task in plan["tasks"]:
        metric_path = Path(task["output_metrics_json"])
        metric_path.parent.mkdir(parents=True)
        metric_path.write_text(
            json.dumps(
                {
                    "evaluator": MODULE.FINAL_EVALUATOR,
                    "evaluation_mask": MODULE.EVALUATION_MASK,
                    "matching_policy": "maximum_cardinality_one_to_one",
                    "iou_threshold": 0.5,
                    "split": "dev" if task["split"] == "development" else "test",
                    "target": task["target"],
                    "plot_id": task["safe_plot_id"],
                    "relative_path": task["relative_path"],
                    "status": "evaluated",
                    "safe_for_scoring": True,
                    "semantic_ignore": {
                        "ignored_semantic_classes": [3],
                        "raw_prediction_instance_count": 2,
                        "ignored_predicted_point_count": 5,
                    },
                    "prediction_instance_count": 1,
                    "reference_instance_count": 3,
                    "true_positives": 1,
                    "false_positives": 0,
                    "false_negatives": 2,
                    "precision": 1.0,
                    "recall": 1 / 3,
                    "f1": 0.5,
                    "mean_matched_iou": 0.75,
                    "oversegmented_reference_count": 0,
                    "undersegmented_prediction_count": 0,
                    "retained_prediction_evaluation_provenance": {
                        "evaluation_plan_sha256": plan_sha,
                        "alignment_metadata_sha256": task[
                            "alignment_metadata_sha256"
                        ],
                        "inference_rerun": False,
                        "configuration_changed": False,
                    },
                }
            )
        )
    summary = MODULE.summarise_plan(plan_path=plan_path, plan_sha256=plan_sha)
    test_summary = MODULE.held_out_summary(summary)
    assert summary["status"] == "retained_prediction_evaluation_completed"
    assert summary["valid_metric_count"] == 106
    assert test_summary["status"] == "held_out_test_completed"
    assert test_summary["workflow_run_id"] == "test-run"
    assert test_summary["final_selection_sha256"] == digest(final)
    assert test_summary["valid_metric_count"] == 22
    assert len(test_summary["plot_metrics"]) == 22
    assert {row["task_index"] for row in test_summary["plot_metrics"]} == set(
        range(11)
    )
    assert test_summary["configuration_changed_after_test"] is False
    assert test_summary["inference_rerun"] is False
    assert test_summary["test_metrics_used_for_configuration_selection"] is False
    load_finaliser().validate_summary(test_summary, "test-run", digest(final))


def test_summary_exact_resume_creates_missing_table_directory(tmp_path: Path) -> None:
    payload = {
        "created_at_utc": "2026-07-20T19:40:00+00:00",
        "plot_metrics": [{"plot": "A", "f1": 0.25}],
        "development_aggregates": [
            {
                "split": "development",
                "candidate_id": "p02",
                "target": "leaf_on",
                "micro_f1": 0.25,
                "mean_collection_f1": {"SITE": 0.25},
            }
        ],
        "test_aggregates": [],
    }
    output_json = tmp_path / "metadata" / "summary.json"
    plot_csv = tmp_path / "tables" / "plot.csv"
    aggregate_csv = tmp_path / "tables" / "aggregate.csv"
    output_json.parent.mkdir(parents=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    MODULE.write_summary(
        payload=payload,
        output_json=output_json,
        plot_csv=plot_csv,
        aggregate_csv=aggregate_csv,
        resume_exact_partial=True,
    )
    assert plot_csv.is_file()
    assert aggregate_csv.is_file()

    MODULE.write_summary(
        payload=payload,
        output_json=output_json,
        plot_csv=plot_csv,
        aggregate_csv=aggregate_csv,
        resume_exact_partial=True,
    )
    plot_csv.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match recomputed content"):
        MODULE.write_summary(
            payload=payload,
            output_json=output_json,
            plot_csv=plot_csv,
            aggregate_csv=aggregate_csv,
            resume_exact_partial=True,
        )


def test_retained_evaluation_ignores_class3_and_verifies_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.las"
    write_reference(source)
    aligned = tmp_path / "source_row_predictions.npz"
    np.savez_compressed(
        aligned,
        source_row_index=np.arange(3),
        predicted_instance_id=np.asarray([1, 1, 1]),
        prediction_names=np.asarray(["tree_1.leafon.ply"]),
        source_las_sha256=np.asarray(digest(source)),
    )
    alignment = tmp_path / "alignment_metadata.json"
    alignment.write_text(
        json.dumps(
            {
                "schema_version": "tls2trees_for_instance_alignment",
                "point_correspondence": "source_row_via_voxel_representative",
                "raw_coordinate_evaluation_permitted": False,
                "coordinate_frame": {
                    "source": "source_crs",
                    "aligned_predictions": "source_row_indices_no_coordinates",
                    "raw_predictions": "grid_aligned_local_shift",
                    "units": "metres",
                    "local_shift_m": [0.0, 0.0, 0.0],
                    "predictions_restored_to_source": False,
                },
                "source_las": {
                    "scale_m": [0.001, 0.001, 0.001],
                    "offset_m": [0.0, 0.0, 0.0],
                },
                "matching": {
                    "coordinate_tolerance_m": 0.001,
                    "distance_metric": "euclidean",
                },
            }
        )
    )
    output = tmp_path / "final" / "plot_metrics.json"
    task = {
        "evaluation_task_index": 0,
        "target": "leaf_on",
        "safe_plot_id": "plot",
        "relative_path": "SITE/plot.las",
        "split": "test",
        "source_workflow_run_id": "test-run",
        "source_candidate_run_id": "candidate-run",
        "alignment_metadata_json": str(alignment),
        "alignment_metadata_sha256": digest(alignment),
        "aligned_predictions_npz": str(aligned),
        "aligned_predictions_npz_sha256": digest(aligned),
        "input_las": str(source),
        "source_las_sha256": digest(source),
        "output_root": str(output.parent),
        "output_metrics_json": str(output),
        "output_matches_csv": str(output.with_name("matches.csv")),
        "output_unmatched_predictions_csv": str(output.with_name("unmatched_predictions.csv")),
        "output_unmatched_references_csv": str(output.with_name("unmatched_references.csv")),
    }
    tasks = [{**task, "evaluation_task_index": index} for index in range(106)]
    plan = {
        "status": "retained_prediction_evaluation_plan_frozen",
        "expected_task_count": 106,
        "tasks": tasks,
        "evaluator_path": str(EVALUATOR),
        "evaluator_sha256": digest(EVALUATOR),
        "iou_threshold": 0.5,
        "evaluation_run_id": "evaluation-run",
        "benchmark_commit": "abc123",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = MODULE.run_task(
        plan_path=plan_path, plan_sha256=digest(plan_path), task_index=0
    )
    assert result["true_positives"] == 1
    assert result["f1"] == 1.0
    assert result["semantic_ignore"]["ignored_predicted_point_count"] == 1
    provenance = result["retained_prediction_evaluation_provenance"]
    assert provenance["inference_rerun"] is False
    assert provenance["aligned_predictions_npz_sha256"] == digest(aligned)
    assert provenance["source_las_sha256"] == digest(source)


def test_retained_evaluation_slurm_chain_is_cpu_only_guarded_and_syntactically_valid() -> None:
    names = [
        "evaluate_retained_predictions.sbatch",
        "summarise_retained_predictions.sbatch",
        "submit_retained_predictions_evaluation.sh",
        "resume_retained_predictions_summary.sh",
        "monitor_retained_predictions_evaluation.sh",
    ]
    sources = {}
    for name in names:
        path = SLURM / name
        sources[name] = path.read_text(encoding="utf-8")
        checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
        assert "set -euo pipefail" in sources[name]
        assert "/Users/" not in sources[name]
    submit = sources["submit_retained_predictions_evaluation.sh"]
    task = sources["evaluate_retained_predictions.sbatch"]
    assert "TLS2TREES_FINAL_EVALUATION_CONFIRMED" in submit + task
    assert '--array="0-105%4"' in submit
    assert 'dependency="afterok:$EVALUATION_JOB"' in submit
    assert "run-task" in task
    assert "run_for_instance_tls2trees" not in task
    assert "#SBATCH --gres" not in task
    assert "TLS2TREES_FINAL_EVALUATION_INFERENCE_ALLOWED" in task
    assert '"$TABLE_ROOT"' in submit
    assert "--resume-exact-partial" in sources["summarise_retained_predictions.sbatch"]
    resume = sources["resume_retained_predictions_summary.sh"]
    assert "TLS2TREES_FINAL_EVALUATION_SUMMARY_RESUME_CONFIRMED" in resume
    assert "COMPLETED_TASKS" in resume
    assert "--format=JobID%30,State" in resume
    assert "Expected 106 completed evaluation tasks" in resume
    assert "evaluation_tasks_rerun=false" in resume
