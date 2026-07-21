from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "methods/tls2trees/scripts/evaluation"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"


def load_finaliser():
    path = EVALUATION / "finalise_tls2trees_held_out_results.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def fixture(tmp_path: Path, module):
    run_id = "tls2trees_for-instance_development_tuned_held_out_test_20260719_110219"
    project = tmp_path / "project"
    project.mkdir()
    final_selection = project / "final_selection.json"
    final_selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "development_tuned_configuration_frozen",
                "dataset": "FOR-instance",
                "method": "TLS2trees",
                "variant": "development_tuned",
                "selection_split": "development",
                "benchmark_commit": "9" * 40,
                "selected_by_target": {
                    "leaf_off": {
                        "candidate_id": "p04_min_points_50_lower_band",
                        "stage1_candidate_index": 2,
                        "parameters": {"find_stems_min_points": 50},
                        "development_metrics": {"micro_f1": 0.004684},
                        "selection_reason": "best frozen leaf-off development score",
                    },
                    "leaf_on": {
                        "candidate_id": "p02_min_points_50",
                        "stage1_candidate_index": 0,
                        "parameters": {"find_stems_min_points": 50},
                        "development_metrics": {"micro_f1": 0.009217},
                        "selection_reason": "best frozen leaf-on development score",
                    },
                },
                "development_metric_count": 84,
                "development_plot_count": 21,
                "development_accuracy_metrics_used": True,
                "held_out_test_accessed": False,
                "held_out_test_runnable": False,
                "final_configuration_selected": True,
                "review_required_before_held_out_test": True,
            }
        )
    )
    final_sha = module.sha256(final_selection)
    evaluation_run_id = "tls2trees_for-instance_final_evaluation_20260720_193810"
    evaluation_benchmark_commit = "a" * 40
    evaluator_path = (
        project
        / "methods/tls2trees/scripts/evaluation/"
        "evaluate_for_instance_tls2trees_plot.py"
    )
    evaluator_path.parent.mkdir(parents=True)
    evaluator_path.write_text("# frozen retained-prediction evaluator fixture\n")
    evaluator_sha256 = module.sha256(evaluator_path)
    evaluation_plan = (
        project
        / "results/metadata/tls2trees/for_instance/development_tuned/"
        "final_evaluation/evaluation_plan.json"
    )
    evaluation_plan.parent.mkdir(parents=True)
    evaluation_plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "retained_prediction_evaluation_plan_frozen",
                "dataset": "FOR-instance",
                "method": "TLS2trees",
                "variant": "development_tuned",
                "evaluation_run_id": evaluation_run_id,
                "benchmark_commit": evaluation_benchmark_commit,
                "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                "evaluation_protocol": "for_instance_pointwise_class3_ignore",
                "evaluation_mask": (
                    "union_of_reference_target_and_predicted_target_points_"
                    "excluding_class3_outpoints"
                ),
                "evaluator_path": str(evaluator_path),
                "evaluator_sha256": evaluator_sha256,
                "iou_threshold": 0.5,
                "matching_policy": "maximum_cardinality_one_to_one",
                "inference_rerun": False,
                "prediction_adapter_rerun": False,
                "configuration_changed": False,
                "configuration_selection_performed": False,
                "expected_task_count": 106,
                "tasks": [],
            }
        )
    )
    evaluation_plan_sha256 = module.sha256(evaluation_plan)
    plots = []
    plot_metrics = []
    for index in range(11):
        collection = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")[index % 5]
        plot = {
            "task_index": index,
            "safe_plot_id": f"plot_{index:02d}",
            "relative_path": f"{collection}/plot_{index:02d}.las",
            "collection": collection,
            "point_count": 8,
            "reference_tree_count": 323 if index == 0 else 0,
        }
        input_las = project / "data/for_instance" / plot["relative_path"]
        input_las.parent.mkdir(parents=True, exist_ok=True)
        input_las.write_bytes(f"synthetic-las-{index}\n".encode())
        plot["input_las"] = str(input_las)
        plot["input_sha256"] = module.sha256(input_las)
        plots.append(plot)
        for target, candidate, predictions in (
            ("leaf_off", "p04_min_points_50_lower_band", 1),
            ("leaf_on", "p02_min_points_50", 2),
        ):
            candidate_run = f"{run_id}__{target}__{candidate}"
            root = (
                project
                / "data/predictions/tls2trees/for_instance/development_tuned/test"
                / candidate_run
                / plot["safe_plot_id"]
            )
            aligned_root = root / "predictions/aligned" / target
            metric_root = (
                project
                / "results/metadata/tls2trees/final_evaluation/metrics"
                / target
                / plot["safe_plot_id"]
            )
            aligned_root.mkdir(parents=True)
            metric_root.mkdir(parents=True)
            aligned = aligned_root / "source_row_predictions.npz"
            predicted = np.zeros(8, dtype=np.int64)
            predicted[:predictions] = np.arange(1, predictions + 1)
            np.savez_compressed(
                aligned,
                source_row_index=np.arange(8),
                predicted_instance_id=predicted,
                prediction_names=np.asarray(
                    [f"tree_{item}.{target.replace('_', '')}.ply" for item in range(predictions)]
                ),
            )
            aligned_sha = module.sha256(aligned)
            alignment_metadata = aligned_root / "alignment_metadata.json"
            alignment_metadata.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "target": target,
                        "aligned_prediction_npz_sha256": aligned_sha,
                        "source_row_count": 8,
                        "prediction_instance_count": predictions,
                    }
                )
            )
            references = 323 if index == 0 else 0
            evaluated_predictions = predictions - int(
                index == 0 and target == "leaf_on"
            )
            alignment_metadata_sha = module.sha256(alignment_metadata)
            metric = {
                "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                "status": "evaluated",
                "safe_for_scoring": True,
                "split": "test",
                "target": target,
                "plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "matching_policy": "maximum_cardinality_one_to_one",
                "evaluation_mask": (
                    "union_of_reference_target_and_predicted_target_points_"
                    "excluding_class3_outpoints"
                ),
                "semantic_ignore": {
                    "ignored_semantic_classes": [3],
                    "raw_prediction_instance_count": predictions,
                },
                "iou_threshold": 0.5,
                "prediction_instance_count": evaluated_predictions,
                "reference_instance_count": references,
                "true_positives": 0,
                "false_positives": evaluated_predictions,
                "false_negatives": references,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mean_matched_iou": 0.0,
                "aligned_predictions_npz": str(aligned),
                "alignment_metadata_json": str(alignment_metadata),
                "point_correspondence": {
                    "source_row_count": 8,
                    "status": "passed",
                },
                "retained_prediction_evaluation_provenance": {
                    "schema_version": 1,
                    "evaluation_run_id": evaluation_run_id,
                    "benchmark_commit": evaluation_benchmark_commit,
                    "evaluator_sha256": evaluator_sha256,
                    "evaluation_plan": str(evaluation_plan),
                    "evaluation_plan_sha256": evaluation_plan_sha256,
                    "aligned_predictions_npz_sha256": aligned_sha,
                    "alignment_metadata_sha256": alignment_metadata_sha,
                    "inference_rerun": False,
                    "prediction_adapter_rerun": False,
                    "retained_sources_changed": False,
                    "configuration_changed": False,
                },
            }
            metric_path = metric_root / "plot_metrics.json"
            metric_path.write_text(json.dumps(metric))
            plot_metrics.append(
                {
                    "target": target,
                    "candidate_id": candidate,
                    "task_index": index,
                    "collection": collection,
                    "safe_plot_id": plot["safe_plot_id"],
                    "relative_path": plot["relative_path"],
                    "status": "evaluated",
                    "safe_for_scoring": True,
                    "raw_prediction_instance_count": predictions,
                    "prediction_instance_count": evaluated_predictions,
                    "reference_instance_count": references,
                    "true_positives": 0,
                    "false_positives": evaluated_predictions,
                    "false_negatives": references,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "mean_matched_iou": 0.0,
                    "metrics_path": str(metric_path),
                    "metrics_sha256": module.sha256(metric_path),
                    "aligned_predictions_npz": str(aligned),
                    "aligned_predictions_npz_sha256": aligned_sha,
                    "alignment_metadata_json": str(alignment_metadata),
                    "alignment_metadata_sha256": alignment_metadata_sha,
                }
            )

    # Freeze a production-shaped 106-task plan after the retained fixture paths
    # exist, then bind every held-out metric to its exact plan task.
    plan_tasks = []
    development_candidates = (
        ("p04_min_points_50_lower_band", 2),
        ("p02_min_points_50", 0),
    )
    for candidate_id, candidate_index in development_candidates:
        for plot_index in range(21):
            collection = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")[
                plot_index % 5
            ]
            for target in ("leaf_off", "leaf_on"):
                task_index = len(plan_tasks)
                task_root = (
                    project
                    / "results/metadata/tls2trees/final_evaluation/metrics"
                    / "development"
                    / candidate_id
                    / f"dev_plot_{plot_index:02d}"
                    / target
                )
                aligned_root = (
                    project
                    / "data/predictions/tls2trees/for_instance/development_tuned/"
                    "development"
                    / f"dev_run__{candidate_id}"
                    / f"dev_plot_{plot_index:02d}"
                    / "predictions/aligned"
                    / target
                )
                plan_tasks.append(
                    {
                        "evaluation_task_index": task_index,
                        "split": "development",
                        "source_workflow_run_id": "dev_run",
                        "source_candidate_run_id": f"dev_run__{candidate_id}",
                        "candidate_id": candidate_id,
                        "candidate_index": candidate_index,
                        "target": target,
                        "manifest_task_index": plot_index,
                        "collection": collection,
                        "safe_plot_id": f"dev_plot_{plot_index:02d}",
                        "relative_path": f"{collection}/dev_plot_{plot_index:02d}.las",
                        "input_las": str(
                            project
                            / "data/for_instance"
                            / collection
                            / f"dev_plot_{plot_index:02d}.las"
                        ),
                        "aligned_predictions_npz": str(
                            aligned_root / "source_row_predictions.npz"
                        ),
                        "aligned_predictions_npz_sha256": f"{1000 + task_index:064x}",
                        "alignment_metadata_json": str(
                            aligned_root / "alignment_metadata.json"
                        ),
                        "alignment_metadata_sha256": f"{2000 + task_index:064x}",
                        "source_las_sha256": f"{3000 + task_index:064x}",
                        "output_root": str(task_root),
                        "output_metrics_json": str(task_root / "plot_metrics.json"),
                        "output_matches_csv": str(task_root / "matches.csv"),
                        "output_unmatched_predictions_csv": str(
                            task_root / "unmatched_predictions.csv"
                        ),
                        "output_unmatched_references_csv": str(
                            task_root / "unmatched_references.csv"
                        ),
                    }
                )

    plots_by_index = {int(plot["task_index"]): plot for plot in plots}
    rows_by_key = {
        (str(row["target"]), int(row["task_index"])): row for row in plot_metrics
    }
    held_out_tasks = {}
    for target in ("leaf_off", "leaf_on"):
        for plot_index in range(11):
            row = rows_by_key[(target, plot_index)]
            plot = plots_by_index[plot_index]
            aligned = Path(row["aligned_predictions_npz"])
            metadata = Path(row["alignment_metadata_json"])
            metrics_path = Path(row["metrics_path"])
            candidate_id = str(row["candidate_id"])
            candidate_index = (
                2 if candidate_id == "p04_min_points_50_lower_band" else 0
            )
            task_index = len(plan_tasks)
            task = {
                "evaluation_task_index": task_index,
                "split": "test",
                "source_workflow_run_id": run_id,
                "source_candidate_run_id": f"{run_id}__{target}__{candidate_id}",
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "target": target,
                "manifest_task_index": plot_index,
                "collection": plot["collection"],
                "safe_plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "input_las": plot["input_las"],
                "aligned_predictions_npz": str(aligned),
                "aligned_predictions_npz_sha256": row[
                    "aligned_predictions_npz_sha256"
                ],
                "alignment_metadata_json": str(metadata),
                "alignment_metadata_sha256": row["alignment_metadata_sha256"],
                "source_las_sha256": module.sha256(Path(plot["input_las"])),
                "output_root": str(metrics_path.parent),
                "output_metrics_json": str(metrics_path),
                "output_matches_csv": str(metrics_path.with_name("matches.csv")),
                "output_unmatched_predictions_csv": str(
                    metrics_path.with_name("unmatched_predictions.csv")
                ),
                "output_unmatched_references_csv": str(
                    metrics_path.with_name("unmatched_references.csv")
                ),
            }
            plan_tasks.append(task)
            held_out_tasks[(target, plot_index)] = task
            row.update(
                {
                    "evaluation_task_index": task_index,
                    "candidate_index": candidate_index,
                    "source_las": task["input_las"],
                    "source_las_sha256": task["source_las_sha256"],
                }
            )

    manifest = project / "manifest.json"
    manifest.write_text(json.dumps({"dataset_split": "test", "plots": plots}))
    plan = json.loads(evaluation_plan.read_text())
    plan["tasks"] = plan_tasks
    plan["development"] = {
        "workflow_run_id": "dev_run",
        "manifest": str(project / "development_manifest.json"),
        "manifest_sha256": "7" * 64,
        "selection": str(project / "stage2_selection.json"),
        "selection_sha256": "8" * 64,
        "held_out_test_accessed": False,
    }
    plan["test"] = {
        "workflow_run_id": run_id,
        "manifest": str(manifest),
        "manifest_sha256": module.sha256(manifest),
        "final_selection": str(final_selection),
        "final_selection_sha256": final_sha,
        "retention_manifest": str(project / "source_retention.json"),
        "retention_manifest_sha256": "6" * 64,
        "held_out_test_already_accessed": True,
        "test_metrics_used_for_configuration_selection": False,
    }
    evaluation_plan.write_text(json.dumps(plan))
    evaluation_plan_sha256 = module.sha256(evaluation_plan)
    for key, task in held_out_tasks.items():
        row = rows_by_key[key]
        metric_path = Path(row["metrics_path"])
        metric = json.loads(metric_path.read_text())
        metric["reference_source"] = task["input_las"]
        provenance = metric["retained_prediction_evaluation_provenance"]
        provenance.update(
            {
                "evaluation_task_index": task["evaluation_task_index"],
                "evaluation_plan_sha256": evaluation_plan_sha256,
                "source_workflow_run_id": task["source_workflow_run_id"],
                "source_candidate_run_id": task["source_candidate_run_id"],
                "source_las_sha256": task["source_las_sha256"],
            }
        )
        metric_path.write_text(json.dumps(metric))
        row["metrics_sha256"] = module.sha256(metric_path)

    summary = project / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "held_out_test_completed",
                "dataset": "FOR-instance",
                "method": "TLS2trees",
                "variant": "development_tuned",
                "split": "test",
                "workflow_run_id": run_id,
                "manifest_sha256": module.sha256(manifest),
                "final_selection_sha256": final_sha,
                "expected_plot_count": 11,
                "expected_metric_count": 22,
                "valid_metric_count": 22,
                "incomplete_tasks": [],
                "held_out_test_accessed": True,
                "held_out_accuracy_metrics_computed": True,
                "configuration_changed_after_test": False,
                "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                "evaluation_protocol": "for_instance_pointwise_class3_ignore",
                "evaluation_mask": (
                    "union_of_reference_target_and_predicted_target_points_"
                    "excluding_class3_outpoints"
                ),
                "evaluation_run_id": evaluation_run_id,
                "evaluation_plan": str(evaluation_plan),
                "evaluation_plan_sha256": evaluation_plan_sha256,
                "inference_rerun": False,
                "prediction_adapter_rerun": False,
                "retained_sources_unchanged": True,
                "test_metrics_used_for_configuration_selection": False,
                "plot_metrics": plot_metrics,
                "aggregates": [
                    {
                        "split": "test",
                        "target": "leaf_off",
                        "expected_plot_count": 11,
                        "evaluated_plot_count": 11,
                        "failed_or_invalid_plot_count": 0,
                        "prediction_instance_count": 11,
                        "reference_instance_count": 323,
                        "true_positives": 0,
                        "false_positives": 11,
                        "false_negatives": 323,
                        "precision": 0.0,
                        "recall": 0.0,
                        "micro_f1": 0.0,
                        "mean_plot_f1": 0.0,
                    },
                    {
                        "split": "test",
                        "target": "leaf_on",
                        "expected_plot_count": 11,
                        "evaluated_plot_count": 11,
                        "failed_or_invalid_plot_count": 0,
                        "prediction_instance_count": 21,
                        "reference_instance_count": 323,
                        "true_positives": 0,
                        "false_positives": 21,
                        "false_negatives": 323,
                        "precision": 0.0,
                        "recall": 0.0,
                        "micro_f1": 0.0,
                        "mean_plot_f1": 0.0,
                    },
                ],
            }
        )
    )
    outputs = project / "outputs"
    results = outputs / "results.csv"
    diagnostics = outputs / "diagnostics.csv"
    retention = outputs / "retention.csv"
    write_csv(results, module.RESULT_FIELDS)
    write_csv(diagnostics, module.RESULT_FIELDS)
    write_csv(retention, module.RETENTION_FIELDS)
    examples_dir = project / "methods/tls2trees/examples"
    examples_dir.mkdir(parents=True)
    retained_files = []
    for row in plot_metrics:
        aligned = Path(row["aligned_predictions_npz"])
        retained_files.append(
            {
                "target": row["target"],
                "candidate_id": row["candidate_id"],
                "plot_index": row["task_index"],
                "plot_id": row["safe_plot_id"],
                "relative_path": aligned.relative_to(project).as_posix(),
                "size_bytes": aligned.stat().st_size,
                "sha256": module.sha256(aligned),
                "format": "npz",
                "point_correspondence": "source_row_index",
            }
        )
    retained_files.sort(key=lambda row: (row["target"], row["plot_index"]))
    retained_manifest = (
        examples_dir
        / "tls2trees_development_tuned_prediction_retention_manifest.json"
    )
    retained_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "retention_verified",
                "dataset": "FOR-instance",
                "method": "TLS2trees",
                "variant": "development_tuned",
                "training_mode": "external_training_only",
                "run_id": run_id,
                "dataset_split": "test",
                "targets": ["leaf_off", "leaf_on"],
                "expected_plots_per_target": 11,
                "verified_prediction_files": 22,
                "verified_prediction_size_bytes": sum(
                    row["size_bytes"] for row in retained_files
                ),
                "hash_algorithm": "sha256",
                "prediction_contract": "one_instance_label_per_source_row",
                "future_metrics_without_inference": True,
                "held_out_test_accessed": True,
                "configuration_changed_after_test": False,
                "files": retained_files,
            }
        )
    )
    args = argparse.Namespace(
        project_root=project,
        summary_json=summary,
        manifest_json=manifest,
        final_selection_json=final_selection,
        final_selection_sha256=final_sha,
        run_id=run_id,
        benchmark_commit="b" * 40,
        evaluation_benchmark_commit=evaluation_benchmark_commit,
        evaluation_run_id=evaluation_run_id,
        evaluation_plan_json=evaluation_plan,
        evaluation_plan_sha256=evaluation_plan_sha256,
        evaluation_evaluator=evaluator_path,
        evaluation_evaluator_sha256=evaluator_sha256,
        examples_dir=examples_dir,
        results_csv=results,
        diagnostics_csv=diagnostics,
        retention_registry=retention,
        receipt_json=project / "receipt.json",
    )
    return args


def refreeze_plan_fixture(args: argparse.Namespace, module, plan: dict) -> None:
    """Rewrite the synthetic frozen plan and keep its metric bindings coherent."""

    summary = json.loads(args.summary_json.read_text())
    plan_path = Path(summary["evaluation_plan"])
    plan_path.write_text(json.dumps(plan))
    plan_sha256 = module.sha256(plan_path)
    summary["evaluation_plan_sha256"] = plan_sha256
    args.evaluation_plan_sha256 = plan_sha256
    for source in summary["plot_metrics"]:
        metric_path = Path(source["metrics_path"])
        metric = json.loads(metric_path.read_text())
        metric["retained_prediction_evaluation_provenance"][
            "evaluation_plan_sha256"
        ] = plan_sha256
        metric_path.write_text(json.dumps(metric))
        source["metrics_sha256"] = module.sha256(metric_path)
    args.summary_json.write_text(json.dumps(summary))


def test_finaliser_retains_predictions_and_adds_canonical_result(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    retention_path = (
        args.examples_dir
        / "tls2trees_development_tuned_prediction_retention_manifest.json"
    )
    retention_sha = module.sha256(retention_path)
    payload = module.finalise(args)

    assert payload["status"] == "tls2trees_results_finalised"
    assert payload["verified_prediction_files"] == 22
    assert payload["canonical_result"]["predicted_instances"] == 21
    assert payload["canonical_result"]["reference_instances"] == 323
    assert payload["canonical_result"]["true_positives"] == 0
    assert payload["canonical_result"]["comparable_group"] == (
        "held_out_test_tls2trees_class3_ignore"
    )
    assert payload["diagnostic_result"]["comparable_group"] == (
        "tls2trees_leaf_off_class3_ignore_diagnostic"
    )

    with args.results_csv.open(newline="") as handle:
        results = list(csv.DictReader(handle))
    assert len(results) == 1
    assert results[0]["method_slug"] == "tls2trees"
    assert results[0]["variant"] == "development_tuned"
    assert results[0]["training_mode"] == "external_training_only"
    assert results[0]["evaluation_protocol"] == (
        "for_instance_pointwise_class3_ignore"
    )
    with args.retention_registry.open(newline="") as handle:
        registry = list(csv.DictReader(handle))
    assert registry[0]["retention_profile"] == (
        "held_out_test_tls2trees_class3_ignore"
    )
    assert registry[0]["storage_status"] == "run_scoped_retained"

    assert module.sha256(retention_path) == retention_sha
    retained = json.loads(retention_path.read_text())
    assert retained["verified_prediction_files"] == 22
    assert len(retained["files"]) == 22
    assert all(len(row["sha256"]) == 64 for row in retained["files"])

    provenance_path = (
        args.examples_dir / "tls2trees_development_tuned_test_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    assert provenance["benchmark_commit"] == "b" * 40
    assert provenance["evaluation_benchmark_commit"] == "a" * 40
    summary = json.loads(args.summary_json.read_text())
    plan = json.loads(Path(summary["evaluation_plan"]).read_text())
    assert provenance["evaluation_evaluator_sha256"] == plan["evaluator_sha256"]
    assert provenance["upstream_commit"] == (
        "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
    )
    assert provenance["model_sha256"] == (
        "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
    )
    assert "inference_" + "benchmark_commit" not in provenance
    assert str(args.project_root) not in provenance_path.read_text()


def test_finaliser_rejects_manifest_derived_private_text_before_publication(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    private_collection = "/users/private_account/private_collection"

    manifest = json.loads(args.manifest_json.read_text())
    manifest["plots"][0]["collection"] = private_collection
    args.manifest_json.write_text(json.dumps(manifest))

    summary = json.loads(args.summary_json.read_text())
    summary["manifest_sha256"] = module.sha256(args.manifest_json)
    for row in summary["plot_metrics"]:
        if int(row["task_index"]) == 0:
            row["collection"] = private_collection
    args.summary_json.write_text(json.dumps(summary))

    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["test"]["manifest_sha256"] = summary["manifest_sha256"]
    for task in plan["tasks"]:
        if task["split"] == "test" and int(task["manifest_task_index"]) == 0:
            task["collection"] = private_collection
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="Private path or host token"):
        module.finalise(args)

    assert not (
        args.examples_dir / "tls2trees_development_tuned_test_plot_results.csv"
    ).exists()


def test_finaliser_replaces_one_existing_neutral_result_without_duplicates(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    module.finalise(args)

    plot_path = (
        args.examples_dir / "tls2trees_development_tuned_test_plot_results.csv"
    )
    provenance_path = (
        args.examples_dir / "tls2trees_development_tuned_test_provenance.json"
    )
    plot_path.write_text("stale public result\n")
    provenance_path.write_text("stale provenance\n")
    with args.results_csv.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
    existing[0]["brief_note"] = "stale registry row"
    args.results_csv.write_text(module.csv_text(module.RESULT_FIELDS, existing))

    retention_path = (
        args.examples_dir
        / "tls2trees_development_tuned_prediction_retention_manifest.json"
    )
    retention_sha = module.sha256(retention_path)
    module.finalise(args)

    with args.results_csv.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    with args.diagnostics_csv.open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    with args.retention_registry.open(encoding="utf-8", newline="") as handle:
        retention_rows = list(csv.DictReader(handle))
    assert len(results) == len(diagnostics) == len(retention_rows) == 1
    assert results[0]["brief_note"] != "stale registry row"
    assert plot_path.read_text().startswith("method_slug,variant,training_mode")
    assert json.loads(provenance_path.read_text())["status"] == (
        "completed_tls2trees_held_out_test"
    )
    assert module.sha256(retention_path) == retention_sha


def test_finaliser_rejects_duplicate_existing_registry_keys(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    module.finalise(args)
    with args.results_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.results_csv.write_text(module.csv_text(module.RESULT_FIELDS, [rows[0], rows[0]]))

    with pytest.raises(ValueError, match="Result registry contains duplicate key"):
        module.finalise(args)


def test_finaliser_rejects_retention_manifest_mismatch(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    retention_path = (
        args.examples_dir
        / "tls2trees_development_tuned_prediction_retention_manifest.json"
    )
    retention = json.loads(retention_path.read_text())
    retention["files"][0]["sha256"] = "0" * 64
    retention_path.write_text(json.dumps(retention))

    with pytest.raises(
        ValueError, match="Retained-prediction manifest has unexpected files"
    ):
        module.finalise(args)


def test_finaliser_rejects_noncanonical_metrics(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    first = summary["plot_metrics"][0]
    metric_path = Path(first["metrics_path"])
    metric = json.loads(metric_path.read_text())
    metric["evaluator"] = "for_instance_tls2trees_source_row_unmasked"
    metric["evaluation_mask"] = (
        "union_of_reference_target_and_predicted_target_points"
    )
    metric_path.write_text(json.dumps(metric))
    first["metrics_sha256"] = module.sha256(metric_path)
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Metric contract changed"):
        module.finalise(args)


def test_finaliser_rejects_changed_frozen_evaluation_plan(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    plan_path = Path(summary["evaluation_plan"])
    plan = json.loads(plan_path.read_text())
    plan["status"] = "changed_after_evaluation"
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(
        ValueError, match="Frozen evaluation plan is missing or its SHA-256 changed"
    ):
        module.finalise(args)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("evaluation_run_id", "different-evaluation-run"),
        ("benchmark_commit", "c" * 40),
        ("evaluator", "different_evaluator"),
    ],
)
def test_finaliser_rejects_frozen_plan_identity_mismatch(
    tmp_path: Path, field: str, replacement: str
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    plan_path = Path(summary["evaluation_plan"])
    plan = json.loads(plan_path.read_text())
    plan[field] = replacement
    plan_path.write_text(json.dumps(plan))
    summary["evaluation_plan_sha256"] = module.sha256(plan_path)
    args.evaluation_plan_sha256 = summary["evaluation_plan_sha256"]
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=rf"unexpected {field}"):
        module.finalise(args)


@pytest.mark.parametrize(
    ("attribute", "replacement", "message"),
    [
        (
            "evaluation_run_id",
            "different-frozen-evaluation-run",
            "evaluation run ID changed from frozen state",
        ),
        (
            "evaluation_plan_json",
            Path("different-frozen-plan.json"),
            "evaluation plan path changed from frozen state",
        ),
        (
            "evaluation_plan_sha256",
            "0" * 64,
            "evaluation plan SHA-256 changed from frozen state",
        ),
        (
            "evaluation_evaluator",
            Path("different-frozen-evaluator.py"),
            "evaluator path changed from evaluation state",
        ),
        (
            "evaluation_evaluator_sha256",
            "0" * 64,
            "evaluator SHA-256 changed from evaluation state",
        ),
    ],
)
def test_finaliser_rejects_values_that_differ_from_frozen_evaluation_state(
    tmp_path: Path, attribute: str, replacement: str | Path, message: str
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    setattr(args, attribute, replacement)

    with pytest.raises(ValueError, match=message):
        module.finalise(args)


def test_finaliser_rejects_duplicate_frozen_plan_task_index(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["tasks"][85]["evaluation_task_index"] = 84
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="task indices are not exact and unique"):
        module.finalise(args)


def test_finaliser_rejects_held_out_plan_task_swap(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["tasks"][84], plan["tasks"][85] = plan["tasks"][85], plan["tasks"][84]
    plan["tasks"][84]["evaluation_task_index"] = 84
    plan["tasks"][85]["evaluation_task_index"] = 85
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="Invalid plot summary provenance"):
        module.finalise(args)


def test_finaliser_rejects_plan_candidates_not_in_reviewed_selection(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    selection = json.loads(args.final_selection_json.read_text())
    selection["selected_by_target"]["leaf_on"].update(
        {
            "candidate_id": "p04_min_points_50_lower_band",
            "stage1_candidate_index": 2,
        }
    )
    args.final_selection_json.write_text(json.dumps(selection))
    args.final_selection_sha256 = module.sha256(args.final_selection_json)

    summary = json.loads(args.summary_json.read_text())
    summary["final_selection_sha256"] = args.final_selection_sha256
    args.summary_json.write_text(json.dumps(summary))
    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["test"]["final_selection_sha256"] = args.final_selection_sha256
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="differs from the reviewed final selection"):
        module.finalise(args)


def test_finaliser_rejects_plan_source_block_mismatch(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["development"]["held_out_test_accessed"] = True
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="development source accessed"):
        module.finalise(args)


def test_finaliser_rejects_task_source_not_in_frozen_manifest(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    replacement = tmp_path / "replacement-source.las"
    replacement.write_bytes(b"different source\n")
    manifest = json.loads(args.manifest_json.read_text())
    manifest["plots"][0]["input_las"] = str(replacement)
    manifest["plots"][0]["input_sha256"] = module.sha256(replacement)
    args.manifest_json.write_text(json.dumps(manifest))

    summary = json.loads(args.summary_json.read_text())
    summary["manifest_sha256"] = module.sha256(args.manifest_json)
    args.summary_json.write_text(json.dumps(summary))
    plan = json.loads(args.evaluation_plan_json.read_text())
    plan["test"]["manifest_sha256"] = summary["manifest_sha256"]
    refreeze_plan_fixture(args, module, plan)

    with pytest.raises(ValueError, match="Invalid plot summary provenance"):
        module.finalise(args)


def test_finaliser_accepts_changed_current_evaluator_for_historical_plan(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    plan = json.loads(Path(summary["evaluation_plan"]).read_text())
    Path(plan["evaluator_path"]).write_text("# evaluator changed after evaluation\n")

    module.finalise(args)

    provenance = json.loads(
        (
            args.examples_dir
            / "tls2trees_development_tuned_test_provenance.json"
        ).read_text()
    )
    assert provenance["evaluation_evaluator_sha256"] == plan["evaluator_sha256"]
    assert provenance["evaluation_evaluator_sha256"] != module.sha256(
        Path(plan["evaluator_path"])
    )


def test_finaliser_rejects_invalid_frozen_evaluator_sha256(tmp_path: Path) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    plan_path = Path(summary["evaluation_plan"])
    plan = json.loads(plan_path.read_text())
    plan["evaluator_sha256"] = "not-a-sha256"
    plan_path.write_text(json.dumps(plan))
    summary["evaluation_plan_sha256"] = module.sha256(plan_path)
    args.evaluation_plan_sha256 = summary["evaluation_plan_sha256"]
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="invalid evaluator_sha256"):
        module.finalise(args)


@pytest.mark.parametrize(
    "field",
    [
        "evaluation_plan",
        "evaluation_plan_sha256",
        "benchmark_commit",
        "evaluator_sha256",
    ],
)
def test_finaliser_rejects_metric_provenance_not_bound_to_plan(
    tmp_path: Path, field: str
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    first = summary["plot_metrics"][0]
    metric_path = Path(first["metrics_path"])
    metric = json.loads(metric_path.read_text())
    provenance = metric["retained_prediction_evaluation_provenance"]
    provenance[field] = (
        str(tmp_path / "different_evaluation_plan.json")
        if field == "evaluation_plan"
        else "0" * 64
        if field in {"evaluation_plan_sha256", "evaluator_sha256"}
        else "c" * 40
    )
    metric_path.write_text(json.dumps(metric))
    first["metrics_sha256"] = module.sha256(metric_path)
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Metric contract changed"):
        module.finalise(args)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prediction_instance_count", 99),
        ("reference_instance_count", 99),
        ("true_positives", 1),
        ("false_positives", 99),
        ("false_negatives", 99),
        ("precision", 0.25),
        ("recall", 0.25),
        ("f1", 0.25),
        ("mean_matched_iou", 0.25),
        ("raw_prediction_instance_count", 99),
    ],
)
def test_finaliser_rejects_summary_plot_values_that_differ_from_hashed_metric(
    tmp_path: Path, field: str, replacement: int | float
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    summary["plot_metrics"][0][field] = replacement
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Plot metric evidence mismatch"):
        module.finalise(args)


def test_finaliser_rejects_hashed_metric_value_that_differs_from_summary(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    first = summary["plot_metrics"][0]
    metric_path = Path(first["metrics_path"])
    metric = json.loads(metric_path.read_text())
    metric["f1"] = 0.25
    metric_path.write_text(json.dumps(metric))
    first["metrics_sha256"] = module.sha256(metric_path)
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Plot metric evidence mismatch"):
        module.finalise(args)


def test_finaliser_rejects_public_aggregate_value_that_does_not_reconcile(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    summary = json.loads(args.summary_json.read_text())
    summary["aggregates"][0]["mean_plot_f1"] = 0.25
    args.summary_json.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Aggregate metric evidence mismatch"):
        module.finalise(args)


def test_finaliser_recovers_exact_publication_after_interrupted_registry_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    real_replace = module.os.replace
    replacements = 0

    def interrupt_ninth_replace(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 9:
            raise OSError("simulated publication interruption")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", interrupt_ninth_replace)
    with pytest.raises(OSError, match="simulated publication interruption"):
        module.finalise(args)
    monkeypatch.setattr(module.os, "replace", real_replace)

    payload = module.finalise(args)
    assert payload["status"] == "tls2trees_results_finalised"
    with args.results_csv.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with args.diagnostics_csv.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with args.retention_registry.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert not list(args.project_root.rglob("*.tls2trees-held-out-finalisation.tmp"))


def test_finaliser_accepts_exact_existing_publication_without_replacing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    first = module.finalise(args)

    def unexpected_replace(source: Path, destination: Path) -> None:
        raise AssertionError(f"unexpected replacement: {source} -> {destination}")

    monkeypatch.setattr(module.os, "replace", unexpected_replace)
    second = module.finalise(args)
    assert second == first


def test_finaliser_rejects_receipt_symlink_before_publication(
    tmp_path: Path,
) -> None:
    module = load_finaliser()
    args = fixture(tmp_path, module)
    external = tmp_path / "external-receipt.json"
    external.write_text("external remains unchanged\n", encoding="utf-8")
    args.receipt_json.symlink_to(external)

    with pytest.raises(ValueError, match="Publication target is a symlink"):
        module.finalise(args)

    assert external.read_text(encoding="utf-8") == "external remains unchanged\n"
    assert not (args.examples_dir / "tls2trees_development_tuned_test_results.csv").exists()


def test_finalisation_shell_routes_are_guarded_and_syntax_valid() -> None:
    submit = SLURM / "submit_held_out_results_finalisation.sh"
    monitor = SLURM / "monitor_held_out_results_finalisation.sh"
    batch = SLURM / "finalise_held_out_results.sbatch"
    worktree_gate = SLURM / "held_out_finalisation_worktree_gate.sh"
    for path in (submit, monitor, batch, worktree_gate):
        checked = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stderr
    source = submit.read_text(encoding="utf-8")
    assert "TLS2TREES_FINALIZE_RESULTS_CONFIRMED" in source
    assert "latest_final_evaluation_state_file.txt" in source
    assert "TLS2TREES_FINAL_EVALUATION_TEST_SUMMARY_JSON" in source
    assert "held_out_test_completed" in source
    assert "sbatch" in source
    assert "git merge-base --is-ancestor" in source
    assert "test ! -e" not in source
    assert "TLS2TREES_TEST_" + "BENCHMARK_COMMIT" not in source
    assert "TLS2TREES_FINALIZE_RESULTS_RECOVERY_CONFIRMED" in source
    assert "held_out_finalisation_worktree_gate.sh" in source
    batch_source = batch.read_text(encoding="utf-8")
    assert "run_for_instance_tls2trees" not in batch_source
    assert "--inference-" + "benchmark-commit" not in batch_source
    assert "--benchmark-commit" in batch_source
    assert "--evaluation-benchmark-commit" in batch_source
    assert "--evaluation-run-id" in batch_source
    assert "--evaluation-plan-json" in batch_source
    assert "--evaluation-plan-sha256" in batch_source
    assert "--evaluation-evaluator" in batch_source
    assert "--evaluation-evaluator-sha256" in batch_source
    for variable in (
        "TLS2TREES_FINAL_EVALUATION_RUN_ID",
        "TLS2TREES_FINAL_EVALUATION_PLAN_JSON",
        "TLS2TREES_FINAL_EVALUATION_PLAN_SHA256",
        "TLS2TREES_FINAL_EVALUATION_EVALUATOR",
        "TLS2TREES_FINAL_EVALUATION_EVALUATOR_SHA256",
    ):
        assert variable in source
        assert variable in batch_source
    assert "TLS2TREES_FINALIZE_RESULTS_RECOVERY_CONFIRMED" in batch_source
    assert "held_out_finalisation_worktree_gate.sh" in batch_source


def test_finalisation_recovery_worktree_gate_is_exact_and_opt_in(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    allowed = repository / (
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_benchmark_results.csv"
    )
    unrelated = repository / "unrelated.txt"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("original\n")
    unrelated.write_text("original\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    gate = SLURM / "held_out_finalisation_worktree_gate.sh"

    def validate(recovery: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; tls2trees_validate_held_out_finalisation_worktree "$2" "$3"',
                "gate-test",
                str(gate),
                str(repository),
                recovery,
            ],
            capture_output=True,
            text=True,
        )

    assert validate("0").returncode == 0
    assert validate("1").returncode == 0
    assert validate("unexpected").returncode == 2

    allowed.write_text("partial finalisation\n")
    assert validate("0").returncode == 2
    assert validate("1").returncode == 0

    unrelated.write_text("unrelated edit\n")
    rejected = validate("1")
    assert rejected.returncode == 2
    assert "unrelated worktree path: unrelated.txt" in rejected.stderr
    unrelated.write_text("original\n")

    retention_manifest = repository / (
        "methods/tls2trees/examples/"
        "tls2trees_development_tuned_prediction_retention_manifest.json"
    )
    retention_manifest.parent.mkdir(parents=True)
    retention_manifest.write_text("unexpected input-evidence change\n")
    rejected = validate("1")
    assert rejected.returncode == 2
    assert retention_manifest.relative_to(repository).as_posix() in rejected.stderr
    retention_manifest.unlink()

    temporary = allowed.with_name(
        f".{allowed.name}.tls2trees-held-out-finalisation.tmp"
    )
    temporary.write_text("exact staged content\n")
    assert validate("1").returncode == 0
    near_miss = allowed.with_name(f".{allowed.name}.tmp")
    near_miss.write_text("not an exact finaliser temporary file\n")
    rejected = validate("1")
    assert rejected.returncode == 2
    assert near_miss.relative_to(repository).as_posix() in rejected.stderr

    near_miss.unlink()
    temporary.unlink()
    external = tmp_path / "external-publication-target.txt"
    external.write_text("must remain external\n")
    public_symlink = repository / (
        "methods/tls2trees/examples/"
        "tls2trees_development_tuned_test_results.csv"
    )
    public_symlink.parent.mkdir(parents=True, exist_ok=True)
    public_symlink.symlink_to(external)
    rejected = validate("1")
    assert rejected.returncode == 2
    assert "symbolic link at publication path" in rejected.stderr

    public_symlink.unlink()
    temporary.symlink_to(external)
    rejected = validate("1")
    assert rejected.returncode == 2
    assert "symbolic link at publication path" in rejected.stderr
