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
    final_selection.write_text('{"status":"development_tuned_configuration_frozen"}\n')
    final_sha = module.sha256(final_selection)
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
            metric = {
                "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
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
            }
            metric_path = metric_root / "plot_metrics.json"
            metric_path.write_text(json.dumps(metric))
            references = 323 if index == 0 else 0
            evaluated_predictions = predictions - int(
                index == 0 and target == "leaf_on"
            )
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
                    "alignment_metadata_sha256": module.sha256(alignment_metadata),
                }
            )
    manifest = project / "manifest.json"
    manifest.write_text(json.dumps({"dataset_split": "test", "plots": plots}))
    summary = project / "summary.json"
    summary.write_text(
        json.dumps(
            {
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
                "inference_rerun": False,
                "prediction_adapter_rerun": False,
                "retained_sources_unchanged": True,
                "test_metrics_used_for_configuration_selection": False,
                "plot_metrics": plot_metrics,
                "aggregates": [
                    {
                        "target": "leaf_off",
                        "prediction_instance_count": 11,
                        "reference_instance_count": 323,
                        "true_positives": 0,
                        "false_positives": 11,
                        "false_negatives": 323,
                    },
                    {
                        "target": "leaf_on",
                        "prediction_instance_count": 21,
                        "reference_instance_count": 323,
                        "true_positives": 0,
                        "false_positives": 21,
                        "false_negatives": 323,
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
        evaluation_benchmark_commit="a" * 40,
        examples_dir=examples_dir,
        results_csv=results,
        diagnostics_csv=diagnostics,
        retention_registry=retention,
        receipt_json=project / "receipt.json",
    )
    return args


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
    assert provenance["upstream_commit"] == (
        "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
    )
    assert provenance["model_sha256"] == (
        "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
    )
    assert "inference_" + "benchmark_commit" not in provenance
    assert str(args.project_root) not in provenance_path.read_text()


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


def test_finalisation_shell_routes_are_guarded_and_syntax_valid() -> None:
    submit = SLURM / "submit_held_out_results_finalisation.sh"
    monitor = SLURM / "monitor_held_out_results_finalisation.sh"
    batch = SLURM / "finalise_held_out_results.sbatch"
    for path in (submit, monitor, batch):
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
    batch_source = batch.read_text(encoding="utf-8")
    assert "run_for_instance_tls2trees" not in batch_source
    assert "--inference-" + "benchmark-commit" not in batch_source
    assert "--benchmark-commit" in batch_source
    assert "--evaluation-benchmark-commit" in batch_source
