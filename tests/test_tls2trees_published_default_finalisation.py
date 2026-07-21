from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/"
    "finalise_tls2trees_published_default_results.py"
)
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
spec = importlib.util.spec_from_file_location("published_default_finaliser", SCRIPT)
assert spec and spec.loader
finaliser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finaliser)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_registry(path: Path, fields: list[str], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: "" for field in fields}
    row["method_slug"] = method
    row["variant"] = "unchanged"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def source_aggregate(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target"] == target]
    tp = sum(int(row["true_positives"]) for row in selected)
    fp = sum(int(row["false_positives"]) for row in selected)
    fn = sum(int(row["false_negatives"]) for row in selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "target": target,
        "configuration_id": "published_default",
        "expected_plot_count": 11,
        "evaluated_plot_count": 11,
        "failed_or_invalid_plot_count": 0,
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in selected
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in selected
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": sum(float(row["f1"]) for row in selected) / 11,
        "mean_collection_f1": {},
        "oversegmented_reference_count": 0,
        "undersegmented_prediction_count": 0,
    }


def make_fixture(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, Path]]:
    project = tmp_path / "repository"
    examples = project / "methods/tls2trees/examples"
    workflow_path = project / "workflow.yml"
    published_path = project / "published.yml"
    benchmark_path = project / "benchmark.yml"
    manifest_path = project / "runtime/test_manifest.json"
    summary_path = project / "runtime/published_default_test_summary.json"
    source_plot_csv = project / "runtime/plot_metrics.csv"
    source_target_csv = project / "runtime/target_summary.csv"
    source_retention_path = project / "runtime/prediction_retention_manifest.json"

    workflow = yaml.safe_load(
        (ROOT / "methods/tls2trees/configs/for_instance_published_default_test.yml")
        .read_text()
    )
    published = yaml.safe_load(
        (ROOT / "methods/tls2trees/configs/for_instance_published_default.yml")
        .read_text()
    )
    workflow["method"]["source_config"] = str(published_path)
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(yaml.safe_dump(workflow, sort_keys=False))
    published_path.write_text(yaml.safe_dump(published, sort_keys=False))
    benchmark_path.write_text("evaluation: neutral\n")

    point_counts = [1] * 10 + [49_709_912]
    reference_counts = [29] * 10 + [33]
    plots = []
    for task, relative_path in enumerate(
        workflow["dataset"]["exact_relative_paths"]
    ):
        plots.append(
            {
                "task_index": task,
                "relative_path": relative_path,
                "safe_plot_id": f"plot_{task:02d}",
                "collection": relative_path.split("/", 1)[0],
                "point_count": point_counts[task],
                "reference_tree_count": reference_counts[task],
            }
        )
    manifest = {"dataset_split": "test", "plots": plots}
    write_json(manifest_path, manifest)

    run_id = "tls2trees_for-instance_published_default_held_out_test_20260721_000000"
    rows: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for target in finaliser.TARGETS:
        for plot in plots:
            task = plot["task_index"]
            output = (
                project
                / "data/predictions/tls2trees/for_instance/published_default/test"
                / run_id
                / plot["safe_plot_id"]
                / "predictions/aligned"
                / target
            )
            prediction = output / "source_row_predictions.npz"
            alignment = output / "alignment_metadata.json"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            prediction.write_bytes(f"{target}:{task}".encode())
            write_json(alignment, {"status": "passed", "target": target})
            metric_path = (
                project
                / "runtime/metrics"
                / target
                / f"plot_{task:02d}.json"
            )
            tp = int(target == "leaf_on" and task < 3)
            predicted = 1
            fp = predicted - tp
            references = plot["reference_tree_count"]
            fn = references - tp
            precision = tp / predicted
            recall = tp / references
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            metric = {
                "evaluator": finaliser.EVALUATOR,
                "evaluation_mask": finaliser.SOURCE_MASK,
                "matching_policy": finaliser.MATCHING,
                "iou_threshold": 0.5,
                "semantic_ignore": {"ignored_semantic_classes": [3]},
                "status": "evaluated",
                "safe_for_scoring": True,
            }
            write_json(metric_path, metric)
            row = {
                "target": target,
                "configuration_id": "published_default",
                "task_index": task,
                "collection": plot["collection"],
                "safe_plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "status": "evaluated",
                "safe_for_scoring": True,
                "raw_prediction_instance_count": predicted,
                "prediction_instance_count": predicted,
                "reference_instance_count": references,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": 0.75 if tp else 0.0,
                "oversegmented_reference_count": 0,
                "undersegmented_prediction_count": 0,
                "semantic_cache_reused": False,
                "instance_runtime_seconds": 1.0,
                "instance_peak_rss_gb": 0.5,
                "adapter_runtime_seconds": 0.5,
                "metrics_path": str(metric_path),
                "metrics_sha256": sha256(metric_path),
                "prediction_path": str(prediction),
                "prediction_sha256": sha256(prediction),
                "alignment_metadata_sha256": sha256(alignment),
            }
            rows.append(row)
            retained.append(
                {
                    "target": target,
                    "configuration_id": "published_default",
                    "plot_index": task,
                    "plot_id": plot["safe_plot_id"],
                    "relative_path": prediction.relative_to(project).as_posix(),
                    "format": "npz",
                    "point_correspondence": "source_row_index",
                    "sha256": sha256(prediction),
                    "size_bytes": prediction.stat().st_size,
                    "alignment_metadata_relative_path": alignment.relative_to(
                        project
                    ).as_posix(),
                    "alignment_metadata_sha256": sha256(alignment),
                }
            )
    aggregates = [source_aggregate(rows, target) for target in finaliser.TARGETS]
    source_retention = {
        "status": "retention_verified",
        "dataset": "FOR-instance",
        "dataset_split": "test",
        "method": "TLS2trees",
        "variant": "published_default",
        "run_id": run_id,
        "expected_files": 22,
        "verified_prediction_files": 22,
        "verified_prediction_size_bytes": sum(row["size_bytes"] for row in retained),
        "manifest_sha256": sha256(manifest_path),
        "workflow_config_sha256": sha256(workflow_path),
        "published_config_sha256": sha256(published_path),
        "benchmark_config_sha256": sha256(benchmark_path),
        "configuration_changed_after_test": False,
        "files": retained,
    }
    write_json(source_retention_path, source_retention)
    summary = {
        "status": "published_default_test_completed",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "published_default",
        "split": "test",
        "workflow_run_id": run_id,
        "manifest_sha256": sha256(manifest_path),
        "workflow_config_sha256": sha256(workflow_path),
        "published_config_sha256": sha256(published_path),
        "benchmark_config_sha256": sha256(benchmark_path),
        "expected_plot_count": 11,
        "expected_metric_count": 22,
        "valid_metric_count": 22,
        "held_out_test_accessed": True,
        "held_out_accuracy_metrics_computed": True,
        "configuration_selected_from_for_instance_metrics": False,
        "configuration_changed_after_test": False,
        "retention_manifest_sha256": sha256(source_retention_path),
        "plot_metrics": rows,
        "aggregates": aggregates,
    }
    write_json(summary_path, summary)
    source_plot_csv.write_text(finaliser.csv_text(list(rows[0]), rows))
    source_target_csv.write_text(
        finaliser.csv_text(list(aggregates[0]), aggregates)
    )

    results_csv = project / "outputs/results.csv"
    diagnostics_csv = project / "outputs/diagnostics.csv"
    retention_csv = project / "outputs/retention.csv"
    write_registry(results_csv, finaliser.RESULT_FIELDS, "treelearn")
    write_registry(diagnostics_csv, finaliser.RESULT_FIELDS, "treex")
    write_registry(retention_csv, finaliser.RETENTION_FIELDS, "segmentanytree")
    args = argparse.Namespace(
        project_root=project,
        run_id=run_id,
        benchmark_commit="a" * 40,
        upstream_commit=finaliser.EXPECTED_UPSTREAM_COMMIT,
        model_sha256=finaliser.EXPECTED_MODEL_SHA256,
        workflow_config=workflow_path,
        workflow_config_sha256=sha256(workflow_path),
        published_config=published_path,
        published_config_sha256=sha256(published_path),
        benchmark_config=benchmark_path,
        benchmark_config_sha256=sha256(benchmark_path),
        manifest_json=manifest_path,
        manifest_sha256=sha256(manifest_path),
        summary_json=summary_path,
        plot_csv=source_plot_csv,
        target_csv=source_target_csv,
        source_retention_json=source_retention_path,
        examples_dir=examples,
        results_csv=results_csv,
        diagnostics_csv=diagnostics_csv,
        retention_registry=retention_csv,
        receipt_json=project / "runtime/publication_receipt.json",
    )
    return args, {
        "project": project,
        "examples": examples,
        "results": results_csv,
        "diagnostics": diagnostics_csv,
        "retention": retention_csv,
        "first_prediction": project / retained[0]["relative_path"],
    }


def matching_rows(path: Path, fields: dict[str, str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if all(row[key] == value for key, value in fields.items())
        ]


def test_finaliser_verifies_and_safely_upserts_public_evidence(tmp_path: Path) -> None:
    args, paths = make_fixture(tmp_path)
    first = finaliser.finalise(args)
    assert first["status"] == "tls2trees_published_default_results_finalised"
    assert first["verified_prediction_files"] == 22
    assert len(list(paths["examples"].iterdir())) == 8

    result_match = {
        "method_slug": "tls2trees",
        "variant": "published_default",
        "comparable_group": finaliser.HEADLINE_GROUP,
    }
    diagnostic_match = {
        "method_slug": "tls2trees",
        "variant": "published_default",
        "comparable_group": finaliser.DIAGNOSTIC_GROUP,
    }
    retention_match = {
        "method_slug": "tls2trees",
        "variant": "published_default",
        "retention_profile": finaliser.HEADLINE_GROUP,
    }
    assert len(matching_rows(paths["results"], result_match)) == 1
    assert len(matching_rows(paths["diagnostics"], diagnostic_match)) == 1
    assert len(matching_rows(paths["retention"], retention_match)) == 1
    assert matching_rows(paths["results"], {"method_slug": "treelearn"})

    # A recovery finalisation replaces the one matching row and never duplicates it.
    finaliser.finalise(args)
    assert len(matching_rows(paths["results"], result_match)) == 1
    assert len(matching_rows(paths["diagnostics"], diagnostic_match)) == 1
    assert len(matching_rows(paths["retention"], retention_match)) == 1

    for output in paths["examples"].iterdir():
        content = output.read_text(encoding="utf-8").lower()
        assert str(paths["project"]).lower() not in content
        assert "/users/" not in content
        assert "/mnt/" not in content
        assert "fastscratch" not in content
        assert "barkla" not in content


def test_finaliser_rejects_changed_retained_prediction(tmp_path: Path) -> None:
    args, paths = make_fixture(tmp_path)
    paths["first_prediction"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="retained prediction"):
        finaliser.finalise(args)
    assert not paths["examples"].exists()


def test_finaliser_refuses_duplicate_registry_rows(tmp_path: Path) -> None:
    args, paths = make_fixture(tmp_path)
    finaliser.finalise(args)
    with paths["results"].open("a", encoding="utf-8") as handle:
        row = matching_rows(
            paths["results"],
            {
                "method_slug": "tls2trees",
                "variant": "published_default",
                "comparable_group": finaliser.HEADLINE_GROUP,
            },
        )[0]
        writer = csv.DictWriter(
            handle, fieldnames=finaliser.RESULT_FIELDS, lineterminator="\n"
        )
        writer.writerow(row)
    with pytest.raises(ValueError, match="duplicate rows"):
        finaliser.finalise(args)


def test_guarded_slurm_entrypoints_are_syntactically_valid() -> None:
    wrapper = SLURM / "finalise_published_default_results.sh"
    batch = SLURM / "finalise_published_default_results.sbatch"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", "-n", str(batch)], check=True)
    text = wrapper.read_text()
    assert "TLS2TREES_PUBLISHED_DEFAULT_RESULTS_CONFIRMED" in text
    assert "TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256" in text
    assert "SUMMARY_STATE" in text
    assert "COMPLETED" in text
    assert "git merge-base --is-ancestor" in text
    assert "git merge-base --is-ancestor" in batch.read_text()
