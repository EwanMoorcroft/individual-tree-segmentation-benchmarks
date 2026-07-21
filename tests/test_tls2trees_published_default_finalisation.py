from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
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


def make_fixture(
    tmp_path: Path,
    *,
    include_metric_prediction_hash: bool = True,
) -> tuple[argparse.Namespace, dict[str, Path]]:
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
            write_json(
                alignment,
                {
                    "status": "passed",
                    "target": target,
                    "aligned_prediction_npz": str(prediction.resolve()),
                    "aligned_prediction_npz_sha256": sha256(prediction),
                },
            )
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
                "split": "test",
                "target": target,
                "plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "aligned_predictions_npz": str(prediction.resolve()),
                "alignment_metadata_json": str(alignment.resolve()),
                "alignment_metadata_sha256": sha256(alignment),
                "semantic_ignore": {
                    "ignored_semantic_classes": [3],
                    "raw_prediction_instance_count": predicted,
                },
                "status": "evaluated",
                "safe_for_scoring": True,
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
            }
            if include_metric_prediction_hash:
                metric["aligned_predictions_npz_sha256"] = sha256(prediction)
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
        publication_benchmark_commit="b" * 40,
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
        "first_metric": Path(rows[0]["metrics_path"]),
        "summary": summary_path,
        "source_plot_csv": source_plot_csv,
        "source_target_csv": source_target_csv,
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

    provenance = json.loads(
        (
            paths["examples"]
            / "tls2trees_published_default_test_provenance.json"
        ).read_text(encoding="utf-8")
    )
    public_retention = (
        paths["examples"]
        / "tls2trees_published_default_prediction_retention_manifest.json"
    )
    assert provenance["inference_executed"] is True
    assert provenance["inference_execution_scope"] == (
        "dedicated_published_default_held_out_test"
    )
    assert provenance["inference_rerun"] is False
    assert provenance["benchmark_commit"] == "a" * 40
    assert provenance["publication_benchmark_commit"] == "b" * 40
    assert provenance["retention_manifest_sha256"] == sha256(public_retention)
    assert provenance["public_retention_manifest_sha256"] == sha256(
        public_retention
    )

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


def test_finaliser_accepts_db4051d_metric_without_redundant_prediction_hash(
    tmp_path: Path,
) -> None:
    success_args, success_paths = make_fixture(
        tmp_path / "success",
        include_metric_prediction_hash=False,
    )
    metric = json.loads(success_paths["first_metric"].read_text(encoding="utf-8"))
    assert "aligned_predictions_npz_sha256" not in metric
    payload = finaliser.finalise(success_args)
    assert payload["status"] == "tls2trees_published_default_results_finalised"

    tamper_args, tamper_paths = make_fixture(
        tmp_path / "tamper",
        include_metric_prediction_hash=False,
    )
    tamper_paths["first_prediction"].write_bytes(b"changed after evaluation")
    with pytest.raises(ValueError, match="retained prediction"):
        finaliser.finalise(tamper_args)
    assert not tamper_paths["examples"].exists()


def test_finaliser_rejects_metric_not_bound_to_retained_prediction(
    tmp_path: Path,
) -> None:
    args, paths = make_fixture(tmp_path)
    metric = json.loads(paths["first_metric"].read_text(encoding="utf-8"))
    metric["aligned_predictions_npz_sha256"] = "0" * 64
    write_json(paths["first_metric"], metric)

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["plot_metrics"][0]["metrics_sha256"] = sha256(paths["first_metric"])
    write_json(paths["summary"], summary)
    paths["source_plot_csv"].write_text(
        finaliser.csv_text(
            list(summary["plot_metrics"][0]), summary["plot_metrics"]
        )
    )

    with pytest.raises(ValueError, match="Metric retained-evidence binding changed"):
        finaliser.finalise(args)
    assert not paths["examples"].exists()


def test_finaliser_rejects_metric_identity_not_bound_to_summary(
    tmp_path: Path,
) -> None:
    args, paths = make_fixture(tmp_path)
    metric = json.loads(paths["first_metric"].read_text(encoding="utf-8"))
    metric["target"] = "leaf_on"
    write_json(paths["first_metric"], metric)

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["plot_metrics"][0]["metrics_sha256"] = sha256(paths["first_metric"])
    write_json(paths["summary"], summary)
    paths["source_plot_csv"].write_text(
        finaliser.csv_text(
            list(summary["plot_metrics"][0]), summary["plot_metrics"]
        )
    )

    with pytest.raises(
        ValueError,
        match="Published-default summary/metric evidence mismatch: leaf_off:0:target",
    ):
        finaliser.finalise(args)
    assert not paths["examples"].exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("prediction_instance_count", 2), ("f1", 0.25)),
)
def test_finaliser_rejects_summary_count_or_score_not_bound_to_metric(
    tmp_path: Path,
    field: str,
    replacement: int | float,
) -> None:
    args, paths = make_fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["plot_metrics"][0][field] = replacement
    summary["aggregates"] = [
        source_aggregate(summary["plot_metrics"], target)
        for target in finaliser.TARGETS
    ]
    write_json(paths["summary"], summary)
    paths["source_plot_csv"].write_text(
        finaliser.csv_text(
            list(summary["plot_metrics"][0]), summary["plot_metrics"]
        )
    )
    paths["source_target_csv"].write_text(
        finaliser.csv_text(
            list(summary["aggregates"][0]), summary["aggregates"]
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Published-default summary/metric evidence mismatch: "
            rf"leaf_off:0:{field}"
        ),
    ):
        finaliser.finalise(args)
    assert not paths["examples"].exists()


def test_finaliser_recovers_exact_publication_after_interrupted_registry_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, paths = make_fixture(tmp_path)
    real_replace = finaliser.os.replace
    replacements = 0

    def interrupt_ninth_replace(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 9:
            raise OSError("simulated published-default publication interruption")
        real_replace(source, destination)

    monkeypatch.setattr(finaliser.os, "replace", interrupt_ninth_replace)
    with pytest.raises(OSError, match="publication interruption"):
        finaliser.finalise(args)
    monkeypatch.setattr(finaliser.os, "replace", real_replace)

    payload = finaliser.finalise(args)
    assert payload["status"] == "tls2trees_published_default_results_finalised"
    for path in (paths["results"], paths["diagnostics"], paths["retention"]):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
    assert not list(
        paths["project"].rglob(
            "*.tls2trees-published-default-finalisation.tmp"
        )
    )


def test_finaliser_accepts_exact_existing_publication_without_replacing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = make_fixture(tmp_path)
    first = finaliser.finalise(args)

    def unexpected_replace(source: Path, destination: Path) -> None:
        raise AssertionError(f"unexpected replacement: {source} -> {destination}")

    monkeypatch.setattr(finaliser.os, "replace", unexpected_replace)
    second = finaliser.finalise(args)
    assert second == first


def test_finaliser_rejects_receipt_symlink_before_publication(
    tmp_path: Path,
) -> None:
    args, paths = make_fixture(tmp_path)
    external = tmp_path / "external-receipt.json"
    external.write_text("external remains unchanged\n", encoding="utf-8")
    args.receipt_json.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_json.symlink_to(external)

    with pytest.raises(ValueError, match="Publication target is a symlink"):
        finaliser.finalise(args)

    assert external.read_text(encoding="utf-8") == "external remains unchanged\n"
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
    gate = SLURM / "published_default_finalisation_worktree_gate.sh"
    for path in (wrapper, batch, gate):
        subprocess.run(["bash", "-n", str(path)], check=True)
    text = wrapper.read_text()
    assert "TLS2TREES_PUBLISHED_DEFAULT_RESULTS_CONFIRMED" in text
    assert "TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256" in text
    assert "SUMMARY_STATE" in text
    assert "COMPLETED" in text
    assert "git merge-base --is-ancestor" in text
    batch_text = batch.read_text()
    assert "git merge-base --is-ancestor" in batch_text
    assert "PUBLICATION_BENCHMARK_COMMIT=$(git rev-parse HEAD)" in text
    assert "TLS2TREES_PD_FINALISE_PUBLICATION_BENCHMARK_COMMIT" in text
    assert "TLS2TREES_PD_FINALISE_PUBLICATION_BENCHMARK_COMMIT" in batch_text
    assert 'test "$(git rev-parse HEAD)"' in batch_text
    assert "--publication-benchmark-commit" in batch_text
    gate_text = gate.read_text()

    expected_targets = {
        "methods/tls2trees/examples/tls2trees_published_default_test_plot_results.csv",
        "methods/tls2trees/examples/tls2trees_published_default_test_site_results.csv",
        "methods/tls2trees/examples/tls2trees_published_default_test_results.csv",
        "methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_plot_diagnostic.csv",
        "methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_site_diagnostic.csv",
        "methods/tls2trees/examples/tls2trees_published_default_leaf_off_test_diagnostic.csv",
        "methods/tls2trees/examples/tls2trees_published_default_prediction_retention_manifest.json",
        "methods/tls2trees/examples/tls2trees_published_default_test_provenance.json",
        "outputs/for_instance_benchmark_metrics/for_instance_method_benchmark_results.csv",
        "outputs/for_instance_benchmark_metrics/for_instance_method_development_diagnostics.csv",
        "outputs/for_instance_benchmark_metrics/for_instance_prediction_retention_registry.csv",
    }
    for source in (text, batch_text):
        assert "TLS2TREES_PUBLISHED_DEFAULT_RESULTS_RECOVERY_CONFIRMED" in source
        assert "published_default_finalisation_worktree_gate.sh" in source
        assert "tls2trees_validate_published_default_finalisation_worktree" in source
        assert ":(exclude)" not in source
    match = re.search(
        r"publication_targets=\(\n(.*?)\n  \)", gate_text, re.DOTALL
    )
    assert match
    assert set(match.group(1).split()) == expected_targets
    assert 'status=${entry:0:2}' in gate_text
    assert 'changed_path=${entry:3}' in gate_text
    assert '"$status" != " M" && "$status" != "??"' in gate_text
    assert "tls2trees-published-default-finalisation.tmp" in gate_text
    assert ":(exclude)" not in gate_text


def test_published_default_recovery_gate_rejects_index_and_path_corruption(
    tmp_path: Path,
) -> None:
    gate = SLURM / "published_default_finalisation_worktree_gate.sh"
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test User"],
        check=True,
    )
    tracked = (
        repository
        / "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_benchmark_results.csv"
    )
    tracked.parent.mkdir(parents=True)
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture"],
        check=True,
    )

    def check(recovery: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; '
                'tls2trees_validate_published_default_finalisation_worktree "$2" "$3"',
                "gate-test",
                str(gate),
                str(repository),
                recovery,
            ],
            capture_output=True,
            text=True,
        )

    assert check("0").returncode == 0
    tracked.write_text("unstaged publication\n", encoding="utf-8")
    assert check("1").returncode == 0
    assert check("0").returncode == 2

    subprocess.run(["git", "-C", str(repository), "add", str(tracked)], check=True)
    staged = check("1")
    assert staged.returncode == 2
    assert "Git status 'M '" in staged.stderr
    subprocess.run(
        ["git", "-C", str(repository), "restore", "--staged", str(tracked)],
        check=True,
    )
    tracked.write_text("original\n", encoding="utf-8")

    tracked.unlink()
    deleted = check("1")
    assert deleted.returncode == 2
    assert "Git status ' D'" in deleted.stderr
    tracked.write_text("original\n", encoding="utf-8")

    moved = tracked.with_name("renamed.csv")
    subprocess.run(
        ["git", "-C", str(repository), "mv", str(tracked), str(moved)], check=True
    )
    renamed = check("1")
    assert renamed.returncode == 2
    assert "Git status 'R '" in renamed.stderr
    subprocess.run(
        ["git", "-C", str(repository), "mv", str(moved), str(tracked)], check=True
    )

    public = (
        repository
        / "methods/tls2trees/examples/"
        "tls2trees_published_default_test_results.csv"
    )
    public.parent.mkdir(parents=True)
    public.write_text("published\n", encoding="utf-8")
    temporary = public.with_name(
        f".{public.name}.tls2trees-published-default-finalisation.tmp"
    )
    temporary.write_text("staged\n", encoding="utf-8")
    assert check("1").returncode == 0

    unrelated = repository / "unrelated.txt"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    rejected = check("1")
    assert rejected.returncode == 2
    assert "unrelated worktree path" in rejected.stderr

    unrelated.unlink()
    public.unlink()
    temporary.unlink()
    external = tmp_path / "external-publication-target.txt"
    external.write_text("must remain external\n", encoding="utf-8")
    public.symlink_to(external)
    rejected = check("1")
    assert rejected.returncode == 2
    assert "symbolic link at publication path" in rejected.stderr

    public.unlink()
    public.write_text("published\n", encoding="utf-8")
    temporary.symlink_to(external)
    rejected = check("1")
    assert rejected.returncode == 2
    assert "symbolic link at publication path" in rejected.stderr
