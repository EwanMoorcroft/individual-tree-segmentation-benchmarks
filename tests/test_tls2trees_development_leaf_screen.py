from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
LEAF_CONFIG = (
    ROOT
    / "methods/tls2trees/configs/for_instance_development_tuned_leaf_screen.yml"
)
STAGE1_CONFIG = (
    ROOT / "methods/tls2trees/configs/for_instance_development_tuned_stage1.yml"
)
SEARCH_CONFIG = ROOT / "methods/tls2trees/configs/for_instance_search_space.yml"
SUMMARY_SCRIPT = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/summarise_tls2trees_development_leaf_screen.py"
)
FINALISER_SCRIPT = (
    ROOT
    / "methods/tls2trees/scripts/evaluation/"
    "finalise_tls2trees_development_leaf_screen.py"
)
PUBLICATION_SCRIPT = SLURM / "finalise_development_leaf_screen_results.sh"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_p02_parameters() -> dict[str, object]:
    source = yaml.safe_load(STAGE1_CONFIG.read_text(encoding="utf-8"))
    return next(
        candidate["parameters"]
        for candidate in source["candidates"]
        if candidate["candidate_id"] == "p02_min_points_50"
    )


def development_evidence(workflow_run_id: str = "source_stage1") -> dict[str, object]:
    return {
        "status": "stage1_completed",
        "split": "development",
        "workflow_run_id": workflow_run_id,
        "valid_metric_count": 40,
        "expected_metric_count": 40,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "candidate_parameters": {"p02_min_points_50": source_p02_parameters()},
    }


def test_leaf_screen_is_exact_declared_3x3_grid_with_fixed_p02_stems() -> None:
    config = yaml.safe_load(LEAF_CONFIG.read_text(encoding="utf-8"))
    search = yaml.safe_load(SEARCH_CONFIG.read_text(encoding="utf-8"))
    candidates = config["candidates"]
    voxel_values = search["searched_instance_parameters"][
        "add_leaves_voxel_length_m"
    ]["values"]
    edge_values = search["searched_instance_parameters"][
        "add_leaves_edge_length_m"
    ]["values"]
    observed = [
        (
            candidate["parameters"]["add_leaves_voxel_length"],
            candidate["parameters"]["add_leaves_edge_length"],
        )
        for candidate in candidates
    ]
    assert observed == list(itertools.product(voxel_values, edge_values))
    assert [candidate["candidate_index"] for candidate in candidates] == list(range(9))
    assert len({candidate["candidate_id"] for candidate in candidates}) == 9

    leaf_keys = {"add_leaves_voxel_length", "add_leaves_edge_length"}
    fixed_p02 = {
        key: value
        for key, value in source_p02_parameters().items()
        if key not in leaf_keys
    }
    assert all(
        {
            key: value
            for key, value in candidate["parameters"].items()
            if key not in leaf_keys
        }
        == fixed_p02
        for candidate in candidates
    )
    assert config["scope"]["targets"] == ["leaf_on"]
    assert config["scope"]["selection_uses_held_out_test_metrics"] is False
    assert config["scope"]["held_out_test_accessed"] is False
    assert (
        config["evaluation"]["evaluator"]
        == "for_instance_tls2trees_source_row_class3_ignore"
    )
    assert config["evaluation"]["ignored_semantic_classes"] == [3]
    assert config["evaluation"]["evaluation_mask"].endswith(
        "excluding_class3_outpoints"
    )
    assert config["run_gate"]["candidate_plot_task_count"] == 45
    assert config["run_gate"]["semantic_jobs_submitted"] is False
    assert config["run_gate"]["held_out_test_runnable"] is False


def test_candidate_runner_accepts_only_complete_development_evidence(
    tmp_path: Path,
) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        "tls2trees_leaf_screen_runner",
    )
    config, _ = runner.load_stage1_config(str(LEAF_CONFIG))
    evidence_path = tmp_path / "stage1_summary.json"
    evidence_path.write_text(
        json.dumps(development_evidence()), encoding="utf-8"
    )
    payload = runner.verify_development_stage1_evidence(
        evidence_path,
        runner.sha256(evidence_path),
        config,
    )
    assert payload["held_out_test_accessed"] is False

    crossed_boundary = development_evidence()
    crossed_boundary["held_out_test_accessed"] = True
    evidence_path.write_text(json.dumps(crossed_boundary), encoding="utf-8")
    with pytest.raises(ValueError, match="test boundary"):
        runner.verify_development_stage1_evidence(
            evidence_path,
            runner.sha256(evidence_path),
            config,
        )

    evidence_path.write_text(
        json.dumps(development_evidence()), encoding="utf-8"
    )
    altered = yaml.safe_load(LEAF_CONFIG.read_text(encoding="utf-8"))
    altered["candidates"][0]["parameters"]["find_stems_min_points"] = 51
    with pytest.raises(ValueError, match="changed a p02 stem parameter"):
        runner.verify_development_stage1_evidence(
            evidence_path,
            runner.sha256(evidence_path),
            altered,
        )


def test_leaf_screen_candidate_config_cannot_enter_test_split(tmp_path: Path) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_development_candidate.py",
        "tls2trees_leaf_screen_split_gate",
    )
    with pytest.raises(PermissionError, match="cannot access the held-out test"):
        runner.run_candidate(
            manifest_path=tmp_path / "test_manifest.json",
            task_index=0,
            source_plot_root=tmp_path / "semantic",
            output_root=tmp_path / "predictions",
            workflow_run_id="forbidden_test_run",
            candidate_run_id="forbidden_test_candidate",
            candidate_index=0,
            stage1_config_path=str(LEAF_CONFIG),
            tls2trees_repo=tmp_path / "upstream",
            development_evidence_path=tmp_path / "development_summary.json",
            development_evidence_sha256="0" * 64,
            split="test",
            target="leaf_on",
            final_selection_path=tmp_path / "final_selection.json",
            final_selection_sha256="1" * 64,
        )


def write_synthetic_leaf_screen(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    config = yaml.safe_load(LEAF_CONFIG.read_text(encoding="utf-8"))
    collections = ["CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"]
    plots = [
        {
            "task_index": index,
            "safe_plot_id": f"{collection}_plot",
            "relative_path": f"{collection}/plot.las",
            "collection": collection,
        }
        for index, collection in enumerate(collections)
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_split": "development",
                "plots": plots,
                "stage0_selection": [
                    {"stage0_index": index, "task_index": index}
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    source_run_id = "tls2trees_source_stage1"
    evidence_path = tmp_path / "stage1_summary.json"
    evidence_path.write_text(
        json.dumps(development_evidence(source_run_id)), encoding="utf-8"
    )
    evidence_sha256 = load_script(
        SUMMARY_SCRIPT, "tls2trees_leaf_screen_hash_helper"
    ).sha256(evidence_path)
    output_root = tmp_path / "predictions"
    workflow_run_id = "tls2trees_development_leaf_screen"
    config_sha256 = load_script(
        SUMMARY_SCRIPT, "tls2trees_leaf_screen_config_hash_helper"
    ).sha256(LEAF_CONFIG)
    for candidate in config["candidates"]:
        candidate_index = candidate["candidate_index"]
        candidate_id = candidate["candidate_id"]
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
                        "split": "development",
                        "target": "leaf_on",
                        "candidate_id": candidate_id,
                        "workflow_run_id": workflow_run_id,
                        "stage1_config_sha256": config_sha256,
                        "probe_summary": None,
                        "development_evidence_sha256": evidence_sha256,
                        "development_evidence_run_id": source_run_id,
                        "held_out_test_accessed": False,
                        "runtime_seconds": 10 - candidate_index / 10,
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
            tp = candidate_index + 1
            fp = 10 - tp
            fn = 12 - tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            metric_path = plot_root / "evaluation/leaf_on/plot_metrics.json"
            metric_path.parent.mkdir(parents=True)
            metric_path.write_text(
                json.dumps(
                    {
                        "evaluator": (
                            "for_instance_tls2trees_source_row_class3_ignore"
                        ),
                        "evaluation_mask": (
                            "union_of_reference_target_and_predicted_target_points_"
                            "excluding_class3_outpoints"
                        ),
                        "semantic_ignore": {
                            "ignored_semantic_classes": [3],
                            "ignored_predicted_point_count": 100,
                        },
                        "status": "evaluated",
                        "safe_for_scoring": True,
                        "split": "dev",
                        "target": "leaf_on",
                        "plot_id": plot["safe_plot_id"],
                        "prediction_instance_count": tp + fp,
                        "reference_instance_count": tp + fn,
                        "true_positives": tp,
                        "false_positives": fp,
                        "false_negatives": fn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "mean_matched_iou": 0.6,
                        "oversegmented_reference_count": fp,
                        "undersegmented_prediction_count": fn,
                    }
                ),
                encoding="utf-8",
            )
    return output_root, manifest_path, evidence_path, workflow_run_id


def test_leaf_screen_summary_ranks_45_class3_ignore_metrics(
    tmp_path: Path,
) -> None:
    summariser = load_script(SUMMARY_SCRIPT, "tls2trees_leaf_screen_summary")
    output_root, manifest_path, evidence_path, workflow_run_id = (
        write_synthetic_leaf_screen(tmp_path)
    )
    payload = summariser.summarise(
        output_root=output_root,
        workflow_run_id=workflow_run_id,
        manifest_path=manifest_path,
        candidate_config_path=LEAF_CONFIG,
        development_evidence_path=evidence_path,
        development_evidence_sha256=summariser.sha256(evidence_path),
    )
    assert payload["status"] == "development_leaf_screen_completed"
    assert payload["valid_metric_count"] == payload["expected_metric_count"] == 45
    assert len(payload["plot_metrics"]) == 45
    assert len(payload["aggregates"]) == 9
    assert payload["held_out_test_accessed"] is False
    assert payload["final_configuration_selected"] is False
    assert payload["evaluator"] == (
        "for_instance_tls2trees_source_row_class3_ignore"
    )
    assert payload["candidate_ranking_for_review"][0] == "leaf_v100_e20"
    assert len(payload["top_three_candidate_ids_for_review"]) == 3


def test_leaf_screen_summary_rejects_wrong_evaluation_protocol(tmp_path: Path) -> None:
    summariser = load_script(
        SUMMARY_SCRIPT, "tls2trees_leaf_screen_protocol_rejection"
    )
    output_root, manifest_path, evidence_path, workflow_run_id = (
        write_synthetic_leaf_screen(tmp_path)
    )
    metric_path = next(output_root.rglob("evaluation/leaf_on/plot_metrics.json"))
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    metric["evaluator"] = "for_instance_tls2trees_source_row_without_class3_ignore"
    metric_path.write_text(json.dumps(metric), encoding="utf-8")
    with pytest.raises(ValueError, match="Metric protocol"):
        summariser.summarise(
            output_root=output_root,
            workflow_run_id=workflow_run_id,
            manifest_path=manifest_path,
            candidate_config_path=LEAF_CONFIG,
            development_evidence_path=evidence_path,
            development_evidence_sha256=summariser.sha256(evidence_path),
        )


def write_leaf_screen_summary_artifacts(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path, Path]:
    summariser = load_script(
        SUMMARY_SCRIPT, "tls2trees_leaf_screen_publication_source"
    )
    output_root, manifest_path, evidence_path, workflow_run_id = (
        write_synthetic_leaf_screen(tmp_path)
    )
    payload = summariser.summarise(
        output_root=output_root,
        workflow_run_id=workflow_run_id,
        manifest_path=manifest_path,
        candidate_config_path=LEAF_CONFIG,
        development_evidence_path=evidence_path,
        development_evidence_sha256=summariser.sha256(evidence_path),
    )
    summary_path = tmp_path / "leaf_screen_summary.json"
    plot_path = tmp_path / "plot_metrics.csv"
    candidate_path = tmp_path / "candidate_summary.csv"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summariser.write_csv(plot_path, payload["plot_metrics"])
    summariser.write_csv(candidate_path, payload["aggregates"])
    return payload, summary_path, plot_path, candidate_path


def all_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in all_strings(child)]
    return [value] if isinstance(value, str) else []


def run_leaf_screen_finaliser(
    finaliser: ModuleType,
    payload: dict[str, object],
    summary_path: Path,
    plot_path: Path,
    candidate_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    return finaliser.finalise(
        summary_path=summary_path,
        source_plot_csv=plot_path,
        source_candidate_csv=candidate_path,
        candidate_config_path=LEAF_CONFIG,
        output_dir=output_dir,
        project_root=output_dir.parent,
        source_state_sha256="a" * 64,
        expected_run_id=str(payload["workflow_run_id"]),
        expected_source_run_id=str(payload["development_evidence_run_id"]),
        expected_semantic_cache_run_id="semantic_cache_run",
        expected_manifest_sha256=str(payload["manifest_sha256"]),
        expected_source_config_sha256=str(payload["candidate_config_sha256"]),
        expected_development_evidence_sha256=str(
            payload["development_evidence_sha256"]
        ),
        source_benchmark_commit="a" * 40,
        publication_benchmark_commit="b" * 40,
    )


def test_leaf_screen_finaliser_writes_exact_public_safe_evidence(
    tmp_path: Path,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    # The completed Barkla evidence used the same semantics under a numeric tag.
    payload["evaluator"] = "for_instance_tls2trees_source_row_v2_class3_ignore"
    payload["candidate_config_sha256"] = "b" * 64
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_publication_finaliser"
    )
    output_dir = tmp_path / "public"
    provenance = finaliser.finalise(
        summary_path=summary_path,
        source_plot_csv=plot_path,
        source_candidate_csv=candidate_path,
        candidate_config_path=LEAF_CONFIG,
        output_dir=output_dir,
        project_root=output_dir.parent,
        source_state_sha256="a" * 64,
        expected_run_id=str(payload["workflow_run_id"]),
        expected_source_run_id=str(payload["development_evidence_run_id"]),
        expected_semantic_cache_run_id="semantic_cache_run",
        expected_manifest_sha256=str(payload["manifest_sha256"]),
        expected_source_config_sha256=str(payload["candidate_config_sha256"]),
        expected_development_evidence_sha256=str(
            payload["development_evidence_sha256"]
        ),
        source_benchmark_commit="a" * 40,
        publication_benchmark_commit="b" * 40,
    )

    public_plot = output_dir / finaliser.PUBLIC_PLOT_NAME
    public_candidate = output_dir / finaliser.PUBLIC_CANDIDATE_NAME
    public_provenance = output_dir / finaliser.PUBLIC_PROVENANCE_NAME
    assert public_plot.is_file()
    assert public_candidate.is_file()
    assert public_provenance.is_file()
    assert all(
        path.stat().st_nlink == 1
        for path in (public_plot, public_candidate, public_provenance)
    )
    with public_plot.open(encoding="utf-8", newline="") as handle:
        plot_reader = csv.DictReader(handle)
        plot_rows = list(plot_reader)
        assert plot_reader.fieldnames == list(finaliser.PUBLIC_PLOT_FIELDS)
    with public_candidate.open(encoding="utf-8", newline="") as handle:
        candidate_reader = csv.DictReader(handle)
        candidate_rows = list(candidate_reader)
        assert candidate_reader.fieldnames == list(
            finaliser.PUBLIC_CANDIDATE_FIELDS
        )
    assert len(plot_rows) == 45
    assert len(candidate_rows) == 9
    assert "metrics_path" not in plot_rows[0]
    assert all(row["status"] == "evaluated" for row in plot_rows)
    assert all(row["safe_for_scoring"] == "True" for row in plot_rows)
    assert {
        (row["candidate_id"], row["collection"]) for row in plot_rows
    } == {
        (candidate["candidate_id"], collection)
        for candidate in yaml.safe_load(LEAF_CONFIG.read_text())["candidates"]
        for collection in ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")
    }

    first_candidate = candidate_rows[0]
    matching_plots = [
        row
        for row in plot_rows
        if row["candidate_id"] == first_candidate["candidate_id"]
    ]
    assert int(first_candidate["true_positives"]) == sum(
        int(row["true_positives"]) for row in matching_plots
    )
    assert int(first_candidate["false_positives"]) == sum(
        int(row["false_positives"]) for row in matching_plots
    )
    assert int(first_candidate["false_negatives"]) == sum(
        int(row["false_negatives"]) for row in matching_plots
    )

    recorded = json.loads(public_provenance.read_text(encoding="utf-8"))
    assert recorded == provenance
    assert provenance["status"] == (
        "development_leaf_screen_publication_completed"
    )
    assert provenance["valid_metric_count"] == 45
    assert provenance["candidate_count"] == 9
    assert provenance["held_out_test_accessed"] is False
    assert provenance["final_configuration_selected"] is False
    assert provenance["source_benchmark_commit"] == "a" * 40
    assert provenance["publication_benchmark_commit"] == "b" * 40
    assert provenance["evaluation_protocol"] == (
        "for_instance_tls2trees_source_row_class3_ignore"
    )
    assert provenance["source_artifact_hashes"]["summary_sha256"] == (
        hashlib.sha256(summary_path.read_bytes()).hexdigest()
    )
    assert provenance["public_artifacts"]["plot_results"]["sha256"] == (
        hashlib.sha256(public_plot.read_bytes()).hexdigest()
    )
    assert provenance["public_artifacts"]["candidate_results"]["sha256"] == (
        hashlib.sha256(public_candidate.read_bytes()).hexdigest()
    )
    forbidden = ("/users/", "/home/", "/mnt/", "/users/", "fastscratch")
    assert not [
        value
        for value in all_strings([plot_rows, candidate_rows, provenance])
        if Path(value).is_absolute()
        or any(token in value.casefold() for token in forbidden)
    ]


def test_leaf_screen_finaliser_rejects_tampering_and_test_access(
    tmp_path: Path,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_publication_rejection"
    )
    candidate_rows = list(csv.DictReader(candidate_path.open(encoding="utf-8")))
    candidate_rows[0]["true_positives"] = str(
        int(candidate_rows[0]["true_positives"]) + 1
    )
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with pytest.raises(ValueError, match="differs from embedded summary"):
        finaliser.finalise(
            summary_path=summary_path,
            source_plot_csv=plot_path,
            source_candidate_csv=candidate_path,
            candidate_config_path=LEAF_CONFIG,
            output_dir=tmp_path / "tampered-public",
            project_root=tmp_path,
            source_state_sha256="a" * 64,
            expected_run_id=str(payload["workflow_run_id"]),
            expected_source_run_id=str(payload["development_evidence_run_id"]),
            expected_semantic_cache_run_id="semantic_cache_run",
            expected_manifest_sha256=str(payload["manifest_sha256"]),
            expected_source_config_sha256=str(payload["candidate_config_sha256"]),
            expected_development_evidence_sha256=str(
                payload["development_evidence_sha256"]
            ),
            source_benchmark_commit="a" * 40,
            publication_benchmark_commit="b" * 40,
        )

    payload["held_out_test_accessed"] = True
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="held_out_test_accessed"):
        finaliser.finalise(
            summary_path=summary_path,
            source_plot_csv=plot_path,
            source_candidate_csv=candidate_path,
            candidate_config_path=LEAF_CONFIG,
            output_dir=tmp_path / "test-access-public",
            project_root=tmp_path,
            source_state_sha256="a" * 64,
            expected_run_id=str(payload["workflow_run_id"]),
            expected_source_run_id=str(payload["development_evidence_run_id"]),
            expected_semantic_cache_run_id="semantic_cache_run",
            expected_manifest_sha256=str(payload["manifest_sha256"]),
            expected_source_config_sha256=str(payload["candidate_config_sha256"]),
            expected_development_evidence_sha256=str(
                payload["development_evidence_sha256"]
            ),
            source_benchmark_commit="a" * 40,
            publication_benchmark_commit="b" * 40,
        )


def test_leaf_screen_finaliser_normal_rerun_completes_interrupted_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_interrupted_publication"
    )
    output_dir = tmp_path / "interrupted-public"
    real_link = finaliser.os.link
    successful_links = 0

    def interrupt_after_first_link(source: Path, target: Path) -> None:
        nonlocal successful_links
        if successful_links == 1:
            raise RuntimeError("simulated publication interruption")
        real_link(source, target)
        successful_links += 1

    monkeypatch.setattr(finaliser.os, "link", interrupt_after_first_link)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        run_leaf_screen_finaliser(
            finaliser,
            payload,
            summary_path,
            plot_path,
            candidate_path,
            output_dir,
        )

    plot_output = output_dir / finaliser.PUBLIC_PLOT_NAME
    candidate_output = output_dir / finaliser.PUBLIC_CANDIDATE_NAME
    provenance_output = output_dir / finaliser.PUBLIC_PROVENANCE_NAME
    stage_dir = output_dir / finaliser.PUBLICATION_STAGE_NAME
    assert plot_output.is_file()
    assert not candidate_output.exists()
    assert not provenance_output.exists()
    assert not stage_dir.exists()

    monkeypatch.setattr(finaliser.os, "link", real_link)
    run_leaf_screen_finaliser(
        finaliser,
        payload,
        summary_path,
        plot_path,
        candidate_path,
        output_dir,
    )
    retry_output = capsys.readouterr().out
    assert "publication_outputs_created=2" in retry_output
    assert "publication_outputs_retained=1" in retry_output
    outputs = (plot_output, candidate_output, provenance_output)
    before = {path: (path.stat().st_ino, path.read_bytes()) for path in outputs}

    run_leaf_screen_finaliser(
        finaliser,
        payload,
        summary_path,
        plot_path,
        candidate_path,
        output_dir,
    )
    idempotent_output = capsys.readouterr().out
    assert "publication_outputs_created=0" in idempotent_output
    assert "publication_outputs_retained=3" in idempotent_output
    assert before == {
        path: (path.stat().st_ino, path.read_bytes()) for path in outputs
    }
    assert not stage_dir.exists()


def test_leaf_screen_publication_preserves_target_changed_between_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_concurrent_target_change"
    )
    output_dir = tmp_path / "concurrent-target-public"
    real_link = finaliser.os.link
    links = 0

    def change_next_target(source: Path, target: Path) -> None:
        nonlocal links
        real_link(source, target)
        links += 1
        if links == 1:
            (output_dir / finaliser.PUBLIC_CANDIDATE_NAME).write_bytes(
                b"concurrent manual edit\n"
            )

    monkeypatch.setattr(finaliser.os, "link", change_next_target)
    with pytest.raises(FileExistsError, match="conflicts with rendered bundle"):
        run_leaf_screen_finaliser(
            finaliser,
            payload,
            summary_path,
            plot_path,
            candidate_path,
            output_dir,
        )

    assert (output_dir / finaliser.PUBLIC_CANDIDATE_NAME).read_bytes() == (
        b"concurrent manual edit\n"
    )


def test_leaf_screen_publication_verifies_complete_bundle_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_complete_bundle_verification"
    )
    output_dir = tmp_path / "post-publication-change"
    real_link = finaliser.os.link
    links = 0

    def change_first_target_after_last_link(source: Path, target: Path) -> None:
        nonlocal links
        real_link(source, target)
        links += 1
        if links == 3:
            (output_dir / finaliser.PUBLIC_PLOT_NAME).write_bytes(
                b"changed after publication\n"
            )

    monkeypatch.setattr(finaliser.os, "link", change_first_target_after_last_link)
    with pytest.raises(FileExistsError, match="conflicts with rendered bundle"):
        run_leaf_screen_finaliser(
            finaliser,
            payload,
            summary_path,
            plot_path,
            candidate_path,
            output_dir,
        )


def test_leaf_screen_finaliser_rejects_conflict_before_filling_bundle(
    tmp_path: Path,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_publication_conflict"
    )
    output_dir = tmp_path / "conflicting-public"
    run_leaf_screen_finaliser(
        finaliser,
        payload,
        summary_path,
        plot_path,
        candidate_path,
        output_dir,
    )
    plot_output = output_dir / finaliser.PUBLIC_PLOT_NAME
    candidate_output = output_dir / finaliser.PUBLIC_CANDIDATE_NAME
    provenance_output = output_dir / finaliser.PUBLIC_PROVENANCE_NAME
    exact_plot = plot_output.read_bytes()
    candidate_output.write_bytes(b"conflicting candidate evidence\n")
    provenance_output.unlink()
    stage_dir = output_dir / finaliser.PUBLICATION_STAGE_NAME
    stage_dir.mkdir()
    # Model a process killed after linking a staged member into the public
    # bundle but before its finally block removed the staging name.
    finaliser.os.link(
        candidate_output,
        stage_dir / finaliser.PUBLIC_CANDIDATE_NAME,
    )

    with pytest.raises(FileExistsError, match="staging content conflicts"):
        run_leaf_screen_finaliser(
            finaliser,
            payload,
            summary_path,
            plot_path,
            candidate_path,
            output_dir,
        )
    assert plot_output.read_bytes() == exact_plot
    assert candidate_output.read_bytes() == b"conflicting candidate evidence\n"
    assert not provenance_output.exists()
    preserved_stage = stage_dir / finaliser.PUBLIC_CANDIDATE_NAME
    assert preserved_stage.is_file()
    assert preserved_stage.read_bytes() == b"conflicting candidate evidence\n"


def test_leaf_screen_finaliser_validates_full_stage_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, summary_path, plot_path, candidate_path = (
        write_leaf_screen_summary_artifacts(tmp_path)
    )
    finaliser = load_script(
        FINALISER_SCRIPT, "tls2trees_leaf_screen_staged_validation"
    )
    output_dir = tmp_path / "invalid-stage-public"

    def reject_staged_bundle(**_: object) -> None:
        raise ValueError("simulated staged-bundle validation failure")

    monkeypatch.setattr(finaliser, "validate_staged_bundle", reject_staged_bundle)
    with pytest.raises(ValueError, match="staged-bundle validation failure"):
        run_leaf_screen_finaliser(
            finaliser,
            payload,
            summary_path,
            plot_path,
            candidate_path,
            output_dir,
        )
    assert not any(
        (output_dir / name).exists()
        for name in (
            finaliser.PUBLIC_PLOT_NAME,
            finaliser.PUBLIC_CANDIDATE_NAME,
            finaliser.PUBLIC_PROVENANCE_NAME,
        )
    )
    assert not (output_dir / finaliser.PUBLICATION_STAGE_NAME).exists()


def test_leaf_screen_publication_entrypoint_is_guarded_and_syntactically_valid(
) -> None:
    source = PUBLICATION_SCRIPT.read_text(encoding="utf-8")
    assert "TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED" in source
    assert "TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED" in source
    assert '[[ "$RECOVERY_CONFIRMED" == "0" ]]' in source
    assert "latest_leaf_screen_state_file.txt" in source
    assert "development_leaf_screen_chain_submitted" in source
    assert "SUMMARY_STATE" in source and '"COMPLETED"' in source
    assert "SOURCE_STATE_SHA256" in source
    assert "STATE_SNAPSHOT" in source
    assert "POST_SOURCE_STATE_SHA256" in source
    assert 'source "$STATE_SNAPSHOT"' in source
    assert "TLS2TREES_LEAF_SCREEN_MANIFEST_SHA256" in source
    assert "TLS2TREES_LEAF_SCREEN_CONFIG_SHA256" in source
    assert "TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_SHA256" in source
    assert "PUBLIC_OUTPUTS=(" in source
    assert "RECOVERY_PATHS=(\"${PUBLIC_OUTPUTS[@]}\")" in source
    assert "--porcelain=v1 -z --untracked-files=all" in source
    assert '[[ "$STATUS" != " M" && "$STATUS" != "??" ]]' in source
    assert '[[ "$PATHNAME" == "$ALLOWED_PATH" ]]' in source
    assert '[[ -L "$PATHNAME" ]]' in source
    assert "WORKTREE_VIOLATIONS" in source
    assert ":(exclude)" not in source
    assert ".tls2trees_development_leaf_screen_publication.staging" in source
    assert "test ! -e" not in source
    assert "--allow-held-out-test" not in source
    assert "TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT" in source
    assert "PUBLICATION_BENCHMARK_COMMIT=$(git rev-parse HEAD)" in source
    assert "git merge-base --is-ancestor" in source
    assert "TLS2TREES_LEAF_SCREEN_DIVERGED_SOURCE_COMMIT" in source
    assert 'SOURCE_HISTORY_RELATION="reviewed_divergence"' in source
    assert "source_history_relation=$SOURCE_HISTORY_RELATION" in source
    assert '--project-root "$PWD"' in source
    assert '--source-benchmark-commit "$SOURCE_BENCHMARK_COMMIT"' in source
    assert (
        '--publication-benchmark-commit "$PUBLICATION_BENCHMARK_COMMIT"'
        in source
    )
    assert '--recovery-confirmed "$RECOVERY_CONFIRMED"' in source
    subprocess.run(["bash", "-n", str(PUBLICATION_SCRIPT)], check=True)


def test_leaf_screen_publication_recovery_is_explicit_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "bin"
    treebench = tmp_path / "treebench"
    for path in (repository, fake_bin, treebench / "bin"):
        path.mkdir(parents=True)

    finaliser = repository / (
        "methods/tls2trees/scripts/evaluation/"
        "finalise_tls2trees_development_leaf_screen.py"
    )
    config = repository / (
        "methods/tls2trees/configs/"
        "for_instance_development_tuned_leaf_screen.yml"
    )
    for path in (finaliser, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    fake_python = treebench / "bin/python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_sacct = fake_bin / "sacct"
    fake_sacct.write_text(
        "#!/usr/bin/env bash\nprintf '123|COMPLETED|\\n'\n",
        encoding="utf-8",
    )
    fake_sacct.chmod(0o755)

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    evidence_paths = {
        "TLS2TREES_LEAF_SCREEN_SUMMARY_JSON": tmp_path / "summary.json",
        "TLS2TREES_LEAF_SCREEN_PLOT_CSV": tmp_path / "plot.csv",
        "TLS2TREES_LEAF_SCREEN_AGGREGATE_CSV": tmp_path / "candidate.csv",
    }
    for path in evidence_paths.values():
        path.write_text("fixture\n", encoding="utf-8")
    state = tmp_path / "state.env"
    state_values = {
        "TLS2TREES_LEAF_SCREEN_RUN_ID": "leaf_run",
        "TLS2TREES_LEAF_SCREEN_SUBMISSION_STATUS": (
            "development_leaf_screen_chain_submitted"
        ),
        "TLS2TREES_LEAF_SCREEN_SUMMARY_JOB": "123",
        "TLS2TREES_LEAF_SCREEN_SOURCE_RUN_ID": "source_run",
        "TLS2TREES_LEAF_SCREEN_SOURCE_SEMANTIC_CACHE_RUN_ID": "semantic_run",
        "TLS2TREES_LEAF_SCREEN_MANIFEST_SHA256": "a" * 64,
        "TLS2TREES_LEAF_SCREEN_CONFIG_SHA256": "b" * 64,
        "TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_SHA256": "c" * 64,
        "TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT": commit,
        "TLS2TREES_LEAF_SCREEN_TREEBENCH_ENV": str(treebench),
        **{key: str(path) for key, path in evidence_paths.items()},
    }
    state.write_text(
        "".join(f"{key}={json.dumps(value)}\n" for key, value in state_values.items()),
        encoding="utf-8",
    )

    def run(
        recovery: str | None,
        diverged_source_commit: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "TLS2TREES_PROJECT_ROOT": str(repository),
                "TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED": "1",
            }
        )
        if recovery is None:
            environment.pop(
                "TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED", None
            )
        else:
            environment[
                "TLS2TREES_LEAF_SCREEN_PUBLICATION_RECOVERY_CONFIRMED"
            ] = recovery
        if diverged_source_commit is None:
            environment.pop(
                "TLS2TREES_LEAF_SCREEN_DIVERGED_SOURCE_COMMIT", None
            )
        else:
            environment["TLS2TREES_LEAF_SCREEN_DIVERGED_SOURCE_COMMIT"] = (
                diverged_source_commit
            )
        return subprocess.run(
            ["bash", str(PUBLICATION_SCRIPT), str(state)],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
        )

    assert run(None).returncode == 0
    assert run("unexpected").returncode == 2

    empty_tree = subprocess.run(
        ["git", "-C", str(repository), "hash-object", "-t", "tree", "/dev/null"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unrelated_commit = subprocess.run(
        [
            "git", "-C", str(repository), "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "commit-tree", empty_tree,
        ],
        input="unrelated source\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state_values["TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT"] = unrelated_commit
    state.write_text(
        "".join(
            f"{key}={json.dumps(value)}\n"
            for key, value in state_values.items()
        ),
        encoding="utf-8",
    )
    rejected_divergence = run(None)
    assert rejected_divergence.returncode == 2
    assert "not an ancestor" in rejected_divergence.stderr
    assert run(None, "0" * 40).returncode == 2
    assert run(None, unrelated_commit).returncode == 0

    state_values["TLS2TREES_LEAF_SCREEN_BENCHMARK_COMMIT"] = commit
    state.write_text(
        "".join(
            f"{key}={json.dumps(value)}\n"
            for key, value in state_values.items()
        ),
        encoding="utf-8",
    )

    public = repository / (
        "methods/tls2trees/examples/"
        "tls2trees_development_leaf_screen_plot_results.csv"
    )
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text("interrupted publication\n", encoding="utf-8")
    assert run(None).returncode == 2
    assert run("1").returncode == 0

    public.unlink()
    external = tmp_path / "external-publication-target.txt"
    external.write_text("must remain external\n", encoding="utf-8")
    public.symlink_to(external)
    rejected = run("1")
    assert rejected.returncode == 2
    assert "symlink" in rejected.stderr
    assert external.read_text(encoding="utf-8") == "must remain external\n"


def test_leaf_screen_slurm_reuses_semantics_and_submits_no_semantic_job() -> None:
    submit = (SLURM / "submit_development_leaf_screen.sh").read_text(
        encoding="utf-8"
    )
    evaluate = (
        SLURM / "evaluate_development_leaf_screen_candidate.sbatch"
    ).read_text(encoding="utf-8")
    assert '--array="0-44%4"' in submit
    assert "TLS2TREES_SOURCE_SEMANTIC_CACHE_RUN_ID" in submit
    assert "semantic_jobs_submitted=false" in submit
    assert "prepare_semantic" not in submit
    assert "--development-evidence-json" in evaluate
    assert "--target leaf_on" in evaluate
    assert "--split development" in evaluate
    assert "for TARGET in" not in evaluate
    assert "--allow-held-out-test" not in submit + evaluate

    for path in (
        SLURM / "submit_development_leaf_screen.sh",
        SLURM / "monitor_development_leaf_screen.sh",
        SLURM / "evaluate_development_leaf_screen_candidate.sbatch",
        SLURM / "summarise_development_leaf_screen.sbatch",
        PUBLICATION_SCRIPT,
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)
