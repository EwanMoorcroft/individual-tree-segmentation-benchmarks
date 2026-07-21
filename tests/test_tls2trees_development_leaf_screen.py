from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
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
        source_state_sha256="a" * 64,
        expected_run_id=str(payload["workflow_run_id"]),
        expected_source_run_id=str(payload["development_evidence_run_id"]),
        expected_semantic_cache_run_id="semantic_cache_run",
        expected_manifest_sha256=str(payload["manifest_sha256"]),
        expected_source_config_sha256=str(payload["candidate_config_sha256"]),
        expected_development_evidence_sha256=str(
            payload["development_evidence_sha256"]
        ),
    )

    public_plot = output_dir / finaliser.PUBLIC_PLOT_NAME
    public_candidate = output_dir / finaliser.PUBLIC_CANDIDATE_NAME
    public_provenance = output_dir / finaliser.PUBLIC_PROVENANCE_NAME
    assert public_plot.is_file()
    assert public_candidate.is_file()
    assert public_provenance.is_file()
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
            source_state_sha256="a" * 64,
            expected_run_id=str(payload["workflow_run_id"]),
            expected_source_run_id=str(payload["development_evidence_run_id"]),
            expected_semantic_cache_run_id="semantic_cache_run",
            expected_manifest_sha256=str(payload["manifest_sha256"]),
            expected_source_config_sha256=str(payload["candidate_config_sha256"]),
            expected_development_evidence_sha256=str(
                payload["development_evidence_sha256"]
            ),
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
            source_state_sha256="a" * 64,
            expected_run_id=str(payload["workflow_run_id"]),
            expected_source_run_id=str(payload["development_evidence_run_id"]),
            expected_semantic_cache_run_id="semantic_cache_run",
            expected_manifest_sha256=str(payload["manifest_sha256"]),
            expected_source_config_sha256=str(payload["candidate_config_sha256"]),
            expected_development_evidence_sha256=str(
                payload["development_evidence_sha256"]
            ),
        )


def test_leaf_screen_publication_entrypoint_is_guarded_and_syntactically_valid(
) -> None:
    source = PUBLICATION_SCRIPT.read_text(encoding="utf-8")
    assert "TLS2TREES_LEAF_SCREEN_PUBLICATION_CONFIRMED" in source
    assert "latest_leaf_screen_state_file.txt" in source
    assert "development_leaf_screen_chain_submitted" in source
    assert "SUMMARY_STATE" in source and '"COMPLETED"' in source
    assert "SOURCE_STATE_SHA256" in source
    assert "TLS2TREES_LEAF_SCREEN_MANIFEST_SHA256" in source
    assert "TLS2TREES_LEAF_SCREEN_CONFIG_SHA256" in source
    assert "TLS2TREES_LEAF_SCREEN_DEVELOPMENT_EVIDENCE_SHA256" in source
    assert "--allow-held-out-test" not in source
    subprocess.run(["bash", "-n", str(PUBLICATION_SCRIPT)], check=True)


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
