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
EVALUATION = ROOT / "methods/tls2trees/scripts/evaluation"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
CONFIGS = ROOT / "methods/tls2trees/configs"
for entry in (ROOT, RUNTIME):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_published_default_test_configuration_is_exactly_frozen() -> None:
    common = load_module(
        RUNTIME / "published_default_test_common.py",
        "published_default_test_config_contract",
    )
    workflow, _, published, _ = common.validate_frozen_configuration(
        CONFIGS / "for_instance_published_default_test.yml",
        CONFIGS / "for_instance_published_default.yml",
    )
    assert workflow["method"]["selected_from_for_instance_metrics"] is False
    assert workflow["method"]["configuration_changes_after_test_permitted"] is False
    assert workflow["frozen_semantic_parameters"] == published["semantic_parameters"]
    assert workflow["frozen_instance_parameters"] == published["instance_parameters"]
    assert workflow["targets"] == ["leaf_off", "leaf_on"]
    assert workflow["evaluation"]["expected_metric_count"] == 22
    assert workflow["retention"]["expected_file_count"] == 22


def test_shared_entrypoints_fail_closed_then_allow_published_default_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    common = load_module(
        RUNTIME / "for_instance_published_common.py",
        "published_default_test_split_contract",
    )
    observed: dict[str, object] = {}

    def fake_loader(path: Path, **kwargs):
        observed.update(kwargs)
        return {}, {"safe_plot_id": "CULS_plot_2_annotated"}

    monkeypatch.setattr(common, "load_and_verify_manifest_plot", fake_loader)
    plot_root, _ = common.resolve_held_out_test_plot_context(
        manifest_path=tmp_path / "manifest.json",
        task_index=0,
        output_root=tmp_path,
        run_id="published-test",
        variant="published_default",
    )
    assert observed["expected_split"] == "test"
    assert observed["allow_held_out_test"] is True
    assert plot_root.parts[-4:] == (
        "published_default",
        "test",
        "published-test",
        "CULS_plot_2_annotated",
    )
    with pytest.raises(ValueError, match="requires one of"):
        common.resolve_held_out_test_plot_context(
            manifest_path=tmp_path / "manifest.json",
            task_index=0,
            output_root=tmp_path,
            run_id="bad-test",
            variant="unreviewed_variant",
        )

    input_source = (
        ROOT
        / "methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
    ).read_text(encoding="utf-8")
    semantic_source = (
        RUNTIME / "run_for_instance_tls2trees_semantic.py"
    ).read_text(encoding="utf-8")
    instance_source = (
        RUNTIME / "run_for_instance_tls2trees_instance.py"
    ).read_text(encoding="utf-8")
    adapter_source = (
        EVALUATION / "adapt_for_instance_tls2trees_predictions.py"
    ).read_text(encoding="utf-8")
    for source in (input_source, semantic_source, instance_source, adapter_source):
        assert "--allow-held-out-test" in source
        assert "allow_held_out_test" in source
    assert "Held-out instance inference requires --allow-held-out-test" in instance_source


def _write_cache_fixture(
    tmp_path: Path, module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    published_path = tmp_path / "published.yml"
    published = {
        "method": {
            "variant": "published_default",
            "bundled_fsct_model": {"sha256": "model-sha"},
        },
        "published_preprocessing": {
            "tile_edge_length_m": 10.0,
            "downsample_voxel_length_m": 0.02,
        },
    }
    published_path.write_text(yaml.safe_dump(published), encoding="utf-8")
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text("status: frozen\n", encoding="utf-8")
    workflow = {"dataset": {"expected_plot_count": 11}}
    monkeypatch.setattr(
        module,
        "validate_frozen_configuration",
        lambda *_: (workflow, workflow_path, published, published_path),
    )
    monkeypatch.setattr(module, "validate_exact_manifest", lambda *_: None)
    monkeypatch.setattr(
        module,
        "verify_upstream",
        lambda *_: {
            "actual_commit": "upstream-commit",
            "model_sha256": "model-sha",
        },
    )
    row = {
        "task_index": 0,
        "safe_plot_id": "CULS_plot_2_annotated",
        "relative_path": "CULS/plot_2_annotated.las",
        "input_las": str(tmp_path / "plot.las"),
        "input_sha256": "input-sha",
        "point_count": 100,
        "reference_tree_count": 3,
    }
    manifest_payload = {"dataset_split": "test", "plots": [row]}
    monkeypatch.setattr(
        module,
        "load_and_verify_manifest_plot",
        lambda *_args, **_kwargs: (manifest_payload, row),
    )
    monkeypatch.setattr(
        module,
        "resolve_held_out_test_plot_context",
        lambda **kwargs: (
            Path(kwargs["output_root"])
            / "tls2trees/for_instance/published_default/test"
            / kwargs["run_id"]
            / row["safe_plot_id"],
            row,
        ),
    )
    manifest = tmp_path / "manifest.json"
    source_manifest = tmp_path / "source_manifest.json"
    manifest_text = json.dumps(manifest_payload, sort_keys=True)
    manifest.write_text(manifest_text, encoding="utf-8")
    source_manifest.write_text(manifest_text, encoding="utf-8")
    source_state = tmp_path / "source.env"
    source_state.write_text("SOURCE_STATE=verified\n", encoding="utf-8")
    source_output = tmp_path / "source_predictions"
    source_plot = (
        source_output
        / "tls2trees/for_instance/development_tuned/test/cache-run"
        / row["safe_plot_id"]
    )
    converted = source_plot / "converted"
    semantic_root = source_plot / "semantic"
    metadata = source_plot / "metadata"
    converted.mkdir(parents=True)
    semantic_root.mkdir()
    metadata.mkdir()
    source_map = converted / "source_map.npz"
    tile_index = converted / "tile_index.dat"
    tile = converted / "tile.ply"
    semantic_output = semantic_root / "tile.segmented.ply"
    source_map.write_bytes(b"source-map")
    tile_index.write_text("tile index\n", encoding="utf-8")
    tile.write_bytes(b"tile")
    semantic_output.write_bytes(b"semantic")
    conversion_path = converted / "conversion_metadata.json"
    conversion = {
        "split": "test",
        "task_index": 0,
        "safe_plot_id": row["safe_plot_id"],
        "relative_path": row["relative_path"],
        "input_sha256": row["input_sha256"],
        "manifest_sha256": module.sha256(source_manifest),
        "labels_stripped": True,
        "tile_size_m": 10.0,
        "downsample_voxel_size_m": 0.02,
        "source_map": str(source_map),
        "source_map_sha256": module.sha256(source_map),
        "tile_index": str(tile_index),
        "tile_index_sha256": module.sha256(tile_index),
        "tiles": [{"path": str(tile), "sha256": module.sha256(tile)}],
    }
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    semantic_path = metadata / "semantic_run.json"
    semantic = {
        "status": "completed",
        "split": "test",
        "task_index": 0,
        "safe_plot_id": row["safe_plot_id"],
        "relative_path": row["relative_path"],
        "config_sha256": module.sha256(published_path),
        "held_out_test_accessed": True,
        "tls2trees": {
            "actual_commit": "upstream-commit",
            "model_sha256": "model-sha",
        },
        "outputs": [
            {"path": str(semantic_output), "sha256": module.sha256(semantic_output)}
        ],
    }
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    return {
        "manifest": manifest,
        "source_manifest": source_manifest,
        "source_state": source_state,
        "source_output": source_output,
        "workflow_path": workflow_path,
        "published_path": published_path,
        "row": row,
    }


def test_semantic_cache_reuse_requires_exact_hashes_and_writes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(
        RUNTIME / "prepare_published_default_semantic_cache.py",
        "published_default_cache_contract",
    )
    fixture = _write_cache_fixture(tmp_path, module, monkeypatch)
    destination_root = tmp_path / "destination"
    evidence = module.verify_cache(
        manifest_path=fixture["manifest"],
        source_manifest_path=fixture["source_manifest"],
        task_index=0,
        source_output_root=fixture["source_output"],
        source_run_id="cache-run",
        source_variant="development_tuned",
        output_root=destination_root,
        run_id="published-run",
        workflow_config_path=fixture["workflow_path"],
        published_config_path=fixture["published_path"],
        tls2trees_repo=tmp_path / "upstream",
        source_state_path=fixture["source_state"],
        source_state_sha256=module.sha256(fixture["source_state"]),
    )
    assert evidence["status"] == "semantic_cache_reused"
    assert evidence["manifest_sha256"] == module.sha256(fixture["manifest"])
    assert evidence["input_las_sha256"] == "input-sha"
    assert evidence["published_config_sha256"] == module.sha256(
        fixture["published_path"]
    )
    plot_root = (
        destination_root
        / "tls2trees/for_instance/published_default/test/published-run"
        / fixture["row"]["safe_plot_id"]
    )
    assert (plot_root / "converted").is_symlink()
    assert (plot_root / "semantic").is_symlink()
    assert (plot_root / "metadata/semantic_cache_reuse.json").is_file()


def test_semantic_cache_mismatch_falls_back_before_creating_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module(
        RUNTIME / "prepare_published_default_semantic_cache.py",
        "published_default_cache_mismatch",
    )
    fixture = _write_cache_fixture(tmp_path, module, monkeypatch)
    fixture["source_manifest"].write_text("changed", encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(module.CacheNotReusable, match="byte-identical"):
        module.verify_cache(
            manifest_path=fixture["manifest"],
            source_manifest_path=fixture["source_manifest"],
            task_index=0,
            source_output_root=fixture["source_output"],
            source_run_id="cache-run",
            source_variant="development_tuned",
            output_root=destination,
            run_id="published-run",
            workflow_config_path=fixture["workflow_path"],
            published_config_path=fixture["published_path"],
            tls2trees_repo=tmp_path / "upstream",
            source_state_path=fixture["source_state"],
            source_state_sha256=module.sha256(fixture["source_state"]),
        )
    assert not destination.exists()


def test_published_default_summary_accepts_22_valid_empty_metrics(
    tmp_path: Path,
) -> None:
    summary_module = load_module(
        EVALUATION / "summarise_tls2trees_published_default_test.py",
        "published_default_summary_contract",
    )
    instance_module = load_module(
        RUNTIME / "run_for_instance_tls2trees_instance.py",
        "published_default_summary_parameters",
    )
    workflow_path = CONFIGS / "for_instance_published_default_test.yml"
    published_path = CONFIGS / "for_instance_published_default.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    published = yaml.safe_load(published_path.read_text(encoding="utf-8"))
    plots = []
    for index, relative_path in enumerate(
        workflow["dataset"]["exact_relative_paths"]
    ):
        plots.append(
            {
                "task_index": index,
                "safe_plot_id": relative_path.removesuffix(".las").replace("/", "_"),
                "relative_path": relative_path,
                "collection": relative_path.split("/", 1)[0],
                "input_sha256": f"input-{index}",
                "point_count": 49_709_922 if index == 0 else 0,
                "reference_tree_count": 323 if index == 0 else 0,
            }
        )
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(
        json.dumps({"dataset_split": "test", "plots": plots}), encoding="utf-8"
    )
    benchmark_path = tmp_path / "benchmark.yml"
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "evaluation": {
                    "protocol": "for_instance_tls2trees_source_row_class3_ignore",
                    "primary_mask": (
                        "union_of_reference_target_and_predicted_target_points_"
                        "excluding_class3_outpoints"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "data/predictions"
    run_id = "published-test"
    parameters = instance_module.resolved_instance_parameters(published)
    for plot in plots:
        plot_root = (
            output_root
            / "tls2trees/for_instance/published_default/test"
            / run_id
            / plot["safe_plot_id"]
        )
        metadata = plot_root / "metadata"
        metadata.mkdir(parents=True)
        common = {
            "split": "test",
            "task_index": plot["task_index"],
            "safe_plot_id": plot["safe_plot_id"],
            "relative_path": plot["relative_path"],
        }
        (metadata / "instance_run.json").write_text(
            json.dumps(
                {
                    **common,
                    "status": "completed_no_predictions",
                    "variant": "published_default",
                    "config_sha256": summary_module.sha256(published_path),
                    "held_out_test_accessed": True,
                    "resolved_instance_parameters": parameters,
                    "runtime_seconds": 1.0,
                    "peak_rss_gb": 0.1,
                }
            ),
            encoding="utf-8",
        )
        (metadata / "adapter_run.json").write_text(
            json.dumps(
                {
                    **common,
                    "status": "completed",
                    "variant": "published_default",
                    "held_out_test_accessed": True,
                    "runtime_seconds": 0.1,
                }
            ),
            encoding="utf-8",
        )
        (metadata / "semantic_run.json").write_text(
            json.dumps(
                {
                    **common,
                    "status": "completed",
                    "variant": "published_default",
                    "config_sha256": summary_module.sha256(published_path),
                    "held_out_test_accessed": True,
                }
            ),
            encoding="utf-8",
        )
        for target in ("leaf_off", "leaf_on"):
            aligned = plot_root / "predictions/aligned" / target
            aligned.mkdir(parents=True)
            (aligned / "source_row_predictions.npz").write_bytes(b"empty prediction")
            (aligned / "alignment_metadata.json").write_text(
                json.dumps({"schema_version": "tls2trees_for_instance_alignment"}),
                encoding="utf-8",
            )
            evaluation = plot_root / "evaluation" / target
            evaluation.mkdir(parents=True)
            (evaluation / "plot_metrics.json").write_text(
                json.dumps(
                    {
                        "split": "test",
                        "target": target,
                        "plot_id": plot["safe_plot_id"],
                        "relative_path": plot["relative_path"],
                        "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                        "evaluation_mask": (
                            "union_of_reference_target_and_predicted_target_points_"
                            "excluding_class3_outpoints"
                        ),
                        "semantic_ignore": {
                            "ignored_semantic_classes": [3],
                            "raw_prediction_instance_count": 0,
                        },
                        "status": "evaluated",
                        "safe_for_scoring": True,
                        "prediction_instance_count": 0,
                        "reference_instance_count": plot["reference_tree_count"],
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": plot["reference_tree_count"],
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "mean_matched_iou": 0.0,
                        "oversegmented_reference_count": 0,
                        "undersegmented_prediction_count": 0,
                    }
                ),
                encoding="utf-8",
            )
    summary, retention = summary_module.summarise(
        project_root=tmp_path,
        output_root=output_root,
        run_id=run_id,
        manifest_path=manifest_path,
        manifest_sha256=summary_module.sha256(manifest_path),
        workflow_config_path=workflow_path,
        workflow_config_sha256=summary_module.sha256(workflow_path),
        published_config_path=published_path,
        published_config_sha256=summary_module.sha256(published_path),
        benchmark_config_path=benchmark_path,
        benchmark_config_sha256=summary_module.sha256(benchmark_path),
    )
    assert summary["status"] == "published_default_test_completed"
    assert summary["valid_metric_count"] == 22
    assert summary["configuration_changed_after_test"] is False
    assert summary["semantic_cache_reused_plot_count"] == 0
    assert all(row["micro_f1"] == 0.0 for row in summary["aggregates"])
    assert retention["status"] == "retention_verified"
    assert retention["verified_prediction_files"] == 22
    assert all(not Path(row["relative_path"]).is_absolute() for row in retention["files"])


def test_published_default_slurm_chain_is_guarded_and_barkla_sized() -> None:
    paths = [
        SLURM / "submit_published_default_held_out_test.sh",
        SLURM / "monitor_published_default_held_out_test.sh",
        SLURM / "published_default_test_common.sh",
        SLURM / "prepare_published_default_test_manifest.sbatch",
        SLURM / "prepare_published_default_test_semantic.sbatch",
        SLURM / "evaluate_published_default_test_plot.sbatch",
        SLURM / "summarise_published_default_test.sbatch",
    ]
    for path in paths:
        checked = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True
        )
        assert checked.returncode == 0, f"{path}: {checked.stderr}"
    submit = paths[0].read_text(encoding="utf-8")
    assert "TLS2TREES_PUBLISHED_DEFAULT_TEST_CONFIRMED" in submit
    assert "TLS2TREES_REVIEWED_PUBLISHED_DEFAULT_CONFIG_SHA256" in submit
    assert "latest_published_default_test_state_file.txt" in submit
    assert '--array="0-10%2"' in submit
    assert '--array="0-10%4"' in submit
    assert 'dependency="afterok:$EVALUATE_JOB"' in submit
    semantic = paths[4].read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu-l40s-low" in semantic
    assert "#SBATCH --mem=192G" in semantic
    assert "semantic_fallback=dedicated_gpu_inference" in semantic
    evaluate = paths[5].read_text(encoding="utf-8")
    assert "#SBATCH --partition=nodes" in evaluate
    assert "#SBATCH --mem=32G" in evaluate
    assert "for TARGET in leaf_off leaf_on" in evaluate
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "configuration_changed_after_test=false" in combined
    for historical_suffix in (1, 2):
        assert f"v{historical_suffix}" not in combined
