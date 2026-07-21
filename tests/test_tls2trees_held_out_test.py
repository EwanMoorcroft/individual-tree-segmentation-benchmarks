from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
EVALUATION = ROOT / "methods/tls2trees/scripts/evaluation"


def load_summary_module():
    path = EVALUATION / "summarise_tls2trees_held_out_test.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "test_manifest.json"
    plots = [
        {
            "task_index": index,
            "safe_plot_id": f"site_plot_{index}",
            "relative_path": f"SITE/plot_{index}.las",
            "collection": "SITE",
            "point_count": 49_709_922 if index == 0 else 0,
            "reference_tree_count": 323 if index == 0 else 0,
        }
        for index in range(11)
    ]
    manifest.write_text(
        json.dumps({"dataset_split": "test", "plots": plots}), encoding="utf-8"
    )
    final = tmp_path / "final_selection.json"
    final.write_text(
        json.dumps(
            {
                "status": "development_tuned_configuration_frozen",
                "final_configuration_selected": True,
                "held_out_test_accessed": False,
                "selected_by_target": {
                    "leaf_off": {
                        "candidate_id": "p04_min_points_50_lower_band",
                        "stage1_candidate_index": 2,
                    },
                    "leaf_on": {
                        "candidate_id": "p02_min_points_50",
                        "stage1_candidate_index": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "predictions"
    workflow = "held-out-run"
    for target, candidate in (
        ("leaf_off", "p04_min_points_50_lower_band"),
        ("leaf_on", "p02_min_points_50"),
    ):
        run_id = f"{workflow}__{target}__{candidate}"
        for plot in plots:
            metrics = (
                output_root / "tls2trees" / "for_instance" / "development_tuned"
                / "test" / run_id / plot["safe_plot_id"] / "evaluation" / target
                / "plot_metrics.json"
            )
            metrics.parent.mkdir(parents=True)
            metrics.write_text(
                json.dumps(
                    {
                        "split": "test", "target": target,
                        "plot_id": plot["safe_plot_id"],
                        "relative_path": plot["relative_path"],
                        "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                        "evaluation_mask": (
                            "union_of_reference_target_and_predicted_target_points_"
                            "excluding_class3_outpoints"
                        ),
                        "semantic_ignore": {
                            "ignored_semantic_classes": [3],
                            "raw_prediction_instance_count": 2,
                        },
                        "status": "evaluated", "safe_for_scoring": True,
                        "prediction_instance_count": 2, "reference_instance_count": 3,
                        "true_positives": 1, "false_positives": 1, "false_negatives": 2,
                        "precision": 0.5, "recall": 1 / 3, "f1": 0.4,
                        "mean_matched_iou": 0.6,
                        "oversegmented_reference_count": 0,
                        "undersegmented_prediction_count": 0,
                    }
                ),
                encoding="utf-8",
            )
    return manifest, final, output_root


def test_held_out_summary_reports_only_frozen_target_routes(tmp_path: Path) -> None:
    module = load_summary_module()
    manifest, final, output_root = write_fixtures(tmp_path)
    payload = module.summarise(
        output_root=output_root,
        workflow_run_id="held-out-run",
        manifest_path=manifest,
        final_selection_path=final,
        final_selection_sha256=module.sha256(final),
    )
    assert payload["status"] == "held_out_test_completed"
    assert payload["valid_metric_count"] == 22
    assert payload["held_out_test_accessed"] is True
    assert payload["configuration_changed_after_test"] is False
    assert [(row["target"], row["candidate_id"]) for row in payload["aggregates"]] == [
        ("leaf_off", "p04_min_points_50_lower_band"),
        ("leaf_on", "p02_min_points_50"),
    ]
    assert all(row["micro_f1"] == pytest.approx(0.4) for row in payload["aggregates"])


def test_held_out_summary_rejects_unreviewed_freeze_hash(tmp_path: Path) -> None:
    module = load_summary_module()
    manifest, final, output_root = write_fixtures(tmp_path)
    with pytest.raises(RuntimeError, match="checksum changed"):
        module.summarise(
            output_root=output_root,
            workflow_run_id="held-out-run",
            manifest_path=manifest,
            final_selection_path=final,
            final_selection_sha256="0" * 64,
        )


def test_held_out_slurm_chain_is_one_time_guarded_and_syntactically_valid() -> None:
    paths = [
        SLURM / "submit_held_out_test.sh",
        SLURM / "monitor_held_out_test.sh",
        SLURM / "prepare_held_out_test_manifest.sbatch",
        SLURM / "prepare_semantic_held_out_test.sbatch",
        SLURM / "evaluate_held_out_test_candidate.sbatch",
        SLURM / "summarise_held_out_test.sbatch",
        SLURM / "recover_held_out_test_manifest_gate.sh",
    ]
    for path in paths:
        checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert checked.returncode == 0, f"{path}: {checked.stderr}"
    submit = paths[0].read_text(encoding="utf-8")
    assert "TLS2TREES_HELD_OUT_TEST_CONFIRMED" in submit
    assert "TLS2TREES_REVIEWED_FINAL_SELECTION_SHA256" in submit
    assert "latest_held_out_test_state_file.txt" in submit
    assert 'test ! -e "$LATEST_POINTER"' not in submit
    assert '--array="0-10%2"' in submit
    assert '--array="0-21%4"' in submit
    assert 'dependency="afterany:$CANDIDATE_JOB"' in submit


def test_development_entrypoints_require_explicit_test_authorisation() -> None:
    conversion = (
        ROOT / "methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
    ).read_text(encoding="utf-8")
    semantic = (
        ROOT / "methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT / "methods/tls2trees/scripts/evaluation/adapt_for_instance_tls2trees_predictions.py"
    ).read_text(encoding="utf-8")
    for source in (conversion, semantic, adapter):
        assert "--allow-held-out-test" in source
        assert "allow_held_out_test" in source


def test_test_plot_loaders_forward_explicit_held_out_authorisation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_module = load_module(
        ROOT / "methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py",
        "tls2trees_test_input_gate",
    )
    observed: dict[str, object] = {}

    def fake_loader(path: Path, **kwargs):
        observed.update(kwargs)
        return {}, {}

    monkeypatch.setattr(data_module, "load_and_verify_manifest_plot", fake_loader)
    data_module.load_manifest_plot(
        tmp_path / "manifest.json",
        0,
        "test",
        allow_held_out_test=True,
    )
    assert observed["expected_split"] == "test"
    assert observed["allow_held_out_test"] is True

    runtime = ROOT / "methods/tls2trees/scripts/runtime"
    monkeypatch.syspath_prepend(str(runtime))
    common = load_module(
        runtime / "for_instance_published_common.py",
        "tls2trees_test_common_gate",
    )
    observed.clear()

    def fake_common_loader(path: Path, **kwargs):
        observed.update(kwargs)
        return {}, {"safe_plot_id": "test_plot"}

    monkeypatch.setattr(common, "load_and_verify_manifest_plot", fake_common_loader)
    plot_root, _ = common.resolve_held_out_test_plot_context(
        manifest_path=tmp_path / "manifest.json",
        task_index=0,
        output_root=tmp_path,
        run_id="test-run",
        variant="development_tuned",
    )
    assert observed["expected_split"] == "test"
    assert observed["allow_held_out_test"] is True
    assert plot_root.parts[-4:] == (
        "development_tuned",
        "test",
        "test-run",
        "test_plot",
    )


def test_manifest_gate_recovery_is_signature_and_zero_metric_guarded() -> None:
    source = (SLURM / "recover_held_out_test_manifest_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "TLS2TREES_HELD_OUT_TEST_RECOVERY_CONFIRMED" in source
    assert "Held-out test manifest validation requires allow_held_out_test=True" in source
    assert 'p["valid_metric_count"] == 0' in source
    assert 'test "$FAILED_COUNT" = "11"' in source
    assert "--format=JobID%30,State" in source
    assert "recovery_preflight_failed_at_line" in source
    assert "mv \"$path\" \"$RECOVERY_ROOT/\"" in source
    assert "configuration_changed=false" in source
