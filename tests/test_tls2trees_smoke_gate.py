from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def load_gate() -> ModuleType:
    path = (
        ROOT
        / "methods/tls2trees/scripts/evaluation/"
        "validate_for_instance_tls2trees_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("tls2trees_smoke_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_smoke_fixture(tmp_path: Path) -> tuple[ModuleType, Path, Path, str]:
    gate = load_gate()
    manifest = tmp_path / "manifest.json"
    output_root = tmp_path / "outputs"
    run_id = "smoke-001"
    safe_plot_id = "CULS_plot_1_annotated"
    write_json(
        manifest,
        {
            "dataset_split": "development",
            "plots": [
                {
                    "task_index": 0,
                    "split": "development",
                    "relative_path": "CULS/plot_1_annotated.las",
                    "safe_plot_id": safe_plot_id,
                }
            ],
        },
    )
    plot_root = (
        output_root
        / "tls2trees/for_instance/published_default/development"
        / run_id
        / safe_plot_id
    )
    write_json(
        plot_root / "converted/conversion_metadata.json",
        {
            "status": "prepared",
            "split": "development",
            "labels_stripped": True,
            "coordinate_frame": {"maximum_round_trip_delta_m": 0.0},
        },
    )
    for stage in ("semantic", "adapter"):
        write_json(
            plot_root / f"metadata/{stage}_run.json",
            {
                "status": "completed",
                "held_out_test_accessed": False,
                "runtime_seconds": 1.0,
                "peak_rss_gb": 2.0,
            },
        )
    write_json(
        plot_root / "metadata/instance_run.json",
        {
            "status": "completed",
            "held_out_test_accessed": False,
            "runtime_seconds": 1.0,
            "peak_rss_gb": 2.0,
            "prediction_inventory": {"leaf_off": [], "leaf_on": []},
        },
    )
    for target in ("leaf_off", "leaf_on"):
        target_root = plot_root / "predictions/aligned" / target
        target_root.mkdir(parents=True)
        npz = target_root / "source_row_predictions.npz"
        np.savez_compressed(npz, source_row_index=np.arange(2))
        write_json(
            target_root / "alignment_metadata.json",
            {
                "status": "passed",
                "aligned_prediction_npz": str(npz),
                "aligned_prediction_npz_sha256": gate.sha256(npz),
            },
        )
        write_json(
            plot_root / f"evaluation/{target}/plot_metrics.json",
            {
                "status": "evaluated",
                "safe_for_scoring": True,
                "evaluator": "for_instance_tls2trees_source_row_class3_ignore",
                "semantic_ignore": {"ignored_semantic_classes": [3]},
                "target": target,
                "split": "dev",
                "prediction_instance_count": 1,
                "reference_instance_count": 1,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mean_matched_iou": 0.0,
                "mean_unweighted_coverage": 0.0,
                "mean_weighted_coverage": 0.0,
                "evaluated_point_count": 2,
                "oversegmented_reference_count": 0,
                "undersegmented_prediction_count": 0,
            },
        )
    # Manifest identity is covered separately; these tests isolate the gate.
    gate.resolve_plot_context = lambda **kwargs: (
        plot_root,
        {
            "relative_path": "CULS/plot_1_annotated.las",
            "safe_plot_id": safe_plot_id,
        },
    )
    return gate, manifest, output_root, run_id


def test_smoke_gate_accepts_both_source_row_targets_and_still_blocks_full_runs(
    tmp_path: Path,
) -> None:
    gate, manifest, output_root, run_id = build_smoke_fixture(tmp_path)

    result = gate.validate_smoke(
        manifest_path=manifest,
        task_index=0,
        output_root=output_root,
        run_id=run_id,
    )

    assert result["status"] == "passed_automated_gates"
    assert result["held_out_test_accessed"] is False
    assert result["manual_alignment_review_required"] is True
    assert result["full_development_authorised"] is False
    assert result["held_out_test_authorised"] is False
    assert Path(result["metrics_csv"]).is_file()


def test_smoke_gate_rejects_stage_metadata_without_split_isolation(
    tmp_path: Path,
) -> None:
    gate, manifest, output_root, run_id = build_smoke_fixture(tmp_path)
    semantic = next(output_root.rglob("metadata/semantic_run.json"))
    value = json.loads(semantic.read_text(encoding="utf-8"))
    value["held_out_test_accessed"] = True
    write_json(semantic, value)

    with pytest.raises(ValueError, match="held-out isolation"):
        gate.validate_smoke(
            manifest_path=manifest,
            task_index=0,
            output_root=output_root,
            run_id=run_id,
        )
