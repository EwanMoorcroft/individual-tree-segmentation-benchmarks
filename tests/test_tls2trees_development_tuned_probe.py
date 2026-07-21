from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"
MANIFEST = (
    ROOT
    / "methods/tls2trees/configs/for_instance_development_tuned_compatibility_probe.yml"
)
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_candidates_are_exact_ordered_and_label_free() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert scope["variant"] == "development_tuned"
    assert scope["split"] == "development"
    assert scope["probe_target"] == "leaf_off"
    assert scope["held_out_test_accessed"] is False
    assert scope["reference_labels_accessed"] is False
    assert scope["accuracy_metrics_accessed"] is False
    assert scope["selection_uses_accuracy_metrics"] is False
    assert scope["benchmark_result"] is False
    assert payload["candidate_generation"]["ordering_frozen"] is True
    assert payload["candidate_generation"][
        "generated_before_development_tuned_accuracy_metrics"
    ] is True

    candidates = payload["candidates"]
    assert [item["candidate_index"] for item in candidates] == list(range(6))
    assert len({item["candidate_id"] for item in candidates}) == 6
    assert [item["candidate_id"] for item in candidates] == [
        "p00_published_instance_leaf_off_control",
        "p01_min_points_100",
        "p02_min_points_50",
        "p03_min_points_50_radius_015",
        "p04_min_points_50_lower_band",
        "p05_min_points_50_graph_3_gap_5",
    ]
    assert [item["parameters"]["find_stems_min_points"] for item in candidates] == [
        200,
        100,
        50,
        50,
        50,
        50,
    ]
    assert all(item["parameters"]["add_leaves"] is False for item in candidates)
    assert all(item["parameters"]["n_tiles"] == 5 for item in candidates)
    assert all(item["parameters"]["slice_thickness"] == 0.5 for item in candidates)
    assert payload["decision_gate"]["select_final_configuration"] is False
    assert payload["decision_gate"]["held_out_test_runnable"] is False


def test_probe_runner_validates_frozen_manifest_and_source_hashes(tmp_path: Path) -> None:
    runner = load_script(
        RUNTIME / "run_for_instance_tls2trees_compatibility_probe.py",
        "tls2trees_development_probe_runner",
    )
    manifest, resolved = runner.load_probe_manifest(str(MANIFEST))
    assert resolved == MANIFEST
    assert len(manifest["candidates"]) == 6

    plot_root = tmp_path / "plot_a"
    converted = plot_root / "converted"
    semantic_root = plot_root / "semantic"
    metadata = plot_root / "metadata"
    converted.mkdir(parents=True)
    semantic_root.mkdir()
    metadata.mkdir()
    tile_index = converted / "tile_index.dat"
    semantic_tile = semantic_root / "000000.downsample.segmented.ply"
    tile_index.write_text("tile index\n", encoding="utf-8")
    semantic_tile.write_text("semantic tile\n", encoding="utf-8")
    conversion_path = converted / "conversion_metadata.json"
    conversion_path.write_text(
        json.dumps(
            {
                "tile_index": str(tile_index),
                "tile_index_sha256": runner.sha256(tile_index),
            }
        ),
        encoding="utf-8",
    )
    semantic_path = metadata / "semantic_run.json"
    semantic_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "variant": "published_default",
                "split": "development",
                "held_out_test_accessed": False,
                "safe_plot_id": "plot_a",
                "relative_path": "CULS/plot_a.las",
                "conversion_metadata": str(conversion_path),
                "outputs": [
                    {"path": str(semantic_tile), "sha256": runner.sha256(semantic_tile)}
                ],
            }
        ),
        encoding="utf-8",
    )

    source = runner.verified_source(plot_root)
    assert source["outputs"] == [semantic_tile]
    assert source["tile_index"] == tile_index

    payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    payload["split"] = "test"
    semantic_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        runner.verified_source(plot_root)
    except ValueError as exc:
        assert "published-default development" in str(exc)
    else:
        raise AssertionError("test semantic source was not rejected")


def test_probe_summary_uses_nonmetric_viability_only(tmp_path: Path) -> None:
    summariser = load_script(
        ROOT / "methods/tls2trees/scripts/evaluation/summarise_tls2trees_compatibility_probe.py",
        "tls2trees_development_probe_summary",
    )
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    run_root = tmp_path / "tls2trees_for-instance_development_tuned_compatibility_probe_20260718_120000"
    probe_root = run_root / "plot_a" / "compatibility_probe"
    manifest_hash = summariser.sha256(MANIFEST)
    for candidate in manifest["candidates"]:
        metadata = (
            probe_root / candidate["candidate_id"] / "metadata" / "probe_run.json"
        )
        metadata.parent.mkdir(parents=True)
        viable = candidate["candidate_index"] == 3
        metadata.write_text(
            json.dumps(
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_manifest_sha256": manifest_hash,
                    "status": "viable_nonempty" if viable else "completed_no_predictions",
                    "prediction_inventory": {
                        "leaf_off": (
                            [{"point_count": 321, "path": "tree.leafoff.ply"}]
                            if viable
                            else []
                        ),
                        "leaf_on": [],
                    },
                    "runtime_seconds": 12.5,
                    "peak_rss_gb": 1.25,
                    "reference_labels_accessed": False,
                    "accuracy_metrics_accessed": False,
                    "selection_uses_accuracy_metrics": False,
                    "held_out_test_accessed": False,
                    "benchmark_result": False,
                }
            ),
            encoding="utf-8",
        )

    summary = summariser.summarise(run_root, MANIFEST)
    assert summary["status"] == "viable_candidates_found"
    assert summary["viable_candidate_ids"] == ["p03_min_points_50_radius_015"]
    assert summary["incomplete_candidate_ids"] == []
    assert summary["final_configuration_selected"] is False
    assert summary["accuracy_metrics_accessed"] is False
    assert summary["held_out_test_accessed"] is False
    assert summary["candidates"][3]["leaf_off_prediction_point_count"] == 321


def test_probe_slurm_route_is_guarded_development_only_and_syntax_valid() -> None:
    names = [
        "run_development_tuned_compatibility_probe.sbatch",
        "summarise_development_tuned_compatibility_probe.sbatch",
        "submit_development_tuned_compatibility_probe.sh",
        "monitor_development_tuned_compatibility_probe.sh",
    ]
    sources = {}
    for name in names:
        path = SLURM / name
        source = path.read_text(encoding="utf-8")
        sources[name] = source
        assert "set -euo pipefail" in source
        assert "/Users/" not in source
        completed = subprocess.run(
            ["bash", "-n", str(path)], check=False, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr

    submit = sources["submit_development_tuned_compatibility_probe.sh"]
    task = sources["run_development_tuned_compatibility_probe.sbatch"]
    summary = sources["summarise_development_tuned_compatibility_probe.sbatch"]
    monitor = sources["monitor_development_tuned_compatibility_probe.sh"]
    for name in (
        "submit_development_tuned_compatibility_probe.sh",
        "monitor_development_tuned_compatibility_probe.sh",
    ):
        assert (SLURM / name).stat().st_mode & 0o111, f"{name} is not executable"
    assert "TLS2TREES_DEV_TUNED_PROBE_CONFIRMED" in submit
    assert "development_tuned" in submit
    assert 'TLS2TREES_REQUESTED_SPLIT=development' in submit
    assert '--array="0-$((CANDIDATE_COUNT - 1))%2"' in submit
    assert 'afterany:$PROBE_JOB' in submit
    assert "evaluate_for_instance" not in submit
    assert "adapt_for_instance" not in submit
    assert "--split test" not in submit + task + summary
    assert "reference_labels_accessed=false" in submit
    assert "accuracy_metrics_accessed=false" in task
    assert "--skip-model-load" in task
    assert "squeue -j" in monitor
    assert "sacct -X -j" in monitor
    assert "viable_candidate_ids" in monitor
