from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "methods/tls2trees/configs"
EVALUATION = ROOT / "methods/tls2trees/scripts/evaluation"
SLURM = ROOT / "methods/tls2trees/slurm/for_instance"


def load_freezer():
    path = EVALUATION / "freeze_tls2trees_development_final.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_summary(path: Path) -> None:
    rows = []
    values = {
        ("p04_min_points_50_lower_band", "leaf_off"): (0.004684, 0.003393, 0.042553, 0.002478),
        ("p02_min_points_50", "leaf_off"): (0.004608, 0.003034, 0.032787, 0.002478),
        ("p04_min_points_50_lower_band", "leaf_on"): (0.007026, 0.031746, 0.063830, 0.003717),
        ("p02_min_points_50", "leaf_on"): (0.009217, 0.034188, 0.065574, 0.004957),
    }
    for (candidate, target), (micro, mean, precision, recall) in values.items():
        rows.append(
            {
                "candidate_id": candidate,
                "target": target,
                "evaluated_plot_count": 21,
                "failed_or_invalid_plot_count": 0,
                "micro_f1": micro,
                "mean_plot_f1": mean,
                "precision": precision,
                "recall": recall,
            }
        )
    path.write_text(
        json.dumps(
            {
                "status": "stage2_completed",
                "valid_metric_count": 84,
                "expected_metric_count": 84,
                "held_out_test_accessed": False,
                "final_configuration_selected": False,
                "workflow_run_id": "stage2-run",
                "candidate_rankings_for_review": {
                    "leaf_off": ["p04_min_points_50_lower_band", "p02_min_points_50"],
                    "leaf_on": ["p02_min_points_50", "p04_min_points_50_lower_band"],
                },
                "aggregates": rows,
            }
        ),
        encoding="utf-8",
    )


def test_final_config_selects_one_candidate_per_target() -> None:
    payload = yaml.safe_load(
        (CONFIGS / "for_instance_development_tuned_final.yml").read_text()
    )
    selected = payload["selection"]["selected_by_target"]
    assert selected["leaf_off"]["candidate_id"] == "p04_min_points_50_lower_band"
    assert selected["leaf_on"]["candidate_id"] == "p02_min_points_50"
    assert payload["integrity"]["final_configuration_selected"] is True
    assert payload["integrity"]["held_out_test_accessed"] is False
    assert payload["run_gate"]["held_out_test_runnable"] is False


def test_final_freeze_binds_complete_development_evidence(tmp_path: Path) -> None:
    module = load_freezer()
    summary = tmp_path / "summary.json"
    write_summary(summary)
    payload = module.freeze(
        stage2_summary_path=summary,
        stage1_config_path=CONFIGS / "for_instance_development_tuned_stage1.yml",
        final_config_path=CONFIGS / "for_instance_development_tuned_final.yml",
        benchmark_commit="synthetic-commit",
    )

    assert payload["status"] == "development_tuned_configuration_frozen"
    assert payload["selected_by_target"]["leaf_off"]["candidate_id"].startswith("p04")
    assert payload["selected_by_target"]["leaf_on"]["candidate_id"].startswith("p02")
    assert payload["development_metric_count"] == 84
    assert payload["source_stage2_summary_sha256"] == module.sha256(summary)
    assert payload["final_configuration_selected"] is True
    assert payload["held_out_test_accessed"] is False
    assert payload["held_out_test_runnable"] is False
    assert payload["review_required_before_held_out_test"] is True


def test_final_freeze_shell_is_guarded_and_has_no_test_route() -> None:
    path = SLURM / "freeze_development_tuned_configuration.sh"
    source = path.read_text(encoding="utf-8")
    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert "TLS2TREES_FINAL_FREEZE_CONFIRMED" in source
    assert 'valid_metric_count\"] == 84' in source
    assert "held_out_test_accessed" in source
    assert "held_out_test_runnable" not in source
    assert "--split test" not in source
    assert "sbatch" not in source
