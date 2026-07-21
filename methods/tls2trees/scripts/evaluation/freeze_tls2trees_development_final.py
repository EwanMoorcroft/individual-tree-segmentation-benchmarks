"""Freeze one TLS2trees development-tuned configuration per target."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPECTED_SELECTION = {
    "leaf_off": "p04_min_points_50_lower_band",
    "leaf_on": "p02_min_points_50",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def freeze(
    *,
    stage2_summary_path: Path,
    stage1_config_path: Path,
    final_config_path: Path,
    benchmark_commit: str,
) -> dict[str, Any]:
    stage2_summary_path = stage2_summary_path.expanduser().resolve()
    stage1_config_path = stage1_config_path.expanduser().resolve()
    final_config_path = final_config_path.expanduser().resolve()
    summary = json.loads(stage2_summary_path.read_text(encoding="utf-8"))
    stage1 = load_yaml(stage1_config_path)
    final = load_yaml(final_config_path)

    if (
        summary.get("status") != "stage2_completed"
        or int(summary.get("valid_metric_count", -1)) != 84
        or int(summary.get("expected_metric_count", -1)) != 84
        or summary.get("held_out_test_accessed") is not False
        or summary.get("final_configuration_selected") is not False
    ):
        raise ValueError("Stage 2 summary is not complete and development-only")
    integrity = final.get("integrity", {})
    if (
        final.get("dataset", {}).get("selection_split") != "development"
        or integrity.get("selection_uses_held_out_test_metrics") is not False
        or integrity.get("held_out_test_accessed") is not False
        or integrity.get("final_configuration_selected") is not True
        or final.get("run_gate", {}).get("held_out_test_runnable") is not False
    ):
        raise ValueError("Final configuration crosses the pre-test review boundary")

    configured = {
        target: record["candidate_id"]
        for target, record in final["selection"]["selected_by_target"].items()
    }
    if configured != EXPECTED_SELECTION:
        raise ValueError("Unexpected target-specific final configuration")
    rankings = summary.get("candidate_rankings_for_review", {})
    if any(rankings.get(target, [None])[0] != candidate for target, candidate in configured.items()):
        raise ValueError("Final configuration does not match the complete Stage 2 rankings")

    aggregates = {
        (row["candidate_id"], row["target"]): row for row in summary["aggregates"]
    }
    for target in EXPECTED_SELECTION:
        for candidate in ("p04_min_points_50_lower_band", "p02_min_points_50"):
            row = aggregates[(candidate, target)]
            if (
                int(row["evaluated_plot_count"]) != 21
                or int(row["failed_or_invalid_plot_count"]) != 0
            ):
                raise ValueError(f"Incomplete Stage 2 aggregate: {candidate}:{target}")
    p04_off = aggregates[("p04_min_points_50_lower_band", "leaf_off")]
    p02_off = aggregates[("p02_min_points_50", "leaf_off")]
    p02_on = aggregates[("p02_min_points_50", "leaf_on")]
    p04_on = aggregates[("p04_min_points_50_lower_band", "leaf_on")]
    if not (
        p04_off["micro_f1"] > p02_off["micro_f1"]
        and p04_off["mean_plot_f1"] > p02_off["mean_plot_f1"]
        and p04_off["precision"] > p02_off["precision"]
        and p04_off["recall"] == p02_off["recall"]
    ):
        raise ValueError("Leaf-off evidence does not support p04")
    if not (
        p02_on["micro_f1"] > p04_on["micro_f1"]
        and p02_on["mean_plot_f1"] > p04_on["mean_plot_f1"]
        and p02_on["precision"] > p04_on["precision"]
        and p02_on["recall"] > p04_on["recall"]
    ):
        raise ValueError("Leaf-on evidence does not support p02")

    stage1_candidates = {row["candidate_id"]: row for row in stage1["candidates"]}
    frozen_targets: dict[str, Any] = {}
    for target, candidate_id in configured.items():
        source = stage1_candidates[candidate_id]
        configured_index = final["selection"]["selected_by_target"][target][
            "stage1_candidate_index"
        ]
        if int(source["candidate_index"]) != int(configured_index):
            raise ValueError(f"Candidate index mismatch for {target}")
        frozen_targets[target] = {
            "candidate_id": candidate_id,
            "stage1_candidate_index": int(source["candidate_index"]),
            "parameters": source["parameters"],
            "development_metrics": aggregates[(candidate_id, target)],
            "selection_reason": final["selection"]["selected_by_target"][target][
                "reason"
            ],
        }

    return {
        "schema_version": 1,
        "status": "development_tuned_configuration_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "selection_split": "development",
        "source_stage2_run_id": summary["workflow_run_id"],
        "source_stage2_summary": str(stage2_summary_path),
        "source_stage2_summary_sha256": sha256(stage2_summary_path),
        "source_stage1_config": str(stage1_config_path),
        "source_stage1_config_sha256": sha256(stage1_config_path),
        "final_config": str(final_config_path),
        "final_config_sha256": sha256(final_config_path),
        "benchmark_commit": benchmark_commit,
        "selected_by_target": frozen_targets,
        "development_metric_count": 84,
        "development_plot_count": 21,
        "development_accuracy_metrics_used": True,
        "held_out_test_accessed": False,
        "held_out_test_runnable": False,
        "final_configuration_selected": True,
        "review_required_before_held_out_test": True,
        "next_gate": "review_frozen_configuration_before_one_time_held_out_test",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-summary-json", required=True)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--final-config", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Final selection already exists: {output}")
    payload = freeze(
        stage2_summary_path=Path(args.stage2_summary_json),
        stage1_config_path=Path(args.stage1_config),
        final_config_path=Path(args.final_config),
        benchmark_commit=args.benchmark_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"status={payload['status']}")
    for target, record in payload["selected_by_target"].items():
        metrics = record["development_metrics"]
        print(
            f"{target}={record['candidate_id']} "
            f"micro_f1={metrics['micro_f1']:.6f} "
            f"mean_plot_f1={metrics['mean_plot_f1']:.6f}"
        )
    print("final_configuration_selected=true")
    print("held_out_test_accessed=false")
    print("held_out_test_runnable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
