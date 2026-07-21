"""Freeze the bounded TLS2trees Stage 2 candidate set from Stage 1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


TARGETS = ("leaf_off", "leaf_on")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def freeze(
    *,
    stage1_summary_path: Path,
    stage1_config_path: Path,
    stage2_config_path: Path,
    benchmark_commit: str,
) -> dict[str, Any]:
    stage1_summary_path = stage1_summary_path.expanduser().resolve()
    stage1_config_path = stage1_config_path.expanduser().resolve()
    stage2_config_path = stage2_config_path.expanduser().resolve()
    summary = json.loads(stage1_summary_path.read_text(encoding="utf-8"))
    stage1 = load_yaml(stage1_config_path)
    stage2 = load_yaml(stage2_config_path)

    if (
        summary.get("status") != "stage1_completed"
        or int(summary.get("valid_metric_count", -1)) != 40
        or summary.get("held_out_test_accessed") is not False
        or summary.get("final_configuration_selected") is not False
    ):
        raise ValueError("Stage 1 summary is not a complete development-only result")
    if stage2.get("dataset", {}).get("allowed_split") != "development":
        raise ValueError("Stage 2 config is not development-only")
    scope = stage2.get("scope", {})
    if (
        scope.get("held_out_test_accessed") is not False
        or scope.get("selection_uses_held_out_test_metrics") is not False
        or scope.get("final_configuration_selected") is not False
    ):
        raise ValueError("Stage 2 config crosses the selection boundary")

    stage1_candidates = {item["candidate_id"]: item for item in stage1["candidates"]}
    selected = stage2["selection"]["selected_candidates"]
    selected_ids = [item["candidate_id"] for item in selected]
    if selected_ids != ["p04_min_points_50_lower_band", "p02_min_points_50"]:
        raise ValueError("Unexpected Stage 2 candidate ordering")
    if len(selected_ids) > 3 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Stage 2 must freeze at most three unique candidates")
    for item in selected:
        source = stage1_candidates.get(item["candidate_id"])
        if source is None or int(source["candidate_index"]) != int(
            item["stage1_candidate_index"]
        ):
            raise ValueError("Stage 2 candidate does not match the frozen Stage 1 config")
        if item.get("targets") != list(TARGETS):
            raise ValueError("Each Stage 2 candidate must cover both explicit targets")

    aggregates = {
        (item["candidate_id"], item["target"]): item
        for item in summary["aggregates"]
    }
    rankings = summary.get("candidate_rankings_for_review", {})
    if any(rankings.get(target, [None])[0] != selected_ids[0] for target in TARGETS):
        raise ValueError("p04 is not the Stage 1 leader for both targets")
    p04 = [aggregates[(selected_ids[0], target)] for target in TARGETS]
    p02 = [aggregates[(selected_ids[1], target)] for target in TARGETS]
    if any(
        leader["micro_f1"] < comparator["micro_f1"]
        or leader["mean_plot_f1"] < comparator["mean_plot_f1"]
        or leader["failed_or_invalid_plot_count"] != 0
        or comparator["failed_or_invalid_plot_count"] != 0
        for leader, comparator in zip(p04, p02, strict=True)
    ):
        raise ValueError("Stage 1 evidence does not support the frozen candidate set")
    if float(p04[0]["total_instance_runtime_seconds"]) >= float(
        p02[0]["total_instance_runtime_seconds"]
    ):
        raise ValueError("p04 did not retain its Stage 1 runtime advantage")

    plot_keys = {
        (row["safe_plot_id"], row["relative_path"], row["collection"])
        for row in summary["plot_metrics"]
    }
    if len(plot_keys) != 5:
        raise ValueError("Stage 1 selection evidence does not cover five unique sites")

    selected_records = []
    for item in selected:
        candidate_id = item["candidate_id"]
        selected_records.append(
            {
                **item,
                "parameters": stage1_candidates[candidate_id]["parameters"],
                "stage1_target_metrics": {
                    target: aggregates[(candidate_id, target)] for target in TARGETS
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "frozen_for_full_development_stage2",
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "development",
        "source_stage1_run_id": summary["workflow_run_id"],
        "source_stage1_summary": str(stage1_summary_path),
        "source_stage1_summary_sha256": sha256(stage1_summary_path),
        "source_stage1_config": str(stage1_config_path),
        "source_stage1_config_sha256": sha256(stage1_config_path),
        "stage2_config": str(stage2_config_path),
        "stage2_config_sha256": sha256(stage2_config_path),
        "benchmark_commit": benchmark_commit,
        "selection_criterion": stage2["selection"]["decision_basis"],
        "selection_rationale": stage2["selection"]["rationale"],
        "development_plots_used": [
            {
                "safe_plot_id": safe_plot_id,
                "relative_path": relative_path,
                "collection": collection,
            }
            for safe_plot_id, relative_path, collection in sorted(plot_keys)
        ],
        "selected_candidates": selected_records,
        "excluded_candidates": stage2["selection"]["excluded_candidates"],
        "full_development_plot_count": 21,
        "expected_stage2_metric_count": 84,
        "confirmation_no_test_metrics_used": True,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "held_out_test_runnable": False,
        "next_gate": "run_and_review_full_development_stage2",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-summary-json", required=True)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_json).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Stage 2 selection manifest already exists: {output}")
    payload = freeze(
        stage1_summary_path=Path(args.stage1_summary_json),
        stage1_config_path=Path(args.stage1_config),
        stage2_config_path=Path(args.stage2_config),
        benchmark_commit=args.benchmark_commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"status={payload['status']}")
    print(
        "selected_candidate_ids="
        + ",".join(item["candidate_id"] for item in payload["selected_candidates"])
    )
    print("held_out_test_accessed=false")
    print("final_configuration_selected=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
