"""Aggregate five-site TLS2trees development metrics without selecting a winner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from for_instance_published_common import sha256, utc_now, write_json


TARGETS = ("leaf_off", "leaf_on")


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def stage0_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_task = {int(row["task_index"]): row for row in manifest["plots"]}
    rows = []
    for selected in manifest["stage0_selection"]:
        row = by_task[int(selected["task_index"])]
        rows.append({"stage0_index": int(selected["stage0_index"]), **row})
    if [row["stage0_index"] for row in rows] != list(range(5)):
        raise ValueError("Development manifest does not contain the frozen five-site Stage 0")
    if {row["collection"] for row in rows} != {"CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"}:
        raise ValueError("Development Stage 0 does not cover the five required collections")
    return rows


def aggregate(rows: list[dict[str, Any]], candidate_id: str, target: str) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["target"] == target
        and row["status"] == "evaluated"
        and row["safe_for_scoring"] is True
    ]
    tp = sum(int(row["true_positives"]) for row in valid)
    fp = sum(int(row["false_positives"]) for row in valid)
    fn = sum(int(row["false_negatives"]) for row in valid)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "candidate_id": candidate_id,
        "target": target,
        "expected_plot_count": 5,
        "evaluated_plot_count": len(valid),
        "failed_or_invalid_plot_count": 5 - len(valid),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": (
            sum(float(row["f1"]) for row in valid) / len(valid) if valid else 0.0
        ),
        "total_instance_runtime_seconds": sum(
            float(row["instance_runtime_seconds"] or 0.0) for row in valid
        ),
        "maximum_instance_peak_rss_gb": max(
            (float(row["instance_peak_rss_gb"] or 0.0) for row in valid), default=0.0
        ),
        "per_site_f1": {row["collection"]: row["f1"] for row in valid},
    }


def summarise(
    *,
    output_root: Path,
    workflow_run_id: str,
    manifest_path: Path,
    stage1_config_path: Path,
    probe_summary_path: Path,
    probe_summary_sha256: str,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    stage1_config_path = stage1_config_path.expanduser().resolve()
    probe_summary_path = probe_summary_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(stage1_config_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_split") != "development":
        raise ValueError("Stage 1 summary only accepts the exact development manifest")
    if sha256(probe_summary_path) != probe_summary_sha256:
        raise RuntimeError("Probe summary checksum changed before Stage 1 aggregation")
    probe = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    required = config["probe_promotion"]["required_viable_candidate_ids"]
    if probe.get("status") != "viable_candidates_found" or probe.get("viable_candidate_ids") != required:
        raise ValueError("Probe evidence does not match the frozen Stage 1 promotion")
    plots = stage0_rows(manifest)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in config["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_run_id = f"{workflow_run_id}__{candidate_id}"
        for plot in plots:
            plot_root = (
                output_root
                / "tls2trees"
                / "for_instance"
                / "development_tuned"
                / "development"
                / candidate_run_id
                / plot["safe_plot_id"]
            )
            instance_path = plot_root / "metadata" / "instance_run.json"
            adapter_path = plot_root / "metadata" / "adapter_run.json"
            instance: dict[str, Any] = {}
            adapter: dict[str, Any] = {}
            if instance_path.is_file():
                instance = json.loads(instance_path.read_text(encoding="utf-8"))
                if (
                    instance.get("candidate_id") != candidate_id
                    or instance.get("workflow_run_id") != workflow_run_id
                    or instance.get("stage1_config_sha256") != sha256(stage1_config_path)
                    or instance.get("probe_summary_sha256") != probe_summary_sha256
                    or instance.get("held_out_test_accessed") is not False
                ):
                    raise ValueError(f"Instance provenance mismatch: {instance_path}")
            if adapter_path.is_file():
                adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
                if (
                    adapter.get("variant") != "development_tuned"
                    or adapter.get("split") != "development"
                    or adapter.get("held_out_test_accessed") is not False
                ):
                    raise ValueError(f"Adapter provenance mismatch: {adapter_path}")
            for target in TARGETS:
                metric_path = plot_root / "evaluation" / target / "plot_metrics.json"
                row: dict[str, Any] = {
                    "candidate_index": candidate["candidate_index"],
                    "candidate_id": candidate_id,
                    "stage0_index": plot["stage0_index"],
                    "collection": plot["collection"],
                    "safe_plot_id": plot["safe_plot_id"],
                    "relative_path": plot["relative_path"],
                    "target": target,
                    "status": "missing",
                    "safe_for_scoring": False,
                    "prediction_instance_count": None,
                    "reference_instance_count": None,
                    "true_positives": None,
                    "false_positives": None,
                    "false_negatives": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "mean_matched_iou": None,
                    "instance_runtime_seconds": instance.get("runtime_seconds"),
                    "instance_peak_rss_gb": instance.get("peak_rss_gb"),
                    "adapter_runtime_seconds": adapter.get("runtime_seconds"),
                    "metrics_path": str(metric_path),
                    "metrics_sha256": None,
                    "error": instance.get("error"),
                }
                if metric_path.is_file():
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    if metrics.get("split") != "dev" or metrics.get("target") != target:
                        raise ValueError(f"Metric provenance mismatch: {metric_path}")
                    row.update(
                        {
                            key: metrics.get(key)
                            for key in (
                                "status",
                                "safe_for_scoring",
                                "prediction_instance_count",
                                "reference_instance_count",
                                "true_positives",
                                "false_positives",
                                "false_negatives",
                                "precision",
                                "recall",
                                "f1",
                                "mean_matched_iou",
                            )
                        }
                    )
                    row["metrics_sha256"] = sha256(metric_path)
                if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
                    missing.append(f"{candidate_id}:{plot['safe_plot_id']}:{target}")
                rows.append(row)
    aggregates = [
        aggregate(rows, candidate["candidate_id"], target)
        for candidate in config["candidates"]
        for target in TARGETS
    ]
    rankings = {
        target: [
            item["candidate_id"]
            for item in sorted(
                (item for item in aggregates if item["target"] == target),
                key=lambda item: (
                    -item["micro_f1"],
                    -item["mean_plot_f1"],
                    item["failed_or_invalid_plot_count"],
                    item["candidate_id"],
                ),
            )
        ]
        for target in TARGETS
    }
    return {
        "schema_version": 1,
        "status": "stage1_completed" if not missing else "stage1_incomplete",
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "development",
        "workflow_run_id": workflow_run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "stage1_config": str(stage1_config_path),
        "stage1_config_sha256": sha256(stage1_config_path),
        "probe_summary": str(probe_summary_path),
        "probe_summary_sha256": probe_summary_sha256,
        "expected_metric_count": 40,
        "valid_metric_count": 40 - len(missing),
        "incomplete_tasks": missing,
        "development_reference_labels_accessed": True,
        "development_accuracy_metrics_computed": True,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "candidate_rankings_for_review": rankings,
        "candidate_parameters": {
            candidate["candidate_id"]: candidate["parameters"]
            for candidate in config["candidates"]
        },
        "plot_metrics": rows,
        "aggregates": aggregates,
        "next_gate": "review_metrics_then_freeze_at_most_three_candidates_per_target",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()}
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--probe-summary-json", required=True)
    parser.add_argument("--probe-summary-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plot-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [Path(args.output_json), Path(args.plot_csv), Path(args.aggregate_csv)]
    if any(path.exists() for path in outputs):
        print("Refusing existing Stage 1 summary output", file=sys.stderr)
        return 2
    try:
        payload = summarise(
            output_root=Path(args.output_root),
            workflow_run_id=args.workflow_run_id,
            manifest_path=Path(args.manifest_json),
            stage1_config_path=Path(args.stage1_config),
            probe_summary_path=Path(args.probe_summary_json),
            probe_summary_sha256=args.probe_summary_sha256,
        )
        write_json(outputs[0], payload)
        write_csv(outputs[1], payload["plot_metrics"])
        write_csv(outputs[2], payload["aggregates"])
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    for target, ranking in payload["candidate_rankings_for_review"].items():
        print(f"{target}_ranking=" + ",".join(ranking))
    print("final_configuration_selected=false")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
