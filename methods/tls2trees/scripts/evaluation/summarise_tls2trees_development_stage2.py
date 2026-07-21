"""Aggregate the two frozen TLS2trees candidates across all development plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = ("leaf_off", "leaf_on")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def aggregate(rows: list[dict[str, Any]], candidate_id: str, target: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["target"] == target
        and row["status"] == "evaluated"
        and row["safe_for_scoring"] is True
    ]
    tp = sum(int(row["true_positives"]) for row in selected)
    fp = sum(int(row["false_positives"]) for row in selected)
    fn = sum(int(row["false_negatives"]) for row in selected)
    precision, recall, f1 = prf(tp, fp, fn)
    by_collection: dict[str, list[float]] = {}
    for row in selected:
        by_collection.setdefault(row["collection"], []).append(float(row["f1"]))
    return {
        "candidate_id": candidate_id,
        "target": target,
        "expected_plot_count": 21,
        "evaluated_plot_count": len(selected),
        "failed_or_invalid_plot_count": 21 - len(selected),
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in selected
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in selected
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": (
            sum(float(row["f1"]) for row in selected) / len(selected)
            if selected
            else 0.0
        ),
        "mean_collection_f1": {
            collection: sum(values) / len(values)
            for collection, values in sorted(by_collection.items())
        },
        "oversegmented_reference_count": sum(
            int(row["oversegmented_reference_count"]) for row in selected
        ),
        "undersegmented_prediction_count": sum(
            int(row["undersegmented_prediction_count"]) for row in selected
        ),
        "total_instance_runtime_seconds": sum(
            float(row["instance_runtime_seconds"] or 0.0) for row in selected
        ),
        "maximum_instance_peak_rss_gb": max(
            (float(row["instance_peak_rss_gb"] or 0.0) for row in selected),
            default=0.0,
        ),
    }


def summarise(
    *,
    output_root: Path,
    workflow_run_id: str,
    manifest_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_split") != "development" or len(manifest["plots"]) != 21:
        raise ValueError("Stage 2 requires the exact 21-plot development manifest")
    if [int(row["task_index"]) for row in manifest["plots"]] != list(range(21)):
        raise ValueError("Development task indexes must be contiguous 0..20")
    if (
        selection.get("status") != "frozen_for_full_development_stage2"
        or selection.get("held_out_test_accessed") is not False
        or selection.get("final_configuration_selected") is not False
        or selection.get("confirmation_no_test_metrics_used") is not True
    ):
        raise ValueError("Stage 2 selection manifest is not frozen and development-only")
    selected_candidates = selection["selected_candidates"]
    candidate_ids = [item["candidate_id"] for item in selected_candidates]
    if candidate_ids != ["p04_min_points_50_lower_band", "p02_min_points_50"]:
        raise ValueError("Unexpected Stage 2 candidate set")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for candidate in selected_candidates:
        candidate_id = candidate["candidate_id"]
        candidate_run_id = f"{workflow_run_id}__{candidate_id}"
        for plot in manifest["plots"]:
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
                    "stage2_candidate_index": candidate["stage2_candidate_index"],
                    "stage1_candidate_index": candidate["stage1_candidate_index"],
                    "candidate_id": candidate_id,
                    "task_index": int(plot["task_index"]),
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
                    "oversegmented_reference_count": None,
                    "undersegmented_prediction_count": None,
                    "instance_runtime_seconds": instance.get("runtime_seconds"),
                    "instance_peak_rss_gb": instance.get("peak_rss_gb"),
                    "adapter_runtime_seconds": adapter.get("runtime_seconds"),
                    "metrics_path": str(metric_path),
                    "metrics_sha256": None,
                    "error": instance.get("error"),
                }
                if metric_path.is_file():
                    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    if (
                        metrics.get("split") != "dev"
                        or metrics.get("target") != target
                        or metrics.get("plot_id") != plot["safe_plot_id"]
                    ):
                        raise ValueError(f"Metric provenance mismatch: {metric_path}")
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
                        "oversegmented_reference_count",
                        "undersegmented_prediction_count",
                    ):
                        row[key] = metrics.get(key)
                    row["metrics_sha256"] = sha256(metric_path)
                if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
                    missing.append(f"{candidate_id}:{plot['safe_plot_id']}:{target}")
                rows.append(row)

    aggregates = [
        aggregate(rows, candidate_id, target)
        for candidate_id in candidate_ids
        for target in TARGETS
    ]
    rankings = {
        target: [
            item["candidate_id"]
            for item in sorted(
                (item for item in aggregates if item["target"] == target),
                key=lambda item: (
                    item["failed_or_invalid_plot_count"],
                    -item["mean_plot_f1"],
                    -item["micro_f1"],
                    item["oversegmented_reference_count"],
                    item["undersegmented_prediction_count"],
                    item["total_instance_runtime_seconds"],
                    item["candidate_id"],
                ),
            )
        ]
        for target in TARGETS
    }
    return {
        "schema_version": 1,
        "status": "stage2_completed" if not missing else "stage2_incomplete",
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "development",
        "workflow_run_id": workflow_run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256(selection_path),
        "expected_plot_count": 21,
        "expected_metric_count": 84,
        "valid_metric_count": 84 - len(missing),
        "incomplete_tasks": missing,
        "development_reference_labels_accessed": True,
        "development_accuracy_metrics_computed": True,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "candidate_rankings_for_review": rankings,
        "plot_metrics": rows,
        "aggregates": aggregates,
        "next_gate": "review_full_development_metrics_then_freeze_one_candidate_per_target",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, dict)
                    else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plot-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [Path(args.output_json), Path(args.plot_csv), Path(args.aggregate_csv)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing existing Stage 2 summary output")
    payload = summarise(
        output_root=Path(args.output_root),
        workflow_run_id=args.workflow_run_id,
        manifest_path=Path(args.manifest_json),
        selection_path=Path(args.selection_json),
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(outputs[1], payload["plot_metrics"])
    write_csv(outputs[2], payload["aggregates"])
    print(f"status={payload['status']}")
    print(f"valid_metrics={payload['valid_metric_count']}/84")
    for target, ranking in payload["candidate_rankings_for_review"].items():
        print(f"{target}_ranking=" + ",".join(ranking))
    print("final_configuration_selected=false")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
