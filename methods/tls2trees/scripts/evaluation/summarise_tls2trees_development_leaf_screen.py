"""Aggregate the frozen TLS2trees leaf-attachment grid on five development plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


EXPECTED_EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
EXPECTED_EVALUATION_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
REQUIRED_COLLECTIONS = {"CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"}


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


def stage0_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("dataset_split") != "development":
        raise ValueError("Leaf screen only accepts the development manifest")
    by_task = {int(row["task_index"]): row for row in manifest["plots"]}
    rows = [
        {
            "stage0_index": int(selected["stage0_index"]),
            **by_task[int(selected["task_index"])],
        }
        for selected in manifest["stage0_selection"]
    ]
    if [row["stage0_index"] for row in rows] != list(range(5)):
        raise ValueError("Development manifest has no frozen five-plot Stage 0")
    if {row["collection"] for row in rows} != REQUIRED_COLLECTIONS:
        raise ValueError("Development Stage 0 must cover all five collections")
    return rows


def aggregate(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row["candidate_id"] == candidate_id
        and row["status"] == "evaluated"
        and row["safe_for_scoring"] is True
    ]
    tp = sum(int(row["true_positives"]) for row in valid)
    fp = sum(int(row["false_positives"]) for row in valid)
    fn = sum(int(row["false_negatives"]) for row in valid)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "candidate_id": candidate_id,
        "target": "leaf_on",
        "expected_plot_count": 5,
        "evaluated_plot_count": len(valid),
        "failed_or_invalid_plot_count": 5 - len(valid),
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in valid
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in valid
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": (
            sum(float(row["f1"]) for row in valid) / len(valid) if valid else 0.0
        ),
        "per_site_f1": {
            row["collection"]: float(row["f1"]) for row in valid
        },
        "oversegmented_reference_count": sum(
            int(row["oversegmented_reference_count"]) for row in valid
        ),
        "undersegmented_prediction_count": sum(
            int(row["undersegmented_prediction_count"]) for row in valid
        ),
        "total_instance_runtime_seconds": sum(
            float(row["instance_runtime_seconds"] or 0.0) for row in valid
        ),
        "maximum_instance_peak_rss_gb": max(
            (float(row["instance_peak_rss_gb"] or 0.0) for row in valid),
            default=0.0,
        ),
    }


def summarise(
    *,
    output_root: Path,
    workflow_run_id: str,
    manifest_path: Path,
    candidate_config_path: Path,
    development_evidence_path: Path,
    development_evidence_sha256: str,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    candidate_config_path = candidate_config_path.expanduser().resolve()
    development_evidence_path = development_evidence_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    if sha256(development_evidence_path) != development_evidence_sha256:
        raise RuntimeError("Development evidence checksum changed before aggregation")
    evidence = json.loads(development_evidence_path.read_text(encoding="utf-8"))
    required_count = config["development_evidence"]["required_valid_metric_count"]
    if (
        evidence.get("status")
        != config["development_evidence"]["required_summary_status"]
        or evidence.get("valid_metric_count") != required_count
        or evidence.get("expected_metric_count") != required_count
        or evidence.get("held_out_test_accessed") is not False
        or evidence.get("final_configuration_selected") is not False
        or evidence.get("split") != "development"
    ):
        raise ValueError("Leaf-screen development evidence is invalid")
    if config.get("scope", {}).get("targets") != ["leaf_on"]:
        raise ValueError("Leaf screen must evaluate leaf_on only")
    if config.get("scope", {}).get("held_out_test_accessed") is not False:
        raise ValueError("Leaf-screen config crossed the held-out-test boundary")
    candidates = config.get("candidates", [])
    if len(candidates) != 9:
        raise ValueError("Leaf screen requires exactly nine candidates")
    plots = stage0_rows(manifest)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    candidate_config_sha256 = sha256(candidate_config_path)
    source_run_id = evidence.get("workflow_run_id")
    for candidate in candidates:
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
            metric_path = plot_root / "evaluation" / "leaf_on" / "plot_metrics.json"
            instance: dict[str, Any] = {}
            adapter: dict[str, Any] = {}
            if instance_path.is_file():
                instance = json.loads(instance_path.read_text(encoding="utf-8"))
                if (
                    instance.get("candidate_id") != candidate_id
                    or instance.get("workflow_run_id") != workflow_run_id
                    or instance.get("stage1_config_sha256")
                    != candidate_config_sha256
                    or instance.get("development_evidence_sha256")
                    != development_evidence_sha256
                    or instance.get("development_evidence_run_id") != source_run_id
                    or instance.get("probe_summary") is not None
                    or instance.get("held_out_test_accessed") is not False
                    or instance.get("split") != "development"
                    or instance.get("target") != "leaf_on"
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
            row: dict[str, Any] = {
                "candidate_index": candidate["candidate_index"],
                "candidate_id": candidate_id,
                "add_leaves_voxel_length": candidate["parameters"][
                    "add_leaves_voxel_length"
                ],
                "add_leaves_edge_length": candidate["parameters"][
                    "add_leaves_edge_length"
                ],
                "stage0_index": plot["stage0_index"],
                "collection": plot["collection"],
                "safe_plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "target": "leaf_on",
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
                "ignored_class3_predicted_point_count": None,
                "instance_runtime_seconds": instance.get("runtime_seconds"),
                "instance_peak_rss_gb": instance.get("peak_rss_gb"),
                "adapter_runtime_seconds": adapter.get("runtime_seconds"),
                "metrics_path": str(metric_path),
                "metrics_sha256": None,
                "error": instance.get("error"),
            }
            if metric_path.is_file():
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                semantic_ignore = metrics.get("semantic_ignore", {})
                if (
                    metrics.get("split") != "dev"
                    or metrics.get("target") != "leaf_on"
                    or metrics.get("plot_id") != plot["safe_plot_id"]
                    or metrics.get("evaluator") != EXPECTED_EVALUATOR
                    or metrics.get("evaluation_mask") != EXPECTED_EVALUATION_MASK
                    or semantic_ignore.get("ignored_semantic_classes") != [3]
                ):
                    raise ValueError(
                        f"Metric protocol or provenance mismatch: {metric_path}"
                    )
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
                row["ignored_class3_predicted_point_count"] = semantic_ignore.get(
                    "ignored_predicted_point_count"
                )
                row["metrics_sha256"] = sha256(metric_path)
            if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
                missing.append(f"{candidate_id}:{plot['safe_plot_id']}:leaf_on")
            rows.append(row)

    aggregates = [aggregate(rows, candidate["candidate_id"]) for candidate in candidates]
    ranking = [
        item["candidate_id"]
        for item in sorted(
            aggregates,
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
    expected_metric_count = len(candidates) * len(plots)
    return {
        "schema_version": 1,
        "status": (
            "development_leaf_screen_completed"
            if not missing
            else "development_leaf_screen_incomplete"
        ),
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "development",
        "target": "leaf_on",
        "workflow_run_id": workflow_run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "candidate_config": str(candidate_config_path),
        "candidate_config_sha256": candidate_config_sha256,
        "development_evidence": str(development_evidence_path),
        "development_evidence_sha256": development_evidence_sha256,
        "development_evidence_run_id": source_run_id,
        "evaluator": EXPECTED_EVALUATOR,
        "evaluation_mask": EXPECTED_EVALUATION_MASK,
        "ignored_semantic_classes": [3],
        "expected_metric_count": expected_metric_count,
        "valid_metric_count": expected_metric_count - len(missing),
        "incomplete_tasks": missing,
        "development_reference_labels_accessed": True,
        "development_accuracy_metrics_computed": True,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "candidate_ranking_for_review": ranking,
        "top_three_candidate_ids_for_review": ranking[:3],
        "candidate_parameters": {
            candidate["candidate_id"]: candidate["parameters"]
            for candidate in candidates
        },
        "plot_metrics": rows,
        "aggregates": aggregates,
        "next_gate": "publish_development_leaf_screen_evidence_without_configuration_change",
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
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--development-evidence-json", required=True)
    parser.add_argument("--development-evidence-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plot-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = [Path(args.output_json), Path(args.plot_csv), Path(args.aggregate_csv)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing existing leaf-screen summary output")
    payload = summarise(
        output_root=Path(args.output_root),
        workflow_run_id=args.workflow_run_id,
        manifest_path=Path(args.manifest_json),
        candidate_config_path=Path(args.candidate_config),
        development_evidence_path=Path(args.development_evidence_json),
        development_evidence_sha256=args.development_evidence_sha256,
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(outputs[1], payload["plot_metrics"])
    write_csv(outputs[2], payload["aggregates"])
    print(f"status={payload['status']}")
    print(
        f"valid_metrics={payload['valid_metric_count']}/"
        f"{payload['expected_metric_count']}"
    )
    print("leaf_on_ranking=" + ",".join(payload["candidate_ranking_for_review"]))
    print(
        "top_three_for_review="
        + ",".join(payload["top_three_candidate_ids_for_review"])
    )
    print("final_configuration_selected=false")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
