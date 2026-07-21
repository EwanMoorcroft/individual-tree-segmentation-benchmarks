"""Summarise the single authorised TLS2trees held-out FOR-instance run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGETS = ("leaf_off", "leaf_on")
EXPECTED_PLOTS = 11
EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
EVALUATION_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target"] == target]
    valid = [
        row for row in selected
        if row["status"] == "evaluated" and row["safe_for_scoring"] is True
    ]
    tp = sum(int(row["true_positives"]) for row in valid)
    fp = sum(int(row["false_positives"]) for row in valid)
    fn = sum(int(row["false_negatives"]) for row in valid)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_collection: dict[str, list[float]] = {}
    for row in valid:
        by_collection.setdefault(row["collection"], []).append(float(row["f1"]))
    return {
        "target": target,
        "candidate_id": selected[0]["candidate_id"],
        "expected_plot_count": EXPECTED_PLOTS,
        "evaluated_plot_count": len(valid),
        "failed_or_invalid_plot_count": EXPECTED_PLOTS - len(valid),
        "prediction_instance_count": sum(int(row["prediction_instance_count"]) for row in valid),
        "reference_instance_count": sum(int(row["reference_instance_count"]) for row in valid),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": sum(float(row["f1"]) for row in valid) / len(valid) if valid else 0.0,
        "mean_collection_f1": {
            key: sum(values) / len(values)
            for key, values in sorted(by_collection.items())
        },
        "oversegmented_reference_count": sum(int(row["oversegmented_reference_count"]) for row in valid),
        "undersegmented_prediction_count": sum(int(row["undersegmented_prediction_count"]) for row in valid),
    }


def summarise(
    *, output_root: Path, workflow_run_id: str, manifest_path: Path,
    final_selection_path: Path, final_selection_sha256: str,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    final_selection_path = final_selection_path.expanduser().resolve()
    if sha256(final_selection_path) != final_selection_sha256:
        raise RuntimeError("Reviewed final selection checksum changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final = json.loads(final_selection_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_split") != "test"
        or len(manifest.get("plots", [])) != EXPECTED_PLOTS
        or [int(row["task_index"]) for row in manifest["plots"]] != list(range(EXPECTED_PLOTS))
        or sum(int(row["point_count"]) for row in manifest["plots"]) != 49_709_922
        or sum(int(row["reference_tree_count"]) for row in manifest["plots"]) != 323
    ):
        raise ValueError("Held-out summary requires the exact 11-plot test manifest")
    if (
        final.get("status") != "development_tuned_configuration_frozen"
        or final.get("final_configuration_selected") is not True
        or final.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Final selection is not a clean pre-test freeze")

    rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for target in TARGETS:
        selected = final["selected_by_target"][target]
        candidate_id = selected["candidate_id"]
        candidate_run_id = f"{workflow_run_id}__{target}__{candidate_id}"
        for plot in manifest["plots"]:
            plot_root = (
                output_root.expanduser().resolve() / "tls2trees" / "for_instance"
                / "development_tuned" / "test" / candidate_run_id / plot["safe_plot_id"]
            )
            instance_path = plot_root / "metadata" / "instance_run.json"
            adapter_path = plot_root / "metadata" / "adapter_run.json"
            metric_path = plot_root / "evaluation" / target / "plot_metrics.json"
            instance = json.loads(instance_path.read_text(encoding="utf-8")) if instance_path.is_file() else {}
            adapter = json.loads(adapter_path.read_text(encoding="utf-8")) if adapter_path.is_file() else {}
            if instance and (
                instance.get("split") != "test"
                or instance.get("target") != target
                or instance.get("candidate_id") != candidate_id
                or instance.get("held_out_test_accessed") is not True
                or instance.get("final_selection_sha256") != final_selection_sha256
            ):
                raise ValueError(f"Held-out instance provenance mismatch: {instance_path}")
            if adapter and (
                adapter.get("split") != "test"
                or adapter.get("held_out_test_accessed") is not True
            ):
                raise ValueError(f"Held-out adapter provenance mismatch: {adapter_path}")
            row: dict[str, Any] = {
                "target": target, "candidate_id": candidate_id,
                "stage1_candidate_index": int(selected["stage1_candidate_index"]),
                "task_index": int(plot["task_index"]), "collection": plot["collection"],
                "safe_plot_id": plot["safe_plot_id"], "relative_path": plot["relative_path"],
                "status": "missing", "safe_for_scoring": False,
                "raw_prediction_instance_count": None,
                "prediction_instance_count": None, "reference_instance_count": None,
                "true_positives": None, "false_positives": None, "false_negatives": None,
                "precision": None, "recall": None, "f1": None,
                "mean_matched_iou": None, "oversegmented_reference_count": None,
                "undersegmented_prediction_count": None,
                "instance_runtime_seconds": instance.get("runtime_seconds"),
                "instance_peak_rss_gb": instance.get("peak_rss_gb"),
                "adapter_runtime_seconds": adapter.get("runtime_seconds"),
                "metrics_path": str(metric_path), "metrics_sha256": None,
                "error": instance.get("error"),
            }
            if metric_path.is_file():
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                if (
                    metrics.get("split") != "test"
                    or metrics.get("target") != target
                    or metrics.get("plot_id") != plot["safe_plot_id"]
                    or metrics.get("relative_path") != plot["relative_path"]
                    or metrics.get("evaluator") != EVALUATOR
                    or metrics.get("evaluation_mask") != EVALUATION_MASK
                    or metrics.get("semantic_ignore", {}).get(
                        "ignored_semantic_classes"
                    )
                    != [3]
                ):
                    raise ValueError(f"Held-out metric provenance mismatch: {metric_path}")
                for key in (
                    "status", "safe_for_scoring", "prediction_instance_count",
                    "reference_instance_count", "true_positives", "false_positives",
                    "false_negatives", "precision", "recall", "f1", "mean_matched_iou",
                    "oversegmented_reference_count", "undersegmented_prediction_count",
                ):
                    row[key] = metrics.get(key)
                row["raw_prediction_instance_count"] = metrics[
                    "semantic_ignore"
                ]["raw_prediction_instance_count"]
                row["metrics_sha256"] = sha256(metric_path)
            if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
                incomplete.append(f"{target}:{plot['safe_plot_id']}")
            rows.append(row)
    aggregates = [aggregate(rows, target) for target in TARGETS]
    return {
        "schema_version": 1,
        "status": "held_out_test_completed" if not incomplete else "held_out_test_incomplete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FOR-instance", "method": "TLS2trees",
        "variant": "development_tuned", "split": "test",
        "workflow_run_id": workflow_run_id,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "final_selection": str(final_selection_path),
        "final_selection_sha256": final_selection_sha256,
        "expected_plot_count": EXPECTED_PLOTS, "expected_metric_count": 22,
        "valid_metric_count": 22 - len(incomplete), "incomplete_tasks": incomplete,
        "held_out_test_accessed": True, "held_out_accuracy_metrics_computed": True,
        "configuration_changed_after_test": False,
        "plot_metrics": rows, "aggregates": aggregates,
        "next_gate": "report_frozen_held_out_results_without_retuning",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--final-selection-json", required=True)
    parser.add_argument("--final-selection-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plot-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    args = parser.parse_args()
    outputs = [Path(args.output_json), Path(args.plot_csv), Path(args.aggregate_csv)]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Held-out summary output already exists")
    payload = summarise(
        output_root=Path(args.output_root), workflow_run_id=args.workflow_run_id,
        manifest_path=Path(args.manifest_json),
        final_selection_path=Path(args.final_selection_json),
        final_selection_sha256=args.final_selection_sha256,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(Path(args.plot_csv), payload["plot_metrics"])
    write_csv(Path(args.aggregate_csv), payload["aggregates"])
    print(f"status={payload['status']}")
    print(f"valid_metrics={payload['valid_metric_count']}/{payload['expected_metric_count']}")
    for row in payload["aggregates"]:
        print(
            f"{row['target']}={row['candidate_id']} micro_f1={row['micro_f1']:.6f} "
            f"mean_plot_f1={row['mean_plot_f1']:.6f} precision={row['precision']:.6f} "
            f"recall={row['recall']:.6f} invalid={row['failed_or_invalid_plot_count']}"
        )
    print("held_out_test_accessed=true")
    print("configuration_changed_after_test=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
