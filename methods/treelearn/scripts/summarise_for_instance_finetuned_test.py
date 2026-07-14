"""Validate, retain and summarise the one-time TreeLearn held-out test."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from for_instance_test_common import (
    EXPECTED_TEST_SITE_COUNTS,
    load_test_manifest,
)
from summarise_for_instance_development import (
    AGGREGATE_FIELDS,
    CONSOLIDATED_PREFIX_FIELDS,
    FAILURE_FIELDS,
    MATCH_FIELDS,
    PER_PLOT_FIELDS,
    UNMATCHED_PREDICTION_FIELDS,
    UNMATCHED_REFERENCE_FIELDS,
    artifact_entry,
    load_object,
    prefix_records,
    sha256,
    validate_evaluation_tables,
    validate_retention,
    write_csv,
)


EXPECTED_UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
EXPECTED_SOURCE_MD5 = "106a80de2991c5f23484a3f9d03e3b16"


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_inference(
    metadata: dict[str, Any],
    row: dict[str, Any],
    run_id: str,
    checkpoint_sha256: str,
    benchmark_commit: str,
    training_mode: str,
    checkpoint_source_md5: str,
    execution_recovery: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    for field, expected in (
        ("method", "treelearn"),
        ("dataset", "for-instance"),
        ("dataset_split", "test"),
        ("held_out_test_accessed", True),
        ("run_id", run_id),
        ("training_mode", training_mode),
        ("status", "completed"),
        ("return_code", 0),
        ("evaluation_scope", "held_out_test"),
    ):
        require_equal(metadata.get(field), expected, f"Inference {field} mismatch")
    plot = metadata.get("plot") or {}
    for field in ("plot_id", "safe_plot_id", "relative_path", "collection", "split"):
        require_equal(plot.get(field), row[field], f"Inference plot {field} mismatch")
    checkpoint = metadata.get("checkpoint") or {}
    require_equal(
        checkpoint.get("source_md5"), checkpoint_source_md5, "Checkpoint source mismatch"
    )
    require_equal(
        checkpoint.get("sha256"), checkpoint_sha256, "Checkpoint SHA-256 mismatch"
    )
    repositories = metadata.get("environment") or {}
    upstream = repositories.get("treelearn_repository") or {}
    benchmark = repositories.get("benchmark_repository") or {}
    if upstream.get("commit") != EXPECTED_UPSTREAM_COMMIT or upstream.get("dirty") is not False:
        raise ValueError("TreeLearn upstream provenance differs from the frozen contract")
    if benchmark.get("commit") != benchmark_commit or benchmark.get("dirty") is not False:
        raise ValueError("Benchmark provenance differs from submission")
    recovery = metadata.get("execution_recovery") or {}
    if execution_recovery is None:
        if recovery.get("enabled") is True:
            raise ValueError("Unexpected execution recovery in test inference")
    else:
        expected_recovery = {
            "enabled": True,
            "policy": execution_recovery["policy"],
        }
        for field, value in expected_recovery.items():
            require_equal(recovery.get(field), value, f"Execution recovery {field} mismatch")
        outcome = recovery.get("outcome") or {}
        for field, value in (
            ("status", "empty_group_mapped_to_background"),
            ("triggered", True),
            ("reference_points", 0),
        ):
            require_equal(outcome.get(field), value, f"Execution recovery outcome {field} mismatch")
    dataset = metadata.get("dataset_validation") or {}
    for field in ("input_sha256", "split_metadata_sha256"):
        require_equal(dataset.get(field), row[field], f"Dataset {field} mismatch")
    require_equal(int(dataset.get("point_count", -1)), row["point_count"], "Point count mismatch")
    require_equal(
        int(dataset.get("reference_tree_count", -1)),
        row["reference_tree_count"],
        "Reference count mismatch",
    )
    alignment = metadata.get("validation") or {}
    if alignment.get("row_count_match") is not True or alignment.get("row_order_preserved") is not True:
        raise ValueError(f"Point alignment failed for {row['relative_path']}")
    if int(alignment.get("source_point_count", -1)) != row["point_count"]:
        raise ValueError(f"Source row count changed for {row['relative_path']}")
    if int(alignment.get("prediction_point_count", -1)) != row["point_count"]:
        raise ValueError(f"Prediction row count changed for {row['relative_path']}")
    delta = float(alignment.get("max_abs_coordinate_delta_m", math.inf))
    tolerance = float(alignment.get("row_coordinate_tolerance_m", -1.0))
    if not (math.isfinite(delta) and 0.0 <= delta <= tolerance):
        raise ValueError(f"Coordinate alignment failed for {row['relative_path']}")
    return validate_retention(metadata, row)


def validate_metrics(
    metrics: dict[str, Any],
    metadata_path: Path,
    retained: list[dict[str, Any]],
    row: dict[str, Any],
    run_id: str,
) -> None:
    expected = {
        "status": "completed_aligned_pointwise_held_out_test_plot",
        "evaluation_scope": "held_out_test",
        "held_out_test_accessed": True,
        "run_id": run_id,
        "plot_id": row["plot_id"],
        "relative_path": row["relative_path"],
        "split": "test",
        "dataset_split": "test",
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "point_correspondence": "source_row_index",
        "iou_threshold": 0.5,
        "tuned_prediction_filtering": False,
        "min_predicted_instance_points": 0,
        "min_predicted_tree_fraction": 0.0,
    }
    for field, value in expected.items():
        require_equal(metrics.get(field), value, f"Metric {field} mismatch")
    if Path(str(metrics.get("inference_metadata", ""))).resolve() != metadata_path.resolve():
        raise ValueError("Metric inference-metadata path mismatch")
    adapted = next(item for item in retained if item["role"] == "adapted_npz")
    if Path(str(metrics.get("prediction_npz", ""))).resolve() != Path(adapted["path"]):
        raise ValueError("Metric prediction path mismatch")
    require_equal(metrics.get("prediction_npz_sha256"), adapted["sha256"], "Prediction hash mismatch")
    require_equal(int(metrics.get("point_count", -1)), row["point_count"], "Metric point count mismatch")
    require_equal(
        int(metrics.get("reference_instance_count", -1)),
        row["reference_tree_count"],
        "Metric reference count mismatch",
    )
    tp = int(metrics.get("true_positives", -1))
    fp = int(metrics.get("false_positives", -1))
    fn = int(metrics.get("false_negatives", -1))
    predictions = int(metrics.get("prediction_instance_count", -1))
    references = int(metrics.get("reference_instance_count", -1))
    if min(tp, fp, fn, predictions, references) < 0:
        raise ValueError("Negative metric counts")
    if predictions != tp + fp or references != tp + fn:
        raise ValueError("Metric count identities failed")
    expected_values = {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }
    for field, value in expected_values.items():
        if not math.isclose(float(metrics.get(field, -1)), value, abs_tol=1e-12):
            raise ValueError(f"Metric {field} is inconsistent")
    harmonized = metrics.get("harmonized") or {}
    for field in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1", "mean_matched_iou"):
        require_equal(harmonized.get(field), metrics.get(field), f"Harmonized {field} mismatch")


def mean(rows: list[dict[str, Any]], field: str) -> float | str:
    return statistics.fmean(float(row[field]) for row in rows) if rows else ""


def aggregate(
    rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    site: str,
    expected_plots: int,
    variant: str,
) -> dict[str, Any]:
    completed = [row for row in rows if row["result_status"] == "completed"]
    tp = sum(int(row["true_positives"]) for row in completed)
    fp = sum(int(row["false_positives"]) for row in completed)
    fn = sum(int(row["false_negatives"]) for row in completed)
    failed = expected_plots - len(completed)
    return {
        "method": "TreeLearn",
        "variant": variant,
        "dataset_split": "test",
        "site": site,
        "expected_plots": expected_plots,
        "completed_plots": len(completed),
        "failed_plots": failed,
        "point_count": sum(int(row["point_count"]) for row in completed),
        "evaluated_point_count": sum(int(row["evaluated_point_count"]) for row in completed),
        "prediction_instance_count": sum(int(row["prediction_instance_count"]) for row in completed),
        "reference_instance_count": sum(int(row["reference_instance_count"]) for row in completed),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_precision": mean(completed, "precision"),
        "mean_plot_recall": mean(completed, "recall"),
        "mean_plot_f1": mean(completed, "f1"),
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_plot_matched_iou": mean(completed, "mean_matched_iou"),
        "mean_matched_iou_across_pairs": (
            statistics.fmean(float(row["iou"]) for row in match_rows)
            if match_rows else 0.0
        ),
        "mean_unweighted_coverage": mean(completed, "mean_unweighted_coverage"),
        "mean_weighted_coverage": mean(completed, "mean_weighted_coverage"),
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "result_status": (
            "completed_aligned_pointwise_test"
            if failed == 0 else "held_out_test_with_documented_failures"
        ),
        "held_out_test_accessed": "true",
    }


def summarise(
    manifest_path: Path,
    run_id: str,
    evaluation_root: Path,
    metadata_root: Path,
    output_root: Path,
    expected_benchmark_commit: str,
    recovery_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("Unsafe TreeLearn test run ID")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_benchmark_commit):
        raise ValueError("Expected benchmark commit must be a full SHA-1")
    rows, manifest = load_test_manifest(manifest_path.resolve())
    require_equal(manifest.get("run_id"), run_id, "Test run ID mismatch")
    checkpoint_sha256 = str(manifest["checkpoint_sha256"])
    checkpoint_source_md5 = str(
        (manifest.get("checkpoint_provenance") or {}).get("source_md5", "")
    )
    if checkpoint_source_md5 != EXPECTED_SOURCE_MD5:
        raise ValueError("Test manifest checkpoint source differs from the clean contract")
    variant = str(manifest["variant"])
    training_mode = str(manifest["training_mode"])
    execution_recovery: dict[str, Any] | None = None
    if recovery_manifest_path is not None:
        execution_recovery = load_object(recovery_manifest_path.resolve())
        expected_recovery = {
            "status": "prepared_single_execution_recovery",
            "run_id": run_id,
            "variant": variant,
            "training_mode": training_mode,
            "dataset_split": "test",
            "held_out_test_accessed": True,
            "repeat_test_for_setting_selection_permitted": False,
            "model_or_parameter_selection_performed": False,
            "task_index": 8,
            "relative_path": "SCION/plot_31_annotated.las",
            "checkpoint_sha256": checkpoint_sha256,
            "recovery_benchmark_commit": expected_benchmark_commit,
            "policy": "map_all_unassigned_to_background_when_initial_grouping_is_empty",
        }
        for field, value in expected_recovery.items():
            require_equal(
                execution_recovery.get(field), value, f"Recovery manifest {field} mismatch"
            )
        if not re.fullmatch(
            r"[0-9a-f]{40}", str(execution_recovery.get("original_benchmark_commit", ""))
        ):
            raise ValueError("Recovery manifest original commit is invalid")
    evaluation_root = evaluation_root.resolve()
    metadata_root = metadata_root.resolve()
    output_root = output_root.resolve()
    targets = {
        "plot_summary": output_root / "plot_summary.csv",
        "site_summary": output_root / "site_summary.csv",
        "final_summary": output_root / "final_summary.csv",
        "failures": output_root / "failures.csv",
        "matches": output_root / "matches.csv",
        "unmatched_predictions": output_root / "unmatched_predictions.csv",
        "unmatched_references": output_root / "unmatched_references.csv",
        "retention_manifest": output_root / "retention_manifest.json",
        "run_summary": output_root / "run_summary.json",
    }
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("TreeLearn held-out summary already exists")
    plot_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    all_unmatched_predictions: list[dict[str, Any]] = []
    all_unmatched_references: list[dict[str, Any]] = []
    retention_records: list[dict[str, Any]] = []
    for row in rows:
        plot_root = evaluation_root / row["safe_plot_id"]
        metadata_path = metadata_root / f"{row['safe_plot_id']}_inference.json"
        metrics_path = plot_root / "metrics.json"
        failure_path = plot_root / "status.json"
        if metrics_path.is_file() and failure_path.is_file():
            raise ValueError(f"Test plot has both metrics and failure: {row['relative_path']}")
        if failure_path.is_file():
            failure = load_object(failure_path)
            if failure.get("status") not in {
                "documented_inference_failure",
                "documented_evaluation_failure",
            }:
                raise ValueError(f"Unrecognised test failure status: {failure_path}")
            for field, expected in (
                ("run_id", run_id), ("task_index", row["task_index"]),
                ("relative_path", row["relative_path"]), ("split", "test"),
                ("held_out_test_accessed", True),
            ):
                require_equal(failure.get(field), expected, f"Failure {field} mismatch")
            reason = json.dumps(failure.get("error") or failure.get("inference_error") or {}, sort_keys=True)
            failures.append({
                **{field: row[field] for field in ("task_index", "plot_id", "safe_plot_id", "relative_path", "collection", "split")},
                "run_id": run_id,
                "status": failure["status"],
                "reason": reason,
                "inference_status": failure.get("inference_status", ""),
                "inference_metadata": str(metadata_path) if metadata_path.is_file() else "",
                "failure_record": str(failure_path),
                "held_out_test_accessed": "true",
            })
            plot_rows.append({
                **{field: row[field] for field in ("task_index", "plot_id", "safe_plot_id", "relative_path", "collection", "split")},
                "run_id": run_id, "result_status": failure["status"],
                "point_count": row["point_count"],
                "inference_metadata": str(metadata_path) if metadata_path.is_file() else "",
                "metrics_json": "", "failure_reason": reason,
                "held_out_test_accessed": "true",
            })
            if failure.get("status") == "documented_evaluation_failure":
                metadata = load_object(metadata_path)
                row_commit = (
                    str(execution_recovery["recovery_benchmark_commit"])
                    if execution_recovery and row["task_index"] == execution_recovery["task_index"]
                    else (
                        str(execution_recovery["original_benchmark_commit"])
                        if execution_recovery else expected_benchmark_commit
                    )
                )
                retained = validate_inference(
                    metadata,
                    row,
                    run_id,
                    checkpoint_sha256,
                    row_commit,
                    training_mode,
                    checkpoint_source_md5,
                    execution_recovery if execution_recovery and row["task_index"] == execution_recovery["task_index"] else None,
                )
                retention_records.append({
                    "task_index": row["task_index"], "plot_id": row["plot_id"],
                    "relative_path": row["relative_path"], "collection": row["collection"],
                    "split": "test", "prediction_files": retained,
                    "inference_metadata": artifact_entry(metadata_path),
                    "evaluation_artifacts": [artifact_entry(failure_path)],
                    "retention_verified": True,
                    "evaluation_status": "documented_evaluation_failure",
                })
            continue
        if not metrics_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Missing accounted test output for {row['relative_path']}")
        metadata = load_object(metadata_path)
        row_commit = (
            str(execution_recovery["recovery_benchmark_commit"])
            if execution_recovery and row["task_index"] == execution_recovery["task_index"]
            else (
                str(execution_recovery["original_benchmark_commit"])
                if execution_recovery else expected_benchmark_commit
            )
        )
        retained = validate_inference(
            metadata,
            row,
            run_id,
            checkpoint_sha256,
            row_commit,
            training_mode,
            checkpoint_source_md5,
            execution_recovery if execution_recovery and row["task_index"] == execution_recovery["task_index"] else None,
        )
        metrics = load_object(metrics_path)
        validate_metrics(metrics, metadata_path, retained, row, run_id)
        matched, unmatched_predictions, unmatched_references = validate_evaluation_tables(plot_root, row, metrics)
        all_matches.extend(prefix_records(matched, row))
        all_unmatched_predictions.extend(prefix_records(unmatched_predictions, row))
        all_unmatched_references.extend(prefix_records(unmatched_references, row))
        metric_fields = (
            "point_count", "evaluated_point_count", "prediction_instance_count",
            "reference_instance_count", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1", "mean_matched_iou",
            "mean_unweighted_coverage", "mean_weighted_coverage",
        )
        plot_rows.append({
            **{field: row[field] for field in ("task_index", "plot_id", "safe_plot_id", "relative_path", "collection", "split")},
            "run_id": run_id, "result_status": "completed",
            **{field: metrics[field] for field in metric_fields},
            "inference_metadata": str(metadata_path), "metrics_json": str(metrics_path),
            "failure_reason": "", "held_out_test_accessed": "true",
        })
        retention_records.append({
            "task_index": row["task_index"], "plot_id": row["plot_id"],
            "relative_path": row["relative_path"], "collection": row["collection"],
            "split": "test", "prediction_files": retained,
            "inference_metadata": artifact_entry(metadata_path),
            "evaluation_artifacts": [artifact_entry(plot_root / name) for name in (
                "metrics.json", "matches.csv", "unmatched_predictions.csv", "unmatched_references.csv"
            )],
            "retention_verified": True, "evaluation_status": "completed",
        })
    plot_rows.sort(key=lambda row: int(row["task_index"]))
    site_rows = []
    for site, count in EXPECTED_TEST_SITE_COUNTS.items():
        site_rows.append(aggregate(
            [row for row in plot_rows if row["collection"] == site],
            [row for row in all_matches if row["collection"] == site],
            site,
            count,
            variant,
        ))
    overall = aggregate(plot_rows, all_matches, "ALL", len(rows), variant)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(targets["plot_summary"], plot_rows, PER_PLOT_FIELDS)
    write_csv(targets["site_summary"], site_rows, AGGREGATE_FIELDS)
    write_csv(targets["final_summary"], [overall], AGGREGATE_FIELDS)
    write_csv(targets["failures"], failures, FAILURE_FIELDS)
    write_csv(targets["matches"], all_matches, CONSOLIDATED_PREFIX_FIELDS + MATCH_FIELDS)
    write_csv(targets["unmatched_predictions"], all_unmatched_predictions, CONSOLIDATED_PREFIX_FIELDS + UNMATCHED_PREDICTION_FIELDS)
    write_csv(targets["unmatched_references"], all_unmatched_references, CONSOLIDATED_PREFIX_FIELDS + UNMATCHED_REFERENCE_FIELDS)
    completed_count = sum(row["result_status"] == "completed" for row in plot_rows)
    retained_files = sum(len(item["prediction_files"]) for item in retention_records)
    retained_bytes = sum(
        int(file["size_bytes"])
        for item in retention_records for file in item["prediction_files"]
    )
    retention = {
        "schema_version": 1,
        "status": "retention_verified" if not failures else "retention_verified_for_completed_plots_with_documented_failures",
        "method": "TreeLearn", "dataset": "FOR-instance", "run_id": run_id,
        "variant": variant, "training_mode": training_mode,
        "dataset_split": "test", "held_out_test_accessed": True,
        "checkpoint_sha256": checkpoint_sha256,
        "expected_plots": len(rows), "completed_plots": completed_count,
        "inference_outputs_retained": len(retention_records),
        "documented_failures": len(failures),
        "verified_prediction_file_count": retained_files,
        "verified_prediction_size_bytes": retained_bytes,
        "complete_test_prediction_set_retained": len(retention_records) == len(rows),
        "execution_recovery": execution_recovery,
        "plots": retention_records,
    }
    targets["retention_manifest"].write_text(json.dumps(retention, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1, "status": overall["result_status"],
        "method": "TreeLearn", "dataset": "FOR-instance", "variant": variant,
        "training_mode": training_mode,
        "run_id": run_id, "dataset_split": "test", "held_out_test_accessed": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit": expected_benchmark_commit,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "expected_plots": len(rows), "completed_plots": completed_count,
        "documented_failures": len(failures), "retention_status": retention["status"],
        "site_counts": EXPECTED_TEST_SITE_COUNTS, "test_metrics": overall,
        "execution_recovery": execution_recovery,
        "outputs": {key: artifact_entry(path) for key, path in targets.items() if key != "run_summary" and path.is_file()},
        "next_gate": "treelearn_benchmark_complete" if not failures else "resolve_test_execution_failures_without_model_selection",
    }
    targets["run_summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-benchmark-commit", required=True)
    parser.add_argument("--recovery-manifest", type=Path)
    args = parser.parse_args()
    summary = summarise(
        Path(args.manifest), args.run_id, Path(args.evaluation_root),
        Path(args.metadata_root), Path(args.output_root), args.expected_benchmark_commit,
        args.recovery_manifest,
    )
    print(f"status={summary['status']}")
    print(f"completed_plots={summary['completed_plots']}/{summary['expected_plots']}")
    print(f"documented_failures={summary['documented_failures']}")
    print(f"final_summary={Path(args.output_root).resolve() / 'final_summary.csv'}")
    print("held_out_test_accessed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
