"""Validate and summarise a frozen TreeLearn FOR-instance development run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from for_instance_development_common import (
    EXPECTED_DEVELOPMENT_SITE_COUNTS,
    load_manifest,
)


EXPECTED_UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
EXPECTED_CHECKPOINT_MD5 = "56a3d78f689ae7f1190906b975700311"
EXPECTED_CHECKPOINT_SHA256 = (
    "5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"
)


MATCH_FIELDS = [
    "plot_id",
    "pred_tree_id",
    "target_tree_id",
    "intersection_points",
    "predicted_points",
    "reference_points",
    "union_points",
    "iou",
]
UNMATCHED_PREDICTION_FIELDS = [
    "plot_id",
    "pred_tree_id",
    "predicted_points",
    "best_target_tree_id",
    "best_iou",
]
UNMATCHED_REFERENCE_FIELDS = [
    "plot_id",
    "target_tree_id",
    "reference_points",
    "best_pred_tree_id",
    "best_iou",
]
CONSOLIDATED_PREFIX_FIELDS = [
    "task_index",
    "relative_path",
    "collection",
    "split",
]
PER_PLOT_FIELDS = [
    "task_index",
    "run_id",
    "plot_id",
    "safe_plot_id",
    "relative_path",
    "collection",
    "split",
    "result_status",
    "point_count",
    "evaluated_point_count",
    "prediction_instance_count",
    "reference_instance_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "mean_unweighted_coverage",
    "mean_weighted_coverage",
    "inference_metadata",
    "metrics_json",
    "failure_reason",
    "held_out_test_accessed",
]
AGGREGATE_FIELDS = [
    "method",
    "variant",
    "dataset_split",
    "site",
    "expected_plots",
    "completed_plots",
    "failed_plots",
    "point_count",
    "evaluated_point_count",
    "prediction_instance_count",
    "reference_instance_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "mean_plot_precision",
    "mean_plot_recall",
    "mean_plot_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_plot_matched_iou",
    "mean_matched_iou_across_pairs",
    "mean_unweighted_coverage",
    "mean_weighted_coverage",
    "evaluation_protocol",
    "matching_policy",
    "result_status",
    "held_out_test_accessed",
]
FAILURE_FIELDS = [
    "task_index",
    "run_id",
    "plot_id",
    "safe_plot_id",
    "relative_path",
    "collection",
    "split",
    "status",
    "reason",
    "inference_status",
    "inference_metadata",
    "failure_record",
    "held_out_test_accessed",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = set(required_fields) - fieldnames
        if missing:
            raise ValueError(f"{path} is missing fields {sorted(missing)}")
        return list(reader)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected {expected!r}, found {actual!r}")


def validate_inference_identity(
    metadata: dict[str, Any],
    row: dict[str, Any],
    run_id: str,
    *,
    require_completed: bool,
) -> dict[str, str] | None:
    _require_equal(metadata.get("run_id"), run_id, "Inference run ID mismatch")
    if require_completed:
        _require_equal(metadata.get("status"), "completed", "Inference status mismatch")
        _require_equal(
            metadata.get("evaluation_scope"),
            "development_full",
            "Inference evaluation scope mismatch",
        )
    plot = metadata.get("plot")
    if not isinstance(plot, dict):
        raise ValueError(f"Inference metadata has no plot object for {row['relative_path']}")
    for field in ("plot_id", "safe_plot_id", "relative_path", "collection", "split"):
        _require_equal(plot.get(field), row[field], f"Inference plot {field} mismatch")
    if plot.get("split") != "dev":
        raise ValueError(f"Non-development inference metadata: {row['relative_path']}")
    if metadata.get("held_out_test_accessed") is not False:
        raise ValueError(
            "Inference metadata does not explicitly lock held-out test access: "
            f"{row['relative_path']}"
        )
    if require_completed or metadata.get("status") == "completed":
        for field, expected in (
            ("method", "treelearn"),
            ("dataset", "for-instance"),
            ("dataset_split", "dev"),
            ("training_mode", "published_pretrained"),
        ):
            _require_equal(
                metadata.get(field),
                expected,
                f"Inference provenance {field} mismatch",
            )
        checkpoint = metadata.get("checkpoint")
        environment = metadata.get("environment")
        if not isinstance(checkpoint, dict) or not isinstance(environment, dict):
            raise ValueError(
                f"Inference provenance is incomplete for {row['relative_path']}"
            )
        _require_equal(
            checkpoint.get("md5"),
            EXPECTED_CHECKPOINT_MD5,
            "Inference checkpoint MD5 mismatch",
        )
        _require_equal(
            checkpoint.get("sha256"),
            EXPECTED_CHECKPOINT_SHA256,
            "Inference checkpoint SHA-256 mismatch",
        )
        upstream = environment.get("treelearn_repository")
        benchmark = environment.get("benchmark_repository")
        if not isinstance(upstream, dict) or not isinstance(benchmark, dict):
            raise ValueError(
                f"Inference repository provenance is incomplete for {row['relative_path']}"
            )
        if (
            upstream.get("commit") != EXPECTED_UPSTREAM_COMMIT
            or upstream.get("dirty") is not False
        ):
            raise ValueError(
                f"Inference upstream provenance differs for {row['relative_path']}"
            )
        benchmark_commit = str(benchmark.get("commit", ""))
        if (
            not re.fullmatch(r"[0-9a-f]{40}", benchmark_commit)
            or benchmark.get("dirty") is not False
        ):
            raise ValueError(
                f"Inference benchmark provenance is invalid for {row['relative_path']}"
            )
        return {
            "training_mode": "published_pretrained",
            "checkpoint_md5": EXPECTED_CHECKPOINT_MD5,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "benchmark_commit": benchmark_commit,
        }
    return None


def validate_retention(
    metadata: dict[str, Any],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    retention = metadata.get("retention")
    outputs = metadata.get("outputs")
    if not isinstance(retention, dict) or not isinstance(outputs, dict):
        raise ValueError(f"Missing retention/outputs metadata for {row['relative_path']}")
    for flag in (
        "raw_pointwise_output_retained",
        "raw_full_forest_output_retained",
        "adapted_point_aligned_output_retained",
    ):
        if retention.get(flag) is not True:
            raise ValueError(f"Retention flag {flag} is not true for {row['relative_path']}")
    entries = retention.get("files")
    if not isinstance(entries, list) or len(entries) != 5:
        raise ValueError(
            f"Expected five retained prediction artefacts for {row['relative_path']}"
        )
    expected_roles = (
        "raw_prediction_laz",
        "raw_prediction_npz",
        "raw_pointwise_npz",
        "adapted_npz",
        "adapted_las",
    )
    output_paths = {
        role: Path(str(outputs.get(role, ""))).expanduser().resolve()
        for role in expected_roles
    }
    if any(not str(outputs.get(role, "")) for role in expected_roles):
        raise ValueError(f"Inference outputs are incomplete for {row['relative_path']}")
    verified: list[dict[str, Any]] = []
    observed_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid retention entry for {row['relative_path']}")
        path = Path(str(entry.get("path", ""))).expanduser().resolve()
        if path in observed_paths:
            raise ValueError(f"Duplicate retained path for {row['relative_path']}: {path}")
        observed_paths.add(path)
        role_matches = [role for role, expected in output_paths.items() if expected == path]
        if len(role_matches) != 1:
            raise ValueError(f"Unexpected retained path for {row['relative_path']}: {path}")
        if entry.get("exists") is not True or not path.is_file():
            raise ValueError(f"Retained prediction is missing: {path}")
        size = path.stat().st_size
        try:
            recorded_size = int(entry.get("size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid retained size for {path}") from exc
        if recorded_size != size:
            raise ValueError(f"Retained prediction size changed: {path}")
        digest = sha256(path)
        if entry.get("sha256") != digest:
            raise ValueError(f"Retained prediction SHA-256 changed: {path}")
        verified.append(
            {
                "role": role_matches[0],
                "path": str(path),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if set(observed_paths) != set(output_paths.values()):
        raise ValueError(f"Retention inventory is incomplete for {row['relative_path']}")
    return sorted(verified, key=lambda item: item["role"])


def validate_metrics(
    metrics: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
    retained: list[dict[str, Any]],
    row: dict[str, Any],
    run_id: str,
) -> None:
    expected = {
        "status": "completed_aligned_pointwise_development_plot",
        "evaluation_scope": "development_full",
        "run_id": run_id,
        "plot_id": row["plot_id"],
        "relative_path": row["relative_path"],
        "split": "dev",
        "dataset_split": "dev",
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "held_out_test_accessed": False,
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "iou_threshold": 0.5,
        "iou_threshold_operator": ">=",
        "point_correspondence": "source_row_index",
        "prediction_semantic_mapping": "pred_tree_id > 0 -> class 4; else 0",
        "reference_tree_classes": [4, 5, 6],
        "prediction_tree_classes": [4],
        "ignored_instance_labels": [-1, 0],
        "tuned_prediction_filtering": False,
        "min_predicted_instance_points": 0,
        "min_predicted_tree_fraction": 0.0,
    }
    for field, value in expected.items():
        _require_equal(metrics.get(field), value, f"Metric {field} mismatch")
    if not _same_path(metrics.get("inference_metadata", ""), metadata_path):
        raise ValueError(f"Metric inference path mismatch for {row['relative_path']}")
    adapted = next(item for item in retained if item["role"] == "adapted_npz")
    if not _same_path(metrics.get("prediction_npz", ""), adapted["path"]):
        raise ValueError(f"Metric prediction path mismatch for {row['relative_path']}")
    _require_equal(
        metrics.get("prediction_npz_sha256"),
        adapted["sha256"],
        "Metric prediction hash mismatch",
    )
    _require_equal(
        int(metrics.get("prediction_npz_size_bytes", -1)),
        adapted["size_bytes"],
        "Metric prediction size mismatch",
    )
    for metric_field, manifest_field in (
        ("point_count", "point_count"),
        ("reference_instance_count", "reference_tree_count"),
    ):
        _require_equal(
            int(metrics.get(metric_field, -1)),
            int(row[manifest_field]),
            f"Metric {metric_field} mismatch",
        )
    validation = metadata.get("dataset_validation", {})
    for field in ("input_sha256", "split_metadata_sha256"):
        _require_equal(validation.get(field), row[field], f"Inference {field} mismatch")
    _require_equal(
        int(validation.get("point_count", -1)),
        int(row["point_count"]),
        "Inference point count mismatch",
    )
    _require_equal(
        int(validation.get("reference_tree_count", -1)),
        int(row["reference_tree_count"]),
        "Inference reference count mismatch",
    )
    alignment = metadata.get("validation", {})
    if (
        alignment.get("row_count_match") is not True
        or alignment.get("row_order_preserved") is not True
    ):
        raise ValueError(f"Point alignment is not preserved for {row['relative_path']}")
    for field in ("source_point_count", "prediction_point_count"):
        _require_equal(
            int(alignment.get(field, -1)),
            int(row["point_count"]),
            f"Inference {field} mismatch",
        )
    coordinate_delta = float(alignment.get("max_abs_coordinate_delta_m", math.inf))
    coordinate_tolerance = float(
        alignment.get("row_coordinate_tolerance_m", -math.inf)
    )
    if (
        not math.isfinite(coordinate_delta)
        or not math.isfinite(coordinate_tolerance)
        or coordinate_delta < 0.0
        or coordinate_tolerance <= 0.0
        or coordinate_delta > coordinate_tolerance
    ):
        raise ValueError(f"Invalid coordinate alignment for {row['relative_path']}")

    tp = int(metrics.get("true_positives", -1))
    fp = int(metrics.get("false_positives", -1))
    fn = int(metrics.get("false_negatives", -1))
    predictions = int(metrics.get("prediction_instance_count", -1))
    references = int(metrics.get("reference_instance_count", -1))
    if min(tp, fp, fn, predictions, references) < 0:
        raise ValueError(f"Negative or missing metric counts for {row['relative_path']}")
    if predictions != tp + fp or references != tp + fn:
        raise ValueError(f"Metric instance/count identity failed for {row['relative_path']}")
    expected_precision = tp / (tp + fp) if tp + fp else 0.0
    expected_recall = tp / (tp + fn) if tp + fn else 0.0
    expected_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    for field, expected_value in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
    ):
        if not math.isclose(float(metrics.get(field, -1.0)), expected_value, abs_tol=1e-12):
            raise ValueError(f"Metric {field} is inconsistent for {row['relative_path']}")
    for field in (
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "mean_unweighted_coverage",
        "mean_weighted_coverage",
    ):
        value = float(metrics.get(field, -1.0))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Metric {field} is outside [0, 1] for {row['relative_path']}")
    evaluated_points = int(metrics.get("evaluated_point_count", -1))
    if not 0 < evaluated_points <= int(metrics["point_count"]):
        raise ValueError(f"Invalid evaluated point count for {row['relative_path']}")
    harmonized = metrics.get("harmonized")
    if not isinstance(harmonized, dict):
        raise ValueError(f"Metrics lack harmonized evidence for {row['relative_path']}")
    for field in (
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
    ):
        if harmonized.get(field) != metrics.get(field):
            raise ValueError(
                f"Top-level and harmonized {field} differ for {row['relative_path']}"
            )


def validate_evaluation_tables(
    evaluation_root: Path,
    row: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    matches = read_csv(evaluation_root / "matches.csv", MATCH_FIELDS)
    unmatched_predictions = read_csv(
        evaluation_root / "unmatched_predictions.csv",
        UNMATCHED_PREDICTION_FIELDS,
    )
    unmatched_references = read_csv(
        evaluation_root / "unmatched_references.csv",
        UNMATCHED_REFERENCE_FIELDS,
    )
    for records, expected_count, label in (
        (matches, int(metrics["true_positives"]), "matches"),
        (unmatched_predictions, int(metrics["false_positives"]), "unmatched predictions"),
        (unmatched_references, int(metrics["false_negatives"]), "unmatched references"),
    ):
        if len(records) != expected_count:
            raise ValueError(
                f"{label} count differs from metrics for {row['relative_path']}: "
                f"{len(records)} != {expected_count}"
            )
        if any(record.get("plot_id") != row["plot_id"] for record in records):
            raise ValueError(f"{label} contain a different plot ID for {row['relative_path']}")
    matched_predicted = [int(record["pred_tree_id"]) for record in matches]
    matched_references = [int(record["target_tree_id"]) for record in matches]
    unmatched_predicted = [
        int(record["pred_tree_id"]) for record in unmatched_predictions
    ]
    unmatched_reference_ids = [
        int(record["target_tree_id"]) for record in unmatched_references
    ]
    for values, label in (
        (matched_predicted, "matched prediction IDs"),
        (matched_references, "matched reference IDs"),
        (unmatched_predicted, "unmatched prediction IDs"),
        (unmatched_reference_ids, "unmatched reference IDs"),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {label} for {row['relative_path']}")
    if set(matched_predicted) & set(unmatched_predicted):
        raise ValueError(f"Matched prediction also marked unmatched for {row['relative_path']}")
    if set(matched_references) & set(unmatched_reference_ids):
        raise ValueError(f"Matched reference also marked unmatched for {row['relative_path']}")
    match_ious = [float(record["iou"]) for record in matches]
    if any(not 0.5 <= value <= 1.0 for value in match_ious):
        raise ValueError(f"Matched IoU violates the frozen threshold for {row['relative_path']}")
    observed_mean = statistics.fmean(match_ious) if match_ious else 0.0
    if not math.isclose(
        observed_mean,
        float(metrics["mean_matched_iou"]),
        abs_tol=1e-12,
    ):
        raise ValueError(f"Matched-pair IoUs differ from metrics for {row['relative_path']}")
    return matches, unmatched_predictions, unmatched_references


def prefix_records(
    records: list[dict[str, str]],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix = {field: row[field] for field in CONSOLIDATED_PREFIX_FIELDS}
    return [{**prefix, **record} for record in records]


def _mean(rows: list[dict[str, Any]], field: str) -> float | str:
    if not rows:
        return ""
    return statistics.fmean(float(row[field]) for row in rows)


def aggregate_rows(
    rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    site: str,
    expected_plots: int,
) -> dict[str, Any]:
    completed = [row for row in rows if row["result_status"] == "completed"]
    tp = sum(int(row["true_positives"]) for row in completed)
    fp = sum(int(row["false_positives"]) for row in completed)
    fn = sum(int(row["false_negatives"]) for row in completed)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    failed = expected_plots - len(completed)
    return {
        "method": "TreeLearn",
        "variant": "published_pretrained",
        "dataset_split": "dev",
        "site": site,
        "expected_plots": expected_plots,
        "completed_plots": len(completed),
        "failed_plots": failed,
        "point_count": sum(int(row["point_count"]) for row in completed),
        "evaluated_point_count": sum(
            int(row["evaluated_point_count"]) for row in completed
        ),
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in completed
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in completed
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_precision": _mean(completed, "precision"),
        "mean_plot_recall": _mean(completed, "recall"),
        "mean_plot_f1": _mean(completed, "f1"),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_plot_matched_iou": _mean(completed, "mean_matched_iou"),
        "mean_matched_iou_across_pairs": (
            statistics.fmean(float(row["iou"]) for row in match_rows)
            if match_rows
            else 0.0
        ),
        "mean_unweighted_coverage": _mean(completed, "mean_unweighted_coverage"),
        "mean_weighted_coverage": _mean(completed, "mean_weighted_coverage"),
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "result_status": (
            "completed_aligned_pointwise_development"
            if failed == 0
            else "development_with_documented_failures"
        ),
        "held_out_test_accessed": "false",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def summarise(
    manifest_path: Path,
    run_id: str,
    evaluation_root: Path,
    metadata_root: Path,
    output_root: Path,
    expected_benchmark_commit: str | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(f"Unsafe TreeLearn development run ID: {run_id!r}")
    if expected_benchmark_commit is not None and not re.fullmatch(
        r"[0-9a-f]{40}", expected_benchmark_commit
    ):
        raise ValueError("Expected benchmark commit must be a full lowercase SHA-1")
    manifest_rows, manifest_metadata = load_manifest(manifest_path)
    if manifest_metadata.get("held_out_test_accessed") is not False:
        raise ValueError("Manifest does not lock held-out test access")
    evaluation_root = evaluation_root.expanduser().resolve()
    metadata_root = metadata_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Inference metadata root does not exist: {metadata_root}")
    expected_safe_ids = {row["safe_plot_id"] for row in manifest_rows}
    observed_evaluation_dirs = {
        path.name for path in evaluation_root.iterdir() if path.is_dir()
    }
    partial_evaluation_dirs: dict[str, list[str]] = defaultdict(list)
    unexpected_evaluation_dirs: set[str] = set()
    for directory in observed_evaluation_dirs - expected_safe_ids:
        matching_safe_ids = [
            safe_id
            for safe_id in expected_safe_ids
            if directory.startswith(f".{safe_id}.partial.")
        ]
        if len(matching_safe_ids) == 1:
            partial_evaluation_dirs[matching_safe_ids[0]].append(directory)
        else:
            unexpected_evaluation_dirs.add(directory)
    if unexpected_evaluation_dirs:
        raise ValueError(
            "Per-plot evaluation directories contain plots outside the frozen "
            f"manifest: {sorted(unexpected_evaluation_dirs)}"
        )
    unexpected_inference = sorted(
        path.name
        for path in metadata_root.glob("*_inference.json")
        if path.name.removesuffix("_inference.json") not in expected_safe_ids
    )
    if unexpected_inference:
        raise ValueError(
            "Inference metadata contains plots outside the frozen development "
            f"manifest: {unexpected_inference}"
        )

    target_paths = {
        "plot_summary": output_root / "plot_summary.csv",
        "site_summary": output_root / "site_summary.csv",
        "development_summary": output_root / "development_summary.csv",
        "failures": output_root / "failures.csv",
        "matches": output_root / "matches.csv",
        "unmatched_predictions": output_root / "unmatched_predictions.csv",
        "unmatched_references": output_root / "unmatched_references.csv",
        "retention_manifest": output_root / "retention_manifest.json",
        "run_summary": output_root / "run_summary.json",
    }
    collisions = [path for path in target_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Development summary output already exists: "
            + ", ".join(str(path) for path in collisions)
        )

    plot_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    unmatched_references: list[dict[str, Any]] = []
    retention_records: list[dict[str, Any]] = []
    completed_provenance: list[dict[str, str]] = []

    for row in manifest_rows:
        per_plot_root = evaluation_root / row["safe_plot_id"]
        metadata_path = metadata_root / f"{row['safe_plot_id']}_inference.json"
        metrics_path = per_plot_root / "metrics.json"
        failure_path = per_plot_root / "status.json"
        if metrics_path.is_file() and failure_path.is_file():
            raise ValueError(f"Plot has both metrics and failure status: {row['relative_path']}")
        if not metrics_path.is_file() and not failure_path.is_file():
            inference_status = "missing"
            metadata_error = None
            if metadata_path.is_file():
                try:
                    metadata = load_object(metadata_path)
                    provenance = validate_inference_identity(
                        metadata,
                        row,
                        run_id,
                        require_completed=False,
                    )
                    if provenance is not None:
                        completed_provenance.append(provenance)
                    inference_status = str(metadata.get("status", "unknown"))
                    if inference_status == "completed":
                        retained = validate_retention(metadata, row)
                        retention_records.append(
                            {
                                "task_index": row["task_index"],
                                "plot_id": row["plot_id"],
                                "relative_path": row["relative_path"],
                                "collection": row["collection"],
                                "split": "dev",
                                "prediction_files": retained,
                                "inference_metadata": artifact_entry(metadata_path),
                                "evaluation_artifacts": [],
                                "retention_verified": True,
                                "evaluation_status": "documented_missing_task_output",
                            }
                        )
                except Exception as exc:  # preserve a final accounting record
                    inference_status = "invalid_metadata"
                    metadata_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            reason_payload = {
                "type": "MissingTaskOutput",
                "message": "Array task wrote neither metrics.json nor status.json",
            }
            if partial_evaluation_dirs.get(row["safe_plot_id"]):
                reason_payload["partial_evaluation_directories"] = sorted(
                    partial_evaluation_dirs[row["safe_plot_id"]]
                )
            if metadata_error is not None:
                reason_payload["metadata_error"] = metadata_error
            reason = json.dumps(reason_payload, sort_keys=True)
            failure_row = {
                **{
                    field: row[field]
                    for field in (
                        "task_index",
                        "plot_id",
                        "safe_plot_id",
                        "relative_path",
                        "collection",
                        "split",
                    )
                },
                "run_id": run_id,
                "status": "documented_missing_task_output",
                "reason": reason,
                "inference_status": inference_status,
                "inference_metadata": (
                    str(metadata_path) if metadata_path.is_file() else ""
                ),
                "failure_record": "",
                "held_out_test_accessed": "false",
            }
            failures.append(failure_row)
            plot_rows.append(
                {
                    **{
                        field: row[field]
                        for field in (
                            "task_index",
                            "plot_id",
                            "safe_plot_id",
                            "relative_path",
                            "collection",
                            "split",
                        )
                    },
                    "run_id": run_id,
                    "result_status": "documented_missing_task_output",
                    "point_count": row["point_count"],
                    "inference_metadata": failure_row["inference_metadata"],
                    "metrics_json": "",
                    "failure_reason": reason,
                    "held_out_test_accessed": "false",
                }
            )
            continue

        if failure_path.is_file():
            failure = load_object(failure_path)
            expected_failure_statuses = {
                "documented_inference_failure",
                "documented_evaluation_failure",
            }
            if failure.get("status") not in expected_failure_statuses:
                raise ValueError(f"Unrecognised failure status: {failure_path}")
            for field, expected in (
                ("run_id", run_id),
                ("task_index", row["task_index"]),
                ("plot_id", row["plot_id"]),
                ("relative_path", row["relative_path"]),
                ("collection", row["collection"]),
                ("split", "dev"),
            ):
                _require_equal(failure.get(field), expected, f"Failure {field} mismatch")
            if failure.get("held_out_test_accessed") is not False:
                raise ValueError(f"Failure record does not lock held-out test: {failure_path}")
            inference_status = "missing"
            failure_status = str(failure["status"])
            if failure_status == "documented_evaluation_failure":
                metadata = load_object(metadata_path)
                provenance = validate_inference_identity(
                    metadata,
                    row,
                    run_id,
                    require_completed=True,
                )
                if provenance is not None:
                    completed_provenance.append(provenance)
                retained = validate_retention(metadata, row)
                inference_status = "completed"
                retention_records.append(
                    {
                        "task_index": row["task_index"],
                        "plot_id": row["plot_id"],
                        "relative_path": row["relative_path"],
                        "collection": row["collection"],
                        "split": "dev",
                        "prediction_files": retained,
                        "inference_metadata": artifact_entry(metadata_path),
                        "evaluation_artifacts": [artifact_entry(failure_path)],
                        "retention_verified": True,
                        "evaluation_status": failure_status,
                    }
                )
            elif metadata_path.is_file():
                metadata = load_object(metadata_path)
                provenance = validate_inference_identity(
                    metadata,
                    row,
                    run_id,
                    require_completed=False,
                )
                if provenance is not None:
                    completed_provenance.append(provenance)
                if metadata.get("status") == "completed":
                    raise ValueError(
                        f"Documented inference failure has completed metadata: {metadata_path}"
                    )
                inference_status = str(metadata.get("status", "unknown"))
            reason_payload = failure.get("error") or failure.get("inference_error")
            if not reason_payload:
                raise ValueError(f"Documented failure has no reason: {failure_path}")
            reason = json.dumps(reason_payload, sort_keys=True)
            failure_row = {
                **{
                    field: row[field]
                    for field in (
                        "task_index",
                        "plot_id",
                        "safe_plot_id",
                        "relative_path",
                        "collection",
                        "split",
                    )
                },
                "run_id": run_id,
                "status": failure["status"],
                "reason": reason,
                "inference_status": inference_status,
                "inference_metadata": str(metadata_path) if metadata_path.is_file() else "",
                "failure_record": str(failure_path),
                "held_out_test_accessed": "false",
            }
            failures.append(failure_row)
            plot_rows.append(
                {
                    **{
                        field: row[field]
                        for field in (
                            "task_index",
                            "plot_id",
                            "safe_plot_id",
                            "relative_path",
                            "collection",
                            "split",
                        )
                    },
                    "run_id": run_id,
                    "result_status": failure["status"],
                    "point_count": row["point_count"],
                    "inference_metadata": failure_row["inference_metadata"],
                    "metrics_json": "",
                    "failure_reason": reason,
                    "held_out_test_accessed": "false",
                }
            )
            continue

        metadata = load_object(metadata_path)
        provenance = validate_inference_identity(
            metadata, row, run_id, require_completed=True
        )
        if provenance is not None:
            completed_provenance.append(provenance)
        retained = validate_retention(metadata, row)
        metrics = load_object(metrics_path)
        validate_metrics(metrics, metadata_path, metadata, retained, row, run_id)
        plot_matches, plot_unmatched_predictions, plot_unmatched_references = (
            validate_evaluation_tables(per_plot_root, row, metrics)
        )
        matches.extend(prefix_records(plot_matches, row))
        unmatched_predictions.extend(prefix_records(plot_unmatched_predictions, row))
        unmatched_references.extend(prefix_records(plot_unmatched_references, row))
        metric_fields = (
            "point_count",
            "evaluated_point_count",
            "prediction_instance_count",
            "reference_instance_count",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "mean_unweighted_coverage",
            "mean_weighted_coverage",
        )
        plot_rows.append(
            {
                **{
                    field: row[field]
                    for field in (
                        "task_index",
                        "plot_id",
                        "safe_plot_id",
                        "relative_path",
                        "collection",
                        "split",
                    )
                },
                "run_id": run_id,
                "result_status": "completed",
                **{field: metrics[field] for field in metric_fields},
                "inference_metadata": str(metadata_path),
                "metrics_json": str(metrics_path),
                "failure_reason": "",
                "held_out_test_accessed": "false",
            }
        )
        evaluation_artifacts = [
            artifact_entry(per_plot_root / name)
            for name in (
                "metrics.json",
                "matches.csv",
                "unmatched_predictions.csv",
                "unmatched_references.csv",
            )
        ]
        retention_records.append(
            {
                "task_index": row["task_index"],
                "plot_id": row["plot_id"],
                "relative_path": row["relative_path"],
                "collection": row["collection"],
                "split": "dev",
                "prediction_files": retained,
                "inference_metadata": artifact_entry(metadata_path),
                "evaluation_artifacts": evaluation_artifacts,
                "retention_verified": True,
                "evaluation_status": "completed",
            }
        )

    benchmark_commits = {
        record["benchmark_commit"] for record in completed_provenance
    }
    if len(benchmark_commits) > 1:
        raise ValueError(
            "Completed TreeLearn inference records use different benchmark commits: "
            f"{sorted(benchmark_commits)}"
        )
    observed_benchmark_commit = (
        next(iter(benchmark_commits)) if benchmark_commits else None
    )
    if (
        expected_benchmark_commit is not None
        and observed_benchmark_commit is not None
        and observed_benchmark_commit != expected_benchmark_commit
    ):
        raise ValueError(
            "Completed TreeLearn inference benchmark commit differs from submission: "
            f"expected {expected_benchmark_commit}, found {observed_benchmark_commit}"
        )
    run_provenance = {
        "training_mode": "published_pretrained",
        "checkpoint_md5": EXPECTED_CHECKPOINT_MD5,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "benchmark_commit": observed_benchmark_commit,
        "expected_benchmark_commit": expected_benchmark_commit,
        "validated_completed_inference_records": len(completed_provenance),
    }

    plot_rows.sort(key=lambda row: int(row["task_index"]))
    failures.sort(key=lambda row: int(row["task_index"]))
    matches.sort(key=lambda row: (int(row["task_index"]), int(row["pred_tree_id"])))
    unmatched_predictions.sort(
        key=lambda row: (int(row["task_index"]), int(row["pred_tree_id"]))
    )
    unmatched_references.sort(
        key=lambda row: (int(row["task_index"]), int(row["target_tree_id"]))
    )

    site_rows: list[dict[str, Any]] = []
    for site, expected_count in EXPECTED_DEVELOPMENT_SITE_COUNTS.items():
        site_plot_rows = [row for row in plot_rows if row["collection"] == site]
        site_match_rows = [row for row in matches if row["collection"] == site]
        site_rows.append(
            aggregate_rows(site_plot_rows, site_match_rows, site, expected_count)
        )
    development_row = aggregate_rows(
        plot_rows,
        matches,
        "ALL",
        len(manifest_rows),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(target_paths["plot_summary"], plot_rows, PER_PLOT_FIELDS)
    write_csv(target_paths["site_summary"], site_rows, AGGREGATE_FIELDS)
    write_csv(target_paths["development_summary"], [development_row], AGGREGATE_FIELDS)
    write_csv(target_paths["failures"], failures, FAILURE_FIELDS)
    write_csv(
        target_paths["matches"],
        matches,
        CONSOLIDATED_PREFIX_FIELDS + MATCH_FIELDS,
    )
    write_csv(
        target_paths["unmatched_predictions"],
        unmatched_predictions,
        CONSOLIDATED_PREFIX_FIELDS + UNMATCHED_PREDICTION_FIELDS,
    )
    write_csv(
        target_paths["unmatched_references"],
        unmatched_references,
        CONSOLIDATED_PREFIX_FIELDS + UNMATCHED_REFERENCE_FIELDS,
    )

    completed_plot_count = sum(
        row["result_status"] == "completed" for row in plot_rows
    )
    retained_file_count = sum(
        len(record["prediction_files"]) for record in retention_records
    )
    retained_size = sum(
        int(item["size_bytes"])
        for record in retention_records
        for item in record["prediction_files"]
    )
    retention_manifest = {
        "schema_version": 1,
        "status": (
            "retention_verified"
            if not failures
            else "retention_verified_for_completed_plots_with_documented_failures"
        ),
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "run_id": run_id,
        "dataset_split": "dev",
        "held_out_test_accessed": False,
        "provenance": run_provenance,
        "manifest": artifact_entry(manifest_path.expanduser().resolve()),
        "expected_plots": len(manifest_rows),
        "completed_plots": completed_plot_count,
        "inference_outputs_retained": len(retention_records),
        "documented_failures": len(failures),
        "verified_prediction_file_count": retained_file_count,
        "verified_prediction_size_bytes": retained_size,
        "all_completed_prediction_retention_verified": True,
        "complete_development_prediction_set_retained": (
            len(retention_records) == len(manifest_rows)
        ),
        "plots": retention_records,
    }
    target_paths["retention_manifest"].write_text(
        json.dumps(retention_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_artifacts = {
        key: artifact_entry(path)
        for key, path in target_paths.items()
        if key != "run_summary" and path.is_file()
    }
    run_summary = {
        "schema_version": 1,
        "status": development_row["result_status"],
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "variant": "published_pretrained",
        "run_id": run_id,
        "dataset_split": "dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_test_accessed": False,
        "provenance": run_provenance,
        "expected_plots": len(manifest_rows),
        "completed_plots": completed_plot_count,
        "documented_failures": len(failures),
        "retention_status": retention_manifest["status"],
        "site_counts": EXPECTED_DEVELOPMENT_SITE_COUNTS,
        "development_metrics": development_row,
        "outputs": output_artifacts,
        "next_gate": (
            "manual_review_before_any_held_out_test_route"
            if not failures
            else "resolve_documented_development_failures_before_any_test_route"
        ),
    }
    target_paths["run_summary"].write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarise one frozen TreeLearn development run."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-benchmark-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarise(
        Path(args.manifest),
        args.run_id,
        Path(args.evaluation_root),
        Path(args.metadata_root),
        Path(args.output_root),
        args.expected_benchmark_commit,
    )
    print(f"status={summary['status']}")
    print(f"completed_plots={summary['completed_plots']}")
    print(f"documented_failures={summary['documented_failures']}")
    summary_path = (
        Path(args.output_root).expanduser().resolve() / "development_summary.csv"
    )
    print(f"development_summary={summary_path}")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
