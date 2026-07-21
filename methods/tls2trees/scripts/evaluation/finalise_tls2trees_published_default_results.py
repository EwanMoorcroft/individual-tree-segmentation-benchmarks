"""Publish a completed TLS2trees published-default FOR-instance test result.

The command is deliberately post-inference.  It verifies the immutable test
configuration, the exact 11-plot/22-metric result, and every retained aligned
prediction before writing repository-safe evidence and upserting the three
TLS2trees registry rows.  It never selects or changes a configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
EVALUATION = Path(__file__).resolve().parent
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from published_default_test_common import (  # noqa: E402
    TARGETS,
    validate_exact_manifest,
    validate_frozen_configuration,
)
from tls2trees_publication import (  # noqa: E402
    preflight_text_target,
    publication_lock,
    publish_text_bundle,
    validate_git_worktree,
)


EXPECTED_PLOTS = 11
EXPECTED_REFERENCES = 323
EXPECTED_UPSTREAM_COMMIT = "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
EXPECTED_MODEL_SHA256 = (
    "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"
)
VARIANT = "published_default"
TRAINING_MODE = "external_training_only"
EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
PROTOCOL = "for_instance_pointwise_class3_ignore"
SOURCE_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
PUBLIC_MASK = (
    "classification_3_excluded_then_union_of_reference_tree_and_predicted_tree_points"
)
MATCHING = "maximum_cardinality_one_to_one"
IOU_THRESHOLD = 0.5
HEADLINE_GROUP = "held_out_test_tls2trees_class3_ignore"
DIAGNOSTIC_GROUP = "tls2trees_leaf_off_class3_ignore_diagnostic"

PLOT_FIELDS = [
    "method_slug", "variant", "training_mode", "target", "run_id",
    "dataset_split", "plot_index", "relative_path", "collection",
    "point_count", "predicted_instances", "reference_instances",
    "true_positives", "false_positives", "false_negatives", "precision",
    "recall", "f1", "mean_matched_iou", "evaluation_protocol",
    "matching_policy", "evaluation_mask", "iou_threshold",
    "source_metrics_sha256", "aligned_prediction_sha256",
]
SUMMARY_FIELDS = [
    "method_slug", "run_id", "variant", "training_mode", "target",
    "dataset_split", "site", "plots", "point_count",
    "predicted_instances", "reference_instances", "true_positives",
    "false_positives", "false_negatives", "mean_plot_f1",
    "mean_plot_precision", "mean_plot_recall", "micro_precision",
    "micro_recall", "micro_f1", "evaluation_protocol", "matching_policy",
    "evaluation_mask", "held_out_test_accessed", "retention_status",
    "result_status",
]
RESULT_FIELDS = [
    "dataset_slug", "method_slug", "variant", "run_id", "training_mode",
    "result_role", "evaluation_protocol", "matching_policy",
    "evaluation_mask", "evaluation_split", "comparable_group", "plots",
    "predicted_instances", "reference_instances", "true_positives",
    "false_positives", "false_negatives", "mean_plot_f1",
    "micro_precision", "micro_recall", "micro_f1", "retention_status",
    "result_status", "held_out_test_accessed", "brief_note",
]
RETENTION_FIELDS = [
    "method_slug", "variant", "retention_profile", "run_id",
    "evaluation_split", "prediction_scope", "retained_file_count",
    "retained_size_bytes", "hash_status", "storage_status",
    "future_metrics_without_inference", "prediction_root", "metrics_root",
    "retention_manifest", "retention_manifest_sha256", "evidence_path",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def csv_text(fields: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = StringIO(newline="")
    writer = csv.DictWriter(
        handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected CSV schema in {path}")
        return list(reader)


def publish_writes(
    writes: dict[Path, str],
    *,
    project_root: Path,
    enforce_head_baseline: bool,
    expected_head: str | None,
) -> None:
    """Commit a deterministic, symlink-safe publication while locked."""

    publish_text_bundle(
        writes,
        project_root=project_root,
        staging_suffix=".tls2trees-published-default-finalisation.tmp",
        replace=os.replace,
        enforce_head_baseline=enforce_head_baseline,
        expected_head=expected_head,
    )


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"{label} is missing or its SHA-256 changed: {path}")


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Evidence is outside the project root: {path}") from exc


def upsert_one(
    path: Path,
    fields: list[str],
    row: dict[str, Any],
    match: dict[str, str],
) -> str:
    rows = read_csv(path, fields)
    indices = [
        index
        for index, existing in enumerate(rows)
        if all(existing[field] == value for field, value in match.items())
    ]
    if len(indices) > 1:
        raise ValueError(f"Registry contains duplicate rows for {match}")
    if indices:
        rows[indices[0]] = row
    else:
        rows.append(row)
    return csv_text(fields, rows)


def division(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate(rows: list[dict[str, Any]], site: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty result set")
    predicted = sum(int(row["predicted_instances"]) for row in rows)
    references = sum(int(row["reference_instances"]) for row in rows)
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    return {
        "method_slug": "tls2trees",
        "run_id": rows[0]["run_id"],
        "variant": VARIANT,
        "training_mode": TRAINING_MODE,
        "target": rows[0]["target"],
        "dataset_split": "test",
        "site": site,
        "plots": len(rows),
        "point_count": sum(int(row["point_count"]) for row in rows),
        "predicted_instances": predicted,
        "reference_instances": references,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "mean_plot_precision": sum(float(row["precision"]) for row in rows)
        / len(rows),
        "mean_plot_recall": sum(float(row["recall"]) for row in rows) / len(rows),
        "micro_precision": division(tp, tp + fp),
        "micro_recall": division(tp, tp + fn),
        "micro_f1": division(2 * tp, 2 * tp + fp + fn),
        "evaluation_protocol": PROTOCOL,
        "matching_policy": MATCHING,
        "evaluation_mask": PUBLIC_MASK,
        "held_out_test_accessed": "true",
        "retention_status": "retention_verified",
        "result_status": "completed_aligned_pointwise_test",
    }


def equal_number(actual: Any, expected: Any) -> bool:
    return abs(float(actual) - float(expected)) <= 1e-12


def reconcile_summary_metric(
    *,
    source: dict[str, Any],
    metrics: dict[str, Any],
    target: str,
    task: int,
    plot: dict[str, Any],
) -> None:
    """Bind the copied summary row to its exact hashed metric evidence."""

    context = f"{target}:{task}"
    expected_identity = {
        "split": "test",
        "target": target,
        "plot_id": plot["safe_plot_id"],
        "relative_path": plot["relative_path"],
        "status": source.get("status"),
        "safe_for_scoring": source.get("safe_for_scoring"),
    }
    for field, expected in expected_identity.items():
        if metrics.get(field) != expected:
            raise ValueError(
                "Published-default summary/metric evidence mismatch: "
                f"{context}:{field}"
            )

    count_fields = (
        "prediction_instance_count",
        "reference_instance_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "oversegmented_reference_count",
        "undersegmented_prediction_count",
    )
    for field in count_fields:
        summary_value = source.get(field)
        metric_value = metrics.get(field)
        if (
            type(summary_value) is not int
            or type(metric_value) is not int
            or summary_value != metric_value
        ):
            raise ValueError(
                "Published-default summary/metric evidence mismatch: "
                f"{context}:{field}"
            )

    summary_raw = source.get("raw_prediction_instance_count")
    metric_raw = metrics.get("semantic_ignore", {}).get(
        "raw_prediction_instance_count"
    )
    if (
        type(summary_raw) is not int
        or type(metric_raw) is not int
        or summary_raw != metric_raw
    ):
        raise ValueError(
            "Published-default summary/metric evidence mismatch: "
            f"{context}:raw_prediction_instance_count"
        )

    for field in ("precision", "recall", "f1", "mean_matched_iou"):
        summary_value = source.get(field)
        metric_value = metrics.get(field)
        if (
            type(summary_value) not in (int, float)
            or type(metric_value) not in (int, float)
            or not (-float("inf") < float(summary_value) < float("inf"))
            or not (-float("inf") < float(metric_value) < float("inf"))
            or summary_value != metric_value
        ):
            raise ValueError(
                "Published-default summary/metric evidence mismatch: "
                f"{context}:{field}"
            )


def verify_source_tables(
    summary: dict[str, Any], plot_csv: Path, target_csv: Path
) -> None:
    plot_rows = summary.get("plot_metrics")
    target_rows = summary.get("aggregates")
    if not isinstance(plot_rows, list) or not plot_rows:
        raise ValueError("Summary has no plot metrics")
    if not isinstance(target_rows, list) or not target_rows:
        raise ValueError("Summary has no target aggregates")
    for path, expected_rows, label in (
        (plot_csv, plot_rows, "plot_metrics.csv"),
        (target_csv, target_rows, "target_summary.csv"),
    ):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            actual_rows = list(reader)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(set(reader.fieldnames))
            or set(reader.fieldnames) != set(expected_rows[0])
            or len(actual_rows) != len(expected_rows)
        ):
            raise ValueError(f"{label} schema or row count differs from the summary")
        for actual, expected in zip(actual_rows, expected_rows):
            for field in reader.fieldnames:
                expected_text = "" if expected[field] is None else str(expected[field])
                if actual[field] != expected_text:
                    raise ValueError(f"{label} differs from the completed summary")


def verify_summary(
    *,
    summary: dict[str, Any],
    run_id: str,
    workflow_sha256: str,
    published_sha256: str,
    benchmark_sha256: str,
) -> None:
    expected = {
        "status": "published_default_test_completed",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "split": "test",
        "workflow_run_id": run_id,
        "workflow_config_sha256": workflow_sha256,
        "published_config_sha256": published_sha256,
        "benchmark_config_sha256": benchmark_sha256,
        "expected_plot_count": EXPECTED_PLOTS,
        "expected_metric_count": EXPECTED_PLOTS * len(TARGETS),
        "valid_metric_count": EXPECTED_PLOTS * len(TARGETS),
        "held_out_test_accessed": True,
        "held_out_accuracy_metrics_computed": True,
        "configuration_selected_from_for_instance_metrics": False,
        "configuration_changed_after_test": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Published-default summary has unexpected {field}")


def collect_results(
    *,
    project_root: Path,
    run_id: str,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    retention: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    plots = {int(row["task_index"]): row for row in manifest["plots"]}
    retained_files = retention.get("files")
    if not isinstance(retained_files, list) or len(retained_files) != 22:
        raise ValueError("Retention evidence must contain exactly 22 files")
    retained_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in retained_files:
        key = (str(record.get("target")), int(record.get("plot_index", -1)))
        if key in retained_by_key:
            raise ValueError(f"Duplicate retained prediction: {key}")
        retained_by_key[key] = record

    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    public_retained: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in summary.get("plot_metrics", []):
        target = str(source.get("target"))
        task = int(source.get("task_index", -1))
        key = (target, task)
        if target not in TARGETS or task not in plots or key in seen:
            raise ValueError(f"Unknown or duplicate plot result: {key}")
        seen.add(key)
        plot = plots[task]
        if (
            source.get("configuration_id") != VARIANT
            or source.get("status") != "evaluated"
            or source.get("safe_for_scoring") is not True
            or source.get("collection") != plot["collection"]
            or source.get("safe_plot_id") != plot["safe_plot_id"]
            or source.get("relative_path") != plot["relative_path"]
        ):
            raise ValueError(f"Plot provenance changed: {key}")
        metrics_path = Path(str(source["metrics_path"])).expanduser().resolve()
        require_hash(metrics_path, str(source["metrics_sha256"]), "plot metric")
        metrics = load_object(metrics_path)
        if (
            metrics.get("evaluator") != EVALUATOR
            or metrics.get("evaluation_mask") != SOURCE_MASK
            or metrics.get("matching_policy") != MATCHING
            or float(metrics.get("iou_threshold", -1)) != IOU_THRESHOLD
            or metrics.get("semantic_ignore", {}).get("ignored_semantic_classes")
            != [3]
            or metrics.get("status") != "evaluated"
            or metrics.get("safe_for_scoring") is not True
        ):
            raise ValueError(f"Metric protocol changed: {metrics_path}")
        reconcile_summary_metric(
            source=source,
            metrics=metrics,
            target=target,
            task=task,
            plot=plot,
        )
        record = retained_by_key.get(key)
        if record is None:
            raise ValueError(f"Retained prediction is missing: {key}")
        prediction = project_root / str(record["relative_path"])
        alignment = project_root / str(record["alignment_metadata_relative_path"])
        require_hash(prediction, str(record["sha256"]), "retained prediction")
        require_hash(
            alignment,
            str(record["alignment_metadata_sha256"]),
            "alignment metadata",
        )
        alignment_metadata = load_object(alignment)
        alignment_prediction = alignment_metadata.get("aligned_prediction_npz")
        if (
            alignment_metadata.get("status") != "passed"
            or alignment_metadata.get("target") != target
            or not isinstance(alignment_prediction, str)
            or Path(alignment_prediction).expanduser().resolve()
            != prediction.resolve()
            or alignment_metadata.get("aligned_prediction_npz_sha256")
            != digest(prediction)
        ):
            raise ValueError(
                f"Alignment metadata retained-evidence binding changed: {key}"
            )
        metric_prediction = metrics.get("aligned_predictions_npz")
        metric_alignment = metrics.get("alignment_metadata_json")
        metric_prediction_sha256 = metrics.get("aligned_predictions_npz_sha256")
        if (
            not isinstance(metric_prediction, str)
            or not isinstance(metric_alignment, str)
            or Path(metric_prediction).expanduser().resolve() != prediction.resolve()
            or (
                metric_prediction_sha256 is not None
                and metric_prediction_sha256 != digest(prediction)
            )
            or Path(metric_alignment).expanduser().resolve() != alignment.resolve()
            or metrics.get("alignment_metadata_sha256") != digest(alignment)
        ):
            raise ValueError(f"Metric retained-evidence binding changed: {key}")
        if (
            prediction.resolve() != Path(str(source["prediction_path"])).resolve()
            or digest(prediction) != source.get("prediction_sha256")
            or digest(alignment) != source.get("alignment_metadata_sha256")
            or int(record.get("size_bytes", -1)) != prediction.stat().st_size
            or record.get("format") != "npz"
            or record.get("point_correspondence") != "source_row_index"
            or record.get("plot_id") != plot["safe_plot_id"]
            or record.get("relative_path", "").startswith("/")
            or record.get("alignment_metadata_relative_path", "").startswith("/")
        ):
            raise ValueError(f"Retained prediction provenance changed: {key}")
        row = {
            "method_slug": "tls2trees",
            "variant": VARIANT,
            "training_mode": TRAINING_MODE,
            "target": target,
            "run_id": run_id,
            "dataset_split": "test",
            "plot_index": task,
            "relative_path": plot["relative_path"],
            "collection": plot["collection"],
            "point_count": int(plot["point_count"]),
            "predicted_instances": int(metrics["prediction_instance_count"]),
            "reference_instances": int(metrics["reference_instance_count"]),
            "true_positives": int(metrics["true_positives"]),
            "false_positives": int(metrics["false_positives"]),
            "false_negatives": int(metrics["false_negatives"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "mean_matched_iou": float(metrics["mean_matched_iou"]),
            "evaluation_protocol": PROTOCOL,
            "matching_policy": MATCHING,
            "evaluation_mask": PUBLIC_MASK,
            "iou_threshold": IOU_THRESHOLD,
            "source_metrics_sha256": digest(metrics_path),
            "aligned_prediction_sha256": digest(prediction),
        }
        rows_by_target[target].append(row)
        public_retained.append(
            {
                "target": target,
                "plot_index": task,
                "plot_id": plot["safe_plot_id"],
                "relative_path": project_relative(prediction, project_root),
                "size_bytes": prediction.stat().st_size,
                "sha256": digest(prediction),
                "format": "npz",
                "point_correspondence": "source_row_index",
                "alignment_metadata_relative_path": project_relative(
                    alignment, project_root
                ),
                "alignment_metadata_sha256": digest(alignment),
            }
        )
    expected_keys = {
        (target, task) for target in TARGETS for task in range(EXPECTED_PLOTS)
    }
    if seen != expected_keys or set(retained_by_key) != expected_keys:
        raise ValueError("Evidence does not contain the exact 22 target/plot pairs")
    for target in TARGETS:
        rows_by_target[target].sort(key=lambda row: int(row["plot_index"]))
        overall = aggregate(rows_by_target[target], "ALL")
        if overall["plots"] != EXPECTED_PLOTS:
            raise ValueError(f"Unexpected plot count for {target}")
        if overall["reference_instances"] != EXPECTED_REFERENCES:
            raise ValueError(f"Unexpected reference count for {target}")
        recorded = [
            row for row in summary["aggregates"] if row.get("target") == target
        ]
        if len(recorded) != 1:
            raise ValueError(f"Summary requires one aggregate for {target}")
        for public_field, source_field in (
            ("predicted_instances", "prediction_instance_count"),
            ("reference_instances", "reference_instance_count"),
            ("true_positives", "true_positives"),
            ("false_positives", "false_positives"),
            ("false_negatives", "false_negatives"),
            ("micro_precision", "precision"),
            ("micro_recall", "recall"),
            ("micro_f1", "micro_f1"),
            ("mean_plot_f1", "mean_plot_f1"),
        ):
            if not equal_number(overall[public_field], recorded[0][source_field]):
                raise ValueError(f"Aggregate mismatch for {target}:{public_field}")
    public_retained.sort(key=lambda row: (row["target"], row["plot_index"]))
    return dict(rows_by_target), public_retained


def result_row(
    overall: dict[str, Any], role: str, group: str, note: str
) -> dict[str, Any]:
    return {
        "dataset_slug": "for-instance",
        "method_slug": "tls2trees",
        "variant": VARIANT,
        "run_id": overall["run_id"],
        "training_mode": TRAINING_MODE,
        "result_role": role,
        "evaluation_protocol": PROTOCOL,
        "matching_policy": MATCHING,
        "evaluation_mask": PUBLIC_MASK,
        "evaluation_split": "test",
        "comparable_group": group,
        "plots": overall["plots"],
        "predicted_instances": overall["predicted_instances"],
        "reference_instances": overall["reference_instances"],
        "true_positives": overall["true_positives"],
        "false_positives": overall["false_positives"],
        "false_negatives": overall["false_negatives"],
        "mean_plot_f1": overall["mean_plot_f1"],
        "micro_precision": overall["micro_precision"],
        "micro_recall": overall["micro_recall"],
        "micro_f1": overall["micro_f1"],
        "retention_status": "retention_verified",
        "result_status": "completed_aligned_pointwise_test",
        "held_out_test_accessed": "true",
        "brief_note": note,
    }


def reject_private_text(writes: dict[Path, str], project_root: Path) -> None:
    forbidden = (
        str(project_root), "/users/", "/home/", "/mnt/", "fastscratch",
        "barkla", "sgemoorc",
    )
    for path, content in writes.items():
        for token in forbidden:
            if token and token.lower() in content.lower():
                raise ValueError(f"Private path or host token in public output {path}")


def _finalise_locked(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.expanduser().resolve()
    workflow_config = args.workflow_config.expanduser().resolve()
    published_config = args.published_config.expanduser().resolve()
    benchmark_config = args.benchmark_config.expanduser().resolve()
    manifest_path = args.manifest_json.expanduser().resolve()
    summary_path = args.summary_json.expanduser().resolve()
    plot_csv = args.plot_csv.expanduser().resolve()
    target_csv = args.target_csv.expanduser().resolve()
    source_retention_path = args.source_retention_json.expanduser().resolve()
    if args.upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("Unexpected TLS2trees upstream commit")
    if args.model_sha256 != EXPECTED_MODEL_SHA256:
        raise ValueError("Unexpected bundled model SHA-256")
    require_hash(workflow_config, args.workflow_config_sha256, "workflow config")
    require_hash(published_config, args.published_config_sha256, "published config")
    require_hash(benchmark_config, args.benchmark_config_sha256, "benchmark config")
    workflow, _, published, _ = validate_frozen_configuration(
        workflow_config, published_config
    )
    if (
        published.get("method", {}).get("executable_pin", {}).get("commit")
        != args.upstream_commit
        or published.get("method", {}).get("bundled_fsct_model", {}).get("sha256")
        != args.model_sha256
    ):
        raise ValueError("Published configuration provenance changed")
    manifest = load_object(manifest_path)
    require_hash(manifest_path, args.manifest_sha256, "test manifest")
    validate_exact_manifest(manifest, workflow)
    summary = load_object(summary_path)
    verify_summary(
        summary=summary,
        run_id=args.run_id,
        workflow_sha256=args.workflow_config_sha256,
        published_sha256=args.published_config_sha256,
        benchmark_sha256=args.benchmark_config_sha256,
    )
    if summary.get("manifest_sha256") != args.manifest_sha256:
        raise ValueError("Summary manifest SHA-256 changed")
    require_hash(
        source_retention_path,
        str(summary.get("retention_manifest_sha256")),
        "source retention manifest",
    )
    verify_source_tables(summary, plot_csv, target_csv)
    source_retention = load_object(source_retention_path)
    retention_expected = {
        "status": "retention_verified",
        "dataset": "FOR-instance",
        "dataset_split": "test",
        "method": "TLS2trees",
        "variant": VARIANT,
        "run_id": args.run_id,
        "expected_files": 22,
        "verified_prediction_files": 22,
        "manifest_sha256": args.manifest_sha256,
        "workflow_config_sha256": args.workflow_config_sha256,
        "published_config_sha256": args.published_config_sha256,
        "benchmark_config_sha256": args.benchmark_config_sha256,
        "configuration_changed_after_test": False,
    }
    for field, value in retention_expected.items():
        if source_retention.get(field) != value:
            raise ValueError(f"Retention evidence has unexpected {field}")
    rows_by_target, retained = collect_results(
        project_root=project_root,
        run_id=args.run_id,
        summary=summary,
        manifest=manifest,
        retention=source_retention,
    )
    retained_size = sum(int(row["size_bytes"]) for row in retained)
    if int(source_retention.get("verified_prediction_size_bytes", -1)) != retained_size:
        raise ValueError("Retained prediction size total changed")

    sites: dict[str, list[dict[str, Any]]] = {}
    overall: dict[str, dict[str, Any]] = {}
    for target, plot_rows in rows_by_target.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in plot_rows:
            grouped[str(row["collection"])].append(row)
        sites[target] = [aggregate(grouped[name], name) for name in sorted(grouped)]
        overall[target] = aggregate(plot_rows, "ALL")

    prefix = "tls2trees_published_default"
    public_paths = {
        "leaf_on_plot": args.examples_dir / f"{prefix}_test_plot_results.csv",
        "leaf_on_site": args.examples_dir / f"{prefix}_test_site_results.csv",
        "leaf_on_overall": args.examples_dir / f"{prefix}_test_results.csv",
        "leaf_off_plot": args.examples_dir
        / f"{prefix}_leaf_off_test_plot_diagnostic.csv",
        "leaf_off_site": args.examples_dir
        / f"{prefix}_leaf_off_test_site_diagnostic.csv",
        "leaf_off_overall": args.examples_dir
        / f"{prefix}_leaf_off_test_diagnostic.csv",
        "retention": args.examples_dir
        / f"{prefix}_prediction_retention_manifest.json",
        "provenance": args.examples_dir / f"{prefix}_test_provenance.json",
    }
    public_retention = {
        "status": "retention_verified",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "training_mode": TRAINING_MODE,
        "run_id": args.run_id,
        "dataset_split": "test",
        "targets": list(TARGETS),
        "expected_plots_per_target": EXPECTED_PLOTS,
        "verified_prediction_files": len(retained),
        "verified_prediction_size_bytes": retained_size,
        "hash_algorithm": "sha256",
        "prediction_contract": "one_instance_label_per_source_row",
        "future_metrics_without_inference": True,
        "held_out_test_accessed": True,
        "configuration_selected_from_for_instance_metrics": False,
        "configuration_changed_after_test": False,
        "published_config_sha256": args.published_config_sha256,
        "upstream_commit": args.upstream_commit,
        "bundled_model_sha256": args.model_sha256,
        "files": retained,
    }
    retention_text = json_text(public_retention)
    retention_sha256 = hashlib.sha256(retention_text.encode("utf-8")).hexdigest()
    provenance = {
        "status": "completed_tls2trees_published_default_test",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "training_mode": TRAINING_MODE,
        "run_id": args.run_id,
        "canonical_target": "leaf_on",
        "diagnostic_target": "leaf_off",
        "evaluation_evaluator": EVALUATOR,
        "evaluation_protocol": PROTOCOL,
        "inference_executed": True,
        "inference_execution_scope": "dedicated_published_default_held_out_test",
        "inference_rerun": False,
        "held_out_test_accessed": True,
        "configuration_selected_from_for_instance_metrics": False,
        "configuration_changed_after_test": False,
        "repeat_test_for_setting_selection_permitted": False,
        "benchmark_commit": args.benchmark_commit,
        "publication_benchmark_commit": args.publication_benchmark_commit,
        "upstream_commit": args.upstream_commit,
        "bundled_model_sha256": args.model_sha256,
        "workflow_config_sha256": args.workflow_config_sha256,
        "published_config_sha256": args.published_config_sha256,
        "benchmark_config_sha256": args.benchmark_config_sha256,
        "manifest_sha256": args.manifest_sha256,
        "published_default_summary_sha256": digest(summary_path),
        "plot_metrics_sha256": digest(plot_csv),
        "target_summary_sha256": digest(target_csv),
        "source_retention_manifest_sha256": digest(source_retention_path),
        "public_retention_manifest_sha256": retention_sha256,
        "retention_manifest_sha256": retention_sha256,
        "verified_prediction_files": len(retained),
        "public_result": overall["leaf_on"],
        "diagnostic_result": overall["leaf_off"],
    }
    headline = result_row(
        overall["leaf_on"],
        "completed_primary_result",
        HEADLINE_GROUP,
        "TLS2trees publication parameters evaluated without FOR-instance metric "
        "selection; leaf_on scores retained source-row predictions after "
        "excluding unlabelled class-3 out-points.",
    )
    diagnostic = result_row(
        overall["leaf_off"],
        "completed_target_specific_diagnostic",
        DIAGNOSTIC_GROUP,
        "TLS2trees leaf_off excludes foliage-labelled prediction points and is "
        "reported as a method-specific diagnostic.",
    )
    relative_retention = project_relative(public_paths["retention"], project_root)
    relative_evidence = project_relative(public_paths["leaf_on_plot"], project_root)
    retention_row = {
        "method_slug": "tls2trees",
        "variant": VARIANT,
        "retention_profile": HEADLINE_GROUP,
        "run_id": args.run_id,
        "evaluation_split": "test",
        "prediction_scope": "22_source_row_aligned_npz_files_leaf_off_and_leaf_on",
        "retained_file_count": len(retained),
        "retained_size_bytes": retained_size,
        "hash_status": "sha256_verified",
        "storage_status": "run_scoped_retained",
        "future_metrics_without_inference": "true",
        "prediction_root": "data/predictions/tls2trees/for_instance/published_default/test",
        "metrics_root": (
            "results/metadata/tls2trees/for_instance/published_default/test/"
            + args.run_id
        ),
        "retention_manifest": relative_retention,
        "retention_manifest_sha256": retention_sha256,
        "evidence_path": relative_evidence,
    }
    public_writes = {
        public_paths["leaf_on_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_on"]),
        public_paths["leaf_on_site"]: csv_text(SUMMARY_FIELDS, sites["leaf_on"]),
        public_paths["leaf_on_overall"]: csv_text(
            SUMMARY_FIELDS, [overall["leaf_on"]]
        ),
        public_paths["leaf_off_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_off"]),
        public_paths["leaf_off_site"]: csv_text(SUMMARY_FIELDS, sites["leaf_off"]),
        public_paths["leaf_off_overall"]: csv_text(
            SUMMARY_FIELDS, [overall["leaf_off"]]
        ),
        public_paths["retention"]: retention_text,
        public_paths["provenance"]: json_text(provenance),
    }
    reject_private_text(
        {
            **public_writes,
            Path("published_default_result_row"): csv_text(RESULT_FIELDS, [headline]),
            Path("published_default_diagnostic_row"): csv_text(
                RESULT_FIELDS, [diagnostic]
            ),
            Path("published_default_retention_row"): csv_text(
                RETENTION_FIELDS, [retention_row]
            ),
        },
        project_root,
    )
    writes = {
        **public_writes,
        args.results_csv: upsert_one(
            args.results_csv,
            RESULT_FIELDS,
            headline,
            {
                "method_slug": "tls2trees",
                "variant": VARIANT,
                "comparable_group": HEADLINE_GROUP,
            },
        ),
        args.diagnostics_csv: upsert_one(
            args.diagnostics_csv,
            RESULT_FIELDS,
            diagnostic,
            {
                "method_slug": "tls2trees",
                "variant": VARIANT,
                "comparable_group": DIAGNOSTIC_GROUP,
            },
        ),
        args.retention_registry: upsert_one(
            args.retention_registry,
            RETENTION_FIELDS,
            retention_row,
            {
                "method_slug": "tls2trees",
                "variant": VARIANT,
                "retention_profile": HEADLINE_GROUP,
            },
        ),
    }
    publish_writes(
        writes,
        project_root=project_root,
        enforce_head_baseline=bool(getattr(args, "validate_worktree", False)),
        expected_head=(
            args.publication_benchmark_commit
            if getattr(args, "validate_worktree", False)
            else None
        ),
    )
    return {
        "status": "tls2trees_published_default_results_finalised",
        "run_id": args.run_id,
        "verified_prediction_files": len(retained),
        "verified_prediction_size_bytes": retained_size,
        "public_retention_manifest_sha256": retention_sha256,
        "canonical_result": headline,
        "diagnostic_result": diagnostic,
        "written_files": [project_relative(path, project_root) for path in writes],
    }


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, render, publish, and receipt one locked public transaction."""

    project_root = args.project_root.expanduser().resolve()
    with publication_lock(project_root):
        preflight_text_target(
            args.receipt_json,
            project_root=project_root,
            staging_suffix=".tls2trees-published-default-receipt.tmp",
        )
        if getattr(args, "validate_worktree", False):
            public_paths = (
                "methods/tls2trees/examples/"
                "tls2trees_published_default_test_plot_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_test_site_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_test_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_leaf_off_test_plot_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_leaf_off_test_site_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_leaf_off_test_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_prediction_retention_manifest.json",
                "methods/tls2trees/examples/"
                "tls2trees_published_default_test_provenance.json",
                "outputs/for_instance_benchmark_metrics/"
                "for_instance_method_benchmark_results.csv",
                "outputs/for_instance_benchmark_metrics/"
                "for_instance_method_development_diagnostics.csv",
                "outputs/for_instance_benchmark_metrics/"
                "for_instance_prediction_retention_registry.csv",
            )
            recovery_paths = set(public_paths)
            for path in public_paths:
                parent, name = path.rsplit("/", 1)
                recovery_paths.add(
                    f"{parent}/.{name}.tls2trees-published-default-finalisation.tmp"
                )
            validate_git_worktree(
                project_root,
                recovery_confirmed=bool(args.recovery_confirmed),
                recovery_paths=recovery_paths,
                expected_head=args.publication_benchmark_commit,
            )
        payload = _finalise_locked(args)
        publish_text_bundle(
            {args.receipt_json: json_text(payload)},
            project_root=project_root,
            staging_suffix=".tls2trees-published-default-receipt.tmp",
            replace=os.replace,
            expected_head=(
                args.publication_benchmark_commit
                if getattr(args, "validate_worktree", False)
                else None
            ),
        )
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--publication-benchmark-commit", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--workflow-config", type=Path, required=True)
    parser.add_argument("--workflow-config-sha256", required=True)
    parser.add_argument("--published-config", type=Path, required=True)
    parser.add_argument("--published-config-sha256", required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--benchmark-config-sha256", required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--plot-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--source-retention-json", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--retention-registry", type=Path, required=True)
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--recovery-confirmed", type=int, choices=(0, 1), default=0)
    parser.set_defaults(validate_worktree=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = finalise(args)
    result = payload["canonical_result"]
    print(f"status={payload['status']}")
    print(f"run_id={payload['run_id']}")
    print(f"verified_prediction_files={payload['verified_prediction_files']}")
    print(
        "leaf_on "
        f"predictions={result['predicted_instances']} "
        f"references={result['reference_instances']} "
        f"tp={result['true_positives']} fp={result['false_positives']} "
        f"fn={result['false_negatives']} "
        f"micro_f1={float(result['micro_f1']):.6f}"
    )
    print("repository_result_files_written=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
