"""Finalise retained TLS2trees FOR-instance predictions and public results.

This command is intentionally post-inference.  It validates the completed,
frozen held-out summary, inventories the 22 source-row-aligned prediction
files, and writes public-safe result evidence plus the repository registries.
It never runs inference or changes the selected TLS2trees configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TARGETS = ("leaf_off", "leaf_on")
CANONICAL_TARGET = "leaf_on"
EXPECTED_PLOTS = 11
EXPECTED_REFERENCES = 323
PROTOCOL = "for_instance_pointwise_class3_ignore"
MATCHING = "maximum_cardinality_one_to_one"
MASK = "classification_3_excluded_then_union_of_reference_tree_and_predicted_tree_points"
TLS_EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
TLS_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
IOU_THRESHOLD = 0.5
VARIANT = "development_tuned"
TRAINING_MODE = "external_training_only"
UPSTREAM_TLS2TREES_COMMIT = "ca12cb73b2c736d80b020e8025f8d975d42e6f01"
MODEL_SHA256 = "1a8bb6372394600f7c4b15f76beb98c32cb47ed25f8f729a84117ccfa410e72b"

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(
        handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Runtime artefact is outside the project root: {path}") from exc


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def read_csv(path: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"Unexpected CSV schema in {path}")
        return list(reader)


def upsert_unique(
    path: Path,
    fieldnames: list[str],
    new_rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> str:
    rows = read_csv(path, fieldnames)
    requested: set[tuple[str, ...]] = set()
    for row in new_rows:
        key = tuple(str(row[field]) for field in key_fields)
        if key in requested:
            raise ValueError(f"Upsert request contains duplicate key {key}")
        requested.add(key)
        matches = [
            index
            for index, existing in enumerate(rows)
            if tuple(existing[field] for field in key_fields) == key
        ]
        if len(matches) > 1:
            raise ValueError(f"Result registry contains duplicate key {key}")
        normalized = {field: row[field] for field in fieldnames}
        if matches:
            rows[matches[0]] = normalized
        else:
            rows.append(normalized)
    return csv_text(fieldnames, rows)


def division(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate(rows: list[dict[str, Any]], site: str) -> dict[str, Any]:
    predicted = sum(int(row["predicted_instances"]) for row in rows)
    references = sum(int(row["reference_instances"]) for row in rows)
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = division(tp, tp + fp)
    recall = division(tp, tp + fn)
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
        "mean_plot_precision": sum(float(row["precision"]) for row in rows) / len(rows),
        "mean_plot_recall": sum(float(row["recall"]) for row in rows) / len(rows),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": division(2 * tp, 2 * tp + fp + fn),
        "evaluation_protocol": PROTOCOL,
        "matching_policy": MATCHING,
        "evaluation_mask": MASK,
        "held_out_test_accessed": "true",
        "retention_status": "retention_verified",
        "result_status": "completed_aligned_pointwise_test",
    }


def validate_summary(
    summary: dict[str, Any], run_id: str, final_selection_sha256: str
) -> None:
    expected = {
        "status": "held_out_test_completed",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "split": "test",
        "workflow_run_id": run_id,
        "final_selection_sha256": final_selection_sha256,
        "expected_plot_count": EXPECTED_PLOTS,
        "expected_metric_count": EXPECTED_PLOTS * len(TARGETS),
        "valid_metric_count": EXPECTED_PLOTS * len(TARGETS),
        "held_out_test_accessed": True,
        "held_out_accuracy_metrics_computed": True,
        "configuration_changed_after_test": False,
        "evaluator": TLS_EVALUATOR,
        "evaluation_protocol": PROTOCOL,
        "evaluation_mask": TLS_MASK,
        "inference_rerun": False,
        "prediction_adapter_rerun": False,
        "retained_sources_unchanged": True,
        "test_metrics_used_for_configuration_selection": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Held-out summary has unexpected {field}")
    if summary.get("incomplete_tasks") != []:
        raise ValueError("Held-out summary still contains incomplete tasks")


def collect(
    *, summary: dict[str, Any], manifest: dict[str, Any], project_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    manifest_plots = {int(row["task_index"]): row for row in manifest["plots"]}
    if set(manifest_plots) != set(range(EXPECTED_PLOTS)):
        raise ValueError("Manifest does not contain the exact 11 test task indices")
    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in summary["plot_metrics"]:
        target = str(source["target"])
        task_index = int(source["task_index"])
        if target not in TARGETS or (target, task_index) in seen:
            raise ValueError("Held-out summary contains duplicate or unknown target tasks")
        seen.add((target, task_index))
        plot = manifest_plots[task_index]
        if (
            source.get("status") != "evaluated"
            or source.get("safe_for_scoring") is not True
            or source.get("relative_path") != plot["relative_path"]
            or source.get("safe_plot_id") != plot["safe_plot_id"]
            or source.get("collection") != plot["collection"]
        ):
            raise ValueError(f"Invalid plot summary provenance for {target}:{task_index}")
        metrics_path = Path(source["metrics_path"]).expanduser().resolve()
        if not metrics_path.is_file() or sha256(metrics_path) != source["metrics_sha256"]:
            raise ValueError(f"Metric file missing or changed: {metrics_path}")
        metrics = load_object(metrics_path)
        if (
            metrics.get("evaluator") != TLS_EVALUATOR
            or metrics.get("matching_policy") != MATCHING
            or metrics.get("evaluation_mask") != TLS_MASK
            or float(metrics.get("iou_threshold", -1)) != IOU_THRESHOLD
            or metrics.get("semantic_ignore", {}).get("ignored_semantic_classes")
            != [3]
        ):
            raise ValueError(f"Metric contract changed: {metrics_path}")
        direct_aligned = source.get("aligned_predictions_npz")
        direct_metadata = source.get("alignment_metadata_json")
        if bool(direct_aligned) != bool(direct_metadata):
            raise ValueError(
                f"Final summary has incomplete retained paths: {target}:{task_index}"
            )
        if direct_aligned:
            aligned = Path(str(direct_aligned)).expanduser().resolve()
            metadata_path = Path(str(direct_metadata)).expanduser().resolve()
        else:
            # Backward-compatible route for a native held-out summary whose metric
            # still lives beneath the original per-plot evaluation directory.
            plot_root = metrics_path.parents[2]
            aligned = (
                plot_root
                / "predictions"
                / "aligned"
                / target
                / "source_row_predictions.npz"
            )
            metadata_path = aligned.with_name("alignment_metadata.json")
        if not aligned.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Aligned prediction is incomplete: {aligned}")
        aligned_sha = sha256(aligned)
        metadata_sha = sha256(metadata_path)
        if (
            source.get("aligned_predictions_npz_sha256", aligned_sha) != aligned_sha
            or source.get("alignment_metadata_sha256", metadata_sha) != metadata_sha
        ):
            raise ValueError(
                f"Final summary retained-path checksum changed: {target}:{task_index}"
            )
        metadata = load_object(metadata_path)
        if (
            metadata.get("status") != "passed"
            or metadata.get("target") != target
            or metadata.get("aligned_prediction_npz_sha256") != aligned_sha
            or int(metadata.get("source_row_count", -1)) != int(plot["point_count"])
            or int(metadata.get("prediction_instance_count", -1))
            != int(source["raw_prediction_instance_count"])
        ):
            raise ValueError(f"Alignment metadata does not match: {metadata_path}")
        with np.load(aligned, allow_pickle=False) as arrays:
            required = {"source_row_index", "predicted_instance_id", "prediction_names"}
            if not required <= set(arrays.files):
                raise ValueError(f"Aligned prediction has missing arrays: {aligned}")
            source_rows = np.asarray(arrays["source_row_index"])
            predicted_ids = np.asarray(arrays["predicted_instance_id"])
            names = np.asarray(arrays["prediction_names"])
            point_count = int(plot["point_count"])
            if (
                len(source_rows) != point_count
                or len(predicted_ids) != point_count
                or len(names) != int(source["raw_prediction_instance_count"])
                or not np.array_equal(source_rows, np.arange(point_count))
                or np.any(predicted_ids < 0)
            ):
                raise ValueError(f"Aligned source-row prediction is invalid: {aligned}")
        row = {
            "method_slug": "tls2trees",
            "variant": VARIANT,
            "training_mode": TRAINING_MODE,
            "target": target,
            "run_id": summary["workflow_run_id"],
            "dataset_split": "test",
            "plot_index": task_index,
            "relative_path": plot["relative_path"],
            "collection": plot["collection"],
            "point_count": int(plot["point_count"]),
            "predicted_instances": int(source["prediction_instance_count"]),
            "reference_instances": int(source["reference_instance_count"]),
            "true_positives": int(source["true_positives"]),
            "false_positives": int(source["false_positives"]),
            "false_negatives": int(source["false_negatives"]),
            "precision": float(source["precision"]),
            "recall": float(source["recall"]),
            "f1": float(source["f1"]),
            "mean_matched_iou": source["mean_matched_iou"],
            "evaluation_protocol": PROTOCOL,
            "matching_policy": MATCHING,
            "evaluation_mask": MASK,
            "iou_threshold": IOU_THRESHOLD,
            "source_metrics_sha256": source["metrics_sha256"],
            "aligned_prediction_sha256": aligned_sha,
        }
        rows_by_target[target].append(row)
        retained.append(
            {
                "target": target,
                "candidate_id": source["candidate_id"],
                "plot_index": task_index,
                "plot_id": plot["safe_plot_id"],
                "relative_path": relative_to_project(aligned, project_root),
                "size_bytes": aligned.stat().st_size,
                "sha256": aligned_sha,
                "format": "npz",
                "point_correspondence": "source_row_index",
            }
        )
    if seen != {(target, index) for target in TARGETS for index in range(EXPECTED_PLOTS)}:
        raise ValueError("Held-out summary does not contain all 22 target/plot results")
    for target in TARGETS:
        rows_by_target[target].sort(key=lambda row: int(row["plot_index"]))
        overall = aggregate(rows_by_target[target], "ALL")
        recorded = next(row for row in summary["aggregates"] if row["target"] == target)
        for field, recorded_field in (
            ("predicted_instances", "prediction_instance_count"),
            ("reference_instances", "reference_instance_count"),
            ("true_positives", "true_positives"),
            ("false_positives", "false_positives"),
            ("false_negatives", "false_negatives"),
        ):
            if overall[field] != int(recorded[recorded_field]):
                raise ValueError(f"Aggregate count mismatch for {target}:{field}")
        if overall["reference_instances"] != EXPECTED_REFERENCES:
            raise ValueError(f"Unexpected reference inventory for {target}")
    retained.sort(key=lambda row: (row["target"], int(row["plot_index"])))
    return dict(rows_by_target), retained


def registry_row(
    overall: dict[str, Any], result_role: str, group: str, note: str
) -> dict[str, Any]:
    return {
        "dataset_slug": "for-instance",
        "method_slug": "tls2trees",
        "variant": VARIANT,
        "run_id": overall["run_id"],
        "training_mode": TRAINING_MODE,
        "result_role": result_role,
        "evaluation_protocol": PROTOCOL,
        "matching_policy": MATCHING,
        "evaluation_mask": MASK,
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


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.expanduser().resolve()
    summary_path = args.summary_json.expanduser().resolve()
    manifest_path = args.manifest_json.expanduser().resolve()
    final_selection_path = args.final_selection_json.expanduser().resolve()
    if sha256(final_selection_path) != args.final_selection_sha256:
        raise ValueError("Frozen final-selection checksum changed")
    summary = load_object(summary_path)
    manifest = load_object(manifest_path)
    validate_summary(summary, args.run_id, args.final_selection_sha256)
    if sha256(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("Held-out manifest checksum changed")
    if manifest.get("dataset_split") != "test":
        raise ValueError("Finalisation requires the held-out test manifest")
    rows_by_target, retained = collect(
        summary=summary, manifest=manifest, project_root=project_root
    )
    examples_dir = args.examples_dir
    prefix = "tls2trees_development_tuned"
    paths = {
        "leaf_on_plot": examples_dir / f"{prefix}_test_plot_results.csv",
        "leaf_on_site": examples_dir / f"{prefix}_test_site_results.csv",
        "leaf_on_overall": examples_dir / f"{prefix}_test_results.csv",
        "leaf_off_plot": examples_dir
        / f"{prefix}_leaf_off_test_plot_diagnostic.csv",
        "leaf_off_site": examples_dir
        / f"{prefix}_leaf_off_test_site_diagnostic.csv",
        "leaf_off_overall": examples_dir
        / f"{prefix}_leaf_off_test_diagnostic.csv",
        "retention": examples_dir
        / f"{prefix}_prediction_retention_manifest.json",
        "provenance": examples_dir / f"{prefix}_test_provenance.json",
    }
    if not paths["retention"].is_file():
        raise FileNotFoundError(
            "The neutral retained-prediction manifest must exist before finalisation"
        )
    site_rows: dict[str, list[dict[str, Any]]] = {}
    overall_rows: dict[str, dict[str, Any]] = {}
    for target, plot_rows in rows_by_target.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in plot_rows:
            groups[str(row["collection"])].append(row)
        site_rows[target] = [aggregate(groups[site], site) for site in sorted(groups)]
        overall_rows[target] = aggregate(plot_rows, "ALL")
    retention_payload = load_object(paths["retention"])
    expected_retention = {
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
        "verified_prediction_size_bytes": sum(
            int(row["size_bytes"]) for row in retained
        ),
        "hash_algorithm": "sha256",
        "prediction_contract": "one_instance_label_per_source_row",
        "future_metrics_without_inference": True,
        "held_out_test_accessed": True,
        "configuration_changed_after_test": False,
        "files": retained,
    }
    for field, value in expected_retention.items():
        if retention_payload.get(field) != value:
            raise ValueError(f"Retained-prediction manifest has unexpected {field}")
    retention_sha = sha256(paths["retention"])
    summary_sha = sha256(summary_path)
    provenance = {
        "schema_version": 1,
        "status": "completed_tls2trees_held_out_test",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "training_mode": TRAINING_MODE,
        "run_id": args.run_id,
        "canonical_target": CANONICAL_TARGET,
        "diagnostic_target": "leaf_off",
        "held_out_test_accessed": True,
        "configuration_changed_after_test": False,
        "repeat_test_for_setting_selection_permitted": False,
        "benchmark_commit": args.benchmark_commit,
        "evaluation_benchmark_commit": args.evaluation_benchmark_commit,
        "evaluation_run_id": summary.get("evaluation_run_id"),
        "evaluation_plan_sha256": summary.get("evaluation_plan_sha256"),
        "evaluation_evaluator": summary.get("evaluator"),
        "evaluation_summary_sha256": summary_sha,
        "inference_rerun": False,
        "upstream_commit": UPSTREAM_TLS2TREES_COMMIT,
        "model_sha256": MODEL_SHA256,
        "final_selection_sha256": args.final_selection_sha256,
        "manifest_sha256": sha256(manifest_path),
        "retention_manifest_sha256": retention_sha,
        "verified_prediction_files": len(retained),
        "public_result": overall_rows[CANONICAL_TARGET],
        "diagnostic_result": overall_rows["leaf_off"],
    }
    relative_retention = relative_to_project(paths["retention"], project_root)
    relative_evidence = relative_to_project(paths["leaf_on_plot"], project_root)
    metrics_root = relative_to_project(summary_path.parent, project_root)
    prediction_root = "data/predictions/tls2trees/for_instance/development_tuned/test"
    retention_row = {
        "method_slug": "tls2trees",
        "variant": VARIANT,
        "retention_profile": "held_out_test_tls2trees_class3_ignore",
        "run_id": args.run_id,
        "evaluation_split": "test",
        "prediction_scope": "22_source_row_aligned_npz_files_leaf_off_and_leaf_on",
        "retained_file_count": len(retained),
        "retained_size_bytes": retention_payload["verified_prediction_size_bytes"],
        "hash_status": "sha256_verified",
        "storage_status": "barkla_run_scoped_retained",
        "future_metrics_without_inference": "true",
        "prediction_root": prediction_root,
        "metrics_root": metrics_root,
        "retention_manifest": relative_retention,
        "retention_manifest_sha256": retention_sha,
        "evidence_path": relative_evidence,
    }
    headline = registry_row(
        overall_rows["leaf_on"],
        "completed_aligned_pointwise_result",
        "held_out_test_tls2trees_class3_ignore",
        "Frozen development-selected TLS2trees parameters; leaf_on evaluates "
        "all FOR-instance tree classes through retained source-row predictions "
        "after excluding unlabelled class-3 out-points from evaluation.",
    )
    diagnostic = registry_row(
        overall_rows["leaf_off"],
        "completed_target_specific_diagnostic",
        "tls2trees_leaf_off_class3_ignore_diagnostic",
        "TLS2trees leaf_off target excludes class-5 foliage points and is not "
        "a shared-protocol headline result.",
    )
    writes = {
        paths["leaf_on_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_on"]),
        paths["leaf_on_site"]: csv_text(SUMMARY_FIELDS, site_rows["leaf_on"]),
        paths["leaf_on_overall"]: csv_text(SUMMARY_FIELDS, [overall_rows["leaf_on"]]),
        paths["leaf_off_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_off"]),
        paths["leaf_off_site"]: csv_text(SUMMARY_FIELDS, site_rows["leaf_off"]),
        paths["leaf_off_overall"]: csv_text(SUMMARY_FIELDS, [overall_rows["leaf_off"]]),
        paths["provenance"]: json_text(provenance),
        args.results_csv: upsert_unique(
            args.results_csv,
            RESULT_FIELDS,
            [headline],
            ("method_slug", "variant", "run_id", "comparable_group"),
        ),
        args.diagnostics_csv: upsert_unique(
            args.diagnostics_csv,
            RESULT_FIELDS,
            [diagnostic],
            ("method_slug", "variant", "run_id", "comparable_group"),
        ),
        args.retention_registry: upsert_unique(
            args.retention_registry,
            RETENTION_FIELDS,
            [retention_row],
            ("method_slug", "retention_profile", "run_id"),
        ),
    }
    for path, text in writes.items():
        atomic_write(path, text)
    return {
        "status": "tls2trees_results_finalised",
        "run_id": args.run_id,
        "canonical_target": CANONICAL_TARGET,
        "diagnostic_target": "leaf_off",
        "verified_prediction_files": len(retained),
        "verified_prediction_size_bytes": retention_payload["verified_prediction_size_bytes"],
        "retention_manifest_sha256": retention_sha,
        "canonical_result": headline,
        "diagnostic_result": diagnostic,
        "written_files": [relative_to_project(path, project_root) for path in writes],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--final-selection-json", type=Path, required=True)
    parser.add_argument("--final-selection-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--evaluation-benchmark-commit", required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--retention-registry", type=Path, required=True)
    parser.add_argument("--receipt-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = finalise(args)
    atomic_write(args.receipt_json, json_text(payload))
    result = payload["canonical_result"]
    print(f"status={payload['status']}")
    print(f"run_id={payload['run_id']}")
    print(f"verified_prediction_files={payload['verified_prediction_files']}")
    print(f"verified_prediction_size_bytes={payload['verified_prediction_size_bytes']}")
    print(f"retention_manifest_sha256={payload['retention_manifest_sha256']}")
    print(
        "leaf_on "
        f"predictions={result['predicted_instances']} references={result['reference_instances']} "
        f"tp={result['true_positives']} fp={result['false_positives']} "
        f"fn={result['false_negatives']} micro_f1={float(result['micro_f1']):.6f}"
    )
    print("repository_result_files_written=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
