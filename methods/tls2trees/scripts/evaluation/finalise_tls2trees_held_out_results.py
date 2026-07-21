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
import math
import os
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np


EVALUATION = Path(__file__).resolve().parent
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from tls2trees_publication import (  # noqa: E402
    preflight_text_target,
    publication_lock,
    publish_text_bundle,
    validate_git_worktree,
)


TARGETS = ("leaf_off", "leaf_on")
CANONICAL_TARGET = "leaf_on"
EXPECTED_PLOTS = 11
EXPECTED_REFERENCES = 323
EXPECTED_EVALUATION_TASKS = 106
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
        staging_suffix=".tls2trees-held-out-finalisation.tmp",
        replace=os.replace,
        enforce_head_baseline=enforce_head_baseline,
        expected_head=expected_head,
    )


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


def exact_number(value: Any, *, field: str, integer: bool) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric evidence for {field}")
    if integer:
        if not isinstance(value, int):
            raise ValueError(f"Non-integral evidence for {field}")
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite evidence for {field}")
    return number


def reconcile_plot_metric(
    *, source: dict[str, Any], metrics: dict[str, Any], target: str, task_index: int
) -> None:
    """Require the summary copy to equal the exact hashed metric evidence."""

    context = f"{target}:{task_index}"
    for field in ("status", "safe_for_scoring"):
        if source.get(field) != metrics.get(field):
            raise ValueError(f"Plot metric evidence mismatch for {context}:{field}")
    for field in (
        "prediction_instance_count",
        "reference_instance_count",
        "true_positives",
        "false_positives",
        "false_negatives",
    ):
        summary_value = exact_number(source.get(field), field=field, integer=True)
        metric_value = exact_number(metrics.get(field), field=field, integer=True)
        if summary_value != metric_value:
            raise ValueError(f"Plot metric evidence mismatch for {context}:{field}")
    for field in ("precision", "recall", "f1", "mean_matched_iou"):
        summary_value = exact_number(source.get(field), field=field, integer=False)
        metric_value = exact_number(metrics.get(field), field=field, integer=False)
        if summary_value != metric_value:
            raise ValueError(f"Plot metric evidence mismatch for {context}:{field}")
    summary_raw = exact_number(
        source.get("raw_prediction_instance_count"),
        field="raw_prediction_instance_count",
        integer=True,
    )
    metric_raw = exact_number(
        metrics.get("semantic_ignore", {}).get("raw_prediction_instance_count"),
        field="raw_prediction_instance_count",
        integer=True,
    )
    if summary_raw != metric_raw:
        raise ValueError(
            f"Plot metric evidence mismatch for {context}:raw_prediction_instance_count"
        )


def reconcile_aggregate(
    *, overall: dict[str, Any], recorded: dict[str, Any], target: str
) -> None:
    mappings = (
        ("predicted_instances", "prediction_instance_count", True),
        ("reference_instances", "reference_instance_count", True),
        ("true_positives", "true_positives", True),
        ("false_positives", "false_positives", True),
        ("false_negatives", "false_negatives", True),
        ("micro_precision", "precision", False),
        ("micro_recall", "recall", False),
        ("micro_f1", "micro_f1", False),
        ("mean_plot_f1", "mean_plot_f1", False),
    )
    for public_field, source_field, integer in mappings:
        calculated = exact_number(
            overall.get(public_field), field=public_field, integer=integer
        )
        source_value = exact_number(
            recorded.get(source_field), field=source_field, integer=integer
        )
        if integer:
            equal = calculated == source_value
        else:
            equal = math.isclose(
                float(calculated), float(source_value), rel_tol=1e-12, abs_tol=1e-15
            )
        if not equal:
            raise ValueError(f"Aggregate metric evidence mismatch for {target}:{public_field}")


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
        "schema_version": 2,
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
    if not isinstance(summary.get("evaluation_run_id"), str) or not summary[
        "evaluation_run_id"
    ]:
        raise ValueError("Held-out summary has no evaluation run ID")
    plan_sha = summary.get("evaluation_plan_sha256")
    if (
        not isinstance(plan_sha, str)
        or len(plan_sha) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha)
    ):
        raise ValueError("Held-out summary has invalid evaluation plan SHA-256")


def validate_final_selection(
    selection: dict[str, Any],
) -> dict[str, tuple[str, int]]:
    """Return the exact reviewed candidate identity for each held-out target."""

    expected = {
        "schema_version": 1,
        "status": "development_tuned_configuration_frozen",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "selection_split": "development",
        "development_metric_count": 84,
        "development_plot_count": 21,
        "development_accuracy_metrics_used": True,
        "held_out_test_accessed": False,
        "held_out_test_runnable": False,
        "final_configuration_selected": True,
        "review_required_before_held_out_test": True,
    }
    for field, value in expected.items():
        if selection.get(field) != value:
            raise ValueError(f"Frozen final selection has unexpected {field}")
    benchmark_commit = selection.get("benchmark_commit")
    if (
        not isinstance(benchmark_commit, str)
        or len(benchmark_commit) != 40
        or any(character not in "0123456789abcdef" for character in benchmark_commit)
    ):
        raise ValueError("Frozen final selection has invalid benchmark_commit")
    selected = selection.get("selected_by_target")
    if not isinstance(selected, dict) or set(selected) != set(TARGETS):
        raise ValueError("Frozen final selection does not contain both exact targets")
    identities: dict[str, tuple[str, int]] = {}
    for target in TARGETS:
        record = selected[target]
        if not isinstance(record, dict):
            raise ValueError(f"Frozen final selection has invalid {target} record")
        candidate_id = record.get("candidate_id")
        candidate_index = record.get("stage1_candidate_index")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or candidate_index < 0
            or not isinstance(record.get("parameters"), dict)
            or not isinstance(record.get("development_metrics"), dict)
            or not isinstance(record.get("selection_reason"), str)
            or not record["selection_reason"]
        ):
            raise ValueError(f"Frozen final selection has invalid {target} identity")
        identities[target] = (candidate_id, candidate_index)
    return identities


def validate_evaluation_plan(
    *,
    summary: dict[str, Any],
    project_root: Path,
    evaluation_benchmark_commit: str,
    expected_evaluation_run_id: str,
    expected_plan_path: Path,
    expected_plan_sha256: str,
    expected_evaluator_path: Path,
    expected_evaluator_sha256: str,
    expected_manifest_path: Path,
    expected_manifest_sha256: str,
    expected_final_selection_path: Path,
    expected_final_selection_sha256: str,
    selected_by_target: dict[str, tuple[str, int]],
) -> tuple[Path, str, dict[tuple[str, int], dict[str, Any]]]:
    """Bind the held-out summary to its SHA-verified historical plan."""

    plan_value = summary.get("evaluation_plan")
    if not isinstance(plan_value, str) or not plan_value:
        raise ValueError("Held-out summary has no evaluation plan path")
    plan_path = Path(plan_value).expanduser().resolve()
    plan_sha256 = str(summary["evaluation_plan_sha256"])
    expected_plan_path = expected_plan_path.expanduser().resolve()
    expected_evaluator_path = expected_evaluator_path.expanduser().resolve()
    if summary.get("evaluation_run_id") != expected_evaluation_run_id:
        raise ValueError("Held-out summary evaluation run ID changed from frozen state")
    if plan_path != expected_plan_path:
        raise ValueError("Held-out summary evaluation plan path changed from frozen state")
    if plan_sha256 != expected_plan_sha256:
        raise ValueError("Held-out summary evaluation plan SHA-256 changed from frozen state")
    if not plan_path.is_file() or sha256(plan_path) != plan_sha256:
        raise ValueError("Frozen evaluation plan is missing or its SHA-256 changed")
    plan = load_object(plan_path)

    evaluator_value = plan.get("evaluator_path")
    if not isinstance(evaluator_value, str) or not evaluator_value:
        raise ValueError("Frozen evaluation plan has no evaluator path")
    evaluator_path_value = Path(evaluator_value).expanduser()
    evaluator_path = (
        evaluator_path_value
        if evaluator_path_value.is_absolute()
        else project_root / evaluator_path_value
    ).resolve()
    canonical_evaluator_path = (
        project_root
        / "methods/tls2trees/scripts/evaluation/"
        "evaluate_for_instance_tls2trees_plot.py"
    ).resolve()
    if evaluator_path != canonical_evaluator_path:
        raise ValueError("Frozen evaluation plan does not select the canonical evaluator")
    evaluator_sha256 = plan.get("evaluator_sha256")
    if (
        not isinstance(evaluator_sha256, str)
        or len(evaluator_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evaluator_sha256)
    ):
        raise ValueError("Frozen evaluation plan has invalid evaluator_sha256")
    if evaluator_path != expected_evaluator_path:
        raise ValueError("Frozen evaluator path changed from evaluation state")
    if evaluator_sha256 != expected_evaluator_sha256:
        raise ValueError("Frozen evaluator SHA-256 changed from evaluation state")

    expected = {
        "schema_version": 1,
        "status": "retained_prediction_evaluation_plan_frozen",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": VARIANT,
        "evaluation_run_id": expected_evaluation_run_id,
        "benchmark_commit": evaluation_benchmark_commit,
        "evaluator": TLS_EVALUATOR,
        "evaluation_protocol": PROTOCOL,
        "evaluation_mask": TLS_MASK,
        "iou_threshold": IOU_THRESHOLD,
        "matching_policy": MATCHING,
        "inference_rerun": False,
        "prediction_adapter_rerun": False,
        "configuration_changed": False,
        "configuration_selection_performed": False,
        "expected_task_count": EXPECTED_EVALUATION_TASKS,
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"Frozen evaluation plan has unexpected {field}")
    development_source = plan.get("development")
    test_source = plan.get("test")
    if not isinstance(development_source, dict) or not isinstance(test_source, dict):
        raise ValueError("Frozen evaluation plan has invalid source blocks")
    for field in ("workflow_run_id", "manifest", "selection"):
        if not isinstance(development_source.get(field), str) or not development_source[
            field
        ]:
            raise ValueError(f"Frozen development source has invalid {field}")
    for field in ("manifest_sha256", "selection_sha256"):
        _required_task_sha256(development_source, field)
    if development_source.get("held_out_test_accessed") is not False:
        raise ValueError("Frozen development source accessed the held-out test")

    expected_manifest_path = expected_manifest_path.expanduser().resolve()
    expected_final_selection_path = expected_final_selection_path.expanduser().resolve()
    test_expected = {
        "workflow_run_id": summary["workflow_run_id"],
        "manifest_sha256": expected_manifest_sha256,
        "final_selection_sha256": expected_final_selection_sha256,
        "held_out_test_already_accessed": True,
        "test_metrics_used_for_configuration_selection": False,
    }
    for field, value in test_expected.items():
        if test_source.get(field) != value:
            raise ValueError(f"Frozen held-out source has unexpected {field}")
    for field, expected_path in (
        ("manifest", expected_manifest_path),
        ("final_selection", expected_final_selection_path),
    ):
        value = test_source.get(field)
        if (
            not isinstance(value, str)
            or Path(value).expanduser().resolve() != expected_path
        ):
            raise ValueError(f"Frozen held-out source has unexpected {field}")
    for field in ("retention_manifest", "retention_manifest_sha256"):
        if field.endswith("sha256"):
            _required_task_sha256(test_source, field)
        elif not isinstance(test_source.get(field), str) or not test_source[field]:
            raise ValueError(f"Frozen held-out source has invalid {field}")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != EXPECTED_EVALUATION_TASKS:
        raise ValueError("Frozen evaluation plan does not contain all 106 tasks")
    test_tasks = validate_evaluation_tasks(
        tasks, selected_by_target=selected_by_target
    )
    return plan_path, evaluator_sha256, test_tasks


def _required_task_path(task: dict[str, Any], field: str) -> Path:
    value = task.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Frozen evaluation task has invalid {field}")
    return Path(value).expanduser().resolve()


def _required_task_sha256(task: dict[str, Any], field: str) -> str:
    value = task.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Frozen evaluation task has invalid {field}")
    return value


def validate_evaluation_tasks(
    tasks: list[Any],
    *,
    selected_by_target: dict[str, tuple[str, int]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate the full plan schema and return its exact held-out task map."""

    path_fields = (
        "input_las",
        "aligned_predictions_npz",
        "alignment_metadata_json",
        "output_root",
        "output_metrics_json",
        "output_matches_csv",
        "output_unmatched_predictions_csv",
        "output_unmatched_references_csv",
    )
    sha_fields = (
        "aligned_predictions_npz_sha256",
        "alignment_metadata_sha256",
        "source_las_sha256",
    )
    identity_fields = (
        "source_workflow_run_id",
        "source_candidate_run_id",
        "candidate_id",
        "collection",
        "safe_plot_id",
        "relative_path",
    )
    seen_outputs: set[Path] = set()
    coverage: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    test_tasks: dict[tuple[str, int], dict[str, Any]] = {}
    development_candidates: set[str] = set()

    for position, raw_task in enumerate(tasks):
        if not isinstance(raw_task, dict):
            raise ValueError("Frozen evaluation task is not an object")
        task = raw_task
        evaluation_task_index = task.get("evaluation_task_index")
        if (
            not isinstance(evaluation_task_index, int)
            or isinstance(evaluation_task_index, bool)
            or evaluation_task_index != position
        ):
            raise ValueError("Frozen evaluation task indices are not exact and unique")
        split = task.get("split")
        target = task.get("target")
        if split not in {"development", "test"} or target not in TARGETS:
            raise ValueError("Frozen evaluation task has invalid split or target")
        for field in identity_fields:
            if not isinstance(task.get(field), str) or not task[field]:
                raise ValueError(f"Frozen evaluation task has invalid {field}")
        candidate_index = task.get("candidate_index")
        manifest_task_index = task.get("manifest_task_index")
        if (
            not isinstance(candidate_index, int)
            or isinstance(candidate_index, bool)
            or candidate_index < 0
            or not isinstance(manifest_task_index, int)
            or isinstance(manifest_task_index, bool)
            or manifest_task_index < 0
        ):
            raise ValueError("Frozen evaluation task has invalid integer identity")
        paths = {field: _required_task_path(task, field) for field in path_fields}
        for field in sha_fields:
            _required_task_sha256(task, field)
        output_root = paths["output_root"]
        output_paths = tuple(paths[field] for field in path_fields[4:])
        if any(path.parent != output_root for path in output_paths):
            raise ValueError("Frozen evaluation task output paths escape output_root")
        if any(path in seen_outputs for path in output_paths):
            raise ValueError("Frozen evaluation task output paths are not unique")
        seen_outputs.update(output_paths)
        if paths["aligned_predictions_npz"].name != "source_row_predictions.npz":
            raise ValueError("Frozen evaluation task has unexpected aligned filename")
        if paths["alignment_metadata_json"].name != "alignment_metadata.json":
            raise ValueError("Frozen evaluation task has unexpected alignment filename")
        expected_output_names = {
            "output_metrics_json": "plot_metrics.json",
            "output_matches_csv": "matches.csv",
            "output_unmatched_predictions_csv": "unmatched_predictions.csv",
            "output_unmatched_references_csv": "unmatched_references.csv",
        }
        if any(paths[field].name != name for field, name in expected_output_names.items()):
            raise ValueError("Frozen evaluation task has unexpected output filename")

        route = (str(split), str(task["candidate_id"]), str(target))
        if manifest_task_index in coverage[route]:
            raise ValueError("Frozen evaluation plan contains a duplicate route task")
        coverage[route].add(manifest_task_index)
        if split == "development":
            development_candidates.add(str(task["candidate_id"]))
        else:
            selected_candidate_id, selected_candidate_index = selected_by_target[
                str(target)
            ]
            if (
                task["candidate_id"] != selected_candidate_id
                or candidate_index != selected_candidate_index
            ):
                raise ValueError(
                    "Frozen held-out task differs from the reviewed final selection"
                )
            key = (str(target), manifest_task_index)
            if key in test_tasks:
                raise ValueError("Frozen evaluation plan contains duplicate held-out tasks")
            test_tasks[key] = task

    if len(development_candidates) != 2:
        raise ValueError("Frozen evaluation plan does not contain two development candidates")
    expected_development_indices = set(range(21))
    expected_test_indices = set(range(EXPECTED_PLOTS))
    expected_development_routes = {
        ("development", candidate, target)
        for candidate in development_candidates
        for target in TARGETS
    }
    actual_development_routes = {
        route for route in coverage if route[0] == "development"
    }
    if actual_development_routes != expected_development_routes or any(
        coverage[route] != expected_development_indices
        for route in expected_development_routes
    ):
        raise ValueError("Frozen evaluation plan development coverage is not exact")
    if set(test_tasks) != {
        (target, index) for target in TARGETS for index in expected_test_indices
    }:
        raise ValueError("Frozen evaluation plan held-out coverage is not exact")
    if any(
        indices != expected_test_indices
        for route, indices in coverage.items()
        if route[0] == "test"
    ):
        raise ValueError("Frozen evaluation plan held-out candidate routes are not exact")
    return test_tasks


def collect(
    *,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    project_root: Path,
    evaluation_plan_path: Path,
    evaluation_benchmark_commit: str,
    evaluation_evaluator_sha256: str,
    evaluation_test_tasks: dict[tuple[str, int], dict[str, Any]],
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
        plan_task = evaluation_test_tasks.get((target, task_index))
        if plan_task is None:
            raise ValueError(f"Frozen plan has no held-out task for {target}:{task_index}")
        plan_aligned = _required_task_path(plan_task, "aligned_predictions_npz")
        plan_metadata = _required_task_path(plan_task, "alignment_metadata_json")
        plan_metrics = _required_task_path(plan_task, "output_metrics_json")
        plan_source_las = _required_task_path(plan_task, "input_las")
        manifest_input_las = Path(str(plot.get("input_las", ""))).expanduser().resolve()
        if (
            source.get("status") != "evaluated"
            or source.get("safe_for_scoring") is not True
            or source.get("relative_path") != plot["relative_path"]
            or source.get("safe_plot_id") != plot["safe_plot_id"]
            or source.get("collection") != plot["collection"]
            or int(source.get("evaluation_task_index", -1))
            != int(plan_task["evaluation_task_index"])
            or source.get("candidate_id") != plan_task["candidate_id"]
            or int(source.get("candidate_index", -1))
            != int(plan_task["candidate_index"])
            or plan_task["split"] != "test"
            or plan_task["target"] != target
            or int(plan_task["manifest_task_index"]) != task_index
            or plan_task["collection"] != plot["collection"]
            or plan_task["safe_plot_id"] != plot["safe_plot_id"]
            or plan_task["relative_path"] != plot["relative_path"]
            or plan_task["source_workflow_run_id"] != summary["workflow_run_id"]
            or plan_source_las != manifest_input_las
            or plan_task["source_las_sha256"] != plot.get("input_sha256")
        ):
            raise ValueError(f"Invalid plot summary provenance for {target}:{task_index}")
        metrics_path = Path(source["metrics_path"]).expanduser().resolve()
        if (
            metrics_path != plan_metrics
            or not metrics_path.is_file()
            or sha256(metrics_path) != source["metrics_sha256"]
        ):
            raise ValueError(f"Metric file missing or changed: {metrics_path}")
        metrics = load_object(metrics_path)
        metric_provenance = metrics.get("retained_prediction_evaluation_provenance")
        if (
            metrics.get("evaluator") != TLS_EVALUATOR
            or metrics.get("status") != "evaluated"
            or metrics.get("safe_for_scoring") is not True
            or metrics.get("split") != "test"
            or metrics.get("target") != target
            or metrics.get("plot_id") != plot["safe_plot_id"]
            or metrics.get("relative_path") != plot["relative_path"]
            or metrics.get("matching_policy") != MATCHING
            or metrics.get("evaluation_mask") != TLS_MASK
            or float(metrics.get("iou_threshold", -1)) != IOU_THRESHOLD
            or metrics.get("semantic_ignore", {}).get("ignored_semantic_classes")
            != [3]
            or not isinstance(metric_provenance, dict)
            or metric_provenance.get("evaluation_run_id")
            != summary["evaluation_run_id"]
            or metric_provenance.get("evaluation_plan_sha256")
            != summary["evaluation_plan_sha256"]
            or not isinstance(metric_provenance.get("evaluation_plan"), str)
            or Path(metric_provenance["evaluation_plan"]).expanduser().resolve()
            != evaluation_plan_path
            or metric_provenance.get("benchmark_commit")
            != evaluation_benchmark_commit
            or metric_provenance.get("evaluator_sha256")
            != evaluation_evaluator_sha256
            or int(metric_provenance.get("evaluation_task_index", -1))
            != int(plan_task["evaluation_task_index"])
            or metric_provenance.get("source_workflow_run_id")
            != plan_task["source_workflow_run_id"]
            or metric_provenance.get("source_candidate_run_id")
            != plan_task["source_candidate_run_id"]
            or metric_provenance.get("aligned_predictions_npz_sha256")
            != plan_task["aligned_predictions_npz_sha256"]
            or metric_provenance.get("alignment_metadata_sha256")
            != plan_task["alignment_metadata_sha256"]
            or metric_provenance.get("source_las_sha256")
            != plan_task["source_las_sha256"]
            or metric_provenance.get("inference_rerun") is not False
            or metric_provenance.get("prediction_adapter_rerun") is not False
            or metric_provenance.get("retained_sources_changed") is not False
            or metric_provenance.get("configuration_changed") is not False
        ):
            raise ValueError(f"Metric contract changed: {metrics_path}")
        reconcile_plot_metric(
            source=source, metrics=metrics, target=target, task_index=task_index
        )
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
        if (
            aligned != plan_aligned
            or metadata_path != plan_metadata
            or source.get("aligned_predictions_npz_sha256")
            != plan_task["aligned_predictions_npz_sha256"]
            or source.get("alignment_metadata_sha256")
            != plan_task["alignment_metadata_sha256"]
            or not isinstance(source.get("source_las"), str)
            or Path(source["source_las"]).expanduser().resolve() != plan_source_las
            or source.get("source_las_sha256") != plan_task["source_las_sha256"]
        ):
            raise ValueError(
                f"Held-out summary does not match frozen plan task: {target}:{task_index}"
            )
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
        metric_aligned_path = Path(
            str(metrics.get("aligned_predictions_npz", ""))
        ).expanduser().resolve()
        metric_metadata_path = Path(
            str(metrics.get("alignment_metadata_json", ""))
        ).expanduser().resolve()
        metric_source_las = Path(
            str(metrics.get("reference_source", ""))
        ).expanduser().resolve()
        if (
            metadata.get("status") != "passed"
            or metadata.get("target") != target
            or metadata.get("aligned_prediction_npz_sha256") != aligned_sha
            or int(metadata.get("source_row_count", -1)) != int(plot["point_count"])
            or int(metadata.get("prediction_instance_count", -1))
            != int(metrics["semantic_ignore"]["raw_prediction_instance_count"])
            or metric_aligned_path != aligned
            or metric_metadata_path != metadata_path
            or metric_source_las != plan_source_las
            or metric_provenance.get("aligned_predictions_npz_sha256") != aligned_sha
            or metric_provenance.get("alignment_metadata_sha256") != metadata_sha
            or int(
                metrics.get("point_correspondence", {}).get("source_row_count", -1)
            )
            != int(plot["point_count"])
            or metrics.get("point_correspondence", {}).get("status") != "passed"
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
                or len(names)
                != int(metrics["semantic_ignore"]["raw_prediction_instance_count"])
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
        recorded_rows = [
            row for row in summary.get("aggregates", []) if row.get("target") == target
        ]
        if len(recorded_rows) != 1:
            raise ValueError(f"Held-out summary requires one aggregate for {target}")
        recorded = recorded_rows[0]
        if (
            recorded.get("split") != "test"
            or int(recorded.get("expected_plot_count", -1)) != EXPECTED_PLOTS
            or int(recorded.get("evaluated_plot_count", -1)) != EXPECTED_PLOTS
            or int(recorded.get("failed_or_invalid_plot_count", -1)) != 0
        ):
            raise ValueError(f"Held-out aggregate provenance mismatch for {target}")
        reconcile_aggregate(overall=overall, recorded=recorded, target=target)
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


def reject_private_text(writes: dict[Path, str], project_root: Path) -> None:
    """Reject runtime paths and private host/user tokens before publication."""

    def strings(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    def rendered_values(content: str) -> Iterable[str]:
        stripped = content.lstrip()
        if stripped.startswith(("{", "[")):
            yield from strings(json.loads(content))
            return
        from io import StringIO

        for row in csv.reader(StringIO(content)):
            yield from row

    forbidden = (
        str(project_root),
        "/users/",
        "/home/",
        "/mnt/",
        "/private/",
        "/tmp/",
        "fastscratch",
        "barkla",
        "alces.network",
    )
    for path, content in writes.items():
        lowered = content.casefold()
        has_forbidden_token = any(
            token and token.casefold() in lowered for token in forbidden
        )
        has_absolute_path = any(
            Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
            for value in rendered_values(content)
        )
        if has_forbidden_token or has_absolute_path:
            raise ValueError(f"Private path or host token in public output {path}")


def _finalise_locked(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.expanduser().resolve()
    summary_path = args.summary_json.expanduser().resolve()
    manifest_path = args.manifest_json.expanduser().resolve()
    final_selection_path = args.final_selection_json.expanduser().resolve()
    if sha256(final_selection_path) != args.final_selection_sha256:
        raise ValueError("Frozen final-selection checksum changed")
    summary = load_object(summary_path)
    manifest = load_object(manifest_path)
    final_selection = load_object(final_selection_path)
    selected_by_target = validate_final_selection(final_selection)
    validate_summary(summary, args.run_id, args.final_selection_sha256)
    (
        evaluation_plan_path,
        evaluation_evaluator_sha256,
        evaluation_test_tasks,
    ) = validate_evaluation_plan(
        summary=summary,
        project_root=project_root,
        evaluation_benchmark_commit=args.evaluation_benchmark_commit,
        expected_evaluation_run_id=args.evaluation_run_id,
        expected_plan_path=args.evaluation_plan_json,
        expected_plan_sha256=args.evaluation_plan_sha256,
        expected_evaluator_path=args.evaluation_evaluator,
        expected_evaluator_sha256=args.evaluation_evaluator_sha256,
        expected_manifest_path=manifest_path,
        expected_manifest_sha256=summary["manifest_sha256"],
        expected_final_selection_path=final_selection_path,
        expected_final_selection_sha256=args.final_selection_sha256,
        selected_by_target=selected_by_target,
    )
    if sha256(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("Held-out manifest checksum changed")
    if manifest.get("dataset_split") != "test":
        raise ValueError("Finalisation requires the held-out test manifest")
    rows_by_target, retained = collect(
        summary=summary,
        manifest=manifest,
        project_root=project_root,
        evaluation_plan_path=evaluation_plan_path,
        evaluation_benchmark_commit=args.evaluation_benchmark_commit,
        evaluation_evaluator_sha256=evaluation_evaluator_sha256,
        evaluation_test_tasks=evaluation_test_tasks,
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
        "evaluation_evaluator_sha256": evaluation_evaluator_sha256,
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
        "storage_status": "run_scoped_retained",
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
    public_writes = {
        paths["leaf_on_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_on"]),
        paths["leaf_on_site"]: csv_text(SUMMARY_FIELDS, site_rows["leaf_on"]),
        paths["leaf_on_overall"]: csv_text(SUMMARY_FIELDS, [overall_rows["leaf_on"]]),
        paths["leaf_off_plot"]: csv_text(PLOT_FIELDS, rows_by_target["leaf_off"]),
        paths["leaf_off_site"]: csv_text(SUMMARY_FIELDS, site_rows["leaf_off"]),
        paths["leaf_off_overall"]: csv_text(SUMMARY_FIELDS, [overall_rows["leaf_off"]]),
        paths["provenance"]: json_text(provenance),
    }
    reject_private_text(
        {
            **public_writes,
            Path("development_tuned_result_row"): csv_text(
                RESULT_FIELDS, [headline]
            ),
            Path("development_tuned_diagnostic_row"): csv_text(
                RESULT_FIELDS, [diagnostic]
            ),
            Path("development_tuned_retention_row"): csv_text(
                RETENTION_FIELDS, [retention_row]
            ),
        },
        project_root,
    )
    writes = {
        **public_writes,
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
    publish_writes(
        writes,
        project_root=project_root,
        enforce_head_baseline=bool(getattr(args, "validate_worktree", False)),
        expected_head=(
            args.benchmark_commit
            if getattr(args, "validate_worktree", False)
            else None
        ),
    )
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


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, render, publish, and receipt one locked public transaction."""

    project_root = args.project_root.expanduser().resolve()
    with publication_lock(project_root):
        preflight_text_target(
            args.receipt_json,
            project_root=project_root,
            staging_suffix=".tls2trees-held-out-receipt.tmp",
        )
        if getattr(args, "validate_worktree", False):
            public_paths = (
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_test_plot_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_test_site_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_test_results.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_leaf_off_test_plot_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_leaf_off_test_site_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_leaf_off_test_diagnostic.csv",
                "methods/tls2trees/examples/"
                "tls2trees_development_tuned_test_provenance.json",
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
                    f"{parent}/.{name}.tls2trees-held-out-finalisation.tmp"
                )
            validate_git_worktree(
                project_root,
                recovery_confirmed=bool(args.recovery_confirmed),
                recovery_paths=recovery_paths,
                expected_head=args.benchmark_commit,
            )
        payload = _finalise_locked(args)
        publish_text_bundle(
            {args.receipt_json: json_text(payload)},
            project_root=project_root,
            staging_suffix=".tls2trees-held-out-receipt.tmp",
            replace=os.replace,
            expected_head=(
                args.benchmark_commit
                if getattr(args, "validate_worktree", False)
                else None
            ),
        )
        return payload


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
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--evaluation-plan-json", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--evaluation-evaluator", type=Path, required=True)
    parser.add_argument("--evaluation-evaluator-sha256", required=True)
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
