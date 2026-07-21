"""Evaluate retained TLS2trees source-row predictions with the final protocol.

This workflow is deliberately evaluation-only. It binds the frozen development
and held-out configurations directly to retained NPZ predictions, their
alignment metadata, the source LAS files, and the held-out retention manifest.
It never invokes TLS2trees inference or selects a configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


TARGETS = ("leaf_off", "leaf_on")
FINAL_EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
EVALUATION_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
EXPECTED_DEVELOPMENT_PLOTS = 21
EXPECTED_TEST_PLOTS = 11
EXPECTED_TASKS = 106
ROOT = Path(__file__).resolve().parents[4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def require_exact_manifest(
    manifest: Mapping[str, Any], *, split: str, plot_count: int
) -> list[dict[str, Any]]:
    plots = manifest.get("plots")
    if manifest.get("dataset_split") != split or not isinstance(plots, list):
        raise ValueError(f"Expected the exact {split!r} FOR-instance manifest")
    if len(plots) != plot_count:
        raise ValueError(f"Expected {plot_count} {split} plots; found {len(plots)}")
    if [int(row["task_index"]) for row in plots] != list(range(plot_count)):
        raise ValueError(f"{split} task indexes must be contiguous from zero")
    required = {"task_index", "safe_plot_id", "relative_path", "collection", "input_las"}
    for row in plots:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Manifest plot is missing fields {missing}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(row["safe_plot_id"])):
            raise ValueError(f"Unsafe manifest plot ID: {row['safe_plot_id']!r}")
    return plots


def retained_test_predictions(
    path: Path, expected_sha256: str, *, workflow_run_id: str
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise RuntimeError("Committed held-out retention manifest is missing or changed")
    payload = read_json(path)
    files = payload.get("files")
    if (
        payload.get("status") != "retention_verified"
        or payload.get("run_id") != workflow_run_id
        or payload.get("dataset_split") != "test"
        or payload.get("verified_prediction_files") != 22
        or not isinstance(files, list)
        or len(files) != 22
    ):
        raise ValueError("Held-out retention manifest is not the frozen 22-file set")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in files:
        key = (str(row.get("target", "")), int(row.get("plot_index", -1)))
        relative = Path(str(row.get("relative_path", ""))).expanduser()
        prediction = relative if relative.is_absolute() else ROOT / relative
        prediction = prediction.resolve()
        digest = str(row.get("sha256", ""))
        if (
            key in indexed
            or key[0] not in TARGETS
            or key[1] not in range(EXPECTED_TEST_PLOTS)
            or not prediction.is_file()
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or sha256(prediction) != digest
        ):
            raise ValueError("Held-out retained prediction evidence is invalid")
        indexed[key] = {**row, "resolved_path": str(prediction)}
    if set(indexed) != {
        (target, task_index)
        for target in TARGETS
        for task_index in range(EXPECTED_TEST_PLOTS)
    }:
        raise ValueError("Held-out retention manifest does not cover all target plots")
    return indexed


def validate_retained_source(
    *,
    aligned_path: Path,
    alignment_path: Path,
    input_las: Path,
    target: str,
    plot: Mapping[str, Any],
) -> dict[str, Any]:
    for path in (aligned_path, alignment_path, input_las):
        if not path.is_file():
            raise FileNotFoundError(f"Required retained source is missing: {path}")
    alignment = read_json(alignment_path)
    aligned_sha256 = sha256(aligned_path)
    source_sha256 = sha256(input_las)
    if (
        alignment.get("status") != "passed"
        or alignment.get("target") != target
        or alignment.get("point_correspondence")
        != "source_row_via_voxel_representative"
        or alignment.get("raw_coordinate_evaluation_permitted") is not False
        or resolve(alignment.get("aligned_prediction_npz", "")) != aligned_path
        or resolve(alignment.get("source_las_path", "")) != input_las
        or alignment.get("aligned_prediction_npz_sha256") != aligned_sha256
        or alignment.get("source_las_sha256") != source_sha256
        or int(alignment.get("source_row_count", -1))
        != int(plot.get("point_count", -2))
    ):
        raise ValueError(f"Retained alignment provenance mismatch: {alignment_path}")
    return alignment


def _task(
    *,
    task_index: int,
    split: str,
    workflow_run_id: str,
    candidate_id: str,
    candidate_index: int,
    target: str,
    plot: Mapping[str, Any],
    output_root: Path,
    metrics_root: Path,
    retained_test_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split == "development":
        candidate_run_id = f"{workflow_run_id}__{candidate_id}"
    else:
        candidate_run_id = f"{workflow_run_id}__{target}__{candidate_id}"
    plot_root = (
        output_root
        / "tls2trees"
        / "for_instance"
        / "development_tuned"
        / split
        / candidate_run_id
        / str(plot["safe_plot_id"])
    )
    aligned_root = plot_root / "predictions" / "aligned" / target
    aligned_path = aligned_root / "source_row_predictions.npz"
    alignment_path = aligned_root / "alignment_metadata.json"
    input_las = resolve(str(plot["input_las"]))
    alignment = validate_retained_source(
        aligned_path=aligned_path,
        alignment_path=alignment_path,
        input_las=input_las,
        target=target,
        plot=plot,
    )
    aligned_sha256 = sha256(aligned_path)
    if retained_test_prediction is not None:
        if (
            resolve(retained_test_prediction["resolved_path"]) != aligned_path
            or retained_test_prediction.get("sha256") != aligned_sha256
            or retained_test_prediction.get("candidate_id") != candidate_id
            or retained_test_prediction.get("target") != target
            or int(retained_test_prediction.get("plot_index", -1))
            != int(plot["task_index"])
            or retained_test_prediction.get("plot_id") != plot["safe_plot_id"]
        ):
            raise ValueError("Held-out task does not match committed retention evidence")
    task_root = (
        metrics_root
        / split
        / candidate_id
        / str(plot["safe_plot_id"])
        / target
    )
    return {
        "evaluation_task_index": task_index,
        "split": split,
        "source_workflow_run_id": workflow_run_id,
        "source_candidate_run_id": candidate_run_id,
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "target": target,
        "manifest_task_index": int(plot["task_index"]),
        "collection": plot["collection"],
        "safe_plot_id": plot["safe_plot_id"],
        "relative_path": plot["relative_path"],
        "input_las": str(input_las),
        "aligned_predictions_npz": str(aligned_path),
        "aligned_predictions_npz_sha256": aligned_sha256,
        "alignment_metadata_json": str(alignment_path),
        "alignment_metadata_sha256": sha256(alignment_path),
        "source_las_sha256": alignment["source_las_sha256"],
        "output_root": str(task_root),
        "output_metrics_json": str(task_root / "plot_metrics.json"),
        "output_matches_csv": str(task_root / "matches.csv"),
        "output_unmatched_predictions_csv": str(
            task_root / "unmatched_predictions.csv"
        ),
        "output_unmatched_references_csv": str(
            task_root / "unmatched_references.csv"
        ),
    }


def build_plan(
    *,
    evaluation_run_id: str,
    benchmark_commit: str,
    evaluator_path: Path,
    metrics_root: Path,
    development_output_root: Path,
    development_workflow_run_id: str,
    development_manifest_path: Path,
    development_selection_path: Path,
    test_output_root: Path,
    test_workflow_run_id: str,
    test_manifest_path: Path,
    final_selection_path: Path,
    final_selection_sha256: str,
    test_retention_manifest_path: Path,
    test_retention_manifest_sha256: str,
) -> dict[str, Any]:
    paths = [
        evaluator_path,
        development_manifest_path,
        development_selection_path,
        test_manifest_path,
        final_selection_path,
        test_retention_manifest_path,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    if FINAL_EVALUATOR not in evaluator_source or EVALUATION_MASK not in evaluator_source:
        raise ValueError("Evaluator is not the required class-3-ignore implementation")

    development_manifest = read_json(development_manifest_path)
    development_plots = require_exact_manifest(
        development_manifest,
        split="development",
        plot_count=EXPECTED_DEVELOPMENT_PLOTS,
    )
    selection = read_json(development_selection_path)
    if (
        selection.get("status") != "frozen_for_full_development_stage2"
        or selection.get("held_out_test_accessed") is not False
        or selection.get("confirmation_no_test_metrics_used") is not True
    ):
        raise ValueError("Stage-2 selection is not a frozen development-only selection")
    candidates = selection.get("selected_candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("Final evaluation requires exactly two frozen candidates")

    test_manifest = read_json(test_manifest_path)
    test_plots = require_exact_manifest(
        test_manifest, split="test", plot_count=EXPECTED_TEST_PLOTS
    )
    if sum(int(row.get("point_count", 0)) for row in test_plots) != 49_709_922:
        raise ValueError("Held-out manifest point count is not the frozen 49,709,922")
    if sum(int(row.get("reference_tree_count", 0)) for row in test_plots) != 323:
        raise ValueError("Held-out manifest tree count is not the frozen 323")
    if {row["relative_path"] for row in development_plots} & {
        row["relative_path"] for row in test_plots
    }:
        raise ValueError("Development and held-out manifests overlap")
    if sha256(final_selection_path) != final_selection_sha256:
        raise ValueError("Reviewed final-selection checksum changed")
    final_selection = read_json(final_selection_path)
    if (
        final_selection.get("status") != "development_tuned_configuration_frozen"
        or final_selection.get("final_configuration_selected") is not True
        or final_selection.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Final selection is not the frozen pre-test configuration")
    retained_predictions = retained_test_predictions(
        test_retention_manifest_path,
        test_retention_manifest_sha256,
        workflow_run_id=test_workflow_run_id,
    )

    tasks: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", candidate_id):
            raise ValueError(f"Unsafe Stage-2 candidate ID: {candidate_id!r}")
        candidate_index = int(candidate["stage1_candidate_index"])
        for plot in development_plots:
            for target in TARGETS:
                tasks.append(
                    _task(
                        task_index=len(tasks),
                        split="development",
                        workflow_run_id=development_workflow_run_id,
                        candidate_id=candidate_id,
                        candidate_index=candidate_index,
                        target=target,
                        plot=plot,
                        output_root=development_output_root,
                        metrics_root=metrics_root,
                    )
                )
    for target in TARGETS:
        selected = final_selection["selected_by_target"][target]
        candidate_id = str(selected["candidate_id"])
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", candidate_id):
            raise ValueError(f"Unsafe held-out candidate ID: {candidate_id!r}")
        candidate_index = int(selected["stage1_candidate_index"])
        for plot in test_plots:
            tasks.append(
                _task(
                    task_index=len(tasks),
                    split="test",
                    workflow_run_id=test_workflow_run_id,
                    candidate_id=candidate_id,
                    candidate_index=candidate_index,
                    target=target,
                    plot=plot,
                    output_root=test_output_root,
                    metrics_root=metrics_root,
                    retained_test_prediction=retained_predictions[
                        (target, int(plot["task_index"]))
                    ],
                )
            )
    if len(tasks) != EXPECTED_TASKS:
        raise AssertionError(
            f"Expected {EXPECTED_TASKS} retained-evaluation tasks; built {len(tasks)}"
        )
    output_paths = [task["output_metrics_json"] for task in tasks]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Retained-evaluation task outputs are not unique")
    if any(Path(path).exists() for path in output_paths):
        raise FileExistsError("A retained-evaluation metric output already exists")
    return {
        "schema_version": 1,
        "status": "retained_prediction_evaluation_plan_frozen",
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "evaluation_run_id": evaluation_run_id,
        "benchmark_commit": benchmark_commit,
        "evaluator": FINAL_EVALUATOR,
        "evaluation_protocol": "for_instance_pointwise_class3_ignore",
        "evaluation_mask": EVALUATION_MASK,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256(evaluator_path),
        "iou_threshold": 0.5,
        "matching_policy": "maximum_cardinality_one_to_one",
        "source_point_correspondence": "source_row_via_voxel_representative",
        "inference_rerun": False,
        "prediction_adapter_rerun": False,
        "configuration_changed": False,
        "configuration_selection_performed": False,
        "development": {
            "workflow_run_id": development_workflow_run_id,
            "manifest": str(development_manifest_path),
            "manifest_sha256": sha256(development_manifest_path),
            "selection": str(development_selection_path),
            "selection_sha256": sha256(development_selection_path),
            "held_out_test_accessed": False,
        },
        "test": {
            "workflow_run_id": test_workflow_run_id,
            "manifest": str(test_manifest_path),
            "manifest_sha256": sha256(test_manifest_path),
            "final_selection": str(final_selection_path),
            "final_selection_sha256": final_selection_sha256,
            "retention_manifest": str(test_retention_manifest_path),
            "retention_manifest_sha256": test_retention_manifest_sha256,
            "held_out_test_already_accessed": True,
            "test_metrics_used_for_configuration_selection": False,
        },
        "expected_task_count": EXPECTED_TASKS,
        "tasks": tasks,
    }


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location(
        "tls2trees_retained_prediction_evaluator", path
    )
    if not spec or not spec.loader:
        raise ImportError(f"Unable to load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv_exclusive(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _write_or_verify_text(
    path: Path, text: str, *, resume_exact_partial: bool
) -> None:
    if path.exists():
        if not resume_exact_partial:
            raise FileExistsError(f"Refusing to overwrite immutable output: {path}")
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing partial output does not match recomputed content: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def run_task(*, plan_path: Path, plan_sha256: str, task_index: int) -> dict[str, Any]:
    if sha256(plan_path) != plan_sha256:
        raise RuntimeError("Frozen evaluation plan checksum changed")
    plan = read_json(plan_path)
    if (
        plan.get("status") != "retained_prediction_evaluation_plan_frozen"
        or plan.get("expected_task_count") != EXPECTED_TASKS
        or len(plan.get("tasks", [])) != EXPECTED_TASKS
    ):
        raise ValueError("Evaluation plan is incomplete or not frozen")
    task = plan["tasks"][task_index]
    if int(task["evaluation_task_index"]) != task_index:
        raise ValueError("Evaluation task index mismatch")
    evaluator_path = resolve(plan["evaluator_path"])
    if sha256(evaluator_path) != plan["evaluator_sha256"]:
        raise RuntimeError("Evaluator checksum changed after evaluation submission")
    alignment_path = resolve(task["alignment_metadata_json"])
    aligned_path = resolve(task["aligned_predictions_npz"])
    input_las = resolve(task["input_las"])
    if sha256(alignment_path) != task["alignment_metadata_sha256"]:
        raise RuntimeError("Retained alignment metadata changed after plan freeze")
    if sha256(aligned_path) != task["aligned_predictions_npz_sha256"]:
        raise RuntimeError("Retained source-row prediction checksum changed")
    if sha256(input_las) != task["source_las_sha256"]:
        raise RuntimeError("FOR-instance source LAS checksum changed")

    output_root = resolve(task["output_root"])
    metrics_path = resolve(task["output_metrics_json"])
    outputs = [
        metrics_path,
        resolve(task["output_matches_csv"]),
        resolve(task["output_unmatched_predictions_csv"]),
        resolve(task["output_unmatched_references_csv"]),
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Refusing to overwrite an immutable retained-evaluation output"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    evaluator = load_evaluator(evaluator_path)
    alignment = read_json(alignment_path)
    # The retained evidence is validated field-by-field above. Canonicalise the
    # schema label in memory so historical files remain evaluable without
    # weakening any path, checksum, row-count, or coordinate-frame checks.
    alignment["schema_version"] = evaluator.ALIGNMENT_SCHEMA
    result = evaluator.evaluate_aligned_plot(
        target=task["target"],
        aligned_predictions=aligned_path,
        reference_source=input_las,
        alignment_metadata=alignment,
        iou_threshold=float(plan["iou_threshold"]),
    )
    if (
        result.get("evaluator") != FINAL_EVALUATOR
        or result.get("evaluation_mask") != EVALUATION_MASK
        or result.get("semantic_ignore", {}).get("ignored_semantic_classes") != [3]
    ):
        raise RuntimeError("Evaluator did not emit the required class-3-ignore contract")
    result.update(
        {
            "timestamp_utc": utc_now(),
            "plot_id": task["safe_plot_id"],
            "relative_path": task["relative_path"],
            "split": "dev" if task["split"] == "development" else "test",
            "prediction_directory": None,
            "aligned_predictions_npz": str(aligned_path),
            "reference_source": str(input_las),
            "alignment_metadata_json": str(alignment_path),
            "alignment_metadata_sha256": task["alignment_metadata_sha256"],
            "retained_prediction_evaluation_provenance": {
                "schema_version": 1,
                "evaluation_run_id": plan["evaluation_run_id"],
                "evaluation_task_index": task_index,
                "benchmark_commit": plan["benchmark_commit"],
                "evaluator_sha256": plan["evaluator_sha256"],
                "evaluation_plan": str(plan_path),
                "evaluation_plan_sha256": plan_sha256,
                "source_workflow_run_id": task["source_workflow_run_id"],
                "source_candidate_run_id": task["source_candidate_run_id"],
                "alignment_metadata_sha256": task["alignment_metadata_sha256"],
                "aligned_predictions_npz_sha256": task[
                    "aligned_predictions_npz_sha256"
                ],
                "source_las_sha256": task["source_las_sha256"],
                "inference_rerun": False,
                "prediction_adapter_rerun": False,
                "retained_sources_changed": False,
                "configuration_changed": False,
            },
        }
    )
    with metrics_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv_exclusive(
        outputs[1],
        [
            "prediction",
            "reference_tree_id",
            "intersection",
            "predicted_points",
            "reference_points",
            "union",
            "iou",
        ],
        result["matches"],
    )
    _write_csv_exclusive(
        outputs[2],
        ["prediction"],
        [{"prediction": value} for value in result["unmatched_predictions"]],
    )
    _write_csv_exclusive(
        outputs[3],
        ["reference_tree_id"],
        [
            {"reference_tree_id": value}
            for value in result["unmatched_references"]
        ],
    )
    if sha256(alignment_path) != task["alignment_metadata_sha256"]:
        raise RuntimeError("Retained alignment metadata changed during evaluation")
    if sha256(aligned_path) != task["aligned_predictions_npz_sha256"]:
        raise RuntimeError("Retained source-row prediction changed during evaluation")
    if sha256(input_las) != task["source_las_sha256"]:
        raise RuntimeError("FOR-instance source LAS changed during evaluation")
    return result


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _aggregate(
    rows: list[dict[str, Any]], *, split: str, candidate: str, target: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["split"] == split
        and row["candidate_id"] == candidate
        and row["target"] == target
    ]
    valid = [
        row
        for row in selected
        if row["status"] == "evaluated" and row["safe_for_scoring"] is True
    ]
    tp = sum(int(row["true_positives"]) for row in valid)
    fp = sum(int(row["false_positives"]) for row in valid)
    fn = sum(int(row["false_negatives"]) for row in valid)
    precision, recall, f1 = _prf(tp, fp, fn)
    by_collection: dict[str, list[float]] = {}
    for row in valid:
        by_collection.setdefault(row["collection"], []).append(float(row["f1"]))
    return {
        "split": split,
        "candidate_id": candidate,
        "target": target,
        "expected_plot_count": len(selected),
        "evaluated_plot_count": len(valid),
        "failed_or_invalid_plot_count": len(selected) - len(valid),
        "raw_prediction_instance_count": sum(
            int(row["raw_prediction_instance_count"]) for row in valid
        ),
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in valid
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in valid
        ),
        "ignored_predicted_point_count": sum(
            int(row["ignored_predicted_point_count"]) for row in valid
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": (
            sum(float(row["f1"]) for row in valid) / len(valid) if valid else 0.0
        ),
        "mean_collection_f1": {
            key: sum(values) / len(values)
            for key, values in sorted(by_collection.items())
        },
        "oversegmented_reference_count": sum(
            int(row["oversegmented_reference_count"]) for row in valid
        ),
        "undersegmented_prediction_count": sum(
            int(row["undersegmented_prediction_count"]) for row in valid
        ),
    }


def summarise_plan(*, plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    if sha256(plan_path) != plan_sha256:
        raise RuntimeError("Frozen evaluation plan checksum changed")
    plan = read_json(plan_path)
    if len(plan.get("tasks", [])) != EXPECTED_TASKS:
        raise ValueError("Expected the complete 106-task evaluation plan")
    if sha256(resolve(plan["evaluator_path"])) != plan["evaluator_sha256"]:
        raise RuntimeError("Evaluator checksum changed during evaluation")
    rows: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for task in plan["tasks"]:
        metric_path = resolve(task["output_metrics_json"])
        row: dict[str, Any] = {
            "evaluation_task_index": task["evaluation_task_index"],
            "split": task["split"],
            "candidate_id": task["candidate_id"],
            "candidate_index": task["candidate_index"],
            "target": task["target"],
            "manifest_task_index": task["manifest_task_index"],
            "collection": task["collection"],
            "safe_plot_id": task["safe_plot_id"],
            "relative_path": task["relative_path"],
            "status": "missing",
            "safe_for_scoring": False,
            "raw_prediction_instance_count": None,
            "prediction_instance_count": None,
            "reference_instance_count": None,
            "ignored_predicted_point_count": None,
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "mean_matched_iou": None,
            "oversegmented_reference_count": None,
            "undersegmented_prediction_count": None,
            "aligned_predictions_npz": task["aligned_predictions_npz"],
            "aligned_predictions_npz_sha256": task[
                "aligned_predictions_npz_sha256"
            ],
            "alignment_metadata_json": task["alignment_metadata_json"],
            "alignment_metadata_sha256": task["alignment_metadata_sha256"],
            "source_las": task["input_las"],
            "source_las_sha256": task["source_las_sha256"],
            "metrics_path": str(metric_path),
            "metrics_sha256": None,
        }
        if metric_path.is_file():
            metric = read_json(metric_path)
            provenance = metric.get("retained_prediction_evaluation_provenance", {})
            expected_split = "dev" if task["split"] == "development" else "test"
            if (
                metric.get("evaluator") != FINAL_EVALUATOR
                or metric.get("evaluation_mask") != EVALUATION_MASK
                or metric.get("split") != expected_split
                or metric.get("target") != task["target"]
                or metric.get("plot_id") != task["safe_plot_id"]
                or metric.get("relative_path") != task["relative_path"]
                or metric.get("semantic_ignore", {}).get("ignored_semantic_classes")
                != [3]
                or provenance.get("evaluation_plan_sha256") != plan_sha256
                or provenance.get("alignment_metadata_sha256")
                != task["alignment_metadata_sha256"]
                or provenance.get("inference_rerun") is not False
                or provenance.get("configuration_changed") is not False
            ):
                raise ValueError(
                    f"Retained-evaluation metric provenance mismatch: {metric_path}"
                )
            if sha256(resolve(task["alignment_metadata_json"])) != task[
                "alignment_metadata_sha256"
            ]:
                raise RuntimeError("Retained alignment evidence changed")
            if sha256(resolve(task["aligned_predictions_npz"])) != task[
                "aligned_predictions_npz_sha256"
            ]:
                raise RuntimeError("Retained prediction evidence changed")
            if sha256(resolve(task["input_las"])) != task["source_las_sha256"]:
                raise RuntimeError("Source LAS evidence changed")
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
                row[key] = metric.get(key)
            row["raw_prediction_instance_count"] = metric["semantic_ignore"][
                "raw_prediction_instance_count"
            ]
            row["ignored_predicted_point_count"] = metric["semantic_ignore"][
                "ignored_predicted_point_count"
            ]
            row["metrics_sha256"] = sha256(metric_path)
        if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
            incomplete.append(
                f"{task['split']}:{task['candidate_id']}:{task['safe_plot_id']}:{task['target']}"
            )
        rows.append(row)

    development_routes = sorted(
        {(row["candidate_id"], row["target"]) for row in rows if row["split"] == "development"}
    )
    test_routes = sorted(
        {(row["candidate_id"], row["target"]) for row in rows if row["split"] == "test"}
    )
    development_aggregates = [
        _aggregate(rows, split="development", candidate=candidate, target=target)
        for candidate, target in development_routes
    ]
    test_aggregates = [
        _aggregate(rows, split="test", candidate=candidate, target=target)
        for candidate, target in test_routes
    ]
    rankings = {
        target: [
            item["candidate_id"]
            for item in sorted(
                (row for row in development_aggregates if row["target"] == target),
                key=lambda row: (
                    row["failed_or_invalid_plot_count"],
                    -row["mean_plot_f1"],
                    -row["micro_f1"],
                    row["candidate_id"],
                ),
            )
        ]
        for target in TARGETS
    }
    return {
        "schema_version": 1,
        "status": (
            "retained_prediction_evaluation_completed"
            if not incomplete
            else "retained_prediction_evaluation_incomplete"
        ),
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "evaluation_run_id": plan["evaluation_run_id"],
        "evaluator": FINAL_EVALUATOR,
        "evaluation_protocol": "for_instance_pointwise_class3_ignore",
        "evaluation_mask": EVALUATION_MASK,
        "evaluator_sha256": plan["evaluator_sha256"],
        "evaluation_plan": str(plan_path),
        "evaluation_plan_sha256": plan_sha256,
        "expected_metric_count": EXPECTED_TASKS,
        "valid_metric_count": EXPECTED_TASKS - len(incomplete),
        "incomplete_tasks": incomplete,
        "inference_rerun": False,
        "prediction_adapter_rerun": False,
        "retained_sources_unchanged": True,
        "configuration_changed_after_test": False,
        "configuration_selection_performed": False,
        "test_metrics_used_for_configuration_selection": False,
        "development_source": plan["development"],
        "test_source": plan["test"],
        "development_candidate_rankings_for_diagnostic_review": rankings,
        "plot_metrics": rows,
        "development_aggregates": development_aggregates,
        "test_aggregates": test_aggregates,
        "next_gate": "review_final_protocol_metrics_without_retuning_from_test",
    }


def write_summary(
    *,
    payload: dict[str, Any],
    output_json: Path,
    plot_csv: Path,
    aggregate_csv: Path,
    resume_exact_partial: bool = False,
) -> None:
    outputs = (output_json, plot_csv, aggregate_csv)
    if not resume_exact_partial and any(path.exists() for path in outputs):
        raise FileExistsError(
            "Refusing to overwrite immutable retained-evaluation summary output"
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows = payload["development_aggregates"] + payload["test_aggregates"]
    csv_rows = [
        {
            **row,
            "mean_collection_f1": json.dumps(row["mean_collection_f1"], sort_keys=True),
        }
        for row in aggregate_rows
    ]
    contents = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        _csv_text(list(payload["plot_metrics"][0]), payload["plot_metrics"]),
        _csv_text(list(csv_rows[0]), csv_rows),
    )
    for path, text in zip(outputs, contents, strict=True):
        _write_or_verify_text(
            path, text, resume_exact_partial=resume_exact_partial
        )


def held_out_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the final 22-row held-out summary consumed by finalisation."""

    test_source = payload["test_source"]
    rows = []
    for source in payload["plot_metrics"]:
        if source["split"] != "test":
            continue
        row = dict(source)
        row["task_index"] = row.pop("manifest_task_index")
        rows.append(row)
    incomplete = [
        f"{row['target']}:{row['safe_plot_id']}"
        for row in rows
        if row["status"] != "evaluated" or row["safe_for_scoring"] is not True
    ]
    return {
        "schema_version": 2,
        "status": (
            "held_out_test_completed" if not incomplete else "held_out_test_incomplete"
        ),
        "created_at_utc": payload["created_at_utc"],
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "test",
        "workflow_run_id": test_source["workflow_run_id"],
        "manifest": test_source["manifest"],
        "manifest_sha256": test_source["manifest_sha256"],
        "final_selection": test_source["final_selection"],
        "final_selection_sha256": test_source["final_selection_sha256"],
        "expected_plot_count": EXPECTED_TEST_PLOTS,
        "expected_metric_count": EXPECTED_TEST_PLOTS * len(TARGETS),
        "valid_metric_count": len(rows) - len(incomplete),
        "incomplete_tasks": incomplete,
        "held_out_test_accessed": True,
        "held_out_accuracy_metrics_computed": True,
        "configuration_changed_after_test": False,
        "evaluator": FINAL_EVALUATOR,
        "evaluation_protocol": "for_instance_pointwise_class3_ignore",
        "evaluation_mask": EVALUATION_MASK,
        "evaluation_run_id": payload["evaluation_run_id"],
        "evaluation_plan": payload["evaluation_plan"],
        "evaluation_plan_sha256": payload["evaluation_plan_sha256"],
        "inference_rerun": False,
        "prediction_adapter_rerun": False,
        "retained_sources_unchanged": True,
        "test_metrics_used_for_configuration_selection": False,
        "plot_metrics": rows,
        "aggregates": payload["test_aggregates"],
        "next_gate": "finalise_held_out_results_without_retuning",
    }


def write_held_out_summary(
    *,
    payload: dict[str, Any],
    output_json: Path,
    plot_csv: Path,
    aggregate_csv: Path,
    resume_exact_partial: bool = False,
) -> None:
    outputs = (output_json, plot_csv, aggregate_csv)
    if not resume_exact_partial and any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite final held-out summary output")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_rows = [
        {
            **row,
            "mean_collection_f1": json.dumps(row["mean_collection_f1"], sort_keys=True),
        }
        for row in payload["aggregates"]
    ]
    contents = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        _csv_text(list(payload["plot_metrics"][0]), payload["plot_metrics"]),
        _csv_text(list(aggregate_rows[0]), aggregate_rows),
    )
    for path, text in zip(outputs, contents, strict=True):
        _write_or_verify_text(
            path, text, resume_exact_partial=resume_exact_partial
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("build-plan")
    for name in (
        "evaluation-run-id",
        "benchmark-commit",
        "evaluator",
        "metrics-root",
        "development-output-root",
        "development-workflow-run-id",
        "development-manifest-json",
        "development-selection-json",
        "test-output-root",
        "test-workflow-run-id",
        "test-manifest-json",
        "final-selection-json",
        "final-selection-sha256",
        "test-retention-manifest-json",
        "test-retention-manifest-sha256",
        "output-plan-json",
    ):
        plan.add_argument(f"--{name}", required=True)
    task = subparsers.add_parser("run-task")
    task.add_argument("--plan-json", required=True)
    task.add_argument("--plan-sha256", required=True)
    task.add_argument("--task-index", required=True, type=int)
    summary = subparsers.add_parser("summarise")
    summary.add_argument("--plan-json", required=True)
    summary.add_argument("--plan-sha256", required=True)
    summary.add_argument("--output-json", required=True)
    summary.add_argument("--plot-csv", required=True)
    summary.add_argument("--aggregate-csv", required=True)
    summary.add_argument("--test-output-json", required=True)
    summary.add_argument("--test-plot-csv", required=True)
    summary.add_argument("--test-aggregate-csv", required=True)
    summary.add_argument("--resume-exact-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-plan":
        output = resolve(args.output_plan_json)
        if output.exists():
            raise FileExistsError("Refusing to overwrite a frozen evaluation plan")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = build_plan(
            evaluation_run_id=args.evaluation_run_id,
            benchmark_commit=args.benchmark_commit,
            evaluator_path=resolve(args.evaluator),
            metrics_root=resolve(args.metrics_root),
            development_output_root=resolve(args.development_output_root),
            development_workflow_run_id=args.development_workflow_run_id,
            development_manifest_path=resolve(args.development_manifest_json),
            development_selection_path=resolve(args.development_selection_json),
            test_output_root=resolve(args.test_output_root),
            test_workflow_run_id=args.test_workflow_run_id,
            test_manifest_path=resolve(args.test_manifest_json),
            final_selection_path=resolve(args.final_selection_json),
            final_selection_sha256=args.final_selection_sha256,
            test_retention_manifest_path=resolve(
                args.test_retention_manifest_json
            ),
            test_retention_manifest_sha256=args.test_retention_manifest_sha256,
        )
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"status={payload['status']}")
        print(f"tasks={payload['expected_task_count']}")
        print(f"plan={output}")
        return 0
    if args.command == "run-task":
        result = run_task(
            plan_path=resolve(args.plan_json),
            plan_sha256=args.plan_sha256,
            task_index=args.task_index,
        )
        print("status=retained_prediction_evaluation_task_completed")
        print(f"evaluator={result['evaluator']}")
        return 0
    payload = summarise_plan(
        plan_path=resolve(args.plan_json), plan_sha256=args.plan_sha256
    )
    output_json = resolve(args.output_json)
    if args.resume_exact_partial and output_json.exists():
        existing = read_json(output_json)
        created_at = existing.get("created_at_utc")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("Existing partial summary has no valid creation timestamp")
        payload["created_at_utc"] = created_at
    write_summary(
        payload=payload,
        output_json=output_json,
        plot_csv=resolve(args.plot_csv),
        aggregate_csv=resolve(args.aggregate_csv),
        resume_exact_partial=args.resume_exact_partial,
    )
    test_summary = held_out_summary(payload)
    write_held_out_summary(
        payload=test_summary,
        output_json=resolve(args.test_output_json),
        plot_csv=resolve(args.test_plot_csv),
        aggregate_csv=resolve(args.test_aggregate_csv),
        resume_exact_partial=args.resume_exact_partial,
    )
    print(f"status={payload['status']}")
    print(
        f"valid_metrics={payload['valid_metric_count']}/{payload['expected_metric_count']}"
    )
    for row in payload["test_aggregates"]:
        print(
            f"test {row['target']}={row['candidate_id']} "
            f"micro_f1={row['micro_f1']:.6f} precision={row['precision']:.6f} "
            f"recall={row['recall']:.6f} invalid={row['failed_or_invalid_plot_count']}"
        )
    print("inference_rerun=false")
    print("retained_sources_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
