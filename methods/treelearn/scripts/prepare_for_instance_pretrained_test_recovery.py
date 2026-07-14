"""Prepare one execution-only recovery of a frozen TreeLearn test task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from for_instance_test_common import load_test_manifest


TASK_INDEX = 8
RELATIVE_PATH = "SCION/plot_31_annotated.las"
SAFE_PLOT_ID = "SCION_plot_31_annotated"
POLICY = "map_all_unassigned_to_background_when_initial_grouping_is_empty"
AGGREGATES = (
    "plot_summary.csv",
    "site_summary.csv",
    "final_summary.csv",
    "failures.csv",
    "matches.csv",
    "unmatched_predictions.csv",
    "unmatched_references.csv",
    "retention_manifest.json",
    "run_summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_inventory(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path),
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }


def move_to_archive(source: Path, archive_root: Path, label: str) -> dict[str, Any]:
    target = archive_root / label
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        entry = {
            "source": str(source),
            "archive": str(target),
            "size_bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    else:
        entry = {"source": str(source), "archive": str(target), **directory_inventory(source)}
    shutil.move(str(source), str(target))
    return entry


def prepare(
    manifest_path: Path,
    runtime_root: Path,
    prediction_root: Path,
    metadata_root: Path,
    table_root: Path,
    archive_root: Path,
    output: Path,
    original_benchmark_commit: str,
    recovery_benchmark_commit: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", original_benchmark_commit):
        raise ValueError("Original benchmark commit must be a full SHA-1")
    if not re.fullmatch(r"[0-9a-f]{40}", recovery_benchmark_commit):
        raise ValueError("Recovery benchmark commit must be a full SHA-1")
    if original_benchmark_commit == recovery_benchmark_commit:
        raise ValueError("Recovery must use a reviewed benchmark-code revision")
    rows, manifest = load_test_manifest(manifest_path.resolve())
    if manifest.get("variant") != "published_pretrained":
        raise ValueError("Recovery is restricted to the published-pretrained run")
    if manifest.get("training_mode") != "published_pretrained":
        raise ValueError("Recovery cannot change the frozen training mode")
    matches = [row for row in rows if row["task_index"] == TASK_INDEX]
    if len(matches) != 1 or matches[0]["relative_path"] != RELATIVE_PATH:
        raise ValueError("Frozen recovery task differs from SCION plot 31")
    row = matches[0]
    run_id = str(manifest["run_id"])
    expected_runtime = runtime_root.resolve() / SAFE_PLOT_ID
    expected_prediction = prediction_root.resolve() / SAFE_PLOT_ID
    expected_metadata = metadata_root.resolve() / f"{SAFE_PLOT_ID}_inference.json"
    expected_evaluation = table_root.resolve() / "per_plot" / SAFE_PLOT_ID
    failures_path = table_root.resolve() / "failures.csv"
    run_summary_path = table_root.resolve() / "run_summary.json"
    for required in (expected_runtime, expected_metadata, expected_evaluation, failures_path, run_summary_path):
        if not required.exists():
            raise FileNotFoundError(required)

    with failures_path.open(encoding="utf-8", newline="") as handle:
        failures = list(csv.DictReader(handle))
    if len(failures) != 1:
        raise ValueError("Recovery requires exactly one documented test failure")
    failure = failures[0]
    if int(failure["task_index"]) != TASK_INDEX or failure["relative_path"] != RELATIVE_PATH:
        raise ValueError("Documented failure differs from the authorized recovery task")
    if failure["status"] != "documented_inference_failure":
        raise ValueError("Recovery requires the documented upstream inference failure")
    original_metadata = json.loads(expected_metadata.read_text(encoding="utf-8"))
    error_text = json.dumps(original_metadata.get("error") or {}, sort_keys=True)
    if "0 sample(s)" not in error_text and "non-zero exit status 1" not in error_text:
        raise ValueError("Inference failure does not match the empty-group execution fault")
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if run_summary.get("benchmark_commit") != original_benchmark_commit:
        raise ValueError("Original summary benchmark commit differs from the state file")
    if run_summary.get("completed_plots") != 10 or run_summary.get("documented_failures") != 1:
        raise ValueError("Recovery requires the accounted 10-of-11 partial result")
    for other in rows:
        if other["task_index"] == TASK_INDEX:
            continue
        metrics = table_root.resolve() / "per_plot" / other["safe_plot_id"] / "metrics.json"
        if not metrics.is_file():
            raise FileNotFoundError(f"Completed plot metric missing: {metrics}")

    final_outputs = list((expected_runtime / "results" / "full_forest").glob("*"))
    final_outputs += list((expected_runtime / "results" / "pointwise_results").glob("*"))
    if any(path.is_file() for path in final_outputs):
        raise ValueError("Failed runtime unexpectedly contains final prediction outputs")
    if output.exists() or archive_root.exists():
        raise FileExistsError("Recovery record or archive already exists")
    archive_root.mkdir(parents=True)
    archived: list[dict[str, Any]] = []
    archived.append(move_to_archive(expected_metadata, archive_root, "failure/inference.json"))
    archived.append(move_to_archive(expected_evaluation, archive_root, "failure/evaluation"))
    if expected_prediction.exists():
        archived.append(move_to_archive(expected_prediction, archive_root, "failure/predictions"))
    for name in AGGREGATES:
        path = table_root.resolve() / name
        if path.exists():
            archived.append(move_to_archive(path, archive_root, f"aggregate/{name}"))
    runtime_inventory = directory_inventory(expected_runtime)
    shutil.rmtree(expected_runtime)

    payload = {
        "schema_version": 1,
        "status": "prepared_single_execution_recovery",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "run_id": run_id,
        "variant": "published_pretrained",
        "training_mode": "published_pretrained",
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "repeat_test_for_setting_selection_permitted": False,
        "model_or_parameter_selection_performed": False,
        "task_index": TASK_INDEX,
        "relative_path": RELATIVE_PATH,
        "safe_plot_id": SAFE_PLOT_ID,
        "input_sha256": row["input_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "original_benchmark_commit": original_benchmark_commit,
        "recovery_benchmark_commit": recovery_benchmark_commit,
        "policy": POLICY,
        "scientific_interpretation": "zero_predicted_instances",
        "original_failure": original_metadata,
        "archived_artifacts": archived,
        "discarded_non_prediction_intermediate_runtime": runtime_inventory,
        "archive_root": str(archive_root),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--table-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-benchmark-commit", required=True)
    parser.add_argument("--recovery-benchmark-commit", required=True)
    args = parser.parse_args()
    payload = prepare(
        args.manifest,
        args.runtime_root,
        args.prediction_root,
        args.metadata_root,
        args.table_root,
        args.archive_root,
        args.output,
        args.original_benchmark_commit,
        args.recovery_benchmark_commit,
    )
    print(f"status={payload['status']}")
    print(f"task_index={payload['task_index']}")
    print(f"plot={payload['relative_path']}")
    print("held_out_test_accessed=true")
    print("model_or_parameter_selection_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
