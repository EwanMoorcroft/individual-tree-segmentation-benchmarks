"""Independently verify a completed ForestFormer3D development run.

The verifier reads only an immutable development run root.  It re-hashes every
retained artefact, reconciles task metadata and aggregate metrics, and checks
the exact source-row identity arrays stored in all harmonised predictions.
Verification evidence is written to a separate, non-existing output root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shared.for_instance_manifest import sha256_file


EXPECTED_TASK_COUNT = 21
EXPECTED_RETAINED_PER_TASK = 14
REQUIRED_PREDICTION_FIELDS = {
    "classification",
    "pred_classification",
    "pred_tree_id",
    "source_row_index",
    "target_tree_id",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _require_development_record(
    payload: dict[str, Any],
    *,
    path: Path,
    status: str,
) -> None:
    if (
        payload.get("status") != status
        or payload.get("split") != "development"
        or payload.get("held_out_access") is not False
    ):
        raise ValueError(f"Invalid development-only record: {path}")


def _read_summary_hashes(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name in rows or "/" in name or "\\" in name:
            raise ValueError(f"Invalid summary hash entry: {line!r}")
        rows[name] = digest
    expected = {"summary.json", "per_plot_metrics.csv", "retention_manifest.json"}
    if set(rows) != expected:
        raise ValueError("Summary hash inventory does not name the three outputs")
    return rows


def verify(run_root: Path, output_root: Path) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(run_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing existing verification root: {output_root}")
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise ValueError("Verification output must be outside the immutable run root")

    summary_root = run_root / "summary"
    summary = _load_json(summary_root / "summary.json")
    if (
        summary.get("schema") != "forestformer3d_development_summary_v1"
        or summary.get("status")
        != "complete_published_pretrained_development_diagnostics"
        or summary.get("split") != "development"
        or summary.get("held_out_access") is not False
        or summary.get("plot_count") != EXPECTED_TASK_COUNT
    ):
        raise ValueError("Invalid ForestFormer3D development summary")
    if not (summary_root / "summary.complete").is_file():
        raise FileNotFoundError("Missing summary.complete")
    if not (run_root / "development.complete").is_file():
        raise FileNotFoundError("Missing development.complete")

    summary_hashes = _read_summary_hashes(summary_root / "artifact_sha256.txt")
    for name, expected in summary_hashes.items():
        if sha256_file(summary_root / name) != expected:
            raise ValueError(f"Summary artifact hash mismatch: {name}")

    with (summary_root / "per_plot_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != EXPECTED_TASK_COUNT:
        raise ValueError("Per-plot CSV does not contain exactly 21 rows")
    csv_by_plot = {row["plot_id"]: row for row in csv_rows}
    if len(csv_by_plot) != EXPECTED_TASK_COUNT:
        raise ValueError("Per-plot CSV contains duplicate plot identifiers")

    task_roots = sorted(path.parent for path in run_root.glob("tasks/*/task.complete"))
    if len(task_roots) != EXPECTED_TASK_COUNT:
        raise ValueError("Run does not contain exactly 21 completed task roots")
    if list(run_root.glob("tasks/*/task.failed")):
        raise ValueError("Run contains a task.failed marker")

    point_count = true_positives = false_positives = false_negatives = 0
    verified_prediction_bytes = 0
    task_checks: list[dict[str, Any]] = []
    for task_root in task_roots:
        input_path = task_root / "staged_input/input_manifest.json"
        validation_path = task_root / "validation/validation.json"
        metrics_path = task_root / "evaluation/metrics.json"
        prediction_path = task_root / "validation/predictions.npz"
        input_manifest = _load_json(input_path)
        validation = _load_json(validation_path)
        metrics = _load_json(metrics_path)
        _require_development_record(validation, path=validation_path, status="passed")
        _require_development_record(metrics, path=metrics_path, status="completed")
        if (
            input_manifest.get("split") != "development"
            or input_manifest.get("held_out_access") is not False
            or input_manifest.get("source_row_index") != "zero_based_identity"
        ):
            raise ValueError(f"Invalid development input manifest: {input_path}")

        plot_id = input_manifest.get("plot_id")
        if plot_id not in csv_by_plot:
            raise ValueError(f"Task plot missing from per-plot CSV: {plot_id}")
        expected_points = int(input_manifest["point_count"])
        if not (
            validation.get("plot_id") == metrics.get("plot_id") == plot_id
            and validation.get("relative_path")
            == metrics.get("relative_path")
            == input_manifest.get("relative_path")
            and int(validation["point_count"])
            == int(metrics["point_count"])
            == expected_points
            and validation.get("exact_row_alignment") is True
        ):
            raise ValueError(f"Task identity/alignment mismatch: {plot_id}")

        prediction_sha256 = sha256_file(prediction_path)
        if not (
            validation.get("prediction_npz_sha256")
            == metrics.get("prediction_npz_sha256")
            == prediction_sha256
        ):
            raise ValueError(f"Prediction hash mismatch: {plot_id}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            if set(arrays.files) != REQUIRED_PREDICTION_FIELDS:
                raise ValueError(f"Prediction fields mismatch: {plot_id}")
            if any(len(arrays[name]) != expected_points for name in arrays.files):
                raise ValueError(f"Prediction length mismatch: {plot_id}")
            if not np.array_equal(
                arrays["source_row_index"],
                np.arange(expected_points, dtype=arrays["source_row_index"].dtype),
            ):
                raise ValueError(f"source_row_index is not identity: {plot_id}")
        verified_prediction_bytes += prediction_path.stat().st_size

        row = csv_by_plot[plot_id]
        fields = {
            "point_count": expected_points,
            "true_positives": int(metrics["true_positives"]),
            "false_positives": int(metrics["false_positives"]),
            "false_negatives": int(metrics["false_negatives"]),
        }
        if any(int(row[name]) != value for name, value in fields.items()):
            raise ValueError(f"Per-plot CSV differs from metrics JSON: {plot_id}")
        point_count += fields["point_count"]
        true_positives += fields["true_positives"]
        false_positives += fields["false_positives"]
        false_negatives += fields["false_negatives"]
        task_checks.append(
            {
                "plot_id": plot_id,
                "point_count": expected_points,
                "prediction_npz_sha256": prediction_sha256,
                "source_row_identity": True,
            }
        )

    aggregates = {
        "point_count": point_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    if any(int(summary[name]) != value for name, value in aggregates.items()):
        raise ValueError("Summary counts do not reconcile with per-task metrics")

    retention_path = summary_root / "retention_manifest.json"
    retention = _load_json(retention_path)
    artifacts = retention.get("artifacts")
    expected_artifact_count = EXPECTED_TASK_COUNT * EXPECTED_RETAINED_PER_TASK
    if (
        retention.get("schema") != "forestformer3d_retention_manifest_v1"
        or retention.get("run_id") != summary.get("run_id")
        or retention.get("immutable_run_root") is not True
        or retention.get("held_out_access") is not False
        or retention.get("artifact_count") != expected_artifact_count
        or not isinstance(artifacts, list)
        or len(artifacts) != expected_artifact_count
    ):
        raise ValueError("Invalid retention manifest")
    relative_paths = [row["relative_path"] for row in artifacts]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Retention manifest contains duplicate artifact paths")

    retained_bytes = 0
    for row in artifacts:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe retained path: {relative}")
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing retained regular file: {relative}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Retained size mismatch: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Retained SHA-256 mismatch: {relative}")
        retained_bytes += path.stat().st_size
    if (
        retained_bytes != int(summary["retained_bytes"])
        or len(artifacts) != int(summary["retained_artifact_count"])
    ):
        raise ValueError("Retention totals do not reconcile with summary")

    result: dict[str, Any] = {
        "schema": "forestformer3d_development_verification_v1",
        "status": "verified",
        "run_id": summary["run_id"],
        "benchmark_commit": summary["benchmark_commit"],
        "split": "development",
        "held_out_access": False,
        "task_count": len(task_checks),
        "task_failed_count": 0,
        "exact_source_row_alignment": True,
        "summary_hashes_verified": len(summary_hashes),
        "retained_artifact_count": len(artifacts),
        "retained_bytes": retained_bytes,
        "verified_prediction_bytes": verified_prediction_bytes,
        "aggregates": aggregates,
        "tasks": task_checks,
    }
    output_root.mkdir(parents=True)
    result_path = output_root / "verification.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "verification.sha256").write_text(
        f"{sha256_file(result_path)}  verification.json\n", encoding="utf-8"
    )
    (output_root / "verification.complete").touch(exist_ok=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.run_root, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
