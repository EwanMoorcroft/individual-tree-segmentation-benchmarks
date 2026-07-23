"""Validate, aggregate, and inventory a complete 21-plot development run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import (  # noqa: E402
    load_and_validate_manifest,
    sha256_file,
)

RETAINED = (
    ("converted_input", "staged_input/input_manifest.json"),
    ("converted_input", "staged_input/points/forestformer3d_development_test.bin"),
    ("converted_input", "staged_input/semantic_mask/reference.bin"),
    ("converted_input", "staged_input/instance_mask/reference.bin"),
    ("converted_input", "staged_input/reference.pkl"),
    ("source_identity_metadata", "staged_input/evaluation_sidecar.npz"),
    ("raw_official_output", "raw/forestformer3d_development_test.ply"),
    ("run_provenance", "raw/model_input_fingerprint.json"),
    ("run_provenance", "raw/effective_predict_audit.json"),
    ("run_provenance", "raw/checkpoint_entrypoint_adapter.json"),
    ("run_provenance", "raw/resource_usage.json"),
    ("harmonised_source_row_predictions", "validation/predictions.npz"),
    ("run_provenance", "validation/validation.json"),
    ("evaluation_table", "evaluation/metrics.json"),
)


def summarise(
    manifest_path: Path,
    run_root: Path,
    output_root: Path,
    *,
    run_id: str,
    benchmark_commit: str,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Refusing existing summary root: {output_root}")
    manifest = load_and_validate_manifest(
        manifest_path,
        expected_split="development",
        allow_held_out_test=False,
    )
    if manifest.get("schema") != "forestformer3d_development_manifest_v1":
        raise ValueError("Manifest is not the ForestFormer3D development preflight")
    rows: list[dict[str, object]] = []
    retention: list[dict[str, object]] = []
    for plot in manifest["plots"]:
        task_root = run_root / "tasks" / plot["safe_plot_id"]
        if not (task_root / "task.complete").is_file():
            raise FileNotFoundError(f"Incomplete development task: {plot['plot_id']}")
        metrics_path = task_root / "evaluation/metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            metrics.get("status") != "completed"
            or metrics.get("split") != "development"
            or metrics.get("held_out_access") is not False
            or metrics.get("relative_path") != plot["relative_path"]
            or metrics.get("benchmark_commit") != benchmark_commit
        ):
            raise ValueError(f"Invalid task metrics: {plot['plot_id']}")
        rows.append(metrics)
        for role, relative in RETAINED:
            path = task_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing retained artifact: {path}")
            retention.append(
                {
                    "logical_role": role,
                    "task_index": plot["task_index"],
                    "plot_id": plot["plot_id"],
                    "relative_path": path.relative_to(run_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_iou = (
        sum(float(row["mean_matched_iou"]) * int(row["true_positives"]) for row in rows)
        / tp
        if tp
        else 0.0
    )
    total_points = sum(int(row["point_count"]) for row in rows)
    summary: dict[str, object] = {
        "schema": "forestformer3d_development_summary_v1",
        "status": "complete_published_pretrained_development_diagnostics",
        "method": "ForestFormer3D",
        "training_mode": "published_pretrained",
        "run_id": run_id,
        "benchmark_commit": benchmark_commit,
        "split": "development",
        "held_out_access": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "aggregation": "sum_counts_then_compute_micro_metrics",
        "plot_count": len(rows),
        "point_count": total_points,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": matched_iou,
        "retained_artifact_count": len(retention),
        "retained_bytes": sum(int(row["size_bytes"]) for row in retention),
        "manifest_sha256": sha256_file(manifest_path),
        "next_gate": "development_only_fine_tuning_design_review",
    }
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "plot_id", "relative_path", "point_count", "reference_instance_count",
        "prediction_instance_count", "true_positives", "false_positives",
        "false_negatives", "precision", "recall", "f1", "mean_matched_iou",
    ]
    with (output_root / "per_plot_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    (output_root / "retention_manifest.json").write_text(
        json.dumps(
            {
                "schema": "forestformer3d_retention_manifest_v1",
                "run_id": run_id,
                "immutable_run_root": True,
                "held_out_access": False,
                "artifact_count": len(retention),
                "artifacts": retention,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_root / "artifact_sha256.txt").open("x", encoding="utf-8") as handle:
        for name in ("summary.json", "per_plot_metrics.csv", "retention_manifest.json"):
            handle.write(f"{sha256_file(output_root / name)}  {name}\n")
    (output_root / "summary.complete").touch(exist_ok=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(summarise(
        args.manifest, args.run_root, args.output_root,
        run_id=args.run_id, benchmark_commit=args.benchmark_commit,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
