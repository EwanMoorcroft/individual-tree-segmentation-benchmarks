"""Validate and summarise all 21 ForAINet development diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_PLOTS = 21
EXPECTED_PROTOCOL = "for_instance_pointwise_v1"
EXPECTED_RETENTION_ROLES = {
    "aligned_prediction",
    "aligned_prediction_metadata",
    "checkpoint_provenance",
    "environment_manifest",
    "input_conversion",
    "label_independence_probe",
    "matched_pairs",
    "merge_alignment",
    "official_raw_output",
    "plot_metadata",
    "plot_metrics",
    "raw_output_inventory",
    "unmatched_predictions",
    "unmatched_references",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_retention(plot_root: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema") != "forainet_retention_manifest_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"invalid plot retention manifest: {plot_root}")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 14:
        raise ValueError(f"incomplete plot retention manifest: {plot_root}")
    roles = set()
    for row in files:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe retained path: {relative}")
        path = plot_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise ValueError(f"retained file changed: {path}")
        roles.add(str(row["role"]))
    if roles != EXPECTED_RETENTION_ROLES or len(roles) != len(files):
        raise ValueError(f"retention roles differ: {plot_root}")


def metric_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = metric_ratio(tp, tp + fp)
    recall = metric_ratio(tp, tp + fn)
    f1 = metric_ratio(2 * precision * recall, precision + recall)
    return {
        "plot_count": len(rows),
        "evaluated_point_count": sum(
            int(row["evaluated_point_count"]) for row in rows
        ),
        "reference_instance_count": sum(
            int(row["reference_instances"]) for row in rows
        ),
        "prediction_instance_count": sum(
            int(row["prediction_instances"]) for row in rows
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_plot_precision": statistics.fmean(
            float(row["precision"]) for row in rows
        ),
        "mean_plot_recall": statistics.fmean(float(row["recall"]) for row in rows),
        "mean_plot_f1": statistics.fmean(float(row["f1"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(
    run_root: Path,
    run_id: str,
    manifest_csv: Path,
    manifest_json: Path,
    benchmark_commit: str,
    recovery_root: Path | None = None,
    recovery_benchmark_commit: str | None = None,
) -> dict[str, Any]:
    if (run_root / "final_gate.json").exists():
        raise FileExistsError("development final gate already exists")
    manifest_payload = load_json(manifest_json)
    if (
        manifest_payload.get("schema") != "forainet_development_manifest_v1"
        or manifest_payload.get("status") != "complete"
        or manifest_payload.get("expected_plot_count") != EXPECTED_PLOTS
        or manifest_payload.get("held_out_paths_included") is not False
    ):
        raise ValueError("development manifest is not frozen and complete")
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != EXPECTED_PLOTS:
        raise ValueError("development CSV must contain exactly 21 plots")

    plot_rows = []
    child_manifests = []
    recovered_task_indices = []
    for manifest_row in manifest_rows:
        task_index = int(manifest_row["task_index"])
        relative_path = manifest_row["relative_path"]
        canonical_plot_root = run_root / "plots" / f"task_{task_index:03d}"
        recovery_plot_root = (
            recovery_root / "plots" / f"task_{task_index:03d}"
            if recovery_root is not None
            else None
        )
        plot_root = canonical_plot_root
        result_source = "original"
        expected_plot_commit = benchmark_commit
        if not (canonical_plot_root / "final_gate.json").is_file():
            if (
                recovery_plot_root is None
                or recovery_benchmark_commit is None
                or not (recovery_plot_root / "final_gate.json").is_file()
            ):
                raise FileNotFoundError(canonical_plot_root / "final_gate.json")
            plot_root = recovery_plot_root
            result_source = "recovery"
            expected_plot_commit = recovery_benchmark_commit
            recovered_task_indices.append(task_index)
        gate_path = plot_root / "final_gate.json"
        metrics_path = plot_root / "evaluation" / "metrics.json"
        plot_metadata_path = plot_root / "metadata" / "plot.json"
        retention_path = plot_root / "retention" / "manifest.json"
        for path in (gate_path, metrics_path, plot_metadata_path, retention_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        gate = load_json(gate_path)
        metrics = load_json(metrics_path)
        plot_metadata = load_json(plot_metadata_path)
        retention = load_json(retention_path)
        validate_retention(plot_root, retention)
        if (
            gate.get("schema") != "forainet_development_plot_final_gate_v1"
            or gate.get("status") != "complete"
            or gate.get("held_out_access") is not False
            or gate.get("relative_path") != relative_path
            or gate.get("development_task_index") != task_index
            or gate.get("retention_manifest_sha256") != sha256(retention_path)
        ):
            raise ValueError(f"invalid development plot gate: {relative_path}")
        if (
            metrics.get("protocol_id") != EXPECTED_PROTOCOL
            or metrics.get("split") != "dev"
            or metrics.get("coordinate_matching") is not False
            or plot_metadata.get("route") != "development"
            or plot_metadata.get("benchmark_commit") != expected_plot_commit
            or plot_metadata.get("relative_path") != relative_path
            or plot_metadata.get("reference_labels_supplied_to_model") is not False
        ):
            raise ValueError(f"plot protocol or provenance differs: {relative_path}")
        plot_rows.append(
            {
                "task_index": task_index,
                "result_source": result_source,
                "benchmark_commit": expected_plot_commit,
                "relative_path": relative_path,
                "site": relative_path.split("/", 1)[0],
                "point_count": int(plot_metadata["point_count"]),
                "evaluated_point_count": int(metrics["evaluated_point_count"]),
                "reference_instances": int(metrics["reference_instance_count"]),
                "prediction_instances": int(metrics["prediction_instance_count"]),
                "true_positives": int(metrics["true_positives"]),
                "false_positives": int(metrics["false_positives"]),
                "false_negatives": int(metrics["false_negatives"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "wall_runtime_seconds": float(
                    plot_metadata["wall_runtime_seconds"]
                ),
                "peak_child_rss_kb": int(plot_metadata["peak_child_rss_kb"]),
                "aligned_prediction_sha256": plot_metadata[
                    "aligned_prediction_sha256"
                ],
                "metrics_sha256": sha256(metrics_path),
                "retention_manifest_sha256": sha256(retention_path),
            }
        )
        child_manifests.append(
            {
                "task_index": task_index,
                "relative_path": relative_path,
                "result_source": result_source,
                "retention_manifest_sha256": sha256(retention_path),
            }
        )
    plot_rows.sort(key=lambda row: int(row["task_index"]))

    site_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        site_groups[str(row["site"])].append(row)
    site_rows = []
    for site in sorted(site_groups):
        values = aggregate(site_groups[site])
        site_rows.append({"site": site, **values})

    summary_root = run_root / "summary"
    retention_root = run_root / "retention"
    summary_root.mkdir(parents=True, exist_ok=False)
    retention_root.mkdir(parents=True, exist_ok=False)
    plot_csv = summary_root / "plots.csv"
    site_csv = summary_root / "sites.csv"
    summary_json = summary_root / "metrics.json"
    write_csv(plot_csv, plot_rows)
    write_csv(site_csv, site_rows)
    overall = aggregate(plot_rows)
    summary_payload = {
        "schema": "forainet_development_summary_v1",
        "status": "complete",
        "run_id": run_id,
        "variant": "published_pretrained",
        "split": "dev",
        "protocol_id": EXPECTED_PROTOCOL,
        "benchmark_commit": benchmark_commit,
        "recovery_benchmark_commit": recovery_benchmark_commit,
        "recovered_task_indices": recovered_task_indices,
        "implementation_commits": sorted(
            {str(row["benchmark_commit"]) for row in plot_rows}
        ),
        "expected_plots": EXPECTED_PLOTS,
        "completed_plots": len(plot_rows),
        "held_out_access": False,
        "total_point_count": sum(int(row["point_count"]) for row in plot_rows),
        "total_reference_instances": sum(
            int(row["reference_instances"]) for row in plot_rows
        ),
        "total_prediction_instances": sum(
            int(row["prediction_instances"]) for row in plot_rows
        ),
        "overall": overall,
        "total_wall_runtime_seconds": sum(
            float(row["wall_runtime_seconds"]) for row in plot_rows
        ),
        "maximum_peak_child_rss_kb": max(
            int(row["peak_child_rss_kb"]) for row in plot_rows
        ),
        "next_gate": "freeze_checkpoint_initialised_fine_tuning_plan",
    }
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    retention_payload = {
        "schema": "forainet_development_retention_v1",
        "status": "complete",
        "run_id": run_id,
        "held_out_access": False,
        "recovered_task_indices": recovered_task_indices,
        "development_manifest_csv_sha256": sha256(manifest_csv),
        "development_manifest_json_sha256": sha256(manifest_json),
        "plot_summary_sha256": sha256(plot_csv),
        "site_summary_sha256": sha256(site_csv),
        "metrics_summary_sha256": sha256(summary_json),
        "child_manifests": child_manifests,
    }
    retention_path = retention_root / "manifest.json"
    retention_path.write_text(
        json.dumps(retention_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_gate = {
        "schema": "forainet_development_final_gate_v1",
        "status": "complete",
        "run_id": run_id,
        "variant": "published_pretrained",
        "expected_plots": EXPECTED_PLOTS,
        "completed_plots": len(plot_rows),
        "protocol_id": EXPECTED_PROTOCOL,
        "benchmark_commit": benchmark_commit,
        "recovery_benchmark_commit": recovery_benchmark_commit,
        "recovered_task_indices": recovered_task_indices,
        "held_out_access": False,
        "summary_metrics_sha256": sha256(summary_json),
        "retention_manifest_sha256": sha256(retention_path),
        "next_gate": "freeze_checkpoint_initialised_fine_tuning_plan",
    }
    (run_root / "final_gate.json").write_text(
        json.dumps(final_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--recovery-benchmark-commit")
    args = parser.parse_args()
    if (args.recovery_root is None) != (
        args.recovery_benchmark_commit is None
    ):
        raise ValueError(
            "recovery root and recovery benchmark commit must be supplied together"
        )
    payload = summarise(
        args.run_root,
        args.run_id,
        args.manifest_csv,
        args.manifest_json,
        args.benchmark_commit,
        args.recovery_root,
        args.recovery_benchmark_commit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
