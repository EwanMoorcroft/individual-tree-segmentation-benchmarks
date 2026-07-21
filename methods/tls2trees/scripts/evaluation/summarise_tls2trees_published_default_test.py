"""Summarise the immutable TLS2trees published-default FOR-instance test run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
for entry in (ROOT, RUNTIME):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from published_default_test_common import (  # noqa: E402
    TARGETS,
    load_json,
    load_yaml,
    validate_exact_manifest,
    validate_frozen_configuration,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{label} is missing or its SHA-256 changed: {path}")


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(
            project_root.expanduser().resolve()
        ).as_posix()
    except ValueError as exc:
        raise ValueError(f"Retained prediction is outside the project root: {path}") from exc


def aggregate(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target"] == target]
    tp = sum(int(row["true_positives"]) for row in selected)
    fp = sum(int(row["false_positives"]) for row in selected)
    fn = sum(int(row["false_negatives"]) for row in selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_collection: dict[str, list[float]] = {}
    for row in selected:
        by_collection.setdefault(row["collection"], []).append(float(row["f1"]))
    return {
        "target": target,
        "configuration_id": "published_default",
        "expected_plot_count": 11,
        "evaluated_plot_count": len(selected),
        "failed_or_invalid_plot_count": 11 - len(selected),
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in selected
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in selected
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": sum(float(row["f1"]) for row in selected) / len(selected),
        "mean_collection_f1": {
            key: sum(values) / len(values)
            for key, values in sorted(by_collection.items())
        },
        "oversegmented_reference_count": sum(
            int(row["oversegmented_reference_count"]) for row in selected
        ),
        "undersegmented_prediction_count": sum(
            int(row["undersegmented_prediction_count"]) for row in selected
        ),
    }


def summarise(
    *,
    project_root: Path,
    output_root: Path,
    run_id: str,
    manifest_path: Path,
    manifest_sha256: str,
    workflow_config_path: Path,
    workflow_config_sha256: str,
    published_config_path: Path,
    published_config_sha256: str,
    benchmark_config_path: Path,
    benchmark_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = project_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    workflow_config_path = workflow_config_path.expanduser().resolve()
    published_config_path = published_config_path.expanduser().resolve()
    benchmark_config_path = benchmark_config_path.expanduser().resolve()
    require_hash(manifest_path, manifest_sha256, "test manifest")
    require_hash(workflow_config_path, workflow_config_sha256, "workflow config")
    require_hash(published_config_path, published_config_sha256, "published config")
    require_hash(benchmark_config_path, benchmark_config_sha256, "benchmark config")
    workflow, _, published, _ = validate_frozen_configuration(
        workflow_config_path, published_config_path
    )
    manifest, _ = load_json(manifest_path)
    plots = validate_exact_manifest(manifest, workflow)
    benchmark, _ = load_yaml(benchmark_config_path)
    evaluator = benchmark.get("evaluation", {}).get("protocol")
    evaluation_mask = benchmark.get("evaluation", {}).get("primary_mask")
    if evaluator != "for_instance_tls2trees_source_row_class3_ignore":
        raise ValueError("Benchmark config does not select the neutral source-row protocol")
    if evaluation_mask != (
        "union_of_reference_target_and_predicted_target_points_"
        "excluding_class3_outpoints"
    ):
        raise ValueError("Benchmark config does not select the class-3-ignore mask")

    rows: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    semantic_reused_count = 0
    runtime_root = (
        output_root.expanduser().resolve()
        / "tls2trees"
        / "for_instance"
        / "published_default"
        / "test"
        / run_id
    )
    for plot in plots:
        plot_root = runtime_root / plot["safe_plot_id"]
        instance_path = plot_root / "metadata" / "instance_run.json"
        adapter_path = plot_root / "metadata" / "adapter_run.json"
        semantic_path = plot_root / "metadata" / "semantic_run.json"
        cache_path = plot_root / "metadata" / "semantic_cache_reuse.json"
        instance, _ = load_json(instance_path)
        adapter, _ = load_json(adapter_path)
        semantic, _ = load_json(semantic_path)
        expected_common = {
            "split": "test",
            "task_index": int(plot["task_index"]),
            "safe_plot_id": plot["safe_plot_id"],
            "relative_path": plot["relative_path"],
        }
        for label, payload in (
            ("instance", instance),
            ("adapter", adapter),
            ("semantic", semantic),
        ):
            for key, expected in expected_common.items():
                if payload.get(key) != expected:
                    raise ValueError(
                        f"Published-default {label} provenance mismatch for {key}: "
                        f"{plot_root}"
                    )
        if (
            instance.get("status") not in {"completed", "completed_no_predictions"}
            or instance.get("variant") != "published_default"
            or instance.get("config_sha256") != published_config_sha256
            or instance.get("held_out_test_accessed") is not True
            or instance.get("resolved_instance_parameters")
            != {
                "n_tiles": published["instance_parameters"]["n_tiles"],
                "n_zeros": published["instance_parameters"]["n_zeros"],
                "overlap": published["instance_parameters"]["overlap"],
                "slice_thickness": published["instance_parameters"]["slice_thickness_m"],
                "find_stems_boundary": published["instance_parameters"]["find_stems_boundary_m"],
                "find_stems_min_radius": published["instance_parameters"]["find_stems_min_radius_m"],
                "find_stems_min_points": published["instance_parameters"]["find_stems_min_points"],
                "graph_edge_length": published["instance_parameters"]["graph_edge_length_m"],
                "graph_maximum_cumulative_gap": published["instance_parameters"]["graph_maximum_cumulative_gap_m"],
                "min_points_per_tree": published["instance_parameters"]["min_points_per_tree"],
                "add_leaves": published["instance_parameters"]["add_leaves"],
                "add_leaves_voxel_length": published["instance_parameters"]["add_leaves_voxel_length_m"],
                "add_leaves_edge_length": published["instance_parameters"]["add_leaves_edge_length_m"],
                "save_diameter_class": published["instance_parameters"]["save_diameter_class"],
                "ignore_missing_tiles": published["instance_parameters"]["ignore_missing_tiles"],
                "pandarallel": published["instance_parameters"]["pandarallel"],
                "verbose": published["instance_parameters"]["verbose"],
            }
        ):
            raise ValueError(f"Published-default instance contract mismatch: {plot_root}")
        if (
            adapter.get("status") != "completed"
            or adapter.get("variant") != "published_default"
            or adapter.get("held_out_test_accessed") is not True
        ):
            raise ValueError(f"Published-default adapter contract mismatch: {plot_root}")
        if cache_path.is_file():
            cache, _ = load_json(cache_path)
            if (
                cache.get("status") != "semantic_cache_reused"
                or cache.get("manifest_sha256") != manifest_sha256
                or cache.get("input_las_sha256") != plot["input_sha256"]
                or cache.get("published_config_sha256") != published_config_sha256
                or cache.get("bundled_model_sha256")
                != published["method"]["bundled_fsct_model"]["sha256"]
                or cache.get("inference_rerun") is not False
            ):
                raise ValueError(f"Semantic-cache evidence mismatch: {cache_path}")
            semantic_reused_count += 1
        elif (
            semantic.get("status") != "completed"
            or semantic.get("variant") != "published_default"
            or semantic.get("config_sha256") != published_config_sha256
            or semantic.get("held_out_test_accessed") is not True
        ):
            raise ValueError(f"Dedicated semantic provenance mismatch: {semantic_path}")

        for target in TARGETS:
            aligned = (
                plot_root
                / "predictions"
                / "aligned"
                / target
                / "source_row_predictions.npz"
            )
            alignment = aligned.with_name("alignment_metadata.json")
            metric_path = plot_root / "evaluation" / target / "plot_metrics.json"
            metrics, _ = load_json(metric_path)
            if (
                metrics.get("split") != "test"
                or metrics.get("target") != target
                or metrics.get("plot_id") != plot["safe_plot_id"]
                or metrics.get("relative_path") != plot["relative_path"]
                or metrics.get("evaluator") != evaluator
                or metrics.get("evaluation_mask") != evaluation_mask
                or metrics.get("semantic_ignore", {}).get(
                    "ignored_semantic_classes"
                )
                != [3]
                or metrics.get("status") != "evaluated"
                or metrics.get("safe_for_scoring") is not True
            ):
                raise ValueError(f"Published-default metric provenance mismatch: {metric_path}")
            if not aligned.is_file() or not alignment.is_file():
                raise FileNotFoundError(f"Aligned prediction evidence is missing: {aligned}")
            if int(metrics["prediction_instance_count"]) == 0 and (
                int(metrics["true_positives"]) != 0
                or int(metrics["false_positives"]) != 0
                or float(metrics["precision"]) != 0.0
                or float(metrics["recall"]) != 0.0
                or float(metrics["f1"]) != 0.0
            ):
                raise ValueError("Empty predictions must be evaluated as a valid zero")
            row = {
                "target": target,
                "configuration_id": "published_default",
                "task_index": int(plot["task_index"]),
                "collection": plot["collection"],
                "safe_plot_id": plot["safe_plot_id"],
                "relative_path": plot["relative_path"],
                "status": metrics["status"],
                "safe_for_scoring": metrics["safe_for_scoring"],
                "raw_prediction_instance_count": metrics["semantic_ignore"][
                    "raw_prediction_instance_count"
                ],
                "prediction_instance_count": metrics["prediction_instance_count"],
                "reference_instance_count": metrics["reference_instance_count"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_matched_iou": metrics["mean_matched_iou"],
                "oversegmented_reference_count": metrics[
                    "oversegmented_reference_count"
                ],
                "undersegmented_prediction_count": metrics[
                    "undersegmented_prediction_count"
                ],
                "semantic_cache_reused": cache_path.is_file(),
                "instance_runtime_seconds": instance.get("runtime_seconds"),
                "instance_peak_rss_gb": instance.get("peak_rss_gb"),
                "adapter_runtime_seconds": adapter.get("runtime_seconds"),
                "metrics_path": str(metric_path),
                "metrics_sha256": sha256(metric_path),
                "prediction_path": str(aligned),
                "prediction_sha256": sha256(aligned),
                "alignment_metadata_sha256": sha256(alignment),
            }
            rows.append(row)
            retained.append(
                {
                    "target": target,
                    "configuration_id": "published_default",
                    "plot_index": int(plot["task_index"]),
                    "plot_id": plot["safe_plot_id"],
                    "relative_path": relative_to_project(aligned, project_root),
                    "format": "npz",
                    "point_correspondence": "source_row_index",
                    "sha256": row["prediction_sha256"],
                    "size_bytes": aligned.stat().st_size,
                    "alignment_metadata_relative_path": relative_to_project(
                        alignment, project_root
                    ),
                    "alignment_metadata_sha256": row[
                        "alignment_metadata_sha256"
                    ],
                }
            )
    if len(rows) != 22 or len(retained) != 22:
        raise ValueError("Published-default summary requires exactly 22 valid metrics")
    aggregates = [aggregate(rows, target) for target in TARGETS]
    retention = {
        "schema_version": 1,
        "status": "retention_verified",
        "dataset": "FOR-instance",
        "dataset_split": "test",
        "method": "TLS2trees",
        "variant": "published_default",
        "run_id": run_id,
        "expected_files": 22,
        "verified_prediction_files": len(retained),
        "verified_prediction_size_bytes": sum(
            int(record["size_bytes"]) for record in retained
        ),
        "manifest_sha256": manifest_sha256,
        "workflow_config_sha256": workflow_config_sha256,
        "published_config_sha256": published_config_sha256,
        "benchmark_config_sha256": benchmark_config_sha256,
        "configuration_changed_after_test": False,
        "files": retained,
    }
    summary = {
        "schema_version": 1,
        "status": "published_default_test_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "published_default",
        "split": "test",
        "workflow_run_id": run_id,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "workflow_config": str(workflow_config_path),
        "workflow_config_sha256": workflow_config_sha256,
        "published_config": str(published_config_path),
        "published_config_sha256": published_config_sha256,
        "benchmark_config": str(benchmark_config_path),
        "benchmark_config_sha256": benchmark_config_sha256,
        "expected_plot_count": 11,
        "expected_metric_count": 22,
        "valid_metric_count": 22,
        "semantic_cache_reused_plot_count": semantic_reused_count,
        "dedicated_semantic_plot_count": 11 - semantic_reused_count,
        "held_out_test_accessed": True,
        "held_out_accuracy_metrics_computed": True,
        "configuration_selected_from_for_instance_metrics": False,
        "configuration_changed_after_test": False,
        "plot_metrics": rows,
        "aggregates": aggregates,
    }
    return summary, retention


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--workflow-config-sha256", required=True)
    parser.add_argument("--published-config", required=True)
    parser.add_argument("--published-config-sha256", required=True)
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--benchmark-config-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--plot-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--retention-json", required=True)
    args = parser.parse_args()
    output_paths = [
        Path(args.output_json),
        Path(args.plot_csv),
        Path(args.aggregate_csv),
        Path(args.retention_json),
    ]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Published-default summary output already exists")
    summary, retention = summarise(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root),
        run_id=args.run_id,
        manifest_path=Path(args.manifest_json),
        manifest_sha256=args.manifest_sha256,
        workflow_config_path=Path(args.workflow_config),
        workflow_config_sha256=args.workflow_config_sha256,
        published_config_path=Path(args.published_config),
        published_config_sha256=args.published_config_sha256,
        benchmark_config_path=Path(args.benchmark_config),
        benchmark_config_sha256=args.benchmark_config_sha256,
    )
    retention_path = Path(args.retention_json)
    retention_path.parent.mkdir(parents=True, exist_ok=True)
    retention_text = json.dumps(retention, indent=2, sort_keys=True) + "\n"
    retention_path.write_text(retention_text, encoding="utf-8")
    summary["retention_manifest"] = str(retention_path.resolve())
    summary["retention_manifest_sha256"] = hashlib.sha256(
        retention_text.encode("utf-8")
    ).hexdigest()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(Path(args.plot_csv), summary["plot_metrics"])
    write_csv(Path(args.aggregate_csv), summary["aggregates"])
    print("status=published_default_test_completed")
    print("valid_metrics=22/22")
    for row in summary["aggregates"]:
        print(
            f"{row['target']}: micro_f1={row['micro_f1']:.6f} "
            f"precision={row['precision']:.6f} recall={row['recall']:.6f} "
            f"predictions={row['prediction_instance_count']} "
            f"references={row['reference_instance_count']}"
        )
    print(f"semantic_cache_reused_plots={summary['semantic_cache_reused_plot_count']}")
    print(f"dedicated_semantic_plots={summary['dedicated_semantic_plot_count']}")
    print("configuration_changed_after_test=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
