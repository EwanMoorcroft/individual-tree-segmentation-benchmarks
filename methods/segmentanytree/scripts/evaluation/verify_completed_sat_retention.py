"""Verify retained SegmentAnyTree artefacts needed for future metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PUBLISHED_RUN = "segmentanytree_for-instance_published_pretrained_20260710_231601"
FINETUNED_RUN = "segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931"
HISTORICAL_RUN = "sat_for_quicktune_to49_20260706_140730"
EXPECTED_RELEASED_SHA256 = (
    "0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def relative_entry(path: Path, project_root: Path) -> dict[str, Any]:
    try:
        label = str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        label = str(path.resolve())
    return {"path": label, "size_bytes": path.stat().st_size}


def validate_summary(path: Path, split: str, plots: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row: {path}")
    row = rows[0]
    if row.get("result_status") != "completed_aligned_pointwise_test":
        raise ValueError(f"Summary is not complete: {path}")
    if row.get("dataset_split") != split or int(row.get("plots", 0)) != plots:
        raise ValueError(f"Summary has unexpected split or plot count: {path}")
    if int(row.get("predicted_instances", 0)) <= 0:
        raise ValueError(f"Summary contains zero predictions: {path}")
    return row


def validate_aligned_run(
    project_root: Path,
    label: str,
    predictions: Path,
    run_metadata: Path,
    metrics: Path,
    summary: Path,
    split: str,
    expected_plots: int,
) -> dict[str, Any]:
    instances = sorted(predictions.glob("*/*/Instance_results_forEval_0.ply"))
    semantics = sorted(predictions.glob("*/*/semantic_segmentation_*.ply"))
    metadata_files = sorted(run_metadata.glob("*/*_run.json"))
    metric_files = sorted(metrics.rglob("*.json"))
    observed = (len(instances), len(semantics), len(metadata_files), len(metric_files))
    expected = (expected_plots, expected_plots, expected_plots, expected_plots)
    if observed != expected:
        raise ValueError(f"{label} retained counts {observed}, expected {expected}")

    for path in metadata_files:
        record = read_object(path)
        if record.get("status") != "completed" or record.get("split") != split:
            raise ValueError(f"Incomplete run metadata: {path}")
        for key in ("aligned_instance_evaluation", "aligned_semantic_evaluation"):
            retained = Path(str(record.get(key, "")))
            if not retained.is_file():
                raise ValueError(f"Missing retained aligned output from {path}: {key}")
        if not record.get("aligned_instance_evaluation_sha256"):
            raise ValueError(f"Missing instance SHA-256 in metadata: {path}")
        if not record.get("aligned_semantic_evaluation_sha256"):
            raise ValueError(f"Missing semantic SHA-256 in metadata: {path}")

    summary_row = validate_summary(summary, split, expected_plots)
    return {
        "label": label,
        "split": split,
        "expected_plots": expected_plots,
        "prediction_instance_files": [
            relative_entry(path, project_root) for path in instances
        ],
        "prediction_semantic_files": [
            relative_entry(path, project_root) for path in semantics
        ],
        "run_metadata_files": [
            relative_entry(path, project_root) for path in metadata_files
        ],
        "metric_files": [relative_entry(path, project_root) for path in metric_files],
        "summary": relative_entry(summary, project_root),
        "summary_metrics": {
            key: summary_row[key]
            for key in (
                "mean_plot_f1",
                "micro_f1",
                "predicted_instances",
                "reference_instances",
                "true_positives",
                "false_positives",
                "false_negatives",
            )
        },
    }


def build_manifest(project_root: Path, checkpoint_root: Path) -> dict[str, Any]:
    results = project_root / "results"
    metadata = results / "metadata/segmentanytree_for_instance"
    tables = results / "tables/segmentanytree_for_instance"
    predictions = project_root / "data/predictions/segmentanytree"

    published = validate_aligned_run(
        project_root,
        "published_pretrained_test",
        predictions / f"for_instance_variants/{PUBLISHED_RUN}/held_out_test",
        metadata / f"variant_runs/{PUBLISHED_RUN}/held_out_test",
        metadata / f"variants/{PUBLISHED_RUN}/held_out_test",
        tables / f"variants/{PUBLISHED_RUN}/held_out_test/final_summary.csv",
        "test",
        11,
    )
    fine_validation = validate_aligned_run(
        project_root,
        "fine_tuned_development_validation",
        predictions / f"for_instance_trained_validation/{FINETUNED_RUN}",
        metadata / f"trained_validation_runs/{FINETUNED_RUN}",
        metadata / f"trained_validation/{FINETUNED_RUN}",
        tables / f"trained_validation/{FINETUNED_RUN}/validation_summary.csv",
        "dev",
        5,
    )
    fine_test = validate_aligned_run(
        project_root,
        "fine_tuned_test",
        predictions / f"for_instance_trained_test/{FINETUNED_RUN}",
        metadata / f"trained_test_runs/{FINETUNED_RUN}",
        metadata / f"trained_test/{FINETUNED_RUN}",
        tables / f"trained_test/{FINETUNED_RUN}/final_summary.csv",
        "test",
        11,
    )

    released_checkpoint = (
        checkpoint_root.parent
        / "segmentanytree_pretrained/released_model_bundle/PointGroup-PAPER.pt"
    )
    if sha256(released_checkpoint) != EXPECTED_RELEASED_SHA256:
        raise ValueError("Released checkpoint is missing or has changed")
    fine_checkpoint = checkpoint_root / f"{FINETUNED_RUN}/run/PointGroup-PAPER.pt"
    historical_checkpoint = checkpoint_root / f"{HISTORICAL_RUN}/run/PointGroup-PAPER.pt"
    for path in (fine_checkpoint, historical_checkpoint):
        if not path.is_file():
            raise ValueError(f"Required retained checkpoint is missing: {path}")

    global_prediction_files = sorted(
        path for path in predictions.rglob("*") if path.is_file()
    )
    all_aligned_instances = sorted(
        predictions.rglob("Instance_results_forEval_0.ply")
    )
    all_aligned_semantics = sorted(
        path
        for path in predictions.rglob("semantic_segmentation_*.ply")
        if path.is_file()
    )
    historical_hits = [
        path for path in global_prediction_files if HISTORICAL_RUN in str(path)
    ]
    if not historical_hits:
        raise ValueError("No retained historical retrained predictions were found")

    required_evidence = [
        metadata / f"test_freezes/{PUBLISHED_RUN}.json",
        metadata / f"finetune_freezes/{FINETUNED_RUN}.json",
        metadata / f"finetuned_test_freezes/{FINETUNED_RUN}.json",
        metadata / f"training_runs/{FINETUNED_RUN}.json",
    ]
    for path in required_evidence:
        if not path.is_file():
            raise ValueError(f"Required retained evidence is missing: {path}")

    return {
        "status": "retention-verified",
        "benchmark": "for_instance_segmentanytree",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root.resolve()),
        "canonical_runs": [published, fine_validation, fine_test],
        "checkpoints": [
            {**relative_entry(released_checkpoint, project_root), "sha256": EXPECTED_RELEASED_SHA256},
            {**relative_entry(fine_checkpoint, project_root), "sha256": sha256(fine_checkpoint)},
            {**relative_entry(historical_checkpoint, project_root), "sha256": sha256(historical_checkpoint)},
        ],
        "required_evidence": [
            {**relative_entry(path, project_root), "sha256": sha256(path)}
            for path in required_evidence
        ],
        "historical_retrained_prediction_files": [
            relative_entry(path, project_root) for path in historical_hits
        ],
        "all_aligned_instance_files": [
            relative_entry(path, project_root) for path in all_aligned_instances
        ],
        "all_aligned_semantic_files": [
            relative_entry(path, project_root) for path in all_aligned_semantics
        ],
        "all_segmentanytree_prediction_file_count": len(global_prediction_files),
        "all_segmentanytree_prediction_size_bytes": sum(
            path.stat().st_size for path in global_prediction_files
        ),
        "future_metrics_inputs_retained": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify retained SAT predictions, metrics and checkpoints."
    )
    parser.add_argument("--project-root", default="~/scratch/tree-seg-benchmark")
    parser.add_argument(
        "--checkpoint-root",
        default="~/fastscratch/segmentanytree_for_instance_checkpoints",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    payload = build_manifest(project_root, checkpoint_root)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"retention_manifest={output}")
    print("status=retention-verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
