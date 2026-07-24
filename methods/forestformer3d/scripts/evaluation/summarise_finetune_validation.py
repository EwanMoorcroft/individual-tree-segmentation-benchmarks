"""Aggregate the frozen ForestFormer3D validation matrix and select a checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import sha256_file  # noqa: E402

EPOCHS = (7, 14, 21, 28, 35)
RETAINED = (
    "staged_input/input_manifest.json",
    "raw/forestformer3d_development_test.ply",
    "raw/model_input_fingerprint.json",
    "raw/effective_predict_audit.json",
    "raw/checkpoint_entrypoint_adapter.json",
    "raw/resource_usage.json",
    "validation/predictions.npz",
    "validation/validation.json",
    "evaluation/metrics.json",
)


def _micro(rows: list[dict[str, object]]) -> dict[str, float | int]:
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarise(
    run_root: Path,
    validation_root: Path,
    source_run_root: Path,
    output_root: Path,
    *,
    benchmark_commit: str,
) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    validation_root = validation_root.expanduser().resolve()
    source_run_root = source_run_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)

    freeze_path = run_root / "fine_tune_freeze.json"
    inventory_path = run_root / "checkpoint_inventory.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        freeze.get("split", {}).get("held_out_access") is not False
        or inventory.get("held_out_access") is not False
        or inventory.get("epochs") != list(EPOCHS)
    ):
        raise ValueError("Fine-tune evidence is not frozen development-only evidence")

    with (run_root / "fine_tune_split.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        plots = [
            row
            for row in csv.DictReader(handle)
            if row["fine_tune_role"] == "validation"
        ]
    if len(plots) != 5:
        raise ValueError("Expected five frozen validation plots")

    checkpoint_rows: list[dict[str, object]] = []
    per_plot: list[dict[str, object]] = []
    retained: list[dict[str, object]] = []
    for epoch in EPOCHS:
        rows: list[dict[str, object]] = []
        for plot in plots:
            task_key = f"epoch_{epoch:02d}__{plot['safe_plot_id']}"
            task_root = validation_root / "tasks" / task_key
            if not (task_root / "task.complete").is_file():
                raise FileNotFoundError(f"Incomplete validation task: {task_key}")
            metrics = json.loads(
                (task_root / "evaluation/metrics.json").read_text(encoding="utf-8")
            )
            if (
                metrics.get("status") != "completed"
                or metrics.get("training_mode") != "fine_tuned_on_dev"
                or metrics.get("split") != "development"
                or metrics.get("held_out_access") is not False
                or metrics.get("relative_path") != plot["relative_path"]
                or metrics.get("benchmark_commit") != benchmark_commit
            ):
                raise ValueError(f"Invalid validation metrics: {task_key}")
            rows.append(metrics)
            per_plot.append({"checkpoint_epoch": epoch, **metrics})
            for relative in RETAINED:
                path = task_root / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                retained.append(
                    {
                        "checkpoint_epoch": epoch,
                        "plot_id": plot["plot_id"],
                        "relative_path": path.relative_to(validation_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        aggregate = _micro(rows)
        checkpoint_rows.append(
            {
                "checkpoint_epoch": epoch,
                "mean_plot_f1": sum(float(row["f1"]) for row in rows) / len(rows),
                "micro_f1": aggregate["f1"],
                **aggregate,
            }
        )

    baseline_rows: list[dict[str, object]] = []
    for plot in plots:
        path = (
            source_run_root
            / "tasks"
            / plot["safe_plot_id"]
            / "evaluation/metrics.json"
        )
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if (
            metrics.get("training_mode") != "published_pretrained"
            or metrics.get("held_out_access") is not False
            or metrics.get("relative_path") != plot["relative_path"]
        ):
            raise ValueError(f"Invalid matched baseline: {plot['plot_id']}")
        baseline_rows.append(metrics)
    baseline_micro = _micro(baseline_rows)
    baseline = {
        "mean_plot_f1": (
            sum(float(row["f1"]) for row in baseline_rows) / len(baseline_rows)
        ),
        "micro_f1": baseline_micro["f1"],
        **baseline_micro,
    }

    selected = sorted(
        checkpoint_rows,
        key=lambda row: (
            -float(row["mean_plot_f1"]),
            -float(row["micro_f1"]),
            int(row["checkpoint_epoch"]),
        ),
    )[0]
    inventory_by_epoch = {
        int(row["epoch"]): row for row in inventory["checkpoints"]
    }
    selected_entry = inventory_by_epoch[int(selected["checkpoint_epoch"])]
    selection = {
        "schema": "forestformer3d_finetune_selection_v1",
        "status": "selected_on_frozen_development_validation",
        "training_mode": "fine_tuned_on_dev",
        "selection_rule": [
            "maximum_mean_plot_f1",
            "maximum_micro_f1",
            "earliest_checkpoint_epoch",
        ],
        "selected_checkpoint_epoch": selected["checkpoint_epoch"],
        "selected_checkpoint_relative_path": selected_entry["relative_path"],
        "selected_checkpoint_size_bytes": selected_entry["size_bytes"],
        "selected_checkpoint_sha256": selected_entry["sha256"],
        "selected_metrics": selected,
        "matched_published_baseline": baseline,
        "checkpoint_results": checkpoint_rows,
        "validation_plot_count": len(plots),
        "validation_evaluation_count": len(per_plot),
        "evaluation_protocol": "for_instance_pointwise_v1",
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_inventory_sha256": sha256_file(inventory_path),
        "source_development_run_id": source_run_root.name,
        "held_out_access": False,
        "next_gate": "fine_tuned_held_out_readiness_review",
    }

    output_root.mkdir(parents=True)
    (output_root / "selected_checkpoint.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint_fields = [
        "checkpoint_epoch",
        "mean_plot_f1",
        "micro_f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    ]
    with (output_root / "checkpoint_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=checkpoint_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in checkpoint_fields}
            for row in checkpoint_rows
        )
    per_plot_fields = [
        "checkpoint_epoch",
        "plot_id",
        "relative_path",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
    ]
    with (output_root / "per_plot_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_plot_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in per_plot_fields} for row in per_plot
        )
    (output_root / "retention_manifest.json").write_text(
        json.dumps(
            {
                "schema": "forestformer3d_finetune_validation_retention_v1",
                "immutable_validation_root": True,
                "held_out_access": False,
                "artifact_count": len(retained),
                "retained_bytes": sum(int(row["size_bytes"]) for row in retained),
                "artifacts": retained,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_root / "artifact_sha256.txt").open("x", encoding="utf-8") as handle:
        for name in (
            "selected_checkpoint.json",
            "checkpoint_metrics.csv",
            "per_plot_metrics.csv",
            "retention_manifest.json",
        ):
            handle.write(f"{sha256_file(output_root / name)}  {name}\n")
    (output_root / "selection.complete").touch(exist_ok=False)
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--benchmark-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarise(
                args.run_root,
                args.validation_root,
                args.source_run_root,
                args.output_root,
                benchmark_commit=args.benchmark_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
