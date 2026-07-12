"""Aggregate the frozen matrix and select a comparable epoch-35 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from resolve_for_instance_finetune_long_validation_task import EPOCHS


SEEDS = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)
CLEAN_INITIAL_CHECKPOINT_MD5 = "106a80de2991c5f23484a3f9d03e3b16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(text)


def aggregate(metrics: list[dict]) -> dict:
    tp = sum(int(row["true_positives"]) for row in metrics)
    fp = sum(int(row["false_positives"]) for row in metrics)
    fn = sum(int(row["false_negatives"]) for row in metrics)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "plots": len(metrics),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_f1": sum(float(row["f1"]) for row in metrics) / len(metrics),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def validation_rows(manifest: dict) -> list[dict]:
    rows = manifest.get("plots", manifest.get("records", []))
    by_task = {int(row["task_index"]): row for row in rows}
    result = [by_task[index] for index in (0, 3, 7, 8, 20)]
    if len(result) != 5 or any(row.get("split") != "dev" for row in result):
        raise ValueError("Expected the frozen five development-validation plots")
    return result


def choose_candidate(candidates: list[dict]) -> dict:
    """Apply the preregistered deterministic checkpoint-selection rule."""
    return sorted(
        candidates,
        key=lambda row: (
            -row["average_mean_plot_f1"],
            -row["average_micro_f1"],
            row["epoch"],
            row["config_id"],
        ),
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--metadata-base", required=True, type=Path)
    parser.add_argument("--tables-base", required=True, type=Path)
    parser.add_argument("--selection-root", required=True, type=Path)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze must explicitly lock held-out test access")
    initial_checkpoint = Path(freeze["initial_checkpoint"]).expanduser().resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    initial_md5 = md5(initial_checkpoint)
    if initial_md5 != CLEAN_INITIAL_CHECKPOINT_MD5:
        raise ValueError("Long run must start from the clean official L1W checkpoint")
    initial_sha256 = sha256(initial_checkpoint)
    if initial_sha256 != freeze.get("initial_checkpoint_sha256"):
        raise ValueError("Frozen initial checkpoint changed")
    trials = freeze.get("trials", [])
    if len(trials) != 8:
        raise ValueError("Selection requires one fixed configuration and eight seeds")
    for trial in trials:
        trial_index = int(trial["trial_index"])
        completion_path = args.freeze.parent / "trial_completions" / f"trial_{trial_index}.json"
        completion = json.loads(completion_path.read_text())
        if (
            completion.get("status") != "long_finetune_trial_completed"
            or int(completion.get("seed", -1)) != int(trial["seed"])
            or completion.get("bitwise_determinism_guaranteed") is not False
        ):
            raise ValueError(f"Incomplete trial evidence: {completion_path}")
    dev_manifest = json.loads(args.development_manifest.read_text())
    rows = validation_rows(dev_manifest)
    if args.selection_root.exists():
        raise FileExistsError(args.selection_root)
    args.selection_root.mkdir(parents=True)

    checkpoint_hashes: dict[Path, str] = {}
    group_results: list[dict] = []
    retained_paths: set[Path] = set()
    for trial in trials:
        config_id = str(trial["config_id"])
        seed = int(trial["seed"])
        trial_index = int(trial["trial_index"])
        if seed not in SEEDS:
            raise ValueError(f"Unexpected seed: {seed}")
        trial_config_path = Path(
            trial.get("training_config") or trial.get("config_path")
        ).expanduser().resolve()
        if sha256(trial_config_path) != trial.get("training_config_sha256"):
            raise ValueError(f"Frozen trial configuration changed: {trial_config_path}")
        trial_config = read_config(trial_config_path)
        if Path(trial_config["pretrain"]).expanduser().resolve() != initial_checkpoint:
            raise ValueError(f"Trial did not start from the frozen clean checkpoint: {trial_config_path}")
        batch_size = int(trial_config["dataloader"]["train"]["batch_size"])
        if (
            int(trial_config["epochs"]) != 35
            or int(trial_config["examples_per_epoch"]) != 714
            or batch_size != 2
        ):
            raise ValueError("Every long-run trial must use the frozen 24,990-example budget")
        for epoch in EPOCHS:
            checkpoint_root = Path(
                trial.get("checkpoint_root") or trial.get("work_dir")
            ).expanduser().resolve()
            checkpoint = checkpoint_root / f"epoch_{epoch}.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_sha = checkpoint_hashes.setdefault(checkpoint, sha256(checkpoint))
            run_id = (
                f'{freeze["run_id"]}_trial_{trial_index:02d}_{config_id}'
                f"_seed_{seed}_epoch_{epoch}_validation"
            )
            metrics: list[dict] = []
            retained_files = 0
            retained_bytes = 0
            for row in rows:
                safe = row["safe_plot_id"]
                metric_path = args.tables_base / run_id / "per_plot" / safe / "metrics.json"
                metadata_path = args.metadata_base / run_id / f"{safe}_inference.json"
                metric = json.loads(metric_path.read_text())
                metadata = json.loads(metadata_path.read_text())
                if metric.get("status") != "completed_aligned_pointwise_development_plot":
                    raise ValueError(f"Incomplete validation metric: {metric_path}")
                if metadata.get("training_mode") != "fine_tuned_on_dev":
                    raise ValueError(f"Wrong training mode: {metadata_path}")
                if metadata.get("checkpoint", {}).get("sha256") != checkpoint_sha:
                    raise ValueError(f"Checkpoint mismatch: {metadata_path}")
                if metadata.get("held_out_test_accessed") is not False:
                    raise ValueError(f"Held-out test lock missing: {metadata_path}")
                metrics.append(metric)
                retention = metadata["retention"]
                if not (
                    retention.get("raw_pointwise_output_retained") is True
                    and retention.get("raw_full_forest_output_retained") is True
                    and retention.get("adapted_point_aligned_output_retained") is True
                ):
                    raise ValueError(f"Incomplete prediction retention: {metadata_path}")
                entries = retention.get("files", [])
                if len(entries) != 5:
                    raise ValueError(f"Expected five retained artefacts: {metadata_path}")
                for entry in entries:
                    path = Path(entry["path"])
                    if path in retained_paths:
                        raise ValueError(f"Retained prediction reused across runs: {path}")
                    if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
                        raise ValueError(f"Retained prediction missing or resized: {path}")
                    if sha256(path) != entry["sha256"]:
                        raise ValueError(f"Retained prediction hash mismatch: {path}")
                    retained_paths.add(path)
                    retained_files += 1
                    retained_bytes += int(entry["size_bytes"])
            if retained_files != 25:
                raise ValueError(f"Expected 25 retained artefacts for {run_id}")
            group_results.append({
                "trial_index": trial_index,
                "config_id": config_id,
                "seed": seed,
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "retained_prediction_files": retained_files,
                "retained_prediction_bytes": retained_bytes,
                **aggregate(metrics),
            })

    baseline_run_id = f'{freeze["run_id"]}_clean_pretrained_validation'
    baseline_metrics: list[dict] = []
    baseline_retained_files = 0
    baseline_retained_bytes = 0
    for row in rows:
        safe = row["safe_plot_id"]
        metric_path = args.tables_base / baseline_run_id / "per_plot" / safe / "metrics.json"
        metadata_path = args.metadata_base / baseline_run_id / f"{safe}_inference.json"
        metric = json.loads(metric_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        if metric.get("status") != "completed_aligned_pointwise_development_plot":
            raise ValueError(f"Incomplete clean-baseline metric: {metric_path}")
        if metadata.get("training_mode") != "published_pretrained":
            raise ValueError(f"Wrong clean-baseline training mode: {metadata_path}")
        checkpoint = metadata.get("checkpoint", {})
        if checkpoint.get("sha256") != initial_sha256 or checkpoint.get("md5") != initial_md5:
            raise ValueError(f"Clean-baseline checkpoint mismatch: {metadata_path}")
        if metadata.get("held_out_test_accessed") is not False:
            raise ValueError(f"Held-out test lock missing: {metadata_path}")
        baseline_metrics.append(metric)
        retention = metadata["retention"]
        if not (
            retention.get("raw_pointwise_output_retained") is True
            and retention.get("raw_full_forest_output_retained") is True
            and retention.get("adapted_point_aligned_output_retained") is True
        ):
            raise ValueError(f"Incomplete clean-baseline retention: {metadata_path}")
        entries = retention.get("files", [])
        if len(entries) != 5:
            raise ValueError(f"Expected five clean-baseline artefacts: {metadata_path}")
        for entry in entries:
            path = Path(entry["path"])
            if path in retained_paths:
                raise ValueError(f"Retained prediction reused across runs: {path}")
            if not path.is_file() or path.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"Clean-baseline prediction missing or resized: {path}")
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Clean-baseline prediction hash mismatch: {path}")
            retained_paths.add(path)
            baseline_retained_files += 1
            baseline_retained_bytes += int(entry["size_bytes"])
    if baseline_retained_files != 25:
        raise ValueError("Expected 25 retained clean-baseline artefacts")
    clean_baseline = {
        "run_id": baseline_run_id,
        "checkpoint": str(initial_checkpoint),
        "checkpoint_md5": initial_md5,
        "checkpoint_sha256": initial_sha256,
        "retained_prediction_files": baseline_retained_files,
        "retained_prediction_bytes": baseline_retained_bytes,
        **aggregate(baseline_metrics),
    }

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for result in group_results:
        grouped[(result["config_id"], result["epoch"])].append(result)
    candidates = []
    for (config_id, epoch), records in grouped.items():
        if epoch != 35:
            continue
        if sorted(record["seed"] for record in records) != sorted(SEEDS):
            raise ValueError(f"Missing eight-seed evidence for {config_id} epoch {epoch}")
        candidates.append({
            "config_id": config_id,
            "epoch": epoch,
            "seed_count": len(records),
            "average_mean_plot_f1": sum(r["mean_plot_f1"] for r in records) / len(SEEDS),
            "average_micro_f1": sum(r["micro_f1"] for r in records) / len(SEEDS),
            "seed_results": records,
        })
    if len(candidates) != 1:
        raise ValueError("Expected one fixed epoch-35 configuration candidate")
    if len(retained_paths) != 1025:
        raise ValueError("Expected 1,025 unique retained validation artefacts")
    selected = choose_candidate(candidates)

    selected_seed42 = next(
        row for row in selected["seed_results"] if int(row["seed"]) == 42
    )
    selected_checkpoint = Path(selected_seed42["checkpoint"]).resolve()
    if sha256(selected_checkpoint) != selected_seed42["checkpoint_sha256"]:
        raise ValueError("Selected epoch-35 checkpoint changed")

    table_path = args.selection_root / "candidate_selection.csv"
    fields = [
        "config_id", "epoch", "seed_count", "average_mean_plot_f1",
        "average_micro_f1",
    ]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda row: (row["config_id"], row["epoch"])):
            writer.writerow({key: candidate[key] for key in fields})

    diagnostics_path = args.selection_root / "validation_diagnostics.csv"
    diagnostics_fields = [
        "trial_index", "config_id", "seed", "epoch", "plots",
        "true_positives", "false_positives", "false_negatives",
        "mean_plot_f1", "micro_precision", "micro_recall", "micro_f1",
        "checkpoint", "checkpoint_sha256", "retained_prediction_files",
        "retained_prediction_bytes",
    ]
    with diagnostics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics_fields, lineterminator="\n")
        writer.writeheader()
        for result in sorted(
            group_results, key=lambda row: (row["config_id"], row["seed"], row["epoch"])
        ):
            writer.writerow({key: result[key] for key in diagnostics_fields})

    selection_freeze = {
        "schema_version": 1,
        "status": "frozen_comparable_development_selected_checkpoint",
        "method": "TreeLearn",
        "training_mode": "fine_tuned_on_dev",
        "source_long_run_id": freeze["run_id"],
        "held_out_test_accessed": False,
        "selection_split": "dev_validation",
        "selection_rule": (
            "fixed full_lr_1e-5 configuration, seed 42 and epoch 35; "
            "seven additional seeds and earlier checkpoints are diagnostics only"
        ),
        "validation_plot_count": 5,
        "selected": selected,
        "clean_pretrained_validation_baseline": clean_baseline,
        "selected_minus_clean_baseline_mean_plot_f1": (
            selected_seed42["mean_plot_f1"] - clean_baseline["mean_plot_f1"]
        ),
        "selected_minus_clean_baseline_micro_f1": (
            selected_seed42["micro_f1"] - clean_baseline["micro_f1"]
        ),
        "selected_seed": 42,
        "selected_training_split": "fixed_16_development_training_plots",
        "selected_training_plots": 16,
        "selected_epoch_count": 35,
        "selected_examples_per_epoch": 714,
        "selected_examples_seen": 24990,
        "selected_batch_size": 2,
        "selected_optimizer_steps": 12495,
        "selected_initial_checkpoint": str(initial_checkpoint),
        "selected_initial_checkpoint_role": freeze.get(
            "initial_checkpoint_role", "authors_released_l1w_finetuned"
        ),
        "selected_initial_checkpoint_md5": initial_md5,
        "selected_initial_checkpoint_sha256": initial_sha256,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_size_bytes": selected_checkpoint.stat().st_size,
        "selected_checkpoint_sha256": selected_seed42["checkpoint_sha256"],
        "development_manifest": str(args.development_manifest.resolve()),
        "development_manifest_sha256": sha256(args.development_manifest),
        "retention_status": "all_validation_predictions_sha256_verified",
        "retained_prediction_files": len(retained_paths),
        "validation_diagnostics": str(diagnostics_path.resolve()),
        "validation_diagnostics_sha256": sha256(diagnostics_path),
        "next_gate": "manual_review_before_any_held_out_test_submission",
    }
    (args.selection_root / "selection_freeze.json").write_text(
        json.dumps(selection_freeze, indent=2, sort_keys=True) + "\n"
    )
    print(f'selected_config={selected["config_id"]}')
    print(f'selected_epoch={selected["epoch"]}')
    print(f'selected_average_mean_plot_f1={selected["average_mean_plot_f1"]:.6f}')
    print(f'selected_average_micro_f1={selected["average_micro_f1"]:.6f}')
    print(f"selected_checkpoint={selected_checkpoint}")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
