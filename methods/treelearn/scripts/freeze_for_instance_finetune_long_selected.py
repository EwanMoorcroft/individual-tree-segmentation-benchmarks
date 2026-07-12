"""Verify and freeze the comparable TreeLearn epoch-35 selected checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-freeze", required=True, type=Path)
    parser.add_argument("--retention-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection_freeze.read_text())
    if selection.get("status") != "frozen_comparable_development_selected_checkpoint":
        raise ValueError("Comparable development selection is not frozen")
    if selection.get("held_out_test_accessed") is not False:
        raise ValueError("Selection does not lock held-out test access")
    if (
        int(selection.get("selected_epoch_count", -1)) != 35
        or int(selection.get("selected_training_plots", -1)) != 16
    ):
        raise ValueError("Selected checkpoint differs from the common 35-epoch 16-plot contract")
    retention = json.loads(args.retention_manifest.read_text())
    if (
        retention.get("status") != "long_finetune_selection_retained_on_scratch"
        or retention.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Durable selection retention is not verified")
    if retention.get("retained_checkpoint_sha256") != selection.get(
        "selected_checkpoint_sha256"
    ):
        raise ValueError("Durable checkpoint hash differs from selected checkpoint")
    checkpoint = Path(retention["retained_checkpoint"]).expanduser().resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(selection["selected_checkpoint_size_bytes"])
        or sha256(checkpoint) != retention["retained_checkpoint_sha256"]
    ):
        raise ValueError("Selected checkpoint identity changed")
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "schema_version": 1,
        "status": "frozen_selected_checkpoint_pending_manual_held_out_test_authorisation",
        "method": "TreeLearn",
        "training_mode": "fine_tuned_on_dev",
        "source_long_run_id": selection["source_long_run_id"],
        "held_out_test_accessed": False,
        "test_jobs_submitted": 0,
        "selection_split": selection["selection_split"],
        "selection_rule": selection["selection_rule"],
        "clean_pretrained_validation_baseline": selection[
            "clean_pretrained_validation_baseline"
        ],
        "selected_minus_clean_baseline_mean_plot_f1": selection[
            "selected_minus_clean_baseline_mean_plot_f1"
        ],
        "selected_minus_clean_baseline_micro_f1": selection[
            "selected_minus_clean_baseline_micro_f1"
        ],
        "selected_config_id": selection["selected"]["config_id"],
        "selected_epoch": 35,
        "selected_seed": 42,
        "training_split": selection["selected_training_split"],
        "training_plots": 16,
        "examples_per_epoch": selection["selected_examples_per_epoch"],
        "examples_seen": selection["selected_examples_seen"],
        "batch_size": selection["selected_batch_size"],
        "optimizer_steps": selection["selected_optimizer_steps"],
        "initial_checkpoint": selection["selected_initial_checkpoint"],
        "initial_checkpoint_role": selection["selected_initial_checkpoint_role"],
        "initial_checkpoint_md5": selection["selected_initial_checkpoint_md5"],
        "initial_checkpoint_sha256": selection["selected_initial_checkpoint_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "retention_manifest": str(args.retention_manifest.resolve()),
        "retention_status": retention["status"],
        "validation_retention_status": selection["retention_status"],
        "validation_retained_prediction_files": selection["retained_prediction_files"],
        "next_gate": "manual_review_before_any_held_out_test_submission",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"selected_freeze={args.output}")
    print(f"checkpoint={checkpoint}")
    print(f'checkpoint_sha256={payload["checkpoint_sha256"]}')
    print("held_out_test_accessed=false")
    print("No held-out test job was submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
