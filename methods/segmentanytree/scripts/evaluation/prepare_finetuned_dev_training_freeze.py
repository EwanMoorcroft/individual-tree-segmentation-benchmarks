"""Freeze one development-only SegmentAnyTree fine-tuning run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_SHA256 = (
    "0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
)
EXPECTED_EXTERNAL_COMMIT = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
REJECTED_RUN_ID = (
    "segmentanytree_for-instance_fine_tuned_on_dev_20260708_215054_full"
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage1_result(test_freeze_path: Path, final_summary_path: Path) -> None:
    test_freeze = read_object(test_freeze_path)
    if test_freeze.get("status") != "frozen_for_one_time_held_out_evaluation":
        raise ValueError("Stage 1 held-out evaluation was not frozen correctly")
    if test_freeze.get("training_mode") != "published_pretrained":
        raise ValueError("Stage 1 freeze has an unexpected training mode")
    if test_freeze.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Stage 1 freeze records an unexpected checkpoint")
    if test_freeze.get("expected_test_plots") != 11:
        raise ValueError("Stage 1 freeze does not record all 11 test plots")
    if test_freeze.get("repeat_test_for_setting_selection_permitted") is not False:
        raise ValueError("Stage 1 freeze does not prohibit test-based selection")
    stage1_run_id = test_freeze.get("run_id", "")
    if not re.fullmatch(
        r"segmentanytree_for-instance_published_pretrained_\d{8}_\d{6}",
        stage1_run_id,
    ):
        raise ValueError("Stage 1 freeze has an invalid run ID")
    if test_freeze_path.stem != stage1_run_id:
        raise ValueError("Stage 1 freeze filename does not match its run ID")

    with final_summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Stage 1 final summary must contain exactly one result row")
    row = rows[0]
    if row.get("variant") != "published_pretrained":
        raise ValueError("Stage 1 final summary has an unexpected variant")
    if row.get("result_status") != "completed_aligned_pointwise_test":
        raise ValueError("Stage 1 final summary is not complete")
    if row.get("dataset_split") != "test" or int(row.get("plots", 0)) != 11:
        raise ValueError("Stage 1 final summary does not cover the frozen test split")
    if stage1_run_id not in row.get("metrics_root", ""):
        raise ValueError("Stage 1 final summary does not match the frozen run")


def freeze_finetuned_dev_training(
    split_manifest_path: Path,
    checkpoint_bundle: Path,
    stage1_test_freeze_path: Path,
    stage1_final_summary_path: Path,
    run_id: str,
) -> dict[str, Any]:
    if not re.fullmatch(
        r"segmentanytree_for-instance_fine_tuned_on_dev_\d{8}_\d{6}", run_id
    ) or run_id == REJECTED_RUN_ID:
        raise ValueError("Fine-tuning run ID is invalid or belongs to a rejected run")

    validate_stage1_result(stage1_test_freeze_path, stage1_final_summary_path)

    split_manifest = read_object(split_manifest_path)
    counts = split_manifest.get("selected_role_counts", {})
    actual_counts = (
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("held_out_test", 0),
    )
    if actual_counts != (16, 5, 0):
        raise ValueError(f"Unexpected development split counts: {actual_counts}")
    if split_manifest.get("test_data_converted") is not False:
        raise ValueError("Development training manifest does not prove test isolation")

    checkpoint = checkpoint_bundle / "PointGroup-PAPER.pt"
    overrides = checkpoint_bundle / ".hydra" / "overrides.yaml"
    if not checkpoint.is_file() or sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Released checkpoint is missing or has changed")
    if not overrides.is_file():
        raise ValueError("Released checkpoint Hydra overrides are missing")
    overrides_lines = set(overrides.read_text(encoding="utf-8").splitlines())
    if "- job_name=mls_data_run" not in overrides_lines:
        raise ValueError("Released checkpoint is not the reviewed MLS bundle")

    return {
        "status": "frozen_for_development_only_fine_tuning",
        "run_id": run_id,
        "benchmark": "for_instance_segmentanytree",
        "method": "SegmentAnyTree",
        "training_mode": "fine_tuned_on_dev",
        "dataset_split": "dev",
        "training_plots": 16,
        "validation_plots": 5,
        "held_out_test_plots": 0,
        "held_out_test_jobs_permitted": False,
        "stage1_test_result_used_for_setting_selection": False,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "source_checkpoint_job_name": "mls_data_run",
        "initialisation": "released_checkpoint_weights_only",
        "optimizer_state": "fresh",
        "epoch_history": "fresh",
        "pretrained_weight_name": "latest",
        "minimum_compatible_weight_fraction": 0.95,
        "smoke_epochs": 1,
        "training_epochs": 35,
        "batch_size": 8,
        "base_learning_rate": 0.0001,
        "external_commit": EXPECTED_EXTERNAL_COMMIT,
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256(split_manifest_path),
        "stage1_test_freeze": str(stage1_test_freeze_path.resolve()),
        "stage1_test_run_id": read_object(stage1_test_freeze_path)["run_id"],
        "stage1_final_summary": str(stage1_final_summary_path.resolve()),
        "stage1_final_summary_sha256": sha256(stage1_final_summary_path),
        "next_gate": "five_plot_development_validation_with_nonzero_instances",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze released-weight fine-tuning on the development split."
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--checkpoint-bundle", required=True)
    parser.add_argument("--stage1-test-freeze", required=True)
    parser.add_argument("--stage1-final-summary", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze_finetuned_dev_training(
        Path(args.split_manifest).expanduser().resolve(),
        Path(args.checkpoint_bundle).expanduser().resolve(),
        Path(args.stage1_test_freeze).expanduser().resolve(),
        Path(args.stage1_final_summary).expanduser().resolve(),
        args.run_id,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Fine-tuning freeze already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"freeze_manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
