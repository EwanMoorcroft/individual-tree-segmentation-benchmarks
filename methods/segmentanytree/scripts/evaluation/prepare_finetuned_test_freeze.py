"""Freeze one held-out evaluation of a development-selected fine-tuned model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_EXTERNAL_COMMIT = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"
EXPECTED_RELEASED_SHA256 = (
    "0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
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


def freeze_finetuned_test(
    run_id: str,
    development_freeze_path: Path,
    development_summary_path: Path,
    training_metadata_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not re.fullmatch(
        r"segmentanytree_for-instance_fine_tuned_on_dev_\d{8}_\d{6}", run_id
    ):
        raise ValueError("Unexpected fine-tuned run ID")

    development_freeze = read_object(development_freeze_path)
    if development_freeze.get("status") != "frozen_for_development_only_fine_tuning":
        raise ValueError("Development fine-tuning was not frozen")
    if development_freeze.get("run_id") != run_id:
        raise ValueError("Development freeze belongs to a different run")
    if development_freeze.get("training_mode") != "fine_tuned_on_dev":
        raise ValueError("Development freeze has an unexpected training mode")
    if development_freeze.get("held_out_test_jobs_permitted") is not False:
        raise ValueError("Development freeze does not prove held-out isolation")
    if development_freeze.get("next_gate") != (
        "five_plot_development_validation_with_nonzero_instances"
    ):
        raise ValueError("Development freeze has an unexpected next gate")

    with development_summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Development summary must contain exactly one result row")
    row = rows[0]
    if row.get("variant") != "fine_tuned_on_dev_validation":
        raise ValueError("Development summary has an unexpected variant")
    if row.get("result_status") != "completed_aligned_pointwise_test":
        raise ValueError("Development validation did not complete")
    if row.get("dataset_split") != "dev" or int(row.get("plots", 0)) != 5:
        raise ValueError("Development summary does not cover the five-plot gate")
    if int(row.get("predicted_instances", 0)) <= 0:
        raise ValueError("Development validation contains zero predicted instances")
    if run_id not in row.get("metrics_root", ""):
        raise ValueError("Development summary belongs to a different run")

    training = read_object(training_metadata_path)
    expected_training = {
        "run_id": run_id,
        "training_mode": "fine_tuned_on_dev",
        "profile": "full",
        "status": "completed",
        "return_code": 0,
        "requested_epochs": 35,
        "batch_size": 8,
        "external_commit": EXPECTED_EXTERNAL_COMMIT,
        "pretrained_checkpoint_sha256": EXPECTED_RELEASED_SHA256,
        "pretrained_weight_name": "latest",
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"Training metadata has unexpected {key}")
    if float(training.get("base_lr", -1)) != 0.0001:
        raise ValueError("Training metadata has an unexpected base learning rate")
    load_validation = training.get("pretrained_load_validation") or {}
    if float(load_validation.get("compatible_fraction", 0)) < 0.95:
        raise ValueError("Released-weight compatibility was below the frozen gate")

    if not checkpoint_path.is_file():
        raise ValueError("Fine-tuned checkpoint is missing")
    checkpoint_hash = sha256(checkpoint_path)
    if training.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Fine-tuned checkpoint does not match training metadata")

    return {
        "status": "frozen_for_one_time_held_out_evaluation",
        "run_id": run_id,
        "benchmark": "for_instance_segmentanytree",
        "method": "SegmentAnyTree",
        "training_mode": "fine_tuned_on_dev",
        "development_freeze": str(development_freeze_path.resolve()),
        "development_summary": str(development_summary_path.resolve()),
        "development_summary_sha256": sha256(development_summary_path),
        "development_mean_plot_f1": float(row["mean_plot_f1"]),
        "development_predicted_instances": int(row["predicted_instances"]),
        "training_metadata": str(training_metadata_path.resolve()),
        "training_metadata_sha256": sha256(training_metadata_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "test_split": "test",
        "expected_test_plots": 11,
        "weight_updates": False,
        "postprocessing_updates": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": 0.5,
        "repeat_test_for_setting_selection_permitted": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one held-out evaluation of a fine-tuned model."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--development-freeze", required=True)
    parser.add_argument("--development-summary", required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze_finetuned_test(
        args.run_id,
        Path(args.development_freeze).expanduser().resolve(),
        Path(args.development_summary).expanduser().resolve(),
        Path(args.training_metadata).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Fine-tuned test freeze already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"freeze_manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
