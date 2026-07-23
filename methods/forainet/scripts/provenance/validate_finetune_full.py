"""Validate a complete official checkpoint-initialised ForAINet fine-tune."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_finetune_smoke import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TENSOR_COUNT,
    latest_state,
    sha256,
)


EXPECTED_EPOCHS = (30, 60, 90, 120, 149)


def archive_epoch(archive: dict[str, Any], stage: str) -> int:
    stats = archive.get("stats")
    if not isinstance(stats, dict) or not isinstance(stats.get(stage), list):
        raise ValueError(f"checkpoint lacks {stage} statistics")
    values = stats[stage]
    if not values or not isinstance(values[-1], dict):
        raise ValueError(f"checkpoint lacks final {stage} statistics")
    return int(values[-1]["epoch"])


def validate(
    initial_checkpoint: Path,
    full_checkpoint: Path,
    candidate_index: Path,
    candidate_root: Path,
    expected_data_root: Path,
) -> dict[str, Any]:
    if sha256(initial_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("initial checkpoint identity changed")
    index = json.loads(candidate_index.read_text(encoding="utf-8"))
    if (
        index.get("schema") != "forainet_finetune_candidate_index_v1"
        or index.get("status") != "complete"
        or index.get("expected_epochs") != list(EXPECTED_EPOCHS)
        or index.get("held_out_access") is not False
    ):
        raise ValueError("candidate index is not complete and test-locked")
    records = index.get("candidates")
    if not isinstance(records, list) or [
        int(row.get("epoch", -1)) for row in records
    ] != list(EXPECTED_EPOCHS):
        raise ValueError("candidate epochs differ from the frozen sweep")

    for record in records:
        candidate = candidate_root / str(record["filename"])
        if (
            not candidate.is_file()
            or sha256(candidate) != record["sha256"]
            or candidate.stat().st_size != int(record["size_bytes"])
        ):
            raise ValueError(f"candidate identity failed: {candidate}")
        archive = torch.load(candidate, map_location="cpu")
        epoch = int(record["epoch"])
        if (
            archive_epoch(archive, "train") != epoch
            or archive_epoch(archive, "val") != epoch
            or len(latest_state(archive)) != EXPECTED_TENSOR_COUNT
        ):
            raise ValueError(f"candidate epoch failed: {candidate}")

    initial = torch.load(initial_checkpoint, map_location="cpu")
    full = torch.load(full_checkpoint, map_location="cpu")
    initial_state = latest_state(initial)
    full_state = latest_state(full)
    if set(initial_state) != set(full_state):
        raise ValueError("full fine-tune changed model tensor keys")
    changed_tensors = 0
    for key in initial_state:
        if initial_state[key].shape != full_state[key].shape:
            raise ValueError(f"full fine-tune changed tensor shape: {key}")
        if not torch.equal(initial_state[key], full_state[key]):
            changed_tensors += 1
    if changed_tensors == 0:
        raise ValueError("full fine-tune did not update any tensor")
    if (
        archive_epoch(full, "train") != 149
        or archive_epoch(full, "val") != 149
    ):
        raise ValueError("full fine-tune did not finish epoch 149")
    final_candidate = candidate_root / str(records[-1]["filename"])
    if sha256(final_candidate) != sha256(full_checkpoint):
        raise ValueError("epoch-149 candidate differs from final rolling checkpoint")

    run_config = full.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("full checkpoint lacks run configuration")
    data = run_config.get("data")
    training = run_config.get("training")
    if not isinstance(data, dict) or not isinstance(training, dict):
        raise ValueError("full checkpoint configuration is incomplete")
    if Path(str(data.get("dataroot"))).resolve() != expected_data_root.resolve():
        raise ValueError("full checkpoint data root changed")
    if (
        int(training.get("epochs", -1)) != 150
        or int(training.get("batch_size", -1)) != 4
        or bool(training.get("enable_mixed", False))
    ):
        raise ValueError("full checkpoint training configuration changed")

    return {
        "schema": "forainet_finetune_full_validation_v1",
        "status": "verified",
        "initial_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "final_checkpoint_sha256": sha256(full_checkpoint),
        "candidate_index_sha256": sha256(candidate_index),
        "candidate_epochs": list(EXPECTED_EPOCHS),
        "candidate_count": len(records),
        "model_tensor_count": len(full_state),
        "shape_compatible_fraction": 1.0,
        "changed_tensor_count": changed_tensors,
        "configured_epoch_limit_exclusive": 150,
        "final_epoch": 149,
        "batch_size": 4,
        "precision": "fp32",
        "data_root": str(expected_data_root.resolve()),
        "held_out_access": False,
        "next_gate": "canonical_five_plot_candidate_validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--full-checkpoint", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--expected-data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    payload = validate(
        args.initial_checkpoint,
        args.full_checkpoint,
        args.candidate_index,
        args.candidate_root,
        args.expected_data_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
