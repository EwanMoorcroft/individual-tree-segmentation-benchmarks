"""Resolve and verify one frozen long-run TreeLearn trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from consolidate_for_instance_finetune_long_crops import verify_consolidated


EXPECTED_EPOCHS = (7, 14, 21, 28, 35)
EXPECTED_SEEDS = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(freeze_path: Path, trial_index: int) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    matches = [row for row in freeze["trials"] if int(row["trial_index"]) == trial_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one trial for index {trial_index}")
    trial = matches[0]
    if int(trial["seed"]) not in EXPECTED_SEEDS:
        raise ValueError("Unexpected frozen seed")
    if tuple(trial["checkpoint_epochs"]) != EXPECTED_EPOCHS:
        raise ValueError("Unexpected checkpoint schedule")
    config = Path(trial["training_config"])
    if not config.is_file():
        raise FileNotFoundError(config)
    if sha256(config) != trial.get("training_config_sha256"):
        raise ValueError("Frozen trial configuration changed")
    return trial


def recorded_crop_integrity(freeze_path: Path, crop_inventory_path: Path) -> dict:
    return verify_consolidated(
        freeze_path.expanduser().resolve(),
        crop_inventory_path.expanduser().resolve(),
    )


def verify(
    freeze_path: Path, trial_index: int, output: Path, crop_inventory_path: Path,
) -> dict:
    trial = resolve(freeze_path, trial_index)
    crop_integrity = recorded_crop_integrity(freeze_path, crop_inventory_path)
    environment_record = output.parent / f"trial_{trial_index}_environment.json"
    environment = json.loads(environment_record.read_text())
    if (
        environment.get("status") != "treelearn_seeded_training_environment_frozen"
        or int(environment.get("seed", -1)) != int(trial["seed"])
        or environment.get("training_config_sha256") != trial["training_config_sha256"]
        or environment.get("crop_inventory") != crop_integrity["inventory"]
        or environment.get("crop_inventory_sha256")
        != crop_integrity["inventory_sha256"]
        or environment.get("crop_entries_aggregate_sha256")
        != crop_integrity["entries_aggregate_sha256"]
        or int(environment.get("crop_count_verified", -1))
        != int(crop_integrity["crop_count"])
        or int(environment.get("crop_referenced_size_bytes_verified", -1))
        != int(crop_integrity["referenced_size_bytes"])
    ):
        raise ValueError("Trial environment record differs from the frozen contract")
    checkpoint_root = Path(trial["checkpoint_root"])
    checkpoints = []
    for epoch in EXPECTED_EPOCHS:
        checkpoint = checkpoint_root / f"epoch_{epoch}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints.append({
            "epoch": epoch, "path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint),
        })
    payload = {
        "schema_version": 1,
        "status": "long_finetune_trial_completed",
        "trial_index": trial_index,
        "config_id": trial["config_id"],
        "seed": trial["seed"],
        "held_out_test_accessed": False,
        "environment_record": str(environment_record.resolve()),
        "environment_record_sha256": sha256(environment_record),
        "crop_inventory": crop_integrity["inventory"],
        "crop_inventory_sha256": crop_integrity["inventory_sha256"],
        "crop_entries_aggregate_sha256": crop_integrity[
            "entries_aggregate_sha256"
        ],
        "crop_count_verified": crop_integrity["crop_count"],
        "crop_referenced_size_bytes_verified": crop_integrity[
            "referenced_size_bytes"
        ],
        "bitwise_determinism_guaranteed": False,
        "checkpoints": checkpoints,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--crop-inventory", type=Path)
    parser.add_argument("--verify-output", type=Path)
    args = parser.parse_args()
    if args.verify_output:
        if args.crop_inventory is None:
            parser.error("--crop-inventory is required with --verify-output")
        result = verify(
            args.freeze.resolve(), args.trial_index, args.verify_output.resolve(),
            args.crop_inventory.expanduser().resolve(),
        )
    else:
        result = resolve(args.freeze.resolve(), args.trial_index)
    print(result["training_config"] if "training_config" in result else result["status"])
    if "seed" in result:
        print(result["seed"])
    if "checkpoint_root" in result:
        print(result["checkpoint_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
