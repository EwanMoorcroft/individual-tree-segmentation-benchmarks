"""Resolve one checkpoint task in the TreeLearn long-run validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EPOCHS = (7, 14, 21, 28, 35)
VALIDATION_TASKS = (0, 3, 7, 8, 20)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SEEDS = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_for(trial: dict, epoch: int) -> Path:
    template = trial.get("checkpoint_template")
    if template:
        return Path(str(template).format(epoch=epoch)).expanduser().resolve()
    root = trial.get("checkpoint_root") or trial.get("work_dir")
    if not root:
        raise ValueError("Trial has no checkpoint_template, checkpoint_root or work_dir")
    return (Path(root).expanduser().resolve() / f"epoch_{epoch}.pth")


def verify_checkpoint_identity(path: Path, size_bytes: int, digest: str) -> None:
    if size_bytes <= 0 or not SHA256.fullmatch(digest):
        raise ValueError(f"Invalid frozen checkpoint identity: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size_bytes:
        raise ValueError(f"Checkpoint size differs from trial completion: {path}")
    if sha256(path) != digest:
        raise ValueError(f"Checkpoint SHA-256 differs from trial completion: {path}")


def frozen_initial_checkpoint(freeze: dict) -> tuple[Path, int, str]:
    """Verify the initial checkpoint, accepting legacy freezes without a size."""

    checkpoint = Path(freeze["initial_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = str(freeze.get("initial_checkpoint_sha256", ""))
    recorded_size = freeze.get("initial_checkpoint_size_bytes")
    try:
        size_bytes = (
            checkpoint.stat().st_size
            if recorded_size is None
            else int(recorded_size)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Initial checkpoint identity is incomplete") from exc
    verify_checkpoint_identity(checkpoint, size_bytes, digest)
    return checkpoint, size_bytes, digest


def load_trial_completion(trial: dict, completion_root: Path) -> dict:
    trial_index = int(trial["trial_index"])
    path = completion_root.expanduser().resolve() / f"trial_{trial_index}.json"
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "long_finetune_trial_completed"
        or int(payload.get("trial_index", -1)) != trial_index
        or payload.get("config_id") != trial.get("config_id")
        or int(payload.get("seed", -1)) != int(trial.get("seed", -2))
        or payload.get("held_out_test_accessed") is not False
        or payload.get("bitwise_determinism_guaranteed") is not False
    ):
        raise ValueError(f"Trial completion identity differs from freeze: {path}")
    entries = payload.get("checkpoints")
    if not isinstance(entries, list) or len(entries) != len(EPOCHS):
        raise ValueError(f"Trial completion has an invalid checkpoint set: {path}")
    by_epoch: dict[int, dict] = {}
    for entry in entries:
        try:
            epoch = int(entry["epoch"])
            size_bytes = int(entry["size_bytes"])
            digest = str(entry["sha256"])
            checkpoint = Path(str(entry["path"])).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint evidence in {path}") from exc
        expected = _checkpoint_for(trial, epoch).resolve()
        if epoch not in EPOCHS or epoch in by_epoch or checkpoint != expected:
            raise ValueError(f"Checkpoint path or epoch differs from freeze: {path}")
        if size_bytes <= 0 or not SHA256.fullmatch(digest):
            raise ValueError(f"Invalid checkpoint identity in {path}")
        by_epoch[epoch] = {
            "epoch": epoch,
            "path": str(checkpoint),
            "size_bytes": size_bytes,
            "sha256": digest,
        }
    if tuple(sorted(by_epoch)) != EPOCHS:
        raise ValueError(f"Trial completion checkpoint schedule changed: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "checkpoints": by_epoch,
    }


def completion_checkpoint(trial: dict, epoch: int, completion: dict) -> dict:
    try:
        entry = completion["checkpoints"][epoch]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Trial completion is missing epoch {epoch}") from exc
    checkpoint = Path(entry["path"]).expanduser().resolve()
    verify_checkpoint_identity(checkpoint, int(entry["size_bytes"]), str(entry["sha256"]))
    return dict(entry)


def resolve_task(freeze: dict, array_task_id: int, completion_root: Path) -> dict:
    trials = freeze.get("trials")
    if not isinstance(trials, list) or len(trials) != 8:
        raise ValueError("Long-run freeze must contain exactly eight trials")
    checkpoint_tasks = len(trials) * len(EPOCHS)
    if not 0 <= array_task_id <= checkpoint_tasks:
        raise ValueError("Validation array task must be between 0 and 40")

    if array_task_id == checkpoint_tasks:
        run_id = f'{freeze["run_id"]}_clean_pretrained_validation'
        checkpoint, checkpoint_size, checkpoint_sha256 = frozen_initial_checkpoint(
            freeze
        )
        return {
            "trial_index": -1,
            "config_id": "clean_pretrained",
            "seed": 0,
            "epoch": 0,
            "manifest_task_indices": list(VALIDATION_TASKS),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint_size,
            "checkpoint_sha256": checkpoint_sha256,
            "completion_record": "not_applicable",
            "completion_record_sha256": "not_applicable",
            "validation_run_id": run_id,
            "evaluation_config": str(Path(freeze["evaluation_config"]).expanduser().resolve()),
            "evaluation_config_sha256": str(freeze["evaluation_config_sha256"]),
            "training_mode": "published_pretrained",
        }

    checkpoint_slot = array_task_id % len(EPOCHS)
    trial_slot = array_task_id // len(EPOCHS)
    trial = trials[trial_slot]
    if int(trial.get("trial_index", -1)) != trial_slot:
        raise ValueError("Trials must be ordered by contiguous trial_index")

    config_id = str(trial.get("config_id", ""))
    seed = int(trial.get("seed", -1))
    if not SAFE_COMPONENT.fullmatch(config_id):
        raise ValueError(f"Unsafe config_id: {config_id!r}")
    if seed not in SEEDS:
        raise ValueError(f"Unexpected trial seed: {seed}")

    epoch = EPOCHS[checkpoint_slot]
    completion = load_trial_completion(trial, completion_root)
    checkpoint = completion_checkpoint(trial, epoch, completion)
    run_id = (
        f'{freeze["run_id"]}_trial_{trial_slot:02d}_{config_id}'
        f"_seed_{seed}_epoch_{epoch}_validation"
    )
    if not SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError(f"Unsafe validation run ID: {run_id!r}")
    return {
        "trial_index": trial_slot,
        "config_id": config_id,
        "seed": seed,
        "epoch": epoch,
        "manifest_task_indices": list(VALIDATION_TASKS),
        "checkpoint": checkpoint["path"],
        "checkpoint_size_bytes": checkpoint["size_bytes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "completion_record": completion["path"],
        "completion_record_sha256": completion["sha256"],
        "validation_run_id": run_id,
        "evaluation_config": str(Path(freeze["evaluation_config"]).expanduser().resolve()),
        "evaluation_config_sha256": str(freeze["evaluation_config_sha256"]),
        "training_mode": "fine_tuned_on_dev",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--array-task-id", required=True, type=int)
    parser.add_argument("--completion-root", required=True, type=Path)
    args = parser.parse_args()
    task = resolve_task(
        json.loads(args.freeze.read_text()), args.array_task_id,
        args.completion_root.expanduser().resolve(),
    )
    for key in (
        "trial_index", "config_id", "seed", "epoch", "manifest_task_indices",
        "checkpoint", "checkpoint_size_bytes", "checkpoint_sha256",
        "completion_record", "completion_record_sha256", "validation_run_id",
        "evaluation_config", "evaluation_config_sha256", "training_mode",
    ):
        value = task[key]
        print(",".join(map(str, value)) if isinstance(value, list) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
