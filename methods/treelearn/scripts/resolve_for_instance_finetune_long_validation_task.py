"""Resolve one task in the frozen TreeLearn long-run validation matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCHS = (7, 14, 21, 28, 35)
VALIDATION_TASKS = (0, 3, 7, 8, 20)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SEEDS = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)


def _checkpoint_for(trial: dict, epoch: int) -> Path:
    template = trial.get("checkpoint_template")
    if template:
        return Path(str(template).format(epoch=epoch)).expanduser().resolve()
    root = trial.get("checkpoint_root") or trial.get("work_dir")
    if not root:
        raise ValueError("Trial has no checkpoint_template, checkpoint_root or work_dir")
    return (Path(root).expanduser().resolve() / f"epoch_{epoch}.pth")


def resolve_task(freeze: dict, array_task_id: int) -> dict:
    trials = freeze.get("trials")
    if not isinstance(trials, list) or len(trials) != 8:
        raise ValueError("Long-run freeze must contain exactly eight trials")
    if not 0 <= array_task_id < len(trials) * len(EPOCHS) * len(VALIDATION_TASKS) + 5:
        raise ValueError("Validation array task must be between 0 and 204")

    if array_task_id >= 200:
        plot_slot = array_task_id - 200
        run_id = f'{freeze["run_id"]}_clean_pretrained_validation'
        return {
            "trial_index": -1,
            "config_id": "clean_pretrained",
            "seed": 0,
            "epoch": 0,
            "manifest_task_index": VALIDATION_TASKS[plot_slot],
            "checkpoint": str(Path(freeze["initial_checkpoint"]).expanduser().resolve()),
            "validation_run_id": run_id,
            "evaluation_config": str(Path(freeze["evaluation_config"]).expanduser().resolve()),
            "evaluation_config_sha256": str(freeze["evaluation_config_sha256"]),
            "training_mode": "published_pretrained",
        }

    plot_slot = array_task_id % len(VALIDATION_TASKS)
    checkpoint_slot = (array_task_id // len(VALIDATION_TASKS)) % len(EPOCHS)
    trial_slot = array_task_id // (len(EPOCHS) * len(VALIDATION_TASKS))
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
        "manifest_task_index": VALIDATION_TASKS[plot_slot],
        "checkpoint": str(_checkpoint_for(trial, epoch)),
        "validation_run_id": run_id,
        "evaluation_config": str(Path(freeze["evaluation_config"]).expanduser().resolve()),
        "evaluation_config_sha256": str(freeze["evaluation_config_sha256"]),
        "training_mode": "fine_tuned_on_dev",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--array-task-id", required=True, type=int)
    args = parser.parse_args()
    task = resolve_task(json.loads(args.freeze.read_text()), args.array_task_id)
    for key in (
        "trial_index", "config_id", "seed", "epoch", "manifest_task_index",
        "checkpoint", "validation_run_id", "evaluation_config",
        "evaluation_config_sha256",
        "training_mode",
    ):
        print(task[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
