"""Resolve and verify one task in the frozen 5-checkpoint x 5-plot matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EPOCHS = (7, 14, 21, 28, 35)
TASK_COUNT = 25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(run_root: Path, array_task_id: int) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    if not 0 <= array_task_id < TASK_COUNT:
        raise ValueError("Validation array task must be between 0 and 24")
    freeze_path = run_root / "fine_tune_freeze.json"
    inventory_path = run_root / "checkpoint_inventory.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "forestformer3d_fine_tune_freeze_v1"
        or freeze.get("split", {}).get("held_out_access") is not False
        or freeze.get("selection", {}).get("evaluated_checkpoint_epochs")
        != list(EPOCHS)
        or inventory.get("schema")
        != "forestformer3d_finetune_checkpoint_inventory_v1"
        or inventory.get("status") != "complete"
        or inventory.get("held_out_access") is not False
        or inventory.get("epochs") != list(EPOCHS)
    ):
        raise ValueError("Fine-tune freeze or checkpoint inventory is invalid")

    with (run_root / "fine_tune_split.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        validation = [
            row
            for row in csv.DictReader(handle)
            if row["fine_tune_role"] == "validation"
        ]
    if len(validation) != 5:
        raise ValueError("Frozen split must contain exactly five validation plots")

    checkpoint_slot, plot_slot = divmod(array_task_id, len(validation))
    epoch = EPOCHS[checkpoint_slot]
    entries = {
        int(entry["epoch"]): entry for entry in inventory["checkpoints"]
    }
    if tuple(sorted(entries)) != EPOCHS:
        raise ValueError("Checkpoint inventory does not match frozen epochs")
    entry = entries[epoch]
    checkpoint = run_root / str(entry["relative_path"])
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(entry["size_bytes"])
        or sha256_file(checkpoint) != entry["sha256"]
    ):
        raise ValueError(f"Checkpoint identity changed after inventory: epoch {epoch}")

    plot = validation[plot_slot]
    return {
        "array_task_id": array_task_id,
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": int(entry["size_bytes"]),
        "checkpoint_sha256": entry["sha256"],
        "manifest_task_index": int(plot["task_index"]),
        "plot_id": plot["plot_id"],
        "safe_plot_id": plot["safe_plot_id"],
        "relative_path": plot["relative_path"],
        "task_key": f"epoch_{epoch:02d}__{plot['safe_plot_id']}",
        "freeze_sha256": sha256_file(freeze_path),
        "inventory_sha256": sha256_file(inventory_path),
        "held_out_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--array-task-id", required=True, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            resolve(args.run_root, args.array_task_id),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
