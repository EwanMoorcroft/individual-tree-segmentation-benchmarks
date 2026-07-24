"""Validate and inventory the five frozen ForestFormer3D checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoint_layout import checkpoint_tensor_for_runtime


EPOCHS = (7, 14, 21, 28, 35)
DATA_LOADER_ITERATIONS_PER_EPOCH = 16
OPTIMIZER_STEPS_PER_EPOCH = 8


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(
    initial_path: Path,
    training_root: Path,
    output_path: Path,
) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    initial = torch.load(initial_path, map_location="cpu")
    initial_state = initial.get("state_dict")
    if not isinstance(initial_state, dict):
        raise TypeError("Initial checkpoint has no state_dict mapping")
    rows = []
    for epoch in EPOCHS:
        path = training_root / f"epoch_{epoch}.pth"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict) or set(state) != set(initial_state):
            raise ValueError(f"Checkpoint state_dict mismatch at epoch {epoch}")
        for name in state:
            initial_tensor = checkpoint_tensor_for_runtime(
                name, initial_state[name]
            )
            if (
                state[name].shape != initial_tensor.shape
                or state[name].dtype != initial_tensor.dtype
            ):
                raise ValueError(
                    f"Checkpoint tensor metadata mismatch: epoch={epoch} key={name}"
                )
        metadata = checkpoint.get("meta", {})
        expected_iter = epoch * DATA_LOADER_ITERATIONS_PER_EPOCH
        if (
            int(metadata.get("epoch", -1)) != epoch
            or int(metadata.get("iter", -1)) != expected_iter
        ):
            raise ValueError(f"Checkpoint progress metadata mismatch: epoch {epoch}")
        if not any(
            key in checkpoint
            for key in ("optimizer", "optim_wrapper", "optimizer_state")
        ):
            raise ValueError(f"Optimizer state missing at epoch {epoch}")
        rows.append(
            {
                "epoch": epoch,
                "data_loader_iterations": expected_iter,
                "optimizer_steps": epoch * OPTIMIZER_STEPS_PER_EPOCH,
                "relative_path": path.relative_to(training_root.parent).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    result = {
        "schema": "forestformer3d_finetune_checkpoint_inventory_v1",
        "status": "complete",
        "initial_checkpoint_sha256": sha256_file(initial_path),
        "epochs": list(EPOCHS),
        "examples_per_epoch": 16,
        "batch_size": 1,
        "gradient_accumulation": 2,
        "effective_batch_size": 2,
        "data_loader_iterations_per_epoch": DATA_LOADER_ITERATIONS_PER_EPOCH,
        "total_data_loader_iterations": (
            EPOCHS[-1] * DATA_LOADER_ITERATIONS_PER_EPOCH
        ),
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
        "total_optimizer_steps": EPOCHS[-1] * OPTIMIZER_STEPS_PER_EPOCH,
        "held_out_access": False,
        "checkpoints": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            inventory(
                args.initial_checkpoint,
                args.training_root,
                args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
