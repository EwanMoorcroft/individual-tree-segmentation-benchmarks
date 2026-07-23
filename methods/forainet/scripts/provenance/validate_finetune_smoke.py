"""Validate a bounded official checkpoint-initialised ForAINet training smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


EXPECTED_CHECKPOINT_SHA256 = (
    "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
)
EXPECTED_TENSOR_COUNT = 755
EXPECTED_SMOKE_EPOCH = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_state(archive: dict[str, Any]) -> dict[str, torch.Tensor]:
    models = archive.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("latest"), dict):
        raise ValueError("checkpoint lacks latest model weights")
    return models["latest"]


def last_epoch(archive: dict[str, Any], stage: str) -> int:
    stats = archive.get("stats")
    if not isinstance(stats, dict) or not isinstance(stats.get(stage), list):
        raise ValueError(f"checkpoint lacks {stage} statistics")
    values = stats[stage]
    if not values or not isinstance(values[-1], dict) or "epoch" not in values[-1]:
        raise ValueError(f"checkpoint lacks a final {stage} epoch")
    return int(values[-1]["epoch"])


def validate(
    initial_checkpoint: Path,
    smoke_checkpoint: Path,
    expected_data_root: Path,
) -> dict[str, Any]:
    if sha256(initial_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("initial checkpoint identity changed")
    initial = torch.load(initial_checkpoint, map_location="cpu")
    smoke = torch.load(smoke_checkpoint, map_location="cpu")
    if not isinstance(initial, dict) or not isinstance(smoke, dict):
        raise ValueError("checkpoint archive is not a dictionary")
    initial_state = latest_state(initial)
    smoke_state = latest_state(smoke)
    if len(initial_state) != EXPECTED_TENSOR_COUNT:
        raise ValueError("initial checkpoint tensor count changed")
    if set(initial_state) != set(smoke_state):
        raise ValueError("training smoke changed model tensor keys")

    shape_mismatches = []
    changed_tensors = 0
    for key in initial_state:
        before = initial_state[key]
        after = smoke_state[key]
        if before.shape != after.shape:
            shape_mismatches.append(key)
        elif not torch.equal(before, after):
            changed_tensors += 1
    if shape_mismatches:
        raise ValueError(f"training smoke changed tensor shapes: {shape_mismatches}")
    if changed_tensors == 0:
        raise ValueError("training smoke did not update any checkpoint tensor")

    train_epoch = last_epoch(smoke, "train")
    validation_epoch = last_epoch(smoke, "val")
    if (train_epoch, validation_epoch) != (
        EXPECTED_SMOKE_EPOCH,
        EXPECTED_SMOKE_EPOCH,
    ):
        raise ValueError("training smoke did not complete epoch-one train and val")
    run_config = smoke.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("training smoke lacks saved run configuration")
    data = run_config.get("data")
    training = run_config.get("training")
    models = run_config.get("models")
    if not all(isinstance(value, dict) for value in (data, training, models)):
        raise ValueError("training smoke run configuration is incomplete")
    model = models.get("PointGroup-PAPER")
    if not isinstance(model, dict):
        raise ValueError("training smoke model configuration is missing")
    if Path(str(data.get("dataroot"))).resolve() != expected_data_root.resolve():
        raise ValueError("training smoke data root is not the frozen fine-tune data")
    if (
        Path(str(model.get("path_pretrained"))).resolve()
        != initial_checkpoint.resolve()
        or model.get("weight_name", "latest") != "latest"
    ):
        raise ValueError("training smoke did not use the official initial checkpoint")
    if int(training.get("batch_size", -1)) != 4:
        raise ValueError("training smoke changed official batch size")
    if bool(training.get("enable_mixed", False)):
        raise ValueError("training smoke unexpectedly enabled mixed precision")

    return {
        "schema": "forainet_finetune_smoke_validation_v1",
        "status": "verified",
        "initial_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "smoke_checkpoint_sha256": sha256(smoke_checkpoint),
        "model_tensor_count": len(smoke_state),
        "shape_compatible_fraction": 1.0,
        "changed_tensor_count": changed_tensors,
        "train_epoch": train_epoch,
        "validation_epoch": validation_epoch,
        "batch_size": int(training["batch_size"]),
        "precision": "fp32",
        "checkpoint_initialisation": (
            "models.PointGroup-PAPER.path_pretrained:latest"
        ),
        "data_root": str(expected_data_root.resolve()),
        "held_out_access": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--smoke-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-data-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    payload = validate(
        args.initial_checkpoint,
        args.smoke_checkpoint,
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
