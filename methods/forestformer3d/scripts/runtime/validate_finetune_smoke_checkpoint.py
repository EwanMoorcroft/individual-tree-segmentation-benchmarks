"""Validate the one-step ForestFormer3D fine-tuning smoke checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoint_layout import checkpoint_tensor_for_runtime


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(initial_path: Path, trained_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    initial = torch.load(initial_path, map_location="cpu")
    trained = torch.load(trained_path, map_location="cpu")
    initial_state = initial.get("state_dict")
    trained_state = trained.get("state_dict")
    if not isinstance(initial_state, dict) or not isinstance(trained_state, dict):
        raise TypeError("Checkpoint state_dict mapping is missing")
    if set(initial_state) != set(trained_state):
        raise ValueError("Smoke checkpoint state_dict keys changed")
    changed = 0
    unchanged = 0
    converted_spconv_tensors = 0
    maximum_absolute_change = 0.0
    for name in sorted(initial_state):
        archived = initial_state[name].detach().cpu()
        before = checkpoint_tensor_for_runtime(name, archived)
        if before is not archived:
            converted_spconv_tensors += 1
        after = trained_state[name].detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Smoke checkpoint tensor metadata changed: {name}")
        if torch.equal(before, after):
            unchanged += 1
        else:
            changed += 1
            if before.is_floating_point():
                maximum_absolute_change = max(
                    maximum_absolute_change,
                    float(torch.max(torch.abs(after - before)).item()),
                )
    if changed == 0:
        raise ValueError("Smoke training changed no checkpoint tensors")
    metadata = trained.get("meta", {})
    if int(metadata.get("epoch", -1)) != 1 or int(metadata.get("iter", -1)) != 1:
        raise ValueError("Smoke checkpoint does not record one epoch and one step")
    optimizer_present = any(
        key in trained for key in ("optimizer", "optim_wrapper", "optimizer_state")
    )
    if not optimizer_present:
        raise ValueError("Smoke checkpoint does not retain optimizer state")
    result = {
        "schema": "forestformer3d_finetune_smoke_checkpoint_v1",
        "status": "passed",
        "initial_checkpoint_sha256": sha256_file(initial_path),
        "smoke_checkpoint_sha256": sha256_file(trained_path),
        "state_dict_key_count": len(initial_state),
        "spconv_layout_conversion": "archived_rskc_to_model_permute_4_0_1_2_3",
        "converted_spconv_tensor_count": converted_spconv_tensors,
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "maximum_absolute_change": maximum_absolute_change,
        "epoch": 1,
        "optimizer_step": 1,
        "optimizer_state_retained": True,
        "held_out_access": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--trained-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.initial_checkpoint,
                args.trained_checkpoint,
                args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
