"""Prove exact official-checkpoint loading through the training Runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner

from checkpoint_layout import checkpoint_tensor_for_runtime


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(config_path: Path, checkpoint_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    cfg = Config.fromfile(config_path)
    if cfg.get("resume", None) is not False:
        raise ValueError("Fine-tuning must load weights without resuming epoch state")
    if Path(cfg.load_from) != checkpoint_path:
        raise ValueError("Effective config does not name the supplied checkpoint")
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected = checkpoint.get("state_dict")
    if not isinstance(expected, dict):
        raise TypeError("Initial checkpoint has no state_dict mapping")
    observed = runner.model.state_dict()
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(
            f"Training Runner key mismatch: missing={missing}, unexpected={unexpected}"
        )
    mismatched = []
    converted_spconv_tensors = 0
    for name in sorted(expected):
        checkpoint_tensor = expected[name].detach().cpu()
        source = checkpoint_tensor_for_runtime(name, checkpoint_tensor)
        if source is not checkpoint_tensor:
            converted_spconv_tensors += 1
        loaded = observed[name].detach().cpu()
        if source.shape != loaded.shape or source.dtype != loaded.dtype:
            mismatched.append(name)
        elif not torch.equal(source, loaded):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            "Training Runner did not load checkpoint tensors exactly: "
            + ", ".join(mismatched[:10])
        )
    if runner.epoch != 0 or runner.iter != 0:
        raise ValueError("load_from unexpectedly resumed epoch or iteration state")
    result = {
        "schema": "forestformer3d_finetune_initialization_v1",
        "status": "passed",
        "config_sha256": sha256_file(config_path),
        "initial_checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_key_count": len(expected),
        "exact_tensor_match": True,
        "spconv_layout_conversion": "archived_rskc_to_model_permute_4_0_1_2_3",
        "converted_spconv_tensor_count": converted_spconv_tensors,
        "runner_epoch": runner.epoch,
        "runner_iter": runner.iter,
        "resume": False,
        "held_out_access": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.checkpoint, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
