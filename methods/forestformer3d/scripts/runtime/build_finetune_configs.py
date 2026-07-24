"""Build frozen effective MMEngine configs from the pinned upstream config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


UPSTREAM_CONFIG_SHA256 = (
    "cdf6bff5269dfbd73f4b4c7fe30deffdbfed7fd73668ca2f0bc5bb792f04ec1f"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(
    cfg: dict[str, Any],
    *,
    data_root: str,
    checkpoint: str,
    work_dir: str,
    smoke: bool,
) -> dict[str, Any]:
    configured = deepcopy(cfg)
    train_ann = (
        "for_instance_finetune_smoke.pkl"
        if smoke
        else "for_instance_finetune_train.pkl"
    )
    configured["train_dataloader"]["dataset"]["data_root"] = data_root
    configured["train_dataloader"]["dataset"]["ann_file"] = train_ann
    configured["train_dataloader"]["dataset"]["filter_empty_gt"] = False
    configured["train_dataloader"]["batch_size"] = 1 if smoke else 2
    configured["train_dataloader"]["num_workers"] = 4 if smoke else 12
    configured["train_dataloader"]["prefetch_factor"] = 2
    configured["val_dataloader"]["dataset"]["data_root"] = data_root
    configured["val_dataloader"]["dataset"][
        "ann_file"
    ] = "for_instance_finetune_validation.pkl"
    configured["test_dataloader"]["dataset"]["data_root"] = data_root
    configured["test_dataloader"]["dataset"][
        "ann_file"
    ] = "for_instance_finetune_validation.pkl"

    configured["model"]["prepare_epoch"] = -1
    configured["optim_wrapper"]["optimizer"]["lr"] = 1e-5
    configured["optim_wrapper"]["optimizer"]["weight_decay"] = 0.05
    configured["param_scheduler"] = {
        "type": "PolyLR",
        "begin": 0,
        "end": 1 if smoke else 280,
        "power": 0.9,
        "by_epoch": False,
    }
    configured["train_cfg"] = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": 1 if smoke else 35,
        "val_interval": 2 if smoke else 36,
    }
    configured["default_hooks"]["checkpoint"].update(
        {
            "interval": 1 if smoke else 7,
            "max_keep_ckpts": 1 if smoke else 5,
            "save_optimizer": True,
        }
    )
    configured["load_from"] = checkpoint
    configured["resume"] = False
    configured["work_dir"] = work_dir
    configured["randomness"] = {
        "seed": 42,
        "diff_rank_seed": False,
        "deterministic": False,
    }
    return configured


def build(
    upstream_config: Path,
    output_root: Path,
    *,
    data_root: str,
    checkpoint: str,
    run_root: str,
) -> dict[str, Any]:
    from mmengine.config import Config

    upstream_config = upstream_config.resolve()
    output_root = output_root.resolve()
    if sha256_file(upstream_config) != UPSTREAM_CONFIG_SHA256:
        raise ValueError("Pinned upstream training config SHA-256 mismatch")
    if output_root.exists():
        raise FileExistsError(f"Refusing existing config output: {output_root}")
    output_root.mkdir(parents=True)
    base = Config.fromfile(upstream_config).to_dict()
    outputs = {}
    for name, smoke in (("smoke", True), ("full", False)):
        effective = Config(
            configure(
                base,
                data_root=data_root,
                checkpoint=checkpoint,
                work_dir=f"{run_root}/training_{name}",
                smoke=smoke,
            )
        )
        path = output_root / f"effective_{name}.py"
        effective.dump(path)
        outputs[name] = {
            "relative_path": path.name,
            "sha256": sha256_file(path),
            "max_epochs": 1 if smoke else 35,
            "batch_size": 1 if smoke else 2,
            "optimizer_steps": 1 if smoke else 280,
        }
    manifest = {
        "schema": "forestformer3d_effective_finetune_configs_v1",
        "upstream_config_sha256": UPSTREAM_CONFIG_SHA256,
        "data_root": data_root,
        "checkpoint": checkpoint,
        "run_root": run_root,
        "held_out_access": False,
        "configs": outputs,
    }
    manifest_path = output_root / "config_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "config_manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  config_manifest.json\n", encoding="utf-8"
    )
    (output_root / "configs.complete").touch(exist_ok=False)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.upstream_config,
                args.output_root,
                data_root=args.data_root,
                checkpoint=args.checkpoint,
                run_root=args.run_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
