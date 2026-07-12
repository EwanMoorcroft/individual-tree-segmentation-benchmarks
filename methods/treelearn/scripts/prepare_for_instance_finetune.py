"""Freeze and prepare a bounded, development-only TreeLearn fine-tune."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"


def assign_roles(plots: list[dict], seed: int = 42) -> list[dict]:
    if len(plots) != 21 or any(row.get("split") != "dev" for row in plots):
        raise ValueError("Fine-tuning requires the frozen 21-plot development manifest")
    validation = set(random.Random(seed).sample(range(len(plots)), 5))
    return [
        {**row, "training_role": "validation" if index in validation else "train"}
        for index, row in enumerate(plots)
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(manifest_path: Path, run_root: Path, treelearn_repo: Path,
            checkpoint: Path, seed: int, crop_count: int) -> dict:
    source = json.loads(manifest_path.read_text())
    if source.get("held_out_test_accessed") is not False:
        raise ValueError("Source manifest must lock the held-out test")
    if treelearn_repo.resolve() == run_root.resolve() or run_root.exists():
        raise FileExistsError(f"Refusing existing run root: {run_root}")
    rows = assign_roles(source["plots"], seed)
    train_rows = [row for row in rows if row["training_role"] == "train"]
    validation_rows = [row for row in rows if row["training_role"] == "validation"]

    forests = run_root / "data" / "train" / "forests"
    forests.mkdir(parents=True)
    for row in train_rows:
        source_las = Path(row["input_las"]).resolve()
        if not source_las.is_file():
            raise FileNotFoundError(source_las)
        name = f'{row["collection"]}__{Path(row["relative_path"]).name}'
        (forests / name).symlink_to(source_las)

    modular = treelearn_repo / "configs" / "_modular"
    crop_config = {
        "default_args": [str(modular / "sample_generation.yaml")],
        "base_dir": str(run_root / "data" / "train"),
        "occupancy_res": 1,
        "n_points_to_calculate_occupancy": 100000,
        "min_percent_occupied_fill": 0.9,
        "how_far_fill": 9,
        "min_percent_occupied_choose": 0.45,
        "n_samples_total": crop_count,
        "chunk_size": 35,
    }
    write_json(run_root / "configs" / "generate_crops.yaml", crop_config)

    defaults = [
        str(modular / "model.yaml"),
        str(modular / "dataset_train.yaml"),
        str(modular / "dataset_test.yaml"),
    ]
    crop_root = str(run_root / "data" / "train" / "random_crops" / "npz")
    common = {
        "default_args": defaults,
        "model": {"spatial_shape": [500, 500, 1000]},
        "dataset_train": {"data_root": crop_root},
        "dataset_test": {"data_root": crop_root},
        "dataloader": {
            "train": {"batch_size": 2, "num_workers": 2},
            "test": {"batch_size": 1, "num_workers": 1},
        },
        "optimizer": {"type": "AdamW", "lr": 0.0003, "weight_decay": 0.001},
        "scheduler": {
            "t_initial": 100, "lr_min": 0.000005, "cycle_decay": 1,
            "warmup_lr_init": 0.000001, "warmup_t": 5,
            "cycle_limit": 1, "t_in_epochs": True,
        },
        "fp16": True,
        "pretrain": str(checkpoint.resolve()),
        "grad_norm_clip": True,
        "validation_frequency": 101,
    }
    smoke = {**common, "epochs": 1, "examples_per_epoch": 16, "save_frequency": 1}
    full = {**common, "epochs": 100, "examples_per_epoch": 256, "save_frequency": 10}
    write_json(run_root / "configs" / "finetune_smoke.yaml", smoke)
    write_json(run_root / "configs" / "finetune_full.yaml", full)

    freeze = {
        "schema_version": 1,
        "status": "frozen_for_development_only_fine_tuning",
        "method": "TreeLearn",
        "training_mode": "fine_tuned_on_dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "initial_checkpoint": str(checkpoint.resolve()),
        "split_seed": seed,
        "validation_fraction": 0.25,
        "training_plots": len(train_rows),
        "validation_plots": len(validation_rows),
        "crop_count": crop_count,
        "held_out_test_accessed": False,
        "plots": rows,
    }
    write_json(run_root / "finetune_freeze.json", freeze)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-count", type=int, default=512)
    args = parser.parse_args()
    if not 32 <= args.crop_count <= 2000:
        raise ValueError("crop-count must be between 32 and 2000")
    result = prepare(args.development_manifest.resolve(), args.run_root.resolve(),
                     args.treelearn_repo.resolve(), args.checkpoint.resolve(),
                     args.seed, args.crop_count)
    print(f'training_plots={result["training_plots"]}')
    print(f'validation_plots={result["validation_plots"]}')
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
