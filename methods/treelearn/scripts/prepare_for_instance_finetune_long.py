"""Freeze the development-only TreeLearn long fine-tuning matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
CLEAN_CHECKPOINT_MD5 = "106a80de2991c5f23484a3f9d03e3b16"
CLEAN_CHECKPOINT_FILENAME = "model_weights_finetuned.pth"
CLEAN_CHECKPOINT_ROLE = "authors_released_l1w_finetuned_for_instance_clean"
CLEAN_CHECKPOINT_PERSISTENT_ID = "doi:10.25625/VPMPID/8CIIW0"
CLEAN_CHECKPOINT_SOURCE = (
    "https://data.goettingen-research-online.de/api/access/datafile/"
    ":persistentId?persistentId=doi:10.25625/VPMPID/8CIIW0"
)
SPLIT_SEED = 42
CROPS_PER_PLOT = 32
CROP_GENERATION_ATTEMPTS_PER_PLOT = 48
TRAINING_EPOCHS = 35
EXAMPLES_PER_EPOCH = 714
BATCH_SIZE = 2
CHECKPOINT_EPOCHS = (7, 14, 21, 28, 35)
SEEDS = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)
PROFILES = (
    {"config_id": "full_lr_1e-5", "learning_rate": 1e-5, "fixed_modules": []},
)
EXPONENT_VALUE = re.compile(
    r'(:\s*)(-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+)(?=[,\n])'
)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_training_config(path: Path, payload: dict) -> None:
    """Write JSON whose numeric values also parse numerically under YAML 1.1."""

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    text = EXPONENT_VALUE.sub(
        lambda match: match.group(1) + format(Decimal(match.group(2)), "f"),
        text,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def canonical_metadata_path(value: object) -> str:
    """Apply the same path normalisation as the accepted manifest builder."""

    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if (
        not raw
        or raw.startswith(("/", "./"))
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
        or path.suffix.casefold() != ".las"
    ):
        raise ValueError(f"Unsafe FOR-instance metadata path: {value!r}")
    return raw


def assign_roles(plots: list[dict], seed: int = SPLIT_SEED) -> list[dict]:
    if len(plots) != 21 or any(row.get("split") != "dev" for row in plots):
        raise ValueError("Long fine-tuning requires the frozen 21-plot development manifest")
    validation = set(random.Random(seed).sample(range(21), 5))
    return [
        {**row, "training_role": "validation" if index in validation else "train"}
        for index, row in enumerate(plots)
    ]


def verify_supplied_split(source: dict) -> dict:
    if (
        source.get("status") != "frozen_exact_path_development_manifest"
        or source.get("dataset_split") != "dev"
        or source.get("mapping_rule") != "exact_metadata_path_only"
    ):
        raise ValueError("Source manifest is not the frozen FOR-instance supplied dev split")
    metadata_path = Path(source["split_metadata"]).expanduser().resolve()
    metadata_sha256 = file_hash(metadata_path)
    if metadata_sha256 != source.get("split_metadata_sha256"):
        raise ValueError("FOR-instance supplied split metadata changed")
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata_dev_paths = sorted(
        canonical_metadata_path(row.get("path"))
        for row in rows
        if str(row.get("split", "")).strip() == "dev"
    )
    metadata_test_paths = sorted(
        canonical_metadata_path(row.get("path"))
        for row in rows
        if str(row.get("split", "")).strip() == "test"
    )
    manifest_paths = sorted(str(row["relative_path"]) for row in source["plots"])
    if (
        len(metadata_dev_paths) != 56
        or len(set(metadata_dev_paths)) != 56
        or len(metadata_test_paths) != 26
        or len(set(metadata_test_paths)) != 26
        or set(metadata_dev_paths) & set(metadata_test_paths)
    ):
        raise ValueError("FOR-instance supplied split metadata catalogue is invalid")
    if len(manifest_paths) != 21 or not set(manifest_paths).issubset(metadata_dev_paths):
        raise ValueError("Development manifest is not an exact-path subset of supplied dev rows")
    if any(
        row.get("split") != "dev"
        or row.get("split_metadata_sha256") != metadata_sha256
        for row in source["plots"]
    ):
        raise ValueError("Development row split evidence differs from supplied metadata")
    return {
        "source": str(metadata_path),
        "sha256": metadata_sha256,
        "metadata_development_rows": len(metadata_dev_paths),
        "metadata_test_rows": len(metadata_test_paths),
        "benchmark_development_rows": len(manifest_paths),
        "benchmark_expected_test_rows": 11,
        "mapping": "every benchmark development path is an exact-path member where split == dev",
        "held_out_test_files_opened": False,
    }


def training_config(
    *, treelearn_repo: Path, checkpoint: Path, data_root: Path,
    work_dir: Path, profile: dict, seed: int,
) -> dict:
    modular = treelearn_repo / "configs" / "_modular"
    learning_rate = float(profile["learning_rate"])
    return {
        "default_args": [
            str(modular / "model.yaml"),
            str(modular / "dataset_train.yaml"),
            str(modular / "dataset_test.yaml"),
        ],
        "model": {
            "spatial_shape": [500, 500, 1000],
            "fixed_modules": list(profile["fixed_modules"]),
        },
        "dataset_train": {"data_root": str(data_root)},
        "dataset_test": {"data_root": str(data_root)},
        "dataloader": {
            "train": {"batch_size": BATCH_SIZE, "num_workers": 2},
            "test": {"batch_size": 1, "num_workers": 1},
        },
        "optimizer": {"type": "AdamW", "lr": learning_rate, "weight_decay": 0.001},
        "scheduler": {
            "t_initial": TRAINING_EPOCHS,
            "lr_min": learning_rate / 20,
            "cycle_decay": 1,
            "warmup_lr_init": learning_rate / 10,
            "warmup_t": 2,
            "cycle_limit": 1,
            "t_in_epochs": True,
        },
        "epochs": TRAINING_EPOCHS,
        "examples_per_epoch": EXAMPLES_PER_EPOCH,
        "fp16": True,
        "pretrain": str(checkpoint),
        "grad_norm_clip": True,
        "validation_frequency": TRAINING_EPOCHS + 1,
        "save_frequency": 7,
        "seed": seed,
        "work_dir": str(work_dir),
    }


def write_evaluation_config(
    template: Path, output: Path, checkpoint: Path, checkpoint_sha256: str,
) -> None:
    text = template.read_text()
    replacements = {
        "filename: model_weights_20241213.pth": f"filename: {CLEAN_CHECKPOINT_FILENAME}",
        'default_path: "~/fastscratch/treelearn_checkpoints/model_weights_20241213.pth"':
            f'default_path: "{checkpoint}"',
        "source_dataset_name: model_weights_20241213":
            "source_dataset_name: model_weights_finetuned",
        'source: "TreeLearn upstream default December 2024 model weights"':
            'source: "TreeLearn authors-released L1W-fine-tuned checkpoint; no FOR-instance training"',
        'source_url: "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/IMHF3G"':
            f'source_url: "{CLEAN_CHECKPOINT_SOURCE}"',
        'source_md5: "56a3d78f689ae7f1190906b975700311"':
            f'source_md5: "{CLEAN_CHECKPOINT_MD5}"',
        'training_data_provenance: "Authors\' diverse-data checkpoint subsequently fine-tuned on manually labelled data including FOR-instance validation/test"':
            'training_data_provenance: "Authors-released noisy-label checkpoint subsequently fine-tuned on the L1W benchmark"',
        'released_weight_test_overlap_status: "documented_for_instance_validation_test_training_overlap; exact_plot_manifest_not_bundled"':
            'released_weight_test_overlap_status: "no_for_instance_training_reported_by_authors"',
        "eligible_for_leakage_free_primary_ranking: false":
            "eligible_for_leakage_free_primary_ranking: true",
        'sha256: "5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"':
            f'sha256: "{checkpoint_sha256}"',
        "sha256_status: frozen_from_accepted_development_smoke":
            "sha256_status: frozen_clean_official_initial_checkpoint",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"Evaluation template is missing frozen text: {old}")
        text = text.replace(old, new, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def prepare(
    development_manifest: Path, run_root: Path, treelearn_repo: Path,
    checkpoint: Path, evaluation_template: Path,
    *, crops_per_plot: int = CROPS_PER_PLOT,
) -> dict:
    if run_root.exists():
        raise FileExistsError(f"Refusing existing long-run root: {run_root}")
    if not checkpoint.is_file() or file_hash(checkpoint, "md5") != CLEAN_CHECKPOINT_MD5:
        raise ValueError("Long fine-tuning requires the clean official TreeLearn checkpoint")
    if not treelearn_repo.is_dir() or not evaluation_template.is_file():
        raise FileNotFoundError(treelearn_repo if not treelearn_repo.is_dir() else evaluation_template)
    source = json.loads(development_manifest.read_text())
    if source.get("held_out_test_accessed") is not False:
        raise ValueError("Development manifest must explicitly lock held-out test access")
    supplied_split = verify_supplied_split(source)
    rows = assign_roles(source.get("plots", []), SPLIT_SEED)
    if crops_per_plot != CROPS_PER_PLOT:
        raise ValueError(f"Long protocol requires exactly {CROPS_PER_PLOT} crops per plot")

    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = file_hash(checkpoint)
    run_root.mkdir(parents=True)
    tuning_root = run_root / "data" / "views" / "tuning" / "npz"
    frozen_rows = []
    for index, row in enumerate(rows):
        source_las = Path(row["input_las"]).expanduser().resolve()
        if not source_las.is_file():
            raise FileNotFoundError(source_las)
        safe = str(row["safe_plot_id"])
        plot_root = run_root / "data" / "by_plot" / safe
        frozen_rows.append({
            **row,
            "task_index": index,
            "input_las": str(source_las),
            "normalised_las": str(plot_root / "forests" / f"{safe}.las"),
            "normalisation_metadata": str(plot_root / "normalisation.json"),
            "crop_config": str(run_root / "configs" / "crops" / f"{safe}.yaml"),
            "crop_root": str(plot_root / "random_crops" / "npz"),
            "crop_inventory": str(plot_root / "crop_inventory.json"),
            "crops_expected": crops_per_plot,
            "crops_generate_requested": CROP_GENERATION_ATTEMPTS_PER_PLOT,
            "crop_seed": 42000 + index,
        })

    trials = []
    trial_index = 0
    for profile in PROFILES:
        for seed in SEEDS:
            config_id = profile["config_id"]
            checkpoint_root = (
                run_root / "trials" / f"config_{config_id}" / f"seed_{seed}"
                / "work_dirs" / "finetune_long"
            )
            config_path = run_root / "configs" / "trials" / f"config_{config_id}_seed_{seed}.yaml"
            write_training_config(config_path, training_config(
                treelearn_repo=treelearn_repo.resolve(), checkpoint=checkpoint,
                data_root=tuning_root.resolve(), work_dir=checkpoint_root.resolve(),
                profile=profile, seed=seed,
            ))
            trials.append({
                "trial_index": trial_index,
                "config_id": config_id,
                "seed": seed,
                "learning_rate": profile["learning_rate"],
                "fixed_modules": profile["fixed_modules"],
                "training_config": str(config_path.resolve()),
                "training_config_sha256": file_hash(config_path),
                "checkpoint_root": str(checkpoint_root.resolve()),
                "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            })
            trial_index += 1

    evaluation_config = run_root / "configs" / "for_instance_finetune_long_evaluation.yml"
    write_evaluation_config(evaluation_template, evaluation_config, checkpoint, checkpoint_sha256)
    freeze = {
        "schema_version": 1,
        "status": "frozen_for_development_only_long_fine_tuning",
        "run_id": run_root.name,
        "method": "TreeLearn",
        "training_mode": "fine_tuned_on_dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": UPSTREAM_COMMIT,
        "development_manifest": str(development_manifest.resolve()),
        "development_manifest_sha256": file_hash(development_manifest),
        "supplied_split_contract": supplied_split,
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_filename": CLEAN_CHECKPOINT_FILENAME,
        "initial_checkpoint_role": CLEAN_CHECKPOINT_ROLE,
        "initial_checkpoint_source": CLEAN_CHECKPOINT_SOURCE,
        "initial_checkpoint_persistent_id": CLEAN_CHECKPOINT_PERSISTENT_ID,
        "initial_checkpoint_md5": CLEAN_CHECKPOINT_MD5,
        "initial_checkpoint_sha256": checkpoint_sha256,
        "evaluation_config": str(evaluation_config.resolve()),
        "evaluation_config_sha256": file_hash(evaluation_config),
        "split_seed": SPLIT_SEED,
        "training_plots": 16,
        "validation_plots": 5,
        "all_development_plots": 21,
        "crops_per_plot": crops_per_plot,
        "crop_generation_attempts_per_plot": CROP_GENERATION_ATTEMPTS_PER_PLOT,
        "tuning_crop_count": 16 * crops_per_plot,
        "tuning_data_root": str(tuning_root.resolve()),
        "label_normalisation": {
            "positive_tree": "classification in [4,5,6] and treeID > 0",
            "non_tree": "classification in [1,2] maps to instance label 0",
            "ignored": "all other points map to instance label -1",
        },
        "training_budget": {
            "epochs": TRAINING_EPOCHS,
            "examples_per_epoch": EXAMPLES_PER_EPOCH,
            "examples_seen": TRAINING_EPOCHS * EXAMPLES_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "optimizer_steps": TRAINING_EPOCHS * (EXAMPLES_PER_EPOCH // BATCH_SIZE),
            "paper_finetune_step_comparison": (
                "12,495 local optimizer steps; five fewer than the paper's "
                "12,500-iteration fine-tuning schedule"
            ),
            "tuning_trials": len(PROFILES) * len(SEEDS),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "selection_rule": (
                "fixed full_lr_1e-5 configuration, seed 42 and epoch 35; "
                "seven additional seeds and earlier checkpoints are diagnostics only"
            ),
            "tuning_budget_role": "one_fixed_configuration_with_seed_replicates",
        },
        "held_out_test_accessed": False,
        "plots": frozen_rows,
        "trials": trials,
        "next_gate": "generate_and_consolidate_16_training_plot_crops",
    }
    write_json(run_root / "long_finetune_freeze.json", freeze)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-template", required=True, type=Path)
    parser.add_argument("--crops-per-plot", type=int, default=CROPS_PER_PLOT)
    args = parser.parse_args()
    result = prepare(
        args.development_manifest.expanduser().resolve(),
        args.run_root.expanduser().resolve(),
        args.treelearn_repo.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.evaluation_template.expanduser().resolve(),
        crops_per_plot=args.crops_per_plot,
    )
    print(f'run_id={result["run_id"]}')
    print(f'training_plots={result["training_plots"]}')
    print(f'validation_plots={result["validation_plots"]}')
    print(f'trials={len(result["trials"])}')
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
