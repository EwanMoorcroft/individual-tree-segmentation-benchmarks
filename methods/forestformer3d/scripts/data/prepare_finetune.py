"""Freeze and stage the ForestFormer3D development-only fine-tuning data."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import sha256_file


EXPECTED_DEVELOPMENT_PLOTS = 21
EXPECTED_TRAIN_PLOTS = 16
EXPECTED_VALIDATION_PLOTS = 5
SPLIT_SEED = 42
MAX_EPOCHS = 35
BATCH_SIZE = 2
CHECKPOINT_EPOCHS = (7, 14, 21, 28, 35)
LEARNING_RATE = 1e-5


def assign_roles(plots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if (
        len(plots) != EXPECTED_DEVELOPMENT_PLOTS
        or any(row.get("dataset_split") != "development" for row in plots)
    ):
        raise ValueError("Fine-tuning requires exactly 21 development plots")
    validation_indices = set(
        random.Random(SPLIT_SEED).sample(
            range(EXPECTED_DEVELOPMENT_PLOTS),
            EXPECTED_VALIDATION_PLOTS,
        )
    )
    return [
        {
            **row,
            "fine_tune_role": (
                "validation" if index in validation_indices else "train"
            ),
        }
        for index, row in enumerate(plots)
    ]


def _info(sample_id: str) -> dict[str, Any]:
    return {
        "sample_idx": sample_id,
        "lidar_points": {
            "num_pts_feats": 3,
            "lidar_path": f"{sample_id}.bin",
        },
        "pts_semantic_mask_path": f"{sample_id}.bin",
        "pts_instance_mask_path": f"{sample_id}.bin",
        "axis_align_matrix": np.eye(4, dtype=np.float64).tolist(),
        "instances": [],
    }


def _write_infos(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {
        "metainfo": {
            "categories": {"tree": 0},
            "dataset": "forainetv2",
            "info_version": "1.1",
        },
        "data_list": [_info(row["safe_plot_id"]) for row in rows],
    }
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=4)


def _copy_exclusive(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)


def prepare(
    source_run_root: Path,
    verification_json: Path,
    output_root: Path,
    *,
    benchmark_commit: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    source_run_root = source_run_root.expanduser().resolve()
    verification_json = verification_json.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing fine-tuning root: {output_root}")
    if not (source_run_root / "development.complete").is_file():
        raise FileNotFoundError("Source development run is not complete")

    manifest_path = source_run_root / "development_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_json.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "forestformer3d_development_manifest_v1"
        or manifest.get("dataset_split") != "development"
        or manifest.get("held_out_access") is not False
        or len(manifest.get("plots", [])) != EXPECTED_DEVELOPMENT_PLOTS
    ):
        raise ValueError("Invalid source development manifest")
    if (
        verification.get("schema")
        != "forestformer3d_development_verification_v1"
        or verification.get("status") != "verified"
        or verification.get("run_id") != source_run_root.name
        or verification.get("held_out_access") is not False
        or verification.get("exact_source_row_alignment") is not True
        or verification.get("task_count") != EXPECTED_DEVELOPMENT_PLOTS
    ):
        raise ValueError("Independent source-run verification did not pass")

    rows = assign_roles(manifest["plots"])
    train_rows = [row for row in rows if row["fine_tune_role"] == "train"]
    validation_rows = [
        row for row in rows if row["fine_tune_role"] == "validation"
    ]
    if (
        len(train_rows) != EXPECTED_TRAIN_PLOTS
        or len(validation_rows) != EXPECTED_VALIDATION_PLOTS
    ):
        raise AssertionError("Canonical split count mismatch")

    data_root = output_root / "data/ForAINetV2"
    points_root = data_root / "points"
    semantic_root = data_root / "semantic_mask"
    instance_root = data_root / "instance_mask"
    for path in (points_root, semantic_root, instance_root):
        path.mkdir(parents=True)

    source_artifacts: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["safe_plot_id"]
        task_root = source_run_root / "tasks" / sample_id / "staged_input"
        input_manifest_path = task_root / "input_manifest.json"
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if (
            input_manifest.get("plot_id") != row["plot_id"]
            or input_manifest.get("relative_path") != row["relative_path"]
            or input_manifest.get("split") != "development"
            or input_manifest.get("held_out_access") is not False
            or input_manifest.get("source_row_index") != "zero_based_identity"
        ):
            raise ValueError(f"Invalid source task input: {row['plot_id']}")
        sources = {
            "points": (
                task_root / "points/forestformer3d_development_test.bin",
                points_root / f"{sample_id}.bin",
            ),
            "semantic_mask": (
                task_root / "semantic_mask/reference.bin",
                semantic_root / f"{sample_id}.bin",
            ),
            "instance_mask": (
                task_root / "instance_mask/reference.bin",
                instance_root / f"{sample_id}.bin",
            ),
        }
        for role, (source, destination) in sources.items():
            _copy_exclusive(source, destination)
            source_artifacts.append(
                {
                    "plot_id": row["plot_id"],
                    "fine_tune_role": row["fine_tune_role"],
                    "logical_role": role,
                    "source_relative_path": source.relative_to(
                        source_run_root
                    ).as_posix(),
                    "source_sha256": sha256_file(source),
                    "size_bytes": source.stat().st_size,
                    "staged_relative_path": destination.relative_to(
                        output_root
                    ).as_posix(),
                }
            )

    _write_infos(data_root / "for_instance_finetune_train.pkl", train_rows)
    _write_infos(
        data_root / "for_instance_finetune_validation.pkl", validation_rows
    )
    _write_infos(data_root / "for_instance_finetune_smoke.pkl", train_rows[:1])

    split_path = output_root / "fine_tune_split.csv"
    fields = [
        "task_index",
        "plot_id",
        "safe_plot_id",
        "relative_path",
        "fine_tune_role",
        "point_count",
        "input_sha256",
    ]
    with split_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)

    examples_per_epoch = len(train_rows)
    optimizer_steps_per_epoch = examples_per_epoch // BATCH_SIZE
    freeze: dict[str, Any] = {
        "schema": "forestformer3d_fine_tune_freeze_v1",
        "status": "frozen_for_initialisation_smoke",
        "method": "ForestFormer3D",
        "training_mode": "fine_tuned_on_dev",
        "learning_regime": "supervised",
        "dataset_exposure": ["published_checkpoint", "development_tuned"],
        "source_development_run_id": source_run_root.name,
        "source_development_manifest_sha256": sha256_file(manifest_path),
        "source_verification_sha256": sha256_file(verification_json),
        "benchmark_commit": benchmark_commit,
        "upstream_commit": "6a75c3735e4a4108d02ee944a8b93177f2360a4f",
        "upstream_config": "configs/oneformer3d_qs_radius16_qp300_2many.py",
        "initial_checkpoint_sha256": checkpoint_sha256,
        "split": {
            "dataset_split": "development",
            "seed": SPLIT_SEED,
            "algorithm": "random.Random(seed).sample(range(21), 5)",
            "training_plots": len(train_rows),
            "validation_plots": len(validation_rows),
            "held_out_access": False,
            "split_csv_sha256": sha256_file(split_path),
        },
        "training": {
            "configuration_count": 1,
            "epochs": MAX_EPOCHS,
            "examples_per_epoch": examples_per_epoch,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": 1,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "total_examples": examples_per_epoch * MAX_EPOCHS,
            "total_optimizer_steps": optimizer_steps_per_epoch * MAX_EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.05,
            "scheduler": "PolyLR",
            "scheduler_power": 0.9,
            "precision": "float32",
            "seed": SPLIT_SEED,
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "model_prepare_epoch": -1,
            "architecture_changes": False,
            "official_radius_m": 16,
            "official_query_count": 300,
            "official_voxel_size_m": 0.2,
        },
        "selection": {
            "evaluated_checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "validation_plots_per_checkpoint": EXPECTED_VALIDATION_PLOTS,
            "primary_rule": "maximum_mean_plot_f1",
            "tie_breakers": [
                "maximum_micro_f1",
                "earliest_checkpoint_epoch",
            ],
            "evaluation_protocol": "for_instance_pointwise_v1",
            "held_out_metrics_permitted": False,
        },
        "source_artifact_count": len(source_artifacts),
        "source_artifacts": source_artifacts,
    }
    (output_root / "fine_tune_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "preparation.sha256").write_text(
        "\n".join(
            f"{sha256_file(output_root / name)}  {name}"
            for name in (
                "fine_tune_freeze.json",
                "fine_tune_split.csv",
                "data/ForAINetV2/for_instance_finetune_train.pkl",
                "data/ForAINetV2/for_instance_finetune_validation.pkl",
                "data/ForAINetV2/for_instance_finetune_smoke.pkl",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "preparation.complete").touch(exist_ok=False)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--verification-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    result = prepare(
        args.source_run_root,
        args.verification_json,
        args.output_root,
        benchmark_commit=args.benchmark_commit,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
