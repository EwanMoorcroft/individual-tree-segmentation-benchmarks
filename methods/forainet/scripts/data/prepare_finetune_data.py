"""Prepare the fixed FOR-instance development split for official ForAINet training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from plyfile import PlyData, PlyElement


EXPECTED_MANIFEST_SCHEMA = "forainet_development_manifest_v1"
EXPECTED_DEVELOPMENT_PLOTS = 21
EXPECTED_TRAINING_PLOTS = 16
EXPECTED_VALIDATION_PLOTS = 5
EXPECTED_SPLIT_SEED = 42
EXPECTED_UPSTREAM_COMMIT = "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
OFFICIAL_INSTANCE_CLASSES = (3, 4, 5, 6)
OFFICIAL_SEMANTIC_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    5: 4,
    6: 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_roles(
    plots: list[dict[str, Any]], seed: int = EXPECTED_SPLIT_SEED
) -> list[dict[str, Any]]:
    """Reproduce the official random.sample split over the 21 dev rows."""

    if seed != EXPECTED_SPLIT_SEED:
        raise ValueError("ForAINet fine-tuning requires the frozen split seed 42")
    if len(plots) != EXPECTED_DEVELOPMENT_PLOTS:
        raise ValueError("fine-tuning requires exactly 21 development plots")
    if any(row.get("split") != "dev" for row in plots):
        raise ValueError("fine-tuning manifest contains a non-development row")
    validation_indices = set(
        random.Random(seed).sample(range(len(plots)), EXPECTED_VALIDATION_PLOTS)
    )
    assigned = []
    for index, row in enumerate(plots):
        assigned.append(
            {
                **row,
                "training_role": (
                    "validation" if index in validation_indices else "train"
                ),
            }
        )
    return assigned


def official_keep_mask(
    classification: np.ndarray, tree_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the pinned official Setting-1 point-selection contract."""

    stuff_mask = ~np.isin(classification, OFFICIAL_INSTANCE_CLASSES)
    stuff_instance_ids = np.unique(tree_id[stuff_mask])
    keep = classification != 3
    for source_class in OFFICIAL_INSTANCE_CLASSES:
        keep &= ~(
            (classification == source_class)
            & np.isin(tree_id, stuff_instance_ids)
        )
    return keep, stuff_instance_ids


def convert(
    *,
    source: Path,
    output: Path,
    expected_sha256: str,
    expected_point_count: int,
) -> dict[str, Any]:
    """Write the six fields consumed by the checkpoint-saved dataset route."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256(source) != expected_sha256:
        raise ValueError(f"development source hash changed: {source}")
    cloud = laspy.read(source)
    if len(cloud.points) != expected_point_count:
        raise ValueError(f"development source point count changed: {source}")
    dimensions = set(cloud.point_format.dimension_names)
    if not {"X", "Y", "Z", "intensity", "classification", "treeID"} <= dimensions:
        raise ValueError(f"required LAS dimensions absent: {source}")

    classification = np.asarray(cloud.classification, dtype=np.int64)
    tree_id = np.asarray(cloud["treeID"], dtype=np.int64)
    unsupported = sorted(
        set(np.unique(classification)) - set(OFFICIAL_SEMANTIC_MAPPING) - {3}
    )
    if unsupported:
        raise ValueError(f"unsupported FOR-instance classes {unsupported}: {source}")
    keep, stuff_instance_ids = official_keep_mask(classification, tree_id)
    kept_classification = classification[keep]
    semantic = np.full(kept_classification.shape, 20, dtype=np.float32)
    for source_class, model_label in OFFICIAL_SEMANTIC_MAPPING.items():
        semantic[kept_classification == source_class] = model_label
    if np.any(semantic == 20):
        raise ValueError(f"official semantic remap incomplete: {source}")

    vertices = np.zeros(
        int(keep.sum()),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("intensity", "f4"),
            ("semantic_seg", "f4"),
            ("treeID", "f4"),
        ],
    )
    scales = np.asarray(cloud.header.scales, dtype=np.float64)
    vertices["x"] = (np.asarray(cloud.X)[keep] * scales[0]).astype(np.float32)
    vertices["y"] = (np.asarray(cloud.Y)[keep] * scales[1]).astype(np.float32)
    vertices["z"] = (np.asarray(cloud.Z)[keep] * scales[2]).astype(np.float32)
    vertices["intensity"] = np.asarray(cloud.intensity, dtype=np.float32)[keep]
    vertices["semantic_seg"] = semantic
    vertices["treeID"] = tree_id[keep].astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(
                vertices,
                "vertex",
                comments=["Official ForAINet Setting-1 FOR-data mapping"],
            )
        ],
        byte_order="<",
    ).write(output)

    return {
        "source_sha256": expected_sha256,
        "source_point_count": int(len(classification)),
        "output_sha256": sha256(output),
        "output_point_count": int(keep.sum()),
        "dropped_class_3_points": int((classification == 3).sum()),
        "dropped_instance_class_without_instance_points": int(
            (
                np.isin(classification, OFFICIAL_INSTANCE_CLASSES)
                & np.isin(tree_id, stuff_instance_ids)
                & (classification != 3)
            ).sum()
        ),
        "stuff_instance_ids": [int(value) for value in stuff_instance_ids],
        "source_class_counts": {
            str(int(value)): int((classification == value).sum())
            for value in np.unique(classification)
        },
        "model_semantic_counts": {
            str(int(value)): int((semantic == value).sum())
            for value in np.unique(semantic)
        },
        "coordinate_rule": "las_integer_coordinate_times_scale_without_offset",
    }


def prepare(
    *,
    development_manifest: Path,
    development_final_gate: Path,
    dataset_root: Path,
    output_root: Path,
    seed: int = EXPECTED_SPLIT_SEED,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing existing fine-tune root: {output_root}")
    source_manifest = json.loads(
        development_manifest.read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA
        or source_manifest.get("status") != "complete"
        or source_manifest.get("held_out_paths_included") is not False
    ):
        raise ValueError("source development manifest is not frozen and test-locked")
    final_gate = json.loads(development_final_gate.read_text(encoding="utf-8"))
    if (
        final_gate.get("status") != "complete"
        or final_gate.get("held_out_access") is not False
    ):
        raise ValueError("published development final gate has not passed")

    assigned = assign_roles(source_manifest["plots"], seed)
    output_root.mkdir(parents=True)
    records = []
    for row in assigned:
        relative = Path(row["relative_path"])
        source = dataset_root / relative
        role_suffix = "_val" if row["training_role"] == "validation" else ""
        output_name = f"{relative.parent.name}_{relative.stem}{role_suffix}.ply"
        output = (
            output_root
            / "data_set1_5classes"
            / "treeinsfused"
            / "raw"
            / relative.parent.name
            / output_name
        )
        conversion = convert(
            source=source,
            output=output,
            expected_sha256=row["source_sha256"],
            expected_point_count=int(row["point_count"]),
        )
        records.append(
            {
                **row,
                "training_role": row["training_role"],
                "converted_relative_path": output.relative_to(output_root).as_posix(),
                **conversion,
            }
        )

    role_counts = {
        role: sum(row["training_role"] == role for row in records)
        for role in ("train", "validation")
    }
    if role_counts != {
        "train": EXPECTED_TRAINING_PLOTS,
        "validation": EXPECTED_VALIDATION_PLOTS,
    }:
        raise ValueError(f"unexpected fine-tuning role counts: {role_counts}")
    manifest = {
        "schema": "forainet_finetune_data_manifest_v1",
        "status": "complete",
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "source_development_manifest_sha256": sha256(development_manifest),
        "source_development_final_gate_sha256": sha256(development_final_gate),
        "split_seed": seed,
        "split_algorithm": "python_random_sample_over_manifest_order",
        "role_counts": role_counts,
        "held_out_paths_included": False,
        "held_out_access": False,
        "official_setting": "setting_1_5classes_remove_outpoints",
        "semantic_mapping": {
            str(key): value for key, value in OFFICIAL_SEMANTIC_MAPPING.items()
        },
        "class_3_removed": True,
        "checkpoint_input_features": [],
        "records": records,
    }
    manifest_path = output_root / "finetune_data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--development-final-gate", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=EXPECTED_SPLIT_SEED)
    args = parser.parse_args()
    manifest = prepare(
        development_manifest=args.development_manifest,
        development_final_gate=args.development_final_gate,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(f'training_plots={manifest["role_counts"]["train"]}')
    print(f'validation_plots={manifest["role_counts"]["validation"]}')
    print("held_out_access=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
