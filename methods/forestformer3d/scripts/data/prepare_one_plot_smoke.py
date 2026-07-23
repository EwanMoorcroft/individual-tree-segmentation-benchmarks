"""Prepare the frozen ForestFormer3D development-only smoke input."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from shared.for_instance_manifest import (
    EXPECTED_METADATA_SHA256,
    read_split_metadata,
    sha256_file,
)


RELATIVE_PATH = "CULS/plot_1_annotated.las"
TREE_CLASSES = (4, 5, 6)


def _write_info(path: Path, semantic_name: str, instance_name: str) -> None:
    payload = {
        "metainfo": {
            "categories": {"tree": 0},
            "dataset": "forainetv2",
            "info_version": "1.1",
        },
        "data_list": [
            {
                "sample_idx": "forestformer3d_smoke_test",
                "lidar_points": {
                    "num_pts_feats": 3,
                    "lidar_path": "forestformer3d_smoke_test.bin",
                },
                "pts_semantic_mask_path": semantic_name,
                "pts_instance_mask_path": instance_name,
                "axis_align_matrix": np.eye(4, dtype=np.float64).tolist(),
                "instances": [],
            }
        ],
    }
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=4)


def prepare(
    dataset_root: Path,
    output_root: Path,
    *,
    expected_point_count: int,
    expected_reference_tree_count: int,
    expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
) -> dict[str, object]:
    import laspy

    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing staging root: {output_root}")

    metadata_path = dataset_root / "data_split_metadata.csv"
    rows, metadata_sha256 = read_split_metadata(
        metadata_path, expected_sha256=expected_metadata_sha256
    )
    row = rows[RELATIVE_PATH]
    if row["split"] != "dev":
        raise ValueError(f"Smoke plot is not development data: {row}")

    source_path = dataset_root / RELATIVE_PATH
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing smoke plot: {source_path}")
    cloud = laspy.read(source_path)
    dimensions = set(cloud.point_format.dimension_names)
    missing = {"classification", "treeID"} - dimensions
    if missing:
        raise ValueError(f"Smoke LAS is missing dimensions: {sorted(missing)}")

    point_count = len(cloud.points)
    if point_count != expected_point_count:
        raise ValueError(
            f"Point-count mismatch: expected {expected_point_count}, found {point_count}"
        )
    source_xyz = np.column_stack(
        (
            np.asarray(cloud.x, dtype=np.float64),
            np.asarray(cloud.y, dtype=np.float64),
            np.asarray(cloud.z, dtype=np.float64),
        )
    )
    if not np.isfinite(source_xyz).all():
        raise ValueError("Smoke coordinates contain non-finite values")
    offsets = np.array(
        (
            np.mean(source_xyz[:, 0], dtype=np.float64),
            np.mean(source_xyz[:, 1], dtype=np.float64),
            np.min(source_xyz[:, 2]),
        ),
        dtype=np.float64,
    )
    model_xyz = (source_xyz - offsets).astype(np.float32)

    classification = np.asarray(cloud.classification, dtype=np.int16)
    source_tree_id = np.asarray(cloud["treeID"], dtype=np.int64)
    tree_mask = np.isin(classification, TREE_CLASSES) & (source_tree_id > 0)
    reference_ids = np.unique(source_tree_id[tree_mask])
    if len(reference_ids) != expected_reference_tree_count:
        raise ValueError(
            "Reference-tree-count mismatch: "
            f"expected {expected_reference_tree_count}, found {len(reference_ids)}"
        )

    # ForestFormer3D's prepared masks are internal 0=ground, 1=wood, 2=leaf.
    # The shared FOR-instance scoring classes 4/5/6 are all tree material and
    # are mapped to wood. All other points are loader-required background.
    reference_semantic = tree_mask.astype(np.int64)
    reference_instance = np.where(tree_mask, source_tree_id, 0).astype(np.int64)
    dummy_semantic = np.zeros(point_count, dtype=np.int64)
    dummy_instance = np.zeros(point_count, dtype=np.int64)

    points_dir = output_root / "points"
    semantic_dir = output_root / "semantic_mask"
    instance_dir = output_root / "instance_mask"
    points_dir.mkdir(parents=True)
    semantic_dir.mkdir()
    instance_dir.mkdir()

    model_xyz.tofile(points_dir / "forestformer3d_smoke_test.bin")
    reference_semantic.tofile(semantic_dir / "reference.bin")
    reference_instance.tofile(instance_dir / "reference.bin")
    dummy_semantic.tofile(semantic_dir / "dummy.bin")
    dummy_instance.tofile(instance_dir / "dummy.bin")
    _write_info(output_root / "reference.pkl", "reference.bin", "reference.bin")
    _write_info(output_root / "dummy.pkl", "dummy.bin", "dummy.bin")

    np.savez(
        output_root / "evaluation_sidecar.npz",
        source_xyz=source_xyz,
        model_xyz=model_xyz,
        classification=classification,
        source_tree_id=source_tree_id,
        target_tree_id=np.where(tree_mask, source_tree_id, -1).astype(np.int64),
        source_row_index=np.arange(point_count, dtype=np.int64),
        reference_semantic=reference_semantic,
        reference_instance=reference_instance,
        offsets=offsets,
    )

    artifacts = [
        "points/forestformer3d_smoke_test.bin",
        "semantic_mask/reference.bin",
        "semantic_mask/dummy.bin",
        "instance_mask/reference.bin",
        "instance_mask/dummy.bin",
        "reference.pkl",
        "dummy.pkl",
        "evaluation_sidecar.npz",
    ]
    manifest: dict[str, object] = {
        "schema": "forestformer3d_one_plot_smoke_input_v1",
        "split": "development",
        "relative_path": RELATIVE_PATH,
        "held_out_access": False,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "split_metadata_sha256": metadata_sha256,
        "point_count": point_count,
        "reference_tree_count": len(reference_ids),
        "reference_tree_classes": list(TREE_CLASSES),
        "semantic_mapping": {
            "classes_4_5_6": "forestformer3d_internal_wood_1",
            "all_other_classes": "forestformer3d_internal_ground_0",
        },
        "offsets_xyz": offsets.tolist(),
        "source_row_index": "zero_based_identity",
        "artifacts": {
            relative: {
                "size_bytes": (output_root / relative).stat().st_size,
                "sha256": sha256_file(output_root / relative),
            }
            for relative in artifacts
        },
    }
    manifest_path = output_root / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "input_preparation.complete").touch(exist_ok=False)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-point-count", required=True, type=int)
    parser.add_argument("--expected-reference-tree-count", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare(
        args.dataset_root,
        args.output_root,
        expected_point_count=args.expected_point_count,
        expected_reference_tree_count=args.expected_reference_tree_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
