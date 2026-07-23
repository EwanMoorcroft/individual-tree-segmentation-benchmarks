"""Convert one frozen development plot to the official ForestFormer3D format."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import (  # noqa: E402
    load_and_verify_manifest_plot,
    sha256_file,
)

SAMPLE_ID = "forestformer3d_development_test"
TREE_CLASSES = (4, 5, 6)


def _write_info(path: Path) -> None:
    payload = {
        "metainfo": {
            "categories": {"tree": 0},
            "dataset": "forainetv2",
            "info_version": "1.1",
        },
        "data_list": [
            {
                "sample_idx": SAMPLE_ID,
                "lidar_points": {
                    "num_pts_feats": 3,
                    "lidar_path": f"{SAMPLE_ID}.bin",
                },
                "pts_semantic_mask_path": "reference.bin",
                "pts_instance_mask_path": "reference.bin",
                "axis_align_matrix": np.eye(4, dtype=np.float64).tolist(),
                "instances": [],
            }
        ],
    }
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=4)


def prepare(manifest_path: Path, task_index: int, output_root: Path) -> dict[str, object]:
    import laspy

    _, row = load_and_verify_manifest_plot(
        manifest_path,
        task_index=task_index,
        expected_split="development",
        allow_held_out_test=False,
    )
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing staging root: {output_root}")
    cloud = laspy.read(row["input_las"])
    dimensions = set(cloud.point_format.dimension_names)
    missing = {"classification", "treeID"} - dimensions
    if missing:
        raise ValueError(f"Development LAS is missing fields: {sorted(missing)}")
    point_count = len(cloud.points)
    if point_count != row["point_count"]:
        raise ValueError("Manifest/source point count mismatch")
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Development coordinates contain non-finite values")
    offsets = np.array(
        (np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.min(xyz[:, 2])),
        dtype=np.float64,
    )
    model_xyz = (xyz - offsets).astype(np.float32)
    classification = np.asarray(cloud.classification, dtype=np.int16)
    tree_id = np.asarray(cloud["treeID"], dtype=np.int64)
    tree_mask = np.isin(classification, TREE_CLASSES) & (tree_id > 0)
    reference_ids = np.unique(tree_id[tree_mask])
    if len(reference_ids) != row["reference_tree_count"]:
        raise ValueError("Manifest/source reference-tree count mismatch")
    semantic = tree_mask.astype(np.int64)
    instance = np.where(tree_mask, tree_id, 0).astype(np.int64)

    points = output_root / "points"
    semantics = output_root / "semantic_mask"
    instances = output_root / "instance_mask"
    points.mkdir(parents=True)
    semantics.mkdir()
    instances.mkdir()
    model_xyz.tofile(points / f"{SAMPLE_ID}.bin")
    semantic.tofile(semantics / "reference.bin")
    instance.tofile(instances / "reference.bin")
    _write_info(output_root / "reference.pkl")
    np.savez(
        output_root / "evaluation_sidecar.npz",
        model_xyz=model_xyz,
        classification=classification,
        target_tree_id=np.where(tree_mask, tree_id, -1).astype(np.int64),
        source_row_index=np.arange(point_count, dtype=np.int64),
        reference_semantic=semantic,
        reference_instance=instance,
        offsets=offsets,
    )
    names = (
        f"points/{SAMPLE_ID}.bin",
        "semantic_mask/reference.bin",
        "instance_mask/reference.bin",
        "reference.pkl",
        "evaluation_sidecar.npz",
    )
    result: dict[str, object] = {
        "schema": "forestformer3d_development_plot_input_v1",
        "split": "development",
        "held_out_access": False,
        "task_index": task_index,
        "plot_id": row["plot_id"],
        "safe_plot_id": row["safe_plot_id"],
        "relative_path": row["relative_path"],
        "source_sha256": row["input_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "point_count": point_count,
        "reference_tree_count": len(reference_ids),
        "source_row_index": "zero_based_identity",
        "artifacts": {
            name: {
                "size_bytes": (output_root / name).stat().st_size,
                "sha256": sha256_file(output_root / name),
            }
            for name in names
        },
    }
    (output_root / "input_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "input_preparation.complete").touch(exist_ok=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.manifest, args.task_index, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
