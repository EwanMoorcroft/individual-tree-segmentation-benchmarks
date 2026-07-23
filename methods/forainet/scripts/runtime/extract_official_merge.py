"""Extract a complete source-row prediction from the official tile merger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertex_data(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    return ply["vertex"].data


def extract(
    merged_ply: Path, inference_ply: Path, expected_point_count: int
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    merged = vertex_data(merged_ply)
    source = vertex_data(inference_ply)
    required = {"x", "y", "z", "instance_preds", "semantic_preds"}
    if missing := required - set(merged.dtype.names or ()):
        raise ValueError(f"official merged PLY is missing fields: {sorted(missing)}")
    if len(merged) != expected_point_count or len(source) != expected_point_count:
        raise ValueError(
            "official merge did not return exactly one row per source point"
        )
    source_names = set(source.dtype.names or ())
    if not {"x", "y", "z"} <= source_names:
        raise ValueError("inference PLY is missing coordinates")

    merged_xyz = np.column_stack(
        [np.asarray(merged[name], dtype=np.float32) for name in ("x", "y", "z")]
    )
    source_xyz = np.column_stack(
        [np.asarray(source[name], dtype=np.float32) for name in ("x", "y", "z")]
    )
    exact_coordinate_order = bool(np.array_equal(merged_xyz, source_xyz))
    if not exact_coordinate_order:
        raise ValueError(
            "official merged output is not in exact inference-Ply row order"
        )

    arrays = {
        "source_row_index": np.arange(expected_point_count, dtype=np.int64),
        "pred_semantic_internal": np.asarray(
            merged["semantic_preds"], dtype=np.int64
        ),
        "pred_instance_id": np.asarray(
            merged["instance_preds"], dtype=np.int64
        ),
    }
    metadata = {
        "schema": "forainet_official_merge_extraction_v1",
        "status": "verified",
        "point_count": expected_point_count,
        "correspondence": "official_merge_original_array_order",
        "source_row_index_carrier": "official_tile_indices_and_original_array_order",
        "coordinate_matching_used": False,
        "coordinate_order_validation": "exact_float32_equality",
        "coordinate_order_valid": exact_coordinate_order,
        "merged_ply_sha256": sha256(merged_ply),
        "inference_ply_sha256": sha256(inference_ply),
        "semantic_values": [
            int(value) for value in np.unique(arrays["pred_semantic_internal"])
        ],
        "instance_min": int(np.min(arrays["pred_instance_id"])),
        "instance_max": int(np.max(arrays["pred_instance_id"])),
    }
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-ply", required=True, type=Path)
    parser.add_argument("--inference-ply", required=True, type=Path)
    parser.add_argument("--expected-point-count", required=True, type=int)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()
    for output in (args.output_npz, args.metadata_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    arrays, metadata = extract(
        args.merged_ply, args.inference_ply, args.expected_point_count
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    metadata["raw_prediction_sha256"] = sha256(args.output_npz)
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
