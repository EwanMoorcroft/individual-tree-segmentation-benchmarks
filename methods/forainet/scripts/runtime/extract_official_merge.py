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


def tile_coverage(tile_index_dir: Path, point_count: int) -> np.ndarray:
    index_paths = sorted(tile_index_dir.glob("tile_*_indices.txt"))
    if not index_paths:
        raise ValueError("official tiler produced no source-index carriers")
    chunks = []
    for path in index_paths:
        values = np.atleast_1d(np.loadtxt(path, dtype=np.int64))
        if values.size and (np.min(values) < 0 or np.max(values) >= point_count):
            raise ValueError(f"tile index is outside the source array: {path}")
        chunks.append(values)
    return np.unique(np.concatenate(chunks))


def extract(
    merged_ply: Path,
    inference_ply: Path,
    expected_point_count: int,
    tile_index_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    merged = vertex_data(merged_ply)
    source = vertex_data(inference_ply)
    required = {"x", "y", "z", "instance_preds", "semantic_preds"}
    if missing := required - set(merged.dtype.names or ()):
        raise ValueError(f"official merged PLY is missing fields: {sorted(missing)}")
    if len(source) != expected_point_count:
        raise ValueError("inference PLY does not contain every source point")
    source_names = set(source.dtype.names or ())
    if not {"x", "y", "z"} <= source_names:
        raise ValueError("inference PLY is missing coordinates")

    if len(merged) == expected_point_count:
        covered_indices = np.arange(expected_point_count, dtype=np.int64)
    else:
        if tile_index_dir is None:
            raise ValueError(
                "official merge omitted source rows but no tile-index carrier "
                "was supplied"
            )
        covered_indices = tile_coverage(tile_index_dir, expected_point_count)
        if len(merged) != len(covered_indices):
            raise ValueError(
                "official merge row count differs from official tile coverage"
            )

    merged_xyz = np.column_stack(
        [np.asarray(merged[name], dtype=np.float32) for name in ("x", "y", "z")]
    )
    source_xyz = np.column_stack(
        [np.asarray(source[name], dtype=np.float32) for name in ("x", "y", "z")]
    )
    exact_coordinate_order = bool(
        np.array_equal(merged_xyz, source_xyz[covered_indices])
    )
    if not exact_coordinate_order:
        raise ValueError(
            "official merged output is not in exact covered source-row order"
        )

    semantic = np.full(expected_point_count, -1, dtype=np.int64)
    instance = np.full(expected_point_count, -1, dtype=np.int64)
    semantic[covered_indices] = np.asarray(
        merged["semantic_preds"], dtype=np.int64
    )
    instance[covered_indices] = np.asarray(
        merged["instance_preds"], dtype=np.int64
    )
    uncovered_indices = np.setdiff1d(
        np.arange(expected_point_count, dtype=np.int64),
        covered_indices,
        assume_unique=True,
    )
    uncovered_index_sha256 = hashlib.sha256(
        uncovered_indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    arrays = {
        "source_row_index": np.arange(expected_point_count, dtype=np.int64),
        "pred_semantic_internal": semantic,
        "pred_instance_id": instance,
    }
    metadata = {
        "schema": "forainet_official_merge_extraction_v2",
        "status": "verified",
        "point_count": expected_point_count,
        "covered_source_point_count": int(len(covered_indices)),
        "uncovered_source_point_count": int(len(uncovered_indices)),
        "uncovered_source_row_indices": [
            int(index) for index in uncovered_indices
        ]
        if len(uncovered_indices) <= 1000
        else None,
        "uncovered_source_row_indices_sha256": uncovered_index_sha256,
        "uncovered_prediction_policy": "no_prediction_sentinel_minus_one",
        "correspondence": "official_tile_index_union_and_original_array_order",
        "source_row_index_carrier": "official_tile_indices",
        "coordinate_matching_used": False,
        "coordinate_order_validation": (
            "exact_float32_equality_against_covered_source_rows"
        ),
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
    parser.add_argument("--tile-index-dir", type=Path)
    parser.add_argument("--expected-point-count", required=True, type=int)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()
    for output in (args.output_npz, args.metadata_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    arrays, metadata = extract(
        args.merged_ply,
        args.inference_ply,
        args.expected_point_count,
        args.tile_index_dir,
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
