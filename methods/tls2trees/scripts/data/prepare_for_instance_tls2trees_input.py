"""Prepare one development plot for a provenance-controlled TLS2trees pipeline.

The adapter tiles label-stripped geometry on a 10 m grid, selects the source
point nearest each 2 cm voxel centre, and writes a source-row projection map.
Reference semantic and instance fields are never written to TLS2trees input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from benchmark.ply_io import write_xyz_ply
from shared.for_instance_manifest import load_and_verify_manifest_plot


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPECTED_VARIANT = "published_default"
ALLOWED_DEVELOPMENT_VARIANTS = {EXPECTED_VARIANT, "development_tuned"}
EXPECTED_SPLIT = "development"
DEFAULT_TILE_SIZE_M = 10.0
DEFAULT_VOXEL_SIZE_M = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be one safe path component: {value!r}")
    return value


def normalise_split(value: str) -> str:
    aliases = {"dev": "development", "development": "development", "test": "test"}
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported split: {value!r}"
        ) from exc


def load_manifest_plot(
    manifest_path: Path,
    task_index: int,
    split: str = EXPECTED_SPLIT,
    *,
    allow_held_out_test: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_and_verify_manifest_plot(
        manifest_path,
        task_index=task_index,
        expected_split=split,
        allow_held_out_test=allow_held_out_test,
    )


def local_origin(coordinates: np.ndarray, tile_size_m: float) -> np.ndarray:
    """Return a grid-aligned origin that keeps processing coordinates small."""

    minima = np.min(coordinates, axis=0)
    return np.floor(minima / tile_size_m) * tile_size_m


def select_voxel_representatives(
    local_xyz: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the source row nearest each voxel centre deterministically.

    Returns representative source rows, source-to-representative indices and
    the integer XYZ voxel key for each representative.
    """

    if local_xyz.ndim != 2 or local_xyz.shape[1] != 3:
        raise ValueError(f"Expected XYZ array with shape (N, 3), found {local_xyz.shape}")
    if len(local_xyz) == 0:
        raise ValueError("Cannot prepare an empty point cloud")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be finite and greater than zero")

    scaled = local_xyz / voxel_size_m
    voxel_keys = np.floor(np.nextafter(scaled, np.inf)).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    centres = (voxel_keys.astype(np.float64) + 0.5) * voxel_size_m
    squared_distance = np.einsum(
        "ij,ij->i", local_xyz - centres, local_xyz - centres
    )
    distance_rank = np.round(squared_distance / (voxel_size_m**2), decimals=15)
    source_rows = np.arange(len(local_xyz), dtype=np.int64)
    order = np.lexsort((source_rows, distance_rank, inverse))
    ordered_groups = inverse[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    first[1:] = ordered_groups[1:] != ordered_groups[:-1]
    representative_source_rows = order[first].astype(np.int64, copy=False)
    if len(representative_source_rows) != len(unique_keys):
        raise RuntimeError("Internal voxel representative count mismatch")
    return representative_source_rows, inverse.astype(np.int64, copy=False), unique_keys


def tile_representatives(
    representative_xyz: np.ndarray,
    tile_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(tile_size_m) or tile_size_m <= 0:
        raise ValueError("tile_size_m must be finite and greater than zero")
    tile_xy = np.floor(
        np.nextafter(representative_xyz[:, :2] / tile_size_m, np.inf)
    ).astype(np.int64)
    unique_tiles, inverse = np.unique(tile_xy, axis=0, return_inverse=True)
    return unique_tiles, inverse.astype(np.int32, copy=False)


def read_source_geometry(input_las: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import laspy

    with laspy.open(input_las) as reader:
        header = reader.header
        dimensions = list(header.point_format.dimension_names)
        point_count = int(header.point_count)
        scales = np.asarray(header.scales, dtype=np.float64)
        offsets = np.asarray(header.offsets, dtype=np.float64)
    cloud = laspy.read(input_las)
    xyz = np.column_stack(
        (
            np.asarray(cloud.x, dtype=np.float64),
            np.asarray(cloud.y, dtype=np.float64),
            np.asarray(cloud.z, dtype=np.float64),
        )
    )
    if len(xyz) != point_count:
        raise ValueError(
            f"LAS point-count mismatch: header={point_count}, decoded={len(xyz)}"
        )
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Source LAS contains non-finite coordinates: {input_las}")
    return xyz, {
        "dimensions": dimensions,
        "point_count": point_count,
        "scales": scales,
        "offsets": offsets,
    }


def prepare_plot(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str = EXPECTED_VARIANT,
    split: str = EXPECTED_SPLIT,
    tile_size_m: float = DEFAULT_TILE_SIZE_M,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    """Prepare one immutable TLS2trees plot root."""

    manifest_path = manifest_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    resolved_split = normalise_split(split)
    if variant not in ALLOWED_DEVELOPMENT_VARIANTS:
        raise ValueError(
            "This entrypoint only prepares published_default or development_tuned "
            f"development inputs, received {variant!r}"
        )
    if resolved_split == "test" and not allow_held_out_test:
        raise ValueError(
            "Held-out test input requires --allow-held-out-test"
        )
    safe_component(run_id, "run_id")
    manifest, row = (
        load_manifest_plot(manifest_path, task_index)
        if resolved_split == EXPECTED_SPLIT
        else load_manifest_plot(
            manifest_path,
            task_index,
            resolved_split,
            allow_held_out_test=True,
        )
    )
    safe_plot_id = safe_component(str(row["safe_plot_id"]), "safe_plot_id")
    input_las = Path(str(row["input_las"])).expanduser().resolve()
    if not input_las.is_file():
        raise FileNotFoundError(f"Manifest LAS does not exist: {input_las}")
    if input_las.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Expected LAS/LAZ input, found: {input_las}")
    observed_input_sha256 = str(
        row.get("observed_input_sha256") or sha256(input_las)
    )
    if observed_input_sha256 != row["input_sha256"]:
        raise ValueError(
            "Manifest/source input SHA-256 mismatch before conversion: "
            f"expected {row['input_sha256']}, found {observed_input_sha256}"
        )

    plot_root = (
        output_root
        / "tls2trees"
        / "for_instance"
        / variant
        / resolved_split
        / run_id
        / safe_plot_id
    )
    if plot_root.exists():
        raise FileExistsError(
            f"Immutable plot root already exists; use a new run_id: {plot_root}"
        )
    converted_root = plot_root / "converted"
    tiles_root = converted_root / "tiles"
    tiles_root.mkdir(parents=True)

    source_xyz, las_info = read_source_geometry(input_las)
    if len(source_xyz) != int(row["point_count"]):
        raise ValueError(
            "Manifest/source point-count mismatch: "
            f"manifest={row['point_count']}, source={len(source_xyz)}"
        )
    origin = local_origin(source_xyz, tile_size_m)
    local_xyz = source_xyz - origin
    round_trip_delta = float(np.max(np.abs((local_xyz + origin) - source_xyz)))

    representative_source_rows, source_to_representative, representative_voxels = (
        select_voxel_representatives(local_xyz, voxel_size_m)
    )
    representative_xyz = local_xyz[representative_source_rows]
    unique_tiles, representative_tile_index = tile_representatives(
        representative_xyz, tile_size_m
    )

    tile_records: list[dict[str, Any]] = []
    tile_index_lines: list[str] = []
    tile_name_width = max(6, len(str(max(len(unique_tiles) - 1, 0))))
    for tile_index, tile_xy in enumerate(unique_tiles):
        tile_id = str(tile_index).zfill(tile_name_width)
        mask = representative_tile_index == tile_index
        tile_xyz = representative_xyz[mask]
        tile_path = tiles_root / f"{tile_id}.downsample.ply"
        write_xyz_ply(tile_path, tile_xyz)
        centre_x = (float(tile_xy[0]) + 0.5) * tile_size_m
        centre_y = (float(tile_xy[1]) + 0.5) * tile_size_m
        centre_z = float(np.mean(tile_xyz[:, 2]))
        tile_index_lines.append(
            f"{tile_id} {centre_x:.9f} {centre_y:.9f} {centre_z:.9f} {tile_path}\n"
        )
        tile_records.append(
            {
                "tile_index": tile_index,
                "tile_id": tile_id,
                "grid_xy": [int(tile_xy[0]), int(tile_xy[1])],
                "representative_point_count": int(np.count_nonzero(mask)),
                "path": str(tile_path),
                "sha256": sha256(tile_path),
            }
        )

    tile_index_path = converted_root / "tile_index.dat"
    tile_index_path.write_text("".join(tile_index_lines), encoding="utf-8")
    source_map_path = converted_root / "source_map.npz"
    np.savez_compressed(
        source_map_path,
        source_row_index=np.arange(len(source_xyz), dtype=np.int64),
        source_to_representative_index=source_to_representative,
        representative_source_row_index=representative_source_rows,
        representative_local_xyz=representative_xyz.astype(np.float64, copy=False),
        representative_voxel_xyz=representative_voxels,
        representative_tile_index=representative_tile_index,
        local_origin_xyz=origin,
        las_scales=las_info["scales"],
        las_offsets=las_info["offsets"],
    )

    metadata_path = converted_root / "conversion_metadata.json"
    metadata = {
        "schema_version": 1,
        "status": "prepared",
        "generated_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": variant,
        "split": resolved_split,
        "run_id": run_id,
        "task_index": task_index,
        "relative_path": row["relative_path"],
        "collection": row["collection"],
        "safe_plot_id": safe_plot_id,
        "input_las": str(input_las),
        "input_sha256": observed_input_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "split_metadata_sha256": manifest.get("split_metadata_sha256"),
        "source_point_count": len(source_xyz),
        "manifest_reference_tree_count": int(row["reference_tree_count"]),
        "representative_point_count": len(representative_xyz),
        "representative_retention_fraction": len(representative_xyz) / len(source_xyz),
        "tile_count": len(unique_tiles),
        "tile_size_m": tile_size_m,
        "downsample_voxel_size_m": voxel_size_m,
        "downsample_rule": "source_point_nearest_voxel_centre_then_lowest_source_row",
        "point_correspondence": "source_row_via_voxel_representative",
        "source_map": str(source_map_path),
        "source_map_sha256": sha256(source_map_path),
        "tile_index": str(tile_index_path),
        "tile_index_sha256": sha256(tile_index_path),
        "coordinate_frame": {
            "input": "source_crs",
            "processing": "grid_aligned_local_shift",
            "local_origin_xyz": origin.tolist(),
            "maximum_round_trip_delta_m": round_trip_delta,
            "las_scales": las_info["scales"].tolist(),
            "las_offsets": las_info["offsets"].tolist(),
        },
        "input_dimensions": las_info["dimensions"],
        "method_input_dimensions": ["x", "y", "z"],
        "reference_fields_passed_to_method": [],
        "labels_stripped": True,
        "held_out_test_accessed": resolved_split == "test",
        "tiles": tile_records,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["plot_root"] = str(plot_root)
    metadata["conversion_metadata"] = str(metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one development plot for published TLS2trees inference."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default=EXPECTED_VARIANT)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    parser.add_argument("--tile-size-m", type=float, default=DEFAULT_TILE_SIZE_M)
    parser.add_argument("--voxel-size-m", type=float, default=DEFAULT_VOXEL_SIZE_M)
    parser.add_argument("--allow-held-out-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = prepare_plot(
        manifest_path=Path(args.manifest_json),
        task_index=args.task_index,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        variant=args.variant,
        split=args.split,
        tile_size_m=args.tile_size_m,
        voxel_size_m=args.voxel_size_m,
        allow_held_out_test=args.allow_held_out_test,
    )
    print(f"plot_root={metadata['plot_root']}")
    print(f"conversion_metadata={metadata['conversion_metadata']}")
    print(f"source_points={metadata['source_point_count']}")
    print(f"representative_points={metadata['representative_point_count']}")
    print(f"tiles={metadata['tile_count']}")
    print(
        "held_out_test_accessed="
        + str(bool(metadata["held_out_test_accessed"])).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
