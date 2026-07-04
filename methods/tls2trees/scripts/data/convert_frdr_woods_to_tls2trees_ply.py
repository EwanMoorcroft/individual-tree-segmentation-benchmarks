"""Convert FRDR treeiso LAZ plots to the TLS2trees PLY input schema."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import write_tls2trees_ply


FEASIBILITY_WARNING = (
    "n_z uses the input cloud's local minimum z. This is a feasibility approximation "
    "and is not terrain or ground normalisation."
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def value_key(value: Any) -> str:
    item = value.item() if hasattr(value, "item") else value
    return str(item)


def iter_converted_chunks(
    input_path: Path,
    wood_field: str,
    wood_value: float,
    nonwood_value: float,
    local_min_z: float,
    chunk_size: int,
    unknown_policy: str,
) -> Iterator[dict[str, np.ndarray]]:
    import laspy

    with laspy.open(input_path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            woods = np.asarray(points[wood_field])
            wood_mask = woods == wood_value
            nonwood_mask = woods == nonwood_value
            known_mask = wood_mask | nonwood_mask
            retained_mask = known_mask if unknown_policy == "drop" else np.ones(len(woods), dtype=bool)

            z = np.asarray(points.z, dtype=np.float64)[retained_mask]
            retained_wood = wood_mask[retained_mask]
            yield {
                "x": np.asarray(points.x, dtype=np.float64)[retained_mask],
                "y": np.asarray(points.y, dtype=np.float64)[retained_mask],
                "z": z,
                "n_z": z - local_min_z,
                "label": np.where(retained_wood, 3.0, 1.0),
            }


def scan_input(
    input_path: Path,
    wood_field: str,
    wood_value: float,
    nonwood_value: float,
    chunk_size: int,
    unknown_policy: str,
) -> dict[str, Any]:
    """Scan FRDR labels and coordinates while applying the unknown-value policy."""

    import laspy

    woods_counts: dict[str, int] = {}
    unknown_values: set[str] = set()
    retained_count = 0
    unknown_count = 0
    wood_count = 0
    nonwood_count = 0
    xyz_sum = np.zeros(3, dtype=np.float64)
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)

    with laspy.open(input_path) as reader:
        header = reader.header
        dimensions = list(header.point_format.dimension_names)
        if wood_field not in dimensions:
            raise ValueError(
                f"Input does not contain wood field {wood_field!r}; dimensions: {dimensions}"
            )
        original_count = int(header.point_count)

        for points in reader.chunk_iterator(chunk_size):
            woods = np.asarray(points[wood_field])
            unique, counts = np.unique(woods, return_counts=True)
            for value, count in zip(unique, counts):
                key = value_key(value)
                woods_counts[key] = woods_counts.get(key, 0) + int(count)

            wood_mask = woods == wood_value
            nonwood_mask = woods == nonwood_value
            known_mask = wood_mask | nonwood_mask
            unknown_mask = ~known_mask
            chunk_unknown_count = int(np.count_nonzero(unknown_mask))
            if chunk_unknown_count:
                unknown_values.update(value_key(value) for value in np.unique(woods[unknown_mask]))
                if unknown_policy == "fail":
                    raise ValueError(
                        f"Unexpected {wood_field} value(s) {sorted(unknown_values)}; "
                        "choose --unknown-policy drop or nonwood to handle them explicitly"
                    )

            retained_mask = known_mask if unknown_policy == "drop" else np.ones(len(woods), dtype=bool)
            retained_xyz = np.column_stack(
                [
                    np.asarray(points.x, dtype=np.float64)[retained_mask],
                    np.asarray(points.y, dtype=np.float64)[retained_mask],
                    np.asarray(points.z, dtype=np.float64)[retained_mask],
                ]
            )
            if len(retained_xyz):
                xyz_sum += retained_xyz.sum(axis=0)
                xyz_min = np.minimum(xyz_min, retained_xyz.min(axis=0))
                xyz_max = np.maximum(xyz_max, retained_xyz.max(axis=0))

            retained_count += int(np.count_nonzero(retained_mask))
            unknown_count += chunk_unknown_count
            wood_count += int(np.count_nonzero(wood_mask))
            nonwood_count += int(np.count_nonzero(nonwood_mask))

    if retained_count == 0:
        raise ValueError("No points remain after applying the unknown-value policy")

    dropped_unknown_count = unknown_count if unknown_policy == "drop" else 0
    label_counts = {
        "1": nonwood_count + (unknown_count if unknown_policy == "nonwood" else 0),
        "3": wood_count,
    }
    return {
        "input_dimensions": dimensions,
        "original_point_count": original_count,
        "retained_point_count": retained_count,
        "dropped_unknown_count": dropped_unknown_count,
        "unknown_point_count": unknown_count,
        "unknown_woods_values": sorted(unknown_values),
        "woods_counts": woods_counts,
        "wood_point_count": wood_count,
        "nonwood_point_count": label_counts["1"],
        "label_counts": label_counts,
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        "xyz_mean": (xyz_sum / retained_count).tolist(),
    }


def convert(
    input_path: Path,
    output_dir: Path,
    tile_name: str,
    wood_field: str,
    wood_value: float,
    nonwood_value: float,
    chunk_size: int,
    overwrite: bool,
    unknown_policy: str = "fail",
) -> dict[str, Any]:
    """Convert one FRDR point cloud and return conversion metadata."""

    if input_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"Input must be .las or .laz: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input point cloud does not exist: {input_path}")
    if not tile_name.isdigit():
        raise ValueError("--tile-name must be numeric because TLS2trees parses it as an integer")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    if unknown_policy not in {"fail", "drop", "nonwood"}:
        raise ValueError("--unknown-policy must be one of: fail, drop, nonwood")

    tile_number = int(tile_name)
    tile_file = output_dir / f"{tile_name}.downsample.segmented.ply"
    tile_index = output_dir / "tile_index.dat"
    if not overwrite:
        existing = [path for path in (tile_file, tile_index) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in existing)
            )

    scan = scan_input(
        input_path,
        wood_field,
        wood_value,
        nonwood_value,
        chunk_size,
        unknown_policy,
    )
    local_min_z = float(scan["xyz_min"][2])
    x_centre, y_centre, z_centre = (float(value) for value in scan["xyz_mean"])

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = iter_converted_chunks(
        input_path,
        wood_field,
        wood_value,
        nonwood_value,
        local_min_z,
        chunk_size,
        unknown_policy,
    )
    write_tls2trees_ply(
        tile_file,
        int(scan["retained_point_count"]),
        chunks,
        overwrite=overwrite,
    )
    tile_index.write_text(
        f"{tile_name} {x_centre:.6f} {y_centre:.6f} {z_centre:.6f} {tile_file}\n",
        encoding="utf-8",
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_ply": str(tile_file),
        "output_path": str(tile_file),
        "tile_index": str(tile_index),
        "tile_name": tile_name,
        "tile_number": tile_number,
        "tile_id": tile_name,
        "point_count": scan["retained_point_count"],
        "original_point_count": scan["original_point_count"],
        "retained_point_count": scan["retained_point_count"],
        "dropped_unknown_count": scan["dropped_unknown_count"],
        "unknown_point_count": scan["unknown_point_count"],
        "unknown_woods_values": scan["unknown_woods_values"],
        "input_dimensions": scan["input_dimensions"],
        "output_columns": ["x", "y", "z", "n_z", "label"],
        "wood_field": wood_field,
        "wood_value": wood_value,
        "nonwood_value": nonwood_value,
        "unknown_policy": unknown_policy,
        "label_mapping": {str(wood_value): 3, str(nonwood_value): 1},
        "woods_counts": scan["woods_counts"],
        "label_counts": scan["label_counts"],
        "wood_point_count": scan["wood_point_count"],
        "nonwood_point_count": scan["nonwood_point_count"],
        "z_range": [scan["xyz_min"][2], scan["xyz_max"][2]],
        "n_z_range": [0.0, scan["xyz_max"][2] - scan["xyz_min"][2]],
        "local_min_z": local_min_z,
        "tile_centre_xyz": [x_centre, y_centre, z_centre],
        "chunk_size_points": chunk_size,
        "z_normalisation": "local_minimum",
        "warning": FEASIBILITY_WARNING,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FRDR woods labels into a TLS2trees instance-stage PLY tile."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-name", required=True)
    parser.add_argument("--wood-field", required=True)
    parser.add_argument("--wood-value", required=True, type=float)
    parser.add_argument("--nonwood-value", required=True, type=float)
    parser.add_argument(
        "--unknown-policy",
        choices=("fail", "drop", "nonwood"),
        default="fail",
        help="How to handle wood-field values other than --wood-value/--nonwood-value.",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--metadata-output", help="Defaults to <output-dir>/conversion_metadata.json.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    metadata_path = (
        resolve_path(args.metadata_output)
        if args.metadata_output
        else output_dir / "conversion_metadata.json"
    )

    payload = convert(
        input_path=input_path,
        output_dir=output_dir,
        tile_name=args.tile_name,
        wood_field=args.wood_field,
        wood_value=args.wood_value,
        nonwood_value=args.nonwood_value,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        unknown_policy=args.unknown_policy,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(FEASIBILITY_WARNING)
    print(
        f"Converted {payload['retained_point_count']} of "
        f"{payload['original_point_count']} points to: {payload['output_ply']}"
    )
    print(f"Dropped unknown points: {payload['dropped_unknown_count']}")
    print(f"Tile index: {payload['tile_index']}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
