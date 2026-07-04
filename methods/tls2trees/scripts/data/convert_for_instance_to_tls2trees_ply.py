"""Convert one FOR-instance LAS plot to the TLS2trees PLY input schema."""

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


NORMALISATION_WARNING = (
    "n_z uses the retained cloud's local minimum z. This is a feasibility "
    "approximation and is not terrain or ground normalisation."
)
TILE_SUFFIX = ".downsample.segmented.ply"


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def scalar_key(value: Any) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item)


def dimension_name(dimensions: list[str], requested: str) -> str | None:
    return {name.lower(): name for name in dimensions}.get(requested.lower())


def normalise_tile_name(tile_name: str) -> tuple[str, str]:
    tile_id = tile_name[: -len(TILE_SUFFIX)] if tile_name.endswith(TILE_SUFFIX) else tile_name
    if not tile_id.isdigit():
        raise ValueError("--tile-name must be numeric or '<digits>.downsample.segmented.ply'")
    return tile_id, f"{tile_id}{TILE_SUFFIX}"


def scan_input(
    input_path: Path,
    reference_classes: tuple[int, ...],
    retain_live_branches: bool,
    chunk_size: int,
    semantic_field: str = "classification",
    tree_id_field: str = "treeID",
) -> dict[str, Any]:
    """Scan labels and coordinates before writing the selected leaf-off points."""

    import laspy

    retained_classes = set(reference_classes)
    if retain_live_branches:
        retained_classes.add(5)
    class_counts: dict[str, int] = {}
    label_counts = {"1": 0, "3": 0}
    positive_reference_ids: set[str] = set()
    all_positive_reference_ids: set[str] = set()
    retained_count = 0
    xyz_sum = np.zeros(3, dtype=np.float64)
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)

    with laspy.open(input_path) as reader:
        header = reader.header
        dimensions = list(header.point_format.dimension_names)
        actual_semantic = dimension_name(dimensions, semantic_field)
        actual_tree_id = dimension_name(dimensions, tree_id_field)
        if actual_semantic is None:
            raise ValueError(
                f"Input is missing semantic field {semantic_field!r}; dimensions: {dimensions}"
            )

        for points in reader.chunk_iterator(chunk_size):
            classes = np.asarray(points[actual_semantic])
            unique_classes, counts = np.unique(classes, return_counts=True)
            for value, count in zip(unique_classes, counts):
                key = scalar_key(value)
                class_counts[key] = class_counts.get(key, 0) + int(count)

            retained_mask = np.isin(classes, list(retained_classes))
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
            label_counts["3"] += int(
                np.count_nonzero(np.isin(classes, list(reference_classes)))
            )
            if retain_live_branches:
                label_counts["1"] += int(np.count_nonzero(classes == 5))

            if actual_tree_id is not None:
                tree_ids = np.asarray(points[actual_tree_id])
                all_positive_reference_ids.update(
                    scalar_key(value) for value in np.unique(tree_ids[tree_ids > 0])
                )
                reference_mask = np.isin(classes, list(reference_classes)) & (tree_ids > 0)
                positive_reference_ids.update(
                    scalar_key(value) for value in np.unique(tree_ids[reference_mask])
                )

        original_count = int(header.point_count)

    if retained_count == 0:
        raise ValueError("No points remain after applying the selected reference classes")
    observed_classes = {int(float(key)) for key in class_counts}
    return {
        "input_dimensions": dimensions,
        "original_point_count": original_count,
        "retained_point_count": retained_count,
        "reference_classes": sorted(set(reference_classes)),
        "retained_classes": sorted(retained_classes),
        "ignored_classes": sorted(observed_classes - retained_classes),
        "class_counts": class_counts,
        "label_counts": label_counts,
        "treeID_field_present": actual_tree_id is not None,
        "positive_reference_tree_count": len(positive_reference_ids),
        "input_positive_reference_tree_count": len(all_positive_reference_ids),
        "xyz_min": xyz_min.tolist(),
        "xyz_max": xyz_max.tolist(),
        "xyz_mean": (xyz_sum / retained_count).tolist(),
    }


def iter_converted_chunks(
    input_path: Path,
    retained_classes: tuple[int, ...],
    local_min_z: float,
    chunk_size: int,
    semantic_field: str = "classification",
) -> Iterator[dict[str, np.ndarray]]:
    import laspy

    with laspy.open(input_path) as reader:
        dimensions = list(reader.header.point_format.dimension_names)
        actual_semantic = dimension_name(dimensions, semantic_field)
        if actual_semantic is None:
            raise ValueError(f"Input is missing semantic field {semantic_field!r}")
        for points in reader.chunk_iterator(chunk_size):
            classes = np.asarray(points[actual_semantic])
            retained_mask = np.isin(classes, list(retained_classes))
            z = np.asarray(points.z, dtype=np.float64)[retained_mask]
            retained_semantic = classes[retained_mask]
            yield {
                "x": np.asarray(points.x, dtype=np.float64)[retained_mask],
                "y": np.asarray(points.y, dtype=np.float64)[retained_mask],
                "z": z,
                "n_z": z - local_min_z,
                "label": np.where(retained_semantic == 5, 1.0, 3.0),
            }


def convert(
    input_path: Path,
    output_dir: Path,
    tile_name: str = "001",
    reference_classes: tuple[int, ...] = (4, 6),
    retain_live_branches: bool = False,
    chunk_size: int = 1_000_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one FOR-instance LAS plot and return conversion metadata."""

    if input_path.suffix.lower() != ".las":
        raise ValueError(f"FOR-instance input must be .las: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input LAS does not exist: {input_path}")
    if not reference_classes:
        raise ValueError("At least one --reference-classes value is required")
    if 5 in reference_classes:
        raise ValueError(
            "Class 5 is live branch context; enable it with "
            "--retain-live-branches instead of using it as a leaf-off reference class"
        )
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")

    tile_id, tile_filename = normalise_tile_name(tile_name)
    tile_path = output_dir / tile_filename
    tile_index = output_dir / "tile_index.dat"
    if not overwrite:
        existing = [path for path in (tile_path, tile_index) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in existing)
            )

    scan = scan_input(
        input_path,
        tuple(sorted(set(reference_classes))),
        retain_live_branches,
        chunk_size,
    )
    retained_classes = tuple(scan["retained_classes"])
    local_min_z = float(scan["xyz_min"][2])
    x_centre, y_centre, z_centre = (float(value) for value in scan["xyz_mean"])

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = iter_converted_chunks(
        input_path,
        retained_classes,
        local_min_z,
        chunk_size,
    )
    write_tls2trees_ply(
        tile_path,
        int(scan["retained_point_count"]),
        chunks,
        overwrite=overwrite,
    )
    tile_index.write_text(
        f"{tile_id} {x_centre:.6f} {y_centre:.6f} {z_centre:.6f} {tile_path}\n",
        encoding="utf-8",
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "output_path": str(tile_path),
        "tile_index": str(tile_index),
        "tile_id": tile_id,
        "tile_filename": tile_filename,
        "evaluation_mode": "leaf_off",
        "reference_classes": scan["reference_classes"],
        "retained_classes": scan["retained_classes"],
        "ignored_classes": scan["ignored_classes"],
        "retain_live_branches": retain_live_branches,
        "label_mapping": {"4": 3, "6": 3, "5": 1},
        "input_dimensions": scan["input_dimensions"],
        "original_point_count": scan["original_point_count"],
        "retained_point_count": scan["retained_point_count"],
        "dropped_point_count": (
            scan["original_point_count"] - scan["retained_point_count"]
        ),
        "treeID_field_present": scan["treeID_field_present"],
        "positive_reference_tree_count": scan["positive_reference_tree_count"],
        "input_positive_reference_tree_count": scan[
            "input_positive_reference_tree_count"
        ],
        "class_counts": scan["class_counts"],
        "label_counts": scan["label_counts"],
        "coordinate_bounds": {
            "x_min": scan["xyz_min"][0],
            "x_max": scan["xyz_max"][0],
            "y_min": scan["xyz_min"][1],
            "y_max": scan["xyz_max"][1],
            "z_min": scan["xyz_min"][2],
            "z_max": scan["xyz_max"][2],
        },
        "n_z_range": [0.0, scan["xyz_max"][2] - scan["xyz_min"][2]],
        "local_min_z": local_min_z,
        "tile_centre_xyz": [x_centre, y_centre, z_centre],
        "chunk_size_points": chunk_size,
        "z_normalisation": "local_minimum",
        "warning": NORMALISATION_WARNING,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a FOR-instance LAS plot into a TLS2trees PLY tile."
    )
    parser.add_argument("--input-las", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-name", default="001")
    parser.add_argument("--reference-classes", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--retain-live-branches", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--metadata-output",
        help="Defaults to <output-dir>/conversion_metadata.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input_las)
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
        reference_classes=tuple(args.reference_classes),
        retain_live_branches=args.retain_live_branches,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(NORMALISATION_WARNING)
    print(
        f"Converted {payload['retained_point_count']} of "
        f"{payload['original_point_count']} points"
    )
    print(f"PLY: {payload['output_path']}")
    print(f"Tile index: {payload['tile_index']}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
