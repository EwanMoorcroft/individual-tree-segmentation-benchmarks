from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from select_for_instance_plot import read_split_metadata, split_for_relative_path


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def dimension_name(dimensions: list[str], requested: str) -> str | None:
    by_lower = {name.lower(): name for name in dimensions}
    return by_lower.get(requested.lower())


def scalar_value(value: Any) -> int | float | str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    if isinstance(item, (int, float, str)):
        return item
    return str(item)


def inspect_las(
    path: Path,
    dataset_root: Path,
    chunk_size: int,
    tree_id_field: str = "treeID",
    tree_species_field: str = "treeSP",
    semantic_field: str = "classification",
    split: str = "unassigned",
) -> dict[str, Any]:
    import laspy

    relative_path = path.relative_to(dataset_root)
    record: dict[str, Any] = {
        "relative_path": relative_path.as_posix(),
        "collection": relative_path.parts[0] if len(relative_path.parts) > 1 else "",
        "filename": path.name,
        "split": split,
        "file_size_bytes": path.stat().st_size,
        "error": None,
    }
    try:
        with laspy.open(path) as reader:
            header = reader.header
            dimensions = list(header.point_format.dimension_names)
            actual_tree_id = dimension_name(dimensions, tree_id_field)
            actual_tree_species = dimension_name(dimensions, tree_species_field)
            actual_semantic = dimension_name(dimensions, semantic_field)
            classification_values: set[int | float | str] = set()
            positive_reference_ids: set[int | float | str] = set()
            positive_tree_id_point_count = 0
            zero_tree_id_point_count = 0

            for points in reader.chunk_iterator(chunk_size):
                if actual_semantic is not None:
                    classification_values.update(
                        scalar_value(value)
                        for value in np.unique(np.asarray(points[actual_semantic]))
                    )
                if actual_tree_id is not None:
                    tree_ids = np.asarray(points[actual_tree_id])
                    positive_mask = tree_ids > 0
                    positive_tree_id_point_count += int(np.count_nonzero(positive_mask))
                    zero_tree_id_point_count += int(np.count_nonzero(tree_ids == 0))
                    positive_reference_ids.update(
                        scalar_value(value) for value in np.unique(tree_ids[positive_mask])
                    )

            record.update(
                {
                    "point_count": int(header.point_count),
                    "dimensions": dimensions,
                    "has_treeID": actual_tree_id is not None,
                    "has_treeSP": actual_tree_species is not None,
                    "classification_values": sorted(
                        classification_values, key=lambda value: (str(type(value)), value)
                    ),
                    "treeID_positive_count": positive_tree_id_point_count,
                    "treeID_zero_count": zero_tree_id_point_count,
                    # Retain the original names for consumers created before the
                    # public inventory schema was finalised.
                    "positive_treeID_point_count": positive_tree_id_point_count,
                    "zero_treeID_point_count": zero_tree_id_point_count,
                    "reference_tree_count": len(positive_reference_ids),
                    "x_min": float(header.mins[0]),
                    "x_max": float(header.maxs[0]),
                    "y_min": float(header.mins[1]),
                    "y_max": float(header.maxs[1]),
                    "z_min": float(header.mins[2]),
                    "z_max": float(header.maxs[2]),
                }
            )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_inventory(records: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "file_count": len(records),
        "total_points": sum(int(record.get("point_count", 0)) for record in records),
        "total_reference_trees": sum(
            int(record.get("reference_tree_count", 0)) for record in records
        ),
        "files_with_errors": sum(bool(record.get("error")) for record in records),
        "records": records,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "relative_path",
        "collection",
        "filename",
        "split",
        "file_size_bytes",
        "point_count",
        "dimensions",
        "has_treeID",
        "has_treeSP",
        "classification_values",
        "treeID_positive_count",
        "treeID_zero_count",
        "positive_treeID_point_count",
        "zero_treeID_point_count",
        "reference_tree_count",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["dimensions"] = json.dumps(row.get("dimensions", []))
            row["classification_values"] = json.dumps(
                row.get("classification_values", [])
            )
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory FOR-instance LAS files without modifying source data."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--split-metadata",
        help="Defaults to <dataset-root>/data_split_metadata.csv when present.",
    )
    parser.add_argument(
        "--output-csv",
        "--csv-output",
        dest="output_csv",
        default="results/metadata/for_instance/inventory.csv",
    )
    parser.add_argument(
        "--output-json",
        "--json-output",
        dest="output_json",
        default="results/metadata/for_instance/inventory.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")
    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    files = sorted(dataset_root.rglob("*.las"))
    if not files:
        raise SystemExit(f"No .las files found under: {dataset_root}")

    split_metadata = (
        resolve_path(args.split_metadata)
        if args.split_metadata
        else dataset_root / "data_split_metadata.csv"
    )
    split_lookup = read_split_metadata(split_metadata)
    records = []
    for path in files:
        relative_path = path.relative_to(dataset_root)
        records.append(
            inspect_las(
                path,
                dataset_root,
                args.chunk_size,
                split=split_for_relative_path(relative_path, split_lookup),
            )
        )
    csv_path = resolve_path(args.output_csv)
    json_path = resolve_path(args.output_json)
    write_inventory(records, csv_path, json_path)

    errors = sum(bool(record.get("error")) for record in records)
    print(f"Inspected {len(records)} LAS files; errors: {errors}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(
        f"Split metadata: {split_metadata if split_metadata.is_file() else 'not found'}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
