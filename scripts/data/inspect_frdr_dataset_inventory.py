from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def count_dimension_values(reader: Any, dimension: str, chunk_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for points in reader.chunk_iterator(chunk_size):
        values, value_counts = np.unique(np.asarray(points[dimension]), return_counts=True)
        for value, count in zip(values, value_counts):
            key = str(value.item() if hasattr(value, "item") else value)
            counts[key] = counts.get(key, 0) + int(count)
    return counts


def inspect_laz(
    path: Path,
    wood_field: str,
    wood_value: float,
    nonwood_value: float,
    chunk_size: int,
) -> dict[str, Any]:
    import laspy

    record: dict[str, Any] = {
        "plot_name": path.stem,
        "filename": path.name,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_size_gib": round(path.stat().st_size / (1024**3), 6),
        "error": None,
    }
    try:
        with laspy.open(path) as reader:
            header = reader.header
            dimensions = list(header.point_format.dimension_names)
            record.update(
                {
                    "point_count": int(header.point_count),
                    "dimensions": dimensions,
                    "x_min": float(header.mins[0]),
                    "x_max": float(header.maxs[0]),
                    "y_min": float(header.mins[1]),
                    "y_max": float(header.maxs[1]),
                    "z_min": float(header.mins[2]),
                    "z_max": float(header.maxs[2]),
                    "has_woods": wood_field in dimensions,
                    "woods_counts": {},
                    "unknown_woods_values": [],
                }
            )
            if wood_field in dimensions:
                record["woods_counts"] = count_dimension_values(reader, wood_field, chunk_size)
                expected = {str(wood_value), str(nonwood_value)}
                expected_integer = {
                    str(int(value))
                    for value in (wood_value, nonwood_value)
                    if float(value).is_integer()
                }
                record["unknown_woods_values"] = sorted(
                    key
                    for key in record["woods_counts"]
                    if key not in expected and key not in expected_integer
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
        "files_with_errors": sum(bool(record.get("error")) for record in records),
        "files_with_unknown_woods_values": sum(
            bool(record.get("unknown_woods_values")) for record in records
        ),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "plot_name",
        "filename",
        "path",
        "file_size_bytes",
        "file_size_gib",
        "point_count",
        "dimensions",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
        "has_woods",
        "woods_counts",
        "unknown_woods_values",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["dimensions"] = json.dumps(row.get("dimensions", []))
            row["woods_counts"] = json.dumps(row.get("woods_counts", {}), sort_keys=True)
            row["unknown_woods_values"] = json.dumps(row.get("unknown_woods_values", []))
            writer.writerow({name: row.get(name) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory all FRDR LAZ files without modifying them.")
    parser.add_argument("--dataset-root", required=True, help="Directory containing FRDR .laz files.")
    parser.add_argument("--wood-field", default="woods")
    parser.add_argument("--wood-value", type=float, default=1.0)
    parser.add_argument("--nonwood-value", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--csv-out",
        "--csv-output",
        dest="csv_out",
        default="results/metadata/frdr_dataset_inventory.csv",
    )
    parser.add_argument(
        "--json-out",
        "--json-output",
        dest="json_out",
        default="results/metadata/frdr_dataset_inventory.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {dataset_root}")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than zero")

    files = sorted(dataset_root.rglob("*.laz"))
    if not files:
        raise SystemExit(f"No .laz files found under: {dataset_root}")

    records = [
        inspect_laz(
            path,
            args.wood_field,
            args.wood_value,
            args.nonwood_value,
            args.chunk_size,
        )
        for path in files
    ]
    csv_path = resolve_path(args.csv_out)
    json_path = resolve_path(args.json_out)
    write_inventory(records, csv_path, json_path)

    print(f"Inspected {len(records)} LAZ file(s); errors: {sum(bool(r.get('error')) for r in records)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 1 if any(record.get("error") for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
