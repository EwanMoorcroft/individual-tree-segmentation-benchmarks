"""Freeze the exact 21 available FOR-instance development plots for TreeLearn."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from for_instance_development_common import (
    EXPECTED_DEVELOPMENT_POINTS,
    EXPECTED_DEVELOPMENT_PLOTS,
    EXPECTED_DEVELOPMENT_REFERENCE_TREES,
    EXPECTED_DEVELOPMENT_SITE_COUNTS,
    EXPECTED_DEVELOPMENT_SITE_POINTS,
    EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES,
    MANIFEST_FIELDS,
    plot_id,
    safe_plot_id,
    strict_relative_path,
    validate_manifest_rows,
)


REFERENCE_CLASSES = {4, 5, 6}
IGNORED_TREE_IDS = {-1, 0}
REQUIRED_METADATA_COLUMNS = {"path", "folder", "split"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_las(path: Path) -> tuple[int, int]:
    """Count points and distinct labelled reference trees without copying a cloud."""

    import laspy

    reference_ids: set[int] = set()
    with laspy.open(path) as reader:
        dimensions = set(reader.header.point_format.dimension_names)
        missing = {"treeID", "classification"} - dimensions
        if missing:
            raise ValueError(f"Source LAS is missing fields {sorted(missing)}: {path}")
        point_count = int(reader.header.point_count)
        for points in reader.chunk_iterator(1_000_000):
            classification = np.asarray(points.classification, dtype=np.int64)
            tree_id = np.asarray(points["treeID"], dtype=np.int64)
            mask = np.isin(classification, tuple(REFERENCE_CLASSES)) & ~np.isin(
                tree_id,
                tuple(IGNORED_TREE_IDS),
            )
            reference_ids.update(int(value) for value in np.unique(tree_id[mask]))
    if point_count <= 0:
        raise ValueError(f"Development LAS contains no points: {path}")
    return point_count, len(reference_ids)


def read_available_development_rows(
    dataset_root: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Select only exact existing dev paths; never stat or open test paths."""

    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_METADATA_COLUMNS - fieldnames
        if missing:
            raise ValueError(
                f"FOR-instance metadata is missing columns {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"FOR-instance metadata contains no rows: {metadata_path}")

    available: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []
    seen_dev_paths: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        if split != "dev":
            continue
        relative = strict_relative_path(row.get("path") or "")
        folder = (row.get("folder") or "").strip()
        if folder != Path(relative).parts[0]:
            raise ValueError(
                f"Metadata folder/path mismatch at row {row_number}: "
                f"{folder!r}, {relative!r}"
            )
        if relative in seen_dev_paths:
            raise ValueError(f"Duplicate development metadata path: {relative}")
        seen_dev_paths.add(relative)
        candidate = (dataset_root / relative).resolve()
        try:
            candidate.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"Development path escapes dataset root: {relative}") from exc
        record = {
            "relative_path": relative,
            "collection": folder,
            "input_las": str(candidate),
        }
        if candidate.is_file():
            available.append(record)
        else:
            unavailable.append(record)
    return available, unavailable


def build_manifest(dataset_root: Path, metadata_path: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"FOR-instance dataset root does not exist: {dataset_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"FOR-instance metadata does not exist: {metadata_path}")
    available, unavailable = read_available_development_rows(
        dataset_root,
        metadata_path,
    )
    available.sort(key=lambda row: (row["collection"], row["relative_path"]))
    observed_counts = Counter(row["collection"] for row in available)
    if len(available) != EXPECTED_DEVELOPMENT_PLOTS:
        raise ValueError(
            f"Expected exactly {EXPECTED_DEVELOPMENT_PLOTS} available development "
            f"plots, found {len(available)}; unavailable dev paths={len(unavailable)}"
        )
    if dict(observed_counts) != EXPECTED_DEVELOPMENT_SITE_COUNTS:
        raise ValueError(
            "Available development site counts differ from the frozen contract: "
            f"expected {EXPECTED_DEVELOPMENT_SITE_COUNTS}, "
            f"found {dict(observed_counts)}"
        )

    split_hash = sha256(metadata_path)
    plots: list[dict[str, Any]] = []
    for task_index, source in enumerate(available):
        input_las = Path(source["input_las"])
        point_count, reference_tree_count = inspect_las(input_las)
        identifier = plot_id(source["relative_path"])
        plots.append(
            {
                "task_index": task_index,
                "plot_id": identifier,
                "safe_plot_id": safe_plot_id(identifier),
                "relative_path": source["relative_path"],
                "collection": source["collection"],
                "split": "dev",
                "input_las": str(input_las),
                "point_count": point_count,
                "reference_tree_count": reference_tree_count,
                "input_sha256": sha256(input_las),
                "split_metadata": str(metadata_path),
                "split_metadata_sha256": split_hash,
            }
        )
    plots = validate_manifest_rows(plots)
    return {
        "schema_version": 1,
        "status": "frozen_exact_path_development_manifest",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "dataset_split": "dev",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "split_metadata": str(metadata_path),
        "split_metadata_sha256": split_hash,
        "available_development_plot_count": len(plots),
        "expected_development_plot_count": EXPECTED_DEVELOPMENT_PLOTS,
        "available_site_counts": dict(observed_counts),
        "expected_site_counts": EXPECTED_DEVELOPMENT_SITE_COUNTS,
        "available_total_points": sum(row["point_count"] for row in plots),
        "expected_total_points": EXPECTED_DEVELOPMENT_POINTS,
        "available_reference_tree_count": sum(
            row["reference_tree_count"] for row in plots
        ),
        "expected_reference_tree_count": EXPECTED_DEVELOPMENT_REFERENCE_TREES,
        "expected_site_points": EXPECTED_DEVELOPMENT_SITE_POINTS,
        "expected_site_reference_trees": (
            EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES
        ),
        "unavailable_development_metadata_paths": unavailable,
        "mapping_rule": "exact_metadata_path_only",
        "held_out_test_accessed": False,
        "plots": plots,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in MANIFEST_FIELDS} for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the exact 21 available FOR-instance development plots."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.output_csv).expanduser().resolve()
    json_path = Path(args.output_json).expanduser().resolve()
    if csv_path == json_path:
        raise ValueError("Manifest CSV and JSON outputs must be different paths")
    collisions = [path for path in (csv_path, json_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Development manifest output already exists: "
            + ", ".join(str(path) for path in collisions)
        )
    payload = build_manifest(
        Path(args.dataset_root),
        Path(args.metadata_csv),
    )
    write_csv(csv_path, payload["plots"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"development_manifest_csv={csv_path}")
    print(f"development_manifest_json={json_path}")
    print(f"development_plots={len(payload['plots'])}")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
