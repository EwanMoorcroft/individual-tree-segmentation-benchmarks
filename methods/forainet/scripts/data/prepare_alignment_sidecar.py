"""Create a source-row sidecar for an original FOR-instance LAS file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import laspy
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".las":
        raise ValueError("relative_path must be a safe relative .las path")
    return path.as_posix()


def catalogue_split(metadata: Path, relative_path: str) -> str:
    with metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if (row.get("relative_path") or row.get("path")) == relative_path
    ]
    if len(matches) != 1:
        raise ValueError("relative_path must occur exactly once in split metadata")
    split = matches[0].get("split", "")
    if split not in {"dev", "test"}:
        raise ValueError(f"unsupported split value: {split!r}")
    return split


def prepare(
    *, source: Path, relative_path: str, split_metadata: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    relative_path = validate_relative_path(relative_path)
    split = catalogue_split(split_metadata, relative_path)
    if split != "dev":
        raise ValueError("alignment sidecar preparation is development-only")
    if not source.is_file():
        raise FileNotFoundError(source)

    cloud = laspy.read(source)
    dimensions = {dimension.name for dimension in cloud.point_format.dimensions}
    missing = {"classification", "treeID"} - dimensions
    if missing:
        raise ValueError(f"source LAS is missing required fields: {sorted(missing)}")

    classification = np.asarray(cloud.classification, dtype=np.int64)
    target_tree_id = np.asarray(cloud["treeID"], dtype=np.int64)
    point_count = len(classification)
    if point_count == 0 or len(target_tree_id) != point_count:
        raise ValueError("source LAS fields are empty or misaligned")

    arrays = {
        "x": np.asarray(cloud.x, dtype=np.float64),
        "y": np.asarray(cloud.y, dtype=np.float64),
        "z": np.asarray(cloud.z, dtype=np.float64),
        "classification": classification,
        "target_tree_id": target_tree_id,
        "source_row_index": np.arange(point_count, dtype=np.int64),
    }
    metadata = {
        "schema": "forainet_alignment_sidecar_v1",
        "relative_path": relative_path,
        "split": split,
        "source_sha256": sha256(source),
        "point_count": point_count,
        "dimensions": sorted(dimensions),
        "classification_values": [int(v) for v in np.unique(classification)],
        "positive_tree_id_point_count": int(np.count_nonzero(target_tree_id > 0)),
        "reference_tree_count": int(len(np.unique(target_tree_id[target_tree_id > 0]))),
        "source_row_contract": "exact_zero_based_original_las_order",
    }
    return metadata, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-las", required=True, type=Path)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--split-metadata", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()

    for output in (args.output_npz, args.metadata_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    metadata, arrays = prepare(
        source=args.source_las,
        relative_path=args.relative_path,
        split_metadata=args.split_metadata,
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    metadata["sidecar_sha256"] = sha256(args.output_npz)
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
