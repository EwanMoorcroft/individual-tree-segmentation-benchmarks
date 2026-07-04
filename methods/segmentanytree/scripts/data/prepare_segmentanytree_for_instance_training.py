"""Prepare FOR-instance development plots for SegmentAnyTree training.

The output layout and PLY fields follow the conversion script shipped with the
pinned SegmentAnyTree repository. Test plots are recorded in the manifest but
are never converted into the training data root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np


PATH_COLUMNS = (
    "relative_path",
    "path",
    "file_path",
    "filepath",
    "las_path",
    "filename",
    "file",
)
SPLIT_COLUMNS = ("split", "data_split", "dataset_split", "partition", "set")
COLLECTION_COLUMNS = ("collection", "dataset", "site", "source")
DEVELOPMENT_SPLITS = {"dev", "development", "train", "training"}
TEST_SPLITS = {"test", "testing"}
SEMANTIC_MAPPING = {
    0: 0,
    1: 1,
    2: 1,
    3: 0,
    4: 2,
    5: 2,
    6: 2,
}


def first_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    names = {name.strip().lower(): name for name in fieldnames}
    return next((names[name] for name in candidates if name in names), None)


def normalise_relative_path(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./")


def canonical_dataset_split(value: str) -> str:
    split = value.strip().casefold()
    if split in DEVELOPMENT_SPLITS:
        return "dev"
    if split in TEST_SPLITS:
        return "test"
    raise ValueError(f"Unsupported dataset split: {value!r}")


def read_split_rows(dataset_root: Path, metadata_path: Path) -> list[dict[str, str]]:
    """Read existing LAS rows from the supplied FOR-instance split metadata."""

    available_paths = {
        path.relative_to(dataset_root).as_posix(): path.resolve()
        for path in sorted(dataset_root.rglob("*.las"))
    }
    if not available_paths:
        raise FileNotFoundError(f"No LAS files found under {dataset_root}")

    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        path_column = first_column(fieldnames, PATH_COLUMNS)
        split_column = first_column(fieldnames, SPLIT_COLUMNS)
        collection_column = first_column(fieldnames, COLLECTION_COLUMNS)
        if path_column is None or split_column is None:
            raise ValueError(
                f"Could not identify path/split columns in {metadata_path}; "
                f"columns: {fieldnames}"
            )

        rows: list[dict[str, str]] = []
        assigned_paths: dict[str, str] = {}
        skipped_missing = 0
        for row_number, row in enumerate(reader, start=2):
            relative_path = normalise_relative_path(row.get(path_column) or "")
            if not relative_path:
                continue
            source_path = available_paths.get(relative_path)
            if source_path is None:
                skipped_missing += 1
                continue
            dataset_split = canonical_dataset_split(row.get(split_column) or "")
            previous_split = assigned_paths.get(relative_path)
            if previous_split is not None:
                if previous_split != dataset_split:
                    raise ValueError(
                        f"Conflicting split assignments for {relative_path}: "
                        f"{previous_split!r} and {dataset_split!r} "
                        f"(metadata row {row_number})"
                    )
                continue
            assigned_paths[relative_path] = dataset_split
            collection = (
                (row.get(collection_column) or "").strip()
                if collection_column
                else ""
            )
            if not collection:
                parts = Path(relative_path).parts
                collection = parts[0] if len(parts) > 1 else ""
            rows.append(
                {
                    "relative_path": relative_path,
                    "source_path": str(source_path),
                    "collection": collection,
                    "plot_name": source_path.stem,
                    "dataset_split": dataset_split,
                }
            )
    if not rows:
        raise ValueError(f"No dataset records were read from {metadata_path}")
    unassigned = sorted(set(available_paths) - set(assigned_paths))
    if unassigned:
        preview = ", ".join(unassigned[:10])
        raise ValueError(
            f"{len(unassigned)} local LAS files have no split metadata: {preview}"
        )
    if skipped_missing:
        print(
            f"Ignored {skipped_missing} metadata rows for LAS files that are "
            "not present in this dataset checkout."
        )
    return rows


def assign_training_roles(
    rows: list[dict[str, str]],
    seed: int,
    validation_fraction: float,
) -> list[dict[str, str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    development_indices = [
        index for index, row in enumerate(rows) if row["dataset_split"] == "dev"
    ]
    if len(development_indices) < 2:
        raise ValueError("At least two development plots are required")
    validation_count = int(validation_fraction * len(development_indices))
    if validation_count < 1:
        raise ValueError("Validation fraction selected no development plots")
    selected_offsets = set(
        random.Random(seed).sample(
            range(len(development_indices)),
            validation_count,
        )
    )
    validation_indices = {
        development_indices[offset] for offset in selected_offsets
    }

    assigned: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        record = dict(row)
        if row["dataset_split"] == "test":
            record["training_role"] = "held_out_test"
        elif index in validation_indices:
            record["training_role"] = "val"
        else:
            record["training_role"] = "train"
        assigned.append(record)
    return assigned


def select_profile_records(
    rows: list[dict[str, str]],
    profile: str,
    pilot_train_count: int,
    pilot_val_count: int,
) -> list[dict[str, Any]]:
    if profile not in {"pilot", "full"}:
        raise ValueError(f"Unsupported profile: {profile}")
    selected_train = {
        row["relative_path"]
        for row in [r for r in rows if r["training_role"] == "train"][
            :pilot_train_count
        ]
    }
    selected_val = {
        row["relative_path"]
        for row in [r for r in rows if r["training_role"] == "val"][
            :pilot_val_count
        ]
    }

    output: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = dict(row)
        if profile == "full":
            selected = row["training_role"] in {"train", "val"}
        else:
            selected = (
                row["relative_path"] in selected_train
                or row["relative_path"] in selected_val
            )
        record["selected_for_profile"] = selected
        output.append(record)
    if not any(
        row["selected_for_profile"] and row["training_role"] == "train"
        for row in output
    ):
        raise ValueError("Profile contains no training plots")
    if not any(
        row["selected_for_profile"] and row["training_role"] == "val"
        for row in output
    ):
        raise ValueError("Profile contains no validation plots")
    return output


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_ply(output_path: Path, vertices: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Converted from FOR-instance for SegmentAnyTree training.\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property float semantic_seg\n"
        "property float treeID\n"
        "end_header\n"
    ).encode("ascii")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.astype(vertices.dtype.newbyteorder("<")).tobytes())
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_las_to_training_ply(source_path: Path, output_path: Path) -> dict[str, Any]:
    """Convert one labelled LAS file to the upstream Treeins training schema."""

    cloud = laspy.read(source_path)
    available = set(cloud.point_format.dimension_names)
    required = {"X", "Y", "Z", "intensity", "classification", "treeID"}
    missing = required - available
    if missing:
        raise ValueError(
            f"LAS file is missing required dimensions {sorted(missing)}: "
            f"{source_path}"
        )

    classification = np.asarray(cloud.classification, dtype=np.int64)
    unsupported = sorted(set(np.unique(classification)) - set(SEMANTIC_MAPPING))
    if unsupported:
        raise ValueError(
            f"Unsupported classification values {unsupported}: {source_path}"
        )

    point_count = len(cloud.points)
    vertices = np.zeros(
        point_count,
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("intensity", "f4"),
                ("semantic_seg", "f4"),
                ("treeID", "f4"),
            ]
        ),
    )
    scales = np.asarray(cloud.header.scales, dtype=np.float64)
    vertices["x"] = np.asarray(cloud.X, dtype=np.float64) * scales[0]
    vertices["y"] = np.asarray(cloud.Y, dtype=np.float64) * scales[1]
    vertices["z"] = np.asarray(cloud.Z, dtype=np.float64) * scales[2]
    vertices["intensity"] = np.asarray(cloud.intensity, dtype=np.float32)
    semantic = np.zeros(point_count, dtype=np.float32)
    for source_class, target_class in SEMANTIC_MAPPING.items():
        semantic[classification == source_class] = target_class
    vertices["semantic_seg"] = semantic
    vertices["treeID"] = np.asarray(cloud["treeID"], dtype=np.float32)

    write_training_ply(output_path, vertices)
    return {
        "point_count": point_count,
        "reference_tree_count": int(
            np.unique(np.asarray(cloud["treeID"])[np.asarray(cloud["treeID"]) > 0]).size
        ),
        "source_sha256": sha256(source_path),
        "output_sha256": sha256(output_path),
    }


def prepare(
    dataset_root: Path,
    metadata_path: Path,
    output_root: Path,
    manifest_path: Path,
    profile: str,
    seed: int,
    validation_fraction: float,
    pilot_train_count: int,
    pilot_val_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Prepare a deterministic pilot or full development-only training tree."""

    rows = read_split_rows(dataset_root, metadata_path)
    rows = assign_training_roles(rows, seed, validation_fraction)
    records = select_profile_records(
        rows,
        profile,
        pilot_train_count,
        pilot_val_count,
    )
    raw_root = output_root / "treeinsfused" / "raw"
    if raw_root.exists() and any(raw_root.rglob("*.ply")) and not overwrite:
        raise FileExistsError(
            f"Training PLY files already exist under {raw_root}; "
            "use --overwrite only after archiving the previous preparation"
        )

    for record in records:
        record["converted_ply"] = None
        record["conversion_status"] = "held_out"
        if not record["selected_for_profile"]:
            continue
        if record["training_role"] == "held_out_test":
            raise AssertionError("Held-out test data cannot enter a training profile")
        filename = (
            f"{record['collection']}_{record['plot_name']}_"
            f"{record['training_role']}.ply"
        )
        output_path = raw_root / record["collection"] / filename
        conversion = convert_las_to_training_ply(
            Path(record["source_path"]),
            output_path,
        )
        record.update(conversion)
        record["converted_ply"] = str(output_path.resolve())
        record["conversion_status"] = "completed"

    role_counts = {
        role: sum(
            row["selected_for_profile"] and row["training_role"] == role
            for row in records
        )
        for role in ("train", "val", "held_out_test")
    }
    dataset_role_counts = {
        role: sum(row["training_role"] == role for row in records)
        for role in ("train", "val", "held_out_test")
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "FOR-instance",
        "profile": profile,
        "split_source": str(metadata_path.resolve()),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "coordinate_conversion": (
            "LAS integer coordinates multiplied by scale; LAS offsets omitted "
            "to match the pinned SegmentAnyTree conversion"
        ),
        "semantic_mapping": {str(key): value for key, value in SEMANTIC_MAPPING.items()},
        "test_data_converted": False,
        "dataset_role_counts": dataset_role_counts,
        "selected_role_counts": role_counts,
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FOR-instance development LAS files into the PLY layout "
            "required for SegmentAnyTree training."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-metadata")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--pilot-train-count", type=int, default=2)
    parser.add_argument("--pilot-val-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = (
        Path(args.split_metadata).expanduser().resolve()
        if args.split_metadata
        else dataset_root / "data_split_metadata.csv"
    )
    manifest = prepare(
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        output_root=Path(args.output_root).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve(),
        profile=args.profile,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        pilot_train_count=args.pilot_train_count,
        pilot_val_count=args.pilot_val_count,
        overwrite=args.overwrite,
    )
    counts = manifest["selected_role_counts"]
    print(
        f"Prepared profile={args.profile}: "
        f"train={counts['train']} val={counts['val']} test=0"
    )
    print(Path(args.manifest).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
