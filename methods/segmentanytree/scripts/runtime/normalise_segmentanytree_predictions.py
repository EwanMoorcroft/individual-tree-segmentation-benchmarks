"""Normalise validated SegmentAnyTree outputs to one XYZ PLY per tree."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import read_ply_vertices, write_xyz_ply


SUPPORTED_EXTENSIONS = {".las", ".laz", ".ply"}
SUPPORTED_FORMATS = {"auto", "labelled_point_cloud", "per_tree_directory"}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def resolve_destination_path(path_text: str) -> Path:
    """Return an absolute destination without following its final symlink."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.absolute()


def load_xyz(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        _, points = read_ply_vertices(path, columns=["x", "y", "z"])
        return np.column_stack([points["x"], points["y"], points["z"]])
    if suffix in {".las", ".laz"}:
        import laspy

        cloud = laspy.read(path)
        return np.column_stack([cloud.x, cloud.y, cloud.z])
    raise ValueError(f"Unsupported point-cloud extension: {suffix or '<none>'}")


def load_labelled_cloud(path: Path, instance_field: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        header, points = read_ply_vertices(path)
        if instance_field not in header.columns:
            raise ValueError(
                f"Prediction PLY is missing instance field {instance_field!r}; "
                f"columns: {header.columns}"
            )
        coordinates = np.column_stack([points["x"], points["y"], points["z"]])
        labels = np.asarray(points[instance_field])
    elif suffix in {".las", ".laz"}:
        import laspy

        cloud = laspy.read(path)
        dimensions = list(cloud.point_format.dimension_names)
        if instance_field not in dimensions:
            raise ValueError(
                f"Prediction point cloud is missing instance field {instance_field!r}; "
                f"dimensions: {dimensions}"
            )
        coordinates = np.column_stack([cloud.x, cloud.y, cloud.z])
        labels = np.asarray(cloud[instance_field])
    else:
        raise ValueError(f"Unsupported labelled prediction extension: {suffix or '<none>'}")
    if len(coordinates) != len(labels):
        raise ValueError("Prediction coordinates and instance labels have different lengths")
    return coordinates, labels


def label_text(value: Any) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item)


def safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return component or "unnamed"


def canonical_destination(path: Path) -> Path:
    """Canonicalise existing parents without following the final component."""
    lexical = path.absolute()
    if lexical.is_symlink():
        raise ValueError(f"Refusing symlinked destination: {lexical}")
    for parent in lexical.parents:
        if parent.is_symlink() and not parent.exists():
            raise ValueError(f"Refusing destination below dangling symlink: {parent}")
    return lexical.parent.resolve(strict=False) / lexical.name


def validate_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty; pass --overwrite to replace it: {path}"
            )


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def install_staged_directory(staged: Path, destination: Path) -> None:
    """Atomically install a sibling directory, restoring the old one on failure."""
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / f".{destination.name}.backup.{uuid.uuid4().hex}"
        destination.rename(backup)
    try:
        staged.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def stage_json(destination: Path, payload: dict[str, Any]) -> Path:
    """Validate and stage JSON beside its destination without replacing it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    canonical_after_mkdir = canonical_destination(destination)
    if canonical_after_mkdir != destination:
        raise ValueError(f"Metadata destination changed during preparation: {destination}")
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Metadata output is a directory: {destination}")
    temporary = destination.parent / (
        f".{destination.name}.partial.{uuid.uuid4().hex}"
    )
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def detect_format(input_path: Path, requested: str) -> str:
    if requested not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {requested!r}; choose from {sorted(SUPPORTED_FORMATS)}"
        )
    if requested != "auto":
        return requested
    if input_path.is_dir():
        return "per_tree_directory"
    if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return "labelled_point_cloud"
    raise ValueError(
        "Could not detect prediction format. Supply a supported point-cloud file "
        "or a directory containing one file per predicted tree."
    )


def normalise_labelled_point_cloud(
    input_path: Path,
    output_dir: Path,
    instance_field: str | None,
    ignored_labels: set[str],
) -> tuple[list[dict[str, Any]], int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Labelled prediction file does not exist: {input_path}")
    if not instance_field:
        raise ValueError("--instance-field is required for labelled_point_cloud format")

    coordinates, labels = load_labelled_cloud(input_path, instance_field)
    records: list[dict[str, Any]] = []
    ignored_point_count = 0
    for label in np.unique(labels):
        label_name = label_text(label)
        mask = labels == label
        point_count = int(np.count_nonzero(mask))
        if label_name in ignored_labels:
            ignored_point_count += point_count
            continue
        if point_count == 0:
            continue
        output_path = output_dir / f"instance_{safe_component(label_name)}.ply"
        write_xyz_ply(output_path, coordinates[mask])
        records.append(
            {
                "instance_id": label_name,
                "source": str(input_path),
                "output": str(output_path),
                "point_count": point_count,
            }
        )
    return records, ignored_point_count


def normalise_per_tree_directory(
    input_path: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Per-tree prediction directory does not exist: {input_path}")
    source_files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not source_files:
        raise ValueError(f"No supported per-tree point clouds found in: {input_path}")

    records: list[dict[str, Any]] = []
    for index, source_path in enumerate(source_files, start=1):
        coordinates = load_xyz(source_path)
        if len(coordinates) == 0:
            continue
        relative_stem = safe_component(
            source_path.relative_to(input_path).with_suffix("").as_posix()
        )
        output_path = output_dir / f"instance_{index:04d}_{relative_stem}.ply"
        write_xyz_ply(output_path, coordinates)
        records.append(
            {
                "instance_id": relative_stem,
                "source": str(source_path),
                "output": str(output_path),
                "point_count": len(coordinates),
            }
        )
    if not records:
        raise ValueError("All supported per-tree prediction files were empty")
    return records, 0


def normalise(
    *,
    input_path: Path,
    output_dir: Path,
    requested_format: str,
    instance_field: str | None,
    ignored_labels: set[str],
    overwrite: bool,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = canonical_destination(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"SegmentAnyTree output does not exist: {input_path}")
    detected_format = detect_format(input_path, requested_format)
    if detected_format == "labelled_point_cloud" and not instance_field:
        raise ValueError("--instance-field is required for labelled_point_cloud format")
    if detected_format == "per_tree_directory" and instance_field:
        raise ValueError("--instance-field is not used with per_tree_directory format")

    validate_output_directory(output_dir, overwrite)
    canonical_output = output_dir.resolve(strict=False)
    if paths_overlap(input_path, canonical_output):
        raise ValueError(
            "Input and output paths must not be equal, nested, or contain one another: "
            f"input={input_path}, output={canonical_output}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if canonical_destination(output_dir) != canonical_output:
        raise ValueError(f"Output destination changed during preparation: {output_dir}")
    staged_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial.", dir=str(output_dir.parent)
        )
    )
    external_metadata_stage: Path | None = None
    try:
        if detected_format == "labelled_point_cloud":
            instances, ignored_point_count = normalise_labelled_point_cloud(
                input_path, staged_dir, instance_field, ignored_labels
            )
        elif detected_format == "per_tree_directory":
            instances, ignored_point_count = normalise_per_tree_directory(
                input_path, staged_dir
            )
        else:
            raise ValueError(f"Unsupported detected format: {detected_format}")
        if not instances:
            raise ValueError(
                "No predicted instances remained after ignored labels were removed"
            )
        for record in instances:
            staged_output = Path(record["output"])
            record["output"] = str(
                output_dir / staged_output.relative_to(staged_dir)
            )
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_dir),
            "requested_format": requested_format,
            "detected_format": detected_format,
            "instance_field": instance_field,
            "predicted_instance_count": len(instances),
            "ignored_predicted_labels": sorted(ignored_labels),
            "ignored_point_count": ignored_point_count,
            "input_point_count": sum(record["point_count"] for record in instances)
            + ignored_point_count,
            "output_point_count": sum(record["point_count"] for record in instances),
            "instances": instances,
        }

        if metadata_path is None:
            destination = output_dir / "normalisation_metadata.json"
            (staged_dir / destination.name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            destination = canonical_destination(metadata_path)
            canonical_metadata = destination.resolve(strict=False)
            if paths_overlap(input_path, canonical_metadata):
                raise ValueError(
                    "Metadata output must not overlap the prediction input: "
                    f"input={input_path}, metadata={canonical_metadata}"
                )
            if canonical_metadata == canonical_output or canonical_output in canonical_metadata.parents:
                relative_metadata = canonical_metadata.relative_to(canonical_output)
                staged_metadata = staged_dir / relative_metadata
                staged_metadata.parent.mkdir(parents=True, exist_ok=True)
                staged_metadata.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                staged_metadata = None
                external_metadata_stage = stage_json(destination, payload)

        install_staged_directory(staged_dir, output_dir)
        if external_metadata_stage is not None:
            external_metadata_stage.replace(destination)
        payload["metadata_path"] = str(destination)
        return payload
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        if external_metadata_stage is not None:
            external_metadata_stage.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalise a validated SegmentAnyTree output to one XYZ PLY per "
            "predicted tree."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--format",
        required=True,
        choices=sorted(SUPPORTED_FORMATS),
        help=(
            "Use labelled_point_cloud for one labelled LAS/LAZ/PLY, "
            "per_tree_directory for one file per tree, or auto after inspecting output."
        ),
    )
    parser.add_argument("--instance-field")
    parser.add_argument("--ignore-labels", default="0,-1")
    parser.add_argument("--metadata-output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ignored_labels = {
        value.strip() for value in args.ignore_labels.split(",") if value.strip()
    }
    payload = normalise(
        input_path=resolve_path(args.input),
        output_dir=resolve_destination_path(args.output_dir),
        requested_format=args.format,
        instance_field=args.instance_field,
        ignored_labels=ignored_labels,
        overwrite=args.overwrite,
        metadata_path=resolve_destination_path(args.metadata_output)
        if args.metadata_output
        else None,
    )
    print(f"Detected format: {payload['detected_format']}")
    print(f"Predicted instances: {payload['predicted_instance_count']}")
    print(f"Output points: {payload['output_point_count']}")
    print(f"Metadata: {payload['metadata_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
