"""Method-neutral exact-path manifests for the local FOR-instance benchmark.

The development builder reads point-cloud headers and hashes only the frozen
development subset. Held-out files require a separate, explicit opt-in so that
development search code cannot accidentally touch them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_METADATA_SHA256 = (
    "dd64aa338681f8f4166f8d175879a2b0b0158ecf222497ec6f7f0b23bc4fce94"
)
EXPECTED_PATHS = {
    "dev": (
        "CULS/plot_1_annotated.las",
        "CULS/plot_3_annotated.las",
        "NIBIO/plot_10_annotated.las",
        "NIBIO/plot_11_annotated.las",
        "NIBIO/plot_12_annotated.las",
        "NIBIO/plot_13_annotated.las",
        "NIBIO/plot_16_annotated.las",
        "NIBIO/plot_19_annotated.las",
        "NIBIO/plot_21_annotated.las",
        "NIBIO/plot_2_annotated.las",
        "NIBIO/plot_3_annotated.las",
        "NIBIO/plot_4_annotated.las",
        "NIBIO/plot_6_annotated.las",
        "NIBIO/plot_7_annotated.las",
        "NIBIO/plot_8_annotated.las",
        "NIBIO/plot_9_annotated.las",
        "RMIT/train.las",
        "SCION/plot_35_annotated.las",
        "SCION/plot_39_annotated.las",
        "SCION/plot_87_annotated.las",
        "TUWIEN/train.las",
    ),
    "test": (
        "CULS/plot_2_annotated.las",
        "NIBIO/plot_17_annotated.las",
        "NIBIO/plot_18_annotated.las",
        "NIBIO/plot_1_annotated.las",
        "NIBIO/plot_22_annotated.las",
        "NIBIO/plot_23_annotated.las",
        "NIBIO/plot_5_annotated.las",
        "RMIT/test.las",
        "SCION/plot_31_annotated.las",
        "SCION/plot_61_annotated.las",
        "TUWIEN/test.las",
    ),
}
EXPECTED_SITE_COUNTS = {
    "dev": {"CULS": 2, "NIBIO": 14, "RMIT": 1, "SCION": 3, "TUWIEN": 1},
    "test": {"CULS": 1, "NIBIO": 6, "RMIT": 1, "SCION": 2, "TUWIEN": 1},
}
REQUIRED_METADATA_COLUMNS = {"path", "folder", "split"}
MANIFEST_FIELDS = (
    "task_index",
    "plot_id",
    "safe_plot_id",
    "relative_path",
    "collection",
    "split",
    "input_las",
    "point_count",
    "reference_tree_count",
    "input_sha256",
    "split_metadata",
    "split_metadata_sha256",
    "stage0_selected",
)
STAGE0_SELECTION_RULE = (
    "one_per_site_arithmetic_median_point_count_then_lower_count_then_path"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_relative_las_path(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("./"):
        raise ValueError(f"Unsafe or non-canonical metadata path: {value!r}")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe or non-canonical metadata path: {value!r}")
    if path.as_posix() != raw:
        raise ValueError(f"Non-canonical metadata path: {value!r}")
    if path.suffix.casefold() != ".las":
        raise ValueError(f"FOR-instance path is not a LAS file: {value!r}")
    return raw


def normalise_split(value: str) -> str:
    split = value.strip().casefold()
    if split in {"dev", "development"}:
        return "dev"
    if split == "test":
        return "test"
    raise ValueError(f"Unsupported FOR-instance split: {value!r}")


def plot_id(relative_path: str) -> str:
    return Path(strict_relative_las_path(relative_path)).with_suffix("").as_posix()


def safe_plot_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not result or not re.fullmatch(r"[A-Za-z0-9._-]+", result):
        raise ValueError(f"Could not make a safe plot ID from {value!r}")
    return result


def inspect_las_inventory(path: Path) -> tuple[int, int]:
    """Read point count and positive class-4/5/6 reference-tree inventory."""

    import laspy

    with laspy.open(path) as reader:
        count = int(reader.header.point_count)
        dimensions = set(reader.header.point_format.dimension_names)
        missing = {"classification", "treeID"} - dimensions
        if missing:
            raise ValueError(f"FOR-instance LAS is missing {sorted(missing)}: {path}")
        reference_ids: set[int] = set()
        for points in reader.chunk_iterator(1_000_000):
            classification = np.asarray(points.classification, dtype=np.int64)
            tree_ids = np.asarray(points["treeID"], dtype=np.int64)
            mask = np.isin(classification, (4, 5, 6)) & (tree_ids > 0)
            reference_ids.update(int(value) for value in np.unique(tree_ids[mask]))
    if count <= 0:
        raise ValueError(f"FOR-instance LAS contains no points: {path}")
    if not reference_ids:
        raise ValueError(f"FOR-instance LAS contains no reference trees: {path}")
    return count, len(reference_ids)


def _validate_expected_contract() -> None:
    development = set(EXPECTED_PATHS["dev"])
    held_out = set(EXPECTED_PATHS["test"])
    overlap = development & held_out
    if overlap:
        raise RuntimeError(f"Frozen FOR-instance split paths overlap: {sorted(overlap)}")
    for split, paths in EXPECTED_PATHS.items():
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Frozen {split} path contract contains duplicates")
        counts = Counter(Path(path).parts[0] for path in paths)
        if dict(counts) != EXPECTED_SITE_COUNTS[split]:
            raise RuntimeError(f"Frozen {split} site counts are inconsistent")


def read_split_metadata(
    metadata_path: Path,
    *,
    expected_sha256: str = EXPECTED_METADATA_SHA256,
) -> tuple[dict[str, dict[str, str]], str]:
    """Validate the metadata identity and the split assignment of all 32 paths."""

    _validate_expected_contract()
    path = metadata_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FOR-instance split metadata does not exist: {path}")
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("Expected split metadata SHA-256 is invalid")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "FOR-instance split metadata SHA-256 mismatch: "
            f"expected {expected_sha256}, found {observed_sha256}"
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_METADATA_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Split metadata is missing columns {sorted(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"Split metadata contains no rows: {path}")

    rows: dict[str, dict[str, str]] = {}
    for row_number, source in enumerate(source_rows, start=2):
        relative = strict_relative_las_path(source.get("path") or "")
        collection = (source.get("folder") or "").strip()
        split = (source.get("split") or "").strip()
        if split not in {"dev", "test"}:
            raise ValueError(
                f"Unexpected split at metadata row {row_number}: {split!r}"
            )
        if collection != Path(relative).parts[0]:
            raise ValueError(
                f"Metadata folder/path mismatch at row {row_number}: "
                f"{collection!r}, {relative!r}"
            )
        if relative in rows:
            raise ValueError(f"Duplicate split metadata path: {relative}")
        rows[relative] = {
            "relative_path": relative,
            "collection": collection,
            "split": split,
        }

    for split, expected_paths in EXPECTED_PATHS.items():
        for relative in expected_paths:
            row = rows.get(relative)
            if row is None:
                raise ValueError(
                    f"Frozen {split} path is absent from split metadata: {relative}"
                )
            if row["split"] != split:
                raise ValueError(
                    f"Split leakage for {relative}: expected {split}, "
                    f"metadata records {row['split']}"
                )
    return rows, observed_sha256


def select_stage0_development_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one development plot per site without consulting accuracy labels."""

    if any(normalise_split(str(row.get("split", ""))) != "dev" for row in rows):
        raise ValueError("Stage 0 selection accepts development rows only")
    selected: list[dict[str, Any]] = []
    for site in EXPECTED_SITE_COUNTS["dev"]:
        site_rows = [row for row in rows if row.get("collection") == site]
        if len(site_rows) != EXPECTED_SITE_COUNTS["dev"][site]:
            raise ValueError(
                f"Stage 0 expected {EXPECTED_SITE_COUNTS['dev'][site]} {site} rows, "
                f"found {len(site_rows)}"
            )
        point_counts = [int(row["point_count"]) for row in site_rows]
        if any(count <= 0 for count in point_counts):
            raise ValueError(f"Stage 0 {site} point counts must be positive")
        median_count = statistics.median(point_counts)
        choice = min(
            site_rows,
            key=lambda row: (
                abs(int(row["point_count"]) - median_count),
                int(row["point_count"]),
                str(row["relative_path"]),
            ),
        )
        selected.append(
            {
                "stage0_index": len(selected),
                "task_index": int(choice["task_index"]),
                "collection": site,
                "plot_id": str(choice["plot_id"]),
                "relative_path": str(choice["relative_path"]),
                "point_count": int(choice["point_count"]),
                "site_median_point_count": median_count,
                "selection_rule": STAGE0_SELECTION_RULE,
            }
        )
    return selected


def _path_is_file(path: Path) -> bool:
    return path.is_file()


def build_exact_split_manifest(
    dataset_root: Path,
    metadata_path: Path,
    *,
    split: str = "development",
    allow_held_out_test: bool = False,
    expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
    file_exists: Callable[[Path], bool] = _path_is_file,
    inventory_reader: Callable[[Path], tuple[int, int]] = inspect_las_inventory,
    file_hasher: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Build one exact split manifest, touching no files from the other split."""

    split_code = normalise_split(split)
    if split_code == "test" and not allow_held_out_test:
        raise PermissionError(
            "Held-out test manifest access requires allow_held_out_test=True"
        )
    root = dataset_root.expanduser().resolve()
    metadata = metadata_path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FOR-instance dataset root does not exist: {root}")
    metadata_rows, observed_metadata_sha256 = read_split_metadata(
        metadata,
        expected_sha256=expected_metadata_sha256,
    )

    sources: list[tuple[str, Path]] = []
    missing: list[str] = []
    for relative in EXPECTED_PATHS[split_code]:
        candidate = (root / relative).resolve()
        try:
            resolved_relative = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Dataset path escapes root: {relative}") from exc
        if resolved_relative != relative:
            raise ValueError(
                f"Dataset path resolution changed the frozen path: {relative}"
            )
        if not file_exists(candidate):
            missing.append(relative)
        sources.append((relative, candidate))
    if missing:
        raise FileNotFoundError(
            f"Frozen {split_code} files are missing: {', '.join(missing)}"
        )

    plots: list[dict[str, Any]] = []
    for task_index, (relative, input_las) in enumerate(sources):
        source = metadata_rows[relative]
        count, reference_tree_count = (
            int(value) for value in inventory_reader(input_las)
        )
        if count <= 0:
            raise ValueError(f"FOR-instance LAS contains no points: {input_las}")
        if reference_tree_count <= 0:
            raise ValueError(
                f"FOR-instance LAS contains no reference trees: {input_las}"
            )
        input_sha256 = str(file_hasher(input_las))
        if not _SHA256_PATTERN.fullmatch(input_sha256):
            raise ValueError(f"Invalid input SHA-256 for {relative}")
        identifier = plot_id(relative)
        plots.append(
            {
                "task_index": task_index,
                "plot_id": identifier,
                "safe_plot_id": safe_plot_id(identifier),
                "relative_path": relative,
                "collection": source["collection"],
                "split": "development" if split_code == "dev" else "test",
                "input_las": str(input_las),
                "point_count": count,
                "reference_tree_count": reference_tree_count,
                "input_sha256": input_sha256,
                "split_metadata": str(metadata),
                "split_metadata_sha256": observed_metadata_sha256,
                "stage0_selected": False,
            }
        )

    stage0 = select_stage0_development_rows(plots) if split_code == "dev" else []
    selected_paths = {row["relative_path"] for row in stage0}
    for row in plots:
        row["stage0_selected"] = row["relative_path"] in selected_paths

    payload = {
        "schema_version": 1,
        "status": "frozen_exact_path_split_manifest",
        "dataset": "FOR-instance",
        "dataset_slug": "for-instance",
        "method_neutral": True,
        "dataset_split": "development" if split_code == "dev" else "test",
        "metadata_split": split_code,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "split_metadata": str(metadata),
        "split_metadata_sha256": observed_metadata_sha256,
        "mapping_rule": "exact_metadata_path_only",
        "expected_plot_count": len(EXPECTED_PATHS[split_code]),
        "expected_site_counts": EXPECTED_SITE_COUNTS[split_code],
        "held_out_test_accessed": split_code == "test",
        "held_out_metrics_computed": False,
        "tuning_eligible": split_code == "dev",
        "stage0_selection_rule": STAGE0_SELECTION_RULE if split_code == "dev" else None,
        "stage0_selection": stage0,
        "plots": plots,
    }
    return validate_manifest_payload(
        payload,
        expected_split=split_code,
        expected_metadata_sha256=expected_metadata_sha256,
        allow_held_out_test=allow_held_out_test,
    )


def validate_manifest_payload(
    payload: dict[str, Any],
    *,
    expected_split: str,
    expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    """Validate a frozen manifest without opening any point-cloud files."""

    if not isinstance(payload, dict):
        raise ValueError("FOR-instance manifest must be a JSON object")
    split_code = normalise_split(expected_split)
    if split_code == "test" and not allow_held_out_test:
        raise PermissionError(
            "Held-out test manifest validation requires allow_held_out_test=True"
        )
    expected_top_level = {
        "schema_version": 1,
        "status": "frozen_exact_path_split_manifest",
        "dataset": "FOR-instance",
        "dataset_slug": "for-instance",
        "method_neutral": True,
        "dataset_split": "development" if split_code == "dev" else "test",
        "metadata_split": split_code,
        "split_metadata_sha256": expected_metadata_sha256,
        "mapping_rule": "exact_metadata_path_only",
        "expected_plot_count": len(EXPECTED_PATHS[split_code]),
        "expected_site_counts": EXPECTED_SITE_COUNTS[split_code],
        "held_out_test_accessed": split_code == "test",
        "held_out_metrics_computed": False,
        "tuning_eligible": split_code == "dev",
    }
    for field, value in expected_top_level.items():
        if payload.get(field) != value:
            raise ValueError(f"Manifest has unexpected {field}: {payload.get(field)!r}")

    root = Path(str(payload.get("dataset_root", ""))).expanduser()
    metadata = Path(str(payload.get("split_metadata", ""))).expanduser()
    if not root.is_absolute() or not metadata.is_absolute():
        raise ValueError("Manifest dataset and split metadata paths must be absolute")
    root = root.resolve()
    metadata = metadata.resolve()
    plots = payload.get("plots")
    if not isinstance(plots, list):
        raise ValueError("Manifest plots must be a list")
    if len(plots) != len(EXPECTED_PATHS[split_code]):
        raise ValueError(
            f"Expected {len(EXPECTED_PATHS[split_code])} {split_code} plots, "
            f"found {len(plots)}"
        )

    normalised: list[dict[str, Any]] = []
    for task_index, source in enumerate(plots):
        if not isinstance(source, dict):
            raise ValueError(f"Manifest row {task_index} is not an object")
        missing = set(MANIFEST_FIELDS) - set(source)
        if missing:
            raise ValueError(f"Manifest row is missing fields {sorted(missing)}")
        row = dict(source)
        if int(row["task_index"]) != task_index:
            raise ValueError("Manifest task indexes must be contiguous and zero-based")
        relative = strict_relative_las_path(str(row["relative_path"]))
        if relative != EXPECTED_PATHS[split_code][task_index]:
            raise ValueError(
                f"Manifest paths or order differ from the frozen {split_code} contract"
            )
        collection = str(row["collection"])
        if collection != Path(relative).parts[0]:
            raise ValueError(f"Manifest collection/path mismatch for {relative}")
        expected_row_split = "development" if split_code == "dev" else "test"
        if str(row["split"]) != expected_row_split:
            raise ValueError(f"Split leakage in manifest row: {relative}")
        identifier = plot_id(relative)
        if str(row["plot_id"]) != identifier:
            raise ValueError(f"Manifest plot ID mismatch for {relative}")
        if str(row["safe_plot_id"]) != safe_plot_id(identifier):
            raise ValueError(f"Manifest safe plot ID mismatch for {relative}")
        input_las = Path(str(row["input_las"])).expanduser()
        if not input_las.is_absolute():
            raise ValueError(f"Manifest input LAS is not absolute: {relative}")
        input_las = input_las.resolve()
        try:
            input_relative = input_las.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Manifest input escapes dataset root: {relative}") from exc
        if input_relative != relative:
            raise ValueError(f"Manifest input path mismatch for {relative}")
        if Path(str(row["split_metadata"])).expanduser().resolve() != metadata:
            raise ValueError(f"Manifest split metadata path mismatch for {relative}")
        if str(row["split_metadata_sha256"]) != expected_metadata_sha256:
            raise ValueError(f"Manifest metadata SHA-256 mismatch for {relative}")
        input_sha256 = str(row["input_sha256"])
        if not _SHA256_PATTERN.fullmatch(input_sha256):
            raise ValueError(f"Manifest input SHA-256 is invalid for {relative}")
        point_count = int(row["point_count"])
        reference_tree_count = int(row["reference_tree_count"])
        if point_count <= 0:
            raise ValueError(f"Manifest point count must be positive for {relative}")
        if reference_tree_count <= 0:
            raise ValueError(
                f"Manifest reference-tree count must be positive for {relative}"
            )
        if not isinstance(row["stage0_selected"], bool):
            raise ValueError(f"Manifest Stage 0 flag is not Boolean for {relative}")
        row.update(
            {
                "task_index": task_index,
                "relative_path": relative,
                "collection": collection,
                "split": expected_row_split,
                "input_las": str(input_las),
                "point_count": point_count,
                "reference_tree_count": reference_tree_count,
                "split_metadata": str(metadata),
            }
        )
        normalised.append(row)

    paths = [row["relative_path"] for row in normalised]
    safe_ids = [row["safe_plot_id"] for row in normalised]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest contains duplicate relative paths")
    if len(safe_ids) != len(set(safe_ids)):
        raise ValueError("Manifest contains colliding safe plot IDs")
    site_counts = Counter(row["collection"] for row in normalised)
    if dict(site_counts) != EXPECTED_SITE_COUNTS[split_code]:
        raise ValueError(f"Manifest {split_code} site counts differ from the contract")

    if split_code == "dev":
        if payload.get("stage0_selection_rule") != STAGE0_SELECTION_RULE:
            raise ValueError("Manifest Stage 0 selection rule is not frozen")
        expected_selection = select_stage0_development_rows(normalised)
        if payload.get("stage0_selection") != expected_selection:
            raise ValueError("Manifest Stage 0 selection differs from the median rule")
        selected_paths = {row["relative_path"] for row in expected_selection}
        actual_paths = {
            row["relative_path"] for row in normalised if row["stage0_selected"]
        }
        if actual_paths != selected_paths:
            raise ValueError("Manifest Stage 0 flags differ from the frozen selection")
    else:
        if payload.get("stage0_selection_rule") is not None:
            raise ValueError("Held-out manifest must not define a Stage 0 rule")
        if payload.get("stage0_selection") != []:
            raise ValueError("Held-out manifest must not contain Stage 0 selections")
        if any(row["stage0_selected"] for row in normalised):
            raise ValueError("Held-out manifest must not mark Stage 0 plots")

    result = dict(payload)
    result["dataset_root"] = str(root)
    result["split_metadata"] = str(metadata)
    result["plots"] = normalised
    return result


def load_and_validate_manifest(
    path: Path,
    *,
    expected_split: str,
    allow_held_out_test: bool = False,
    expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FOR-instance manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_manifest_payload(
        payload,
        expected_split=expected_split,
        expected_metadata_sha256=expected_metadata_sha256,
        allow_held_out_test=allow_held_out_test,
    )


def load_and_verify_manifest_plot(
    path: Path,
    *,
    task_index: int,
    expected_split: str,
    allow_held_out_test: bool = False,
    expected_metadata_sha256: str = EXPECTED_METADATA_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one exact manifest row and its current source-file identities.

    Only the selected point cloud is hashed, so development execution never
    opens a held-out LAS while still detecting stale or redirected manifests.
    """

    payload = load_and_validate_manifest(
        path,
        expected_split=expected_split,
        allow_held_out_test=allow_held_out_test,
        expected_metadata_sha256=expected_metadata_sha256,
    )
    metadata = Path(payload["split_metadata"])
    _, observed_metadata_sha256 = read_split_metadata(
        metadata,
        expected_sha256=expected_metadata_sha256,
    )
    if observed_metadata_sha256 != payload["split_metadata_sha256"]:
        raise ValueError("Manifest split metadata differs from the verified file")

    plots = payload["plots"]
    if task_index < 0 or task_index >= len(plots):
        raise ValueError(f"Manifest has no task_index={task_index}")
    row = dict(plots[task_index])
    if int(row["task_index"]) != task_index:
        raise ValueError(f"Manifest has no unique task_index={task_index}")
    input_las = Path(row["input_las"])
    if not input_las.is_file():
        raise FileNotFoundError(f"Manifest LAS does not exist: {input_las}")
    observed_input_sha256 = sha256_file(input_las)
    if observed_input_sha256 != row["input_sha256"]:
        raise ValueError(
            "Manifest/source input SHA-256 mismatch for "
            f"{row['relative_path']}: expected {row['input_sha256']}, "
            f"found {observed_input_sha256}"
        )
    row["observed_input_sha256"] = observed_input_sha256
    return payload, row
