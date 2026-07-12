"""Shared strict contracts for TreeLearn FOR-instance development reporting."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_DEVELOPMENT_SITE_COUNTS = {
    "CULS": 2,
    "NIBIO": 14,
    "RMIT": 1,
    "SCION": 3,
    "TUWIEN": 1,
}
EXPECTED_DEVELOPMENT_PLOTS = sum(EXPECTED_DEVELOPMENT_SITE_COUNTS.values())
EXPECTED_DEVELOPMENT_SITE_POINTS = {
    "CULS": 4_901_588,
    "NIBIO": 79_435_164,
    "RMIT": 1_483_208,
    "SCION": 8_380_233,
    "TUWIEN": 7_568_844,
}
EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES = {
    "CULS": 27,
    "NIBIO": 414,
    "RMIT": 159,
    "SCION": 92,
    "TUWIEN": 115,
}
EXPECTED_DEVELOPMENT_POINTS = sum(EXPECTED_DEVELOPMENT_SITE_POINTS.values())
EXPECTED_DEVELOPMENT_REFERENCE_TREES = sum(
    EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES.values()
)
EXPECTED_DEVELOPMENT_PATHS = (
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
)

MANIFEST_FIELDS = [
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
]


def strict_relative_path(value: str) -> str:
    """Return a normalised safe POSIX relative path without guessing aliases."""

    raw = value.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("./"):
        raise ValueError(f"Unsafe or non-canonical relative path: {value!r}")
    path = Path(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe or non-canonical relative path: {value!r}")
    normalised = path.as_posix()
    if normalised != raw:
        raise ValueError(f"Non-canonical relative path: {value!r}")
    if path.suffix.casefold() != ".las":
        raise ValueError(f"FOR-instance manifest path is not a LAS file: {value!r}")
    return normalised


def plot_id(relative_path: str) -> str:
    return Path(strict_relative_path(relative_path)).with_suffix("").as_posix()


def safe_plot_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    if not safe or not re.fullmatch(r"[A-Za-z0-9._-]+", safe):
        raise ValueError(f"Could not create a safe plot ID from {value!r}")
    return safe


def _as_int(row: dict[str, Any], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Manifest row has invalid {field}: {row!r}") from exc
    if value < 0:
        raise ValueError(f"Manifest row has negative {field}: {row!r}")
    return value


def validate_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalise the frozen 21-plot development manifest."""

    if len(rows) != EXPECTED_DEVELOPMENT_PLOTS:
        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT_PLOTS} development plots, "
            f"found {len(rows)}"
        )
    normalised: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        missing = set(MANIFEST_FIELDS) - set(source)
        if missing:
            raise ValueError(f"Manifest row is missing fields {sorted(missing)}")
        row = dict(source)
        task_index = _as_int(row, "task_index")
        if task_index != position:
            raise ValueError(
                f"Manifest task indexes must be contiguous 0-based values; "
                f"row {position} records {task_index}"
            )
        relative = strict_relative_path(str(row["relative_path"]))
        collection = str(row["collection"]).strip()
        if collection != Path(relative).parts[0]:
            raise ValueError(
                f"Manifest collection/path mismatch: {collection!r}, {relative!r}"
            )
        if str(row["split"]).strip() != "dev":
            raise ValueError(f"Non-development path in TreeLearn manifest: {relative}")
        expected_plot_id = plot_id(relative)
        if str(row["plot_id"]) != expected_plot_id:
            raise ValueError(f"Manifest plot ID mismatch for {relative}")
        if str(row["safe_plot_id"]) != safe_plot_id(expected_plot_id):
            raise ValueError(f"Manifest safe plot ID mismatch for {relative}")
        input_path = Path(str(row["input_las"])).expanduser()
        if not input_path.is_absolute():
            raise ValueError(f"Manifest input LAS is not absolute: {input_path}")
        for field in ("input_sha256", "split_metadata_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row[field])):
                raise ValueError(f"Manifest {field} is not SHA-256 for {relative}")
        row.update(
            {
                "task_index": task_index,
                "relative_path": relative,
                "collection": collection,
                "split": "dev",
                "input_las": str(input_path.resolve()),
                "point_count": _as_int(row, "point_count"),
                "reference_tree_count": _as_int(row, "reference_tree_count"),
                "split_metadata": str(
                    Path(str(row["split_metadata"])).expanduser().resolve()
                ),
            }
        )
        if row["point_count"] <= 0:
            raise ValueError(f"Manifest point count must be positive for {relative}")
        if row["reference_tree_count"] <= 0:
            raise ValueError(
                f"Manifest reference tree count must be positive for {relative}"
            )
        normalised.append(row)

    paths = [row["relative_path"] for row in normalised]
    safe_ids = [row["safe_plot_id"] for row in normalised]
    if len(set(paths)) != len(paths):
        raise ValueError("Development manifest contains duplicate relative paths")
    if tuple(paths) != EXPECTED_DEVELOPMENT_PATHS:
        raise ValueError(
            "Development manifest paths or task order differ from the exact frozen "
            f"contract: expected {EXPECTED_DEVELOPMENT_PATHS}, found {tuple(paths)}"
        )
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("Development manifest contains colliding safe plot IDs")
    observed_counts = Counter(row["collection"] for row in normalised)
    if dict(observed_counts) != EXPECTED_DEVELOPMENT_SITE_COUNTS:
        raise ValueError(
            "Development site counts differ from the frozen contract: "
            f"expected {EXPECTED_DEVELOPMENT_SITE_COUNTS}, "
            f"found {dict(observed_counts)}"
        )
    observed_points = {
        site: sum(
            int(row["point_count"])
            for row in normalised
            if row["collection"] == site
        )
        for site in EXPECTED_DEVELOPMENT_SITE_COUNTS
    }
    if observed_points != EXPECTED_DEVELOPMENT_SITE_POINTS:
        raise ValueError(
            "Development site point counts differ from the frozen contract: "
            f"expected {EXPECTED_DEVELOPMENT_SITE_POINTS}, found {observed_points}"
        )
    observed_references = {
        site: sum(
            int(row["reference_tree_count"])
            for row in normalised
            if row["collection"] == site
        )
        for site in EXPECTED_DEVELOPMENT_SITE_COUNTS
    }
    if observed_references != EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES:
        raise ValueError(
            "Development site reference-tree counts differ from the frozen contract: "
            f"expected {EXPECTED_DEVELOPMENT_SITE_REFERENCE_TREES}, "
            f"found {observed_references}"
        )
    split_hashes = {row["split_metadata_sha256"] for row in normalised}
    split_paths = {row["split_metadata"] for row in normalised}
    if len(split_hashes) != 1 or len(split_paths) != 1:
        raise ValueError("Manifest rows do not share one frozen split metadata file")
    return normalised


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load either the JSON or CSV representation of a development manifest."""

    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Development manifest does not exist: {manifest_path}")
    if manifest_path.suffix.casefold() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("plots"), list):
            raise ValueError(f"Invalid development manifest JSON: {manifest_path}")
        if payload.get("dataset_split") != "dev":
            raise ValueError("Development manifest JSON is not development-only")
        if payload.get("held_out_test_accessed") is not False:
            raise ValueError("Development manifest does not explicitly lock held-out test")
        rows = payload["plots"]
        metadata = payload
    elif manifest_path.suffix.casefold() == ".csv":
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        metadata = {
            "dataset_split": "dev",
            "held_out_test_accessed": False,
            "manifest_path": str(manifest_path),
        }
    else:
        raise ValueError("Development manifest must be a .json or .csv file")
    return validate_manifest_rows(rows), metadata
