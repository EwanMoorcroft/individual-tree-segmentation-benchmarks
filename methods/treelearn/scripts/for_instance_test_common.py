"""Strict manifest contract for the local 11-plot FOR-instance test subset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from for_instance_development_common import (
    MANIFEST_FIELDS,
    plot_id,
    safe_plot_id,
    strict_relative_path,
)


EXPECTED_TEST_SITE_COUNTS = {
    "CULS": 1,
    "NIBIO": 6,
    "RMIT": 1,
    "SCION": 2,
    "TUWIEN": 1,
}
EXPECTED_TEST_SITE_POINTS = {
    "CULS": 3_946_098,
    "NIBIO": 36_559_549,
    "RMIT": 357_435,
    "SCION": 6_566_791,
    "TUWIEN": 2_280_049,
}
EXPECTED_TEST_SITE_REFERENCE_TREES = {
    "CULS": 20,
    "NIBIO": 161,
    "RMIT": 64,
    "SCION": 43,
    "TUWIEN": 35,
}
EXPECTED_TEST_PATHS = (
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
)
EXPECTED_TEST_INVENTORY = {
    "CULS/plot_2_annotated.las": (3_946_098, 20),
    "NIBIO/plot_17_annotated.las": (6_890_118, 30),
    "NIBIO/plot_18_annotated.las": (6_915_118, 27),
    "NIBIO/plot_1_annotated.las": (5_000_698, 37),
    "NIBIO/plot_22_annotated.las": (6_366_607, 20),
    "NIBIO/plot_23_annotated.las": (6_163_377, 28),
    "NIBIO/plot_5_annotated.las": (5_223_631, 19),
    "RMIT/test.las": (357_435, 64),
    "SCION/plot_31_annotated.las": (2_977_537, 25),
    "SCION/plot_61_annotated.las": (3_589_254, 18),
    "TUWIEN/test.las": (2_280_049, 35),
}
EXPECTED_TEST_PLOTS = len(EXPECTED_TEST_PATHS)
EXPECTED_TEST_POINTS = sum(EXPECTED_TEST_SITE_POINTS.values())
EXPECTED_TEST_REFERENCE_TREES = sum(
    EXPECTED_TEST_SITE_REFERENCE_TREES.values()
)


def validate_test_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_TEST_PLOTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_PLOTS} test plots, found {len(rows)}"
        )
    normalised: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        missing = set(MANIFEST_FIELDS) - set(source)
        if missing:
            raise ValueError(f"Test manifest row is missing {sorted(missing)}")
        row = dict(source)
        relative = strict_relative_path(str(row["relative_path"]))
        collection = str(row["collection"])
        if int(row["task_index"]) != position:
            raise ValueError("Test manifest task indexes must be contiguous")
        if collection != Path(relative).parts[0]:
            raise ValueError(f"Test manifest collection mismatch for {relative}")
        if str(row["split"]) != "test":
            raise ValueError(f"Non-test row in held-out manifest: {relative}")
        if str(row["plot_id"]) != plot_id(relative):
            raise ValueError(f"Test manifest plot ID mismatch for {relative}")
        if str(row["safe_plot_id"]) != safe_plot_id(plot_id(relative)):
            raise ValueError(f"Test manifest safe plot ID mismatch for {relative}")
        input_las = Path(str(row["input_las"])).expanduser().resolve()
        split_metadata = Path(str(row["split_metadata"])).expanduser().resolve()
        if not input_las.is_absolute() or not split_metadata.is_absolute():
            raise ValueError("Test manifest paths must be absolute")
        for field in ("input_sha256", "split_metadata_sha256"):
            value = str(row[field])
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"Invalid {field} for {relative}")
        point_count = int(row["point_count"])
        reference_count = int(row["reference_tree_count"])
        if point_count <= 0 or reference_count <= 0:
            raise ValueError(f"Invalid inventory counts for {relative}")
        if (point_count, reference_count) != EXPECTED_TEST_INVENTORY[relative]:
            raise ValueError(f"Test inventory differs from the frozen contract: {relative}")
        row.update(
            task_index=position,
            relative_path=relative,
            collection=collection,
            split="test",
            input_las=str(input_las),
            split_metadata=str(split_metadata),
            point_count=point_count,
            reference_tree_count=reference_count,
        )
        normalised.append(row)

    if tuple(row["relative_path"] for row in normalised) != EXPECTED_TEST_PATHS:
        raise ValueError("Test paths or order differ from the frozen 11-plot contract")
    counts = Counter(row["collection"] for row in normalised)
    if dict(counts) != EXPECTED_TEST_SITE_COUNTS:
        raise ValueError("Test site counts differ from the frozen contract")
    points = {
        site: sum(
            row["point_count"] for row in normalised if row["collection"] == site
        )
        for site in EXPECTED_TEST_SITE_COUNTS
    }
    references = {
        site: sum(
            row["reference_tree_count"]
            for row in normalised
            if row["collection"] == site
        )
        for site in EXPECTED_TEST_SITE_COUNTS
    }
    if points != EXPECTED_TEST_SITE_POINTS:
        raise ValueError("Test point counts differ from the frozen contract")
    if references != EXPECTED_TEST_SITE_REFERENCE_TREES:
        raise ValueError("Test reference-tree counts differ from the frozen contract")
    return normalised


def load_test_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Test manifest must be a JSON object")
    expected = {
        "status": "frozen_for_one_time_held_out_test",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "training_mode": "fine_tuned_on_dev",
        "expected_test_plot_count": EXPECTED_TEST_PLOTS,
        "repeat_test_for_setting_selection_permitted": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Test manifest has unexpected {field}")
    return validate_test_rows(payload.get("plots") or []), payload
