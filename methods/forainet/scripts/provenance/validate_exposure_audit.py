"""Validate the public-safe 32-row ForAINet exposure table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_TEST_PATHS = {
    "CULS/plot_2_annotated.las",
    "NIBIO/plot_1_annotated.las",
    "NIBIO/plot_5_annotated.las",
    "NIBIO/plot_17_annotated.las",
    "NIBIO/plot_18_annotated.las",
    "NIBIO/plot_22_annotated.las",
    "NIBIO/plot_23_annotated.las",
    "RMIT/test.las",
    "SCION/plot_31_annotated.las",
    "SCION/plot_61_annotated.las",
    "TUWIEN/test.las",
}


def validate(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "relative_path",
        "collection",
        "benchmark_split",
        "checkpoint_role",
        "evidence",
        "mapping_status",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("exposure table schema is invalid")
    if len(rows) != 32:
        raise ValueError("exposure table must contain exactly 32 rows")
    paths = [row["relative_path"] for row in rows]
    if len(set(paths)) != 32:
        raise ValueError("exposure table contains duplicate paths")
    test_rows = [row for row in rows if row["benchmark_split"] == "test"]
    dev_rows = [row for row in rows if row["benchmark_split"] == "dev"]
    if len(test_rows) != 11 or len(dev_rows) != 21:
        raise ValueError("exposure table must preserve the 21/11 split")
    if {row["relative_path"] for row in test_rows} != EXPECTED_TEST_PATHS:
        raise ValueError("held-out paths do not equal the official test list")
    if any(row["checkpoint_role"] != "test_only" for row in test_rows):
        raise ValueError("every held-out path must be documented test-only")
    if any(row["checkpoint_role"] != "train_or_validation" for row in dev_rows):
        raise ValueError("every development path must remain in the combined fit pool")
    if any(not row["evidence"] or not row["mapping_status"] for row in rows):
        raise ValueError("every exposure row needs evidence and mapping status")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exposure_csv", type=Path)
    args = parser.parse_args()
    validate(args.exposure_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
