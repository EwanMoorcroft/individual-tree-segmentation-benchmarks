"""Resolve one frozen ForAINet development task for a Slurm array."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path


def resolve(path: Path, task_index: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["task_index"]) == task_index]
    if len(rows) != 21 or len(matches) != 1:
        raise ValueError("development task does not resolve exactly once")
    row = matches[0]
    if row["split"] != "dev":
        raise ValueError("development array cannot resolve a held-out path")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    args = parser.parse_args()
    row = resolve(args.manifest_csv, args.task_index)
    keys = {
        "FORAINET_TASK_RELATIVE_PATH": "relative_path",
        "FORAINET_TASK_SOURCE_SHA256": "source_sha256",
        "FORAINET_TASK_POINT_COUNT": "point_count",
    }
    for shell_name, row_name in keys.items():
        print(f"{shell_name}={shlex.quote(row[row_name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
