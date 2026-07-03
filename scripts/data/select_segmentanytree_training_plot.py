"""Select one plot from a generated SegmentAnyTree training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def select_record(
    manifest_path: Path,
    role: str,
    array_index: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest.get("records", [])
        if record.get("selected_for_profile")
        and record.get("training_role") == role
    ]
    if array_index < 0 or array_index >= len(records):
        raise IndexError(
            f"Array index {array_index} is outside the {role} selection "
            f"0..{max(len(records) - 1, 0)}"
        )
    return records[array_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one plot from a SegmentAnyTree split manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--role",
        choices=("train", "val", "held_out_test"),
        required=True,
    )
    parser.add_argument("--array-index", type=int, required=True)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = select_record(
        Path(args.manifest).expanduser().resolve(),
        args.role,
        args.array_index,
    )
    if args.format == "json":
        print(json.dumps(record, sort_keys=True))
    else:
        for field in (
            "source_path",
            "relative_path",
            "collection",
            "plot_name",
            "dataset_split",
            "training_role",
            "converted_ply",
        ):
            print(record.get(field) or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
