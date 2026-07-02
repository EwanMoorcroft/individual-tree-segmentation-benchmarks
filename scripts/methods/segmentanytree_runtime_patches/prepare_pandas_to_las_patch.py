"""Prepare the narrow SegmentAnyTree LAS export compatibility patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCAN_ANGLE_OLD = "'scan_angle': 'uint16'"
SCAN_ANGLE_NEW = "'scan_angle': 'int16'"
CAST_OLD = (
    "las_file[column] = "
    "df[column].astype(standard_columns_with_data_types[column])"
)
CAST_NEW = (
    "values = df[column].round() if column == \"scan_angle\" else df[column]\n"
    "            las_file[column] = "
    "values.astype(standard_columns_with_data_types[column])"
)


def patch_source(source: str) -> str:
    if source.count(SCAN_ANGLE_OLD) != 1:
        raise ValueError("Expected one unsigned scan_angle declaration")
    if source.count(CAST_OLD) != 1:
        raise ValueError("Expected one standard LAS column cast")
    patched = source.replace(SCAN_ANGLE_OLD, SCAN_ANGLE_NEW)
    patched = patched.replace(CAST_OLD, CAST_NEW)
    if patched == source:
        raise ValueError("No export compatibility changes were applied")
    return patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the upstream pandas_to_las module and correct scan_angle "
            "conversion for LAS point format 6."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Upstream module does not exist: {source_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    patched = patch_source(source_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")

    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "output": str(output_path),
                    "changes": [
                        "scan_angle uses signed int16",
                        "scan_angle values are rounded before integer conversion",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
