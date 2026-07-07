"""Build exact-path-only FOR-instance plot lists for TreeX."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exact-path TreeX development and test plot lists."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--existing-output", required=True)
    parser.add_argument("--missing-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--test-output", required=True)
    return parser.parse_args()


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_id(relative_path: str) -> str:
    path = Path(relative_path)
    return str(path.with_suffix("")).replace("\\", "/")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))

    required_columns = {"path", "folder", "split"}
    if not metadata_rows:
        raise ValueError(f"Metadata CSV contains no rows: {metadata_path}")
    missing_columns = required_columns - set(metadata_rows[0])
    if missing_columns:
        raise ValueError(
            f"Metadata CSV is missing columns {sorted(missing_columns)}"
        )

    existing_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    for row in metadata_rows:
        relative_path = row["path"].strip()
        expected_path = (dataset_root / relative_path).resolve()
        if expected_path.is_file():
            existing_rows.append(
                {
                    "plot_id": plot_id(relative_path),
                    "input_las": str(expected_path),
                    "metadata_path": relative_path,
                    "folder": row["folder"],
                    "split": row["split"],
                    "mapping_rule": "exact",
                }
            )
        else:
            missing_rows.append(
                {
                    "metadata_path": relative_path,
                    "folder": row["folder"],
                    "split": row["split"],
                    "expected_las": str(expected_path),
                }
            )

    existing_rows.sort(key=lambda row: (row["split"], row["plot_id"]))
    missing_rows.sort(key=lambda row: (row["split"], row["metadata_path"]))
    split_fields = ["plot_id", "input_las", "metadata_path", "mapping_rule"]
    write_csv(
        Path(args.existing_output).expanduser().resolve(),
        existing_rows,
        [
            "plot_id",
            "input_las",
            "metadata_path",
            "folder",
            "split",
            "mapping_rule",
        ],
    )
    write_csv(
        Path(args.missing_output).expanduser().resolve(),
        missing_rows,
        ["metadata_path", "folder", "split", "expected_las"],
    )
    write_csv(
        Path(args.dev_output).expanduser().resolve(),
        [
            {field: row[field] for field in split_fields}
            for row in existing_rows
            if row["split"] == "dev"
        ],
        split_fields,
    )
    write_csv(
        Path(args.test_output).expanduser().resolve(),
        [
            {field: row[field] for field in split_fields}
            for row in existing_rows
            if row["split"] == "test"
        ],
        split_fields,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
