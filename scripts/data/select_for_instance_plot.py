from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
PATH_COLUMNS = (
    "relative_path",
    "path",
    "file_path",
    "filepath",
    "las_path",
    "filename",
    "file",
    "plot",
    "plot_name",
)
SPLIT_COLUMNS = ("split", "data_split", "dataset_split", "partition", "set")
COLLECTION_COLUMNS = ("collection", "dataset", "site", "source")


def normalise_key(value: str) -> str:
    return value.strip().replace("\\", "/").removeprefix("./")


def first_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    by_lower = {name.strip().lower(): name for name in fieldnames}
    return next((by_lower[name] for name in candidates if name in by_lower), None)


def read_split_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        path_column = first_column(fieldnames, PATH_COLUMNS)
        split_column = first_column(fieldnames, SPLIT_COLUMNS)
        collection_column = first_column(fieldnames, COLLECTION_COLUMNS)
        if path_column is None or split_column is None:
            raise ValueError(
                f"Could not identify path/split columns in {path}; columns: {fieldnames}"
            )

        aliases: dict[str, set[str]] = {}
        for row in reader:
            split = (row.get(split_column) or "").strip()
            path_value = normalise_key(row.get(path_column) or "")
            if not split or not path_value:
                continue
            collection = (
                normalise_key(row.get(collection_column) or "")
                if collection_column
                else ""
            )
            keys = {
                path_value,
                Path(path_value).name,
                Path(path_value).stem,
            }
            if collection and "/" not in path_value:
                keys.add(f"{collection}/{path_value}")
            for key in keys:
                aliases.setdefault(key, set()).add(split)
    return {
        key: next(iter(values))
        for key, values in aliases.items()
        if len(values) == 1
    }


def split_for_relative_path(relative_path: Path, split_lookup: dict[str, str]) -> str:
    candidates = (
        relative_path.as_posix(),
        relative_path.name,
        relative_path.stem,
    )
    return next(
        (split_lookup[key] for key in candidates if key in split_lookup),
        "unassigned",
    )


def discover_plots(
    dataset_root: Path,
    split_metadata_file: str = "data_split_metadata.csv",
    selected_split: str | None = None,
) -> list[dict[str, Any]]:
    split_lookup = read_split_metadata(dataset_root / split_metadata_file)
    records: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*.las")):
        relative_path = path.relative_to(dataset_root)
        split = split_for_relative_path(relative_path, split_lookup)
        if selected_split and split.casefold() != selected_split.casefold():
            continue
        records.append(
            {
                "absolute_path": str(path.resolve()),
                "relative_path": relative_path.as_posix(),
                "collection": relative_path.parts[0]
                if len(relative_path.parts) > 1
                else "",
                "plot_name": path.stem,
                "split": split,
            }
        )
    for index, record in enumerate(records):
        record["array_index"] = index
    return records


def load_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config, path


def select_plot(
    config: dict[str, Any],
    dataset_root_override: str | None = None,
    plot_path: str | None = None,
    selected_split: str | None = None,
    array_index: int | None = None,
) -> dict[str, Any]:
    dataset_root = Path(
        dataset_root_override or config["dataset"]["root"]
    ).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    split_file = config["dataset"].get(
        "split_metadata_file", "data_split_metadata.csv"
    )
    records = discover_plots(dataset_root, split_file, selected_split)
    if plot_path:
        candidate = Path(plot_path).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        candidate = candidate.resolve()
        matches = [
            record
            for record in records
            if Path(record["absolute_path"]) == candidate
        ]
        if not matches:
            raise ValueError(
                f"Plot is not available in the selected dataset/split: {candidate}"
            )
        return matches[0]
    if array_index is None:
        raise ValueError("Supply --plot-path or --array-index")
    if array_index < 0 or array_index >= len(records):
        raise IndexError(
            f"Array index {array_index} is outside 0..{max(len(records) - 1, 0)}"
        )
    return records[array_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one deterministic FOR-instance plot selection."
    )
    parser.add_argument(
        "--config",
        default="configs/for_instance_segmentanytree_benchmark.yml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--plot-path")
    parser.add_argument("--split")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = load_config(args.config)
    record = select_plot(
        config,
        dataset_root_override=args.dataset_root,
        plot_path=args.plot_path,
        selected_split=args.split,
        array_index=args.array_index,
    )
    if args.format == "json":
        print(json.dumps(record, sort_keys=True))
    else:
        for field in (
            "absolute_path",
            "relative_path",
            "collection",
            "plot_name",
            "split",
            "array_index",
        ):
            print(record[field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
