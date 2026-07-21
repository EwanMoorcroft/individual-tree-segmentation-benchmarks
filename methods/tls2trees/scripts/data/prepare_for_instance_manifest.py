"""Build or validate method-neutral exact FOR-instance split manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import (  # noqa: E402
    MANIFEST_FIELDS,
    STAGE0_SELECTION_RULE,
    build_exact_split_manifest,
    load_and_validate_manifest,
    sha256_file,
)


RESOLVABLE_FIELDS = (
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
)
STAGE0_FIELDS = ("stage0_index", *MANIFEST_FIELDS, "selection_rule")


def add_held_out_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-held-out-test",
        action="store_true",
        help="Explicitly permit test-manifest access; never use during tuning.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or validate exact FOR-instance split manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build one immutable split manifest.")
    build.add_argument("--dataset-root", required=True)
    build.add_argument("--metadata-csv", required=True)
    build.add_argument(
        "--split",
        choices=("development", "test"),
        default="development",
    )
    build.add_argument("--output-json", required=True)
    build.add_argument("--output-csv", required=True)
    add_held_out_flag(build)

    validate = subparsers.add_parser(
        "validate", help="Validate one manifest without opening point clouds."
    )
    validate.add_argument("--manifest-json", required=True)
    validate.add_argument(
        "--expected-split",
        choices=("development", "test"),
        required=True,
    )
    add_held_out_flag(validate)

    select = subparsers.add_parser(
        "select-stage0",
        help="Write the five frozen development Stage 0 rows.",
    )
    select.add_argument("--manifest-json", required=True)
    select.add_argument("--output-json", required=True)
    select.add_argument("--output-csv", required=True)

    resolve = subparsers.add_parser(
        "resolve",
        help="Print one manifest field for a Slurm task index.",
    )
    resolve.add_argument("--manifest-json", required=True)
    resolve.add_argument("--task-index", required=True, type=int)
    resolve.add_argument("--field", required=True, choices=RESOLVABLE_FIELDS)
    resolve.add_argument(
        "--expected-split",
        choices=("development", "test"),
        default="development",
    )
    add_held_out_flag(resolve)

    resolve_stage0_parser = subparsers.add_parser(
        "resolve-stage0",
        help="Print one field for a deterministic Stage 0 array index.",
    )
    resolve_stage0_parser.add_argument("--manifest-json", required=True)
    resolve_stage0_parser.add_argument("--stage0-index", required=True, type=int)
    resolve_stage0_parser.add_argument(
        "--field",
        required=True,
        choices=("stage0_index", *RESOLVABLE_FIELDS),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_csv(path, rows, MANIFEST_FIELDS)


def write_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def stage0_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_task_index = {int(row["task_index"]): row for row in payload["plots"]}
    result: list[dict[str, Any]] = []
    for selected in payload["stage0_selection"]:
        task_index = int(selected["task_index"])
        source = by_task_index.get(task_index)
        if source is None or not source["stage0_selected"]:
            raise ValueError(f"Stage 0 task index is missing or unmarked: {task_index}")
        result.append(
            {
                "stage0_index": int(selected["stage0_index"]),
                **source,
                "selection_rule": STAGE0_SELECTION_RULE,
            }
        )
    if len(result) != 5 or [row["stage0_index"] for row in result] != list(range(5)):
        raise ValueError("Stage 0 selection must contain five contiguous rows")
    return result


def resolve_task(payload: dict[str, Any], task_index: int, field: str) -> Any:
    if field not in RESOLVABLE_FIELDS:
        raise ValueError(f"Unsupported manifest field: {field}")
    matches = [
        row for row in payload["plots"] if int(row["task_index"]) == task_index
    ]
    if len(matches) != 1:
        raise ValueError(f"Manifest has no unique task index {task_index}")
    return matches[0][field]


def resolve_stage0_task(
    payload: dict[str, Any], stage0_index: int, field: str
) -> Any:
    if field not in {"stage0_index", *RESOLVABLE_FIELDS}:
        raise ValueError(f"Unsupported Stage 0 field: {field}")
    matches = [
        row for row in stage0_rows(payload) if row["stage0_index"] == stage0_index
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage 0 selection has no unique index {stage0_index}")
    return matches[0][field]


def build(args: argparse.Namespace) -> int:
    json_path = Path(args.output_json).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve()
    if json_path == csv_path:
        raise ValueError("Manifest JSON and CSV outputs must be different paths")
    collisions = [path for path in (json_path, csv_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Manifest outputs already exist: " + ", ".join(str(path) for path in collisions)
        )
    payload = build_exact_split_manifest(
        Path(args.dataset_root),
        Path(args.metadata_csv),
        split=args.split,
        allow_held_out_test=args.allow_held_out_test,
    )
    write_csv(csv_path, payload["plots"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest_json={json_path}")
    print(f"manifest_csv={csv_path}")
    print(f"split={payload['dataset_split']}")
    print(f"plots={len(payload['plots'])}")
    print(f"held_out_metrics_computed={str(payload['held_out_metrics_computed']).lower()}")
    return 0


def validate(args: argparse.Namespace) -> int:
    payload = load_and_validate_manifest(
        Path(args.manifest_json),
        expected_split=args.expected_split,
        allow_held_out_test=args.allow_held_out_test,
    )
    print(f"manifest_valid=true")
    print(f"split={payload['dataset_split']}")
    print(f"plots={len(payload['plots'])}")
    print(f"held_out_metrics_computed={str(payload['held_out_metrics_computed']).lower()}")
    return 0


def select_stage0(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    payload = load_and_validate_manifest(
        manifest_path,
        expected_split="development",
    )
    json_path = Path(args.output_json).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve()
    if json_path == csv_path:
        raise ValueError("Stage 0 JSON and CSV outputs must be different paths")
    collisions = [path for path in (json_path, csv_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            "Stage 0 outputs already exist: " + ", ".join(str(path) for path in collisions)
        )
    rows = stage0_rows(payload)
    selection = {
        "schema_version": 1,
        "status": "frozen_development_stage0_selection",
        "dataset": "FOR-instance",
        "dataset_split": "development",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "selection_rule": STAGE0_SELECTION_RULE,
        "plot_count": len(rows),
        "original_task_indexes_retained": True,
        "held_out_test_accessed": False,
        "plots": rows,
    }
    write_rows_csv(csv_path, rows, STAGE0_FIELDS)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(f"stage0_json={json_path}")
    print(f"stage0_csv={csv_path}")
    print("plots=5")
    print("held_out_test_accessed=false")
    return 0


def resolve(args: argparse.Namespace) -> int:
    payload = load_and_validate_manifest(
        Path(args.manifest_json),
        expected_split=args.expected_split,
        allow_held_out_test=args.allow_held_out_test,
    )
    print(resolve_task(payload, args.task_index, args.field))
    return 0


def resolve_stage0(args: argparse.Namespace) -> int:
    payload = load_and_validate_manifest(
        Path(args.manifest_json),
        expected_split="development",
    )
    print(resolve_stage0_task(payload, args.stage0_index, args.field))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "validate":
        return validate(args)
    if args.command == "select-stage0":
        return select_stage0(args)
    if args.command == "resolve":
        return resolve(args)
    if args.command == "resolve-stage0":
        return resolve_stage0(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
