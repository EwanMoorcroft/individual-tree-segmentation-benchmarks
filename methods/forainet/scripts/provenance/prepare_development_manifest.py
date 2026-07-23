"""Freeze the exact 21 original FOR-instance development sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import laspy


EXPECTED_DEVELOPMENT_PLOTS = 21
EXPECTED_TEST_PLOTS = 11
EXPECTED_SMOKE_RUN_ID = (
    "forainet__for-instance__published-pretrained__none__dev-smoke__"
    "20260723T202654"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("split metadata is empty")
    path_key = "path" if "path" in rows[0] else "relative_path"
    if path_key not in rows[0] or "split" not in rows[0]:
        raise ValueError("split metadata lacks path or split columns")
    normalised = []
    for row in rows:
        relative = row[path_key].strip().replace("\\", "/")
        split = row["split"].strip()
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe dataset relative path: {relative!r}")
        if split not in {"dev", "test"}:
            raise ValueError(f"unexpected split for {relative}: {split!r}")
        normalised.append({"relative_path": relative, "split": split})
    paths = [row["relative_path"] for row in normalised]
    if len(paths) != len(set(paths)):
        raise ValueError("split metadata contains duplicate paths")
    return normalised


def build(
    *,
    dataset_root: Path,
    split_metadata: Path,
    accepted_smoke: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    smoke = json.loads(accepted_smoke.read_text(encoding="utf-8"))
    if (
        smoke.get("schema") != "forainet_accepted_development_smoke_v1"
        or smoke.get("status") != "accepted"
        or smoke.get("run_id") != EXPECTED_SMOKE_RUN_ID
        or smoke.get("held_out_access") is not False
    ):
        raise ValueError("accepted development smoke evidence is not frozen")
    metadata_rows = read_split_rows(split_metadata)
    metadata_by_path = {row["relative_path"]: row for row in metadata_rows}
    catalogue_paths = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*.las")
        if path.is_file()
    }
    if len(catalogue_paths) != EXPECTED_DEVELOPMENT_PLOTS + EXPECTED_TEST_PLOTS:
        raise ValueError("local LAS catalogue does not contain exactly 32 files")
    missing_metadata = sorted(catalogue_paths - set(metadata_by_path))
    if missing_metadata:
        raise ValueError(f"local LAS files lack split metadata: {missing_metadata}")
    source_rows = [
        row for row in metadata_rows if row["relative_path"] in catalogue_paths
    ]
    development_rows = [
        row for row in source_rows if row["split"] == "dev"
    ]
    test_rows = [row for row in source_rows if row["split"] == "test"]
    if (
        len(development_rows) != EXPECTED_DEVELOPMENT_PLOTS
        or len(test_rows) != EXPECTED_TEST_PLOTS
    ):
        raise ValueError(
            "available LAS catalogue does not preserve the frozen 21/11 boundary"
        )
    plots = []
    for task_index, row in enumerate(
        development_rows
    ):
        relative_path = row["relative_path"]
        source = dataset_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        with laspy.open(source) as handle:
            dimensions = set(handle.header.point_format.dimension_names)
            point_count = int(handle.header.point_count)
        if not {"classification", "treeID"} <= dimensions:
            raise ValueError(f"required LAS dimensions absent: {relative_path}")
        plots.append(
            {
                "task_index": task_index,
                "relative_path": relative_path,
                "split": "dev",
                "source_sha256": sha256(source),
                "size_bytes": source.stat().st_size,
                "point_count": point_count,
            }
        )
    if len(plots) != EXPECTED_DEVELOPMENT_PLOTS:
        raise ValueError("development manifest must contain exactly 21 plots")
    payload = {
        "schema": "forainet_development_manifest_v1",
        "status": "complete",
        "dataset_version": "original_for_instance",
        "expected_plot_count": EXPECTED_DEVELOPMENT_PLOTS,
        "held_out_plot_count": EXPECTED_TEST_PLOTS,
        "held_out_paths_included": False,
        "split_metadata_row_count": len(metadata_rows),
        "available_catalogue_count": len(source_rows),
        "split_metadata_sha256": sha256(split_metadata),
        "accepted_smoke_run_id": smoke["run_id"],
        "accepted_smoke_record_sha256": sha256(accepted_smoke),
        "total_point_count": sum(row["point_count"] for row in plots),
        "total_size_bytes": sum(row["size_bytes"] for row in plots),
        "plots": plots,
    }
    return plots, payload


def write_outputs(
    plots: list[dict[str, Any]],
    payload: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> None:
    for output in (output_csv, output_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plots[0]))
        writer.writeheader()
        writer.writerows(plots)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split-metadata", required=True, type=Path)
    parser.add_argument("--accepted-smoke", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    plots, payload = build(
        dataset_root=args.dataset_root,
        split_metadata=args.split_metadata,
        accepted_smoke=args.accepted_smoke,
    )
    write_outputs(plots, payload, args.output_csv, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
