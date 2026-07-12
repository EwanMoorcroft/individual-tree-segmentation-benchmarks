"""Build the immutable 16-plot tuning crop view from per-plot crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_plot_inventory(row: dict, expected_per_plot: int) -> dict:
    path = Path(row["crop_inventory"])
    inventory = json.loads(path.read_text())
    if (
        inventory.get("status") != "treelearn_plot_crops_sha256_inventoried"
        or inventory.get("held_out_test_accessed") is not False
        or int(inventory.get("crop_seed", -1)) != int(row["crop_seed"])
        or int(inventory.get("crop_count", -1)) != expected_per_plot
    ):
        raise ValueError(f"Invalid crop inventory: {path}")
    files = sorted(Path(row["crop_root"]).glob("*.npz"))
    entries = inventory.get("files", [])
    if len(files) != len(entries) or [item.name for item in files] != [
        item["name"] for item in entries
    ]:
        raise ValueError(f"Crop inventory names changed: {path}")
    if any(file.stat().st_size != int(entry["size_bytes"]) for file, entry in zip(files, entries)):
        raise ValueError(f"Crop inventory sizes changed: {path}")
    if any(sha256(file) != entry["sha256"] for file, entry in zip(files, entries)):
        raise ValueError(f"Crop inventory hashes changed: {path}")
    return {
        "safe_plot_id": row["safe_plot_id"],
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "entries_aggregate_sha256": inventory["entries_aggregate_sha256"],
        "crop_seed": int(row["crop_seed"]),
    }


def link_view(rows: list[dict], target: Path, expected_per_plot: int) -> tuple[int, int]:
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    count = 0
    size = 0
    for row in rows:
        source_files = sorted(Path(row["crop_root"]).glob("*.npz"))
        if len(source_files) != expected_per_plot:
            raise ValueError(
                f'{row["safe_plot_id"]}: expected {expected_per_plot} crops, '
                f"found {len(source_files)}"
            )
        for index, source in enumerate(source_files):
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = target / f'{row["safe_plot_id"]}__{index:04d}.npz'
            destination.symlink_to(source.resolve())
            count += 1
            size += source.stat().st_size
    return count, size


def consolidate(freeze_path: Path, output: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    rows = freeze["plots"]
    if len(rows) != 21 or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Expected exactly 21 development plots")
    expected = int(freeze["crops_per_plot"])
    train_rows = [row for row in rows if row["training_role"] == "train"]
    if len(train_rows) != 16:
        raise ValueError("Expected exactly 16 tuning-training plots")
    plot_inventories = [verify_plot_inventory(row, expected) for row in train_rows]
    tuning_root = Path(freeze["tuning_data_root"])
    tuning_count, tuning_bytes = link_view(train_rows, tuning_root, expected)
    if tuning_count != 16 * expected:
        raise ValueError("Consolidated crop counts differ from the frozen contract")
    payload = {
        "schema_version": 1,
        "status": "development_crops_consolidated",
        "held_out_test_accessed": False,
        "crops_per_plot": expected,
        "tuning": {
            "plots": 16, "crop_count": tuning_count, "referenced_size_bytes": tuning_bytes,
            "root": str(tuning_root.resolve()),
        },
        "plot_crop_inventories": plot_inventories,
        "plot_crop_inventory_count": len(plot_inventories),
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = consolidate(args.freeze.resolve(), args.output.resolve())
    print(f'tuning_crops={result["tuning"]["crop_count"]}')
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
