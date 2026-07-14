"""Build and verify the immutable 16-plot tuning crop view."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_training_rows(freeze_path: Path) -> tuple[dict, list[dict], int]:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    rows = freeze.get("plots", [])
    if len(rows) != 21 or any(row.get("split") != "dev" for row in rows):
        raise ValueError("Expected exactly 21 development plots")
    train_rows = [row for row in rows if row.get("training_role") == "train"]
    if len(train_rows) != 16:
        raise ValueError("Expected exactly 16 tuning-training plots")
    return freeze, train_rows, int(freeze["crops_per_plot"])


def verify_plot_inventory(row: dict, expected_per_plot: int) -> tuple[dict, list[dict]]:
    path = Path(row["crop_inventory"]).expanduser().resolve()
    inventory = json.loads(path.read_text())
    crop_root = Path(row["crop_root"]).expanduser().resolve()
    if (
        inventory.get("status") != "treelearn_plot_crops_sha256_inventoried"
        or inventory.get("held_out_test_accessed") is not False
        or inventory.get("safe_plot_id") != row.get("safe_plot_id")
        or Path(str(inventory.get("crop_root", ""))).expanduser().resolve() != crop_root
        or int(inventory.get("crop_seed", -1)) != int(row["crop_seed"])
        or int(inventory.get("crop_count", -1)) != expected_per_plot
    ):
        raise ValueError(f"Invalid crop inventory: {path}")
    entries = inventory.get("files")
    if not isinstance(entries, list) or len(entries) != expected_per_plot:
        raise ValueError(f"Invalid crop inventory entries: {path}")
    files = sorted(crop_root.glob("*.npz"))
    names = [str(entry.get("name", "")) for entry in entries]
    if len(files) != len(entries) or [item.name for item in files] != names:
        raise ValueError(f"Crop inventory names changed: {path}")

    aggregate = hashlib.sha256()
    verified_files: list[dict] = []
    total_bytes = 0
    for file, entry in zip(files, entries):
        try:
            size = int(entry["size_bytes"])
            digest = str(entry["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid crop entry in {path}") from exc
        if (
            file.is_symlink()
            or not file.is_file()
            or size <= 0
            or not SHA256.fullmatch(digest)
            or Path(entry["name"]).name != entry["name"]
        ):
            raise ValueError(f"Invalid crop file identity: {file}")
        if file.stat().st_size != size:
            raise ValueError(f"Crop inventory sizes changed: {path}")
        if sha256(file) != digest:
            raise ValueError(f"Crop inventory hashes changed: {path}")
        aggregate.update(f'{entry["name"]}\0{size}\0{digest}\n'.encode("utf-8"))
        total_bytes += size
        verified_files.append({
            "source": str(file.resolve(strict=True)),
            "size_bytes": size,
            "sha256": digest,
        })
    if (
        aggregate.hexdigest() != inventory.get("entries_aggregate_sha256")
        or total_bytes != int(inventory.get("total_size_bytes", -1))
    ):
        raise ValueError(f"Crop inventory aggregate changed: {path}")
    summary = {
        "safe_plot_id": row["safe_plot_id"],
        "path": str(path),
        "sha256": sha256(path),
        "entries_aggregate_sha256": inventory["entries_aggregate_sha256"],
        "crop_seed": int(row["crop_seed"]),
    }
    return summary, verified_files


def link_descriptors(verified_rows: list[tuple[dict, list[dict]]]) -> list[dict]:
    descriptors: list[dict] = []
    names: set[str] = set()
    for row, files in verified_rows:
        for index, source in enumerate(files):
            name = f'{row["safe_plot_id"]}__{index:04d}.npz'
            if Path(name).name != name or name in names:
                raise ValueError(f"Unsafe or duplicate consolidated crop name: {name}")
            names.add(name)
            descriptors.append({
                "name": name,
                "target": source["source"],
                "size_bytes": source["size_bytes"],
                "sha256": source["sha256"],
            })
    return sorted(descriptors, key=lambda entry: entry["name"])


def aggregate_links(descriptors: list[dict]) -> str:
    aggregate = hashlib.sha256()
    for entry in descriptors:
        aggregate.update(
            f'{entry["name"]}\0{entry["target"]}\0{entry["size_bytes"]}'
            f'\0{entry["sha256"]}\n'.encode("utf-8")
        )
    return aggregate.hexdigest()


def link_view(descriptors: list[dict], target: Path) -> tuple[int, int, str]:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    for entry in descriptors:
        destination = target / entry["name"]
        destination.symlink_to(Path(entry["target"]))
    return (
        len(descriptors),
        sum(int(entry["size_bytes"]) for entry in descriptors),
        aggregate_links(descriptors),
    )


def consolidate(freeze_path: Path, output: Path) -> dict:
    freeze, train_rows, expected = frozen_training_rows(freeze_path)
    verified = [
        (row, *verify_plot_inventory(row, expected))
        for row in train_rows
    ]
    summaries = [summary for _, summary, _ in verified]
    descriptors = link_descriptors([(row, files) for row, _, files in verified])
    tuning_root = Path(freeze["tuning_data_root"]).expanduser().resolve()
    tuning_count, tuning_bytes, tuning_aggregate = link_view(descriptors, tuning_root)
    if tuning_count != 16 * expected:
        raise ValueError("Consolidated crop counts differ from the frozen contract")
    payload = {
        "schema_version": 2,
        "status": "development_crops_consolidated",
        "held_out_test_accessed": False,
        "crops_per_plot": expected,
        "tuning": {
            "plots": 16,
            "crop_count": tuning_count,
            "referenced_size_bytes": tuning_bytes,
            "root": str(tuning_root),
            "entries_aggregate_sha256": tuning_aggregate,
            "aggregate_fields": "destination_name,target_path,size_bytes,sha256",
            "symlink_targets_verified": True,
        },
        "plot_crop_inventories": summaries,
        "plot_crop_inventory_count": len(summaries),
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verify_consolidated(freeze_path, output)
    return payload


def verify_consolidated(freeze_path: Path, inventory_path: Path) -> dict:
    freeze, train_rows, expected = frozen_training_rows(freeze_path)
    inventory_path = inventory_path.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text())
    verified = [
        (row, *verify_plot_inventory(row, expected))
        for row in train_rows
    ]
    summaries = [summary for _, summary, _ in verified]
    descriptors = link_descriptors([(row, files) for row, _, files in verified])
    tuning = inventory.get("tuning", {})
    tuning_root = Path(freeze["tuning_data_root"]).expanduser().resolve()
    expected_count = 16 * expected
    expected_bytes = sum(int(entry["size_bytes"]) for entry in descriptors)
    expected_aggregate = aggregate_links(descriptors)
    schema_version = inventory.get("schema_version")
    if (
        schema_version not in (1, 2)
        or inventory.get("status") != "development_crops_consolidated"
        or inventory.get("held_out_test_accessed") is not False
        or int(inventory.get("crops_per_plot", -1)) != expected
        or int(inventory.get("plot_crop_inventory_count", -1)) != 16
        or inventory.get("plot_crop_inventories") != summaries
        or int(tuning.get("plots", -1)) != 16
        or int(tuning.get("crop_count", -1)) != expected_count
        or int(tuning.get("referenced_size_bytes", -1)) != expected_bytes
        or Path(str(tuning.get("root", ""))).expanduser().resolve() != tuning_root
    ):
        raise ValueError(f"Consolidated crop inventory changed: {inventory_path}")
    if schema_version == 2 and (
        tuning.get("entries_aggregate_sha256") != expected_aggregate
        or tuning.get("aggregate_fields")
        != "destination_name,target_path,size_bytes,sha256"
        or tuning.get("symlink_targets_verified") is not True
    ):
        raise ValueError(f"Consolidated crop inventory changed: {inventory_path}")
    if not tuning_root.is_dir() or tuning_root.is_symlink():
        raise ValueError(f"Consolidated crop root changed: {tuning_root}")
    destinations = sorted(tuning_root.iterdir(), key=lambda path: path.name)
    if [path.name for path in destinations] != [entry["name"] for entry in descriptors]:
        raise ValueError(f"Consolidated crop names changed: {tuning_root}")
    for destination, entry in zip(destinations, descriptors):
        expected_target = Path(entry["target"])
        if (
            not destination.is_symlink()
            or destination.readlink() != expected_target
            or destination.resolve(strict=True) != expected_target.resolve(strict=True)
            or destination.stat().st_size != int(entry["size_bytes"])
        ):
            raise ValueError(f"Consolidated crop symlink target changed: {destination}")
    return {
        "inventory": str(inventory_path),
        "inventory_sha256": sha256(inventory_path),
        "inventory_schema_version": schema_version,
        "entries_aggregate_sha256": expected_aggregate,
        "crop_count": expected_count,
        "referenced_size_bytes": expected_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify-inventory", type=Path)
    args = parser.parse_args()
    freeze = args.freeze.expanduser().resolve()
    if args.output:
        result = consolidate(freeze, args.output.expanduser().resolve())
        print(f'tuning_crops={result["tuning"]["crop_count"]}')
    else:
        result = verify_consolidated(freeze, args.verify_inventory.expanduser().resolve())
        print(f'tuning_crops={result["crop_count"]}')
        print(f'crop_inventory_sha256={result["inventory_sha256"]}')
        print(f'crop_entries_aggregate_sha256={result["entries_aggregate_sha256"]}')
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
