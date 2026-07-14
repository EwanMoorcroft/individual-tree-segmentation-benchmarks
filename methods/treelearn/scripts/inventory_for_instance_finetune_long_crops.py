"""Hash and inventory one frozen plot's generated TreeLearn crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def managed_path(value: object, managed_root: Path, label: str) -> Path:
    """Resolve one generated path without allowing escapes or symlink traversal."""

    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {raw}")
    lexical_root = Path(os.path.abspath(managed_root))
    lexical_path = Path(os.path.abspath(raw))
    if not lexical_path.is_relative_to(lexical_root):
        raise ValueError(f"{label} escapes the long-run root: {lexical_path}")

    current = lexical_root
    for part in lexical_path.relative_to(lexical_root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlinked path component: {current}")

    resolved_root = lexical_root.resolve()
    resolved_path = lexical_path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{label} resolves outside the long-run root: {resolved_path}")
    return resolved_path


def inventory(freeze_path: Path, task_index: int) -> dict:
    freeze_path = freeze_path.expanduser().resolve()
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    matches = [
        row for row in freeze["plots"]
        if int(row["task_index"]) == task_index and row.get("split") == "dev"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one development plot for task {task_index}")
    row = matches[0]
    managed_root = freeze_path.parent
    output = managed_path(row["crop_inventory"], managed_root, "Crop inventory")
    if output.exists():
        raise FileExistsError(output)
    crop_root = managed_path(row["crop_root"], managed_root, "Crop root")
    if not crop_root.is_dir():
        raise FileNotFoundError(crop_root)
    files = sorted(crop_root.glob("*.npz"))
    unsafe_files = [
        path
        for path in files
        if path.is_symlink()
        or not path.is_file()
        or not path.resolve(strict=True).is_relative_to(crop_root)
    ]
    if unsafe_files:
        raise ValueError(f"Crop root contains unsafe files: {unsafe_files}")
    expected = int(row["crops_expected"])
    requested = int(row["crops_generate_requested"])
    generated_count = len(files)
    if requested < expected or generated_count < expected:
        raise ValueError(
            f"Expected at least {expected} crops from {requested} attempts, "
            f"found {generated_count}"
        )
    pruned = files[expected:]
    for path in pruned:
        path.unlink()
    files = files[:expected]
    entries = []
    total_bytes = 0
    aggregate = hashlib.sha256()
    for path in files:
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"Generated crop is empty: {path}")
        digest = sha256(path)
        entry = {"name": path.name, "size_bytes": size, "sha256": digest}
        entries.append(entry)
        total_bytes += size
        aggregate.update(
            f'{entry["name"]}\0{size}\0{digest}\n'.encode("utf-8")
        )
    payload = {
        "schema_version": 1,
        "status": "treelearn_plot_crops_sha256_inventoried",
        "held_out_test_accessed": False,
        "safe_plot_id": row["safe_plot_id"],
        "task_index": task_index,
        "crop_seed": int(row["crop_seed"]),
        "crop_generation_attempts_requested": requested,
        "crop_generation_outputs_created": generated_count,
        "crop_pruning_rule": "retain_lexicographically_first_expected_npz",
        "pruned_crop_count": len(pruned),
        "pruned_crop_names": [path.name for path in pruned],
        "crop_count": len(entries),
        "total_size_bytes": total_bytes,
        "crop_root": str(crop_root),
        "normalised_las_sha256": sha256(Path(row["normalised_las"])),
        "crop_config_sha256": sha256(Path(row["crop_config"])),
        "entries_aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    args = parser.parse_args()
    result = inventory(args.freeze.resolve(), args.task_index)
    print(f'crop_count={result["crop_count"]}')
    print(f'crop_bytes={result["total_size_bytes"]}')
    print(f'crop_inventory_sha256={result["entries_aggregate_sha256"]}')
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
