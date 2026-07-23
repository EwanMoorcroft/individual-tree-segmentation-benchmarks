#!/usr/bin/env python3
"""Create a non-overwriting SHA-256 inventory for retained run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, relatives: Sequence[str]) -> list[dict[str, object]]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Artifact root does not exist: {resolved_root}")
    if not relatives:
        raise ValueError("At least one artifact path is required")
    if len(relatives) != len(set(relatives)):
        raise ValueError("Artifact paths contain duplicates")

    rows: list[dict[str, object]] = []
    for value in relatives:
        relative = Path(value)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(f"Unsafe artifact path: {value!r}")
        path = (resolved_root / relative).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"Artifact escapes root: {value!r}")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Artifact is not a regular retained file: {path}")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("relative_paths", nargs="+")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Inventory output already exists: {args.output}")
    rows = inventory(args.root, args.relative_paths)
    payload = {
        "schema": "forestformer3d_retention_manifest_v1",
        "root_identity": args.root.name,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
