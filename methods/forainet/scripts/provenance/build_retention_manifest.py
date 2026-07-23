"""Build and verify a hash-complete ForAINet run retention manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_SMOKE_ROLES = {
    "official_raw_output",
    "aligned_prediction",
    "checkpoint_provenance",
    "environment_manifest",
    "plot_metadata",
    "plot_metrics",
    "matched_pairs",
    "unmatched_predictions",
    "unmatched_references",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(root: Path, role_paths: dict[str, Path]) -> dict[str, object]:
    resolved_root = root.resolve()
    if set(role_paths) != REQUIRED_SMOKE_ROLES:
        missing = sorted(REQUIRED_SMOKE_ROLES - set(role_paths))
        extra = sorted(set(role_paths) - REQUIRED_SMOKE_ROLES)
        raise ValueError(f"retention roles differ; missing={missing}, extra={extra}")
    files = []
    resolved_paths = set()
    for role in sorted(role_paths):
        path = role_paths[role]
        resolved = path.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            raise ValueError(f"retained path for {role} is outside the run root")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved in resolved_paths:
            raise ValueError("one retained file cannot satisfy multiple roles")
        resolved_paths.add(resolved)
        files.append(
            {
                "role": role,
                "relative_path": resolved.relative_to(resolved_root).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )
    return {
        "schema": "forainet_retention_manifest_v1",
        "status": "complete",
        "root_policy": "all paths relative to immutable run root",
        "files": files,
    }


def validate(root: Path, manifest: dict[str, object]) -> None:
    if manifest.get("schema") != "forainet_retention_manifest_v1":
        raise ValueError("unexpected retention schema")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("retention files must be a list")
    roles = {str(row.get("role")) for row in files if isinstance(row, dict)}
    if roles != REQUIRED_SMOKE_ROLES or len(files) != len(REQUIRED_SMOKE_ROLES):
        raise ValueError("retention manifest roles are incomplete or duplicated")
    for row in files:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("retention paths must be safe and relative")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"retained file changed: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--file", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    role_paths: dict[str, Path] = {}
    for item in args.file:
        role, separator, value = item.partition("=")
        if not separator or not role or not value or role in role_paths:
            raise ValueError(f"invalid or duplicate --file value: {item!r}")
        role_paths[role] = Path(value)
    manifest = build(args.run_root, role_paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
