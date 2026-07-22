"""Verify pinned external ForAINet source and checkpoint identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
EXPECTED_CHECKPOINT_SHA256 = (
    "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
)
EXPECTED_CHECKPOINT_SIZE = 665805463


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify(upstream: Path, checkpoint: Path) -> dict[str, object]:
    if not upstream.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("upstream checkout or checkpoint is missing")
    commit = git(upstream, "rev-parse", "HEAD")
    dirty = bool(git(upstream, "status", "--porcelain"))
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"unexpected upstream commit: {commit}")
    if dirty:
        raise ValueError("upstream checkout must be clean")
    size = checkpoint.stat().st_size
    digest = sha256(checkpoint)
    if size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError(f"unexpected checkpoint byte size: {size}")
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA-256: {digest}")
    return {
        "schema": "forainet_asset_verification_v1",
        "status": "verified",
        "upstream": {
            "repository": "https://github.com/prs-eth/ForAINet",
            "commit": commit,
            "dirty": dirty,
        },
        "checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": size,
            "sha256": digest,
            "provider_checksum": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    payload = verify(args.upstream_root, args.checkpoint)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
