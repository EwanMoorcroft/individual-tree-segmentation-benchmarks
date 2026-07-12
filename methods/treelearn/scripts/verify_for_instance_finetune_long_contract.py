"""Recheck immutable inputs for every downstream TreeLearn long-run stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
CLEAN_CHECKPOINT_MD5 = "106a80de2991c5f23484a3f9d03e3b16"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def canonical_metadata_path(value: object) -> str:
    """Apply the same path normalisation as the accepted manifest builder."""

    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if (
        not raw
        or raw.startswith(("/", "./"))
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
        or path.suffix.casefold() != ".las"
    ):
        raise ValueError(f"Unsafe FOR-instance metadata path: {value!r}")
    return raw


def verify_supplied_split(freeze: dict, development_manifest: Path) -> None:
    contract = freeze.get("supplied_split_contract", {})
    metadata_path = Path(contract.get("source", "")).expanduser().resolve()
    if not metadata_path.is_file():
        raise ValueError("Frozen FOR-instance supplied split metadata is missing")
    metadata_sha256 = digest(metadata_path, "sha256")
    if metadata_sha256 != contract.get("sha256"):
        raise ValueError("FOR-instance supplied split metadata changed")
    if (
        contract.get("development_rows") != 21
        or contract.get("held_out_test_rows") != 11
        or contract.get("held_out_test_files_opened") is not False
    ):
        raise ValueError("Frozen FOR-instance supplied split contract is invalid")

    source = json.loads(development_manifest.read_text())
    if (
        source.get("dataset_split") != "dev"
        or source.get("mapping_rule") != "exact_metadata_path_only"
        or source.get("split_metadata_sha256") != metadata_sha256
    ):
        raise ValueError("Development manifest no longer matches supplied split evidence")
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    supplied_dev = sorted(
        canonical_metadata_path(row.get("path"))
        for row in metadata_rows
        if str(row.get("split", "")).strip() == "dev"
    )
    supplied_test = [
        canonical_metadata_path(row.get("path")) for row in metadata_rows
        if str(row.get("split", "")).strip() == "test"
    ]
    frozen_dev = sorted(str(row["relative_path"]) for row in source.get("plots", []))
    if supplied_dev != frozen_dev or len(supplied_dev) != 21 or len(supplied_test) != 11:
        raise ValueError("Frozen development paths differ from supplied FOR-instance split")
    if any(
        row.get("split") != "dev"
        or row.get("split_metadata_sha256") != metadata_sha256
        for row in source.get("plots", [])
    ):
        raise ValueError("Frozen development rows differ from supplied split evidence")


def verify(freeze_path: Path, treelearn_repo: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    if git(treelearn_repo, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("Pinned TreeLearn commit changed")
    if git(treelearn_repo, "status", "--porcelain"):
        raise ValueError("Pinned TreeLearn checkout is dirty")
    checkpoint = Path(freeze["initial_checkpoint"]).expanduser().resolve()
    if digest(checkpoint, "md5") != CLEAN_CHECKPOINT_MD5:
        raise ValueError("Clean initial checkpoint MD5 changed")
    if digest(checkpoint, "sha256") != freeze.get("initial_checkpoint_sha256"):
        raise ValueError("Clean initial checkpoint SHA-256 changed")
    development_manifest = Path(freeze["development_manifest"]).resolve()
    if digest(development_manifest, "sha256") != freeze.get("development_manifest_sha256"):
        raise ValueError("Frozen development manifest changed")
    verify_supplied_split(freeze, development_manifest)
    evaluation_config = Path(freeze["evaluation_config"]).resolve()
    if digest(evaluation_config, "sha256") != freeze.get("evaluation_config_sha256"):
        raise ValueError("Frozen evaluation configuration changed")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    args = parser.parse_args()
    verify(args.freeze.resolve(), args.treelearn_repo.expanduser().resolve())
    print("status=treelearn-long-contract-verified")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
