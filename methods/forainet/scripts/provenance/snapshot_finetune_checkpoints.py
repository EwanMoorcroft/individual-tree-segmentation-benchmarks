"""Retain predeclared immutable checkpoints from the official rolling archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch


EXPECTED_EPOCHS = (30, 60, 90, 120, 149)
EXPECTED_TENSOR_COUNT = 755


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(path: Path) -> tuple[int, int]:
    archive = torch.load(path, map_location="cpu")
    if not isinstance(archive, dict):
        raise ValueError("rolling checkpoint is not a dictionary")
    stats = archive.get("stats")
    models = archive.get("models")
    if (
        not isinstance(stats, dict)
        or not isinstance(stats.get("train"), list)
        or not isinstance(stats.get("val"), list)
        or not stats["train"]
        or not stats["val"]
    ):
        raise ValueError("rolling checkpoint lacks train/val statistics")
    if not isinstance(models, dict) or not isinstance(models.get("latest"), dict):
        raise ValueError("rolling checkpoint lacks latest weights")
    if len(models["latest"]) != EXPECTED_TENSOR_COUNT:
        raise ValueError("rolling checkpoint tensor count changed")
    return int(stats["train"][-1]["epoch"]), int(stats["val"][-1]["epoch"])


def write_index(
    output_dir: Path,
    records: list[dict[str, Any]],
    status: str,
) -> None:
    payload = {
        "schema": "forainet_finetune_candidate_index_v1",
        "status": status,
        "expected_epochs": list(EXPECTED_EPOCHS),
        "candidate_count": len(records),
        "held_out_access": False,
        "candidates": records,
    }
    target = output_dir / "index.json"
    temporary = output_dir / ".index.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def snapshot(
    checkpoint: Path,
    output_dir: Path,
    completion_marker: Path,
    poll_seconds: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing candidate root: {output_dir}")
    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    retained: set[int] = set()
    write_index(output_dir, records, "monitoring")
    deadline = time.monotonic() + timeout_seconds
    processed_signature: tuple[int, int] | None = None
    observed_validation_epoch: int | None = None

    while time.monotonic() < deadline:
        if checkpoint.is_file():
            stat = checkpoint.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != processed_signature:
                try:
                    _, observed_validation_epoch = checkpoint_epoch(checkpoint)
                except Exception:
                    pass
                else:
                    processed_signature = signature
        if (
            observed_validation_epoch in EXPECTED_EPOCHS
            and observed_validation_epoch not in retained
        ):
            epoch = int(observed_validation_epoch)
            temporary = output_dir / f".PointGroup-PAPER_epoch_{epoch:03d}.pt.tmp"
            target = output_dir / f"PointGroup-PAPER_epoch_{epoch:03d}.pt"
            temporary.unlink(missing_ok=True)
            try:
                shutil.copyfile(checkpoint, temporary)
                train_epoch, validation_epoch = checkpoint_epoch(temporary)
                if (train_epoch, validation_epoch) != (epoch, epoch):
                    raise ValueError("rolling archive changed during candidate copy")
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
            else:
                record = {
                    "epoch": epoch,
                    "filename": target.name,
                    "sha256": sha256(target),
                    "size_bytes": target.stat().st_size,
                    "train_epoch": train_epoch,
                    "validation_epoch": validation_epoch,
                    "weight_name": "latest",
                }
                records.append(record)
                records.sort(key=lambda row: int(row["epoch"]))
                retained.add(epoch)
                write_index(output_dir, records, "monitoring")

        missing = sorted(set(EXPECTED_EPOCHS) - retained)
        if (
            observed_validation_epoch is not None
            and missing
            and observed_validation_epoch > missing[0]
        ):
            raise RuntimeError(f"candidate epochs were missed: {missing}")

        if completion_marker.is_file():
            completion = completion_marker.read_text(encoding="utf-8").strip()
            if completion != "complete":
                raise RuntimeError(
                    f"training completion marker reports {completion!r}"
                )
            if missing:
                raise RuntimeError(f"training ended without candidates: {missing}")
            write_index(output_dir, records, "complete")
            return records
        time.sleep(poll_seconds)
    raise TimeoutError("checkpoint snapshot monitor timed out")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--completion-marker", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=260000)
    args = parser.parse_args()
    if not 2 <= args.poll_seconds <= 60:
        raise ValueError("poll interval must be between 2 and 60 seconds")
    records = snapshot(
        args.checkpoint,
        args.output_dir,
        args.completion_marker,
        args.poll_seconds,
        args.timeout_seconds,
    )
    print(f"retained_candidates={len(records)}")
    print("held_out_access=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
