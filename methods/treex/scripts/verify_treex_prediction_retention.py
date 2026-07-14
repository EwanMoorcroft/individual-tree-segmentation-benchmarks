"""Verify the frozen TreeX prediction pairs retained outside Git."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(source_summary: Path, prediction_root: Path) -> dict:
    with source_summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32 or len({row["plot_id"] for row in rows}) != 32:
        raise ValueError("TreeX retention requires the frozen 32-plot inventory")
    files = []
    for row in rows:
        split_dir = "for_instance_test" if row["split"] == "test" else "for_instance"
        plot_dir = prediction_root / split_dir / row["plot_id"].replace("/", "_")
        for suffix in (".npz", ".las"):
            matches = sorted(plot_dir.glob(f"*_treex_predictions{suffix}"))
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one retained {suffix} for {row['plot_id']}; found {len(matches)}"
                )
            path = matches[0]
            files.append(
                {
                    "plot_id": row["plot_id"],
                    "split": row["split"],
                    "format": suffix.removeprefix("."),
                    "relative_path": path.relative_to(prediction_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "status": "retention_verified",
        "method": "TreeX",
        "expected_plots": 32,
        "verified_prediction_files": len(files),
        "verified_prediction_size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-summary",
        default="methods/treex/examples/treex_run_metadata.csv",
        type=Path,
    )
    parser.add_argument(
        "--prediction-root",
        default="local_outputs/treex_predictions",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = verify(args.source_summary.resolve(), args.prediction_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f'status={payload["status"]}')
    print(f'verified_prediction_files={payload["verified_prediction_files"]}')
    print(f'verified_prediction_size_bytes={payload["verified_prediction_size_bytes"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
