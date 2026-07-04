"""Inventory SegmentAnyTree PLY outputs and identify evaluation candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import read_ply_header, read_ply_vertices


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def inspect_ply(path: Path, root: Path) -> dict[str, Any]:
    header = read_ply_header(path)
    fields = list(header.columns)
    if header.vertex_count == 0:
        return {
            "relative_path": str(path.relative_to(root)),
            "point_count": 0,
            "fields": fields,
            "candidate_role": "none",
        }
    record: dict[str, Any] = {
        "relative_path": str(path.relative_to(root)),
        "point_count": header.vertex_count,
        "fields": fields,
        "candidate_role": "none",
    }
    if {"preds", "gt"} <= set(fields):
        _, vertices = read_ply_vertices(path, columns=["preds", "gt"])
        predicted = np.asarray(vertices["preds"])
        reference = np.asarray(vertices["gt"])
        predicted_unique = np.unique(predicted)
        reference_unique = np.unique(reference)
        record.update(
            {
                "preds_unique_count": len(predicted_unique),
                "gt_unique_count": len(reference_unique),
                "preds_min": float(np.min(predicted_unique)),
                "preds_max": float(np.max(predicted_unique)),
                "gt_min": float(np.min(reference_unique)),
                "gt_max": float(np.max(reference_unique)),
            }
        )
        if len(reference_unique) > 8 or len(predicted_unique) > 8:
            record["candidate_role"] = "instance_evaluation"
        else:
            record["candidate_role"] = "semantic_evaluation"
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_outputs(root: Path) -> dict[str, Any]:
    files = [
        inspect_ply(path, root)
        for path in sorted(root.rglob("*.ply"))
    ]
    checkpoints = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("PointGroup-PAPER.pt"))
    ]
    return {
        "output_root": str(root),
        "ply_file_count": len(files),
        "checkpoint_files": checkpoints,
        "instance_candidates": [
            row["relative_path"]
            for row in files
            if row["candidate_role"] == "instance_evaluation"
        ],
        "semantic_candidates": [
            row["relative_path"]
            for row in files
            if row["candidate_role"] == "semantic_evaluation"
        ],
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect SegmentAnyTree PLY fields and evaluation candidates."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Prediction output directory does not exist: {output_root}"
        )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **inspect_outputs(output_root),
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PLY files: {payload['ply_file_count']}")
    print(f"Instance candidates: {payload['instance_candidates']}")
    print(f"Semantic candidates: {payload['semantic_candidates']}")
    print(f"Output: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
