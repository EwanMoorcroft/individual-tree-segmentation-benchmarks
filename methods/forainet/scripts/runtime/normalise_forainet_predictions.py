"""Normalise official ForAINet post-merge output to source-row order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from forainet_contract import align_full_resolution_prediction  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(raw_prediction: Path, sidecar: Path) -> dict[str, np.ndarray]:
    with np.load(sidecar, allow_pickle=False) as source, np.load(
        raw_prediction, allow_pickle=False
    ) as raw:
        required_source = {
            "classification",
            "target_tree_id",
            "source_row_index",
        }
        required_raw = {
            "source_row_index",
            "pred_semantic_internal",
            "pred_instance_id",
        }
        if missing := required_source - set(source.files):
            raise ValueError(f"sidecar is missing fields: {sorted(missing)}")
        if missing := required_raw - set(raw.files):
            raise ValueError(f"raw prediction is missing fields: {sorted(missing)}")

        source_rows = np.asarray(source["source_row_index"])
        expected_rows = np.arange(len(source_rows), dtype=np.int64)
        if not np.array_equal(source_rows, expected_rows):
            raise ValueError("sidecar source_row_index is not exact source order")
        aligned = align_full_resolution_prediction(
            source_row_index=raw["source_row_index"],
            pred_semantic_internal=raw["pred_semantic_internal"],
            pred_instance_id=raw["pred_instance_id"],
            expected_point_count=len(source_rows),
        )
        return {
            "pred_tree_id": aligned.pred_tree_id,
            "target_tree_id": np.asarray(source["target_tree_id"], dtype=np.int64),
            "classification": np.asarray(source["classification"], dtype=np.int64),
            "pred_classification": aligned.pred_classification,
            "source_row_index": aligned.source_row_index,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-prediction-npz", required=True, type=Path)
    parser.add_argument("--alignment-sidecar-npz", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--split", choices=("dev",), required=True)
    args = parser.parse_args()

    for output in (args.output_npz, args.metadata_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    arrays = normalise(args.raw_prediction_npz, args.alignment_sidecar_npz)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    metadata = {
        "schema": "forainet_aligned_prediction_v1",
        "status": "completed",
        "run_id": args.run_id,
        "relative_path": args.relative_path,
        "split": args.split,
        "point_count": len(arrays["source_row_index"]),
        "point_correspondence": "exact_source_row_index",
        "coordinate_matching": False,
        "raw_prediction_sha256": sha256(args.raw_prediction_npz),
        "alignment_sidecar_sha256": sha256(args.alignment_sidecar_npz),
        "aligned_prediction_sha256": sha256(args.output_npz),
        "forainet_semantic_mapping": {
            "0": 0,
            "1": 0,
            "2": 4,
            "3": 5,
            "4": 6,
        },
    }
    args.metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
