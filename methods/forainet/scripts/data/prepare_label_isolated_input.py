"""Prepare a row-stable, label-isolated PLY for official ForAINet inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData, PlyElement


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_alignment_sidecar import prepare, sha256  # noqa: E402


def inference_coordinates(cloud: laspy.LasData) -> np.ndarray:
    """Reproduce the official converter's scaled, offset-free coordinates."""

    scales = np.asarray(cloud.header.scales, dtype=np.float64)
    integers = np.column_stack((cloud.X, cloud.Y, cloud.Z)).astype(
        np.float64, copy=False
    )
    return (integers * scales).astype(np.float32)


def write_label_isolated_ply(path: Path, cloud: laspy.LasData) -> dict[str, object]:
    """Write all source rows with dummy fields required only by the loader."""

    point_count = len(cloud.points)
    coordinates = inference_coordinates(cloud)
    vertices = np.zeros(
        point_count,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("intensity", "f4"),
            ("semantic_seg", "f4"),
            ("treeID", "f4"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = coordinates.T
    vertices["intensity"] = np.asarray(cloud.intensity, dtype=np.float32)
    # The official loader subtracts one. A constant low-vegetation bookkeeping
    # label therefore becomes internal class 0 and keeps the upstream tracker
    # numerically defined without exposing any reference label.
    vertices["semantic_seg"] = 1.0
    vertices["treeID"] = -1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex", comments=["FOR-instance input"])],
        byte_order="<",
    ).write(path)
    return {
        "point_count": point_count,
        "coordinate_scale": [float(value) for value in cloud.header.scales],
        "coordinate_offset_ignored": [
            float(value) for value in cloud.header.offsets
        ],
        "coordinate_dtype": "float32",
        "source_order": "exact_original_las_row_order",
        "retained_source_rows": point_count,
        "dropped_source_rows": 0,
        "semantic_seg_dummy_value": 1,
        "semantic_after_official_loader": 0,
        "tree_id_dummy_value": -1,
        "tree_id_after_official_loader": 0,
        "reference_classification_supplied_to_model": False,
        "reference_tree_id_supplied_to_model": False,
        "reference_label_dependent_filtering": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-las", required=True, type=Path)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--split-metadata", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--alignment-sidecar-npz", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()

    outputs = (
        args.output_ply,
        args.alignment_sidecar_npz,
        args.metadata_json,
    )
    if existing := [str(path) for path in outputs if path.exists()]:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    sidecar_metadata, arrays = prepare(
        source=args.source_las,
        relative_path=args.relative_path,
        split_metadata=args.split_metadata,
    )
    cloud = laspy.read(args.source_las)
    conversion = write_label_isolated_ply(args.output_ply, cloud)
    args.alignment_sidecar_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.alignment_sidecar_npz, **arrays)
    payload = {
        "schema": "forainet_label_isolated_input_v1",
        "status": "verified",
        "relative_path": sidecar_metadata["relative_path"],
        "split": sidecar_metadata["split"],
        "source_sha256": sidecar_metadata["source_sha256"],
        "source_point_count": sidecar_metadata["point_count"],
        "reference_tree_count": sidecar_metadata["reference_tree_count"],
        "positive_tree_id_point_count": sidecar_metadata[
            "positive_tree_id_point_count"
        ],
        "classification_values": sidecar_metadata["classification_values"],
        "conversion": conversion,
        "input_ply_sha256": sha256(args.output_ply),
        "alignment_sidecar_sha256": sha256(args.alignment_sidecar_npz),
        "official_preparation_compatibility": {
            "coordinate_rule": "scaled_integer_coordinates_without_las_offset",
            "fields": [
                "x",
                "y",
                "z",
                "intensity",
                "semantic_seg",
                "treeID",
            ],
            "semantic_and_instance_fields": "dummy_loader_bookkeeping_only",
            "class_3_removal": False,
            "class_3_removal_reason": (
                "forbidden_reference_label_dependent_inference_filter"
            ),
        },
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
