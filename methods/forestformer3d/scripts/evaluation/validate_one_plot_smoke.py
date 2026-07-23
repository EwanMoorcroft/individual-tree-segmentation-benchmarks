"""Validate label independence and exact row alignment for the FF3D smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PREDICTION_FIELDS = ("semantic_pred", "instance_pred", "score")
EXPECTED_FIELDS = (
    "x",
    "y",
    "z",
    *PREDICTION_FIELDS,
    "semantic_gt",
    "instance_gt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_vertices(path: Path) -> np.ndarray:
    from plyfile import PlyData

    if not path.is_file():
        raise FileNotFoundError(path)
    vertices = PlyData.read(path)["vertex"].data
    if tuple(vertices.dtype.names or ()) != EXPECTED_FIELDS:
        raise ValueError(
            f"Unexpected PLY fields in {path}: {vertices.dtype.names}"
        )
    return vertices


def validate(
    reference_ply: Path,
    dummy_ply: Path,
    sidecar_path: Path,
    output_root: Path,
) -> dict[str, object]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing validation root: {output_root}")
    reference = _read_vertices(reference_ply)
    dummy = _read_vertices(dummy_ply)
    with np.load(sidecar_path) as sidecar:
        model_xyz = np.asarray(sidecar["model_xyz"], dtype=np.float32)
        source_row_index = np.asarray(sidecar["source_row_index"], dtype=np.int64)
        classification = np.asarray(sidecar["classification"], dtype=np.int16)
        target_tree_id = np.asarray(sidecar["target_tree_id"], dtype=np.int64)
        reference_semantic = np.asarray(
            sidecar["reference_semantic"], dtype=np.int64
        )
        reference_instance = np.asarray(
            sidecar["reference_instance"], dtype=np.int64
        )

    point_count = len(source_row_index)
    if not (
        len(reference) == len(dummy) == len(model_xyz) == point_count
    ):
        raise ValueError("Point count changed across source and raw outputs")
    if not np.array_equal(source_row_index, np.arange(point_count)):
        raise ValueError("source_row_index is not the exact identity map")

    reference_xyz = np.column_stack(
        (reference["x"], reference["y"], reference["z"])
    )
    dummy_xyz = np.column_stack((dummy["x"], dummy["y"], dummy["z"]))
    if not np.array_equal(reference_xyz, model_xyz):
        raise ValueError("Reference-case output rows do not match staged XYZ")
    if not np.array_equal(dummy_xyz, model_xyz):
        raise ValueError("Dummy-case output rows do not match staged XYZ")
    if not np.array_equal(reference["semantic_gt"], reference_semantic):
        raise ValueError("Reference semantic loader labels were not retained")
    if not np.array_equal(reference["instance_gt"], reference_instance):
        raise ValueError("Reference instance loader labels were not retained")
    if np.any(dummy["semantic_gt"] != 0) or np.any(dummy["instance_gt"] != 0):
        raise ValueError("Dummy loader labels are not all zero")

    for field in PREDICTION_FIELDS:
        if not np.array_equal(reference[field], dummy[field]):
            differing = int(np.count_nonzero(reference[field] != dummy[field]))
            raise ValueError(
                f"Label counterfactual changed {field} on {differing} rows"
            )
    if not np.isfinite(reference["score"]).all():
        raise ValueError("Prediction scores contain non-finite values")

    raw_instance = np.asarray(reference["instance_pred"], dtype=np.int64)
    pred_tree_id = np.where(raw_instance >= 0, raw_instance + 1, -1).astype(
        np.int64
    )
    pred_classification = np.where(pred_tree_id > 0, 4, 0).astype(np.int16)

    output_root.mkdir(parents=True)
    harmonised_path = output_root / "forestformer3d_smoke_predictions.npz"
    np.savez_compressed(
        harmonised_path,
        pred_tree_id=pred_tree_id,
        target_tree_id=target_tree_id,
        classification=classification,
        pred_classification=pred_classification,
        source_row_index=source_row_index,
    )
    result: dict[str, object] = {
        "schema": "forestformer3d_one_plot_smoke_validation_v1",
        "status": "passed",
        "split": "development",
        "relative_path": "CULS/plot_1_annotated.las",
        "held_out_access": False,
        "point_count": point_count,
        "exact_row_alignment": True,
        "label_counterfactual": {
            "passed": True,
            "compared_fields": list(PREDICTION_FIELDS),
            "different_loader_semantic_rows": int(
                np.count_nonzero(
                    reference["semantic_gt"] != dummy["semantic_gt"]
                )
            ),
            "different_loader_instance_rows": int(
                np.count_nonzero(
                    reference["instance_gt"] != dummy["instance_gt"]
                )
            ),
        },
        "raw_prediction_inventory": {
            "predicted_tree_points": int(np.count_nonzero(pred_tree_id > 0)),
            "predicted_instances": int(
                len(np.unique(pred_tree_id[pred_tree_id > 0]))
            ),
            "semantic_ids": sorted(
                int(value) for value in np.unique(reference["semantic_pred"])
            ),
        },
        "artifacts": {
            "reference_ply": {
                "path": str(reference_ply),
                "sha256": sha256_file(reference_ply),
            },
            "dummy_ply": {
                "path": str(dummy_ply),
                "sha256": sha256_file(dummy_ply),
            },
            "harmonised_npz": {
                "path": str(harmonised_path),
                "sha256": sha256_file(harmonised_path),
            },
        },
        "manual_alignment_review_required": True,
    }
    result_path = output_root / "smoke_validation.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "smoke_validation.complete").touch(exist_ok=False)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-ply", required=True, type=Path)
    parser.add_argument("--dummy-ply", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate(
        args.reference_ply, args.dummy_ply, args.sidecar, args.output_root
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
