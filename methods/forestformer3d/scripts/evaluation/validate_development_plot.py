"""Validate and normalise one official ForestFormer3D development output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.forestformer3d.scripts.runtime.run_official_test import sha256_file

EXPECTED_FIELDS = (
    "x", "y", "z", "semantic_pred", "instance_pred", "score",
    "semantic_gt", "instance_gt",
)


def validate(
    raw_ply: Path,
    sidecar_path: Path,
    input_manifest_path: Path,
    fingerprint_path: Path,
    audit_path: Path,
    output_root: Path,
) -> dict[str, object]:
    from plyfile import PlyData

    if output_root.exists():
        raise FileExistsError(f"Refusing existing validation root: {output_root}")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (
        input_manifest.get("split") != "development"
        or input_manifest.get("held_out_access") is not False
    ):
        raise ValueError("Input manifest is not development-only")
    vertices = PlyData.read(raw_ply)["vertex"].data
    if tuple(vertices.dtype.names or ()) != EXPECTED_FIELDS:
        raise ValueError("Unexpected official PLY fields")
    with np.load(sidecar_path) as sidecar:
        arrays = {name: np.asarray(sidecar[name]) for name in sidecar.files}
    point_count = len(arrays["source_row_index"])
    if len(vertices) != point_count or point_count != input_manifest["point_count"]:
        raise ValueError("Point count changed during inference")
    if not np.array_equal(arrays["source_row_index"], np.arange(point_count)):
        raise ValueError("source_row_index is not the identity")
    raw_xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    if not np.array_equal(raw_xyz, arrays["model_xyz"]):
        raise ValueError("Official output rows differ from staged XYZ")
    if not np.array_equal(vertices["semantic_gt"], arrays["reference_semantic"]):
        raise ValueError("Official semantic labels differ from staging")
    if not np.array_equal(vertices["instance_gt"], arrays["reference_instance"]):
        raise ValueError("Official instance labels differ from staging")
    if not np.isfinite(vertices["score"]).all():
        raise ValueError("Official prediction scores contain non-finite values")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if fingerprint.get("point_count") != point_count:
        raise ValueError("Model-facing point fingerprint count differs")
    if audit.get("status") != "passed" or audit.get("prediction_uses_ground_truth") is not False:
        raise ValueError("Effective predict audit did not pass")
    raw_instance = np.asarray(vertices["instance_pred"], dtype=np.int64)
    pred_tree_id = np.where(raw_instance >= 0, raw_instance + 1, -1)
    output_root.mkdir(parents=True)
    prediction_path = output_root / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        pred_tree_id=pred_tree_id.astype(np.int64),
        target_tree_id=arrays["target_tree_id"].astype(np.int64),
        classification=arrays["classification"].astype(np.int16),
        pred_classification=np.where(pred_tree_id > 0, 4, 0).astype(np.int16),
        source_row_index=arrays["source_row_index"].astype(np.int64),
    )
    result: dict[str, object] = {
        "schema": "forestformer3d_development_plot_validation_v1",
        "status": "passed",
        "split": "development",
        "held_out_access": False,
        "plot_id": input_manifest["plot_id"],
        "relative_path": input_manifest["relative_path"],
        "point_count": point_count,
        "exact_row_alignment": True,
        "raw_ply_sha256": sha256_file(raw_ply),
        "prediction_npz_sha256": sha256_file(prediction_path),
        "model_input_fingerprint": fingerprint,
        "effective_predict_audit_sha256": sha256_file(audit_path),
    }
    (output_root / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "validation.complete").touch(exist_ok=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-ply", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = validate(
        args.raw_ply, args.sidecar, args.input_manifest, args.fingerprint,
        args.audit, args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
