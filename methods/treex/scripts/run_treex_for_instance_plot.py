"""Run TreeX on one FOR-instance LAS plot."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]


def load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TreeX on one FOR-instance LAS plot."
    )
    parser.add_argument(
        "--config",
        default="methods/treex/configs/for_instance_benchmark.yml",
    )
    parser.add_argument("--input-las", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--performance-csv", required=True)
    parser.add_argument("--plot-id", required=True)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def call_treex(
    algorithm: Any,
    xyz: np.ndarray,
    intensity: np.ndarray,
    plot_id: str,
) -> Any:
    signature = inspect.signature(algorithm.__call__)
    kwargs: dict[str, Any] = {}
    if "point_cloud_id" in signature.parameters:
        kwargs["point_cloud_id"] = plot_id.replace("/", "_")
    if "intensity" in signature.parameters:
        kwargs["intensity"] = intensity
        return algorithm(xyz, **kwargs)
    if "intensities" in signature.parameters:
        kwargs["intensities"] = intensity
        return algorithm(xyz, **kwargs)
    return algorithm(xyz, intensity, **kwargs)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    input_path = Path(args.input_las).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    performance_path = Path(args.performance_csv).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input LAS does not exist: {input_path}")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pointtree.instance_segmentation import (
            TreeXAlgorithm,
            TreeXPresetULS,
        )
    except ImportError as exc:
        raise RuntimeError(
            "pointtree is required in the active environment."
        ) from exc

    started = time.perf_counter()
    cloud = laspy.read(input_path)
    required_fields = {"intensity", "classification", "treeID"}
    available_fields = set(cloud.point_format.dimension_names)
    missing_fields = required_fields - available_fields
    if missing_fields:
        raise ValueError(
            f"Input LAS is missing fields {sorted(missing_fields)}: {input_path}"
        )

    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float64)
    intensity = np.asarray(cloud.intensity, dtype=np.float64)
    classification = np.asarray(cloud.classification, dtype=np.int64)
    target_tree_id = np.asarray(cloud["treeID"], dtype=np.int64)

    tree_classes = set(config["dataset"]["tree_classes"])
    ignored_tree_ids = set(config["dataset"]["ignored_tree_ids"])
    tree_mask = np.isin(classification, sorted(tree_classes))
    reference_mask = tree_mask & ~np.isin(target_tree_id, sorted(ignored_tree_ids))
    reference_tree_count = len(np.unique(target_tree_id[reference_mask]))

    params = dict(TreeXPresetULS())
    params.update(config["method"]["params"])
    if args.workers is not None:
        params["num_workers"] = args.workers
    invalid_tree_id = int(params["invalid_tree_id"])

    algorithm = TreeXAlgorithm(**params)
    result = call_treex(algorithm, xyz, intensity, args.plot_id)
    predicted = np.asarray(
        result[0] if isinstance(result, tuple) else result,
        dtype=np.int64,
    )
    if len(predicted) != len(xyz):
        raise ValueError(
            f"Prediction length {len(predicted)} != point count {len(xyz)}"
        )

    elapsed_seconds = time.perf_counter() - started
    stem = input_path.stem
    prediction_npz = output_dir / f"{stem}_treex_predictions.npz"
    prediction_las = output_dir / f"{stem}_treex_predictions.las"

    np.savez_compressed(
        prediction_npz,
        pred_tree_id=predicted,
        target_tree_id=target_tree_id,
        classification=classification,
        tree_mask=tree_mask,
        intensity=intensity,
    )

    labelled_cloud = laspy.read(input_path)
    if "pred_treeID" not in labelled_cloud.point_format.dimension_names:
        labelled_cloud.add_extra_dim(
            laspy.ExtraBytesParams(name="pred_treeID", type=np.int32)
        )
    labelled_cloud["pred_treeID"] = predicted.astype(np.int32)
    labelled_cloud.write(prediction_las)

    algorithm.performance_metrics().to_csv(performance_path, index=False)

    valid_predictions = predicted[predicted != invalid_tree_id]
    summary = {
        "method": config["method"]["algorithm"],
        "preset": config["method"]["preset"],
        "profile": config["method"]["run_profile"],
        "plot_id": args.plot_id,
        "input_las": str(input_path),
        "prediction_npz": str(prediction_npz),
        "prediction_las": str(prediction_las),
        "total_points": int(len(xyz)),
        "tree_class_points": int(tree_mask.sum()),
        "reference_tree_count_tree_classes": int(reference_tree_count),
        "predicted_instances": int(len(np.unique(valid_predictions))),
        "elapsed_seconds": float(elapsed_seconds),
        "intensity_min": float(np.min(intensity)),
        "intensity_median": float(np.median(intensity)),
        "intensity_max": float(np.max(intensity)),
        "parameters": params,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
