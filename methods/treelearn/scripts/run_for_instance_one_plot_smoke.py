"""Run and adapt the one-plot TreeLearn FOR-instance smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path_text: str, base: Path = ROOT) -> Path:
    expanded = os.path.expandvars(path_text)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = resolve_path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_staged_input(source_las: Path, staged_las: Path, overwrite: bool) -> None:
    staged_las.parent.mkdir(parents=True, exist_ok=True)
    if staged_las.exists() or staged_las.is_symlink():
        if not overwrite:
            return
        staged_las.unlink()
    try:
        staged_las.symlink_to(source_las)
    except OSError:
        shutil.copy2(source_las, staged_las)


def treelearn_pipeline_config(
    forest_path: Path,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    upstream = config["method"]["upstream_pipeline"]
    return {
        "default_args": [
            "configs/_modular/sample_generation.yaml",
            "configs/_modular/model.yaml",
            "configs/_modular/grouping.yaml",
            "configs/_modular/dataset_test.yaml",
        ],
        "model": {
            "spatial_shape": [500, 500, 1000],
        },
        "forest_path": str(forest_path),
        "pretrain": str(checkpoint_path),
        "fp16": True,
        "tile_generation": True,
        "dataloader": {
            "batch_size": 1,
            "num_workers": 2,
        },
        "shape_cfg": {
            "outer_remove": None,
            "alpha": 0.6,
            "buffer_size_to_determine_edge_trees": 0.3,
        },
        "save_cfg": {
            "save_formats": upstream["save_formats"],
            "save_treewise": bool(upstream["save_treewise"]),
            "save_pointwise": bool(upstream["save_pointwise"]),
            "return_type": upstream["return_type"],
            "results_dir": "results",
        },
        "grouping": {
            "use_hdbscan": bool(upstream["use_hdbscan"]),
        },
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def run_pipeline(treelearn_repo: Path, pipeline_config: Path) -> None:
    entrypoint = treelearn_repo / "tools" / "pipeline" / "pipeline.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"TreeLearn pipeline entrypoint missing: {entrypoint}")
    subprocess.run(
        [sys.executable, str(entrypoint), "--config", str(pipeline_config)],
        cwd=treelearn_repo,
        check=True,
    )


def load_prediction_arrays(source_las: Path, raw_prediction_laz: Path) -> dict[str, Any]:
    import laspy
    import numpy as np

    source = laspy.read(source_las)
    prediction = laspy.read(raw_prediction_laz)

    source_count = len(source.points)
    prediction_count = len(prediction.points)
    if source_count != prediction_count:
        raise ValueError(
            f"Prediction point count {prediction_count} != source point count {source_count}"
        )

    source_dimensions = set(source.point_format.dimension_names)
    prediction_dimensions = set(prediction.point_format.dimension_names)
    for field in ("treeID", "classification"):
        if field not in source_dimensions:
            raise ValueError(f"Source LAS is missing {field!r}: {source_las}")
    if "treeID" not in prediction_dimensions:
        raise ValueError(f"Prediction LAS is missing 'treeID': {raw_prediction_laz}")

    source_xyz = np.column_stack((source.x, source.y, source.z))
    prediction_xyz = np.column_stack((prediction.x, prediction.y, prediction.z))
    max_abs_coordinate_delta = float(np.max(np.abs(source_xyz - prediction_xyz)))

    pred_tree_id = np.asarray(prediction["treeID"], dtype=np.int64)
    target_tree_id = np.asarray(source["treeID"], dtype=np.int64)
    classification = np.asarray(source.classification, dtype=np.int64)

    return {
        "source": source,
        "source_count": int(source_count),
        "prediction_count": int(prediction_count),
        "max_abs_coordinate_delta": max_abs_coordinate_delta,
        "pred_tree_id": pred_tree_id,
        "target_tree_id": target_tree_id,
        "classification": classification,
        "source_row_index": np.arange(source_count, dtype=np.int64),
    }


def write_adapted_outputs(
    arrays: dict[str, Any],
    adapted_npz: Path,
    adapted_las: Path,
) -> dict[str, Any]:
    import laspy
    import numpy as np

    pred_tree_id = arrays["pred_tree_id"]
    int32_info = np.iinfo(np.int32)
    pred_min = int(pred_tree_id.min()) if len(pred_tree_id) else 0
    pred_max = int(pred_tree_id.max()) if len(pred_tree_id) else 0
    if pred_min < int32_info.min or pred_max > int32_info.max:
        raise OverflowError(
            f"Predicted tree IDs exceed int32 range: min={pred_min}, max={pred_max}"
        )

    ensure_parent(adapted_npz)
    np.savez_compressed(
        adapted_npz,
        pred_tree_id=pred_tree_id,
        target_tree_id=arrays["target_tree_id"],
        classification=arrays["classification"],
        source_row_index=arrays["source_row_index"],
    )

    adapted = arrays["source"]
    if "pred_treeID" not in adapted.point_format.dimension_names:
        adapted.add_extra_dim(laspy.ExtraBytesParams(name="pred_treeID", type=np.int32))
    adapted["pred_treeID"] = pred_tree_id.astype(np.int32)
    ensure_parent(adapted_las)
    adapted.write(adapted_las)

    positive_prediction_count = int(np.sum(pred_tree_id > 0))
    predicted_tree_count = int(len(np.unique(pred_tree_id[pred_tree_id > 0])))
    return {
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "positive_prediction_count": positive_prediction_count,
        "predicted_tree_count": predicted_tree_count,
    }


def build_metadata(
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
    started: float,
    validation: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = paths["checkpoint"]
    return {
        "method": config["method"]["slug"],
        "dataset": config["dataset"]["slug"],
        "training_mode": config["method"]["checkpoint"]["training_mode"],
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "plot": {
            "relative_path": config["smoke"]["relative_path"],
            "split": config["smoke"]["split"],
            "expected_point_count": config["smoke"]["expected_point_count"],
            "expected_reference_tree_count": config["smoke"][
                "expected_reference_tree_count"
            ],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "environment": {
            "treelearn_repo": str(paths["treelearn_repo"]),
            "python": sys.executable,
        },
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key
            in {
                "pipeline_config",
                "raw_prediction_laz",
                "raw_pointwise_npz",
                "adapted_npz",
                "adapted_las",
                "metadata_json",
            }
        },
        "validation": validation,
        "success_criteria": config["success_criteria"],
        "failure_indicators": config["failure_indicators"],
        "dry_run": bool(args.dry_run),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TreeLearn one-plot FOR-instance smoke test."
    )
    parser.add_argument(
        "--config",
        default="methods/treelearn/configs/for_instance_one_plot_smoke.yml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--treelearn-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--runtime-root")
    parser.add_argument("--predictions-root")
    parser.add_argument("--metadata-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config, _ = load_config(args.config)

    if config["smoke"]["split"] != "dev":
        raise ValueError("The one-plot smoke route must use a development plot.")

    dataset_root = resolve_path(args.dataset_root or config["dataset"]["root"])
    treelearn_repo = resolve_path(args.treelearn_repo or config["method"]["repo_path"])
    checkpoint = resolve_path(
        args.checkpoint or config["method"]["checkpoint"]["default_path"]
    )
    runtime_root = resolve_path(args.runtime_root or config["paths"]["runtime_root"])
    predictions_root = resolve_path(
        args.predictions_root or config["paths"]["predictions_root"]
    )
    metadata_root = resolve_path(args.metadata_root or config["paths"]["metadata_root"])

    safe_plot_id = config["smoke"]["safe_plot_id"]
    source_las = (dataset_root / config["smoke"]["relative_path"]).resolve()
    staged_las = runtime_root / "forest" / f"{safe_plot_id}.las"
    pipeline_config = runtime_root / "treelearn_pipeline_config.yml"
    raw_prediction_laz = runtime_root / "results" / "full_forest" / f"{safe_plot_id}.laz"
    raw_pointwise_npz = (
        runtime_root / "results" / "pointwise_results" / "pointwise_results.npz"
    )
    adapted_npz = predictions_root / f"{safe_plot_id}_treelearn_smoke_predictions.npz"
    adapted_las = predictions_root / f"{safe_plot_id}_treelearn_smoke_predictions.las"
    metadata_json = metadata_root / f"{safe_plot_id}_treelearn_smoke_metadata.json"

    paths = {
        "dataset_root": dataset_root,
        "treelearn_repo": treelearn_repo,
        "checkpoint": checkpoint,
        "source_las": source_las,
        "runtime_root": runtime_root,
        "staged_las": staged_las,
        "pipeline_config": pipeline_config,
        "raw_prediction_laz": raw_prediction_laz,
        "raw_pointwise_npz": raw_pointwise_npz,
        "adapted_npz": adapted_npz,
        "adapted_las": adapted_las,
        "metadata_json": metadata_json,
    }

    for required in (source_las, treelearn_repo, checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"Required path does not exist: {required}")

    prepare_staged_input(source_las, staged_las, args.overwrite)
    write_yaml(pipeline_config, treelearn_pipeline_config(staged_las, checkpoint, config))

    if args.dry_run:
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0

    if not args.skip_pipeline:
        run_pipeline(treelearn_repo, pipeline_config)

    if not raw_prediction_laz.is_file():
        raise FileNotFoundError(f"TreeLearn full-forest output missing: {raw_prediction_laz}")
    if not raw_pointwise_npz.is_file():
        raise FileNotFoundError(f"TreeLearn pointwise output missing: {raw_pointwise_npz}")

    arrays = load_prediction_arrays(source_las, raw_prediction_laz)
    row_tolerance = float(config["smoke"]["row_coordinate_tolerance_m"])
    row_order_preserved = arrays["max_abs_coordinate_delta"] <= row_tolerance
    if not row_order_preserved:
        raise ValueError(
            "Prediction coordinates are not row-aligned with source LAS: "
            f"max delta {arrays['max_abs_coordinate_delta']:.6f} m > {row_tolerance:.6f} m"
        )

    prediction_summary = write_adapted_outputs(arrays, adapted_npz, adapted_las)
    if prediction_summary["positive_prediction_count"] <= 0:
        raise ValueError("Prediction contains no positive TreeLearn instance labels.")

    validation = {
        "source_point_count": arrays["source_count"],
        "prediction_point_count": arrays["prediction_count"],
        "row_count_match": arrays["source_count"] == arrays["prediction_count"],
        "max_abs_coordinate_delta_m": arrays["max_abs_coordinate_delta"],
        "row_coordinate_tolerance_m": row_tolerance,
        "row_order_preserved": row_order_preserved,
        **prediction_summary,
    }
    metadata = build_metadata(config, paths, args, started, validation)
    ensure_parent(metadata_json)
    metadata_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
