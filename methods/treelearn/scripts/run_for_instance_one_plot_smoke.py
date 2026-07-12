"""Run and adapt the one-plot TreeLearn FOR-instance smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_file(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        entry.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    return entry


def validate_checkpoint_identity(
    checkpoint: Path,
    expected_md5: str,
    *,
    allow_derived: bool = False,
) -> dict[str, str]:
    """Require released weights unless the route explicitly uses derived weights."""

    actual_md5 = md5(checkpoint).lower()
    expected_md5 = expected_md5.lower()
    if actual_md5 != expected_md5 and not allow_derived:
        raise ValueError(
            "TreeLearn checkpoint MD5 "
            f"{actual_md5} does not match official {expected_md5}"
        )
    return {"md5": actual_md5, "sha256": sha256(checkpoint)}


def repository_state(repo: Path, expected_commit: str) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if commit != expected_commit:
        raise ValueError(
            f"TreeLearn commit {commit} does not match pinned {expected_commit}"
        )
    if dirty:
        raise ValueError("TreeLearn checkout contains uncommitted or untracked files")
    return {
        "commit": commit,
        "expected_commit": expected_commit,
        "dirty": dirty,
    }


def benchmark_repository_state(
    repo: Path, expected_commit: str | None = None
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(
            f"Benchmark commit {commit} does not match frozen {expected_commit}"
        )
    if dirty:
        raise ValueError("Benchmark checkout contains uncommitted or untracked files")
    return {"commit": commit, "branch": branch, "dirty": dirty}


def validate_dataset_source(
    config: dict[str, Any],
    dataset_root: Path,
    source_las: Path,
    plot_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import laspy
    import numpy as np

    relative_path = source_las.relative_to(dataset_root).as_posix()
    split_path = dataset_root / config["dataset"]["split_metadata_file"]
    if not split_path.is_file():
        raise FileNotFoundError(f"Split metadata does not exist: {split_path}")
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        by_lower = {name.strip().lower(): name for name in fieldnames}
        path_column = next(
            (
                by_lower[name]
                for name in (
                    "relative_path",
                    "path",
                    "file_path",
                    "filepath",
                    "las_path",
                    "filename",
                    "file",
                    "plot",
                    "plot_name",
                )
                if name in by_lower
            ),
            None,
        )
        split_column = next(
            (
                by_lower[name]
                for name in (
                    "split",
                    "data_split",
                    "dataset_split",
                    "partition",
                    "set",
                )
                if name in by_lower
            ),
            None,
        )
        if path_column is None or split_column is None:
            raise ValueError(
                f"Could not identify path/split columns in {split_path}: {fieldnames}"
            )
        collection_column = next(
            (
                by_lower[name]
                for name in ("collection", "dataset", "site", "source")
                if name in by_lower
            ),
            None,
        )
        exact_matches = []
        for row in reader:
            path_value = (row.get(path_column) or "").strip().replace("\\", "/")
            path_value = path_value.removeprefix("./")
            collection = (
                (row.get(collection_column) or "").strip().replace("\\", "/")
                if collection_column
                else ""
            )
            reconstructed = (
                f"{collection}/{path_value}"
                if collection and "/" not in path_value
                else path_value
            )
            split_value = (row.get(split_column) or "").strip()
            if (
                path_value == relative_path
                or reconstructed == relative_path
            ):
                exact_matches.append(split_value)
        matches = exact_matches
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one split record for {relative_path}, found {len(matches)}"
        )
    observed_split = matches[0].casefold()
    contract = plot_contract or config["smoke"]
    expected_split = str(contract["split"]).casefold()
    if observed_split != expected_split or expected_split != "dev":
        raise ValueError(
            f"Configured plot split is {matches[0]!r}; expected development split 'dev'"
        )

    cloud = laspy.read(source_las)
    dimensions = set(cloud.point_format.dimension_names)
    missing = {"treeID", "classification"} - dimensions
    if missing:
        raise ValueError(f"Source LAS is missing fields {sorted(missing)}")
    point_count = len(cloud.points)
    classification = np.asarray(cloud.classification, dtype=np.int64)
    tree_id = np.asarray(cloud["treeID"], dtype=np.int64)
    tree_mask = np.isin(
        classification, config["dataset"]["reference_classes"]
    ) & ~np.isin(tree_id, config["dataset"]["ignored_tree_ids"])
    reference_tree_count = int(len(np.unique(tree_id[tree_mask])))
    expected_points = int(contract["expected_point_count"])
    expected_trees = int(contract["expected_reference_tree_count"])
    if point_count != expected_points:
        raise ValueError(
            f"Source point count {point_count} does not match expected {expected_points}"
        )
    if reference_tree_count != expected_trees:
        raise ValueError(
            "Source reference tree count "
            f"{reference_tree_count} does not match expected {expected_trees}"
        )
    return {
        "relative_path": relative_path,
        "split": matches[0],
        "split_metadata": str(split_path),
        "split_metadata_sha256": sha256(split_path),
        "input_sha256": sha256(source_las),
        "point_count": int(point_count),
        "reference_tree_count": reference_tree_count,
        "classification_values": sorted(
            int(value) for value in np.unique(classification)
        ),
        "retained_points": int(point_count),
        "dropped_points": 0,
        "source_row_identifier": "zero_based_source_row_index",
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_staged_input(source_las: Path, staged_las: Path) -> None:
    staged_las.parent.mkdir(parents=True, exist_ok=True)
    if staged_las.exists() or staged_las.is_symlink():
        raise FileExistsError(f"Refusing to overwrite staged input: {staged_las}")
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


def validate_row_alignment(max_abs_coordinate_delta: float, tolerance: float) -> None:
    if max_abs_coordinate_delta > tolerance:
        raise ValueError(
            "Prediction coordinates are not row-aligned with source LAS: "
            f"max delta {max_abs_coordinate_delta:.6f} m > {tolerance:.6f} m"
        )


def write_adapted_outputs(
    arrays: dict[str, Any],
    adapted_npz: Path,
    adapted_las: Path,
) -> dict[str, Any]:
    import laspy
    import numpy as np

    pred_tree_id = arrays["pred_tree_id"]
    pred_classification = np.where(pred_tree_id > 0, 4, 0).astype(np.uint8)
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
        pred_classification=pred_classification,
        source_row_index=arrays["source_row_index"],
    )

    adapted = arrays["source"]
    if "pred_treeID" not in adapted.point_format.dimension_names:
        adapted.add_extra_dim(laspy.ExtraBytesParams(name="pred_treeID", type=np.int32))
    if "pred_classification" not in adapted.point_format.dimension_names:
        adapted.add_extra_dim(
            laspy.ExtraBytesParams(name="pred_classification", type=np.uint8)
        )
    adapted["pred_treeID"] = pred_tree_id.astype(np.int32)
    adapted["pred_classification"] = pred_classification
    ensure_parent(adapted_las)
    adapted.write(adapted_las)

    positive_prediction_count = int(np.sum(pred_tree_id > 0))
    predicted_tree_count = int(len(np.unique(pred_tree_id[pred_tree_id > 0])))
    return {
        "prediction_min": pred_min,
        "prediction_max": pred_max,
        "positive_prediction_count": positive_prediction_count,
        "predicted_tree_count": predicted_tree_count,
        "predicted_semantic_mapping": {
            "positive_instance_label": 4,
            "background_instance_labels": [0, -1],
            "background_semantic_label": 0,
        },
    }


def build_metadata(
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
    started: float,
    validation: dict[str, Any],
    dataset_validation: dict[str, Any],
    repo_state: dict[str, Any],
    benchmark_repo_state: dict[str, Any],
    plot_contract: dict[str, Any],
    evaluation_scope: str,
    status: str = "completed",
    return_code: int = 0,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint = paths["checkpoint"]
    checkpoint_exists = checkpoint.is_file()
    return {
        "method": config["method"]["slug"],
        "dataset": config["dataset"]["slug"],
        "dataset_split": plot_contract["split"],
        "held_out_test_accessed": False,
        "run_id": args.run_id,
        "training_mode": (
            getattr(args, "training_mode", None)
            or config["method"]["checkpoint"]["training_mode"]
        ),
        "status": status,
        "return_code": return_code,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluation_scope": evaluation_scope,
        "plot": {
            "plot_id": plot_contract["plot_id"],
            "safe_plot_id": plot_contract["safe_plot_id"],
            "relative_path": plot_contract["relative_path"],
            "collection": plot_contract["collection"],
            "split": plot_contract["split"],
            "expected_point_count": plot_contract["expected_point_count"],
            "expected_reference_tree_count": plot_contract[
                "expected_reference_tree_count"
            ],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "exists": checkpoint_exists,
            "sha256": sha256(checkpoint) if checkpoint_exists else None,
            "md5": md5(checkpoint) if checkpoint_exists else None,
            "source_url": config["method"]["checkpoint"]["source_url"],
            "source_dataset_name": config["method"]["checkpoint"][
                "source_dataset_name"
            ],
            "source_md5": config["method"]["checkpoint"]["source_md5"],
        },
        "environment": {
            "treelearn_repo": str(paths["treelearn_repo"]),
            "treelearn_repository": repo_state,
            "benchmark_repository": benchmark_repo_state,
            "python": sys.executable,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "manual"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "unknown"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS", "unknown"),
            "max_rss_kib": max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            ),
        },
        "command": [sys.executable, *sys.argv],
        "upstream_command": [
            sys.executable,
            str(paths["treelearn_repo"] / "tools/pipeline/pipeline.py"),
            "--config",
            str(paths["pipeline_config"]),
        ],
        "dataset_validation": dataset_validation,
        "conversion": {
            "retained_points": dataset_validation.get("retained_points"),
            "dropped_points": dataset_validation.get("dropped_points"),
            "source_row_identifier": dataset_validation.get(
                "source_row_identifier"
            ),
            "reference_semantic_mapping": {
                "tree_classes": config["dataset"]["reference_classes"],
                "ignored_classes": config["dataset"]["ignored_classes"],
            },
            "prediction_semantic_mapping": {
                "positive_pred_tree_id": 4,
                "pred_tree_id_0_or_minus_1": 0,
            },
        },
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key
            in {
                "pipeline_config",
                "raw_prediction_laz",
                "raw_prediction_npz",
                "raw_pointwise_npz",
                "adapted_npz",
                "adapted_las",
                "metadata_json",
            }
        },
        "retention": {
            "prediction_root": str(paths["predictions_root"]),
            "runtime_root": str(paths["runtime_root"]),
            "table_root": str(paths["tables_root"]),
            "raw_pointwise_output_retained": paths["raw_pointwise_npz"].is_file(),
            "raw_full_forest_output_retained": (
                paths["raw_prediction_laz"].is_file()
                and paths["raw_prediction_npz"].is_file()
            ),
            "adapted_point_aligned_output_retained": (
                paths["adapted_npz"].is_file() and paths["adapted_las"].is_file()
            ),
            "files": [
                retained_file(paths[key])
                for key in (
                    "raw_prediction_laz",
                    "raw_prediction_npz",
                    "raw_pointwise_npz",
                    "adapted_npz",
                    "adapted_las",
                )
            ],
        },
        "validation": validation,
        "success_criteria": config["success_criteria"],
        "failure_indicators": config["failure_indicators"],
        "error": error,
        "next_gate": (
            "shared_smoke_evaluation_then_manual_alignment_review"
            if evaluation_scope == "development_smoke"
            else "aggregate_full_development_results_before_any_test_route"
        ),
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
    parser.add_argument(
        "--training-mode",
        choices=("published_pretrained", "fine_tuned_on_dev"),
    )
    parser.add_argument("--runtime-root")
    parser.add_argument("--predictions-root")
    parser.add_argument("--metadata-root")
    parser.add_argument("--tables-root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--relative-path")
    parser.add_argument("--plot-id")
    parser.add_argument("--safe-plot-id")
    parser.add_argument("--expected-split", default="dev")
    parser.add_argument("--expected-point-count", type=int)
    parser.add_argument("--expected-reference-tree-count", type=int)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-split-metadata-sha256")
    parser.add_argument(
        "--evaluation-scope",
        choices=("development_smoke", "development_full"),
        default="development_smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    config, _ = load_config(args.config)

    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError(f"Unsafe TreeLearn run ID: {args.run_id!r}")
    defaults = config.get("plot_defaults") or config.get("smoke") or {}
    relative_path = args.relative_path or defaults.get("relative_path")
    if not relative_path:
        raise ValueError("A development --relative-path is required")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe TreeLearn relative path: {relative_path!r}")
    plot_id = args.plot_id or defaults.get("plot_id") or relative.with_suffix("").as_posix()
    safe_plot_id = (
        args.safe_plot_id
        or defaults.get("safe_plot_id")
        or re.sub(r"[^A-Za-z0-9._-]+", "_", plot_id)
    )
    expected_point_count = args.expected_point_count or defaults.get(
        "expected_point_count"
    )
    expected_reference_tree_count = args.expected_reference_tree_count or defaults.get(
        "expected_reference_tree_count"
    )
    if expected_point_count is None or expected_reference_tree_count is None:
        raise ValueError("Frozen expected point and reference-tree counts are required")
    split = args.expected_split or defaults.get("split")
    if split != "dev":
        raise ValueError("The TreeLearn route must use a development plot")
    plot_contract = {
        "relative_path": relative.as_posix(),
        "plot_id": plot_id,
        "safe_plot_id": safe_plot_id,
        "collection": relative.parts[0] if len(relative.parts) > 1 else "",
        "split": split,
        "expected_point_count": int(expected_point_count),
        "expected_reference_tree_count": int(expected_reference_tree_count),
        "row_coordinate_tolerance_m": float(
            defaults.get("row_coordinate_tolerance_m", 0.005)
        ),
    }

    dataset_root = resolve_path(args.dataset_root or config["dataset"]["root"])
    treelearn_repo = resolve_path(args.treelearn_repo or config["method"]["repo_path"])
    checkpoint = resolve_path(
        args.checkpoint or config["method"]["checkpoint"]["default_path"]
    )
    runtime_base = resolve_path(args.runtime_root or config["paths"]["runtime_root"])
    predictions_base = resolve_path(
        args.predictions_root or config["paths"]["predictions_root"]
    )
    metadata_base = resolve_path(args.metadata_root or config["paths"]["metadata_root"])
    tables_base = resolve_path(args.tables_root or config["paths"]["tables_root"])

    runtime_root = runtime_base / args.run_id / safe_plot_id
    predictions_root = predictions_base / args.run_id / safe_plot_id
    metadata_root = metadata_base / args.run_id
    tables_root = tables_base / args.run_id
    source_las = (dataset_root / plot_contract["relative_path"]).resolve()
    staged_las = runtime_root / "forest" / f"{safe_plot_id}.las"
    pipeline_config = runtime_root / "treelearn_pipeline_config.yml"
    raw_prediction_laz = runtime_root / "results" / "full_forest" / f"{safe_plot_id}.laz"
    raw_prediction_npz = runtime_root / "results" / "full_forest" / f"{safe_plot_id}.npz"
    raw_pointwise_npz = (
        runtime_root / "results" / "pointwise_results" / "pointwise_results.npz"
    )
    artifact_label = (
        "smoke" if args.evaluation_scope == "development_smoke" else "development"
    )
    adapted_npz = (
        predictions_root / f"{safe_plot_id}_treelearn_{artifact_label}_predictions.npz"
    )
    adapted_las = (
        predictions_root / f"{safe_plot_id}_treelearn_{artifact_label}_predictions.las"
    )
    metadata_json = metadata_root / f"{safe_plot_id}_inference.json"

    paths = {
        "dataset_root": dataset_root,
        "treelearn_repo": treelearn_repo,
        "checkpoint": checkpoint,
        "source_las": source_las,
        "runtime_root": runtime_root,
        "predictions_root": predictions_root,
        "metadata_root": metadata_root,
        "staged_las": staged_las,
        "pipeline_config": pipeline_config,
        "raw_prediction_laz": raw_prediction_laz,
        "raw_prediction_npz": raw_prediction_npz,
        "raw_pointwise_npz": raw_pointwise_npz,
        "adapted_npz": adapted_npz,
        "adapted_las": adapted_las,
        "metadata_json": metadata_json,
        "tables_root": tables_root,
    }

    collision_paths = (runtime_root, predictions_root, metadata_json)
    if args.evaluation_scope == "development_smoke":
        collision_paths = (*collision_paths, tables_root)
    collisions = [path for path in collision_paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "TreeLearn run-scoped outputs already exist; use a new run ID: "
            + ", ".join(str(path) for path in collisions)
        )

    repo_state: dict[str, Any] = {}
    benchmark_repo_state: dict[str, Any] = {}
    dataset_validation: dict[str, Any] = {}
    try:
        benchmark_repo_state = benchmark_repository_state(
            ROOT,
            os.environ.get("TREELEARN_EXPECTED_BENCHMARK_COMMIT"),
        )
        for required in (source_las, treelearn_repo, checkpoint):
            if not required.exists():
                raise FileNotFoundError(f"Required path does not exist: {required}")

        expected_commit = config["method"]["upstream_commit"]
        repo_state = repository_state(treelearn_repo, expected_commit)

        validate_checkpoint_identity(
            checkpoint,
            str(config["method"]["checkpoint"]["source_md5"]),
            allow_derived=args.training_mode == "fine_tuned_on_dev",
        )

        dataset_validation = validate_dataset_source(
            config, dataset_root, source_las, plot_contract
        )
        if (
            args.expected_input_sha256
            and dataset_validation["input_sha256"] != args.expected_input_sha256
        ):
            raise ValueError("Source LAS SHA-256 differs from the frozen manifest")
        if (
            args.expected_split_metadata_sha256
            and dataset_validation["split_metadata_sha256"]
            != args.expected_split_metadata_sha256
        ):
            raise ValueError("Split metadata SHA-256 differs from the frozen manifest")
        prepare_staged_input(source_las, staged_las)
        write_yaml(
            pipeline_config,
            treelearn_pipeline_config(staged_las, checkpoint, config),
        )

        run_pipeline(treelearn_repo, pipeline_config)

        if not raw_prediction_laz.is_file():
            raise FileNotFoundError(
                f"TreeLearn full-forest output missing: {raw_prediction_laz}"
            )
        if not raw_prediction_npz.is_file():
            raise FileNotFoundError(
                f"TreeLearn full-forest NPZ output missing: {raw_prediction_npz}"
            )
        if not raw_pointwise_npz.is_file():
            raise FileNotFoundError(
                f"TreeLearn pointwise output missing: {raw_pointwise_npz}"
            )

        arrays = load_prediction_arrays(source_las, raw_prediction_laz)
        row_tolerance = float(plot_contract["row_coordinate_tolerance_m"])
        validate_row_alignment(
            arrays["max_abs_coordinate_delta"], row_tolerance
        )
        row_order_preserved = arrays["max_abs_coordinate_delta"] <= row_tolerance

        prediction_summary = write_adapted_outputs(
            arrays, adapted_npz, adapted_las
        )
        if (
            args.evaluation_scope == "development_smoke"
            and prediction_summary["positive_prediction_count"] <= 0
        ):
            raise ValueError(
                "Prediction contains no positive TreeLearn instance labels."
            )

        validation = {
            "source_point_count": arrays["source_count"],
            "prediction_point_count": arrays["prediction_count"],
            "row_count_match": arrays["source_count"]
            == arrays["prediction_count"],
            "max_abs_coordinate_delta_m": arrays["max_abs_coordinate_delta"],
            "row_coordinate_tolerance_m": row_tolerance,
            "row_order_preserved": row_order_preserved,
            **prediction_summary,
        }
        metadata = build_metadata(
            config,
            paths,
            args,
            started,
            validation,
            dataset_validation,
            repo_state,
            benchmark_repo_state,
            plot_contract,
            args.evaluation_scope,
        )
        ensure_parent(metadata_json)
        metadata_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = build_metadata(
            config,
            paths,
            args,
            started,
            {},
            dataset_validation,
            repo_state,
            benchmark_repo_state,
            plot_contract,
            args.evaluation_scope,
            status="failed",
            return_code=1,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        ensure_parent(metadata_json)
        metadata_json.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
