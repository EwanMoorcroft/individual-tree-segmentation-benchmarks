"""Run the guarded one-plot ForAINet development smoke inside its image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement


EXPECTED_UPSTREAM_COMMIT = "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
EXPECTED_CHECKPOINT_SHA256 = (
    "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
)
EXPECTED_RELATIVE_PATH = "CULS/plot_1_annotated.las"
EXPECTED_POINT_COUNT = 1_816_672
EXPECTED_REFERENCE_TREE_COUNT = 6
ACCEPTED_SMOKE_RUN_ID = (
    "forainet__for-instance__published-pretrained__none__dev-smoke__"
    "20260723T202654"
)
ACCEPTED_SMOKE_LABEL_INDEPENDENCE_SHA256 = (
    "8c31204b4e0bd02bf77dc786e9e3393fa4ab456b8e7f44b635744f624022be79"
)
TILE_SIZE_M = 50
TILE_OVERLAP_M = 5
EVAL_BATCH_SIZE = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path,
    env: dict[str, str] = None,
    accepted_error_markers: tuple[str, ...] = (),
) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as out_handle, stderr.open(
        "w", encoding="utf-8"
    ) as err_handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=out_handle,
            stderr=err_handle,
            text=True,
            check=False,
            env=env,
        )
    if completed.returncode != 0:
        error_text = stderr.read_text(encoding="utf-8", errors="replace")
        if not accepted_error_markers or any(
            marker not in error_text for marker in accepted_error_markers
        ):
            raise RuntimeError(
                f"command failed with exit {completed.returncode}; "
                f"see {stdout} and {stderr}"
            )
    return completed.returncode


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    archive = torch.load(checkpoint, map_location="cpu")
    if not isinstance(archive, dict):
        raise ValueError("official checkpoint archive is not a dictionary")
    run_config = OmegaConf.to_container(
        OmegaConf.create(archive["run_config"]), resolve=True
    )
    if not isinstance(run_config, dict) or not isinstance(
        run_config.get("data"), dict
    ):
        raise ValueError("checkpoint does not retain a usable data configuration")
    data = run_config["data"]
    expected = {
        "task": "panoptic",
        "class": "treeins_set1.TreeinsFusedDataset",
        "radius": 8,
        "grid_size": 0.2,
    }
    observed = {key: data.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"checkpoint data configuration differs from official route: {observed}"
        )
    add_input_features = data.get("add_input_features", [])
    if add_input_features not in ([], None):
        raise ValueError("checkpoint unexpectedly requires labelled auxiliary features")
    models = archive.get("models")
    if not isinstance(models, dict) or "latest" not in models:
        raise ValueError("checkpoint does not contain official latest weights")
    return {
        "schema": "forainet_checkpoint_runtime_provenance_v1",
        "status": "verified",
        "filename": checkpoint.name,
        "sha256": sha256(checkpoint),
        "weight_name": "latest",
        "model_name": run_config.get("model_name"),
        "data_configuration": observed,
        "stored_dataroot_is_absolute": Path(str(data.get("dataroot"))).is_absolute(),
        "runtime_dataroot_override": "data_set1_5classes",
        "add_input_features": [],
        "reference_labels_used_as_features": False,
        "official_eval_entrypoint": "PointCloudSegmentation/eval.py",
        "official_tiler": "PointCloudSegmentation/split_largePC_to_tiles.py",
        "official_merger": "PointCloudSegmentation/merge_tiles.py",
    }


def stage_checkpoint(checkpoint: Path, output: Path) -> str:
    """Copy the archive with only its non-portable data root made relative."""

    if output.exists():
        raise FileExistsError(output)
    archive = torch.load(checkpoint, map_location="cpu")
    run_config = archive.get("run_config")
    if not isinstance(run_config, dict) or not isinstance(
        run_config.get("data"), dict
    ):
        raise ValueError("checkpoint run configuration cannot be staged")
    run_config["data"]["dataroot"] = "data_set1_5classes"
    output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(archive, output)
    staged = torch.load(output, map_location="cpu")
    original_models = archive.get("models")
    staged_models = staged.get("models")
    if not isinstance(original_models, dict) or not isinstance(
        staged_models, dict
    ):
        raise ValueError("staged checkpoint lost model weights")
    if set(original_models) != set(staged_models):
        raise ValueError("staged checkpoint changed available weight names")
    for weight_name in original_models:
        original_state = original_models[weight_name]
        staged_state = staged_models[weight_name]
        if set(original_state) != set(staged_state):
            raise ValueError("staged checkpoint changed tensor keys")
        for key in original_state:
            if not torch.equal(original_state[key], staged_state[key]):
                raise ValueError("staged checkpoint changed tensor values")
    return sha256(output)


def file_inventory(root: Path, paths: list[Path]) -> dict[str, Any]:
    files = []
    for path in sorted(set(paths)):
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise ValueError(f"invalid retained raw path: {path}")
        if not resolved.is_file():
            raise ValueError(f"invalid retained raw path: {path}")
        files.append(
            {
                "relative_path": resolved.relative_to(root.resolve()).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )
    return {
        "schema": "forainet_official_raw_inventory_v1",
        "status": "complete",
        "files": files,
    }


def make_label_probe(source: Path, output: Path) -> None:
    ply = PlyData.read(source)
    vertex = ply["vertex"].data.copy()
    names = set(vertex.dtype.names or ())
    if not {"semantic_seg", "treeID"} <= names:
        raise ValueError("official tile lacks loader bookkeeping fields")
    vertex["semantic_seg"] = 2.0
    vertex["treeID"] = 0.0
    PlyData(
        [PlyElement.describe(vertex, "vertex", comments=["label probe"])],
        byte_order="<",
    ).write(output)


def prediction_values(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    if "preds" not in set(vertex.dtype.names or ()):
        raise ValueError(f"official prediction PLY lacks preds: {path}")
    return np.asarray(vertex["preds"], dtype=np.int64)


def prediction_values_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(b"forainet_prediction_values_v1\n")
    digest.update(
        json.dumps(
            {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-las", required=True, type=Path)
    parser.add_argument("--split-metadata", required=True, type=Path)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument(
        "--route", choices=("smoke", "development"), default="smoke"
    )
    parser.add_argument("--development-task-index", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-point-count", type=int)
    args = parser.parse_args()

    started = time.monotonic()
    if args.route == "smoke" and args.relative_path != EXPECTED_RELATIVE_PATH:
        raise ValueError("smoke route permits only the frozen development plot")
    if args.route == "development" and (
        args.development_task_index is None
        or args.expected_source_sha256 is None
        or args.expected_point_count is None
    ):
        raise ValueError("development route requires frozen task identity")
    if git_commit(args.benchmark_root) != args.benchmark_commit:
        raise ValueError("benchmark checkout changed after submission")
    if git_commit(args.upstream_root) != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("upstream checkout is not the pinned official commit")
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 is not the frozen official asset")
    if not args.run_root.is_dir() or any(args.run_root.iterdir()):
        raise ValueError("run root must exist and be empty")

    input_root = args.run_root / "input"
    raw_root = args.run_root / "raw"
    metadata_root = args.run_root / "metadata"
    aligned_root = args.run_root / "aligned"
    evaluation_root = args.run_root / "evaluation"
    runtime_root = args.run_root / "runtime"
    dataset_raw_root = (
        runtime_root / "data_set1_5classes" / "treeinsfused" / "raw"
    )
    for directory in (
        input_root,
        raw_root,
        metadata_root,
        aligned_root,
        evaluation_root,
        dataset_raw_root,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    checkpoint_payload = checkpoint_provenance(args.checkpoint)
    if checkpoint_payload["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint provenance hash mismatch")
    staged_checkpoint = (
        runtime_root / "checkpoint_stage" / "PointGroup-PAPER.pt"
    )
    checkpoint_payload["staged_checkpoint_sha256"] = stage_checkpoint(
        args.checkpoint, staged_checkpoint
    )
    checkpoint_payload["staged_change"] = (
        "run_config.data.dataroot_only; all model tensors exactly equal"
    )
    write_json(metadata_root / "checkpoint.json", checkpoint_payload)
    cpu_open3d_env = dict(os.environ)
    cpu_open3d_env["LD_LIBRARY_PATH"] = ""
    write_json(
        metadata_root / "environment.json",
        {
            "schema": "forainet_runtime_environment_v1",
            "status": "verified",
            "image_sha256": args.image_sha256,
            "benchmark_commit": args.benchmark_commit,
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "python": sys.version.split()[0],
            "packages": {
                name: package_version(name)
                for name in (
                    "numpy",
                    "torch",
                    "torch-geometric",
                    "MinkowskiEngine",
                    "torchsparse",
                    "hdbscan",
                    "laspy",
                    "plyfile",
                )
            },
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cpu_open3d_child_ld_library_path": "",
            "cpu_open3d_uses_container_glx": True,
            "dataset_raw_catalogue": "hardlink_to_label_isolated_input",
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("official smoke requires a visible CUDA device")

    python = sys.executable
    method_root = args.benchmark_root / "methods" / "forainet"
    input_stem = args.relative_path.replace("/", "__").replace(".las", "")
    inference_ply = input_root / f"{input_stem}_label_isolated.ply"
    sidecar_npz = input_root / "alignment_sidecar.npz"
    input_metadata = metadata_root / "conversion.json"
    run_checked(
        [
            python,
            str(
                method_root
                / "scripts/data/prepare_label_isolated_input.py"
            ),
            "--source-las",
            str(args.source_las),
            "--relative-path",
            args.relative_path,
            "--split-metadata",
            str(args.split_metadata),
            "--output-ply",
            str(inference_ply),
            "--alignment-sidecar-npz",
            str(sidecar_npz),
            "--metadata-json",
            str(input_metadata),
        ],
        cwd=runtime_root,
        stdout=metadata_root / "conversion.stdout",
        stderr=metadata_root / "conversion.stderr",
    )
    conversion = json.loads(input_metadata.read_text(encoding="utf-8"))
    if conversion["split"] != "dev":
        raise ValueError("runtime route permits development plots only")
    if args.route == "smoke":
        if (
            conversion["source_point_count"] != EXPECTED_POINT_COUNT
            or conversion["reference_tree_count"] != EXPECTED_REFERENCE_TREE_COUNT
        ):
            raise ValueError("frozen smoke plot identity failed")
    elif (
        conversion["source_sha256"] != args.expected_source_sha256
        or conversion["source_point_count"] != args.expected_point_count
    ):
        raise ValueError("development source differs from frozen manifest")
    raw_catalogue_input = dataset_raw_root / inference_ply.name
    os.link(inference_ply, raw_catalogue_input)

    pointcloud_root = args.upstream_root / "PointCloudSegmentation"
    run_checked(
        [
            python,
            str(pointcloud_root / "split_largePC_to_tiles.py"),
            "--file_path",
            str(inference_ply),
            "--tile_size",
            str(TILE_SIZE_M),
            "--overlap",
            str(TILE_OVERLAP_M),
        ],
        cwd=runtime_root,
        stdout=metadata_root / "tiling.stdout",
        stderr=metadata_root / "tiling.stderr",
        env=cpu_open3d_env,
    )
    tile_root = input_root / f"tiles_{TILE_SIZE_M}_{inference_ply.stem}"
    tile_list_path = tile_root / "ply_output_file_paths.txt"
    if not tile_list_path.is_file():
        raise FileNotFoundError(tile_list_path)
    tile_paths = [
        Path(line.strip())
        for line in tile_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tile_paths or any(not path.is_file() for path in tile_paths):
        raise ValueError("official tiler did not produce a complete tile list")

    batch_root = raw_root / "eval_batches"
    batch_root.mkdir()
    for batch_number, start in enumerate(
        range(0, len(tile_paths), EVAL_BATCH_SIZE)
    ):
        batch = tile_paths[start : start + EVAL_BATCH_SIZE]
        output = batch_root / f"batch_{batch_number:04d}"
        fold_override = json.dumps([str(path) for path in batch], separators=(",", ":"))
        run_checked(
            [
                python,
                str(pointcloud_root / "eval.py"),
                f"checkpoint_dir={staged_checkpoint.parent}",
                "model_name=PointGroup-PAPER",
                f"data.fold={fold_override}",
                "num_workers=0",
                "batch_size=1",
                "cuda=0",
                "weight_name=latest",
                "voting_runs=1",
                "enable_dropout=false",
                "tracker_options.full_res=true",
                "tracker_options.make_submission=true",
                "tracker_options.ply_output=vote1regular.ply",
                f"hydra.run.dir={output}",
            ],
            cwd=runtime_root,
            stdout=raw_root / f"eval_batch_{batch_number:04d}.stdout",
            stderr=raw_root / f"eval_batch_{batch_number:04d}.stderr",
        )
        for local_index in range(len(batch)):
            for prefix in (
                "Semantic_results_forEval",
                "Instance_Results_forEval",
            ):
                expected = output / f"{prefix}_{local_index}.ply"
                if not expected.is_file():
                    raise FileNotFoundError(expected)

    label_probe_metadata = metadata_root / "label_independence.json"
    if args.route == "smoke":
        label_probe_input = input_root / "label_probe_tile.ply"
        make_label_probe(tile_paths[0], label_probe_input)
        label_probe_output = raw_root / "label_probe"
        label_probe_exit_code = run_checked(
            [
                python,
                str(pointcloud_root / "eval.py"),
                f"checkpoint_dir={staged_checkpoint.parent}",
                "model_name=PointGroup-PAPER",
                (
                    "data.fold="
                    + json.dumps(
                        [str(label_probe_input)], separators=(",", ":")
                    )
                ),
                "num_workers=0",
                "batch_size=1",
                "cuda=0",
                "weight_name=latest",
                "voting_runs=1",
                "enable_dropout=false",
                "tracker_options.full_res=true",
                "tracker_options.make_submission=true",
                "tracker_options.ply_output=vote1regular.ply",
                f"hydra.run.dir={label_probe_output}",
            ],
            cwd=runtime_root,
            stdout=raw_root / "label_probe.stdout",
            stderr=raw_root / "label_probe.stderr",
            accepted_error_markers=(
                "treeins_set1.py\", line 204, in final_eval",
                "ZeroDivisionError: float division by zero",
            ),
        )
        primary_output = batch_root / "batch_0000"
        comparisons = {}
        for kind, filename in (
            ("semantic", "Semantic_results_forEval_0.ply"),
            ("instance", "Instance_Results_forEval_0.ply"),
        ):
            primary = primary_output / filename
            probe = label_probe_output / filename
            primary_values = prediction_values(primary)
            probe_values = prediction_values(probe)
            equal = np.array_equal(primary_values, probe_values)
            comparisons[kind] = {
                "prediction_equal": bool(equal),
                "point_count": int(primary_values.size),
                "primary_file_sha256": sha256(primary),
                "probe_file_sha256": sha256(probe),
                "primary_prediction_values_sha256": prediction_values_sha256(
                    primary_values
                ),
                "probe_prediction_values_sha256": prediction_values_sha256(
                    probe_values
                ),
            }
            if not equal:
                raise ValueError(
                    f"official {kind} prediction changed with bookkeeping labels"
                )
        write_json(
            label_probe_metadata,
            {
                "schema": "forainet_label_independence_probe_v2",
                "status": "verified",
                "reference_labels_used": False,
                "primary_bookkeeping": {"semantic_seg": 1, "treeID": -1},
                "probe_bookkeeping": {"semantic_seg": 2, "treeID": 0},
                "official_eval_exit_code": label_probe_exit_code,
                "accepted_reporting_failure": (
                    "official_final_eval_zero_denominator_on_artificial_labels"
                    if label_probe_exit_code
                    else None
                ),
                "comparison": comparisons,
            },
        )
    else:
        write_json(
            label_probe_metadata,
            {
                "schema": "forainet_label_independence_inheritance_v1",
                "status": "verified_by_accepted_development_smoke",
                "reference_labels_used": False,
                "bookkeeping": {"semantic_seg": 1, "treeID": -1},
                "accepted_smoke_run_id": ACCEPTED_SMOKE_RUN_ID,
                "accepted_smoke_label_independence_sha256": (
                    ACCEPTED_SMOKE_LABEL_INDEPENDENCE_SHA256
                ),
            },
        )

    collected_root = raw_root / "official_collected_tiles"
    merged_ply = raw_root / "official_merged_result.ply"
    run_checked(
        [
            python,
            str(pointcloud_root / "merge_tiles.py"),
            "--src_dir",
            str(batch_root),
            "--ply_output_file",
            str(tile_list_path),
            "--base_dir",
            str(collected_root),
            "--output_file",
            str(merged_ply),
            "--original_point_cloud_file",
            str(inference_ply),
            "--overlap",
            str(TILE_OVERLAP_M),
        ],
        cwd=runtime_root,
        stdout=metadata_root / "merge.stdout",
        stderr=metadata_root / "merge.stderr",
        env=cpu_open3d_env,
    )

    raw_prediction = raw_root / "prediction_source_order.npz"
    merge_metadata = metadata_root / "merge_alignment.json"
    run_checked(
        [
            python,
            str(method_root / "scripts/runtime/extract_official_merge.py"),
            "--merged-ply",
            str(merged_ply),
            "--inference-ply",
            str(inference_ply),
            "--expected-point-count",
            str(EXPECTED_POINT_COUNT),
            "--output-npz",
            str(raw_prediction),
            "--metadata-json",
            str(merge_metadata),
        ],
        cwd=runtime_root,
        stdout=metadata_root / "extraction.stdout",
        stderr=metadata_root / "extraction.stderr",
    )

    aligned_prediction = aligned_root / "prediction.npz"
    aligned_metadata = metadata_root / "aligned_prediction.json"
    run_checked(
        [
            python,
            str(
                method_root
                / "scripts/runtime/normalise_forainet_predictions.py"
            ),
            "--raw-prediction-npz",
            str(raw_prediction),
            "--alignment-sidecar-npz",
            str(sidecar_npz),
            "--output-npz",
            str(aligned_prediction),
            "--metadata-json",
            str(aligned_metadata),
            "--run-id",
            args.run_id,
            "--relative-path",
            args.relative_path,
            "--split",
            "dev",
        ],
        cwd=runtime_root,
        stdout=metadata_root / "normalisation.stdout",
        stderr=metadata_root / "normalisation.stderr",
    )

    metrics = evaluation_root / "metrics.json"
    matches = evaluation_root / "matches.csv"
    unmatched_predictions = evaluation_root / "unmatched_predictions.csv"
    unmatched_references = evaluation_root / "unmatched_references.csv"
    run_checked(
        [
            python,
            str(method_root / "scripts/evaluation/evaluate_for_instance.py"),
            "--prediction-npz",
            str(aligned_prediction),
            "--metrics-json",
            str(metrics),
            "--matches-csv",
            str(matches),
            "--unmatched-predictions-csv",
            str(unmatched_predictions),
            "--unmatched-references-csv",
            str(unmatched_references),
            "--split",
            "dev",
        ],
        cwd=runtime_root,
        stdout=metadata_root / "evaluation.stdout",
        stderr=metadata_root / "evaluation.stderr",
    )

    raw_paths = [
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and path.name not in {"raw_inventory.json", "PointGroup-PAPER.pt"}
    ]
    raw_inventory = metadata_root / "raw_inventory.json"
    write_json(raw_inventory, file_inventory(args.run_root, raw_paths))
    elapsed = time.monotonic() - started
    child_rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    plot_metadata = metadata_root / "plot.json"
    plot_status = (
        "completed_waiting_manual_alignment"
        if args.route == "smoke"
        else "completed_development_diagnostic"
    )
    next_gate = (
        "manual_alignment_review"
        if args.route == "smoke"
        else "development_run_summary"
    )
    write_json(
        plot_metadata,
        {
            "schema": "forainet_plot_runtime_v2",
            "status": plot_status,
            "run_id": args.run_id,
            "route": args.route,
            "development_task_index": args.development_task_index,
            "relative_path": args.relative_path,
            "split": "dev",
            "point_count": conversion["source_point_count"],
            "reference_tree_count": conversion["reference_tree_count"],
            "tile_size_m": TILE_SIZE_M,
            "tile_overlap_m": TILE_OVERLAP_M,
            "tile_count": len(tile_paths),
            "official_eval_batch_size": EVAL_BATCH_SIZE,
            "official_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
            "benchmark_commit": args.benchmark_commit,
            "label_isolation": "constant_nonreference_bookkeeping_fields",
            "reference_labels_supplied_to_model": False,
            "label_independence_probe_sha256": sha256(label_probe_metadata),
            "label_independence_verified": True,
            "source_row_correspondence": "exact_integer_index_through_official_tiles",
            "coordinate_matching": False,
            "wall_runtime_seconds": elapsed,
            "peak_child_rss_kb": child_rss_kb,
            "raw_inventory_sha256": sha256(raw_inventory),
            "aligned_prediction_sha256": sha256(aligned_prediction),
            "metrics_sha256": sha256(metrics),
            "next_gate": next_gate,
        },
    )

    manifest = args.run_root / "retention" / "manifest.json"
    run_checked(
        [
            python,
            str(
                method_root
                / "scripts/provenance/build_retention_manifest.py"
            ),
            "--run-root",
            str(args.run_root),
            "--file",
            f"official_raw_output={merged_ply}",
            "--file",
            f"aligned_prediction={aligned_prediction}",
            "--file",
            f"plot_metadata={plot_metadata}",
            "--file",
            f"plot_metrics={metrics}",
            "--file",
            f"matched_pairs={matches}",
            "--file",
            f"unmatched_predictions={unmatched_predictions}",
            "--file",
            f"unmatched_references={unmatched_references}",
            "--file",
            f"environment_manifest={metadata_root / 'environment.json'}",
            "--file",
            f"checkpoint_provenance={metadata_root / 'checkpoint.json'}",
            "--file",
            f"input_conversion={metadata_root / 'conversion.json'}",
            "--file",
            f"label_independence_probe={label_probe_metadata}",
            "--file",
            f"merge_alignment={merge_metadata}",
            "--file",
            f"aligned_prediction_metadata={aligned_metadata}",
            "--file",
            f"raw_output_inventory={raw_inventory}",
            "--output-json",
            str(manifest),
        ],
        cwd=runtime_root,
        stdout=metadata_root / "retention.stdout",
        stderr=metadata_root / "retention.stderr",
    )
    final_gate = {
        "schema": (
            "forainet_smoke_final_gate_v1"
            if args.route == "smoke"
            else "forainet_development_plot_final_gate_v1"
        ),
        "status": "complete",
        "scientific_status": (
            "waiting_manual_alignment"
            if args.route == "smoke"
            else "development_diagnostic_complete"
        ),
        "run_id": args.run_id,
        "route": args.route,
        "development_task_index": args.development_task_index,
        "relative_path": args.relative_path,
        "retention_manifest_sha256": sha256(manifest),
        "held_out_access": False,
    }
    write_json(args.run_root / "final_gate.json", final_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
