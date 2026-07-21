"""Run published TLS2trees instance segmentation for one development plot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path(__file__).resolve().parent
SRC = ROOT / "src"
for entry in (RUNTIME, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from benchmark.ply_io import read_ply_header
from for_instance_published_common import (
    EXPECTED_SPLIT,
    EXPECTED_VARIANT,
    load_config,
    peak_rss_gb,
    resolve_held_out_test_plot_context,
    resolve_plot_context,
    sha256,
    utc_now,
    verify_upstream,
    write_json,
)
from run_tls2trees_instance_for_plot import build_command


PATCH_WRAPPER = RUNTIME / "patches" / "instance_patched.py"
NO_PREDICTIONS_PATTERN = ".*.tls2trees_no_predictions.txt"
NO_PREDICTIONS_REASONS = {
    "no_clustered_wood_convex_hulls",
    "no_graph_connected_stem_bases",
    "no_in_tile_stem_predictions",
}


def archive_failed_instance_attempt(
    *,
    plot_root: Path,
    raw_root: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Failed instance metadata does not exist: {metadata_path}"
        )
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    if previous.get("status") != "failed":
        raise ValueError("Instance recovery requires status=failed metadata")
    if "Instance tile" not in str(previous.get("error", "")):
        raise ValueError("Instance recovery metadata has an unexpected failure")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Failed raw prediction root does not exist: {raw_root}")
    retained_raw_files = [path for path in raw_root.rglob("*") if path.is_file()]
    if retained_raw_files:
        raise RuntimeError(
            "Refusing instance recovery because the failed raw root contains files: "
            f"{retained_raw_files[0]}"
        )
    logs_root = plot_root / "logs" / "instance"
    if not logs_root.is_dir():
        raise FileNotFoundError(f"Failed instance log root does not exist: {logs_root}")
    tile_errors = sorted(logs_root.glob("tile_*.stderr.log"))
    if not any(
        "sources must not be empty" in path.read_text(encoding="utf-8")
        for path in tile_errors
    ):
        raise ValueError(
            "Instance recovery is restricted to the audited empty graph-source failure"
        )

    recovery_root = plot_root / "recovery" / "instance_failed_attempt_1"
    if recovery_root.exists():
        raise FileExistsError(f"Instance recovery archive already exists: {recovery_root}")
    recovery_root.mkdir(parents=True)
    previous_metadata_sha256 = sha256(metadata_path)
    shutil.move(str(raw_root), str(recovery_root / "raw"))
    shutil.move(str(logs_root), str(recovery_root / "logs"))
    shutil.move(str(metadata_path), str(recovery_root / "instance_run.json"))
    return {
        "status": "failed_attempt_archived",
        "archive_root": str(recovery_root),
        "previous_metadata": str(recovery_root / "instance_run.json"),
        "previous_metadata_sha256": previous_metadata_sha256,
        "previous_error": previous.get("error"),
        "previous_benchmark_commit": os.environ.get(
            "TLS2TREES_RECOVERY_FROM_BENCHMARK_COMMIT"
        ),
    }


def resolved_instance_parameters(config: dict[str, Any]) -> dict[str, Any]:
    source = config["instance_parameters"]
    return {
        "n_tiles": source["n_tiles"],
        "n_zeros": source["n_zeros"],
        "overlap": source["overlap"],
        "slice_thickness": source["slice_thickness_m"],
        "find_stems_boundary": source["find_stems_boundary_m"],
        "find_stems_min_radius": source["find_stems_min_radius_m"],
        "find_stems_min_points": source["find_stems_min_points"],
        "graph_edge_length": source["graph_edge_length_m"],
        "graph_maximum_cumulative_gap": source[
            "graph_maximum_cumulative_gap_m"
        ],
        "min_points_per_tree": source["min_points_per_tree"],
        "add_leaves": source["add_leaves"],
        "add_leaves_voxel_length": source["add_leaves_voxel_length_m"],
        "add_leaves_edge_length": source["add_leaves_edge_length_m"],
        "save_diameter_class": source["save_diameter_class"],
        "ignore_missing_tiles": source["ignore_missing_tiles"],
        "pandarallel": source["pandarallel"],
        "verbose": source["verbose"],
    }


def prediction_inventory(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, suffix in (("leaf_off", "*.leafoff.ply"), ("leaf_on", "*.leafon.ply")):
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob(suffix)):
            header = read_ply_header(path)
            records.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "point_count": header.vertex_count,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        result[target] = records
    return result


def run_instance(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    tls2trees_repo: Path,
    config_path: str,
    variant: str = EXPECTED_VARIANT,
    split: str = EXPECTED_SPLIT,
    dry_run: bool = False,
    resume_failed_empty_output: bool = False,
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, resolved_config = load_config(config_path)
    is_held_out_test = split == "test"
    if is_held_out_test:
        if not allow_held_out_test:
            raise ValueError(
                "Held-out instance inference requires --allow-held-out-test"
            )
        plot_root, row = resolve_held_out_test_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
        )
    else:
        plot_root, row = resolve_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
            split=split,
        )
    upstream = verify_upstream(config, tls2trees_repo)
    semantic_metadata_path = plot_root / "metadata" / "semantic_run.json"
    if not semantic_metadata_path.is_file():
        raise FileNotFoundError(f"Semantic run metadata does not exist: {semantic_metadata_path}")
    semantic_metadata = json.loads(
        semantic_metadata_path.read_text(encoding="utf-8")
    )
    if semantic_metadata.get("status") != "completed":
        raise ValueError("Semantic inference is not completed")
    semantic_root = plot_root / "semantic"
    converted_root = plot_root / "converted"
    tile_index = converted_root / "tile_index.dat"
    conversion_metadata_path = converted_root / "conversion_metadata.json"
    conversion = json.loads(conversion_metadata_path.read_text(encoding="utf-8"))
    if not tile_index.is_file() or sha256(tile_index) != conversion["tile_index_sha256"]:
        raise RuntimeError("Converted tile index is missing or has changed")
    semantic_outputs = [Path(record["path"]) for record in semantic_metadata["outputs"]]
    if not semantic_outputs:
        raise ValueError("Semantic metadata contains no tile outputs")
    for record, path in zip(semantic_metadata["outputs"], semantic_outputs):
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Semantic output is missing or changed: {path}")

    raw_root = plot_root / "predictions" / "raw"
    metadata_path = plot_root / "metadata" / (
        "instance_dry_run.json" if dry_run else "instance_run.json"
    )
    recovery: dict[str, Any] | None = None
    if resume_failed_empty_output:
        if dry_run:
            raise ValueError("Instance recovery cannot be combined with --dry-run")
        recovery = archive_failed_instance_attempt(
            plot_root=plot_root,
            raw_root=raw_root,
            metadata_path=metadata_path,
        )
    elif raw_root.exists():
        raise FileExistsError(
            f"Raw prediction root already exists; use a new run_id: {raw_root}"
        )
    parameters = resolved_instance_parameters(config)
    commands = [
        build_command(PATCH_WRAPPER, tile, tile_index, raw_root, parameters)
        for tile in semantic_outputs
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "running",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": variant,
        "split": split,
        "run_id": run_id,
        "task_index": task_index,
        "relative_path": row["relative_path"],
        "safe_plot_id": row["safe_plot_id"],
        "hostname": platform.node(),
        "config_path": str(resolved_config),
        "config_sha256": sha256(resolved_config),
        "conversion_metadata": str(conversion_metadata_path),
        "semantic_metadata": str(semantic_metadata_path),
        "tls2trees": upstream,
        "patch_wrapper": str(PATCH_WRAPPER),
        "patch_wrapper_sha256": sha256(PATCH_WRAPPER),
        "compatibility_patches": [
            "pandas_groupby_apply_clstr_restore",
            "empty_groupby_apply_recorded_as_no_predictions",
            "parsed_leaf_graph_edge_length",
            "empty_graph_sources_recorded_as_no_predictions",
            "empty_in_tile_stems_recorded_as_no_predictions",
            "empty_leaf_tip_graph_preserves_stem_only_leaf_on_predictions",
            "small_wood_graph_neighbours_capped_to_available_samples",
            "deterministic_numpy_and_python_seed",
        ],
        "reproducibility_seed": config["reproducibility_controls"][
            "numpy_random_seed"
        ],
        "resolved_instance_parameters": parameters,
        "commands": commands,
        "raw_prediction_root": str(raw_root),
        "prediction_inventory": {},
        "return_code": None,
        "runtime_seconds": None,
        "peak_rss_gb": None,
        "held_out_test_accessed": is_held_out_test,
        "recovery": recovery,
    }
    if dry_run:
        payload.update(
            {
                "ended_at_utc": utc_now(),
                "return_code": 0,
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "peak_rss_gb": peak_rss_gb(),
            }
        )
        write_json(metadata_path, payload)
        return payload

    raw_root.mkdir(parents=True)
    logs_root = plot_root / "logs" / "instance"
    logs_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TLS2TREES_REPO"] = upstream["repo"]
    environment["TLS2TREES_SEED"] = str(payload["reproducibility_seed"])
    environment["PYTHONHASHSEED"] = str(payload["reproducibility_seed"])
    python_paths = [upstream["repo"], str(Path(upstream["repo"]) / "tls2trees")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    try:
        for index, command in enumerate(commands):
            stdout_path = logs_root / f"tile_{index:06d}.stdout.log"
            stderr_path = logs_root / f"tile_{index:06d}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=upstream["repo"],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    shell=False,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Instance tile {index} failed with return code {completed.returncode}; "
                    f"see {stderr_path}"
                )
        inventory = prediction_inventory(raw_root)
        no_prediction_evidence = []
        for path in sorted(raw_root.glob(NO_PREDICTIONS_PATTERN)):
            reason = path.read_text(encoding="utf-8").strip()
            if reason not in NO_PREDICTIONS_REASONS:
                raise ValueError(f"Unexpected no-predictions reason in {path}: {reason}")
            no_prediction_evidence.append(
                {"path": str(path), "reason": reason, "sha256": sha256(path)}
            )
        if parameters["add_leaves"] and len(inventory["leaf_off"]) != len(
            inventory["leaf_on"]
        ):
            raise ValueError(
                "Leaf-off and leaf-on prediction counts differ despite add_leaves=true"
            )
        payload["prediction_inventory"] = inventory
        payload["no_prediction_evidence"] = no_prediction_evidence
        if not inventory["leaf_off"] and len(no_prediction_evidence) != len(commands):
            raise RuntimeError(
                "TLS2trees emitted no predictions without one empty-graph reason "
                "for every semantic tile"
            )
        payload.update(
            {
                "status": (
                    "completed"
                    if inventory["leaf_off"]
                    else "completed_no_predictions"
                ),
                "return_code": 0,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "return_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        payload["ended_at_utc"] = utc_now()
        payload["runtime_seconds"] = round(time.perf_counter() - started, 6)
        payload["peak_rss_gb"] = peak_rss_gb()
        write_json(metadata_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run published TLS2trees instances for one development plot."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tls2trees-repo", required=True)
    parser.add_argument(
        "--config",
        default="methods/tls2trees/configs/for_instance_published_default.yml",
    )
    parser.add_argument("--variant", default=EXPECTED_VARIANT)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-failed-empty-output", action="store_true")
    parser.add_argument("--allow-held-out-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = run_instance(
            manifest_path=Path(args.manifest_json),
            task_index=args.task_index,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            tls2trees_repo=Path(args.tls2trees_repo),
            config_path=args.config,
            variant=args.variant,
            split=args.split,
            dry_run=args.dry_run,
            resume_failed_empty_output=args.resume_failed_empty_output,
            allow_held_out_test=args.allow_held_out_test,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print(
        f"leaf_off_predictions={len(payload.get('prediction_inventory', {}).get('leaf_off', []))}"
    )
    print(
        f"leaf_on_predictions={len(payload.get('prediction_inventory', {}).get('leaf_on', []))}"
    )
    print(
        "held_out_test_accessed="
        + str(bool(payload["held_out_test_accessed"])).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
