"""Run pinned TLS2trees semantic inference for one prepared development plot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from for_instance_published_common import (
    EXPECTED_SPLIT,
    EXPECTED_VARIANT,
    load_config,
    peak_rss_gb,
    resolve_development_plot_context,
    resolve_held_out_test_plot_context,
    resolve_plot_context,
    sha256,
    utc_now,
    verify_upstream,
    write_json,
)


PATCH_WRAPPER = RUNTIME / "patches" / "semantic_patched.py"


def build_semantic_command(
    *,
    input_tile: Path,
    tile_index: Path,
    output_dir: Path,
    model_path: Path,
    config: dict[str, Any],
) -> list[str]:
    parameters = config["semantic_parameters"]
    operational = parameters["operational_defaults"]
    command = [
        sys.executable,
        str(PATCH_WRAPPER),
        "--point-cloud",
        str(input_tile),
        "--tile-index",
        str(tile_index),
        "--buffer",
        str(parameters["buffer_m"]),
        "--zeros",
        str(operational["zeros"]),
        "--batch_size",
        str(parameters["batch_size"]),
        "--num_procs",
        str(parameters["num_processes"]),
        "--is-wood",
        str(parameters["is_wood_probability_threshold"]),
        "--model",
        str(model_path),
        "--output_fmt",
        str(parameters["output_format"]),
        "--step",
        str(operational["step"]),
        "--odir",
        str(output_dir),
        "--verbose",
    ]
    if operational.get("redo") is not None:
        command.extend(["--redo", str(operational["redo"])])
    if operational.get("keep_npy") is True:
        command.append("--keep-npy")
    return command


def run_semantic(
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
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    config, resolved_config = load_config(config_path)
    is_held_out_test = split == "test"
    if is_held_out_test:
        if not allow_held_out_test:
            raise ValueError("Held-out semantic inference requires --allow-held-out-test")
        plot_root, row = resolve_held_out_test_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
        )
    elif variant == EXPECTED_VARIANT:
        plot_root, row = resolve_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
            split=split,
        )
    else:
        if split != EXPECTED_SPLIT:
            raise ValueError("TLS2trees semantic inference remains development-only")
        plot_root, row = resolve_development_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
            allowed_variants={"development_tuned"},
        )
    converted = plot_root / "converted"
    conversion_metadata_path = converted / "conversion_metadata.json"
    if not conversion_metadata_path.is_file():
        raise FileNotFoundError(
            f"Conversion metadata does not exist: {conversion_metadata_path}"
        )
    conversion = json.loads(conversion_metadata_path.read_text(encoding="utf-8"))
    if (
        conversion.get("variant") != variant
        or conversion.get("split") != split
        or not conversion.get("labels_stripped")
    ):
        raise ValueError("Conversion is not an accepted label-stripped input")
    if float(conversion["tile_size_m"]) != float(
        config["published_preprocessing"]["tile_edge_length_m"]
    ) or float(conversion["downsample_voxel_size_m"]) != float(
        config["published_preprocessing"]["downsample_voxel_length_m"]
    ):
        raise ValueError("Conversion parameters do not match published/default provenance")
    required_buffer_tiles = int(
        config["semantic_parameters"]["hidden_fixed_values"][
            "buffer_tile_neighbours"
        ]
    )
    if (
        float(config["semantic_parameters"]["buffer_m"]) > 0
        and len(conversion["tiles"]) < required_buffer_tiles
    ):
        raise ValueError(
            "Published upstream semantic buffering requires at least "
            f"{required_buffer_tiles} tiles; found {len(conversion['tiles'])}. "
            "No neighbour-count fallback is permitted."
        )

    upstream = verify_upstream(config, tls2trees_repo)
    wrapper_sha = sha256(PATCH_WRAPPER)
    tile_index = Path(conversion["tile_index"])
    if sha256(tile_index) != conversion["tile_index_sha256"]:
        raise RuntimeError("Converted tile index checksum has changed")
    semantic_root = plot_root / "semantic"
    if semantic_root.exists():
        raise FileExistsError(
            f"Semantic output already exists; use a new run_id: {semantic_root}"
        )

    commands: list[list[str]] = []
    expected_outputs: list[Path] = []
    for record in conversion["tiles"]:
        tile = Path(record["path"])
        if not tile.is_file() or sha256(tile) != record["sha256"]:
            raise RuntimeError(f"Converted tile is missing or changed: {tile}")
        commands.append(
            build_semantic_command(
                input_tile=tile,
                tile_index=tile_index,
                output_dir=semantic_root,
                model_path=Path(upstream["model"]),
                config=config,
            )
        )
        expected_outputs.append(semantic_root / f"{tile.stem}.segmented.ply")

    metadata_path = plot_root / "metadata" / (
        "semantic_dry_run.json" if dry_run else "semantic_run.json"
    )
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
        "tls2trees": upstream,
        "patch_wrapper": str(PATCH_WRAPPER),
        "patch_wrapper_sha256": wrapper_sha,
        "compatibility_patches": [
            "pandas_dataframe_append_api_compatibility",
            "semantic_batch_local_shift_indexing",
            "seeded_numpy_torch_and_python_random_state",
            "upstream_compatible_nondeterministic_cuda_scatter",
        ],
        "required_upstream_buffer_tile_count": required_buffer_tiles,
        "reproducibility_seed": config["reproducibility_controls"][
            "numpy_random_seed"
        ],
        "deterministic_algorithms_enabled": config["reproducibility_controls"][
            "deterministic_algorithms"
        ],
        "determinism_policy": config["reproducibility_controls"][
            "determinism_policy"
        ],
        "commands": commands,
        "expected_outputs": [str(path) for path in expected_outputs],
        "outputs": [],
        "return_code": None,
        "runtime_seconds": None,
        "peak_rss_gb": None,
        "held_out_test_accessed": is_held_out_test,
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

    semantic_root.mkdir(parents=True)
    logs_root = plot_root / "logs" / "semantic"
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
                    f"Semantic tile {index} failed with return code {completed.returncode}; "
                    f"see {stderr_path}"
                )
        from benchmark.ply_io import read_ply_header

        for output in expected_outputs:
            header = read_ply_header(output)
            if header.vertex_count <= 0:
                raise ValueError(f"Semantic output contains no points: {output}")
            missing = sorted({"x", "y", "z", "n_z", "label"} - set(header.columns))
            if missing:
                raise ValueError(
                    f"Semantic output {output} is missing columns: {', '.join(missing)}"
                )
            payload["outputs"].append(
                {
                    "path": str(output),
                    "point_count": header.vertex_count,
                    "sha256": sha256(output),
                }
            )
        payload.update({"status": "completed", "return_code": 0})
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
        description="Run published TLS2trees semantic inference for one development plot."
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
    parser.add_argument("--allow-held-out-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = run_semantic(
            manifest_path=Path(args.manifest_json),
            task_index=args.task_index,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            tls2trees_repo=Path(args.tls2trees_repo),
            config_path=args.config,
            variant=args.variant,
            split=args.split,
            dry_run=args.dry_run,
            allow_held_out_test=args.allow_held_out_test,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print(f"semantic_tiles={len(payload['expected_outputs'])}")
    print(
        "held_out_test_accessed="
        + str(bool(payload["held_out_test_accessed"])).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
