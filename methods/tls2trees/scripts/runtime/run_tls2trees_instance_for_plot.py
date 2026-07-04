"""Run the TLS2trees instance stage for one prepared plot and record metadata."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
VALUE_FLAGS = {
    "n_tiles": "--n-tiles",
    "n_zeros": "--n-zeros",
    "overlap": "--overlap",
    "slice_thickness": "--slice-thickness",
    "find_stems_boundary": "--find-stems-boundary",
    "find_stems_min_radius": "--find-stems-min-radius",
    "find_stems_min_points": "--find-stems-min-points",
    "graph_edge_length": "--graph-edge-length",
    "graph_maximum_cumulative_gap": "--graph-maximum-cumulative-gap",
    "min_points_per_tree": "--min-points-per-tree",
    "add_leaves_voxel_length": "--add-leaves-voxel-length",
    "add_leaves_edge_length": "--add-leaves-edge-length",
}
BOOLEAN_FLAGS = {
    "add_leaves": "--add-leaves",
    "save_diameter_class": "--save-diameter-class",
    "ignore_missing_tiles": "--ignore-missing-tiles",
    "pandarallel": "--pandarallel",
    "verbose": "--verbose",
}
PACKAGE_DISTRIBUTIONS = {
    "laspy": "laspy",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit_learn": "scikit-learn",
    "scipy": "scipy",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_from_root(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_benchmark_config(path_text: str) -> tuple[dict[str, Any], Path, Path]:
    config_path = Path(path_text).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    try:
        project_root = Path(config["project"]["barkla_root"]).expanduser().resolve()
        config["dataset"]["plots"]
        config["conversion"]["tile_name"]
        config["method"]["patched_instance_script"]
        config["outputs"]["converted_root"]
        config["outputs"]["predictions_root"]
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return config, config_path, project_root


def git_commit(repo_path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[name] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_peak_memory_gb(*paths: Path) -> float | None:
    pattern = re.compile(r"peak memory:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    matches: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
            matches.append(float(match.group(1)))
    return max(matches) if matches else None


def build_command(
    script_path: Path,
    tile_path: Path,
    tile_index: Path,
    output_dir: Path,
    parameters: dict[str, Any],
) -> list[str]:
    unknown = sorted(set(parameters) - set(VALUE_FLAGS) - set(BOOLEAN_FLAGS))
    if unknown:
        raise ValueError(f"Unknown TLS2trees instance parameter(s): {', '.join(unknown)}")

    command = [
        sys.executable,
        str(script_path),
        "--tile",
        str(tile_path),
        "--odir",
        str(output_dir),
        "--tindex",
        str(tile_index),
    ]
    for name, flag in VALUE_FLAGS.items():
        value = parameters.get(name)
        if value is None or value is False:
            continue
        command.append(flag)
        if isinstance(value, list):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    for name, flag in BOOLEAN_FLAGS.items():
        if parameters.get(name) is True:
            command.append(flag)
    return command


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the patched TLS2trees instance stage for one FRDR plot.")
    parser.add_argument("--plot-name", required=True)
    parser.add_argument("--config", default="methods/tls2trees/configs/frdr_benchmark.yml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = utc_now()
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "plot_name": args.plot_name,
        "start_time_utc": start_time,
        "end_time_utc": None,
        "runtime_seconds": None,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "dry_run": args.dry_run,
        "command": None,
        "return_code": None,
        "status": "preflight",
    }
    metadata_path: Path | None = None

    try:
        config, config_path, project_root = load_benchmark_config(args.config)
        plots = config["dataset"]["plots"]
        if args.plot_name not in plots:
            raise ValueError(f"Unknown plot {args.plot_name!r}; expected one of: {', '.join(plots)}")

        converted_root = resolve_from_root(config["outputs"]["converted_root"], project_root)
        predictions_root = resolve_from_root(config["outputs"]["predictions_root"], project_root)
        logs_root = resolve_from_root(config["outputs"]["logs_root"], project_root)
        metadata_root = resolve_from_root(config["outputs"]["run_metadata_root"], project_root)
        repo_path = resolve_from_root(config["method"]["repo_path"], project_root)
        script_path = resolve_from_root(config["method"]["patched_instance_script"], project_root)
        tile_name = str(config["conversion"]["tile_name"])
        tile_filename = str(
            config["conversion"].get(
                "tile_filename", f"{tile_name}.downsample.segmented.ply"
            )
        )
        converted_dir = converted_root / args.plot_name
        tile_path = converted_dir / tile_filename
        tile_index = converted_dir / "tile_index.dat"
        output_dir = predictions_root / args.plot_name
        metadata_path = metadata_root / f"{args.plot_name}_run.json"
        stdout_path = logs_root / f"{args.plot_name}.stdout.log"
        stderr_path = logs_root / f"{args.plot_name}.stderr.log"
        overwrite_enabled = bool(config.get("runtime", {}).get("overwrite", False))

        actual_commit = git_commit(repo_path) if repo_path.is_dir() else None
        expected_commit = config["method"].get("tested_commit")
        payload.update(
            {
                "config_path": str(config_path),
                "project_root": str(project_root),
                "tls2trees_repo": str(repo_path),
                "tls2trees_expected_commit": expected_commit,
                "tls2trees_actual_commit": actual_commit,
                "patched_instance_script": str(script_path),
                "input_ply": str(tile_path),
                "tile_index": str(tile_index),
                "output_directory": str(output_dir),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "overwrite_enabled": overwrite_enabled,
            }
        )

        required_paths = {
            "TLS2trees repository": repo_path,
            "patched instance script": script_path,
            "converted PLY": tile_path,
            "tile index": tile_index,
        }
        missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))
        if config["method"].get("require_tested_commit", True) and actual_commit != expected_commit:
            raise RuntimeError(
                f"TLS2trees commit mismatch: expected {expected_commit}, found {actual_commit}"
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            if not overwrite_enabled:
                raise FileExistsError(
                    "Prediction directory is not empty and runtime.overwrite is false: "
                    f"{output_dir}"
                )
            if predictions_root not in output_dir.parents:
                raise RuntimeError(f"Refusing to replace output outside predictions root: {output_dir}")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)
        command = build_command(
            script_path,
            tile_path,
            tile_index,
            output_dir,
            config["method"].get("instance_parameters", {}),
        )
        payload["command"] = command

        if args.dry_run:
            payload.update({"status": "dry_run", "return_code": 0})
            print("Dry run; command not executed:")
            print(json.dumps(command))
            return_code = 0
        else:
            environment = os.environ.copy()
            package_root = repo_path / "tls2trees"
            existing_pythonpath = environment.get("PYTHONPATH")
            python_paths = [str(repo_path), str(package_root)]
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
            environment["PYTHONPATH"] = os.pathsep.join(python_paths)
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=str(repo_path),
                    env=environment,
                    shell=False,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
            return_code = completed.returncode
            payload.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "peak_memory_gb": parse_peak_memory_gb(stdout_path, stderr_path),
                }
            )
    except Exception as exc:
        payload.update(
            {
                "status": "preflight_failed",
                "return_code": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(payload["error"], file=sys.stderr)
        return_code = 2
    finally:
        payload["end_time_utc"] = utc_now()
        payload["runtime_seconds"] = round(time.perf_counter() - started, 6)
        if metadata_path is None:
            metadata_path = ROOT / "results/metadata/tls2trees_runs" / f"{args.plot_name}_run.json"
        write_metadata(metadata_path, payload)
        print(f"Run metadata: {metadata_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
