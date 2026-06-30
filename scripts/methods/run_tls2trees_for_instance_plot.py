from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = Path(__file__).resolve().parent
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from run_tls2trees_instance_for_plot import (
    build_command,
    git_commit,
    package_versions,
    parse_peak_memory_gb,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_from_root(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(path_text: str) -> tuple[dict[str, Any], Path, Path]:
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
        config["dataset"]["root"]
        config["conversion"]["tile_filename"]
        config["method"]["patched_instance_script"]
        config["outputs"]["converted_root"]
        config["outputs"]["predictions_root"]
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return config, config_path, project_root


def safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be one path component: {value!r}")
    return value


def resolve_plot_path(value: str, dataset_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the patched TLS2trees instance stage for one FOR-instance plot."
    )
    parser.add_argument(
        "--config",
        default="configs/for_instance_tls2trees_accuracy.yml",
    )
    parser.add_argument("--plot-path")
    parser.add_argument("--collection")
    parser.add_argument("--plot-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "start_time_utc": utc_now(),
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
        "peak_memory_gb": None,
    }
    metadata_path: Path | None = None

    try:
        config, config_path, project_root = load_config(args.config)
        dataset_root = Path(config["dataset"]["root"]).expanduser().resolve()
        pilot = config["dataset"]["pilot"]
        plot_value = args.plot_path or pilot["relative_path"]
        plot_path = resolve_plot_path(plot_value, dataset_root)
        collection = safe_component(
            args.collection or pilot["collection"], "collection"
        )
        plot_name = safe_component(
            args.plot_name or Path(plot_value).stem, "plot name"
        )

        converted_root = resolve_from_root(
            config["outputs"]["converted_root"], project_root
        )
        predictions_root = resolve_from_root(
            config["outputs"]["predictions_root"], project_root
        )
        logs_root = resolve_from_root(config["outputs"]["logs_root"], project_root)
        metadata_root = resolve_from_root(
            config["outputs"]["run_metadata_root"], project_root
        )
        repo_path = resolve_from_root(config["method"]["repo_path"], project_root)
        script_path = resolve_from_root(
            config["method"]["patched_instance_script"], project_root
        )
        converted_dir = converted_root / collection / plot_name
        tile_path = converted_dir / config["conversion"]["tile_filename"]
        tile_index = converted_dir / "tile_index.dat"
        output_dir = predictions_root / collection / plot_name
        metadata_path = metadata_root / collection / f"{plot_name}_run.json"
        stdout_path = logs_root / f"{collection}_{plot_name}.stdout.log"
        stderr_path = logs_root / f"{collection}_{plot_name}.stderr.log"
        overwrite = args.overwrite or bool(
            config.get("runtime", {}).get("overwrite", False)
        )

        actual_commit = git_commit(repo_path) if repo_path.is_dir() else None
        expected_commit = config["method"].get("tested_commit")
        payload.update(
            {
                "config_path": str(config_path),
                "project_root": str(project_root),
                "dataset_root": str(dataset_root),
                "plot_path": str(plot_path),
                "collection": collection,
                "plot_name": plot_name,
                "evaluation_mode": config["conversion"]["evaluation_mode"],
                "input_ply": str(tile_path),
                "tile_index": str(tile_index),
                "output_directory": str(output_dir),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "overwrite_enabled": overwrite,
                "tls2trees_repo": str(repo_path),
                "tls2trees_expected_commit": expected_commit,
                "tls2trees_actual_commit": actual_commit,
                "patched_instance_script": str(script_path),
            }
        )

        required_paths = {
            "source plot": plot_path,
            "TLS2trees repository": repo_path,
            "patched instance script": script_path,
            "converted PLY": tile_path,
            "tile index": tile_index,
        }
        missing = [
            f"{label}: {path}"
            for label, path in required_paths.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))
        if (
            config["method"].get("require_tested_commit", True)
            and actual_commit != expected_commit
        ):
            raise RuntimeError(
                f"TLS2trees commit mismatch: expected {expected_commit}, found {actual_commit}"
            )
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                "Prediction directory is not empty; pass --overwrite to replace it: "
                f"{output_dir}"
            )

        command = build_command(
            script_path,
            tile_path,
            tile_index,
            output_dir,
            config["method"]["instance_parameters"],
        )
        payload["command"] = command

        if args.dry_run:
            payload.update({"status": "dry_run", "return_code": 0})
            print("Dry run; command not executed:")
            print(json.dumps(command))
            return_code = 0
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                if predictions_root not in output_dir.parents:
                    raise RuntimeError(
                        f"Refusing to replace output outside predictions root: {output_dir}"
                    )
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            python_paths = [str(repo_path), str(repo_path / "tls2trees")]
            if environment.get("PYTHONPATH"):
                python_paths.append(environment["PYTHONPATH"])
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
                    "peak_memory_gb": parse_peak_memory_gb(
                        stdout_path, stderr_path
                    ),
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
            metadata_path = (
                ROOT
                / "results/metadata/for_instance_tls2trees/runs/preflight_failed.json"
            )
        write_metadata(metadata_path, payload)
        print(f"Run metadata: {metadata_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
