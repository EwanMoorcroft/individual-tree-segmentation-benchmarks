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


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "scripts/data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

try:
    from select_for_instance_plot import select_plot
except ModuleNotFoundError as exc:
    if exc.name != "select_for_instance_plot":
        raise
    raise ModuleNotFoundError(
        "Required plot selector is missing: "
        f"{DATA_DIR / 'select_for_instance_plot.py'}"
    ) from exc


PACKAGE_DISTRIBUTIONS = {
    "laspy": "laspy",
    "numpy": "numpy",
    "pandas": "pandas",
    "torch": "torch",
    "torch_points3d": "torch-points3d",
}
COMMAND_REQUIRED_MESSAGE = (
    "Native SegmentAnyTree execution is unresolved. Use the configured "
    "Apptainer Slurm workflow or provide an explicit method.command_template."
)
SLURM_EXECUTION_MESSAGE = (
    "This benchmark is configured for Apptainer under Slurm. Submit the "
    "displayed sbatch command instead of running the Python wrapper directly."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[name] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_from_root(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(path_text: str) -> tuple[dict[str, Any], Path, Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    try:
        project_root = Path(config["project"]["barkla_root"]).expanduser().resolve()
        config["dataset"]["root"]
        config["method"]["repo_path"]
        config["paths"]["predictions_root"]
        config["paths"]["staged_inputs_root"]
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc
    return config, path, project_root


def git_commit(repo_path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def parse_peak_memory_gb(*paths: Path) -> float | None:
    pattern = re.compile(
        r"peak memory:\s*([0-9]+(?:\.[0-9]+)?)\s*(gib|gb|mib|mb)\b",
        re.IGNORECASE,
    )
    values: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        for match in pattern.finditer(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == "mib":
                value /= 1024
            elif unit == "mb":
                value /= 1000
            values.append(value)
    return max(values) if values else None


def build_command(
    template: Any,
    values: dict[str, str],
) -> list[str]:
    if template is None:
        raise ValueError(COMMAND_REQUIRED_MESSAGE)
    if not isinstance(template, list) or not template or not all(
        isinstance(item, str) for item in template
    ):
        raise ValueError("method.command_template must be a non-empty YAML list")
    try:
        command = [item.format_map(values) for item in template]
    except KeyError as exc:
        raise ValueError(f"Unknown command-template placeholder: {exc}") from exc
    if any(not item for item in command):
        raise ValueError("method.command_template produced an empty argument")
    return command


def build_slurm_command(
    selection: dict[str, Any],
    pilot_relative_path: str | None,
) -> list[str]:
    if selection["relative_path"] == pilot_relative_path:
        script = (
            "scripts/slurm/"
            "run_segmentanytree_for_instance_pilot_apptainer.sbatch"
        )
        return [
            "sbatch",
            "--export=ALL,SEGMENTANYTREE_EXECUTE=1",
            script,
        ]
    return [
        "sbatch",
        f"--array={selection['array_index']}",
        "--export=ALL,SEGMENTANYTREE_EXECUTE=1",
        "scripts/slurm/run_segmentanytree_for_instance_array.sbatch",
    ]


def prepare_empty_directory(
    path: Path,
    root: Path,
    overwrite: bool,
) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlinked managed directory: {path}")
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Managed path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Directory is not empty; pass --overwrite to replace it: {path}"
            )
        if root not in path.parents:
            raise RuntimeError(f"Refusing to replace directory outside managed root: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one configured SegmentAnyTree FOR-instance prediction."
    )
    parser.add_argument(
        "--config",
        default="configs/for_instance_segmentanytree_benchmark.yml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--plot-path")
    parser.add_argument("--split")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "dataset": "FOR-instance",
        "method": "SegmentAnyTree",
        "start_time_utc": utc_now(),
        "end_time_utc": None,
        "runtime_seconds": None,
        "return_code": None,
        "status": "preflight",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "command": None,
        "dry_run": args.dry_run,
        "peak_memory_gb": None,
    }
    metadata_path: Path | None = None

    try:
        config, config_path, project_root = load_config(args.config)
        selection = select_plot(
            config,
            dataset_root_override=args.dataset_root,
            plot_path=args.plot_path,
            selected_split=args.split,
            array_index=args.array_index,
        )
        repo_path = resolve_from_root(config["method"]["repo_path"], project_root)
        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"SegmentAnyTree checkout does not exist: {repo_path}"
            )
        input_path = Path(selection["absolute_path"])
        if not input_path.is_file():
            raise FileNotFoundError(f"Input LAS does not exist: {input_path}")

        collection = selection["collection"]
        plot_name = selection["plot_name"]
        predictions_root = resolve_from_root(
            config["paths"]["predictions_root"], project_root
        )
        staged_root = resolve_from_root(
            config["paths"]["staged_inputs_root"], project_root
        )
        metadata_root = resolve_from_root(
            config["paths"]["run_metadata_root"], project_root
        )
        logs_root = resolve_from_root(config["paths"]["logs_root"], project_root)
        output_dir = predictions_root / collection / plot_name
        staged_input_dir = staged_root / collection / plot_name
        metadata_path = metadata_root / collection / f"{plot_name}_run.json"
        stdout_path = logs_root / f"{collection}_{plot_name}.stdout.log"
        stderr_path = logs_root / f"{collection}_{plot_name}.stderr.log"
        overwrite = args.overwrite or bool(
            config.get("runtime", {}).get("overwrite", False)
        )
        execution_mode = config["method"].get("execution_mode", "native")

        if execution_mode == "apptainer_slurm":
            command = build_slurm_command(
                selection,
                config["dataset"].get("pilot", {}).get("relative_path"),
            )
        else:
            if output_dir.is_symlink():
                raise ValueError(
                    f"Refusing symlinked prediction output directory: {output_dir}"
                )
            if output_dir.exists() and not output_dir.is_dir():
                raise FileExistsError(
                    f"Prediction output path is not a directory: {output_dir}"
                )
            if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
                raise FileExistsError(
                    "Prediction directory is not empty; pass --overwrite to "
                    f"replace it: {output_dir}"
                )
            values = {
                "repo_path": str(repo_path),
                "input_path": str(input_path),
                "input_dir": str(input_path.parent),
                "staged_input_dir": str(staged_input_dir),
                "output_dir": str(output_dir),
                "collection": collection,
                "plot_name": plot_name,
                "split": selection["split"],
            }
            command = build_command(
                config["method"].get("command_template"), values
            )
        external_commit = git_commit(repo_path)
        payload.update(
            {
                "benchmark_name": config["project"]["benchmark_name"],
                "config_path": str(config_path),
                "input_file": str(input_path),
                "relative_path": selection["relative_path"],
                "collection": collection,
                "plot_name": plot_name,
                "split": selection["split"],
                "array_index": selection["array_index"],
                "external_repo_path": str(repo_path),
                "external_commit": external_commit,
                "output_directory": str(output_dir),
                "staged_input_directory": str(staged_input_dir),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "command": command,
                "execution_mode": execution_mode,
                "overwrite_enabled": overwrite,
            }
        )

        if args.dry_run:
            payload.update({"status": "dry_run", "return_code": 0})
            print("Dry run; command not executed:")
            print(json.dumps(command))
            print(f"Input: {input_path}")
            print(f"Output: {output_dir}")
            return_code = 0
        else:
            if execution_mode == "apptainer_slurm":
                raise RuntimeError(SLURM_EXECUTION_MESSAGE)
            prepare_empty_directory(staged_input_dir, staged_root, overwrite)
            prepare_empty_directory(output_dir, predictions_root, overwrite)
            (staged_input_dir / input_path.name).symlink_to(input_path)
            logs_root.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(repo_path)] + ([existing_pythonpath] if existing_pythonpath else [])
            )
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
            output_file_count = sum(
                path.is_file() for path in output_dir.rglob("*")
            )
            if return_code == 0 and output_file_count == 0:
                return_code = 3
                payload["error"] = (
                    "SegmentAnyTree returned zero but produced no output files: "
                    f"{output_dir}"
                )
            payload.update(
                {
                    "return_code": return_code,
                    "status": "completed" if return_code == 0 else "failed",
                    "output_file_count": output_file_count,
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
                / "results/metadata/segmentanytree_for_instance/runs/preflight_failed.json"
            )
        write_metadata(metadata_path, payload)
        print(f"Run metadata: {metadata_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
