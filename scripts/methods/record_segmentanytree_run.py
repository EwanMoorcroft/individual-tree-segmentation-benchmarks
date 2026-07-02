"""Write one public-structure SegmentAnyTree run metadata record."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def peak_memory_gb(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    prefix = "Maximum resident set size (kbytes):"
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith(prefix):
            values.append(float(line.split(":", maxsplit=1)[1].strip()) / 1024**2)
    return max(values) if values else None


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one SegmentAnyTree run.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--plot-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--prediction-directory", required=True)
    parser.add_argument("--final-prediction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--external-repo", required=True)
    parser.add_argument("--python-userbase", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--return-code", required=True, type=int)
    parser.add_argument("--runtime-seconds", required=True, type=float)
    parser.add_argument("--time-log")
    parser.add_argument("--package-versions-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    repo_path = Path(args.external_repo).expanduser().resolve()
    prediction_directory = Path(args.prediction_directory).expanduser().resolve()
    final_prediction = (
        Path(args.final_prediction).expanduser().resolve()
        if args.final_prediction
        else None
    )
    output_file_count = (
        sum(path.is_file() for path in prediction_directory.rglob("*"))
        if prediction_directory.is_dir()
        else 0
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "for_instance_segmentanytree",
        "dataset": "FOR-instance",
        "method": "SegmentAnyTree",
        "execution_mode": "apptainer_slurm",
        "input_file": str(Path(args.input_file).expanduser().resolve()),
        "relative_path": args.relative_path,
        "collection": args.collection,
        "plot_name": args.plot_name,
        "split": args.split,
        "prediction_directory": str(prediction_directory),
        "final_prediction": str(final_prediction) if final_prediction else None,
        "final_prediction_exists": bool(
            final_prediction and final_prediction.is_file()
        ),
        "output_file_count": output_file_count,
        "container_image": str(Path(args.image).expanduser().resolve()),
        "external_repo_path": str(repo_path),
        "external_commit": git_commit(repo_path),
        "python_userbase": str(Path(args.python_userbase).expanduser().resolve()),
        "package_versions": read_json(
            Path(args.package_versions_json).expanduser().resolve()
            if args.package_versions_json
            else None
        ),
        "runtime_seconds": args.runtime_seconds,
        "peak_memory_gb": peak_memory_gb(
            Path(args.time_log).expanduser().resolve() if args.time_log else None
        ),
        "return_code": args.return_code,
        "status": args.status,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
