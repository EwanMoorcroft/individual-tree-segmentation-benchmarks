"""Record reproducibility metadata for one SegmentAnyTree training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path | None, chunk_size: int = 1024 * 1024) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def parse_time_log(path: Path | None) -> dict[str, Any]:
    text = read_text(path)
    if not text:
        return {}
    result: dict[str, Any] = {"time_log": str(path)}
    patterns = {
        "maximum_resident_set_kb": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "elapsed_wall_clock": r"Elapsed \(wall clock\) time.*:\s*(\S+)",
        "cpu_percent": r"Percent of CPU this job got:\s*(\S+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value: Any = match.group(1)
            if key == "maximum_resident_set_kb":
                value = int(value)
            result[key] = value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write metadata for a SegmentAnyTree training job."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--run-type",
        choices=("pilot_training", "full_training"),
        required=True,
    )
    parser.add_argument(
        "--training-mode",
        choices=("retrained_from_dev", "fine_tuned_on_dev"),
        required=True,
    )
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--training-data-root", required=True)
    parser.add_argument("--training-output-root", required=True)
    parser.add_argument("--external-repo", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python-userbase", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--package-versions-json")
    parser.add_argument("--time-log")
    parser.add_argument("--checkpoint")
    parser.add_argument("--requested-epochs", type=int, required=True)
    parser.add_argument("--hydra-stop-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--status", choices=("completed", "failed"), required=True)
    parser.add_argument("--return-code", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    )
    external_repo = Path(args.external_repo).expanduser().resolve()
    image = Path(args.image).expanduser().resolve()
    package_path = (
        Path(args.package_versions_json).expanduser().resolve()
        if args.package_versions_json
        else None
    )
    time_path = (
        Path(args.time_log).expanduser().resolve() if args.time_log else None
    )
    command_path = Path(args.command_file).expanduser().resolve()
    manifest = read_json(manifest_path)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "run_type": args.run_type,
        "training_mode": args.training_mode,
        "profile": args.profile,
        "dataset": "FOR-instance",
        "training_dataset_split": "dev",
        "test_data_used_for_training": False,
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256(manifest_path),
        "split_summary": (
            {
                "dataset_role_counts": manifest.get("dataset_role_counts"),
                "selected_role_counts": manifest.get("selected_role_counts"),
            }
            if isinstance(manifest, dict)
            else None
        ),
        "training_data_root": str(
            Path(args.training_data_root).expanduser().resolve()
        ),
        "training_output_root": str(
            Path(args.training_output_root).expanduser().resolve()
        ),
        "external_repo": str(external_repo),
        "external_commit": git_commit(external_repo),
        "container_image": str(image),
        "container_image_size_bytes": (
            image.stat().st_size if image.is_file() else None
        ),
        "python_userbase": str(
            Path(args.python_userbase).expanduser().resolve()
        ),
        "command": read_text(command_path),
        "package_versions": read_json(package_path),
        "requested_epochs": args.requested_epochs,
        "hydra_stop_epoch": args.hydra_stop_epoch,
        "batch_size": args.batch_size,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
        "checkpoint_sha256": sha256(checkpoint),
        "status": args.status,
        "return_code": args.return_code,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "resources": parse_time_log(time_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
