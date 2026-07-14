"""Run one frozen TreeLearn FOR-instance development manifest task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_for_instance_one_plot_smoke as runner
from for_instance_development_common import load_manifest


def read_manifest_row(path: Path, task_index: int) -> dict[str, Any]:
    rows, metadata = load_manifest(path)
    if metadata.get("held_out_test_accessed") is not False:
        raise ValueError("TreeLearn development manifest does not lock held-out test")
    matches = [row for row in rows if int(row["task_index"]) == task_index]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one development manifest row for task {task_index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def write_preflight_failure(
    config_path: str,
    run_id: str,
    task_index: int,
    row: dict[str, str] | None,
    error: Exception,
) -> None:
    config, _ = runner.load_config(config_path)
    metadata_base = runner.resolve_path(config["paths"]["metadata_root"])
    safe_plot_id = (row or {}).get("safe_plot_id", f"task_{task_index:03d}")
    output = metadata_base / run_id / f"{safe_plot_id}_inference.json"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "method": "treelearn",
        "dataset": "for-instance",
        "run_id": run_id,
        "evaluation_scope": "development_full",
        "status": "failed_preflight",
        "return_code": 1,
        "held_out_test_accessed": False,
        "task_index": task_index,
        "plot": {
            "plot_id": (row or {}).get("plot_id"),
            "safe_plot_id": safe_plot_id,
            "relative_path": (row or {}).get("relative_path"),
            "collection": (row or {}).get("collection"),
            "split": (row or {}).get("split", "dev"),
        },
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--treelearn-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--training-mode",
        choices=("published_pretrained", "fine_tuned_on_dev"),
        default="published_pretrained",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    row: dict[str, str] | None = None
    try:
        row = read_manifest_row(manifest, args.task_index)
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        expected_input = (dataset_root / row["relative_path"]).resolve()
        recorded_input = Path(row["input_las"]).expanduser().resolve()
        if recorded_input != expected_input or not expected_input.is_file():
            raise ValueError("Manifest input path does not match the dataset root")
        command = [
            sys.executable,
            str(runner.ROOT / "methods/treelearn/scripts/run_for_instance_one_plot_smoke.py"),
            "--config",
            args.config,
            "--dataset-root",
            str(dataset_root),
            "--treelearn-repo",
            args.treelearn_repo,
            "--checkpoint",
            args.checkpoint,
            "--training-mode",
            args.training_mode,
            "--run-id",
            args.run_id,
            "--relative-path",
            row["relative_path"],
            "--plot-id",
            row["plot_id"],
            "--safe-plot-id",
            row["safe_plot_id"],
            "--expected-split",
            "dev",
            "--expected-point-count",
            str(row["point_count"]),
            "--expected-reference-tree-count",
            str(row["reference_tree_count"]),
            "--expected-input-sha256",
            row["input_sha256"],
            "--expected-split-metadata-sha256",
            row["split_metadata_sha256"],
            "--evaluation-scope",
            "development_full",
        ]
        subprocess.run(command, cwd=runner.ROOT, check=True)
        return 0
    except Exception as exc:
        write_preflight_failure(
            args.config,
            args.run_id,
            args.task_index,
            row,
            exc,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
