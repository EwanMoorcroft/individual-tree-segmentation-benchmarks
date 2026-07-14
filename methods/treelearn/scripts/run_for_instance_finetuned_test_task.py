"""Run one authorized frozen TreeLearn held-out test task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_for_instance_one_plot_smoke as runner
from for_instance_test_common import load_test_manifest


def read_row(path: Path, task_index: int, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, manifest = load_test_manifest(path)
    if manifest.get("run_id") != run_id:
        raise ValueError("Test manifest belongs to a different run")
    matches = [row for row in rows if int(row["task_index"]) == task_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one test row for task {task_index}")
    return matches[0], manifest


def write_preflight_failure(
    metadata_root: Path,
    run_id: str,
    task_index: int,
    row: dict[str, Any] | None,
    error: Exception,
    training_mode: str | None,
) -> None:
    safe_id = (row or {}).get("safe_plot_id", f"task_{task_index:03d}")
    output = metadata_root / run_id / f"{safe_id}_inference.json"
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "treelearn",
        "dataset": "for-instance",
        "run_id": run_id,
        "training_mode": training_mode,
        "evaluation_scope": "held_out_test",
        "status": "failed_preflight",
        "return_code": 1,
        "held_out_test_accessed": True,
        "task_index": task_index,
        "plot": {
            "plot_id": (row or {}).get("plot_id"),
            "safe_plot_id": safe_id,
            "relative_path": (row or {}).get("relative_path"),
            "collection": (row or {}).get("collection"),
            "split": "test",
        },
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--treelearn-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--tables-root", required=True)
    parser.add_argument("--allow-empty-group-recovery", action="store_true")
    args = parser.parse_args()
    row: dict[str, Any] | None = None
    training_mode: str | None = None
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    try:
        row, manifest = read_row(
            Path(args.manifest).expanduser().resolve(), args.task_index, args.run_id
        )
        training_mode = str(manifest["training_mode"])
        if args.allow_empty_group_recovery and not (
            training_mode == "published_pretrained"
            and args.task_index == 8
            and row["relative_path"] == "SCION/plot_31_annotated.las"
        ):
            raise ValueError("Empty-group recovery is restricted to frozen test task 8")
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if (dataset_root / row["relative_path"]).resolve() != Path(
            row["input_las"]
        ).resolve():
            raise ValueError("Test manifest input path differs from dataset root")
        if checkpoint != Path(manifest["checkpoint"]).resolve():
            raise ValueError("Submitted checkpoint differs from frozen selection")
        provenance = manifest.get("checkpoint_provenance") or {}
        for field in ("source_md5", "source_url", "source_dataset_name"):
            if not provenance.get(field):
                raise ValueError(f"Checkpoint provenance is missing {field}")
        command = [
            sys.executable,
            str(runner.ROOT / "methods/treelearn/scripts/run_for_instance_one_plot_smoke.py"),
            "--config", args.config,
            "--dataset-root", str(dataset_root),
            "--treelearn-repo", args.treelearn_repo,
            "--checkpoint", str(checkpoint),
            "--checkpoint-source-md5", provenance["source_md5"],
            "--checkpoint-source-url", provenance["source_url"],
            "--checkpoint-source-dataset-name", provenance["source_dataset_name"],
            "--training-mode", manifest["training_mode"],
            "--runtime-root", args.runtime_root,
            "--predictions-root", args.predictions_root,
            "--metadata-root", args.metadata_root,
            "--tables-root", args.tables_root,
            "--run-id", args.run_id,
            "--relative-path", row["relative_path"],
            "--plot-id", row["plot_id"],
            "--safe-plot-id", row["safe_plot_id"],
            "--expected-split", "test",
            "--held-out-test-authorized",
            "--expected-point-count", str(row["point_count"]),
            "--expected-reference-tree-count", str(row["reference_tree_count"]),
            "--expected-input-sha256", row["input_sha256"],
            "--expected-split-metadata-sha256", row["split_metadata_sha256"],
            "--evaluation-scope", "held_out_test",
        ]
        if args.allow_empty_group_recovery:
            command.append("--allow-empty-group-recovery")
        subprocess.run(command, cwd=runner.ROOT, check=True)
        return 0
    except Exception as exc:
        write_preflight_failure(
            metadata_root, args.run_id, args.task_index, row, exc, training_mode
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
