"""Evaluate or document one authorized TreeLearn held-out test task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import run_for_instance_one_plot_smoke as runner
from for_instance_test_common import load_test_manifest


def write_failure(
    path: Path,
    args: argparse.Namespace,
    row: dict[str, Any],
    status: str,
    error: Exception | None,
    inference: dict[str, Any] | None,
) -> None:
    payload = {
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "run_id": args.run_id,
        "task_index": args.task_index,
        "plot_id": row["plot_id"],
        "relative_path": row["relative_path"],
        "collection": row["collection"],
        "split": "test",
        "dataset_split": "test",
        "status": status,
        "held_out_test_accessed": True,
        "inference_status": (inference or {}).get("status"),
        "inference_error": (inference or {}).get("error"),
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
        "next_gate": "resolve_documented_test_failure_without_model_selection",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--tables-root", required=True)
    args = parser.parse_args()
    rows, manifest = load_test_manifest(Path(args.manifest).expanduser().resolve())
    if manifest.get("source_long_run_id") != args.run_id:
        raise ValueError("Test manifest belongs to a different long run")
    matches = [row for row in rows if row["task_index"] == args.task_index]
    if len(matches) != 1:
        raise ValueError(f"Expected one test row for task {args.task_index}")
    row = matches[0]
    safe_id = row["safe_plot_id"]
    prediction = (
        Path(args.predictions_root).expanduser().resolve()
        / args.run_id
        / safe_id
        / f"{safe_id}_treelearn_test_predictions.npz"
    )
    inference_path = (
        Path(args.metadata_root).expanduser().resolve()
        / args.run_id
        / f"{safe_id}_inference.json"
    )
    final_root = (
        Path(args.tables_root).expanduser().resolve()
        / args.run_id
        / "per_plot"
        / safe_id
    )
    if final_root.exists():
        raise FileExistsError(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(
            prefix=f".{safe_id}.partial.{os.environ.get('SLURM_JOB_ID', 'manual')}.",
            dir=final_root.parent,
        )
    )
    inference = None
    if inference_path.is_file():
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if not inference or inference.get("status") != "completed":
        write_failure(
            partial / "status.json",
            args,
            row,
            "documented_inference_failure",
            None,
            inference,
        )
    else:
        command = [
            sys.executable,
            str(runner.ROOT / "methods/treelearn/scripts/evaluate_for_instance_one_plot_smoke.py"),
            "--prediction-npz", str(prediction),
            "--inference-metadata", str(inference_path),
            "--run-id", args.run_id,
            "--plot-id", row["plot_id"],
            "--relative-path", row["relative_path"],
            "--split", "test",
            "--metrics-json", str(partial / "metrics.json"),
            "--harmonized-matches-csv", str(partial / "matches.csv"),
            "--unmatched-predictions-csv", str(partial / "unmatched_predictions.csv"),
            "--unmatched-references-csv", str(partial / "unmatched_references.csv"),
            "--evaluation-scope", "held_out_test",
        ]
        try:
            subprocess.run(command, cwd=runner.ROOT, check=True)
        except Exception as exc:
            write_failure(
                partial / "status.json",
                args,
                row,
                "documented_evaluation_failure",
                exc,
                inference,
            )
    if final_root.exists():
        raise FileExistsError(final_root)
    partial.replace(final_root)
    print(f"held_out_test_task={args.task_index}")
    print(f"evaluation_root={final_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
