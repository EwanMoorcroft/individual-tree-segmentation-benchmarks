"""Evaluate or document one TreeLearn development-array task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import run_for_instance_one_plot_smoke as runner
from run_for_instance_development_task import read_manifest_row, validate_run_id


def write_failure(
    output: Path,
    args: argparse.Namespace,
    row: dict[str, str],
    status: str,
    error: Exception | None,
    inference_metadata: dict[str, object] | None,
) -> None:
    payload = {
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "run_id": args.run_id,
        "task_index": args.task_index,
        "plot_id": row["plot_id"],
        "relative_path": row["relative_path"],
        "collection": row["collection"],
        "split": "dev",
        "dataset_split": "dev",
        "status": status,
        "held_out_test_accessed": False,
        "inference_status": (inference_metadata or {}).get("status"),
        "inference_error": (inference_metadata or {}).get("error"),
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
        "next_gate": "resolve_development_failure_before_any_test_route",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = validate_run_id(args.run_id)
    args.run_id = run_id
    config, _ = runner.load_config(args.config)
    row = read_manifest_row(
        Path(args.manifest).expanduser().resolve(), args.task_index
    )
    safe_plot_id = runner.validate_safe_plot_id(row["safe_plot_id"])
    predictions_base = runner.resolve_path(config["paths"]["predictions_root"])
    metadata_base = runner.resolve_path(config["paths"]["metadata_root"])
    tables_base = runner.resolve_path(config["paths"]["tables_root"])
    prediction = runner.contained_path(
        predictions_base,
        run_id,
        safe_plot_id,
        f"{safe_plot_id}_treelearn_development_predictions.npz",
    )
    inference_path = runner.contained_path(
        metadata_base, run_id, f"{safe_plot_id}_inference.json"
    )
    final_root = runner.contained_path(
        tables_base, run_id, "per_plot", safe_plot_id
    )
    if final_root.exists():
        raise FileExistsError(f"Refusing existing evaluation root: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f".{safe_plot_id}.partial.{os.environ.get('SLURM_JOB_ID', 'manual')}.",
            dir=final_root.parent,
        )
    )

    inference_metadata = None
    if inference_path.is_file():
        inference_metadata = json.loads(inference_path.read_text(encoding="utf-8"))
    if not inference_metadata or inference_metadata.get("status") != "completed":
        write_failure(
            partial_root / "status.json",
            args,
            row,
            "documented_inference_failure",
            None,
            inference_metadata,
        )
    else:
        command = [
            sys.executable,
            str(
                runner.ROOT
                / "methods/treelearn/scripts/evaluate_for_instance_one_plot_smoke.py"
            ),
            "--prediction-npz",
            str(prediction),
            "--inference-metadata",
            str(inference_path),
            "--run-id",
            run_id,
            "--plot-id",
            row["plot_id"],
            "--relative-path",
            row["relative_path"],
            "--split",
            "dev",
            "--metrics-json",
            str(partial_root / "metrics.json"),
            "--harmonized-matches-csv",
            str(partial_root / "matches.csv"),
            "--unmatched-predictions-csv",
            str(partial_root / "unmatched_predictions.csv"),
            "--unmatched-references-csv",
            str(partial_root / "unmatched_references.csv"),
            "--evaluation-scope",
            "development_full",
        ]
        try:
            subprocess.run(command, cwd=runner.ROOT, check=True)
        except Exception as exc:
            write_failure(
                partial_root / "status.json",
                args,
                row,
                "documented_evaluation_failure",
                exc,
                inference_metadata,
            )
    if final_root.exists():
        raise FileExistsError(f"Evaluation root appeared concurrently: {final_root}")
    partial_root.replace(final_root)
    print(f"development_task={args.task_index}")
    print(f"evaluation_root={final_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
