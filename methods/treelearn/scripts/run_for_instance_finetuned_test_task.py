"""Run one authorized frozen TreeLearn held-out test task."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_for_instance_one_plot_smoke as runner
from for_instance_test_common import load_test_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_checkpoint(manifest: dict[str, Any], submitted: Path) -> Path:
    checkpoint = submitted.expanduser().resolve()
    frozen = Path(str(manifest.get("checkpoint", ""))).expanduser().resolve()
    if checkpoint != frozen:
        raise ValueError("Submitted checkpoint differs from frozen selection")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        expected_size = int(manifest["checkpoint_size_bytes"])
        expected_sha256 = str(manifest["checkpoint_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frozen checkpoint identity is incomplete") from exc
    if expected_size <= 0 or len(expected_sha256) != 64:
        raise ValueError("Frozen checkpoint identity is invalid")
    if checkpoint.stat().st_size != expected_size:
        raise ValueError("Frozen checkpoint size changed before test task")
    if sha256(checkpoint) != expected_sha256:
        raise ValueError("Frozen checkpoint SHA-256 changed before test task")
    return checkpoint


def validate_dataset_row(dataset_root: Path, row: dict[str, Any]) -> None:
    if (dataset_root / row["relative_path"]).resolve() != Path(
        row["input_las"]
    ).resolve():
        raise ValueError("Test manifest input path differs from dataset root")


def validate_run_id(value: Any) -> str:
    if not isinstance(value, str) or not runner.RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Unsafe TreeLearn run ID: {value!r}")
    return value


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
    safe_run_id = validate_run_id(run_id)
    safe_id = runner.validate_safe_plot_id(
        (row or {}).get("safe_plot_id", f"task_{task_index:03d}")
    )
    output = runner.contained_path(
        metadata_root, safe_run_id, f"{safe_id}_inference.json"
    )
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
    run_id = validate_run_id(args.run_id)
    row: dict[str, Any] | None = None
    training_mode: str | None = None
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    try:
        row, manifest = read_row(
            Path(args.manifest).expanduser().resolve(), args.task_index, run_id
        )
        training_mode = str(manifest["training_mode"])
        if args.allow_empty_group_recovery and not (
            training_mode == "published_pretrained"
            and args.task_index == 8
            and row["relative_path"] == "SCION/plot_31_annotated.las"
        ):
            raise ValueError("Empty-group recovery is restricted to frozen test task 8")
        checkpoint = validate_frozen_checkpoint(manifest, Path(args.checkpoint))
        provenance = manifest.get("checkpoint_provenance") or {}
        for field in ("source_md5", "source_url", "source_dataset_name"):
            if not provenance.get(field):
                raise ValueError(f"Checkpoint provenance is missing {field}")
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        validate_dataset_row(dataset_root, row)
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
            "--run-id", run_id,
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
            metadata_root, run_id, args.task_index, row, exc, training_mode
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
