"""Resolve one immutable candidate-checkpoint by validation-plot task."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


EXPECTED_EPOCHS = (30, 60, 90, 120, 149)
VALIDATION_TASKS = 25


def resolve(
    finetune_manifest: Path,
    candidate_index: Path,
    task_index: int,
) -> dict[str, Any]:
    if not 0 <= task_index < VALIDATION_TASKS:
        raise ValueError("fine-tune validation task index is outside 0..24")
    data = json.loads(finetune_manifest.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_index.read_text(encoding="utf-8"))
    if (
        data.get("schema") != "forainet_finetune_data_manifest_v1"
        or data.get("status") != "complete"
        or data.get("held_out_access") is not False
        or data.get("held_out_paths_included") is not False
    ):
        raise ValueError("fine-tune data manifest is not complete and test-locked")
    validation_rows = [
        row
        for row in data.get("records", [])
        if row.get("training_role") == "validation"
    ]
    if len(validation_rows) != 5 or any(
        row.get("split") != "dev" for row in validation_rows
    ):
        raise ValueError("fine-tune manifest does not contain exactly five dev-val plots")
    if (
        candidates.get("schema") != "forainet_finetune_candidate_index_v1"
        or candidates.get("status") != "complete"
        or candidates.get("held_out_access") is not False
    ):
        raise ValueError("candidate index is not complete and test-locked")
    candidate_rows = candidates.get("candidates")
    if not isinstance(candidate_rows, list) or [
        int(row.get("epoch", -1)) for row in candidate_rows
    ] != list(EXPECTED_EPOCHS):
        raise ValueError("candidate epochs differ from frozen validation sweep")

    candidate_offset, plot_offset = divmod(task_index, 5)
    candidate = candidate_rows[candidate_offset]
    plot = validation_rows[plot_offset]
    return {
        "validation_task_index": task_index,
        "candidate_epoch": int(candidate["epoch"]),
        "checkpoint_filename": str(candidate["filename"]),
        "checkpoint_sha256": str(candidate["sha256"]),
        "plot_offset": plot_offset,
        "development_task_index": int(plot["task_index"]),
        "relative_path": str(plot["relative_path"]),
        "source_sha256": str(plot["source_sha256"]),
        "point_count": int(plot["source_point_count"]),
    }


def shell_assignments(payload: dict[str, Any]) -> str:
    names = {
        "FORAINET_VALIDATION_TASK_INDEX": "validation_task_index",
        "FORAINET_CANDIDATE_EPOCH": "candidate_epoch",
        "FORAINET_CANDIDATE_FILENAME": "checkpoint_filename",
        "FORAINET_CANDIDATE_SHA256": "checkpoint_sha256",
        "FORAINET_VALIDATION_PLOT_OFFSET": "plot_offset",
        "FORAINET_TASK_DEVELOPMENT_INDEX": "development_task_index",
        "FORAINET_TASK_RELATIVE_PATH": "relative_path",
        "FORAINET_TASK_SOURCE_SHA256": "source_sha256",
        "FORAINET_TASK_POINT_COUNT": "point_count",
    }
    return "\n".join(
        f"{name}={shlex.quote(str(payload[key]))}" for name, key in names.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-manifest", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    args = parser.parse_args()
    payload = resolve(
        args.finetune_manifest,
        args.candidate_index,
        args.task_index,
    )
    print(shell_assignments(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
