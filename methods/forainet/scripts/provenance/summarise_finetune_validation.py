"""Validate five candidates on the fixed five development-validation plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarise_development_run import (  # noqa: E402
    EXPECTED_PROTOCOL,
    aggregate,
    load_json,
    sha256,
    validate_retention,
    write_csv,
)


EXPECTED_EPOCHS = (30, 60, 90, 120, 149)


def select_candidate(
    candidate_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_summaries:
        raise ValueError("candidate summary is empty")
    return min(
        candidate_summaries,
        key=lambda row: (
            -float(row["f1"]),
            int(row["false_positives"]),
            int(row["candidate_epoch"]),
        ),
    )


def summarise(
    validation_root: Path,
    validation_run_id: str,
    finetune_root: Path,
    benchmark_commit: str,
) -> dict[str, Any]:
    if (validation_root / "final_gate.json").exists():
        raise FileExistsError("fine-tune validation final gate already exists")
    data_manifest_path = finetune_root / "finetune_data_manifest.json"
    candidate_index_path = finetune_root / "full" / "candidates" / "index.json"
    full_gate_path = finetune_root / "full" / "final_gate.json"
    for path in (data_manifest_path, candidate_index_path, full_gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = load_json(data_manifest_path)
    index = load_json(candidate_index_path)
    full_gate = load_json(full_gate_path)
    if (
        data.get("status") != "complete"
        or data.get("held_out_access") is not False
        or data.get("held_out_paths_included") is not False
        or index.get("status") != "complete"
        or index.get("held_out_access") is not False
        or full_gate.get("status") != "complete"
        or full_gate.get("held_out_access") is not False
    ):
        raise ValueError("fine-tune evidence is not complete and test-locked")
    validation_rows = [
        row
        for row in data["records"]
        if row.get("training_role") == "validation"
    ]
    candidate_rows = index.get("candidates")
    if len(validation_rows) != 5 or not isinstance(candidate_rows, list):
        raise ValueError("validation inputs are incomplete")
    if [int(row["epoch"]) for row in candidate_rows] != list(EXPECTED_EPOCHS):
        raise ValueError("candidate epochs differ from the frozen sweep")

    plots = []
    child_manifests = []
    for candidate in candidate_rows:
        epoch = int(candidate["epoch"])
        checkpoint_sha256 = str(candidate["sha256"])
        for plot_offset, source in enumerate(validation_rows):
            task_index = int(source["task_index"])
            plot_root = (
                validation_root
                / "candidates"
                / f"epoch_{epoch:03d}"
                / f"plot_{plot_offset:03d}"
            )
            gate_path = plot_root / "final_gate.json"
            metrics_path = plot_root / "evaluation" / "metrics.json"
            plot_metadata_path = plot_root / "metadata" / "plot.json"
            retention_path = plot_root / "retention" / "manifest.json"
            for path in (
                gate_path,
                metrics_path,
                plot_metadata_path,
                retention_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
            gate = load_json(gate_path)
            metrics = load_json(metrics_path)
            metadata = load_json(plot_metadata_path)
            retention = load_json(retention_path)
            validate_retention(plot_root, retention)
            if (
                gate.get("schema")
                != "forainet_finetune_validation_plot_final_gate_v1"
                or gate.get("status") != "complete"
                or gate.get("held_out_access") is not False
                or gate.get("relative_path") != source["relative_path"]
                or gate.get("development_task_index") != task_index
                or gate.get("retention_manifest_sha256")
                != sha256(retention_path)
            ):
                raise ValueError(f"invalid validation gate: {plot_root}")
            if (
                metrics.get("protocol_id") != EXPECTED_PROTOCOL
                or metrics.get("split") != "dev"
                or metrics.get("coordinate_matching") is not False
                or metadata.get("route") != "finetune_validation"
                or metadata.get("checkpoint_kind") != "fine_tuned_on_dev"
                or int(metadata.get("checkpoint_epoch", -1)) != epoch
                or metadata.get("checkpoint_sha256") != checkpoint_sha256
                or metadata.get("benchmark_commit") != benchmark_commit
                or metadata.get("relative_path") != source["relative_path"]
            ):
                raise ValueError(f"validation protocol differs: {plot_root}")
            plots.append(
                {
                    "candidate_epoch": epoch,
                    "checkpoint_sha256": checkpoint_sha256,
                    "plot_offset": plot_offset,
                    "development_task_index": task_index,
                    "relative_path": source["relative_path"],
                    "site": str(source["relative_path"]).split("/", 1)[0],
                    "evaluated_point_count": int(metrics["evaluated_point_count"]),
                    "reference_instances": int(metrics["reference_instance_count"]),
                    "prediction_instances": int(metrics["prediction_instance_count"]),
                    "true_positives": int(metrics["true_positives"]),
                    "false_positives": int(metrics["false_positives"]),
                    "false_negatives": int(metrics["false_negatives"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "aligned_prediction_sha256": metadata[
                        "aligned_prediction_sha256"
                    ],
                    "retention_manifest_sha256": sha256(retention_path),
                }
            )
            child_manifests.append(
                {
                    "candidate_epoch": epoch,
                    "plot_offset": plot_offset,
                    "relative_path": source["relative_path"],
                    "retention_manifest_sha256": sha256(retention_path),
                }
            )
    if len(plots) != 25:
        raise ValueError("fine-tune validation must contain exactly 25 results")

    candidate_summaries = []
    for candidate in candidate_rows:
        epoch = int(candidate["epoch"])
        rows = [row for row in plots if int(row["candidate_epoch"]) == epoch]
        if len(rows) != 5:
            raise ValueError(f"candidate {epoch} lacks five validation plots")
        candidate_summaries.append(
            {
                "candidate_epoch": epoch,
                "checkpoint_filename": candidate["filename"],
                "checkpoint_sha256": candidate["sha256"],
                **aggregate(rows),
            }
        )
    selected = select_candidate(candidate_summaries)

    summary_root = validation_root / "summary"
    retention_root = validation_root / "retention"
    summary_root.mkdir(parents=True, exist_ok=False)
    retention_root.mkdir(parents=True, exist_ok=False)
    plot_csv = summary_root / "plots.csv"
    candidate_csv = summary_root / "candidates.csv"
    selection_json = summary_root / "selection.json"
    write_csv(plot_csv, plots)
    write_csv(candidate_csv, candidate_summaries)
    selection = {
        "schema": "forainet_finetune_validation_selection_v1",
        "status": "selected",
        "run_id": validation_run_id,
        "protocol_id": EXPECTED_PROTOCOL,
        "split": "dev_validation",
        "validation_plot_count": 5,
        "candidate_count": 5,
        "selection_metric": "micro_f1",
        "tie_breakers": ["lower_false_positives", "earlier_epoch"],
        "selected": selected,
        "held_out_access": False,
        "next_gate": "freeze_finetuned_test_readiness",
    }
    selection_json.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    retention = {
        "schema": "forainet_finetune_validation_retention_v1",
        "status": "complete",
        "run_id": validation_run_id,
        "data_manifest_sha256": sha256(data_manifest_path),
        "candidate_index_sha256": sha256(candidate_index_path),
        "full_finetune_gate_sha256": sha256(full_gate_path),
        "plot_summary_sha256": sha256(plot_csv),
        "candidate_summary_sha256": sha256(candidate_csv),
        "selection_sha256": sha256(selection_json),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "child_manifests": child_manifests,
        "held_out_access": False,
    }
    retention_path = retention_root / "manifest.json"
    retention_path.write_text(
        json.dumps(retention, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_gate = {
        "schema": "forainet_finetune_validation_final_gate_v1",
        "status": "complete",
        "run_id": validation_run_id,
        "benchmark_commit": benchmark_commit,
        "selected_epoch": selected["candidate_epoch"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selection_sha256": sha256(selection_json),
        "retention_manifest_sha256": sha256(retention_path),
        "held_out_access": False,
        "next_gate": "freeze_finetuned_test_readiness",
    }
    (validation_root / "final_gate.json").write_text(
        json.dumps(final_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--validation-run-id", required=True)
    parser.add_argument("--finetune-root", required=True, type=Path)
    parser.add_argument("--benchmark-commit", required=True)
    args = parser.parse_args()
    payload = summarise(
        args.validation_root,
        args.validation_run_id,
        args.finetune_root,
        args.benchmark_commit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
