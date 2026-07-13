"""Final gate for the one-time TreeLearn fine-tuned held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from for_instance_test_common import EXPECTED_TEST_PLOTS
from summarise_for_instance_development import artifact_entry, sha256


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-freeze", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--final-summary", required=True, type=Path)
    parser.add_argument("--retention-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-benchmark-commit", required=True)
    args = parser.parse_args()
    freeze = load(args.test_freeze)
    run = load(args.run_summary)
    retention = load(args.retention_manifest)
    expected_run_id = freeze.get("source_long_run_id")
    expected_checkpoint = freeze.get("checkpoint_sha256")
    expected_run = {
        "status": "completed_aligned_pointwise_test",
        "method": "TreeLearn",
        "variant": "fine_tuned_on_dev_long_epoch_35",
        "run_id": expected_run_id,
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "benchmark_commit": args.expected_benchmark_commit,
        "checkpoint_sha256": expected_checkpoint,
        "expected_plots": EXPECTED_TEST_PLOTS,
        "completed_plots": EXPECTED_TEST_PLOTS,
        "documented_failures": 0,
        "retention_status": "retention_verified",
        "next_gate": "treelearn_benchmark_complete",
    }
    for field, value in expected_run.items():
        if run.get(field) != value:
            raise ValueError(f"Run summary has unexpected {field}")
    expected_retention = {
        "status": "retention_verified",
        "run_id": expected_run_id,
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "checkpoint_sha256": expected_checkpoint,
        "expected_plots": EXPECTED_TEST_PLOTS,
        "completed_plots": EXPECTED_TEST_PLOTS,
        "documented_failures": 0,
        "verified_prediction_file_count": EXPECTED_TEST_PLOTS * 5,
        "complete_test_prediction_set_retained": True,
    }
    for field, value in expected_retention.items():
        if retention.get(field) != value:
            raise ValueError(f"Retention manifest has unexpected {field}")
    if len(retention.get("plots") or []) != EXPECTED_TEST_PLOTS:
        raise ValueError("Retention manifest does not inventory all test plots")
    for plot in retention["plots"]:
        files = plot.get("prediction_files") or []
        if len(files) != 5 or plot.get("retention_verified") is not True:
            raise ValueError("Per-plot prediction retention is incomplete")
        for item in files:
            path = Path(item["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(item["size_bytes"])
                or sha256(path) != item["sha256"]
            ):
                raise ValueError(f"Retained prediction changed: {path}")
    with args.final_summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Final summary must contain exactly one row")
    result = rows[0]
    expected_csv = {
        "method": "TreeLearn",
        "variant": "fine_tuned_on_dev_long_epoch_35",
        "dataset_split": "test",
        "site": "ALL",
        "expected_plots": str(EXPECTED_TEST_PLOTS),
        "completed_plots": str(EXPECTED_TEST_PLOTS),
        "failed_plots": "0",
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "result_status": "completed_aligned_pointwise_test",
        "held_out_test_accessed": "true",
    }
    for field, value in expected_csv.items():
        if result.get(field) != value:
            raise ValueError(f"Final result has unexpected {field}")
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "schema_version": 1,
        "status": "completed_treelearn_finetuned_held_out_test",
        "run_id": expected_run_id,
        "method": "TreeLearn",
        "variant": "fine_tuned_on_dev_long_epoch_35",
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "repeat_test_for_setting_selection_permitted": False,
        "checkpoint_sha256": expected_checkpoint,
        "completed_plots": EXPECTED_TEST_PLOTS,
        "verified_prediction_files": EXPECTED_TEST_PLOTS * 5,
        "run_summary": artifact_entry(args.run_summary),
        "final_summary": artifact_entry(args.final_summary),
        "retention_manifest": artifact_entry(args.retention_manifest),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "next_gate": "update_public_results_from_frozen_outputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"status={payload['status']}")
    print(f"completion_gate={args.output.resolve()}")
    print(f"verified_prediction_files={payload['verified_prediction_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
