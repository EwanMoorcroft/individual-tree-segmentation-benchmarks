"""Freeze the authorized one-time TreeLearn fine-tuned held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from for_instance_development_common import MANIFEST_FIELDS, plot_id, safe_plot_id
from for_instance_test_common import (
    EXPECTED_TEST_PATHS,
    EXPECTED_TEST_PLOTS,
    EXPECTED_TEST_POINTS,
    EXPECTED_TEST_REFERENCE_TREES,
    EXPECTED_TEST_SITE_COUNTS,
    EXPECTED_TEST_SITE_POINTS,
    EXPECTED_TEST_SITE_REFERENCE_TREES,
    validate_test_rows,
)
from prepare_for_instance_development_manifest import inspect_las, sha256


EXPECTED_INITIAL_MD5 = "106a80de2991c5f23484a3f9d03e3b16"
EXPECTED_UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_test_metadata(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    found: dict[str, str] = {}
    for row in rows:
        if (row.get("split") or "").strip() != "test":
            continue
        relative = (row.get("path") or "").strip().replace("\\", "/")
        folder = (row.get("folder") or "").strip()
        if relative in EXPECTED_TEST_PATHS:
            if relative in found:
                raise ValueError(f"Duplicate test metadata path: {relative}")
            if folder != Path(relative).parts[0]:
                raise ValueError(f"Test metadata folder mismatch for {relative}")
            found[relative] = folder
    if set(found) != set(EXPECTED_TEST_PATHS):
        raise ValueError("Official split metadata does not contain the frozen test subset")
    return found


def prepare(
    selected_freeze_path: Path,
    dataset_root: Path,
    metadata_path: Path,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    selected_freeze_path = selected_freeze_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    selected = load_object(selected_freeze_path)
    expected_selection = {
        "status": "frozen_selected_checkpoint_pending_manual_held_out_test_authorisation",
        "method": "TreeLearn",
        "training_mode": "fine_tuned_on_dev",
        "held_out_test_accessed": False,
        "test_jobs_submitted": 0,
        "selected_config_id": "full_lr_1e-5",
        "selected_epoch": 35,
        "selected_seed": 42,
        "training_plots": 16,
        "initial_checkpoint_md5": EXPECTED_INITIAL_MD5,
        "next_gate": "manual_review_before_any_held_out_test_submission",
    }
    for field, value in expected_selection.items():
        if selected.get(field) != value:
            raise ValueError(f"Selected checkpoint freeze has unexpected {field}")
    if expected_run_id is not None and selected.get("source_long_run_id") != expected_run_id:
        raise ValueError("Selected checkpoint freeze belongs to a different long run")
    checkpoint = Path(str(selected["checkpoint"])).expanduser().resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(selected["checkpoint_size_bytes"])
        or sha256(checkpoint) != selected["checkpoint_sha256"]
    ):
        raise ValueError("Selected checkpoint identity changed before test")
    if not dataset_root.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError("FOR-instance dataset or split metadata is missing")
    metadata_rows = read_test_metadata(metadata_path)
    split_hash = sha256(metadata_path)
    plots = []
    for task_index, relative in enumerate(EXPECTED_TEST_PATHS):
        input_las = (dataset_root / relative).resolve()
        try:
            input_las.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(f"Test path escapes dataset root: {relative}") from exc
        if not input_las.is_file():
            raise FileNotFoundError(input_las)
        point_count, reference_tree_count = inspect_las(input_las)
        identifier = plot_id(relative)
        plots.append(
            {
                "task_index": task_index,
                "plot_id": identifier,
                "safe_plot_id": safe_plot_id(identifier),
                "relative_path": relative,
                "collection": metadata_rows[relative],
                "split": "test",
                "input_las": str(input_las),
                "point_count": point_count,
                "reference_tree_count": reference_tree_count,
                "input_sha256": sha256(input_las),
                "split_metadata": str(metadata_path),
                "split_metadata_sha256": split_hash,
            }
        )
    plots = validate_test_rows(plots)
    return {
        "schema_version": 1,
        "status": "frozen_for_one_time_held_out_test",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "training_mode": "fine_tuned_on_dev",
        "source_long_run_id": selected["source_long_run_id"],
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "manual_authorisation_recorded": True,
        "test_jobs_submitted_at_freeze": 0,
        "repeat_test_for_setting_selection_permitted": False,
        "weight_updates": False,
        "postprocessing_updates": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": 0.5,
        "selected_checkpoint_freeze": str(selected_freeze_path),
        "selected_checkpoint_freeze_sha256": sha256(selected_freeze_path),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "initial_checkpoint_md5": selected["initial_checkpoint_md5"],
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "dataset_root": str(dataset_root),
        "split_metadata": str(metadata_path),
        "split_metadata_sha256": split_hash,
        "expected_test_plot_count": EXPECTED_TEST_PLOTS,
        "expected_site_counts": EXPECTED_TEST_SITE_COUNTS,
        "expected_site_points": EXPECTED_TEST_SITE_POINTS,
        "expected_site_reference_trees": EXPECTED_TEST_SITE_REFERENCE_TREES,
        "expected_total_points": EXPECTED_TEST_POINTS,
        "expected_reference_tree_count": EXPECTED_TEST_REFERENCE_TREES,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plots": plots,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-freeze", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    json_path = Path(args.output_json).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve()
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("One-time TreeLearn test freeze already exists")
    payload = prepare(
        Path(args.selected_freeze),
        Path(args.dataset_root),
        Path(args.metadata_csv),
        args.expected_run_id,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in MANIFEST_FIELDS}
            for row in payload["plots"]
        )
    print(f"test_freeze={json_path}")
    print(f"test_manifest={csv_path}")
    print(f"held_out_test_plots={len(payload['plots'])}")
    print("held_out_test_accessed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
