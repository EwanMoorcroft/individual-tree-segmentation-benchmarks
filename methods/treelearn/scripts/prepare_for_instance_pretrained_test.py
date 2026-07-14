"""Freeze one authorized TreeLearn clean-pretrained held-out evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

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
from prepare_for_instance_finetuned_test import read_test_metadata


EXPECTED_CHECKPOINT_MD5 = "106a80de2991c5f23484a3f9d03e3b16"
EXPECTED_UPSTREAM_COMMIT = "fd240ce7caa4c444fe3418aca454dc578bc557d4"
SOURCE_URL = (
    "https://data.goettingen-research-online.de/api/access/datafile/"
    ":persistentId?persistentId=doi:10.25625/VPMPID/8CIIW0"
)


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    checkpoint: Path,
    dataset_root: Path,
    metadata_path: Path,
    run_id: str,
) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    metadata_path = metadata_path.expanduser().resolve()
    if not re.fullmatch(
        r"treelearn_for-instance_published_pretrained_[0-9]{8}_[0-9]{6}",
        run_id,
    ):
        raise ValueError("Unexpected TreeLearn pretrained run ID")
    if not checkpoint.is_file() or md5(checkpoint) != EXPECTED_CHECKPOINT_MD5:
        raise ValueError("Clean authors-released checkpoint identity differs")
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
    checkpoint_sha256 = sha256(checkpoint)
    return {
        "schema_version": 1,
        "status": "frozen_for_one_time_held_out_test",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "run_id": run_id,
        "variant": "published_pretrained",
        "training_mode": "published_pretrained",
        "dataset_split": "test",
        "held_out_test_accessed": True,
        "manual_authorisation_recorded": True,
        "repeat_test_for_setting_selection_permitted": False,
        "weight_updates": False,
        "postprocessing_updates": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": 0.5,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_md5": EXPECTED_CHECKPOINT_MD5,
        "checkpoint_provenance": {
            "source_md5": EXPECTED_CHECKPOINT_MD5,
            "source_url": SOURCE_URL,
            "source_dataset_name": "model_weights_finetuned",
            "training_data": (
                "Authors-released noisy-label checkpoint fine-tuned on L1W; "
                "no FOR-instance training"
            ),
        },
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    json_path = Path(args.output_json).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve()
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("One-time TreeLearn pretrained test freeze already exists")
    payload = prepare(
        Path(args.checkpoint),
        Path(args.dataset_root),
        Path(args.metadata_csv),
        args.run_id,
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
    print("training_mode=published_pretrained")
    print("held_out_test_accessed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
