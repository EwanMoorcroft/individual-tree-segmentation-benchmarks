"""Validate the final TreeLearn full-development summary gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def validate(
    run_summary_path: Path,
    expected_benchmark_commit: str | None = None,
) -> dict[str, Any]:
    run_summary_path = run_summary_path.expanduser().resolve()
    if not run_summary_path.is_file():
        raise FileNotFoundError(run_summary_path)
    summary = load_object(run_summary_path)
    expected = {
        "status": "completed_aligned_pointwise_development",
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "variant": "published_pretrained",
        "dataset_split": "dev",
        "held_out_test_accessed": False,
        "expected_plots": 21,
        "completed_plots": 21,
        "documented_failures": 0,
        "retention_status": "retention_verified",
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(
                f"Development gate {field} differs: "
                f"expected {value!r}, found {summary.get(field)!r}"
            )

    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Development run summary has no frozen provenance")
    expected_provenance = {
        "training_mode": "published_pretrained",
        "checkpoint_md5": "56a3d78f689ae7f1190906b975700311",
        "checkpoint_sha256": (
            "5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"
        ),
        "upstream_commit": "fd240ce7caa4c444fe3418aca454dc578bc557d4",
        "validated_completed_inference_records": 21,
    }
    for field, value in expected_provenance.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"Development provenance {field} differs: "
                f"expected {value!r}, found {provenance.get(field)!r}"
            )
    benchmark_commit = str(provenance.get("benchmark_commit", ""))
    if len(benchmark_commit) != 40 or any(
        char not in "0123456789abcdef" for char in benchmark_commit
    ):
        raise ValueError("Development provenance benchmark commit is not a full SHA-1")
    if (
        expected_benchmark_commit is not None
        and benchmark_commit != expected_benchmark_commit
    ):
        raise ValueError("Development benchmark commit differs from the submitted commit")
    if provenance.get("expected_benchmark_commit") not in (
        None,
        expected_benchmark_commit,
    ):
        raise ValueError("Development summary records a different expected benchmark commit")

    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Development run summary has no output inventory")
    required = {
        "plot_summary",
        "site_summary",
        "development_summary",
        "failures",
        "matches",
        "unmatched_predictions",
        "unmatched_references",
        "retention_manifest",
    }
    if not required.issubset(outputs):
        raise ValueError(
            f"Development output inventory is missing {sorted(required - set(outputs))}"
        )
    verified: dict[str, Path] = {}
    for name in sorted(required):
        entry = outputs[name]
        path = Path(str(entry.get("path", ""))).expanduser().resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("size_bytes", -1))
            or sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"Development summary artefact failed verification: {name}")
        verified[name] = path

    retention = load_object(verified["retention_manifest"])
    retention_expected = {
        "status": "retention_verified",
        "dataset_split": "dev",
        "held_out_test_accessed": False,
        "expected_plots": 21,
        "completed_plots": 21,
        "documented_failures": 0,
        "inference_outputs_retained": 21,
        "verified_prediction_file_count": 105,
        "all_completed_prediction_retention_verified": True,
        "complete_development_prediction_set_retained": True,
    }
    for field, value in retention_expected.items():
        if retention.get(field) != value:
            raise ValueError(
                f"Retention gate {field} differs: "
                f"expected {value!r}, found {retention.get(field)!r}"
            )
    if retention.get("provenance") != provenance:
        raise ValueError("Retention and run-summary provenance differ")

    with verified["site_summary"].open(encoding="utf-8", newline="") as handle:
        site_rows = list(csv.DictReader(handle))
    if {row.get("site") for row in site_rows} != {
        "CULS",
        "NIBIO",
        "RMIT",
        "SCION",
        "TUWIEN",
    }:
        raise ValueError("Development site summary does not contain the five target sites")
    if any(
        row.get("dataset_split") != "dev"
        or row.get("held_out_test_accessed") != "false"
        for row in site_rows
    ):
        raise ValueError("Development site summary is not explicitly test-locked")

    with verified["development_summary"].open(
        encoding="utf-8", newline=""
    ) as handle:
        development_rows = list(csv.DictReader(handle))
    if len(development_rows) != 1 or development_rows[0].get("site") != "ALL":
        raise ValueError("Development summary must contain exactly one ALL row")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--expected-benchmark-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(Path(args.run_summary), args.expected_benchmark_commit)
    print(f"validated_development_run={summary['run_id']}")
    print("status=completed_aligned_pointwise_development")
    print("completed_plots=21/21")
    print("retained_prediction_files=105")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
