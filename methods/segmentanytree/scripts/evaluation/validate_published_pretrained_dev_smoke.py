"""Validate one aligned released-pretrained SegmentAnyTree development smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_SHA256 = (
    "0b4d74b4644e37a16f59008ad0f5c62894fc4d2d906f3abd803bbfc5b5dd803a"
)
EXPECTED_EXTERNAL_COMMIT = "a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_smoke(
    run_metadata_path: Path,
    metrics_path: Path,
    checkpoint_bundle: Path,
    expected_relative_path: str,
    expected_checkpoint_sha256: str,
    expected_external_commit: str,
) -> dict[str, Any]:
    checkpoint = checkpoint_bundle / "PointGroup-PAPER.pt"
    overrides = checkpoint_bundle / ".hydra" / "overrides.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Released checkpoint is missing: {checkpoint}")
    if not overrides.is_file() or overrides.stat().st_size <= 0:
        raise FileNotFoundError(f"Released Hydra overrides are missing: {overrides}")
    actual_checkpoint_sha256 = sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "Released checkpoint SHA-256 mismatch: "
            f"{actual_checkpoint_sha256} != {expected_checkpoint_sha256}"
        )

    run = read_object(run_metadata_path)
    metrics = read_object(metrics_path)
    if run.get("status") != "completed" or int(run.get("return_code", -1)) != 0:
        raise ValueError("Inference run did not complete successfully")
    if run.get("run_type") != "published_pretrained":
        raise ValueError(f"Unexpected run type: {run.get('run_type')}")
    if run.get("split") != "dev" or metrics.get("split") != "dev":
        raise ValueError("Development smoke contains a non-development split")
    if (
        run.get("relative_path") != expected_relative_path
        or metrics.get("relative_path") != expected_relative_path
    ):
        raise ValueError("Development smoke used an unexpected plot")
    if run.get("external_commit") != expected_external_commit:
        raise ValueError(f"Unexpected SegmentAnyTree commit: {run.get('external_commit')}")
    if run.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("Run metadata does not record the released checkpoint hash")
    if not run.get("aligned_instance_evaluation_exists"):
        raise ValueError("Aligned instance evaluation output is missing")
    if not run.get("aligned_semantic_evaluation_exists"):
        raise ValueError("Aligned semantic evaluation output is missing")
    for path_key, checksum_key in (
        ("input_file", "input_sha256"),
        ("aligned_instance_evaluation", "aligned_instance_evaluation_sha256"),
        ("aligned_semantic_evaluation", "aligned_semantic_evaluation_sha256"),
    ):
        aligned_path = Path(str(run.get(path_key, "")))
        recorded_checksum = run.get(checksum_key)
        if not aligned_path.is_file() or not recorded_checksum:
            raise ValueError(f"Run metadata is missing checksum evidence: {path_key}")
        if sha256(aligned_path) != recorded_checksum:
            raise ValueError(f"Recorded checksum mismatch: {aligned_path}")
    if metrics.get("evaluator") != "pointwise_instance_metrics":
        raise ValueError("Unexpected development-smoke evaluator")
    if metrics.get("input_mode") != "internal_aligned_ply":
        raise ValueError("Development smoke did not use aligned internal predictions")
    if int(metrics.get("point_count", 0)) <= 0:
        raise ValueError("Aligned development-smoke output contains no points")
    if int(metrics.get("reference_instance_count", 0)) <= 0:
        raise ValueError("Development smoke contains no reference instances")
    if int(metrics.get("prediction_instance_count", 0)) <= 0:
        raise ValueError("Development smoke contains zero predicted instances")

    return {
        "status": "smoke-tested",
        "benchmark": "for_instance_segmentanytree",
        "method": "SegmentAnyTree",
        "training_mode": "published_pretrained",
        "dataset_split": "dev",
        "relative_path": expected_relative_path,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_bundle_has_hydra_overrides": True,
        "checkpoint_training_data_provenance": (
            "paper_scenario_1_uls_only_exact_plot_manifest_not_bundled"
        ),
        "released_weight_test_overlap_status": (
            "unresolved_do_not_claim_leakage_free"
        ),
        "external_commit": expected_external_commit,
        "point_correspondence": "internal_aligned_ply",
        "input_sha256": run["input_sha256"],
        "aligned_instance_evaluation_sha256": run[
            "aligned_instance_evaluation_sha256"
        ],
        "aligned_semantic_evaluation_sha256": run[
            "aligned_semantic_evaluation_sha256"
        ],
        "point_count": int(metrics["point_count"]),
        "reference_instance_count": int(metrics["reference_instance_count"]),
        "prediction_instance_count": int(metrics["prediction_instance_count"]),
        "held_out_test_accessed": False,
        "accuracy_benchmark_completed": False,
        "next_gate": "manual_review_before_held_out_test_submission",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata": str(run_metadata_path.resolve()),
        "metrics": str(metrics_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one released-pretrained development smoke."
    )
    parser.add_argument("--run-metadata", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint-bundle", required=True)
    parser.add_argument(
        "--expected-relative-path", default="CULS/plot_1_annotated.las"
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-external-commit", default=EXPECTED_EXTERNAL_COMMIT)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_smoke(
        Path(args.run_metadata).expanduser().resolve(),
        Path(args.metrics).expanduser().resolve(),
        Path(args.checkpoint_bundle).expanduser().resolve(),
        args.expected_relative_path,
        args.expected_checkpoint_sha256,
        args.expected_external_commit,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated_smoke={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
