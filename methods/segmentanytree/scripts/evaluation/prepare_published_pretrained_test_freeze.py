"""Freeze one released-pretrained SegmentAnyTree held-out evaluation."""

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


def freeze_test_evaluation(
    smoke_evidence_path: Path,
    checkpoint_bundle: Path,
    run_id: str,
    accept_unresolved_training_manifest: bool,
) -> dict[str, Any]:
    smoke = read_object(smoke_evidence_path)
    if smoke.get("status") != "smoke-tested":
        raise ValueError("Released-pretrained development smoke has not passed")
    if smoke.get("training_mode") != "published_pretrained":
        raise ValueError("Smoke evidence has an unexpected training mode")
    if smoke.get("dataset_split") != "dev" or smoke.get("held_out_test_accessed"):
        raise ValueError("Smoke evidence does not prove development-only execution")
    if smoke.get("next_gate") != "manual_review_before_held_out_test_submission":
        raise ValueError("Smoke evidence does not require the expected manual gate")
    if int(smoke.get("prediction_instance_count", 0)) <= 0:
        raise ValueError("Smoke evidence contains zero predicted instances")
    if int(smoke.get("reference_instance_count", 0)) <= 0:
        raise ValueError("Smoke evidence contains zero reference instances")
    if smoke.get("external_commit") != EXPECTED_EXTERNAL_COMMIT:
        raise ValueError("Smoke evidence records an unexpected upstream commit")
    if smoke.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Smoke evidence records an unexpected checkpoint hash")
    if (
        smoke.get("released_weight_test_overlap_status")
        != "unresolved_do_not_claim_leakage_free"
    ):
        raise ValueError("Unexpected released-weight provenance status")
    if not accept_unresolved_training_manifest:
        raise ValueError("Explicit acceptance of the unresolved manifest is required")

    checkpoint = checkpoint_bundle / "PointGroup-PAPER.pt"
    overrides = checkpoint_bundle / ".hydra" / "overrides.yaml"
    if not checkpoint.is_file() or sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen released checkpoint is missing or has changed")
    if not overrides.is_file() or overrides.stat().st_size <= 0:
        raise ValueError("Frozen released Hydra overrides are missing")
    overrides_text = overrides.read_text(encoding="utf-8")
    required_overrides = {
        "- task=panoptic",
        "- data=panoptic/treeins_rad8",
        "- models=panoptic/area4_ablation_3heads_5",
        "- model_name=PointGroup-PAPER",
        "- training=treeins",
        "- job_name=mls_data_run",
        "- batch_size=6",
        "- epochs=100",
    }
    missing = sorted(required_overrides - set(overrides_text.splitlines()))
    if missing:
        raise ValueError(f"Released Hydra overrides changed or are incomplete: {missing}")

    return {
        "status": "frozen_for_one_time_held_out_evaluation",
        "run_id": run_id,
        "benchmark": "for_instance_segmentanytree",
        "method": "SegmentAnyTree",
        "training_mode": "published_pretrained",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "hydra_overrides": str(overrides.resolve()),
        "hydra_overrides_sha256": sha256(overrides),
        "released_checkpoint_job_name": "mls_data_run",
        "checkpoint_training_data_provenance": (
            "released_mls_checkpoint_exact_training_plot_manifest_not_bundled"
        ),
        "released_weight_test_overlap_status": (
            "low_risk_unresolved_do_not_claim_leakage_free"
        ),
        "unresolved_training_manifest_accepted": True,
        "external_commit": EXPECTED_EXTERNAL_COMMIT,
        "development_smoke_evidence": str(smoke_evidence_path.resolve()),
        "development_smoke_prediction_instances": int(
            smoke["prediction_instance_count"]
        ),
        "test_split": "test",
        "expected_test_plots": 11,
        "weight_updates": False,
        "postprocessing_updates": False,
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": 0.5,
        "point_correspondence": "internal_aligned_ply",
        "repeat_test_for_setting_selection_permitted": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a released-pretrained held-out evaluation."
    )
    parser.add_argument("--smoke-evidence", required=True)
    parser.add_argument("--checkpoint-bundle", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--accept-unresolved-training-manifest", action="store_true"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = freeze_test_evaluation(
        Path(args.smoke_evidence).expanduser().resolve(),
        Path(args.checkpoint_bundle).expanduser().resolve(),
        args.run_id,
        args.accept_unresolved_training_manifest,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Freeze manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"freeze_manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
