"""Validate the automated gates for one published/default development smoke."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods" / "tls2trees" / "scripts" / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from for_instance_published_common import (
    EXPECTED_SPLIT,
    EXPECTED_VARIANT,
    resolve_plot_context,
    sha256,
    utc_now,
    write_json,
)


TARGETS = ("leaf_off", "leaf_on")
STAGES = ("semantic", "instance", "adapter")


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def validate_smoke(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str = EXPECTED_VARIANT,
    split: str = EXPECTED_SPLIT,
) -> dict[str, Any]:
    plot_root, row = resolve_plot_context(
        manifest_path=manifest_path,
        task_index=task_index,
        output_root=output_root,
        run_id=run_id,
        variant=variant,
        split=split,
    )
    output_path = plot_root / "metadata" / "smoke_gate.json"
    summary_csv = plot_root / "evaluation" / "smoke_metrics.csv"
    if output_path.exists() or summary_csv.exists():
        raise FileExistsError(
            f"Smoke-gate output already exists; use a new run_id: {output_path}"
        )

    conversion = load_json(
        plot_root / "converted" / "conversion_metadata.json", "conversion metadata"
    )
    if conversion.get("status") != "prepared" or conversion.get("split") != split:
        raise ValueError("Conversion metadata is not a prepared development artefact")
    if conversion.get("labels_stripped") is not True:
        raise ValueError("Conversion passed reference labels to the method")
    if conversion.get("coordinate_frame", {}).get("maximum_round_trip_delta_m") != 0.0:
        raise ValueError("Conversion coordinate round trip is not exact")

    stage_metadata: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        metadata = load_json(
            plot_root / "metadata" / f"{stage}_run.json", f"{stage} metadata"
        )
        accepted_status = (
            {"completed", "completed_no_predictions"}
            if stage == "instance"
            else {"completed"}
        )
        if metadata.get("status") not in accepted_status:
            raise ValueError(
                f"{stage} status is not accepted: {metadata.get('status')!r}"
            )
        if metadata.get("held_out_test_accessed") is not False:
            raise ValueError(f"{stage} metadata does not prove held-out isolation")
        stage_metadata[stage] = metadata

    instance_inventory = stage_metadata["instance"].get("prediction_inventory", {})
    for target in TARGETS:
        for record in instance_inventory.get(target, []):
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Retained raw prediction is missing or changed: {path}")

    target_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_root = plot_root / "predictions" / "aligned" / target
        alignment = load_json(
            target_root / "alignment_metadata.json", f"{target} alignment metadata"
        )
        if alignment.get("status") != "passed":
            raise ValueError(f"{target} source-row projection did not pass")
        aligned_npz = Path(alignment["aligned_prediction_npz"])
        if not aligned_npz.is_file() or sha256(aligned_npz) != alignment[
            "aligned_prediction_npz_sha256"
        ]:
            raise RuntimeError(f"{target} aligned prediction is missing or changed")

        result = load_json(
            plot_root / "evaluation" / target / "plot_metrics.json",
            f"{target} evaluation",
        )
        if result.get("status") != "evaluated" or result.get("safe_for_scoring") is not True:
            raise ValueError(f"{target} evaluation is not safe for scoring")
        if result.get("evaluator") != (
            "for_instance_tls2trees_source_row_class3_ignore"
        ):
            raise ValueError(f"{target} did not use the source-row evaluator")
        if result.get("semantic_ignore", {}).get("ignored_semantic_classes") != [3]:
            raise ValueError(f"{target} did not apply the class-3 evaluation mask")
        if result.get("target") != target or result.get("split") != "dev":
            raise ValueError(f"{target} evaluation identity does not match the smoke")
        if int(result.get("prediction_instance_count", 0)) <= 0:
            raise ValueError(
                f"{target} emitted no instances; the target route is not smoke-validated"
            )
        target_results[target] = result
        rows.append(
            {
                "target": target,
                "prediction_instance_count": result["prediction_instance_count"],
                "reference_instance_count": result["reference_instance_count"],
                "true_positives": result["true_positives"],
                "false_positives": result["false_positives"],
                "false_negatives": result["false_negatives"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "mean_matched_iou": result["mean_matched_iou"],
                "mean_unweighted_coverage": result["mean_unweighted_coverage"],
                "mean_weighted_coverage": result["mean_weighted_coverage"],
                "evaluated_point_count": result["evaluated_point_count"],
                "oversegmented_reference_count": result[
                    "oversegmented_reference_count"
                ],
                "undersegmented_prediction_count": result[
                    "undersegmented_prediction_count"
                ],
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1,
        "status": "passed_automated_gates",
        "validated_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": variant,
        "split": split,
        "run_id": run_id,
        "task_index": task_index,
        "relative_path": row["relative_path"],
        "safe_plot_id": row["safe_plot_id"],
        "held_out_test_accessed": False,
        "label_stripped_input": True,
        "source_row_alignment": "passed",
        "target_results": {
            target: {
                key: target_results[target][key]
                for key in (
                    "prediction_instance_count",
                    "reference_instance_count",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                    "mean_matched_iou",
                    "mean_unweighted_coverage",
                    "mean_weighted_coverage",
                    "evaluated_point_count",
                )
            }
            for target in TARGETS
        },
        "stage_resources": {
            stage: {
                "runtime_seconds": stage_metadata[stage].get("runtime_seconds"),
                "peak_rss_gb": stage_metadata[stage].get("peak_rss_gb"),
            }
            for stage in STAGES
        },
        "metrics_csv": str(summary_csv),
        "metrics_csv_sha256": sha256(summary_csv),
        "manual_alignment_review_required": True,
        "full_development_authorised": False,
        "held_out_test_authorised": False,
    }
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one automated TLS2trees development smoke gate."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default=EXPECTED_VARIANT)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = validate_smoke(
            manifest_path=Path(args.manifest_json),
            task_index=args.task_index,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            variant=args.variant,
            split=args.split,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print(f"metrics_csv={payload['metrics_csv']}")
    print("manual_alignment_review_required=true")
    print("held_out_test_authorised=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
