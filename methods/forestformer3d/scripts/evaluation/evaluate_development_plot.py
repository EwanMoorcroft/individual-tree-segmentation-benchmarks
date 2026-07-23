"""Evaluate one aligned ForestFormer3D development prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EVALUATOR_ROOT = ROOT / "methods/segmentanytree/scripts/evaluation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EVALUATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_ROOT))

from pointwise_instance_metrics import PointLabels, evaluate_pointwise  # noqa: E402
from shared.for_instance_manifest import sha256_file  # noqa: E402


def evaluate(
    prediction_path: Path,
    validation_path: Path,
    resource_path: Path,
    output_root: Path,
    *,
    run_id: str,
    benchmark_commit: str,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Refusing existing evaluation root: {output_root}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "passed"
        or validation.get("split") != "development"
        or validation.get("held_out_access") is not False
        or validation.get("exact_row_alignment") is not True
        or validation.get("prediction_npz_sha256") != sha256_file(prediction_path)
    ):
        raise ValueError("Development validation record did not pass")
    with np.load(prediction_path) as data:
        labels = PointLabels(
            predicted_instance=np.asarray(data["pred_tree_id"]),
            reference_instance=np.asarray(data["target_tree_id"]),
            predicted_semantic=np.asarray(data["pred_classification"]),
            reference_semantic=np.asarray(data["classification"]),
        )
    result = evaluate_pointwise(
        labels,
        reference_tree_classes={4.0, 5.0, 6.0},
        prediction_tree_classes={4.0},
        ignored_reference_labels={-1.0, 0.0},
        ignored_prediction_labels={-1.0, 0.0},
        iou_threshold=0.5,
        min_predicted_instance_points=0,
        min_predicted_tree_fraction=0.0,
    )
    harmonized = result["harmonized"]
    metrics: dict[str, object] = {
        "schema": "forestformer3d_development_plot_metrics_v1",
        "status": "completed",
        "method": "ForestFormer3D",
        "training_mode": "published_pretrained",
        "run_id": run_id,
        "split": "development",
        "held_out_access": False,
        "plot_id": validation["plot_id"],
        "relative_path": validation["relative_path"],
        "evaluation_protocol": "for_instance_pointwise_v1",
        "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": 0.5,
        "point_count": int(result["point_count"]),
        "evaluated_point_count": int(result["evaluated_point_count"]),
        "prediction_instance_count": int(result["prediction_instance_count"]),
        "reference_instance_count": int(result["reference_instance_count"]),
        "true_positives": int(harmonized["true_positives"]),
        "false_positives": int(harmonized["false_positives"]),
        "false_negatives": int(harmonized["false_negatives"]),
        "precision": float(harmonized["precision"]),
        "recall": float(harmonized["recall"]),
        "f1": float(harmonized["f1"]),
        "mean_matched_iou": float(harmonized["mean_matched_iou"]),
        "mean_unweighted_coverage": float(result["mean_unweighted_coverage"]),
        "mean_weighted_coverage": float(result["mean_weighted_coverage"]),
        "benchmark_commit": benchmark_commit,
        "prediction_npz_sha256": sha256_file(prediction_path),
        "resource_usage": json.loads(resource_path.read_text(encoding="utf-8")),
    }
    output_root.mkdir(parents=True)
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "evaluation.complete").touch(exist_ok=False)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.prediction, args.validation, args.resource, args.output_root,
        run_id=args.run_id, benchmark_commit=args.benchmark_commit,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
