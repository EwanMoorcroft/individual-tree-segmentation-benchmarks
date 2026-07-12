"""Freeze accepted TreeLearn smoke evidence before full development submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from run_for_instance_one_plot_smoke import md5, sha256


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def prepare_acceptance(
    config: dict[str, object],
    inference_path: Path,
    metrics_path: Path,
    output: Path,
    manual_alignment_review_confirmed: bool,
) -> dict[str, object]:
    if not manual_alignment_review_confirmed:
        raise ValueError("Manual development alignment review is not confirmed")
    expected = config["accepted_smoke"]
    method = config["method"]
    checkpoint_contract = method["checkpoint"]
    inference = load_json(inference_path)
    metrics = load_json(metrics_path)

    if inference.get("status") != "completed":
        raise ValueError("Accepted smoke inference is not completed")
    if metrics.get("status") != "completed_development_smoke_evaluation":
        raise ValueError("Accepted smoke evaluation is not completed")
    for payload in (inference, metrics):
        if payload.get("run_id") != expected["run_id"]:
            raise ValueError("Smoke run ID does not match the frozen contract")
    plot = inference.get("plot", {})
    if (
        plot.get("split") != "dev"
        or plot.get("relative_path") != expected["relative_path"]
    ):
        raise ValueError("Smoke evidence is not the frozen development plot")
    validation = inference.get("validation", {})
    if (
        validation.get("row_count_match") is not True
        or validation.get("row_order_preserved") is not True
        or int(validation.get("source_point_count", 0)) != expected["point_count"]
        or float(validation.get("max_abs_coordinate_delta_m", 1.0))
        != expected["max_abs_coordinate_delta_m"]
    ):
        raise ValueError("Smoke point-alignment evidence differs from the contract")
    for field in (
        "reference_instance_count",
        "prediction_instance_count",
        "true_positives",
        "false_positives",
        "false_negatives",
    ):
        expected_key = {
            "reference_instance_count": "reference_instances",
            "prediction_instance_count": "predicted_instances",
        }.get(field, field)
        if int(metrics.get(field, -1)) != int(expected[expected_key]):
            raise ValueError(f"Smoke metric {field} differs from the frozen contract")
    if abs(float(metrics.get("f1", -1.0)) - float(expected["f1"])) > 1e-12:
        raise ValueError("Smoke F1 differs from the frozen contract")

    checkpoint = inference.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (
        checkpoint.get("md5") != checkpoint_contract["source_md5"]
        or checkpoint.get("sha256") != checkpoint_contract["sha256"]
        or not checkpoint_path.is_file()
        or md5(checkpoint_path) != checkpoint_contract["source_md5"]
        or sha256(checkpoint_path) != checkpoint_contract["sha256"]
    ):
        raise ValueError("Smoke checkpoint identity differs from the frozen contract")
    upstream = inference.get("environment", {}).get("treelearn_repository", {})
    benchmark = inference.get("environment", {}).get("benchmark_repository", {})
    if upstream.get("commit") != method["upstream_commit"] or upstream.get(
        "dirty"
    ) is not False:
        raise ValueError("Smoke upstream repository evidence is not frozen and clean")
    if (
        benchmark.get("commit") != expected["benchmark_commit"]
        or benchmark.get("dirty") is not False
    ):
        raise ValueError("Smoke benchmark repository evidence differs from the contract")

    retained = inference.get("retention", {}).get("files", [])
    outputs = inference.get("outputs", {})
    expected_artifacts = expected["retained_artifacts"]
    expected_roles = tuple(expected_artifacts)
    if len(retained) != len(expected_roles):
        raise ValueError("Smoke retention inventory must contain five artefacts")
    output_paths = {
        role: Path(str(outputs.get(role, ""))).expanduser().resolve()
        for role in expected_roles
    }
    if any(not str(outputs.get(role, "")) for role in expected_roles):
        raise ValueError("Smoke output inventory is missing a frozen artefact role")
    verified_files = []
    observed_paths: set[Path] = set()
    for item in retained:
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if path in observed_paths:
            raise ValueError(f"Smoke retention inventory repeats an artefact: {path}")
        observed_paths.add(path)
        matching_roles = [
            role for role, output_path in output_paths.items() if output_path == path
        ]
        if len(matching_roles) != 1:
            raise ValueError(f"Smoke retained artefact has no frozen output role: {path}")
        role = matching_roles[0]
        artifact_contract = expected_artifacts[role]
        if (
            item.get("exists") is not True
            or not path.is_file()
            or path.stat().st_size != int(artifact_contract["size_bytes"])
            or int(item.get("size_bytes", -1)) != int(artifact_contract["size_bytes"])
            or sha256(path) != artifact_contract["sha256"]
            or item.get("sha256") != artifact_contract["sha256"]
        ):
            raise ValueError(f"Smoke retained artefact failed verification: {path}")
        verified_files.append({"role": role, **dict(item)})
    if observed_paths != set(output_paths.values()):
        raise ValueError("Smoke retention inventory does not match all output roles")
    verified_files.sort(key=lambda item: item["role"])

    record = {
        "protocol_id": config["project"]["acceptance_protocol_id"],
        "method": "TreeLearn",
        "dataset": "FOR-instance",
        "status": "accepted_for_full_development_only",
        "training_mode": checkpoint_contract["training_mode"],
        "smoke_run_id": expected["run_id"],
        "smoke_inference_metadata": str(inference_path),
        "smoke_metrics": str(metrics_path),
        "manual_alignment_review_confirmed": True,
        "held_out_test_accessed": False,
        "checkpoint_md5": checkpoint_contract["source_md5"],
        "checkpoint_sha256": checkpoint_contract["sha256"],
        "upstream_commit": method["upstream_commit"],
        "benchmark_commit_at_smoke": benchmark["commit"],
        "alignment": {
            "point_count": validation["source_point_count"],
            "row_count_match": validation["row_count_match"],
            "row_order_preserved": validation["row_order_preserved"],
            "max_abs_coordinate_delta_m": validation[
                "max_abs_coordinate_delta_m"
            ],
        },
        "metrics": {
            "prediction_instances": metrics["prediction_instance_count"],
            "reference_instances": metrics["reference_instance_count"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "verified_retained_files": verified_files,
        "next_gate": "submit_full_development_array_without_test_access",
    }
    if output.exists():
        raise FileExistsError(f"Refusing existing acceptance record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-inference", required=True)
    parser.add_argument("--smoke-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manual-alignment-review-confirmed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    record = prepare_acceptance(
        config,
        Path(args.smoke_inference).expanduser().resolve(),
        Path(args.smoke_metrics).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        args.manual_alignment_review_confirmed,
    )
    print(f"acceptance_status={record['status']}")
    print(f"acceptance_record={Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
