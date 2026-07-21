"""Summarise a frozen TLS2trees compatibility probe without reading metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods" / "tls2trees" / "scripts" / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from for_instance_published_common import sha256, utc_now, write_json


def summarise(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    plot_roots = [path for path in run_root.iterdir() if path.is_dir()]
    if len(plot_roots) != 1:
        raise ValueError(f"Expected exactly one probe plot root; found {len(plot_roots)}")
    probe_root = plot_roots[0] / "compatibility_probe"
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata_path = probe_root / candidate["candidate_id"] / "metadata" / "probe_run.json"
        row: dict[str, Any] = {
            "candidate_index": candidate["candidate_index"],
            "candidate_id": candidate["candidate_id"],
            "status": "missing",
            "leaf_off_prediction_file_count": 0,
            "leaf_off_prediction_point_count": 0,
            "runtime_seconds": None,
            "peak_rss_gb": None,
            "metadata_path": str(metadata_path),
            "metadata_sha256": None,
            "error": "candidate metadata missing",
        }
        if metadata_path.is_file():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("candidate_id") != candidate["candidate_id"]:
                raise ValueError(f"Candidate metadata ID mismatch: {metadata_path}")
            if payload.get("candidate_manifest_sha256") != sha256(manifest_path):
                raise RuntimeError(f"Candidate manifest checksum mismatch: {metadata_path}")
            if any(
                payload.get(key) is not False
                for key in (
                    "reference_labels_accessed",
                    "accuracy_metrics_accessed",
                    "selection_uses_accuracy_metrics",
                    "held_out_test_accessed",
                    "benchmark_result",
                )
            ):
                raise ValueError(f"Candidate crossed the compatibility-probe scope: {metadata_path}")
            predictions = payload.get("prediction_inventory", {}).get("leaf_off", [])
            row.update(
                {
                    "status": payload.get("status", "unknown"),
                    "leaf_off_prediction_file_count": len(predictions),
                    "leaf_off_prediction_point_count": sum(
                        int(record.get("point_count", 0)) for record in predictions
                    ),
                    "runtime_seconds": payload.get("runtime_seconds"),
                    "peak_rss_gb": payload.get("peak_rss_gb"),
                    "metadata_sha256": sha256(metadata_path),
                    "error": payload.get("error"),
                }
            )
        rows.append(row)
    viable = [row["candidate_id"] for row in rows if row["status"] == "viable_nonempty"]
    incomplete = [row["candidate_id"] for row in rows if row["status"] in {"missing", "failed", "unknown"}]
    status = (
        "probe_incomplete"
        if incomplete
        else "viable_candidates_found"
        if viable
        else "no_viable_candidates"
    )
    return {
        "schema_version": 1,
        "status": status,
        "created_at_utc": utc_now(),
        "variant": "development_tuned",
        "split": "development",
        "run_id": run_root.name,
        "safe_plot_id": plot_roots[0].name,
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256(manifest_path),
        "candidate_count": len(rows),
        "viable_candidate_ids": viable,
        "incomplete_candidate_ids": incomplete,
        "selection_rule": "leaf_off_prediction_file_count_greater_than_zero",
        "final_configuration_selected": False,
        "reference_labels_accessed": False,
        "accuracy_metrics_accessed": False,
        "selection_uses_accuracy_metrics": False,
        "held_out_test_accessed": False,
        "benchmark_result": False,
        "candidates": rows,
        "next_action": (
            "cross_site_development_stage0_with_both_leaf_targets"
            if viable and not incomplete
            else "inspect_failed_candidates_before_decision"
            if incomplete
            else "review_hidden_cylinder_acceptance_boundary"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        fields = [
            "candidate_index",
            "candidate_id",
            "status",
            "leaf_off_prediction_file_count",
            "leaf_off_prediction_point_count",
            "runtime_seconds",
            "peak_rss_gb",
            "metadata_path",
            "metadata_sha256",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if output_json.exists() or output_csv.exists():
        print("Refusing existing compatibility-probe summary output", file=sys.stderr)
        return 2
    try:
        payload = summarise(Path(args.run_root), Path(args.candidate_manifest))
        write_json(output_json, payload)
        write_csv(output_csv, payload["candidates"])
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print("viable_candidate_ids=" + ",".join(payload["viable_candidate_ids"]))
    print(f"summary_json={output_json}")
    print("accuracy_metrics_accessed=false")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
