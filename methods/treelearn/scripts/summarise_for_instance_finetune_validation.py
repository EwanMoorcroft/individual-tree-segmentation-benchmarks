"""Summarise the five frozen TreeLearn fine-tune validation plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--table-root", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    rows = [row for row in freeze["plots"] if row["training_role"] == "validation"]
    if len(rows) != 5 or freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Expected five frozen development-validation plots")

    metrics = []
    retained = []
    for row in rows:
        safe = row["safe_plot_id"]
        metric_path = args.table_root / "per_plot" / safe / "metrics.json"
        metadata_path = args.metadata_root / f"{safe}_inference.json"
        metric = json.loads(metric_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        if metric.get("status") != "completed_aligned_pointwise_development_plot":
            raise ValueError(f"Incomplete validation metric: {safe}")
        if metadata.get("training_mode") != "fine_tuned_on_dev":
            raise ValueError(f"Wrong training mode: {safe}")
        if metadata.get("checkpoint", {}).get("sha256") != args.checkpoint_sha256:
            raise ValueError(f"Checkpoint mismatch: {safe}")
        if metadata.get("held_out_test_accessed") is not False:
            raise ValueError(f"Held-out test lock missing: {safe}")
        metric["collection"] = row["collection"]
        metrics.append(metric)
        for entry in metadata["retention"]["files"]:
            path = Path(entry["path"])
            if not path.is_file() or sha256(path) != entry["sha256"]:
                raise ValueError(f"Retention mismatch: {path}")
            retained.append(entry)

    def aggregate(group: list[dict]) -> dict:
        tp = sum(int(row["true_positives"]) for row in group)
        fp = sum(int(row["false_positives"]) for row in group)
        fn = sum(int(row["false_negatives"]) for row in group)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "plots": len(group), "true_positives": tp, "false_positives": fp,
            "false_negatives": fn,
            "mean_plot_f1": sum(float(row["f1"]) for row in group) / len(group),
            "micro_precision": precision, "micro_recall": recall,
            "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    overall = aggregate(metrics)
    sites = defaultdict(list)
    for metric in metrics:
        sites[metric["collection"]].append(metric)
    args.table_root.mkdir(parents=True, exist_ok=True)
    fields = ["site", "plots", "true_positives", "false_positives", "false_negatives",
              "mean_plot_f1", "micro_precision", "micro_recall", "micro_f1"]
    with (args.table_root / "validation_site_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for site, group in sorted(sites.items()):
            writer.writerow({"site": site, **aggregate(group)})
    summary = {
        "status": "completed_internal_development_validation",
        "method": "TreeLearn", "training_mode": "fine_tuned_on_dev",
        "run_id": args.run_id, "dataset_split": "dev_validation",
        "checkpoint_sha256": args.checkpoint_sha256,
        "held_out_test_accessed": False, **overall,
        "retention_status": "retention_verified",
        "retained_prediction_files": len(retained),
        "retained_prediction_bytes": sum(int(row["size_bytes"]) for row in retained),
        "next_gate": "manual_review_before_any_held_out_test_route",
    }
    (args.table_root / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
