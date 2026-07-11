"""Summarise completed aligned SegmentAnyTree results by FOR-instance site."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_SITES = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=METRICS_ROOT for --variant.")
    label, root = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Variant label cannot be empty.")
    return label.strip(), resolve_path(root)


def metric_payloads(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Metrics root does not exist: {root}")
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evaluator") == "pointwise_instance_metrics":
            payloads.append(payload)
    return payloads


def aggregate_site(
    variant: str,
    site: str,
    payloads: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    harmonized = [payload["harmonized"] for payload in payloads]
    paper = [payload["paper_compatible"] for payload in payloads]
    tp = sum(int(metric["true_positives"]) for metric in harmonized)
    fp = sum(int(metric["false_positives"]) for metric in harmonized)
    fn = sum(int(metric["false_negatives"]) for metric in harmonized)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "variant": variant,
        "dataset_split": split,
        "site": site,
        "plots": len(payloads),
        "predicted_instances": sum(
            int(payload["prediction_instance_count"]) for payload in payloads
        ),
        "reference_instances": sum(
            int(payload["reference_instance_count"]) for payload in payloads
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_f1": statistics.fmean(
            float(metric["f1"]) for metric in harmonized
        ),
        "mean_plot_precision": statistics.fmean(
            float(metric["precision"]) for metric in harmonized
        ),
        "mean_plot_recall": statistics.fmean(
            float(metric["recall"]) for metric in harmonized
        ),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_matched_iou": statistics.fmean(
            float(metric["mean_matched_iou"]) for metric in harmonized
        ),
        "mean_unweighted_coverage": statistics.fmean(
            float(payload["mean_unweighted_coverage"]) for payload in payloads
        ),
        "mean_weighted_coverage": statistics.fmean(
            float(payload["mean_weighted_coverage"]) for payload in payloads
        ),
        "paper_mean_f1": statistics.fmean(float(metric["f1"]) for metric in paper),
        "evaluation_protocol": "for_instance_pointwise_v1",
        "matching_policy": "maximum_cardinality_one_to_one",
        "result_status": "completed_aligned_pointwise_site_summary",
    }


def summarise_sites(
    variant: str,
    root: Path,
    expected_plots: int,
    expected_split: str,
) -> list[dict[str, Any]]:
    payloads = metric_payloads(root)
    if len(payloads) != expected_plots:
        raise ValueError(
            f"{variant}: expected {expected_plots} metrics, found {len(payloads)}"
        )
    wrong_splits = sorted(
        {
            str(payload.get("split"))
            for payload in payloads
            if payload.get("split") != expected_split
        }
    )
    if wrong_splits:
        raise ValueError(f"{variant}: unexpected splits: {wrong_splits}")
    plot_keys = [
        (str(payload.get("collection")), str(payload.get("plot_name")))
        for payload in payloads
    ]
    if len(set(plot_keys)) != len(plot_keys):
        raise ValueError(f"{variant}: duplicate site/plot metric records")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[str(payload.get("collection"))].append(payload)
    observed_sites = tuple(sorted(grouped))
    if set(observed_sites) != set(EXPECTED_SITES):
        raise ValueError(
            f"{variant}: expected sites {EXPECTED_SITES}, found {observed_sites}"
        )
    return [
        aggregate_site(variant, site, grouped[site], expected_split)
        for site in EXPECTED_SITES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise completed SAT metric JSONs by FOR-instance site."
    )
    parser.add_argument("--variant", action="append", required=True, type=parse_variant)
    parser.add_argument("--expected-plots", type=int, default=11)
    parser.add_argument("--expected-split", default="test")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for label, root in args.variant:
        rows.extend(
            summarise_sites(
                label,
                root,
                args.expected_plots,
                args.expected_split,
            )
        )
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['variant']} {row['site']}: plots={row['plots']} "
            f"mean_plot_f1={row['mean_plot_f1']:.4f} "
            f"micro_f1={row['micro_f1']:.4f}"
        )
    print(f"site_summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
