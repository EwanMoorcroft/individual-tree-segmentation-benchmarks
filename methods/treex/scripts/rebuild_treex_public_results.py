"""Rebuild public TreeX tables and plots from retained aligned predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_treex_final_summaries import create_final_summaries
from create_treex_split_summary import OUTPUT_FIELDS, RUN_FIELDS
from evaluate_treex_for_instance_plot import evaluate_arrays, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate retained TreeX NPZ files and rebuild public results."
        )
    )
    parser.add_argument(
        "--source-summary",
        default="methods/treex/examples/treex_run_metadata.csv",
        help="Immutable public run metadata and plot identifiers.",
    )
    parser.add_argument(
        "--prediction-root",
        default="local_outputs/treex_predictions",
    )
    parser.add_argument(
        "--config",
        default="methods/treex/configs/for_instance_benchmark.yml",
    )
    parser.add_argument(
        "--output-dir", default="methods/treex/examples"
    )
    parser.add_argument("--plot-dir", default="methods/treex/plots")
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    """Resolve repository-relative command-line paths."""

    path = Path(path_text).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _prediction_path(root: Path, split: str, plot_id: str) -> Path:
    """Locate the single retained NPZ for one split and plot identifier."""

    split_dir = "for_instance_test" if split == "test" else "for_instance"
    plot_dir = root / split_dir / plot_id.replace("/", "_")
    candidates = sorted(plot_dir.glob("*_treex_predictions.npz"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one retained prediction for {split}/{plot_id}; "
            f"found {len(candidates)} in {plot_dir}"
        )
    return candidates[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one per-plot result table with the canonical field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rebuild(
    source_summary: Path,
    prediction_root: Path,
    config: dict[str, Any],
    output_dir: Path,
    plot_dir: Path,
) -> list[Path]:
    """Recompute all per-plot metrics, aggregate tables, and plots."""

    with source_summary.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"Source summary contains no rows: {source_summary}")
    if len({row["plot_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Source summary contains duplicate plot IDs")

    expected_split_counts = {
        "dev": int(config["dataset"]["local_dev_plot_count"]),
        "test": int(config["dataset"]["local_test_plot_count"]),
    }
    observed_split_counts = {
        split: sum(row["split"] == split for row in source_rows)
        for split in expected_split_counts
    }
    if observed_split_counts != expected_split_counts:
        raise ValueError(
            "Source summary split counts do not match the config: "
            f"{observed_split_counts} != {expected_split_counts}"
        )

    split_rows: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    for source in source_rows:
        split = source["split"]
        if split not in split_rows:
            raise ValueError(f"Unsupported split {split!r}")
        prediction_path = _prediction_path(
            prediction_root, split, source["plot_id"]
        )
        with np.load(prediction_path) as data:
            required = {"pred_tree_id", "target_tree_id", "classification"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"{prediction_path} is missing arrays {sorted(missing)}"
                )
            metrics, _, _ = evaluate_arrays(
                data["pred_tree_id"],
                data["target_tree_id"],
                data["classification"],
                config,
                source["plot_id"],
            )
        if int(source["total_points"]) != int(metrics["total_points"]):
            raise ValueError(f"Point-count mismatch for {source['plot_id']}")
        if int(source["reference_tree_count_tree_classes"]) != int(
            metrics["reference_trees"]
        ):
            raise ValueError(
                f"Reference-count mismatch for {source['plot_id']}"
            )
        if int(source["predicted_instances"]) != int(
            metrics["predicted_trees_harmonized_union_mask"]
        ):
            raise ValueError(
                f"Prediction-count mismatch for {source['plot_id']}"
            )
        split_rows[split].append(
            {
                "plot_id": source["plot_id"],
                **{field: source[field] for field in RUN_FIELDS},
                **{
                    field: metrics[field]
                    for field in OUTPUT_FIELDS
                    if field not in {"plot_id", *RUN_FIELDS}
                },
            }
        )

    dev_path = output_dir / "treex_dev_full_summary.csv"
    test_path = output_dir / "treex_test_full_summary.csv"
    _write_csv(dev_path, sorted(split_rows["dev"], key=lambda row: row["plot_id"]))
    _write_csv(test_path, sorted(split_rows["test"], key=lambda row: row["plot_id"]))
    return [
        dev_path,
        test_path,
        *create_final_summaries(dev_path, test_path, output_dir, plot_dir),
    ]


def main() -> int:
    """Run the deterministic public-result rebuild."""

    args = parse_args()
    paths = rebuild(
        _resolve(args.source_summary),
        _resolve(args.prediction_root),
        load_config(args.config),
        _resolve(args.output_dir),
        _resolve(args.plot_dir),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
