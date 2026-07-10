"""Create final TreeX development/test summaries and plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create final TreeX development/test summaries and plots."
    )
    parser.add_argument(
        "--dev-csv",
        default="results/treex_for_instance/treex_dev_full_summary.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="results/treex_for_instance/treex_test_full_summary.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/treex_for_instance"
    )
    parser.add_argument(
        "--plot-dir",
        help="Plot output directory; defaults to <output-dir>/plots.",
    )
    return parser.parse_args()


def _add_micro_metrics(frame: Any, suffix: str) -> None:
    """Add micro scores derived from aggregate TP, FP, and FN counts."""

    tp = frame[f"total_tp_{suffix}"]
    fp = frame[f"total_fp_{suffix}"]
    fn = frame[f"total_fn_{suffix}"]
    frame[f"micro_precision_{suffix}"] = tp.div(tp + fp).fillna(0.0)
    frame[f"micro_recall_{suffix}"] = tp.div(tp + fn).fillna(0.0)
    frame[f"micro_f1_{suffix}"] = (2 * tp).div(
        (2 * tp) + fp + fn
    ).fillna(0.0)


def _aggregate(combined: Any, group_columns: list[str]) -> Any:
    """Aggregate counts first, then derive micro and plot-level summaries."""

    grouped = (
        combined.groupby(group_columns)
        .agg(
            n_plots=("plot_id", "count"),
            total_reference_trees=("reference_trees", "sum"),
            total_predicted_trees_harmonized=(
                "predicted_trees_harmonized_union_mask",
                "sum",
            ),
            total_tp_harmonized=("true_positives_harmonized", "sum"),
            total_fp_harmonized=("false_positives_harmonized", "sum"),
            total_fn_harmonized=("false_negatives_harmonized", "sum"),
            mean_plot_f1_harmonized=("f1_harmonized", "mean"),
            median_plot_f1_harmonized=("f1_harmonized", "median"),
            min_plot_f1_harmonized=("f1_harmonized", "min"),
            max_plot_f1_harmonized=("f1_harmonized", "max"),
            total_predicted_trees_labelled_mask=(
                "predicted_trees_on_reference_labelled_mask",
                "sum",
            ),
            total_tp_labelled_mask=("true_positives_labelled_mask", "sum"),
            total_fp_labelled_mask=("false_positives_labelled_mask", "sum"),
            total_fn_labelled_mask=("false_negatives_labelled_mask", "sum"),
            mean_plot_f1_labelled_mask=("f1_labelled_mask", "mean"),
            median_plot_f1_labelled_mask=("f1_labelled_mask", "median"),
            min_plot_f1_labelled_mask=("f1_labelled_mask", "min"),
            max_plot_f1_labelled_mask=("f1_labelled_mask", "max"),
            mean_of_plot_mean_matched_iou_harmonized=(
                "mean_matched_iou_harmonized",
                "mean",
            ),
            median_of_plot_mean_matched_iou_harmonized=(
                "mean_matched_iou_harmonized",
                "median",
            ),
            mean_runtime_seconds=("elapsed_seconds", "mean"),
            total_runtime_seconds=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )
    _add_micro_metrics(grouped, "harmonized")
    _add_micro_metrics(grouped, "labelled_mask")
    return grouped


def create_final_summaries(
    dev_path: Path,
    test_path: Path,
    output_dir: Path,
    plot_dir: Path,
) -> list[Path]:
    """Build public aggregate tables and plots from per-plot split tables."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    for path in (dev_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    development["split"] = "dev"
    test["split"] = "test"
    combined = pd.concat([development, test], ignore_index=True)
    combined["site"] = combined["plot_id"].str.split("/").str[0]
    combined = combined.sort_values(
        ["split", "site", "plot_id"]
    ).reset_index(drop=True)

    combined_path = output_dir / "treex_combined_dev_test_summary.csv"
    combined.to_csv(combined_path, index=False)
    split_path = output_dir / "treex_split_summary.csv"
    _aggregate(combined, ["split"]).to_csv(split_path, index=False)
    site_path = output_dir / "treex_site_summary.csv"
    _aggregate(combined, ["split", "site"]).to_csv(site_path, index=False)

    worst_path = output_dir / "treex_worst_plots_by_strict_f1.csv"
    best_path = output_dir / "treex_best_plots_by_strict_f1.csv"
    combined.sort_values("f1_harmonized").head(15).to_csv(
        worst_path, index=False
    )
    combined.sort_values("f1_harmonized", ascending=False).head(15).to_csv(
        best_path, index=False
    )

    plot_data = combined.copy()
    plot_data["label"] = plot_data["split"] + " | " + plot_data["plot_id"]

    plt.figure(figsize=(14, 7))
    plt.bar(plot_data["label"], plot_data["f1_harmonized"])
    plt.xticks(rotation=90)
    plt.ylabel("Harmonised one-to-one F1")
    plt.xlabel("Plot")
    plt.title("TreeX harmonised F1 by FOR-instance plot")
    plt.tight_layout()
    strict_plot = plot_dir / "treex_strict_f1_by_plot.png"
    plt.savefig(strict_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.bar(plot_data["label"], plot_data["f1_labelled_mask"])
    plt.xticks(rotation=90)
    plt.ylabel("Reference-labelled-mask F1")
    plt.xlabel("Plot")
    plt.title("TreeX reference-labelled-mask diagnostic F1 by plot")
    plt.tight_layout()
    labelled_plot = plot_dir / "treex_labelled_mask_f1_by_plot.png"
    plt.savefig(labelled_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(combined["elapsed_seconds"], combined["f1_harmonized"])
    plt.xlabel("Runtime per plot (seconds)")
    plt.ylabel("Harmonised one-to-one F1")
    plt.title("TreeX runtime vs harmonised F1")
    plt.tight_layout()
    runtime_plot = plot_dir / "treex_runtime_vs_strict_f1.png"
    plt.savefig(runtime_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        combined["reference_trees"],
        combined["predicted_trees_harmonized_union_mask"],
    )
    maximum = max(
        combined["reference_trees"].max(),
        combined["predicted_trees_harmonized_union_mask"].max(),
    )
    plt.plot([0, maximum], [0, maximum])
    plt.xlabel("Reference trees")
    plt.ylabel("Predicted trees")
    plt.title("TreeX predicted vs reference tree counts")
    plt.tight_layout()
    count_plot = plot_dir / "treex_predicted_vs_reference_counts.png"
    plt.savefig(count_plot, dpi=200, facecolor="white")
    plt.close()

    return [
        combined_path,
        split_path,
        site_path,
        worst_path,
        best_path,
        strict_plot,
        labelled_plot,
        runtime_plot,
        count_plot,
    ]


def main() -> int:
    """Run the summary builder from command-line paths."""

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = create_final_summaries(
        Path(args.dev_csv).expanduser().resolve(),
        Path(args.test_csv).expanduser().resolve(),
        output_dir,
        (
            Path(args.plot_dir).expanduser().resolve()
            if args.plot_dir
            else output_dir / "plots"
        ),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
