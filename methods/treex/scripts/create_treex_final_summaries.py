"""Create final TreeX development/test summaries and plots."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
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
        "--output-dir",
        default="results/treex_for_instance",
    )
    parser.add_argument(
        "--plot-dir",
        help="Plot output directory; defaults to <output-dir>/plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    dev_path = Path(args.dev_csv).expanduser().resolve()
    test_path = Path(args.test_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir
        else output_dir / "plots"
    )
    if not dev_path.is_file():
        raise FileNotFoundError(dev_path)
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
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

    split_summary = (
        combined.groupby("split")
        .agg(
            n_plots=("plot_id", "count"),
            total_reference_trees=("reference_trees", "sum"),
            total_predicted_trees_all=("predicted_trees_all", "sum"),
            total_tp=("true_positives", "sum"),
            total_fp_labelled=("false_positives_labelled_mask", "sum"),
            total_fp_strict=("false_positives_strict", "sum"),
            total_fn=("false_negatives", "sum"),
            mean_f1_labelled=("f1_labelled_mask", "mean"),
            median_f1_labelled=("f1_labelled_mask", "median"),
            min_f1_labelled=("f1_labelled_mask", "min"),
            max_f1_labelled=("f1_labelled_mask", "max"),
            mean_f1_strict=("f1_strict", "mean"),
            median_f1_strict=("f1_strict", "median"),
            min_f1_strict=("f1_strict", "min"),
            max_f1_strict=("f1_strict", "max"),
            mean_iou=("mean_matched_iou", "mean"),
            median_iou=("mean_matched_iou", "median"),
            mean_runtime_s=("elapsed_seconds", "mean"),
            total_runtime_s=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )
    split_path = output_dir / "treex_split_summary.csv"
    split_summary.to_csv(split_path, index=False)

    site_summary = (
        combined.groupby(["split", "site"])
        .agg(
            n_plots=("plot_id", "count"),
            total_reference_trees=("reference_trees", "sum"),
            total_predicted_trees_all=("predicted_trees_all", "sum"),
            total_tp=("true_positives", "sum"),
            total_fp_labelled=("false_positives_labelled_mask", "sum"),
            total_fp_strict=("false_positives_strict", "sum"),
            total_fn=("false_negatives", "sum"),
            mean_f1_labelled=("f1_labelled_mask", "mean"),
            median_f1_labelled=("f1_labelled_mask", "median"),
            mean_f1_strict=("f1_strict", "mean"),
            median_f1_strict=("f1_strict", "median"),
            mean_iou=("mean_matched_iou", "mean"),
            median_iou=("mean_matched_iou", "median"),
            mean_runtime_s=("elapsed_seconds", "mean"),
            total_runtime_s=("elapsed_seconds", "sum"),
        )
        .reset_index()
    )
    site_path = output_dir / "treex_site_summary.csv"
    site_summary.to_csv(site_path, index=False)

    worst_path = output_dir / "treex_worst_plots_by_strict_f1.csv"
    best_path = output_dir / "treex_best_plots_by_strict_f1.csv"
    combined.sort_values("f1_strict").head(15).to_csv(
        worst_path,
        index=False,
    )
    combined.sort_values("f1_strict", ascending=False).head(15).to_csv(
        best_path,
        index=False,
    )

    plot_data = combined.copy()
    plot_data["label"] = (
        plot_data["split"] + " | " + plot_data["plot_id"]
    )

    plt.figure(figsize=(14, 7))
    plt.bar(plot_data["label"], plot_data["f1_strict"])
    plt.xticks(rotation=90)
    plt.ylabel("Strict F1")
    plt.xlabel("Plot")
    plt.title("TreeX strict F1 by FOR-instance plot")
    plt.tight_layout()
    strict_plot = plot_dir / "treex_strict_f1_by_plot.png"
    plt.savefig(strict_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(14, 7))
    plt.bar(plot_data["label"], plot_data["f1_labelled_mask"])
    plt.xticks(rotation=90)
    plt.ylabel("Labelled-mask F1")
    plt.xlabel("Plot")
    plt.title("TreeX labelled-mask F1 by FOR-instance plot")
    plt.tight_layout()
    labelled_plot = plot_dir / "treex_labelled_mask_f1_by_plot.png"
    plt.savefig(labelled_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(combined["elapsed_seconds"], combined["f1_strict"])
    plt.xlabel("Runtime per plot (seconds)")
    plt.ylabel("Strict F1")
    plt.title("TreeX runtime vs strict F1")
    plt.tight_layout()
    runtime_plot = plot_dir / "treex_runtime_vs_strict_f1.png"
    plt.savefig(runtime_plot, dpi=200, facecolor="white")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        combined["reference_trees"],
        combined["predicted_trees_all"],
    )
    maximum = max(
        combined["reference_trees"].max(),
        combined["predicted_trees_all"].max(),
    )
    plt.plot([0, maximum], [0, maximum])
    plt.xlabel("Reference trees")
    plt.ylabel("Predicted trees")
    plt.title("TreeX predicted vs reference tree counts")
    plt.tight_layout()
    count_plot = plot_dir / "treex_predicted_vs_reference_counts.png"
    plt.savefig(count_plot, dpi=200, facecolor="white")
    plt.close()

    for path in (
        combined_path,
        split_path,
        site_path,
        worst_path,
        best_path,
        strict_plot,
        labelled_plot,
        runtime_plot,
        count_plot,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
