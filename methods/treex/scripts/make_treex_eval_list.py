"""Create a TreeX development evaluation list."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TreeX development evaluation list."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--prediction-root",
        default="data/predictions/treex/for_instance",
    )
    parser.add_argument(
        "--results-root",
        default="results/treex_for_instance",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {"plot_id", "input_las"}
    if rows and required_columns - set(rows[0]):
        raise ValueError(
            f"Input CSV is missing columns {sorted(required_columns - set(rows[0]))}"
        )

    output_rows = []
    for row in rows:
        plot_id = row["plot_id"]
        safe_plot_id = plot_id.replace("/", "_")
        stem = Path(row["input_las"]).stem
        output_rows.append(
            {
                "plot_id": plot_id,
                "safe_plot_id": safe_plot_id,
                "summary_json": (
                    f"{args.results_root}/{safe_plot_id}_treex_summary.json"
                ),
                "prediction_npz": (
                    f"{args.prediction_root}/{safe_plot_id}/"
                    f"{stem}_treex_predictions.npz"
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "plot_id",
                "safe_plot_id",
                "summary_json",
                "prediction_npz",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
