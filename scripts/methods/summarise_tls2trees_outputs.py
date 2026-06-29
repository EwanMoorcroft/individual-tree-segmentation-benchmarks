from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.ply_io import read_ply_vertices


TREE_FIELDS = [
    "plot_name",
    "tree_index",
    "file",
    "point_count",
    "columns",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "file_size_bytes",
    "error",
]


def resolve_path(path_text: str, root: Path = ROOT) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def inspect_tree_file(path: Path, output_dir: Path, plot_name: str, index: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "plot_name": plot_name,
        "tree_index": index,
        "file": str(path.relative_to(output_dir)),
        "file_size_bytes": path.stat().st_size,
        "error": None,
    }
    try:
        header, points = read_ply_vertices(path, columns=["x", "y", "z"])
        record.update({"point_count": header.vertex_count, "columns": header.columns})
        for axis in ("x", "y", "z"):
            values = points[axis]
            record[f"{axis}_min"] = float(np.min(values)) if len(values) else None
            record[f"{axis}_max"] = float(np.max(values)) if len(values) else None
    except Exception as exc:
        record.update(
            {
                "point_count": None,
                "columns": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def summarise_plot(
    plot_name: str,
    output_dir: Path,
    json_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    tree_files = sorted(output_dir.rglob("*.leafoff.ply")) if output_dir.is_dir() else []
    records = [
        inspect_tree_file(path, output_dir, plot_name, index)
        for index, path in enumerate(tree_files, start=1)
    ]
    valid_records = [record for record in records if not record.get("error")]

    def aggregate_bound(key: str, function: Any) -> float | None:
        values = [record[key] for record in valid_records if record.get(key) is not None]
        return float(function(values)) if values else None

    status = "missing_output_directory" if not output_dir.is_dir() else "complete"
    if output_dir.is_dir() and not tree_files:
        status = "no_leafoff_outputs"
    if any(record.get("error") for record in records):
        status = "completed_with_errors"
    point_counts = [int(record["point_count"]) for record in valid_records]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "plot_name": plot_name,
        "output_directory": str(output_dir),
        "output_dir": str(output_dir),
        "status": status,
        "leafoff_file_count": len(tree_files),
        "predicted_tree_count": len(tree_files),
        "valid_tree_file_count": len(valid_records),
        "files_with_errors": len(records) - len(valid_records),
        "total_predicted_tree_points": sum(point_counts),
        "predicted_tree_points": sum(point_counts),
        "min_tree_points": min(point_counts) if point_counts else None,
        "max_tree_points": max(point_counts) if point_counts else None,
        "mean_tree_points": float(np.mean(point_counts)) if point_counts else None,
        "bounding_box": {
            "x_min": aggregate_bound("x_min", min),
            "x_max": aggregate_bound("x_max", max),
            "y_min": aggregate_bound("y_min", min),
            "y_max": aggregate_bound("y_max", max),
            "z_min": aggregate_bound("z_min", min),
            "z_max": aggregate_bound("z_max", max),
        },
        "trees": records,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TREE_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["columns"] = json.dumps(row.get("columns", []))
            writer.writerow({field: row.get(field) for field in TREE_FIELDS})
    return summary


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def combined_row(
    summary: dict[str, Any],
    conversion: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    run_status = run.get("status")
    return_code = run.get("return_code")
    if run_status == "completed" and return_code == 0:
        status = summary["status"]
    elif run_status:
        status = run_status
    else:
        status = summary["status"]
    return {
        "plot_name": summary["plot_name"],
        "input_point_count": conversion.get("original_point_count"),
        "retained_point_count": conversion.get("retained_point_count"),
        "dropped_unknown_count": conversion.get("dropped_unknown_count"),
        "wood_point_count": conversion.get("wood_point_count"),
        "nonwood_point_count": conversion.get("nonwood_point_count"),
        "predicted_tree_count": summary["predicted_tree_count"],
        "predicted_tree_points": summary["predicted_tree_points"],
        "min_tree_points": summary["min_tree_points"],
        "max_tree_points": summary["max_tree_points"],
        "mean_tree_points": summary["mean_tree_points"],
        "runtime_seconds": run.get("runtime_seconds"),
        "peak_memory_gb": run.get("peak_memory_gb"),
        "return_code": return_code,
        "status": status,
        "output_dir": summary["output_directory"],
    }


def write_combined(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else [
        "plot_name",
        "input_point_count",
        "retained_point_count",
        "dropped_unknown_count",
        "wood_point_count",
        "nonwood_point_count",
        "predicted_tree_count",
        "predicted_tree_points",
        "min_tree_points",
        "max_tree_points",
        "mean_tree_points",
        "runtime_seconds",
        "peak_memory_gb",
        "return_code",
        "status",
        "output_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_all(config_path: Path) -> tuple[list[dict[str, Any]], Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = Path(config["project"]["barkla_root"]).expanduser().resolve()
    predictions_root = resolve_path(config["outputs"]["predictions_root"], project_root)
    metadata_root = resolve_path(config["outputs"]["output_metadata_root"], project_root)
    conversion_metadata_root = resolve_path(
        config["outputs"].get(
            "conversion_metadata_root",
            "results/metadata/tls2trees_conversions",
        ),
        project_root,
    )
    run_metadata_root = resolve_path(config["outputs"]["run_metadata_root"], project_root)
    tables_root = resolve_path(config["outputs"]["tables_root"], project_root)
    combined_path = resolve_path(config["outputs"]["combined_prediction_summary"], project_root)

    summaries: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for plot_name in config["dataset"]["plots"]:
        summary = summarise_plot(
            plot_name,
            predictions_root / plot_name,
            metadata_root / f"{plot_name}_summary.json",
            tables_root / f"tls2trees_{plot_name}_tree_summary.csv",
        )
        summaries.append(summary)
        combined_rows.append(
            combined_row(
                summary,
                read_json(conversion_metadata_root / f"{plot_name}_conversion.json"),
                read_json(run_metadata_root / f"{plot_name}_run.json"),
            )
        )
    write_combined(combined_path, combined_rows)
    return summaries, combined_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise TLS2trees .leafoff.ply predictions.")
    parser.add_argument("--plot-name")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", default="configs/frdr_tls2trees_benchmark.yml")
    parser.add_argument("--all-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    if args.all_plots:
        if args.plot_name or args.output_dir:
            raise SystemExit("--all-plots cannot be combined with --plot-name or --output-dir")
        summaries, combined_path = summarise_all(config_path)
        completed = sum(summary["status"] == "complete" for summary in summaries)
        print(f"Summarised {len(summaries)} configured plots; complete: {completed}")
        print(f"Combined table: {combined_path}")
        return 1 if any(summary["status"] == "completed_with_errors" for summary in summaries) else 0

    if not args.plot_name or not args.output_dir:
        raise SystemExit("Supply both --plot-name and --output-dir, or use --all-plots")
    output_dir = resolve_path(args.output_dir)
    json_path = ROOT / "results/metadata/tls2trees_outputs" / f"{args.plot_name}_summary.json"
    csv_path = ROOT / "results/tables" / f"tls2trees_{args.plot_name}_tree_summary.csv"
    summary = summarise_plot(args.plot_name, output_dir, json_path, csv_path)
    print(f"Plot: {args.plot_name}")
    print(f"Status: {summary['status']}")
    print(f"Predicted trees: {summary['leafoff_file_count']}")
    print(f"Summary JSON: {json_path}")
    print(f"Tree table: {csv_path}")
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
