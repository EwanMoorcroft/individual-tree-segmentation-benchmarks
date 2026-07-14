"""Run the pinned TreeLearn pipeline with one explicit empty-group guard."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Callable

import numpy as np


POLICY = "map_all_unassigned_to_background_when_initial_grouping_is_empty"
EMPTY_DIAGNOSTICS = {"cluster_coords_initial", "cluster_coords"}


def guarded_assignment(
    coords: np.ndarray,
    predictions: np.ndarray,
    remaining_points_idx: int,
    n_neighbors: int = 5,
    *,
    background_label: int = 0,
    original: Callable | None = None,
    on_empty: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Map unassigned labels to upstream background when no cluster exists."""

    values = np.asarray(predictions)
    reference_idx = np.flatnonzero(values != remaining_points_idx)
    if reference_idx.size == 0:
        if on_empty is not None:
            on_empty(int(np.count_nonzero(values == remaining_points_idx)))
        result = values.astype(np.int64, copy=True)
        result[result == remaining_points_idx] = background_label
        return result
    if original is None:
        raise ValueError("The upstream assignment function is required")
    return original(coords, predictions, remaining_points_idx, n_neighbors)


def guarded_save_data(
    data: np.ndarray,
    save_format: str,
    save_name: str,
    save_folder: str,
    use_offset: bool = True,
    *,
    original: Callable,
    on_empty: Callable[[str, str], None] | None = None,
):
    """Skip only empty optional cluster visualizations."""

    values = np.asarray(data)
    if values.shape[0] == 0 and save_name in EMPTY_DIAGNOSTICS:
        if on_empty is not None:
            on_empty(save_name, save_format)
        return None
    return original(data, save_format, save_name, save_folder, use_offset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    args = parser.parse_args()
    pipeline = args.pipeline.expanduser().resolve()
    config = args.config.expanduser().resolve()
    marker = args.marker.expanduser().resolve()
    if not pipeline.is_file() or pipeline.name != "pipeline.py":
        raise FileNotFoundError(f"Pinned TreeLearn pipeline missing: {pipeline}")
    if not config.is_file():
        raise FileNotFoundError(config)

    import tree_learn.util as treelearn_util

    original = treelearn_util.assign_remaining_points_nearest_neighbor
    original_save_data = treelearn_util.save_data
    record = {
        "schema_version": 1,
        "status": "guard_active_not_triggered",
        "policy": POLICY,
        "triggered": False,
        "query_points": 0,
        "reference_points": None,
        "skipped_empty_diagnostics": [],
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    def on_empty(query_points: int) -> None:
        record.update(
            {
                "status": "empty_group_mapped_to_background",
                "triggered": True,
                "query_points": query_points,
                "reference_points": 0,
            }
        )
        marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(
            "treelearn_empty_group_guard="
            "all_unassigned_mapped_to_background",
            flush=True,
        )

    def replacement(coords, predictions, remaining_points_idx, n_neighbors=5):
        return guarded_assignment(
            coords,
            predictions,
            remaining_points_idx,
            n_neighbors,
            original=original,
            on_empty=on_empty,
        )

    def on_empty_diagnostic(save_name: str, save_format: str) -> None:
        record["skipped_empty_diagnostics"].append(
            {"save_name": save_name, "save_format": save_format}
        )
        marker.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"treelearn_empty_diagnostic_skipped={save_name}.{save_format}", flush=True)

    def save_data_replacement(
        data, save_format, save_name, save_folder, use_offset=True
    ):
        return guarded_save_data(
            data,
            save_format,
            save_name,
            save_folder,
            use_offset,
            original=original_save_data,
            on_empty=on_empty_diagnostic,
        )

    treelearn_util.assign_remaining_points_nearest_neighbor = replacement
    treelearn_util.save_data = save_data_replacement
    old_argv = sys.argv
    try:
        sys.argv = [str(pipeline), "--config", str(config)]
        runpy.run_path(str(pipeline), run_name="__main__")
    finally:
        sys.argv = old_argv
        treelearn_util.assign_remaining_points_nearest_neighbor = original
        treelearn_util.save_data = original_save_data
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
