"""Build a deterministic, local-only visual review for a development prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import laspy
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REQUIRED_ARRAYS = {
    "classification",
    "pred_classification",
    "pred_tree_id",
    "source_row_index",
    "target_tree_id",
}
TREE_CLASSES = (4, 5, 6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(mask: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rows = np.flatnonzero(mask)
    if rows.size <= maximum:
        return rows
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(rows, size=maximum, replace=False))


def instance_colours(instance_ids: np.ndarray) -> np.ndarray:
    palette = plt.get_cmap("tab20").colors
    colours = np.empty((instance_ids.size, 4), dtype=np.float32)
    colours[:] = (0.78, 0.78, 0.78, 0.25)
    positive = instance_ids > 0
    if np.any(positive):
        indices = (
            np.asarray(instance_ids[positive], dtype=np.uint64)
            * np.uint64(2_654_435_761)
        ) % np.uint64(len(palette))
        colours[positive, :3] = np.asarray(palette)[indices.astype(np.int64)]
        colours[positive, 3] = 0.9
    return colours


def validate_prediction(
    source_las: Path, prediction_npz: Path
) -> tuple[Any, dict[str, np.ndarray]]:
    cloud = laspy.read(source_las)
    with np.load(prediction_npz, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(
                "aligned prediction arrays differ; "
                f"observed={sorted(archive.files)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    point_count = len(cloud.points)
    if any(array.shape != (point_count,) for array in arrays.values()):
        raise ValueError("aligned prediction length differs from source LAS")
    expected_rows = np.arange(point_count, dtype=np.int64)
    if not np.array_equal(arrays["source_row_index"], expected_rows):
        raise ValueError("source_row_index is not exact zero-based source order")
    return cloud, arrays


def build_review(
    *,
    source_las: Path,
    prediction_npz: Path,
    relative_path: str,
    output_png: Path,
    output_json: Path,
    maximum_union_points: int,
    maximum_context_points: int,
    seed: int,
) -> dict[str, Any]:
    for output in (output_png, output_json):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    cloud, arrays = validate_prediction(source_las, prediction_npz)
    classification = np.asarray(arrays["classification"], dtype=np.int64)
    pred_classification = np.asarray(
        arrays["pred_classification"], dtype=np.int64
    )
    target_tree_id = np.asarray(arrays["target_tree_id"], dtype=np.int64)
    pred_tree_id = np.asarray(arrays["pred_tree_id"], dtype=np.int64)
    target_mask = np.isin(classification, TREE_CLASSES) & (target_tree_id > 0)
    prediction_mask = np.isin(pred_classification, TREE_CLASSES) & (
        pred_tree_id > 0
    )
    union_mask = target_mask | prediction_mask
    union_rows = selected_rows(union_mask, maximum_union_points, seed)
    context_rows = selected_rows(
        np.ones(len(classification), dtype=bool),
        maximum_context_points,
        seed + 1,
    )

    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    x -= np.min(x)
    y -= np.min(y)
    z -= np.min(z)

    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    panels = (
        (axes[0, 0], target_tree_id, target_mask, x, y, "Reference instances — XY"),
        (axes[0, 1], pred_tree_id, prediction_mask, x, y, "ForAINet instances — XY"),
        (axes[1, 0], target_tree_id, target_mask, x, z, "Reference instances — XZ"),
        (axes[1, 1], pred_tree_id, prediction_mask, x, z, "ForAINet instances — XZ"),
    )
    for axis, instances, tree_mask, horizontal, vertical, title in panels:
        axis.scatter(
            horizontal[context_rows],
            vertical[context_rows],
            c="#dedede",
            s=0.15,
            linewidths=0,
            rasterized=True,
        )
        panel_ids = np.where(tree_mask[union_rows], instances[union_rows], 0)
        axis.scatter(
            horizontal[union_rows],
            vertical[union_rows],
            c=instance_colours(panel_ids),
            s=0.35,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("relative X (m)")
        axis.set_ylabel("relative Y (m)" if title.endswith("XY") else "relative Z (m)")
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "ForAINet development alignment review\n"
        f"{relative_path} — exact source-row correspondence",
        fontsize=14,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)

    payload = {
        "schema": "forainet_manual_alignment_review_v1",
        "status": "waiting_manual_confirmation",
        "relative_path": relative_path,
        "split": "dev",
        "source_las_sha256": sha256(source_las),
        "aligned_prediction_sha256": sha256(prediction_npz),
        "figure_sha256": sha256(output_png),
        "point_count": len(classification),
        "reference_tree_count": int(np.unique(target_tree_id[target_mask]).size),
        "predicted_tree_count": int(np.unique(pred_tree_id[prediction_mask]).size),
        "union_point_count": int(np.count_nonzero(union_mask)),
        "plotted_union_point_count": int(union_rows.size),
        "plotted_context_point_count": int(context_rows.size),
        "sampling_seed": seed,
        "source_row_index_exact": True,
        "coordinate_matching": False,
        "coordinates_in_figure": "plot-relative; global offsets removed",
        "reference_labels_supplied_to_model": False,
        "review_instruction": (
            "Confirm that reference and predicted crowns occupy the same plot "
            "geometry in both XY and XZ views; score plausibility is not the gate."
        ),
    }
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-las", required=True, type=Path)
    parser.add_argument("--prediction-npz", required=True, type=Path)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--maximum-union-points", type=int, default=250_000)
    parser.add_argument("--maximum-context-points", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.maximum_union_points < 1 or args.maximum_context_points < 1:
        raise ValueError("sample limits must be positive")
    build_review(
        source_las=args.source_las,
        prediction_npz=args.prediction_npz,
        relative_path=args.relative_path,
        output_png=args.output_png,
        output_json=args.output_json,
        maximum_union_points=args.maximum_union_points,
        maximum_context_points=args.maximum_context_points,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
