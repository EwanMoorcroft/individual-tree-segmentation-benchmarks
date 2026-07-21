"""Project raw TLS2trees tree files back to FOR-instance source rows."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "methods" / "tls2trees" / "scripts" / "runtime"
SRC = ROOT / "src"
for entry in (RUNTIME, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from benchmark.ply_io import read_ply_vertices
from for_instance_published_common import (
    EXPECTED_SPLIT,
    EXPECTED_VARIANT,
    peak_rss_gb,
    resolve_development_plot_context,
    resolve_held_out_test_plot_context,
    resolve_plot_context,
    sha256,
    utc_now,
    write_json,
)


TARGET_PATTERNS = {
    "leaf_off": "*.leafoff.ply",
    "leaf_on": "*.leafon.ply",
}
ALIGNMENT_SCHEMA = "tls2trees_for_instance_alignment"


def load_raw_predictions(
    raw_root: Path, target: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pattern = TARGET_PATTERNS[target]
    names: list[str] = []
    coordinates: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for owner, path in enumerate(sorted(raw_root.rglob(pattern)), start=1):
        header, points = read_ply_vertices(path, columns=["x", "y", "z"])
        if header.vertex_count == 0:
            raise ValueError(f"Raw predicted tree file is empty: {path}")
        xyz = np.column_stack((points["x"], points["y"], points["z"]))
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"Raw prediction contains non-finite coordinates: {path}")
        names.append(str(path.relative_to(raw_root)))
        coordinates.append(xyz.astype(np.float64, copy=False))
        owners.append(np.full(len(xyz), owner, dtype=np.int64))
    if not coordinates:
        return names, np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    return names, np.concatenate(coordinates), np.concatenate(owners)


def assign_raw_to_representatives(
    prediction_xyz: np.ndarray,
    prediction_owner: np.ndarray,
    representative_xyz: np.ndarray,
    tolerance_m: float,
    *,
    owner_groups: list[list[int]] | None = None,
    merge_cross_tree_conflicts: bool = False,
    conflict_support_by_owner: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("coordinate_tolerance_m must be finite and positive")
    if len(prediction_xyz) != len(prediction_owner):
        raise ValueError("Raw prediction coordinates and owner labels are not aligned")
    if len(prediction_xyz) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {
            "raw_prediction_point_count": 0,
            "matched_raw_prediction_point_count": 0,
            "unmatched_raw_prediction_point_count": 0,
            "ambiguous_raw_prediction_point_count": 0,
            "within_tree_duplicate_representative_count": 0,
            "across_tree_conflicting_representative_count": 0,
            "post_reconciliation_duplicate_representative_count": 0,
            "post_reconciliation_conflicting_representative_count": 0,
            "raw_prediction_tree_count": 0,
            "resolved_prediction_tree_count": 0,
            "merged_owner_component_count": 0,
            "leaf_on_conflicting_representative_count_resolved": 0,
            "cross_tree_conflict_resolution": "not_required",
            "owner_components": [],
            "owner_reconciliation": "not_required",
            "maximum_coordinate_distance_m": None,
            "mean_coordinate_distance_m": None,
            "status": "passed",
        }
    if len(representative_xyz) == 0:
        raise ValueError("Source map contains no representatives")
    if owner_groups is not None and merge_cross_tree_conflicts:
        raise ValueError("Cannot provide owner_groups and request conflict merging")
    if len(prediction_owner) and (
        prediction_owner.min() < 1
        or not np.array_equal(
            np.unique(prediction_owner),
            np.arange(1, int(prediction_owner.max()) + 1),
        )
    ):
        raise ValueError("Raw prediction owners must be contiguous positive integers")

    neighbour_count = min(2, len(representative_xyz))
    distances, indices = cKDTree(representative_xyz).query(
        prediction_xyz,
        k=neighbour_count,
        distance_upper_bound=tolerance_m,
        workers=1,
    )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if neighbour_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    nearest_distance = distances[:, 0]
    nearest_index = indices[:, 0]
    unmatched = ~np.isfinite(nearest_distance) | (nearest_index >= len(representative_xyz))
    tie_atol = max(1e-12, tolerance_m * 1e-9)
    ambiguous = (
        np.isfinite(distances[:, 1])
        & np.isclose(distances[:, 0], distances[:, 1], rtol=0.0, atol=tie_atol)
        if neighbour_count == 2
        else np.zeros(len(prediction_xyz), dtype=bool)
    )
    if np.any(unmatched) or np.any(ambiguous):
        raise ValueError(
            "Raw prediction-to-representative alignment failed: "
            f"unmatched={int(np.count_nonzero(unmatched))}, "
            f"ambiguous={int(np.count_nonzero(ambiguous))}"
        )

    pairs = np.column_stack((prediction_owner, nearest_index))
    within_duplicate_count = len(pairs) - len(np.unique(pairs, axis=0))
    representative_owner_pairs = np.unique(
        np.column_stack((nearest_index, prediction_owner)), axis=0
    )
    _, owner_counts = np.unique(
        representative_owner_pairs[:, 0], return_counts=True
    )
    across_conflict_count = int(np.count_nonzero(owner_counts > 1))
    if within_duplicate_count:
        raise ValueError(
            "Raw prediction ownership is not unique: "
            f"within_tree_duplicates={within_duplicate_count}, "
            f"across_tree_conflicts={across_conflict_count}"
        )

    tree_count = int(prediction_owner.max())
    if owner_groups is None and merge_cross_tree_conflicts:
        parent = np.arange(tree_count + 1, dtype=np.int64)

        def find(owner: int) -> int:
            while parent[owner] != owner:
                parent[owner] = parent[parent[owner]]
                owner = int(parent[owner])
            return owner

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            parent[max(left_root, right_root)] = min(left_root, right_root)

        unique_representatives, starts = np.unique(
            representative_owner_pairs[:, 0], return_index=True
        )
        del unique_representatives
        ends = np.r_[starts[1:], len(representative_owner_pairs)]
        for start, end in zip(starts, ends, strict=True):
            owners = representative_owner_pairs[start:end, 1]
            for owner in owners[1:]:
                union(int(owners[0]), int(owner))
        components: dict[int, list[int]] = {}
        for owner in range(1, tree_count + 1):
            components.setdefault(find(owner), []).append(owner)
        owner_groups = sorted(components.values(), key=lambda group: group[0])
        reconciliation = (
            "shared_representative_connected_components"
            if across_conflict_count
            else "not_required"
        )
    elif owner_groups is None:
        if across_conflict_count:
            raise ValueError(
                "Raw prediction ownership is not unique: "
                f"within_tree_duplicates={within_duplicate_count}, "
                f"across_tree_conflicts={across_conflict_count}"
            )
        owner_groups = [[owner] for owner in range(1, tree_count + 1)]
        reconciliation = "not_required"
    else:
        reconciliation = "provided_leaf_off_owner_components"

    flattened = [owner for group in owner_groups for owner in group]
    if sorted(flattened) != list(range(1, tree_count + 1)) or len(flattened) != len(
        set(flattened)
    ):
        raise ValueError(
            "owner_groups must partition every raw prediction owner exactly once"
        )
    owner_remap = np.zeros(tree_count + 1, dtype=np.int64)
    for resolved_owner, group in enumerate(owner_groups, start=1):
        owner_remap[np.asarray(group, dtype=np.int64)] = resolved_owner
    resolved_prediction_owner = owner_remap[prediction_owner]
    resolved_pairs = np.column_stack((nearest_index, resolved_prediction_owner))
    unique_resolved_pairs = np.unique(resolved_pairs, axis=0)
    resolved_representatives, resolved_owner_counts = np.unique(
        unique_resolved_pairs[:, 0], return_counts=True
    )
    remaining_conflicts = int(np.count_nonzero(resolved_owner_counts > 1))
    resolved_leaf_on_conflicts = 0
    if remaining_conflicts:
        if conflict_support_by_owner is None:
            raise ValueError(
                "Raw prediction ownership remains conflicting after leaf-off "
                f"reconciliation: across_tree_conflicts={remaining_conflicts}"
            )
        resolved_owner_ids = set(range(1, len(owner_groups) + 1))
        if set(conflict_support_by_owner) != resolved_owner_ids:
            raise ValueError(
                "Leaf-off conflict support must cover every resolved prediction owner"
            )
        support_trees: dict[int, cKDTree] = {}
        for owner, support_indices in conflict_support_by_owner.items():
            support_indices = np.asarray(support_indices, dtype=np.int64)
            if len(support_indices) == 0 or np.any(support_indices < 0) or np.any(
                support_indices >= len(representative_xyz)
            ):
                raise ValueError(f"Invalid leaf-off conflict support for owner {owner}")
            support_trees[owner] = cKDTree(representative_xyz[support_indices])

        conflicting_representatives = resolved_representatives[
            resolved_owner_counts > 1
        ]
        for representative in conflicting_representatives:
            candidates = np.unique(
                resolved_prediction_owner[nearest_index == representative]
            )
            ranked = [
                (
                    float(
                        support_trees[int(owner)].query(
                            representative_xyz[int(representative)], k=1
                        )[0]
                    ),
                    int(owner),
                )
                for owner in candidates
            ]
            winner = min(ranked)[1]
            resolved_prediction_owner[nearest_index == representative] = winner

        resolved_pairs = np.column_stack((nearest_index, resolved_prediction_owner))
        unique_resolved_pairs = np.unique(resolved_pairs, axis=0)
        resolved_representatives, resolved_owner_counts = np.unique(
            unique_resolved_pairs[:, 0], return_counts=True
        )
        del resolved_representatives
        unresolved_conflicts = int(np.count_nonzero(resolved_owner_counts > 1))
        if unresolved_conflicts:
            raise RuntimeError(
                "Leaf-off-supported crown ownership did not resolve every conflict: "
                f"remaining={unresolved_conflicts}"
            )
        resolved_leaf_on_conflicts = remaining_conflicts
        remaining_conflicts = 0
        reconciliation += "_then_nearest_leaf_off_support"
    post_reconciliation_duplicates = len(resolved_pairs) - len(unique_resolved_pairs)
    return nearest_index, resolved_prediction_owner, {
        "raw_prediction_point_count": len(prediction_xyz),
        "matched_raw_prediction_point_count": len(prediction_xyz),
        "unmatched_raw_prediction_point_count": 0,
        "ambiguous_raw_prediction_point_count": 0,
        "within_tree_duplicate_representative_count": int(within_duplicate_count),
        "across_tree_conflicting_representative_count": across_conflict_count,
        "post_reconciliation_duplicate_representative_count": int(
            post_reconciliation_duplicates
        ),
        "post_reconciliation_conflicting_representative_count": remaining_conflicts,
        "raw_prediction_tree_count": tree_count,
        "resolved_prediction_tree_count": len(owner_groups),
        "merged_owner_component_count": sum(len(group) > 1 for group in owner_groups),
        "leaf_on_conflicting_representative_count_resolved": resolved_leaf_on_conflicts,
        "cross_tree_conflict_resolution": (
            "nearest_leaf_off_representative_then_lowest_owner"
            if resolved_leaf_on_conflicts
            else "not_required"
        ),
        "owner_components": owner_groups,
        "owner_reconciliation": reconciliation,
        "maximum_coordinate_distance_m": float(np.max(nearest_distance)),
        "mean_coordinate_distance_m": float(np.mean(nearest_distance)),
        "status": "passed",
    }


def adapt_target(
    *,
    target: str,
    raw_root: Path,
    aligned_root: Path,
    source_map_path: Path,
    input_las: Path,
    input_las_sha256: str,
    local_origin_xyz: np.ndarray,
    las_scales: np.ndarray,
    las_offsets: np.ndarray,
    tolerance_m: float,
    owner_groups: list[list[int]] | None = None,
    merge_cross_tree_conflicts: bool = False,
    conflict_support_by_owner: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    names, raw_xyz, raw_owner = load_raw_predictions(raw_root, target)
    with np.load(source_map_path) as source_map:
        source_row_index = np.asarray(source_map["source_row_index"], dtype=np.int64)
        source_to_rep = np.asarray(
            source_map["source_to_representative_index"], dtype=np.int64
        )
        representative_xyz = np.asarray(
            source_map["representative_local_xyz"], dtype=np.float64
        )
    if not np.array_equal(source_row_index, np.arange(len(source_row_index))):
        raise ValueError("source_map source_row_index is not zero-based source order")
    if len(source_to_rep) != len(source_row_index):
        raise ValueError("source_map projection length does not equal source row count")
    if len(source_to_rep) and (
        source_to_rep.min() < 0 or source_to_rep.max() >= len(representative_xyz)
    ):
        raise ValueError("source_map contains an out-of-range representative index")

    raw_to_rep, resolved_raw_owner, diagnostics = assign_raw_to_representatives(
        raw_xyz,
        raw_owner,
        representative_xyz,
        tolerance_m,
        owner_groups=owner_groups,
        merge_cross_tree_conflicts=merge_cross_tree_conflicts,
        conflict_support_by_owner=conflict_support_by_owner,
    )
    resolved_groups = diagnostics["owner_components"]
    prediction_source_files = [
        [names[owner - 1] for owner in group] for group in resolved_groups
    ]
    resolved_names = [
        files[0] if len(files) == 1 else "merged::" + "|".join(files)
        for files in prediction_source_files
    ]
    representative_owner = np.zeros(len(representative_xyz), dtype=np.int64)
    if len(raw_to_rep):
        representative_owner[raw_to_rep] = resolved_raw_owner
    source_prediction = representative_owner[source_to_rep]
    used = np.unique(source_prediction[source_prediction > 0])
    if len(resolved_names) and not np.array_equal(
        used, np.arange(1, len(resolved_names) + 1)
    ):
        raise ValueError("One or more raw predicted trees has no projected source rows")

    target_root = aligned_root / target
    target_root.mkdir(parents=True)
    aligned_npz = target_root / "source_row_predictions.npz"
    np.savez_compressed(
        aligned_npz,
        source_row_index=source_row_index,
        predicted_instance_id=source_prediction,
        predicted_semantic_label=(source_prediction > 0).astype(np.int8),
        prediction_names=np.asarray(resolved_names, dtype=str),
        source_las_sha256=np.asarray(input_las_sha256),
        source_map_sha256=np.asarray(sha256(source_map_path)),
    )
    alignment_metadata_path = target_root / "alignment_metadata.json"
    metadata = {
        "schema_version": ALIGNMENT_SCHEMA,
        "status": "passed",
        "target": target,
        "point_correspondence": "source_row_via_voxel_representative",
        "raw_coordinate_evaluation_permitted": False,
        "aligned_prediction_npz": str(aligned_npz),
        "aligned_prediction_npz_sha256": sha256(aligned_npz),
        "source_map": str(source_map_path),
        "source_map_sha256": sha256(source_map_path),
        "source_las_path": str(input_las),
        "source_las_sha256": input_las_sha256,
        "source_row_count": len(source_row_index),
        "predicted_source_row_count": int(np.count_nonzero(source_prediction > 0)),
        "background_source_row_count": int(np.count_nonzero(source_prediction == 0)),
        "prediction_instance_count": len(resolved_names),
        "prediction_names": resolved_names,
        "prediction_source_files": prediction_source_files,
        "coordinate_frame": {
            "source": "source_crs",
            "aligned_predictions": "source_row_indices_no_coordinates",
            "raw_predictions": "grid_aligned_local_shift",
            "units": "metres",
            "local_shift_m": local_origin_xyz.tolist(),
            "predictions_restored_to_source": False,
        },
        "source_las": {
            "scale_m": las_scales.tolist(),
            "offset_m": las_offsets.tolist(),
        },
        "matching": {
            "coordinate_tolerance_m": tolerance_m,
            "distance_metric": "euclidean",
            "raw_to_representative_assignment": (
                "unique_nearest_within_tolerance_then_nearest_leaf_off_support"
                if diagnostics["leaf_on_conflicting_representative_count_resolved"]
                else (
                    "unique_nearest_within_tolerance"
                    if diagnostics["owner_reconciliation"] == "not_required"
                    else "unique_nearest_within_tolerance_then_leaf_off_shared_"
                    "representative_component_reconciliation"
                )
            ),
            "source_projection": "all_source_rows_in_representative_voxel",
        },
        "raw_alignment_diagnostics": diagnostics,
    }
    write_json(alignment_metadata_path, metadata)
    metadata["alignment_metadata"] = str(alignment_metadata_path)
    return metadata


def run_adapter(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str = EXPECTED_VARIANT,
    split: str = EXPECTED_SPLIT,
    coordinate_tolerance_m: float = 0.001,
    allow_held_out_test: bool = False,
) -> dict[str, Any]:
    started_at_utc = utc_now()
    started = time.perf_counter()
    is_held_out_test = split == "test"
    if is_held_out_test:
        if not allow_held_out_test:
            raise ValueError("Held-out adaptation requires --allow-held-out-test")
        plot_root, row = resolve_held_out_test_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
        )
    elif variant == EXPECTED_VARIANT:
        plot_root, row = resolve_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
            split=split,
        )
    else:
        if split != EXPECTED_SPLIT:
            raise ValueError("TLS2trees prediction adaptation remains development-only")
        plot_root, row = resolve_development_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=run_id,
            variant=variant,
            allowed_variants={"development_tuned"},
        )
    instance_metadata_path = plot_root / "metadata" / "instance_run.json"
    if not instance_metadata_path.is_file():
        raise FileNotFoundError(f"Instance run metadata does not exist: {instance_metadata_path}")
    instance_metadata = json.loads(instance_metadata_path.read_text(encoding="utf-8"))
    if instance_metadata.get("status") not in {"completed", "completed_no_predictions"}:
        raise ValueError("Instance run is not complete")
    conversion_path = plot_root / "converted" / "conversion_metadata.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    input_las = Path(row["input_las"]).expanduser().resolve()
    if sha256(input_las) != conversion["input_sha256"]:
        raise RuntimeError("Source LAS checksum differs from conversion metadata")
    source_map_path = Path(conversion["source_map"])
    if sha256(source_map_path) != conversion["source_map_sha256"]:
        raise RuntimeError("Source-map checksum differs from conversion metadata")
    raw_root = Path(instance_metadata["raw_prediction_root"])
    aligned_root = plot_root / "predictions" / "aligned"
    if aligned_root.exists():
        raise FileExistsError(
            f"Aligned prediction root already exists; use a new run_id: {aligned_root}"
        )

    local_origin = np.asarray(
        conversion["coordinate_frame"]["local_origin_xyz"], dtype=np.float64
    )
    las_scales = np.asarray(conversion["coordinate_frame"]["las_scales"], dtype=np.float64)
    las_offsets = np.asarray(conversion["coordinate_frame"]["las_offsets"], dtype=np.float64)
    results: dict[str, Any] = {}
    try:
        leaf_off_names, _, _ = load_raw_predictions(raw_root, "leaf_off")
        leaf_on_names, _, _ = load_raw_predictions(raw_root, "leaf_on")
        leaf_off_keys = [name.removesuffix(".leafoff.ply") for name in leaf_off_names]
        leaf_on_keys = [name.removesuffix(".leafon.ply") for name in leaf_on_names]
        if leaf_off_keys != leaf_on_keys:
            raise ValueError("Leaf-off and leaf-on raw prediction files do not correspond")

        results["leaf_off"] = adapt_target(
            target="leaf_off",
            raw_root=raw_root,
            aligned_root=aligned_root,
            source_map_path=source_map_path,
            input_las=input_las,
            input_las_sha256=conversion["input_sha256"],
            local_origin_xyz=local_origin,
            las_scales=las_scales,
            las_offsets=las_offsets,
            tolerance_m=coordinate_tolerance_m,
            merge_cross_tree_conflicts=True,
        )
        owner_groups = results["leaf_off"]["raw_alignment_diagnostics"][
            "owner_components"
        ]
        _, leaf_off_xyz, leaf_off_owner = load_raw_predictions(raw_root, "leaf_off")
        with np.load(source_map_path) as source_map:
            representative_xyz = np.asarray(
                source_map["representative_local_xyz"], dtype=np.float64
            )
        leaf_off_to_rep, resolved_leaf_off_owner, _ = assign_raw_to_representatives(
            leaf_off_xyz,
            leaf_off_owner,
            representative_xyz,
            coordinate_tolerance_m,
            owner_groups=owner_groups,
        )
        conflict_support_by_owner = {
            owner: np.unique(leaf_off_to_rep[resolved_leaf_off_owner == owner])
            for owner in range(1, len(owner_groups) + 1)
        }
        results["leaf_on"] = adapt_target(
            target="leaf_on",
            raw_root=raw_root,
            aligned_root=aligned_root,
            source_map_path=source_map_path,
            input_las=input_las,
            input_las_sha256=conversion["input_sha256"],
            local_origin_xyz=local_origin,
            las_scales=las_scales,
            las_offsets=las_offsets,
            tolerance_m=coordinate_tolerance_m,
            owner_groups=owner_groups,
            conflict_support_by_owner=conflict_support_by_owner,
        )
    except Exception:
        if aligned_root.exists():
            shutil.rmtree(aligned_root)
        raise
    payload = {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": started_at_utc,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_gb": peak_rss_gb(),
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": variant,
        "split": split,
        "run_id": run_id,
        "task_index": task_index,
        "relative_path": row["relative_path"],
        "safe_plot_id": row["safe_plot_id"],
        "coordinate_tolerance_m": coordinate_tolerance_m,
        "compatibility_patches": [
            "cross_tile_prediction_ownership_reconciled_from_leaf_off_support"
        ],
        "targets": results,
        "held_out_test_accessed": is_held_out_test,
    }
    write_json(plot_root / "metadata" / "adapter_run.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project raw TLS2trees tree files to FOR-instance source rows."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default=EXPECTED_VARIANT)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    parser.add_argument("--coordinate-tolerance-m", type=float, default=0.001)
    parser.add_argument("--allow-held-out-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = run_adapter(
            manifest_path=Path(args.manifest_json),
            task_index=args.task_index,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            variant=args.variant,
            split=args.split,
            coordinate_tolerance_m=args.coordinate_tolerance_m,
            allow_held_out_test=args.allow_held_out_test,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    for target, result in payload["targets"].items():
        print(f"{target}_instances={result['prediction_instance_count']}")
        print(f"{target}_predicted_source_rows={result['predicted_source_row_count']}")
    print(
        "held_out_test_accessed="
        + str(bool(payload["held_out_test_accessed"])).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
