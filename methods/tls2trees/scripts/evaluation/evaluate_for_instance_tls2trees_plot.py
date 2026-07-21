"""Evaluate one target-explicit TLS2trees FOR-instance plot.

TLS2trees writes one PLY file per predicted tree and does not retain arbitrary
source-row fields.  This evaluator therefore aligns every prediction point to
an indexed source-cloud row before calculating pointwise instance metrics.  It
never collapses repeated coordinates: reference and prediction multiplicity is
retained, and a sparse maximum-cardinality assignment enforces that one source
row can be used by at most one predicted point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.instance_metrics import (  # noqa: E402
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)
from benchmark.ply_io import read_ply_vertices  # noqa: E402


TARGET_CONTRACTS: dict[str, dict[str, Any]] = {
    "leaf_off": {
        "prediction_pattern": "*.leafoff.ply",
        "included_semantic_classes": (4, 6),
        "excluded_semantic_classes": (0, 1, 2, 5),
        "ignored_semantic_classes": (3,),
    },
    "leaf_on": {
        "prediction_pattern": "*.leafon.ply",
        "included_semantic_classes": (4, 5, 6),
        "excluded_semantic_classes": (0, 1, 2),
        "ignored_semantic_classes": (3,),
    },
}
ALIGNMENT_SCHEMA = "tls2trees_for_instance_alignment"
EVALUATION_PROTOCOL = "for_instance_tls2trees_coordinate_class3_ignore"
SOURCE_ROW_EVALUATION_PROTOCOL = "for_instance_tls2trees_source_row_class3_ignore"
EVALUATION_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
DEFAULT_IOU_THRESHOLD = 0.5


def resolve_path(path_text: str) -> Path:
    """Resolve an absolute or repository-relative path."""

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_contract(target: str) -> dict[str, Any]:
    """Return a copy of the explicit leaf target contract."""

    try:
        contract = TARGET_CONTRACTS[target]
    except KeyError as exc:
        raise ValueError(
            f"Target must be one of {sorted(TARGET_CONTRACTS)}; received {target!r}"
        ) from exc
    return {
        "target": target,
        "prediction_pattern": contract["prediction_pattern"],
        "reference_instance_field": "treeID",
        "reference_semantic_field": "classification",
        "included_semantic_classes": list(contract["included_semantic_classes"]),
        "excluded_semantic_classes": list(contract["excluded_semantic_classes"]),
        "ignored_semantic_classes": list(contract["ignored_semantic_classes"]),
        "valid_reference_instance_rule": "treeID > 0",
        "invalid_reference_instance_rule": "treeID <= 0 is background",
        "reference_semantic_postfilter_applied_to_predictions": False,
        "ignored_semantic_class_mask_applied_to_predictions": True,
    }


def prediction_files(directory: Path, target: str) -> list[Path]:
    """Select only the requested target pattern, without fallback behaviour."""

    contract = target_contract(target)
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob(contract["prediction_pattern"])
        if path.is_file()
    )


def _as_xyz(coordinates: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} coordinates must have shape (point_count, 3)")
    if np.any(~np.isfinite(points)):
        raise ValueError(f"{label} coordinates contain non-finite values")
    return points


def _as_integral_vector(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    numeric = np.asarray(array, dtype=np.float64)
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.rint(numeric)):
        raise ValueError(f"{label} must contain finite integer values")
    return numeric.astype(np.int64)


def _triple(values: Any, label: str, *, positive: bool = False) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{label} must contain three finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{label} values must be greater than zero")
    return [float(value) for value in array]


def validate_alignment_metadata(
    metadata: Mapping[str, Any],
    actual_source_las_scale_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalise the coordinate-frame matching contract."""

    if metadata.get("schema_version") != ALIGNMENT_SCHEMA:
        raise ValueError(
            f"Alignment metadata schema_version must be {ALIGNMENT_SCHEMA!r}"
        )
    frame = metadata.get("coordinate_frame")
    source_las = metadata.get("source_las")
    matching = metadata.get("matching")
    if not isinstance(frame, Mapping):
        raise ValueError("Alignment metadata requires coordinate_frame mapping")
    if not isinstance(source_las, Mapping):
        raise ValueError("Alignment metadata requires source_las mapping")
    if not isinstance(matching, Mapping):
        raise ValueError("Alignment metadata requires matching mapping")

    source_frame = str(frame.get("source", "")).strip()
    prediction_frame = str(frame.get("predictions", "")).strip()
    if not source_frame or not prediction_frame:
        raise ValueError("Source and prediction coordinate frames must be named")
    if frame.get("units") != "metres":
        raise ValueError("Coordinate-frame units must be 'metres'")
    if frame.get("predictions_restored_to_source") is not True:
        raise ValueError("Predictions must be restored to the source coordinate frame")
    if prediction_frame not in {source_frame, "restored_source_crs"}:
        raise ValueError(
            "Prediction frame must equal the source frame or restored_source_crs"
        )

    source_scale = _triple(
        source_las.get("scale_m"), "source_las.scale_m", positive=True
    )
    source_offset = _triple(
        source_las.get("offset_m"), "source_las.offset_m"
    )
    local_shift = _triple(
        frame.get("local_shift_m"), "coordinate_frame.local_shift_m"
    )
    tolerance = float(matching.get("coordinate_tolerance_m", math.nan))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("matching.coordinate_tolerance_m must be finite and positive")
    if matching.get("distance_metric") != "euclidean":
        raise ValueError("matching.distance_metric must be 'euclidean'")

    if actual_source_las_scale_m is not None:
        actual_scale = _triple(
            actual_source_las_scale_m,
            "actual source LAS scale",
            positive=True,
        )
        if not np.allclose(source_scale, actual_scale, rtol=0.0, atol=1e-15):
            raise ValueError(
                "Alignment metadata source LAS scale does not match the reference header"
            )

    return {
        "schema_version": ALIGNMENT_SCHEMA,
        "coordinate_frame": {
            "source": source_frame,
            "predictions": prediction_frame,
            "units": "metres",
            "local_shift_m": local_shift,
            "predictions_restored_to_source": True,
        },
        "source_las": {
            "scale_m": source_scale,
            "offset_m": source_offset,
        },
        "matching": {
            "coordinate_tolerance_m": tolerance,
            "distance_metric": "euclidean",
            "source_assignment": "maximum_cardinality_one_to_one",
        },
    }


def validate_source_row_alignment_metadata(
    metadata: Mapping[str, Any],
    actual_source_las_scale_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Validate metadata for aligned labels that contain no prediction XYZ."""

    if metadata.get("schema_version") != ALIGNMENT_SCHEMA:
        raise ValueError(
            f"Alignment metadata schema_version must be {ALIGNMENT_SCHEMA!r}"
        )
    if metadata.get("point_correspondence") != "source_row_via_voxel_representative":
        raise ValueError("Source-row metadata has an unexpected point correspondence")
    if metadata.get("raw_coordinate_evaluation_permitted") is not False:
        raise ValueError("Source-row metadata must disable raw coordinate evaluation")
    frame = metadata.get("coordinate_frame")
    source_las = metadata.get("source_las")
    matching = metadata.get("matching")
    if not isinstance(frame, Mapping) or not isinstance(source_las, Mapping):
        raise ValueError("Source-row metadata requires coordinate and LAS mappings")
    if not isinstance(matching, Mapping):
        raise ValueError("Source-row metadata requires matching mapping")
    if frame.get("source") != "source_crs":
        raise ValueError("Source-row metadata must name the source CRS")
    if frame.get("aligned_predictions") != "source_row_indices_no_coordinates":
        raise ValueError("Aligned predictions must declare source-row indexing")
    if frame.get("raw_predictions") != "grid_aligned_local_shift":
        raise ValueError("Raw TLS2trees predictions must declare their local frame")
    if frame.get("predictions_restored_to_source") is not False:
        raise ValueError("Source-row predictions must not claim restored coordinates")
    if frame.get("units") != "metres":
        raise ValueError("Coordinate-frame units must be 'metres'")
    source_scale = _triple(
        source_las.get("scale_m"), "source_las.scale_m", positive=True
    )
    source_offset = _triple(source_las.get("offset_m"), "source_las.offset_m")
    local_shift = _triple(frame.get("local_shift_m"), "coordinate_frame.local_shift_m")
    tolerance = float(matching.get("coordinate_tolerance_m", math.nan))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("matching.coordinate_tolerance_m must be finite and positive")
    if matching.get("distance_metric") != "euclidean":
        raise ValueError("matching.distance_metric must be 'euclidean'")
    if actual_source_las_scale_m is not None:
        actual_scale = _triple(
            actual_source_las_scale_m, "actual source LAS scale", positive=True
        )
        if not np.allclose(source_scale, actual_scale, rtol=0.0, atol=1e-15):
            raise ValueError(
                "Alignment metadata source LAS scale does not match the reference header"
            )
    return {
        "schema_version": ALIGNMENT_SCHEMA,
        "point_correspondence": "source_row_via_voxel_representative",
        "raw_coordinate_evaluation_permitted": False,
        "coordinate_frame": {
            "source": "source_crs",
            "aligned_predictions": "source_row_indices_no_coordinates",
            "raw_predictions": "grid_aligned_local_shift",
            "units": "metres",
            "local_shift_m": local_shift,
            "predictions_restored_to_source": False,
        },
        "source_las": {"scale_m": source_scale, "offset_m": source_offset},
        "matching": {
            "coordinate_tolerance_m": tolerance,
            "distance_metric": "euclidean",
            "source_assignment": "unique_voxel_representative_then_source_row_projection",
        },
    }


def load_predictions(
    directory: Path,
    target: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load target-specific PLYs while preserving every coordinate row."""

    files = prediction_files(directory, target)
    names: list[str] = []
    chunks: list[np.ndarray] = []
    instance_chunks: list[np.ndarray] = []
    for instance_index, path in enumerate(files):
        header, points = read_ply_vertices(path, columns=["x", "y", "z"])
        if header.vertex_count == 0:
            raise ValueError(f"Predicted instance file is empty: {path}")
        coordinates = _as_xyz(
            np.column_stack([points["x"], points["y"], points["z"]]),
            str(path),
        )
        names.append(str(path.relative_to(directory)))
        chunks.append(coordinates)
        instance_chunks.append(
            np.full(len(coordinates), instance_index, dtype=np.int64)
        )
    if not chunks:
        return names, np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
    return names, np.concatenate(chunks), np.concatenate(instance_chunks)


def duplicate_diagnostics(
    coordinates: np.ndarray,
    instance_index: np.ndarray,
    instance_count: int,
) -> dict[str, int]:
    """Count exact duplicate rows without changing the evaluation arrays."""

    points = _as_xyz(coordinates, "Prediction")
    owners = _as_integral_vector(instance_index, "Prediction instance index")
    if len(points) != len(owners):
        raise ValueError("Prediction coordinates and instance indices are not aligned")

    within_groups = 0
    within_rows = 0
    owner_order = np.argsort(owners, kind="stable")
    owner_counts = np.bincount(owners, minlength=instance_count)
    owner_stops = np.cumsum(owner_counts)
    owner_starts = np.concatenate((np.array([0]), owner_stops[:-1]))
    for start, stop in zip(owner_starts, owner_stops):
        instance_points = points[owner_order[start:stop]]
        if not len(instance_points):
            continue
        _, counts = np.unique(instance_points, axis=0, return_counts=True)
        within_groups += int(np.count_nonzero(counts > 1))
        within_rows += int(np.sum(np.maximum(counts - 1, 0)))

    shared_groups = 0
    shared_rows = 0
    if len(points):
        _, inverse = np.unique(points, axis=0, return_inverse=True)
        coordinate_group_count = int(inverse.max()) + 1
        coordinate_owner_pairs = np.unique(
            np.column_stack([inverse, owners]), axis=0
        )
        owner_counts_per_group = np.bincount(
            coordinate_owner_pairs[:, 0], minlength=coordinate_group_count
        )
        row_counts_per_group = np.bincount(
            inverse, minlength=coordinate_group_count
        )
        shared_mask = owner_counts_per_group > 1
        shared_groups = int(np.count_nonzero(shared_mask))
        shared_rows = int(np.sum(row_counts_per_group[shared_mask]))

    return {
        "duplicate_coordinate_group_count_within_instances": within_groups,
        "duplicate_coordinate_row_count_within_instances": within_rows,
        "shared_coordinate_group_count_across_instances": shared_groups,
        "prediction_row_count_in_shared_coordinate_groups": shared_rows,
    }


def coordinate_multiset_diagnostics(coordinates: np.ndarray) -> dict[str, int]:
    """Summarise coordinate multiplicity without deduplication."""

    points = _as_xyz(coordinates, "Source")
    if not len(points):
        return {
            "coordinate_row_count": 0,
            "unique_coordinate_count": 0,
            "duplicate_coordinate_group_count": 0,
            "duplicate_coordinate_row_count": 0,
        }
    _, counts = np.unique(points, axis=0, return_counts=True)
    return {
        "coordinate_row_count": len(points),
        "unique_coordinate_count": len(counts),
        "duplicate_coordinate_group_count": int(np.count_nonzero(counts > 1)),
        "duplicate_coordinate_row_count": int(np.sum(np.maximum(counts - 1, 0))),
    }


def align_prediction_coordinates(
    prediction_coordinates: np.ndarray,
    source_coordinates: np.ndarray,
    tolerance_m: float,
    source_effective_reference_ids: np.ndarray,
    source_ignored_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign scored prediction rows to distinct source rows within tolerance.

    A prediction whose equally-nearest source candidates are all ignored is
    represented by assignment ``-2`` and does not participate in the bipartite
    graph.  A tie spanning ignored and scored rows remains invalid because its
    evaluation identity cannot be established safely.
    """

    predicted = _as_xyz(prediction_coordinates, "Prediction")
    source = _as_xyz(source_coordinates, "Source")
    effective_ids = _as_integral_vector(
        source_effective_reference_ids, "Effective source reference IDs"
    )
    if len(source) != len(effective_ids):
        raise ValueError("Source coordinates and effective reference IDs are not aligned")
    if source_ignored_mask is None:
        ignored_sources = np.zeros(len(source), dtype=bool)
    else:
        ignored_sources = np.asarray(source_ignored_mask, dtype=bool)
        if ignored_sources.ndim != 1 or len(ignored_sources) != len(source):
            raise ValueError("Source ignored mask is not aligned to source coordinates")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Coordinate tolerance must be finite and positive")

    prediction_count = len(predicted)
    source_count = len(source)
    assignments = np.full(prediction_count, -1, dtype=np.int64)
    if prediction_count == 0 or source_count == 0:
        unmatched = prediction_count
        diagnostics = {
            "prediction_coordinate_count": prediction_count,
            "source_coordinate_count": source_count,
            "within_tolerance_candidate_edge_count": 0,
            "candidate_edge_count": 0,
            "candidate_policy": "nearest_distance_ties_within_tolerance",
            "prediction_coordinate_count_with_multiple_candidates": 0,
            "ambiguous_source_identity_prediction_count": 0,
            "mixed_ignored_scored_tie_count": 0,
            "contended_source_coordinate_count": 0,
            "matched_prediction_coordinate_count": 0,
            "ignored_prediction_coordinate_count": 0,
            "unmatched_prediction_coordinate_count": unmatched,
            "unmatched_prediction_coordinate_count_no_candidate": unmatched,
            "conflicting_many_to_one_prediction_count": 0,
            "mean_coordinate_distance_m": None,
            "max_coordinate_distance_m": None,
            "complete_unique_source_assignment": unmatched == 0,
        }
        return assignments, diagnostics

    tree = cKDTree(source)
    radius_candidates = tree.query_ball_point(predicted, r=tolerance_m)
    candidates: list[np.ndarray] = []
    nearest_candidate_counts = np.zeros(prediction_count, dtype=np.int64)
    ignored_prediction_mask = np.zeros(prediction_count, dtype=bool)
    radius_candidate_edge_count = 0
    mixed_ignored_scored_ties = 0
    ambiguous = 0
    distance_tie_atol = max(1e-12, tolerance_m * 1e-9)
    for row, values in enumerate(radius_candidates):
        candidate_indices = np.asarray(values, dtype=np.int64)
        radius_candidate_edge_count += len(candidate_indices)
        if not len(candidate_indices):
            candidates.append(candidate_indices)
            continue
        distances = np.linalg.norm(source[candidate_indices] - predicted[row], axis=1)
        minimum_distance = float(np.min(distances))
        nearest_mask = np.isclose(
            distances,
            minimum_distance,
            rtol=0.0,
            atol=distance_tie_atol,
        )
        nearest = candidate_indices[nearest_mask]
        nearest_candidate_counts[row] = len(nearest)
        nearest_ignored = ignored_sources[nearest]
        if np.all(nearest_ignored):
            ignored_prediction_mask[row] = True
            candidates.append(np.empty(0, dtype=np.int64))
        elif np.any(nearest_ignored):
            mixed_ignored_scored_ties += 1
            ambiguous += 1
            candidates.append(np.empty(0, dtype=np.int64))
        else:
            candidates.append(nearest)
    del radius_candidates
    candidate_counts = np.fromiter(
        (len(values) for values in candidates),
        dtype=np.int64,
        count=prediction_count,
    )
    active_rows = np.flatnonzero(candidate_counts > 0)
    indptr = np.empty(len(active_rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(candidate_counts[active_rows], out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    for graph_row, row in enumerate(active_rows):
        values = candidates[row]
        start, stop = int(indptr[graph_row]), int(indptr[graph_row + 1])
        candidate_indices = np.asarray(values, dtype=np.int64)
        distances = np.linalg.norm(source[candidate_indices] - predicted[row], axis=1)
        order = np.lexsort((candidate_indices, distances))
        candidate_indices = candidate_indices[order]
        indices[start:stop] = candidate_indices
        if len(np.unique(effective_ids[candidate_indices])) > 1:
            ambiguous += 1

    if len(indices):
        graph = csr_matrix(
            (
                np.ones(len(indices), dtype=np.uint8),
                indices,
                indptr,
            ),
            shape=(len(active_rows), source_count),
        )
        active_assignments = maximum_bipartite_matching(
            graph, perm_type="column"
        ).astype(
            np.int64,
            copy=False,
        )
        assignments[active_rows] = active_assignments
        source_candidate_counts = np.bincount(indices, minlength=source_count)
        contended_source_count = int(
            np.count_nonzero(source_candidate_counts > 1)
        )
    else:
        contended_source_count = 0

    assignments[ignored_prediction_mask] = -2
    matched_mask = assignments >= 0
    candidate_mask = candidate_counts > 0
    unmatched_mask = assignments == -1
    no_candidate_count = int(np.count_nonzero(nearest_candidate_counts == 0))
    conflict_count = int(np.count_nonzero(candidate_mask & unmatched_mask))
    distances = (
        np.linalg.norm(predicted[matched_mask] - source[assignments[matched_mask]], axis=1)
        if np.any(matched_mask)
        else np.empty(0, dtype=np.float64)
    )
    complete = bool(
        not np.any(unmatched_mask)
        and conflict_count == 0
        and ambiguous == 0
        and len(np.unique(assignments[matched_mask])) == np.count_nonzero(matched_mask)
    )
    diagnostics = {
        "prediction_coordinate_count": prediction_count,
        "source_coordinate_count": source_count,
        "within_tolerance_candidate_edge_count": radius_candidate_edge_count,
        "candidate_edge_count": int(len(indices)),
        "candidate_policy": "nearest_distance_ties_within_tolerance",
        "prediction_coordinate_count_with_multiple_candidates": int(
            np.count_nonzero(nearest_candidate_counts > 1)
        ),
        "ambiguous_source_identity_prediction_count": ambiguous,
        "mixed_ignored_scored_tie_count": mixed_ignored_scored_ties,
        "contended_source_coordinate_count": contended_source_count,
        "matched_prediction_coordinate_count": int(np.count_nonzero(matched_mask)),
        "ignored_prediction_coordinate_count": int(
            np.count_nonzero(ignored_prediction_mask)
        ),
        "unmatched_prediction_coordinate_count": int(
            np.count_nonzero(unmatched_mask)
        ),
        "unmatched_prediction_coordinate_count_no_candidate": no_candidate_count,
        "conflicting_many_to_one_prediction_count": conflict_count,
        "mean_coordinate_distance_m": (
            float(np.mean(distances)) if len(distances) else None
        ),
        "max_coordinate_distance_m": (
            float(np.max(distances)) if len(distances) else None
        ),
        "complete_unique_source_assignment": complete,
    }
    return assignments, diagnostics


def _reference_target_mask(
    reference_tree_ids: np.ndarray,
    classification: np.ndarray,
    included_classes: Iterable[int],
) -> np.ndarray:
    return (reference_tree_ids > 0) & np.isin(
        classification, np.asarray(list(included_classes), dtype=np.int64)
    )


def _apply_ignored_semantic_mask(
    prediction_names: list[str],
    prediction_instance_index: np.ndarray,
    assignments: np.ndarray,
    classification: np.ndarray,
    ignored_classes: Iterable[int],
    preignored_prediction_mask: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    """Remove predictions on unlabelled rows and compact surviving IDs.

    FOR-instance class 3 contains unannotated vegetation outside the plot
    boundary.  Those rows are required during inference but must not contribute
    to prediction support, IoU unions, or false-positive instance counts.
    """

    owners = _as_integral_vector(
        prediction_instance_index, "Prediction instance index"
    )
    source_rows = _as_integral_vector(assignments, "Prediction source rows")
    classes = _as_integral_vector(classification, "Reference classification")
    if len(owners) != len(source_rows):
        raise ValueError("Prediction instance indices and source rows are not aligned")
    if preignored_prediction_mask is None:
        preignored = np.zeros(len(source_rows), dtype=bool)
    else:
        preignored = np.asarray(preignored_prediction_mask, dtype=bool)
        if preignored.ndim != 1 or len(preignored) != len(source_rows):
            raise ValueError("Preignored prediction mask is not aligned")
    scored_rows = source_rows[~preignored]
    if len(scored_rows) and (
        scored_rows.min() < 0 or scored_rows.max() >= len(classes)
    ):
        raise ValueError("Prediction source rows are outside reference classification")

    ignored_mask = preignored.copy()
    ignored_mask[~preignored] |= np.isin(
        classes[scored_rows], np.asarray(list(ignored_classes), dtype=np.int64)
    )
    retained_owners = owners[~ignored_mask]
    retained_rows = source_rows[~ignored_mask]
    retained_instance_ids = np.unique(retained_owners)
    remap = np.full(len(prediction_names), -1, dtype=np.int64)
    remap[retained_instance_ids] = np.arange(len(retained_instance_ids))
    compact_owners = remap[retained_owners]
    compact_names = [prediction_names[index] for index in retained_instance_ids]

    diagnostics = {
        "raw_prediction_instance_count": len(prediction_names),
        "raw_predicted_positive_row_count": len(owners),
        "ignored_predicted_point_count": int(np.count_nonzero(ignored_mask)),
        "ignored_prediction_instance_count": (
            len(prediction_names) - len(retained_instance_ids)
        ),
        "evaluated_prediction_instance_count": len(compact_names),
        "evaluated_predicted_positive_row_count": len(compact_owners),
    }
    return compact_names, compact_owners, retained_rows, diagnostics


def _reference_contract_audit(
    tree_ids: np.ndarray,
    classes: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Report excluded background separately from evaluation-ignored rows."""

    positive = tree_ids > 0
    excluded = np.isin(
        classes,
        np.asarray(contract["excluded_semantic_classes"], dtype=np.int64),
    )
    ignored = np.isin(
        classes,
        np.asarray(contract["ignored_semantic_classes"], dtype=np.int64),
    )
    return {
        "source_row_count": len(tree_ids),
        "nonpositive_tree_id_row_count": int(np.count_nonzero(tree_ids <= 0)),
        "positive_tree_id_row_count_excluded_by_semantic_class": int(
            np.count_nonzero(positive & excluded)
        ),
        "positive_tree_id_row_count_ignored_by_semantic_class": int(
            np.count_nonzero(positive & ignored)
        ),
        "ignored_source_row_count": int(np.count_nonzero(ignored)),
        "unexpected_semantic_classes": [],
    }


def _build_iou_evidence(
    prediction_names: list[str],
    prediction_instance_index: np.ndarray,
    assignments: np.ndarray,
    reference_tree_ids: np.ndarray,
    reference_target_mask: np.ndarray,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[int]]:
    """Build pointwise IoU evidence after a complete unique row assignment."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    prediction_count = len(prediction_names)
    reference_ids, reference_point_counts = np.unique(
        reference_tree_ids[reference_target_mask], return_counts=True
    )
    reference_ids = reference_ids.astype(np.int64, copy=False)
    reference_point_counts = reference_point_counts.astype(np.int64, copy=False)
    prediction_point_counts = np.bincount(
        prediction_instance_index, minlength=prediction_count
    ).astype(np.int64)
    intersections = np.zeros(
        (prediction_count, len(reference_ids)), dtype=np.int64
    )
    if len(assignments) and len(reference_ids):
        assigned_target_mask = reference_target_mask[assignments]
        prediction_bins = prediction_instance_index[assigned_target_mask]
        assigned_reference_ids = reference_tree_ids[
            assignments[assigned_target_mask]
        ]
        reference_bins = np.searchsorted(reference_ids, assigned_reference_ids)
        flat_bins = prediction_bins * len(reference_ids) + reference_bins
        occupied_bins, intersection_counts = np.unique(
            flat_bins, return_counts=True
        )
        intersections[
            occupied_bins // len(reference_ids),
            occupied_bins % len(reference_ids),
        ] = intersection_counts

    unions = (
        prediction_point_counts[:, None]
        + reference_point_counts[None, :]
        - intersections
    ).astype(np.float64)
    iou = np.zeros(intersections.shape, dtype=np.float64)
    np.divide(intersections, unions, out=iou, where=unions > 0)
    matched_indices = maximum_cardinality_threshold_matching(iou, iou_threshold)
    matches: list[dict[str, Any]] = []
    for prediction_bin, reference_bin in matched_indices:
        intersection = int(intersections[prediction_bin, reference_bin])
        predicted_points = int(prediction_point_counts[prediction_bin])
        reference_points = int(reference_point_counts[reference_bin])
        matches.append(
            {
                "prediction": prediction_names[prediction_bin],
                "reference_tree_id": int(reference_ids[reference_bin]),
                "intersection": intersection,
                "predicted_points": predicted_points,
                "reference_points": reference_points,
                "union": predicted_points + reference_points - intersection,
                "iou": float(iou[prediction_bin, reference_bin]),
            }
        )

    true_positives = len(matches)
    false_positives = prediction_count - true_positives
    false_negatives = len(reference_ids) - true_positives
    precision, recall, f1 = precision_recall_f1(
        true_positives, false_positives, false_negatives
    )
    matched_ious = [float(row["iou"]) for row in matches]
    reference_best_iou = (
        np.max(iou, axis=0)
        if iou.shape[0]
        else np.zeros(iou.shape[1], dtype=np.float64)
    )
    mean_unweighted_coverage = (
        float(np.mean(reference_best_iou)) if len(reference_best_iou) else 0.0
    )
    mean_weighted_coverage = (
        float(np.average(reference_best_iou, weights=reference_point_counts))
        if len(reference_best_iou) and int(np.sum(reference_point_counts)) > 0
        else 0.0
    )
    predicted_source_rows = np.unique(assignments[assignments >= 0])
    reference_source_rows = np.flatnonzero(reference_target_mask)
    evaluated_point_count = len(
        np.union1d(predicted_source_rows, reference_source_rows)
    )
    overlap = intersections > 0
    fragments_per_reference = np.sum(overlap, axis=0)
    references_per_prediction = np.sum(overlap, axis=1)
    matched_predictions = {row["prediction"] for row in matches}
    matched_references = {int(row["reference_tree_id"]) for row in matches}
    metrics: dict[str, Any] = {
        "prediction_instance_count": prediction_count,
        "reference_instance_count": len(reference_ids),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else 0.0
        ),
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_ious else 0.0
        ),
        "mean_unweighted_coverage": mean_unweighted_coverage,
        "mean_weighted_coverage": mean_weighted_coverage,
        "evaluated_point_count": evaluated_point_count,
        "oversegmented_reference_count": int(
            np.count_nonzero(fragments_per_reference > 1)
        ),
        "oversegmentation_extra_fragment_count": int(
            np.sum(np.maximum(fragments_per_reference - 1, 0))
        ),
        "undersegmented_prediction_count": int(
            np.count_nonzero(references_per_prediction > 1)
        ),
        "undersegmentation_extra_reference_count": int(
            np.sum(np.maximum(references_per_prediction - 1, 0))
        ),
    }
    unmatched_predictions = [
        name for name in prediction_names if name not in matched_predictions
    ]
    unmatched_references = [
        int(tree_id)
        for tree_id in reference_ids
        if int(tree_id) not in matched_references
    ]
    return metrics, matches, unmatched_predictions, unmatched_references


def evaluate_coordinate_arrays(
    *,
    target: str,
    prediction_names: list[str],
    prediction_coordinates: np.ndarray,
    prediction_instance_index: np.ndarray,
    source_coordinates: np.ndarray,
    reference_tree_ids: np.ndarray,
    classification: np.ndarray,
    coordinate_tolerance_m: float,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate synthetic or loaded arrays under the target and alignment gates."""

    contract = target_contract(target)
    predicted = _as_xyz(prediction_coordinates, "Prediction")
    owners = _as_integral_vector(
        prediction_instance_index, "Prediction instance index"
    )
    source = _as_xyz(source_coordinates, "Source")
    tree_ids = _as_integral_vector(reference_tree_ids, "Reference tree IDs")
    classes = _as_integral_vector(classification, "Reference classification")
    if len(predicted) != len(owners):
        raise ValueError("Prediction coordinates and instance indices are not aligned")
    if len(source) != len(tree_ids) or len(source) != len(classes):
        raise ValueError("Source coordinates, tree IDs and classification are not aligned")
    known_semantic_classes = (
        set(contract["included_semantic_classes"])
        | set(contract["excluded_semantic_classes"])
        | set(contract["ignored_semantic_classes"])
    )
    unexpected_semantic_classes = sorted(
        int(value) for value in set(classes.tolist()) - known_semantic_classes
    )
    if unexpected_semantic_classes:
        raise ValueError(
            "Reference classification contains values outside the target contract: "
            f"{unexpected_semantic_classes}"
        )
    if len(owners):
        if owners.min() < 0 or owners.max() >= len(prediction_names):
            raise ValueError("Prediction instance indices are outside prediction_names")
        used = np.unique(owners)
        if not np.array_equal(used, np.arange(len(prediction_names))):
            raise ValueError("Every named prediction instance must contain at least one point")
    elif prediction_names:
        raise ValueError("Named prediction instances cannot be empty")

    target_mask = _reference_target_mask(
        tree_ids, classes, contract["included_semantic_classes"]
    )
    ignored_source_mask = np.isin(
        classes,
        np.asarray(contract["ignored_semantic_classes"], dtype=np.int64),
    )
    effective_reference_ids = np.where(target_mask, tree_ids, 0)
    assignments, alignment = align_prediction_coordinates(
        predicted,
        source,
        coordinate_tolerance_m,
        effective_reference_ids,
        source_ignored_mask=ignored_source_mask,
    )
    duplicates = duplicate_diagnostics(
        predicted, owners, len(prediction_names)
    )
    source_multiplicity = coordinate_multiset_diagnostics(source)
    assigned_source_rows = assignments[assignments >= 0]
    assigned_target_rows = (
        assigned_source_rows[target_mask[assigned_source_rows]]
        if len(assigned_source_rows)
        else np.empty(0, dtype=np.int64)
    )
    target_reference_coordinate_count = int(np.count_nonzero(target_mask))
    matched_reference_coordinate_count = len(np.unique(assigned_target_rows))
    unmatched_reference_coordinate_count = (
        target_reference_coordinate_count - matched_reference_coordinate_count
    )
    retention = (
        matched_reference_coordinate_count / target_reference_coordinate_count
        if target_reference_coordinate_count
        else 1.0
    )
    alignment.update(
        {
            **duplicates,
            "source_coordinate_multiplicity": source_multiplicity,
            "target_reference_coordinate_count": target_reference_coordinate_count,
            "matched_reference_coordinate_count": matched_reference_coordinate_count,
            "unmatched_reference_coordinate_count": unmatched_reference_coordinate_count,
            "reference_coordinate_retention": float(retention),
        }
    )
    safe_for_scoring = bool(alignment["complete_unique_source_assignment"])
    alignment["status"] = "passed" if safe_for_scoring else "failed"

    semantic_ignore: dict[str, Any] = {
        "ignored_semantic_classes": contract["ignored_semantic_classes"],
        "ignored_source_row_count": int(np.count_nonzero(ignored_source_mask)),
    }
    scoring_names = prediction_names
    scoring_owners = owners
    scoring_assignments = assignments
    if safe_for_scoring:
        (
            scoring_names,
            scoring_owners,
            scoring_assignments,
            ignore_diagnostics,
        ) = _apply_ignored_semantic_mask(
            prediction_names,
            owners,
            assignments,
            classes,
            contract["ignored_semantic_classes"],
            preignored_prediction_mask=assignments == -2,
        )
        semantic_ignore.update(ignore_diagnostics)
    else:
        semantic_ignore.update(
            {
                "raw_prediction_instance_count": len(prediction_names),
                "raw_predicted_positive_row_count": len(owners),
                "ignored_predicted_point_count": None,
                "ignored_prediction_instance_count": None,
                "evaluated_prediction_instance_count": None,
                "evaluated_predicted_positive_row_count": None,
            }
        )

    metric_names = (
        "prediction_instance_count",
        "reference_instance_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "median_matched_iou",
        "mean_unweighted_coverage",
        "mean_weighted_coverage",
        "evaluated_point_count",
        "oversegmented_reference_count",
        "oversegmentation_extra_fragment_count",
        "undersegmented_prediction_count",
        "undersegmentation_extra_reference_count",
    )
    if safe_for_scoring:
        metrics, matches, unmatched_predictions, unmatched_references = (
            _build_iou_evidence(
                scoring_names,
                scoring_owners,
                scoring_assignments,
                tree_ids,
                target_mask,
                iou_threshold,
            )
        )
    else:
        metrics = {name: None for name in metric_names}
        matches = []
        unmatched_predictions = []
        unmatched_references = []

    return {
        "evaluator": EVALUATION_PROTOCOL,
        "status": "evaluated" if safe_for_scoring else "invalid_coordinate_alignment",
        "safe_for_scoring": safe_for_scoring,
        "target": target,
        "prediction_pattern": contract["prediction_pattern"],
        "target_contract": contract,
        "reference_contract_audit": _reference_contract_audit(
            tree_ids, classes, contract
        ),
        "semantic_ignore": semantic_ignore,
        "evaluation_mask": EVALUATION_MASK,
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": float(iou_threshold),
        "iou_threshold_operator": ">=",
        "coordinate_tolerance_m": float(coordinate_tolerance_m),
        "coordinate_alignment": alignment,
        **metrics,
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_references": unmatched_references,
    }


def evaluate_source_row_arrays(
    *,
    target: str,
    prediction_names: list[str],
    predicted_instance_ids: np.ndarray,
    source_row_index: np.ndarray,
    reference_tree_ids: np.ndarray,
    classification: np.ndarray,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one prediction label per source row without coordinate recovery."""

    contract = target_contract(target)
    predicted = _as_integral_vector(predicted_instance_ids, "Predicted instance IDs")
    rows = _as_integral_vector(source_row_index, "Source row index")
    tree_ids = _as_integral_vector(reference_tree_ids, "Reference tree IDs")
    classes = _as_integral_vector(classification, "Reference classification")
    point_count = len(tree_ids)
    if len(predicted) != point_count or len(rows) != point_count or len(classes) != point_count:
        raise ValueError("Aligned prediction, source rows and reference arrays differ in length")
    if not np.array_equal(rows, np.arange(point_count)):
        raise ValueError("source_row_index must equal np.arange(source_point_count)")
    if np.any(predicted < 0):
        raise ValueError("Predicted instance IDs must use zero for background and positive IDs")
    known_semantic_classes = (
        set(contract["included_semantic_classes"])
        | set(contract["excluded_semantic_classes"])
        | set(contract["ignored_semantic_classes"])
    )
    unexpected_semantic_classes = sorted(
        int(value) for value in set(classes.tolist()) - known_semantic_classes
    )
    if unexpected_semantic_classes:
        raise ValueError(
            "Reference classification contains values outside the target contract: "
            f"{unexpected_semantic_classes}"
        )
    used = np.unique(predicted[predicted > 0])
    expected = np.arange(1, len(prediction_names) + 1)
    expected_suffix = contract["prediction_pattern"].removeprefix("*")
    if any(not name.endswith(expected_suffix) for name in prediction_names):
        raise ValueError(
            f"Aligned prediction names do not all match target suffix {expected_suffix!r}"
        )
    if not np.array_equal(used, expected):
        raise ValueError(
            "Positive aligned prediction IDs must be contiguous and match prediction_names"
        )

    target_mask = _reference_target_mask(
        tree_ids, classes, contract["included_semantic_classes"]
    )
    positive_mask = predicted > 0
    assignments = rows[positive_mask]
    owners = predicted[positive_mask] - 1
    (
        scoring_names,
        scoring_owners,
        scoring_assignments,
        ignore_diagnostics,
    ) = _apply_ignored_semantic_mask(
        prediction_names,
        owners,
        assignments,
        classes,
        contract["ignored_semantic_classes"],
    )
    metrics, matches, unmatched_predictions, unmatched_references = _build_iou_evidence(
        scoring_names,
        scoring_owners,
        scoring_assignments,
        tree_ids,
        target_mask,
        iou_threshold,
    )
    return {
        "evaluator": SOURCE_ROW_EVALUATION_PROTOCOL,
        "status": "evaluated",
        "safe_for_scoring": True,
        "target": target,
        "prediction_pattern": contract["prediction_pattern"],
        "target_contract": contract,
        "reference_contract_audit": _reference_contract_audit(
            tree_ids, classes, contract
        ),
        "semantic_ignore": {
            "ignored_semantic_classes": contract["ignored_semantic_classes"],
            "ignored_source_row_count": int(
                np.count_nonzero(
                    np.isin(
                        classes,
                        np.asarray(
                            contract["ignored_semantic_classes"], dtype=np.int64
                        ),
                    )
                )
            ),
            **ignore_diagnostics,
        },
        "evaluation_mask": EVALUATION_MASK,
        "matching_policy": "maximum_cardinality_one_to_one",
        "iou_threshold": float(iou_threshold),
        "iou_threshold_operator": ">=",
        "coordinate_tolerance_m": None,
        "point_correspondence": {
            "mode": "source_row_index",
            "source_row_count": point_count,
            "aligned_prediction_row_count": len(predicted),
            "predicted_positive_row_count": ignore_diagnostics[
                "evaluated_predicted_positive_row_count"
            ],
            "raw_predicted_positive_row_count": ignore_diagnostics[
                "raw_predicted_positive_row_count"
            ],
            "ignored_predicted_positive_row_count": ignore_diagnostics[
                "ignored_predicted_point_count"
            ],
            "predicted_background_row_count": int(np.count_nonzero(~positive_mask)),
            "source_row_order_complete": True,
            "status": "passed",
        },
        **metrics,
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_references": unmatched_references,
    }


def load_reference_las(
    path: Path,
    reference_instance_field: str = "treeID",
    reference_semantic_field: str = "classification",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """Load source coordinates and reference fields from a LAS/LAZ file."""

    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("FOR-instance reference source must be LAS or LAZ")
    import laspy

    cloud = laspy.read(path)
    dimensions = set(cloud.point_format.dimension_names)
    missing = {reference_instance_field, reference_semantic_field} - dimensions
    if missing:
        raise ValueError(f"Reference source is missing fields {sorted(missing)}")
    coordinates = _as_xyz(
        np.column_stack([cloud.x, cloud.y, cloud.z]), "Reference source"
    )
    return (
        coordinates,
        _as_integral_vector(cloud[reference_instance_field], reference_instance_field),
        _as_integral_vector(cloud[reference_semantic_field], reference_semantic_field),
        [float(value) for value in cloud.header.scales],
        [float(value) for value in cloud.header.offsets],
    )


def evaluate_plot(
    *,
    target: str,
    prediction_directory: Path,
    reference_source: Path,
    alignment_metadata: Mapping[str, Any],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Load and evaluate one plot, including frame/header verification."""

    if (
        alignment_metadata.get("point_correspondence")
        == "source_row_via_voxel_representative"
        or alignment_metadata.get("raw_coordinate_evaluation_permitted") is False
    ):
        raise ValueError(
            "Raw coordinate evaluation is disabled for source-row alignment metadata"
        )

    source, tree_ids, classes, source_scale, source_offset = load_reference_las(
        reference_source
    )
    canonical_metadata = validate_alignment_metadata(
        alignment_metadata, source_scale
    )
    if not np.allclose(
        canonical_metadata["source_las"]["offset_m"],
        source_offset,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Alignment metadata source LAS offset does not match the reference header"
        )
    names, predicted, owners = load_predictions(prediction_directory, target)
    result = evaluate_coordinate_arrays(
        target=target,
        prediction_names=names,
        prediction_coordinates=predicted,
        prediction_instance_index=owners,
        source_coordinates=source,
        reference_tree_ids=tree_ids,
        classification=classes,
        coordinate_tolerance_m=canonical_metadata["matching"][
            "coordinate_tolerance_m"
        ],
        iou_threshold=iou_threshold,
    )
    result["coordinate_frame_metadata"] = canonical_metadata
    return result


def evaluate_aligned_plot(
    *,
    target: str,
    aligned_predictions: Path,
    reference_source: Path,
    alignment_metadata: Mapping[str, Any],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate the preferred source-row-aligned TLS2trees adapter artefact."""

    _, tree_ids, classes, source_scale, source_offset = load_reference_las(
        reference_source
    )
    canonical_metadata = validate_source_row_alignment_metadata(
        alignment_metadata, source_scale
    )
    if not np.allclose(
        canonical_metadata["source_las"]["offset_m"],
        source_offset,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Alignment metadata source LAS offset does not match the reference header"
        )
    with np.load(aligned_predictions, allow_pickle=False) as arrays:
        required = {"source_row_index", "predicted_instance_id", "prediction_names"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(
                f"Aligned prediction NPZ is missing arrays: {', '.join(missing)}"
            )
        source_rows = np.asarray(arrays["source_row_index"])
        predicted_ids = np.asarray(arrays["predicted_instance_id"])
        prediction_names = [str(value) for value in arrays["prediction_names"].tolist()]
        if "source_las_sha256" in arrays.files:
            recorded_source_sha = str(np.asarray(arrays["source_las_sha256"]).item())
            actual_source_sha = sha256_file(reference_source)
            if recorded_source_sha != actual_source_sha:
                raise ValueError("Aligned prediction source LAS checksum does not match")
    result = evaluate_source_row_arrays(
        target=target,
        prediction_names=prediction_names,
        predicted_instance_ids=predicted_ids,
        source_row_index=source_rows,
        reference_tree_ids=tree_ids,
        classification=classes,
        iou_threshold=iou_threshold,
    )
    result["coordinate_frame_metadata"] = canonical_metadata
    result["raw_alignment_coordinate_tolerance_m"] = canonical_metadata["matching"][
        "coordinate_tolerance_m"
    ]
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a stable CSV schema, including for empty tables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one TLS2trees FOR-instance plot with explicit leaf target "
            "and coordinate-integrity gates."
        )
    )
    parser.add_argument("--target", choices=sorted(TARGET_CONTRACTS), required=True)
    predictions = parser.add_mutually_exclusive_group(required=True)
    predictions.add_argument("--predicted-instance-dir")
    predictions.add_argument("--aligned-predictions-npz")
    parser.add_argument("--reference-labelled-point-cloud", required=True)
    parser.add_argument("--alignment-metadata-json", required=True)
    parser.add_argument("--plot-id", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-matches-csv")
    parser.add_argument("--output-unmatched-predictions-csv")
    parser.add_argument("--output-unmatched-references-csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_source = resolve_path(args.reference_labelled_point_cloud)
    alignment_metadata_path = resolve_path(args.alignment_metadata_json)
    metadata = json.loads(alignment_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("Alignment metadata JSON must contain an object")
    prediction_directory: Path | None = None
    aligned_predictions: Path | None = None
    if args.aligned_predictions_npz:
        aligned_predictions = resolve_path(args.aligned_predictions_npz)
        result = evaluate_aligned_plot(
            target=args.target,
            aligned_predictions=aligned_predictions,
            reference_source=reference_source,
            alignment_metadata=metadata,
            iou_threshold=args.iou_threshold,
        )
    else:
        prediction_directory = resolve_path(args.predicted_instance_dir)
        result = evaluate_plot(
            target=args.target,
            prediction_directory=prediction_directory,
            reference_source=reference_source,
            alignment_metadata=metadata,
            iou_threshold=args.iou_threshold,
        )
    result.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "plot_id": args.plot_id,
            "relative_path": args.relative_path,
            "split": args.split,
            "prediction_directory": (
                str(prediction_directory) if prediction_directory else None
            ),
            "aligned_predictions_npz": (
                str(aligned_predictions) if aligned_predictions else None
            ),
            "reference_source": str(reference_source),
            "alignment_metadata_json": str(alignment_metadata_path),
            "alignment_metadata_sha256": sha256_file(alignment_metadata_path),
        }
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matches_path = (
        resolve_path(args.output_matches_csv)
        if args.output_matches_csv
        else output_json.with_name(f"{output_json.stem}_matches.csv")
    )
    unmatched_predictions_path = (
        resolve_path(args.output_unmatched_predictions_csv)
        if args.output_unmatched_predictions_csv
        else output_json.with_name(f"{output_json.stem}_unmatched_predictions.csv")
    )
    unmatched_references_path = (
        resolve_path(args.output_unmatched_references_csv)
        if args.output_unmatched_references_csv
        else output_json.with_name(f"{output_json.stem}_unmatched_references.csv")
    )
    write_csv(
        matches_path,
        [
            "prediction",
            "reference_tree_id",
            "intersection",
            "predicted_points",
            "reference_points",
            "union",
            "iou",
        ],
        result["matches"],
    )
    write_csv(
        unmatched_predictions_path,
        ["prediction"],
        [
            {"prediction": value}
            for value in result["unmatched_predictions"]
        ],
    )
    write_csv(
        unmatched_references_path,
        ["reference_tree_id"],
        [
            {"reference_tree_id": value}
            for value in result["unmatched_references"]
        ],
    )
    print(f"Status: {result['status']}")
    print(f"Target: {args.target}")
    print(f"Output: {output_json}")
    return 0 if result["safe_for_scoring"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
