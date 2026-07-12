"""Create one TreeLearn training LAS with the frozen FOR-instance label mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TREE_CLASSES = (4, 5, 6)
NON_TREE_CLASSES = (1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validated_integer_tree_ids(tree_id: np.ndarray) -> np.ndarray:
    tree_id = np.asarray(tree_id)
    if not np.issubdtype(tree_id.dtype, np.number):
        raise TypeError("treeID must use a numeric type")
    if not np.all(np.isfinite(tree_id)):
        raise ValueError("treeID contains non-finite values")
    rounded = np.rint(tree_id)
    if not np.array_equal(tree_id, rounded):
        raise ValueError("treeID contains non-integral values")
    limits = np.iinfo(np.int64)
    if np.any(rounded < limits.min) or np.any(rounded > limits.max):
        raise OverflowError("treeID exceeds the int64 range")
    return rounded.astype(np.int64)


def encode_tree_ids(tree_id: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    tree_id = np.asarray(tree_id, dtype=np.int64)
    target_dtype = np.dtype(target_dtype)
    if not np.issubdtype(target_dtype, np.number):
        raise TypeError("LAS treeID target must use a numeric type")
    encoded = tree_id.astype(target_dtype)
    if not np.all(np.isfinite(encoded)) or not np.array_equal(
        encoded.astype(np.int64), tree_id
    ):
        raise OverflowError(f"treeID cannot be represented losslessly as {target_dtype}")
    return encoded


def normalise_instance_labels(classification: np.ndarray, tree_id: np.ndarray) -> np.ndarray:
    classification = np.asarray(classification)
    tree_id = validated_integer_tree_ids(tree_id)
    if classification.ndim != 1 or tree_id.ndim != 1 or classification.shape != tree_id.shape:
        raise ValueError("classification and treeID must be equal-length one-dimensional arrays")
    result = np.full(tree_id.shape, -1, dtype=np.int64)
    tree = np.isin(classification, TREE_CLASSES) & (tree_id > 0)
    non_tree = np.isin(classification, NON_TREE_CLASSES) & ~tree
    result[non_tree] = 0
    result[tree] = tree_id[tree].astype(np.int64, copy=False)
    return result


def resolve_dimension(names: list[str], expected: str) -> str:
    matches = [name for name in names if name.casefold() == expected.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {expected} dimension, found {matches}")
    return matches[0]


def normalise_las(
    source: Path,
    output: Path,
    metadata_path: Path,
    *,
    expected_sha256: str,
    expected_point_count: int,
    expected_reference_tree_count: int,
) -> dict:
    import laspy

    if output.exists() or metadata_path.exists():
        raise FileExistsError(output if output.exists() else metadata_path)
    source_sha256 = sha256(source)
    if source_sha256 != expected_sha256:
        raise ValueError(f"Frozen development input SHA-256 changed: {source}")
    las = laspy.read(source)
    names = list(las.point_format.dimension_names)
    classification_name = resolve_dimension(names, "classification")
    tree_name = resolve_dimension(names, "treeID")
    classification = np.asarray(las[classification_name])
    source_tree_id = np.asarray(las[tree_name])
    normalised = normalise_instance_labels(classification, source_tree_id)
    tree_mask = normalised > 0
    reference_tree_count = int(np.unique(normalised[tree_mask]).size)
    if normalised.size != expected_point_count:
        raise ValueError(f"Frozen development point count changed: {source}")
    if reference_tree_count != expected_reference_tree_count:
        raise ValueError(f"Frozen development reference-tree count changed: {source}")
    target_dtype = source_tree_id.dtype
    # TreeLearn's LAS loader derives ignored labels from treeID == 0 combined
    # with a classification outside [1, 2]. Keeping ignored LAS treeID values
    # at zero also supports FOR-instance files whose extra dimension is unsigned.
    output_tree_id = np.where(normalised > 0, normalised, 0)
    las[tree_name] = encode_tree_ids(output_tree_id, target_dtype)
    output.parent.mkdir(parents=True, exist_ok=False)
    las.write(output)
    record = {
        "schema_version": 1,
        "status": "normalised_for_treelearn_training",
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "point_count": int(normalised.size),
        "positive_tree_points": int(np.count_nonzero(tree_mask)),
        "positive_tree_instances": reference_tree_count,
        "source_tree_id_dtype": str(target_dtype),
        "tree_id_integer_values_validated": True,
        "tree_id_lossless_storage_validated": True,
        "non_tree_points": int(np.count_nonzero(normalised == 0)),
        "ignored_points": int(np.count_nonzero(normalised == -1)),
        "held_out_test_accessed": False,
        "mapping": {
            "positive_tree": "classification in [4,5,6] and source treeID > 0",
            "non_tree": "classification in [1,2] maps to treeID 0",
            "ignored": (
                "all other points use treeID 0 and a non-[1,2] classification; "
                "the pinned TreeLearn loader maps them to instance label -1"
            ),
        },
    }
    metadata_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("held_out_test_accessed") is not False:
        raise ValueError("Long-run freeze does not lock held-out test access")
    matches = [row for row in freeze["plots"] if int(row["task_index"]) == args.task_index]
    if len(matches) != 1 or matches[0].get("split") != "dev":
        raise ValueError(f"Expected one development task for index {args.task_index}")
    row = matches[0]
    output = Path(row["normalised_las"])
    record = normalise_las(
        Path(row["input_las"]),
        output,
        Path(row["normalisation_metadata"]),
        expected_sha256=str(row["input_sha256"]),
        expected_point_count=int(row["point_count"]),
        expected_reference_tree_count=int(row["reference_tree_count"]),
    )
    modular = args.treelearn_repo.resolve() / "configs" / "_modular"
    config = {
        "default_args": [str(modular / "sample_generation.yaml")],
        "base_dir": str(output.parent.parent),
        "occupancy_res": 1,
        "n_points_to_calculate_occupancy": 100000,
        "min_percent_occupied_fill": 0.9,
        "how_far_fill": 9,
        "min_percent_occupied_choose": 0.45,
        "n_samples_total": int(row["crops_generate_requested"]),
        "chunk_size": 35,
    }
    config_path = Path(row["crop_config"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f'normalised_las={record["output"]}')
    print(f"crop_config={config_path}")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
