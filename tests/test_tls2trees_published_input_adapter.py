from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType

import laspy
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script() -> ModuleType:
    path = (
        ROOT
        / "methods/tls2trees/scripts/data/prepare_for_instance_tls2trees_input.py"
    )
    spec = importlib.util.spec_from_file_location("tls2trees_published_input", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_source_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    cloud = laspy.LasData(header)
    cloud.x = np.array([100.001, 100.009, 110.001, 110.021])
    cloud.y = np.array([200.001, 200.009, 200.001, 200.021])
    cloud.z = np.array([5.001, 5.009, 5.001, 5.021])
    cloud.classification = np.array([2, 4, 5, 6], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.array([0, 1, 2, 2], dtype=np.int32)
    cloud.write(path)


def write_manifest(path: Path, input_las: Path, *, split: str = "development") -> None:
    payload = {
        "schema_version": 1,
        "dataset_split": split,
        "split_metadata_sha256": "synthetic-metadata-sha",
        "plots": [
            {
                "task_index": 0,
                "split": split,
                "relative_path": "CULS/plot_1_annotated.las",
                "collection": "CULS",
                "safe_plot_id": "CULS__plot_1_annotated",
                "input_las": str(input_las),
                "point_count": 4,
                "reference_tree_count": 2,
                "input_sha256": "synthetic-input-sha",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def stub_verified_development_manifest(
    adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    input_las: Path,
) -> None:
    digest = hashlib.sha256(input_las.read_bytes()).hexdigest()
    payload = {"split_metadata_sha256": "synthetic-metadata-sha"}
    row = {
        "task_index": 0,
        "split": "development",
        "relative_path": "CULS/plot_1_annotated.las",
        "collection": "CULS",
        "safe_plot_id": "CULS_plot_1_annotated",
        "input_las": str(input_las),
        "point_count": 4,
        "reference_tree_count": 2,
        "input_sha256": digest,
        "observed_input_sha256": digest,
    }
    monkeypatch.setattr(
        adapter,
        "load_manifest_plot",
        lambda manifest_path, task_index: (payload, row),
    )


def test_published_input_is_label_stripped_tiled_and_source_aligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = load_script()
    input_las = tmp_path / "plot.las"
    manifest = tmp_path / "manifest.json"
    output_root = tmp_path / "outputs"
    write_source_las(input_las)
    write_manifest(manifest, input_las)
    stub_verified_development_manifest(adapter, monkeypatch, input_las)

    metadata = adapter.prepare_plot(
        manifest_path=manifest,
        task_index=0,
        output_root=output_root,
        run_id="smoke-001",
    )

    plot_root = Path(metadata["plot_root"])
    converted = plot_root / "converted"
    tile_paths = sorted((converted / "tiles").glob("*.ply"))
    assert len(tile_paths) == 2
    assert metadata["tile_size_m"] == 10.0
    assert metadata["downsample_voxel_size_m"] == 0.02
    assert metadata["labels_stripped"] is True
    assert metadata["reference_fields_passed_to_method"] == []
    assert metadata["method_input_dimensions"] == ["x", "y", "z"]
    assert metadata["source_point_count"] == 4
    assert metadata["input_sha256"] == hashlib.sha256(input_las.read_bytes()).hexdigest()
    assert metadata["representative_point_count"] == 3
    assert metadata["coordinate_frame"]["maximum_round_trip_delta_m"] == 0.0

    source_map = np.load(converted / "source_map.npz")
    assert np.array_equal(source_map["source_row_index"], np.arange(4))
    source_to_rep = source_map["source_to_representative_index"]
    assert source_to_rep[0] == source_to_rep[1]
    assert source_map["representative_source_row_index"].tolist() == [1, 2, 3]
    assert source_map["representative_local_xyz"].shape == (3, 3)

    for tile_path in tile_paths:
        raw_header = tile_path.read_bytes().split(b"end_header\n", 1)[0]
        assert b"property float64 x" in raw_header
        assert b"property float64 y" in raw_header
        assert b"property float64 z" in raw_header
        assert b"classification" not in raw_header
        assert b"treeID" not in raw_header
        assert b"label" not in raw_header


def test_voxel_tie_breaks_on_lowest_source_row() -> None:
    adapter = load_script()
    points = np.array([[0.005, 0.01, 0.01], [0.015, 0.01, 0.01]])
    representatives, source_to_rep, keys = adapter.select_voxel_representatives(
        points, 0.02
    )
    assert representatives.tolist() == [0]
    assert source_to_rep.tolist() == [0, 0]
    assert keys.tolist() == [[0, 0, 0]]


def test_published_input_rejects_held_out_test_manifest(tmp_path: Path) -> None:
    adapter = load_script()
    input_las = tmp_path / "plot.las"
    manifest = tmp_path / "manifest.json"
    write_source_las(input_las)
    write_manifest(manifest, input_las, split="test")

    with pytest.raises(ValueError):
        adapter.prepare_plot(
            manifest_path=manifest,
            task_index=0,
            output_root=tmp_path / "outputs",
            run_id="smoke-001",
        )


def test_published_input_never_overwrites_an_existing_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = load_script()
    input_las = tmp_path / "plot.las"
    manifest = tmp_path / "manifest.json"
    output_root = tmp_path / "outputs"
    write_source_las(input_las)
    write_manifest(manifest, input_las)
    stub_verified_development_manifest(adapter, monkeypatch, input_las)
    kwargs = {
        "manifest_path": manifest,
        "task_index": 0,
        "output_root": output_root,
        "run_id": "smoke-001",
    }

    adapter.prepare_plot(**kwargs)
    with pytest.raises(FileExistsError, match="new run_id"):
        adapter.prepare_plot(**kwargs)
