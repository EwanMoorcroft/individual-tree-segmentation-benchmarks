from __future__ import annotations

import csv
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pytest

from methods.forestformer3d.scripts.data import prepare_one_plot_smoke
from methods.forestformer3d.scripts.evaluation import validate_one_plot_smoke
from shared.for_instance_manifest import EXPECTED_PATHS


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forestformer3d"


def _metadata(dataset_root: Path) -> str:
    path = dataset_root / "data_split_metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "folder", "split"))
        for split, paths in EXPECTED_PATHS.items():
            for relative in paths:
                writer.writerow((relative, Path(relative).parts[0], split))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_las(path: Path) -> None:
    path.parent.mkdir(parents=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.array([10.0, 11.0, 12.0, 13.0])
    cloud.y = np.array([20.0, 22.0, 24.0, 26.0])
    cloud.z = np.array([2.0, 3.0, 4.0, 5.0])
    cloud.classification = np.array([2, 4, 5, 6], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.array([0, 7, 7, 9], dtype=np.int32)
    cloud.write(path)


def _write_ply(
    path: Path,
    xyz: np.ndarray,
    semantic_gt: np.ndarray,
    instance_gt: np.ndarray,
    *,
    change_prediction: bool = False,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("semantic_pred", "i4"),
        ("instance_pred", "i4"),
        ("score", "f4"),
        ("semantic_gt", "i4"),
        ("instance_gt", "i4"),
    ]
    vertices = np.empty(len(xyz), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["semantic_pred"] = np.array([0, 1, 1, 2])
    vertices["instance_pred"] = np.array([-1, 0, 0, 1])
    vertices["score"] = np.array([0.0, 0.8, 0.8, 0.9])
    if change_prediction:
        vertices["instance_pred"][1] = 3
    vertices["semantic_gt"] = semantic_gt
    vertices["instance_gt"] = instance_gt
    path.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _prepared(tmp_path: Path) -> Path:
    dataset = tmp_path / "FORinstance_dataset"
    dataset.mkdir()
    metadata_sha = _metadata(dataset)
    _source_las(dataset / prepare_one_plot_smoke.RELATIVE_PATH)
    output = tmp_path / "staged"
    prepare_one_plot_smoke.prepare(
        dataset,
        output,
        expected_point_count=4,
        expected_reference_tree_count=2,
        expected_metadata_sha256=metadata_sha,
    )
    return output


def test_preparation_is_development_only_and_preserves_identity(tmp_path: Path) -> None:
    output = _prepared(tmp_path)
    with np.load(output / "evaluation_sidecar.npz") as sidecar:
        assert sidecar["source_row_index"].tolist() == [0, 1, 2, 3]
        assert sidecar["reference_semantic"].tolist() == [0, 1, 1, 1]
        assert sidecar["reference_instance"].tolist() == [0, 7, 7, 9]
        assert sidecar["target_tree_id"].tolist() == [-1, 7, 7, 9]
    assert (output / "points/forestformer3d_smoke_test.bin").stat().st_size == 48
    assert (output / "input_preparation.complete").is_file()


@pytest.mark.skipif(
    importlib.util.find_spec("plyfile") is None, reason="plyfile is not installed"
)
def test_validation_proves_counterfactual_and_writes_contract(tmp_path: Path) -> None:
    staged = _prepared(tmp_path)
    with np.load(staged / "evaluation_sidecar.npz") as sidecar:
        xyz = sidecar["model_xyz"].copy()
        ref_sem = sidecar["reference_semantic"].copy()
        ref_ins = sidecar["reference_instance"].copy()
    reference = tmp_path / "raw/reference.ply"
    dummy = tmp_path / "raw/dummy.ply"
    _write_ply(reference, xyz, ref_sem, ref_ins)
    _write_ply(dummy, xyz, np.zeros(4), np.zeros(4))

    result = validate_one_plot_smoke.validate(
        reference, dummy, staged / "evaluation_sidecar.npz", tmp_path / "validated"
    )
    assert result["status"] == "passed"
    assert result["exact_row_alignment"] is True
    assert result["label_counterfactual"]["passed"] is True
    with np.load(
        tmp_path / "validated/forestformer3d_smoke_predictions.npz"
    ) as output:
        assert set(output.files) == {
            "pred_tree_id",
            "target_tree_id",
            "classification",
            "pred_classification",
            "source_row_index",
        }
        assert output["pred_tree_id"].tolist() == [-1, 1, 1, 2]


@pytest.mark.skipif(
    importlib.util.find_spec("plyfile") is None, reason="plyfile is not installed"
)
def test_validation_rejects_label_dependent_prediction(tmp_path: Path) -> None:
    sidecar = tmp_path / "sidecar.npz"
    xyz = np.zeros((4, 3), dtype=np.float32)
    np.savez(
        sidecar,
        model_xyz=xyz,
        source_row_index=np.arange(4),
        classification=np.array([2, 4, 5, 6]),
        target_tree_id=np.array([-1, 7, 7, 9]),
        reference_semantic=np.array([0, 1, 1, 1]),
        reference_instance=np.array([0, 7, 7, 9]),
    )
    reference = tmp_path / "reference.ply"
    dummy = tmp_path / "dummy.ply"
    _write_ply(
        reference,
        xyz,
        np.array([0, 1, 1, 1]),
        np.array([0, 7, 7, 9]),
    )
    _write_ply(
        dummy,
        xyz,
        np.zeros(4),
        np.zeros(4),
        change_prediction=True,
    )
    with pytest.raises(ValueError, match="counterfactual changed instance_pred"):
        validate_one_plot_smoke.validate(
            reference, dummy, sidecar, tmp_path / "validated"
        )


def test_smoke_submitter_is_guarded_development_only_and_monitored() -> None:
    submitter = (METHOD / "slurm/submit_one_plot_smoke.sh").read_text(
        encoding="utf-8"
    )
    job = (METHOD / "slurm/run_one_plot_smoke.sbatch").read_text(encoding="utf-8")
    runner = (
        METHOD / "scripts/runtime/run_official_test.py"
    ).read_text(encoding="utf-8")
    assert "FF3D_SMOKE_CONFIRMED" in submitter
    assert "CULS/plot_1_annotated.las" in submitter
    assert "NIBIO/plot_1_annotated.las" not in submitter
    assert "monitor_workflow.sh" in submitter
    assert 'FF3D_MONITOR_SECONDS:-30' in submitter
    assert "#SBATCH --partition=gpu-a100-lowbig" in job
    assert "run_case reference" in job
    assert "run_case dummy" in job
    assert "validate_one_plot_smoke.py" in job
    assert "runpy.run_path" in runner
    assert 'init_globals={"torch": torch}' in runner
    assert "prepare_entrypoint_checkpoint" in runner
    assert "permute(4, 0, 1, 2, 3)" in runner
    assert "converted != 49" in runner


def test_preparation_cli_resolves_shared_package() -> None:
    completed = subprocess.run(
        [
            "python",
            "methods/forestformer3d/scripts/data/prepare_one_plot_smoke.py",
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--dataset-root" in completed.stdout
