from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest
import yaml
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forainet"


def load_script(relative_path: str, name: str):
    path = METHOD / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_script("scripts/runtime/forainet_contract.py", "forainet_contract")
evaluator = load_script(
    "scripts/evaluation/evaluate_for_instance.py", "forainet_evaluator"
)
sidecar = load_script(
    "scripts/data/prepare_alignment_sidecar.py", "forainet_sidecar"
)
input_adapter = load_script(
    "scripts/data/prepare_label_isolated_input.py", "forainet_input_adapter"
)
merge_extractor = load_script(
    "scripts/runtime/extract_official_merge.py", "forainet_merge_extractor"
)
exposure = load_script(
    "scripts/provenance/validate_exposure_audit.py", "forainet_exposure"
)
retention = load_script(
    "scripts/provenance/build_retention_manifest.py", "forainet_retention"
)


def test_scaffold_and_configs_are_method_local() -> None:
    assert {
        "configs",
        "docs",
        "examples",
        "scripts",
        "slurm",
    } <= {path.name for path in METHOD.iterdir() if path.is_dir()}
    assert {
        "data",
        "runtime",
        "evaluation",
        "provenance",
    } <= {path.name for path in (METHOD / "scripts").iterdir() if path.is_dir()}
    for path in sorted((METHOD / "configs").glob("*.yml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["project"]["dataset_slug"] == "for-instance"
        assert payload["project"]["method_slug"] == "forainet"
        assert payload["dataset"]["slug"] == "for-instance"
        assert payload["method"]["slug"] == "forainet"


def test_qualification_identity_is_frozen() -> None:
    config = yaml.safe_load(
        (METHOD / "configs/qualification.yml").read_text(encoding="utf-8")
    )
    assert config["method"]["selected_release"] == "original"
    assert config["method"]["selection_reason"] == (
        "no_complete_official_forainetv2_release"
    )
    assert config["method"]["upstream_commit"] == (
        "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
    )
    assert config["method"]["checkpoint"]["sha256"] == (
        "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
    )
    assert config["method"]["checkpoint"]["provider_checksum"] is None
    container = config["method"]["container"]
    assert container["base_image_digest"] == (
        "sha256:83e4b2841034cdf45ea5b9a5b472eb2c07b1b23d4836d32666a881db29a8dceb"
    )
    assert container["cuda_arch"] == "8.0"
    assert container["qualification_gpu"] == "a100"
    assert container["barkla_build_probe"]["sha256"] == (
        "2a111b22871288abe8eb205fe4a14424290bc4e2376e6c4c170f82260b3052db"
    )
    assert container["build_toolchain"]["installer_sha256"] == (
        "41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1"
    )
    assert container["build_toolchain"]["release_rpm_sha256"] == (
        "1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605"
    )
    assert config["gates"]["barkla_root_mapped_fakeroot_probe_passed"] is True
    assert config["gates"]["barkla_root_mapped_apt_build_blocked"] is True
    assert config["gates"]["barkla_userlocal_fakeroot_toolchain_verified"] is True
    assert config["gates"]["barkla_image_verified"] is True
    assert config["gates"]["checkpoint_full_load_verified"] is True
    assert container["qualified_image"]["sha256"] == (
        "4b8835107800c5a368e4073aade1fee5b94e436693e13ab351fe8c2a250d898e"
    )
    assert container["checkpoint_load_evidence"]["compatible_fraction"] == 1.0
    assert config["gates"]["held_out_authorised"] is False


def test_container_definition_pins_mutable_upstream_inputs() -> None:
    definition = (
        METHOD / "containers/forainet-cuda111-a100.def"
    ).read_text(encoding="utf-8")
    assert "From: nvidia/cuda@sha256:" in definition
    assert "11.1.1-cudnn8-devel-ubuntu20.04@sha256:" not in definition
    for commit in (
        "9f81ae66b33b883cd08ee4f64d08cf633608b118",
        "74099d10a51c71c14318bce63d6421f698b24f24",
        "ec3b205fbd7da9f1e41b9d83cdf3f6236e2ef1c4",
    ):
        assert commit in definition
    assert "TORCH_CUDA_ARCH_LIST=8.0" in definition
    assert "--requirement /opt/forainet/requirements.lock" in definition
    assert "hdbscan/archive/master" not in definition
    assert "git+https://github.com/NVIDIA/MinkowskiEngine.git" not in definition
    probe = (METHOD / "containers/fakeroot-apt-probe.def").read_text(
        encoding="utf-8"
    )
    assert "From: nvidia/cuda@sha256:" in probe
    assert "apt-get install -y --no-install-recommends less" in probe
    lock = (METHOD / "containers/requirements.lock").read_text(encoding="utf-8")
    assert "pylidar" not in lock
    assert "rios" not in lock


def test_image_build_is_cpu_only_and_qualification_targets_a100() -> None:
    build = (METHOD / "slurm/build_forainet_image.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=nodes" in build
    assert "#SBATCH --gres" not in build
    assert '"$apptainer" build --fakeroot' in build
    assert 'mktemp -d "/tmp/forai-build-${SLURM_JOB_ID}-' in build
    assert "${FORAINET_TOOLCHAIN_ROOT:?set FORAINET_TOOLCHAIN_ROOT}" in build

    installer = (
        METHOD / "slurm/install_forainet_apptainer_toolchain.sbatch"
    ).read_text(encoding="utf-8")
    assert '-d el8 -v "$release_rpm"' in installer
    assert "41574717e85e03cdf40597819c927250d0772186b943b8869c8ec8dfcb5b86d1" in installer
    assert "1890dd3df87b06b0a9b2845b81b5709c0033fcca5673b03cc69ce9cb755e9605" in installer
    assert '"apptainer version 1.3.6-1"' in installer
    assert '"$apptainer" build --fakeroot' in installer
    assert '"apptainer version 1.3.6-1"' in build
    qualification = (METHOD / "slurm/qualify_forainet_assets.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu-a-lowsmall" in qualification
    assert "#SBATCH --gres=gpu:a100:1" in qualification
    assert '"$image" \\\n  python3.8 ' in qualification
    assert 'benchmark_root="$(readlink -f "$FORAINET_BENCHMARK_ROOT")"' in qualification
    assert '--bind "$benchmark_root:$benchmark_root:ro"' in qualification
    checkpoint_probe = (
        METHOD / "scripts/provenance/probe_checkpoint_load.py"
    ).read_text(encoding="utf-8")
    assert "ModelCheckpoint(" in checkpoint_probe
    assert "run_config.data.fold = []" in checkpoint_probe
    assert "checkpoint.dataset_properties" in checkpoint_probe
    assert "PretainedRegistry" not in checkpoint_probe
    smoke = (METHOD / "slurm/run_forainet_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu-a-lowsmall" in smoke
    assert "#SBATCH --gres=gpu:a100:1" in smoke
    assert "CULS/plot_1_annotated.las" in smoke
    assert "FORAINET_SMOKE_CONFIRMED" in smoke
    assert "data_split_metadata.csv" in smoke
    assert 'sub(/\\r$/, "", header)' in smoke
    assert 'sub(/\\r$/, "", value)' in smoke
    assert "ForAINet smoke failed at line" in smoke
    assert 'mkdir -p "$(dirname "$FORAINET_RUN_ROOT")"' in smoke
    assert 'mkdir "$FORAINET_RUN_ROOT"' in smoke
    assert "rev-parse --git-common-dir" in smoke
    assert '--bind "$benchmark_git_common:$benchmark_git_common:ro"' in smoke


def test_exposure_table_is_exact_and_test_only() -> None:
    rows = exposure.validate(METHOD / "examples/checkpoint_exposure_32_plots.csv")
    assert len(rows) == 32
    assert sum(row["benchmark_split"] == "dev" for row in rows) == 21
    assert sum(row["benchmark_split"] == "test" for row in rows) == 11
    assert {
        row["relative_path"]
        for row in rows
        if row["benchmark_split"] == "test"
    } == exposure.EXPECTED_TEST_PATHS

    evidence = json.loads(
        (METHOD / "examples/exposure_evidence_sources.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["retrieved_date"] == "2026-07-22"
    by_id = {row["identifier"]: row for row in evidence["sources"]}
    assert by_id["official_repository"]["git_commit"] == (
        "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
    )
    assert by_id["official_original_test_fold"]["git_blob_sha1"] == (
        "f9886421d90261f1bf80319ddfe7f5218665ddbf"
    )
    assert by_id["official_checkpoint"]["locally_computed_sha256"] == (
        "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
    )


def test_exposure_validator_rejects_test_training_role(tmp_path: Path) -> None:
    source = METHOD / "examples/checkpoint_exposure_32_plots.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["checkpoint_role"] = "train_or_validation"
    invalid = tmp_path / "invalid.csv"
    with invalid.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="test-only"):
        exposure.validate(invalid)


def test_alignment_reorders_by_exact_source_index() -> None:
    result = contract.align_full_resolution_prediction(
        source_row_index=np.asarray([2, 0, 3, 1]),
        pred_semantic_internal=np.asarray([4, 0, 1, 2]),
        pred_instance_id=np.asarray([22, -1, 0, 11]),
        expected_point_count=4,
    )
    assert result.source_row_index.tolist() == [0, 1, 2, 3]
    assert result.pred_classification.tolist() == [0, 4, 6, 0]
    assert result.pred_tree_id.tolist() == [0, 12, 23, 0]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([0, 1, 1], "duplicated or missing"),
        ([0, 1], "exactly one row"),
        ([0, 1, 3], "out-of-range"),
    ],
)
def test_alignment_rejects_invalid_row_maps(rows: list[int], message: str) -> None:
    count = 3
    with pytest.raises(ValueError, match=message):
        contract.align_full_resolution_prediction(
            source_row_index=np.asarray(rows),
            pred_semantic_internal=np.zeros(len(rows), dtype=np.int64),
            pred_instance_id=np.zeros(len(rows), dtype=np.int64),
            expected_point_count=count,
        )


def test_alignment_rejects_unknown_semantics_and_stuff_instances() -> None:
    with pytest.raises(ValueError, match="unknown ForAINet"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([0, 9]),
            pred_instance_id=np.asarray([0, 0]),
            expected_point_count=2,
        )


def test_unassigned_tree_semantics_remain_in_prediction_union() -> None:
    result = contract.align_full_resolution_prediction(
        source_row_index=np.arange(3),
        pred_semantic_internal=np.asarray([2, 3, 4]),
        pred_instance_id=np.asarray([-1, 0, 99]),
        expected_point_count=3,
    )
    assert result.pred_tree_id.tolist() == [0, 1, 100]
    assert result.pred_classification.tolist() == [4, 5, 6]
    with pytest.raises(ValueError, match="stuff classes"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([0, 2]),
            pred_instance_id=np.asarray([5, 7]),
            expected_point_count=2,
        )
    with pytest.raises(ValueError, match="must be -1 or non-negative"):
        contract.align_full_resolution_prediction(
            source_row_index=np.arange(2),
            pred_semantic_internal=np.asarray([2, 3]),
            pred_instance_id=np.asarray([-2, 0]),
            expected_point_count=2,
        )


def test_overlap_rows_collapse_only_when_identical() -> None:
    rows, semantics, instances = contract.collapse_identical_overlap_rows(
        source_row_index=np.asarray([0, 1, 1, 2]),
        pred_semantic_internal=np.asarray([0, 2, 2, 4]),
        pred_instance_id=np.asarray([0, 7, 7, 9]),
    )
    assert rows.tolist() == [0, 1, 2]
    assert semantics.tolist() == [0, 2, 4]
    assert instances.tolist() == [0, 7, 9]
    with pytest.raises(ValueError, match="conflicting overlap"):
        contract.collapse_identical_overlap_rows(
            source_row_index=np.asarray([0, 0]),
            pred_semantic_internal=np.asarray([2, 3]),
            pred_instance_id=np.asarray([7, 8]),
        )


def evaluation_payload() -> dict[str, np.ndarray]:
    return {
        "pred_tree_id": np.asarray([10, 10, 0, 20, 20, 30, 0, 0]),
        "target_tree_id": np.asarray([1, 1, 1, 2, 2, 0, 3, 3]),
        "classification": np.asarray([4, 4, 4, 5, 5, 2, 6, 6]),
        "pred_classification": np.asarray([4, 4, 0, 5, 5, 6, 0, 0]),
        "source_row_index": np.arange(8),
    }


def test_evaluator_uses_union_mask_and_maximum_matching() -> None:
    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate(evaluation_payload())
    )
    assert summary["protocol_id"] == "for_instance_pointwise_v1"
    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["f1"] == pytest.approx(2 / 3)
    assert {(row["pred_tree_id"], row["target_tree_id"]) for row in matches} == {
        (10, 1),
        (20, 2),
    }
    assert [row["pred_tree_id"] for row in unmatched_predictions] == [30]
    assert [row["target_tree_id"] for row in unmatched_references] == [3]


def test_evaluator_handles_all_background_and_noncontiguous_ids() -> None:
    payload = evaluation_payload()
    payload["pred_tree_id"] = np.zeros(8, dtype=np.int64)
    payload["pred_classification"] = np.zeros(8, dtype=np.int64)
    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate(payload)
    )
    assert summary["prediction_instance_count"] == 0
    assert summary["reference_instance_count"] == 3
    assert summary["f1"] == 0.0
    assert matches == []
    assert unmatched_predictions == []
    assert len(unmatched_references) == 3


def test_evaluator_rejects_bad_alignment_and_length() -> None:
    payload = evaluation_payload()
    payload["source_row_index"] = np.asarray([0, 1, 3, 2, 4, 5, 6, 7])
    with pytest.raises(ValueError, match="source_row_index"):
        evaluator.evaluate(payload)
    payload = evaluation_payload()
    payload["classification"] = payload["classification"][:-1]
    with pytest.raises(ValueError, match="mismatched"):
        evaluator.evaluate(payload)
    payload = evaluation_payload()
    del payload["pred_tree_id"]
    with pytest.raises(ValueError, match="missing fields"):
        evaluator.evaluate(payload)


def write_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.0, 1.0, 2.0])
    cloud.y = np.asarray([0.0, 0.0, 0.0])
    cloud.z = np.asarray([1.0, 2.0, 3.0])
    cloud.classification = np.asarray([2, 4, 5], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.asarray([0, 10, 11], dtype=np.int32)
    cloud.write(path)


def test_generated_las_sidecar_preserves_exact_rows(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text(
        "relative_path,split\nSYNTHETIC/plot.las,dev\n", encoding="utf-8"
    )
    metadata, arrays = sidecar.prepare(
        source=source,
        relative_path="SYNTHETIC/plot.las",
        split_metadata=split,
    )
    assert metadata["point_count"] == 3
    assert metadata["reference_tree_count"] == 2
    assert arrays["source_row_index"].tolist() == [0, 1, 2]
    assert arrays["x"].tolist() == [0.0, 1.0, 2.0]
    assert arrays["classification"].tolist() == [2, 4, 5]
    assert arrays["target_tree_id"].tolist() == [0, 10, 11]


def test_label_isolated_input_retains_every_row_and_hides_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text(
        "path,folder,split\nSYNTHETIC/plot.las,SYNTHETIC,dev\n",
        encoding="utf-8",
    )
    metadata, arrays = sidecar.prepare(
        source=source,
        relative_path="SYNTHETIC/plot.las",
        split_metadata=split,
    )
    assert metadata["split"] == "dev"
    assert arrays["source_row_index"].tolist() == [0, 1, 2]

    cloud = laspy.read(source)
    inference_ply = tmp_path / "input.ply"
    conversion = input_adapter.write_label_isolated_ply(inference_ply, cloud)
    vertex = PlyData.read(inference_ply)["vertex"].data
    assert len(vertex) == 3
    assert np.asarray(vertex["semantic_seg"]).tolist() == [1.0, 1.0, 1.0]
    assert np.asarray(vertex["treeID"]).tolist() == [-1.0, -1.0, -1.0]
    assert conversion["reference_classification_supplied_to_model"] is False
    assert conversion["reference_tree_id_supplied_to_model"] is False
    assert conversion["dropped_source_rows"] == 0


def write_prediction_ply(path: Path, source_ply: Path) -> None:
    source = PlyData.read(source_ply)["vertex"].data
    vertices = np.zeros(
        len(source),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("instance_preds", "i2"),
            ("semantic_preds", "i2"),
        ],
    )
    for name in ("x", "y", "z"):
        vertices[name] = source[name]
    vertices["instance_preds"] = np.asarray([-1, 0, 7], dtype=np.int16)
    vertices["semantic_preds"] = np.asarray([0, 2, 4], dtype=np.int16)
    PlyData([PlyElement.describe(vertices, "vertex")], byte_order="<").write(path)


def test_official_merge_extraction_uses_original_array_order(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    cloud = laspy.read(source)
    inference_ply = tmp_path / "input.ply"
    input_adapter.write_label_isolated_ply(inference_ply, cloud)
    merged = tmp_path / "merged.ply"
    write_prediction_ply(merged, inference_ply)
    arrays, metadata = merge_extractor.extract(merged, inference_ply, 3)
    assert arrays["source_row_index"].tolist() == [0, 1, 2]
    assert arrays["pred_instance_id"].tolist() == [-1, 0, 7]
    assert metadata["coordinate_matching_used"] is False
    assert metadata["coordinate_order_valid"] is True

    reordered = PlyData.read(merged)
    reordered["vertex"].data = reordered["vertex"].data[::-1].copy()
    reordered_path = tmp_path / "reordered.ply"
    reordered.write(reordered_path)
    with pytest.raises(ValueError, match="exact inference-Ply row order"):
        merge_extractor.extract(reordered_path, inference_ply, 3)


def test_sidecar_refuses_test_split_and_missing_fields(tmp_path: Path) -> None:
    source = tmp_path / "plot.las"
    write_las(source)
    split = tmp_path / "split.csv"
    split.write_text("relative_path,split\nSITE/plot.las,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="development-only"):
        sidecar.prepare(
            source=source,
            relative_path="SITE/plot.las",
            split_metadata=split,
        )


def test_shell_scripts_are_syntactically_valid() -> None:
    for path in sorted((METHOD / "slurm").iterdir()):
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True
        )
        assert completed.returncode == 0, (path, completed.stderr)


def test_retention_manifest_detects_missing_and_changed_files(tmp_path: Path) -> None:
    role_paths = {}
    for index, role in enumerate(sorted(retention.REQUIRED_SMOKE_ROLES)):
        path = tmp_path / f"artifact_{index}.txt"
        path.write_text(f"{role}\n", encoding="utf-8")
        role_paths[role] = path
    manifest = retention.build(tmp_path, role_paths)
    retention.validate(tmp_path, manifest)
    role_paths["aligned_prediction"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        retention.validate(tmp_path, manifest)
    incomplete = dict(role_paths)
    del incomplete["plot_metrics"]
    with pytest.raises(ValueError, match="retention roles differ"):
        retention.build(tmp_path, incomplete)


def test_no_test_submission_route_exists() -> None:
    names = {path.name for path in (METHOD / "slurm").iterdir()}
    assert not any("test" in name for name in names)
    published = yaml.safe_load(
        (METHOD / "configs/for_instance_published_test.yml").read_text(
            encoding="utf-8"
        )
    )
    assert published["gates"]["current_authorisation"] is False


def test_public_files_do_not_contain_private_paths() -> None:
    forbidden = ("/users/", "/cluster/", "/mnt/")
    for path in METHOD.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_cli_help_is_available() -> None:
    scripts = [
        METHOD / "scripts/data/prepare_alignment_sidecar.py",
        METHOD / "scripts/data/prepare_label_isolated_input.py",
        METHOD / "scripts/runtime/extract_official_merge.py",
        METHOD / "scripts/runtime/normalise_forainet_predictions.py",
        METHOD / "scripts/evaluation/evaluate_for_instance.py",
        METHOD / "scripts/provenance/validate_exposure_audit.py",
        METHOD / "scripts/provenance/verify_forainet_assets.py",
        METHOD / "scripts/provenance/probe_checkpoint_load.py",
        METHOD / "scripts/provenance/build_retention_manifest.py",
    ]
    for path in scripts:
        completed = subprocess.run(
            [sys.executable, str(path), "--help"], capture_output=True, text=True
        )
        assert completed.returncode == 0, (path, completed.stderr)


def test_evaluator_cli_writes_required_tables(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, **evaluation_payload())
    outputs = {
        "metrics": tmp_path / "metrics.json",
        "matches": tmp_path / "matches.csv",
        "unmatched_predictions": tmp_path / "unmatched_predictions.csv",
        "unmatched_references": tmp_path / "unmatched_references.csv",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(METHOD / "scripts/evaluation/evaluate_for_instance.py"),
            "--prediction-npz",
            str(prediction),
            "--metrics-json",
            str(outputs["metrics"]),
            "--matches-csv",
            str(outputs["matches"]),
            "--unmatched-predictions-csv",
            str(outputs["unmatched_predictions"]),
            "--unmatched-references-csv",
            str(outputs["unmatched_references"]),
            "--split",
            "dev",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(outputs["metrics"].read_text())["true_positives"] == 2
    assert all(path.is_file() for path in outputs.values())
