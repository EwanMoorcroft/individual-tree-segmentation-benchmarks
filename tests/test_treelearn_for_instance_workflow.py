from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
import laspy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT / "methods/treelearn/scripts/evaluate_for_instance_one_plot_smoke.py"
)
RUNNER_PATH = ROOT / "methods/treelearn/scripts/run_for_instance_one_plot_smoke.py"
ENVIRONMENT_VALIDATOR_PATH = (
    ROOT / "methods/treelearn/scripts/validate_treelearn_environment.py"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("treelearn_smoke_evaluator", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location("treelearn_smoke_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_environment_validator():
    spec = importlib.util.spec_from_file_location(
        "treelearn_environment_validator", ENVIRONMENT_VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_source_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    cloud.y = np.zeros(5)
    cloud.z = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0])
    cloud.classification = np.asarray([2, 4, 5, 6, 4], dtype=np.uint8)
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    cloud["treeID"] = np.asarray([0, 1, 1, 2, 2], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.write(path)


def known_arrays() -> dict[str, np.ndarray]:
    arrays = {
        "pred_tree_id": np.asarray(
            [10, 10, 10, 10, 20, 20, 20, 0, 0, 0, 20, 30],
            dtype=np.int64,
        ),
        "target_tree_id": np.asarray(
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0],
            dtype=np.int64,
        ),
        "classification": np.asarray(
            [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 2, 2],
            dtype=np.int64,
        ),
        "source_row_index": np.arange(12, dtype=np.int64),
    }
    arrays["pred_classification"] = np.where(
        arrays["pred_tree_id"] > 0, 4, 0
    ).astype(np.uint8)
    return arrays


def synthetic_inference_metadata(prediction: Path, run_id: str) -> dict[str, object]:
    prediction_sha256 = hashlib.sha256(prediction.read_bytes()).hexdigest()
    return {
        "status": "completed",
        "run_id": run_id,
        "plot": {
            "split": "dev",
            "relative_path": "CULS/plot_1_annotated.las",
        },
        "checkpoint": {
            "md5": "56a3d78f689ae7f1190906b975700311",
            "source_md5": "56a3d78f689ae7f1190906b975700311",
        },
        "environment": {
            "treelearn_repository": {
                "commit": "fd240ce7caa4c444fe3418aca454dc578bc557d4",
                "dirty": False,
            },
            "benchmark_repository": {"commit": "1" * 40, "dirty": False},
        },
        "outputs": {"adapted_npz": str(prediction.resolve())},
        "retention": {
            "files": [
                {
                    "path": str(prediction.resolve()),
                    "exists": True,
                    "size_bytes": prediction.stat().st_size,
                    "sha256": prediction_sha256,
                }
            ]
        },
    }


def evaluator_cli_command(
    prediction: Path, inference_metadata: Path, run_id: str, output_root: Path
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--prediction-npz",
        str(prediction),
        "--inference-metadata",
        str(inference_metadata),
        "--run-id",
        run_id,
        "--plot-id",
        "CULS/plot_1",
        "--relative-path",
        "CULS/plot_1_annotated.las",
        "--split",
        "dev",
        "--metrics-json",
        str(output_root / "metrics.json"),
        "--harmonized-matches-csv",
        str(output_root / "matches.csv"),
        "--unmatched-predictions-csv",
        str(output_root / "unmatched_predictions.csv"),
        "--unmatched-references-csv",
        str(output_root / "unmatched_references.csv"),
    ]


def test_treelearn_evaluator_known_matching_counts() -> None:
    evaluator = load_evaluator()
    arrays = known_arrays()

    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate_arrays(**arrays, plot_id="CULS/plot_1", split="dev")
    )

    assert summary["prediction_semantic_mapping"] == (
        "pred_tree_id > 0 -> class 4; else 0"
    )
    assert summary["reference_tree_classes"] == [4, 5, 6]
    assert summary["ignored_instance_labels"] == [-1, 0]
    assert summary["iou_threshold"] == 0.5
    assert summary["iou_threshold_operator"] == ">="
    assert summary["tuned_prediction_filtering"] is False
    assert summary["prediction_instance_count"] == 3
    assert summary["reference_instance_count"] == 3
    assert summary["true_positives"] == 2
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert math.isclose(summary["precision"], 2 / 3)
    assert math.isclose(summary["recall"], 2 / 3)
    assert math.isclose(summary["f1"], 2 / 3)
    assert {(row["pred_tree_id"], row["target_tree_id"]) for row in matches} == {
        (10, 1),
        (20, 2),
    }
    assert [row["pred_tree_id"] for row in unmatched_predictions] == [30]
    assert [row["target_tree_id"] for row in unmatched_references] == [3]


def test_treelearn_evaluator_all_background_predictions() -> None:
    evaluator = load_evaluator()
    target = np.asarray([1, 1, 2, 2, 0], dtype=np.int64)
    summary, matches, unmatched_predictions, unmatched_references = (
        evaluator.evaluate_arrays(
            pred_tree_id=np.zeros(5, dtype=np.int64),
            target_tree_id=target,
            classification=np.asarray([4, 4, 6, 6, 2], dtype=np.int64),
            pred_classification=np.zeros(5, dtype=np.uint8),
            source_row_index=np.arange(5, dtype=np.int64),
            plot_id="CULS/plot_1",
            split="dev",
        )
    )

    assert summary["prediction_instance_count"] == 0
    assert summary["reference_instance_count"] == 2
    assert summary["true_positives"] == 0
    assert summary["false_positives"] == 0
    assert summary["false_negatives"] == 2
    assert summary["f1"] == 0.0
    assert matches == []
    assert unmatched_predictions == []
    assert [row["target_tree_id"] for row in unmatched_references] == [1, 2]


def test_treelearn_evaluator_rejects_source_row_index_mismatch() -> None:
    evaluator = load_evaluator()
    arrays = known_arrays()
    arrays["source_row_index"] = np.asarray(
        [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11],
        dtype=np.int64,
    )

    with pytest.raises(ValueError, match="source_row_index"):
        evaluator.evaluate_arrays(
            **arrays,
            plot_id="CULS/plot_1",
            split="dev",
        )


def test_treelearn_evaluator_rejects_mismatched_and_empty_arrays() -> None:
    evaluator = load_evaluator()
    arrays = known_arrays()
    arrays["classification"] = arrays["classification"][:-1]
    with pytest.raises(ValueError, match="not aligned"):
        evaluator.evaluate_arrays(
            **arrays,
            plot_id="CULS/plot_1",
            split="dev",
        )

    empty = np.asarray([], dtype=np.int64)
    with pytest.raises(ValueError, match="at least one point"):
        evaluator.evaluate_arrays(
            pred_tree_id=empty,
            target_tree_id=empty,
            classification=empty,
            pred_classification=empty,
            source_row_index=empty,
            plot_id="CULS/plot_1",
            split="dev",
        )


def test_treelearn_evaluator_rejects_inconsistent_prediction_semantics() -> None:
    evaluator = load_evaluator()
    arrays = known_arrays()
    arrays["pred_classification"][0] = 0

    with pytest.raises(ValueError, match="pred_classification"):
        evaluator.evaluate_arrays(
            **arrays,
            plot_id="CULS/plot_1",
            split="dev",
        )


def test_treelearn_evaluator_cli_writes_stable_output_schemas(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "adapted.npz"
    np.savez_compressed(prediction, **known_arrays())
    metrics = tmp_path / "metrics.json"
    matches = tmp_path / "matches.csv"
    unmatched_predictions = tmp_path / "unmatched_predictions.csv"
    unmatched_references = tmp_path / "unmatched_references.csv"
    run_id = "treelearn_dev_smoke_synthetic"
    inference_metadata = tmp_path / "inference.json"
    inference_metadata.write_text(
        json.dumps(synthetic_inference_metadata(prediction, run_id)),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--prediction-npz",
            str(prediction),
            "--inference-metadata",
            str(inference_metadata),
            "--run-id",
            run_id,
            "--plot-id",
            "CULS/plot_1",
            "--relative-path",
            "CULS/plot_1_annotated.las",
            "--split",
            "dev",
            "--metrics-json",
            str(metrics),
            "--harmonized-matches-csv",
            str(matches),
            "--unmatched-predictions-csv",
            str(unmatched_predictions),
            "--unmatched-references-csv",
            str(unmatched_references),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["dataset_split"] == "dev"
    assert payload["evaluation_protocol"] == "for_instance_pointwise_v1"
    assert payload["matching_policy"] == "maximum_cardinality_one_to_one"
    assert payload["prediction_npz_sha256"] == hashlib.sha256(
        prediction.read_bytes()
    ).hexdigest()

    evaluator = load_evaluator()
    expected = (
        (matches, evaluator.MATCH_FIELDS),
        (unmatched_predictions, evaluator.UNMATCHED_PREDICTION_FIELDS),
        (unmatched_references, evaluator.UNMATCHED_REFERENCE_FIELDS),
    )
    for path, fieldnames in expected:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert reader.fieldnames == fieldnames
        assert rows

    empty_matches = tmp_path / "empty_matches.csv"
    evaluator.write_csv(empty_matches, [], evaluator.MATCH_FIELDS)
    assert empty_matches.read_text(encoding="utf-8").splitlines() == [
        ",".join(evaluator.MATCH_FIELDS)
    ]


def test_treelearn_evaluator_rejects_missing_field_and_mutated_prediction(
    tmp_path: Path,
) -> None:
    run_id = "treelearn_dev_smoke_integrity"
    incomplete = tmp_path / "incomplete.npz"
    incomplete_arrays = known_arrays()
    incomplete_arrays.pop("source_row_index")
    np.savez_compressed(incomplete, **incomplete_arrays)
    incomplete_metadata = tmp_path / "incomplete_metadata.json"
    incomplete_metadata.write_text(
        json.dumps(synthetic_inference_metadata(incomplete, run_id)),
        encoding="utf-8",
    )
    incomplete_outputs = tmp_path / "incomplete_outputs"

    missing = subprocess.run(
        evaluator_cli_command(
            incomplete, incomplete_metadata, run_id, incomplete_outputs
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert missing.returncode != 0
    assert "missing arrays" in missing.stderr
    assert not (incomplete_outputs / "metrics.json").exists()

    prediction = tmp_path / "mutated.npz"
    arrays = known_arrays()
    np.savez_compressed(prediction, **arrays)
    inference_metadata = tmp_path / "mutated_metadata.json"
    inference_metadata.write_text(
        json.dumps(synthetic_inference_metadata(prediction, run_id)),
        encoding="utf-8",
    )
    arrays["pred_tree_id"][0] = 11
    np.savez_compressed(prediction, **arrays)
    mutated_outputs = tmp_path / "mutated_outputs"

    mutated = subprocess.run(
        evaluator_cli_command(
            prediction, inference_metadata, run_id, mutated_outputs
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert mutated.returncode != 0
    assert "SHA-256 does not match" in mutated.stderr
    assert not (mutated_outputs / "metrics.json").exists()


def test_treelearn_config_freezes_development_smoke_contract() -> None:
    config = yaml.safe_load(
        (
            ROOT / "methods/treelearn/configs/for_instance_one_plot_smoke.yml"
        ).read_text(encoding="utf-8")
    )

    assert config["project"]["status"] == "development_smoke_route_ready_not_run"
    assert config["method"]["upstream_commit"] == (
        "fd240ce7caa4c444fe3418aca454dc578bc557d4"
    )
    assert config["method"]["checkpoint"]["filename"] == (
        "model_weights_20241213.pth"
    )
    assert config["method"]["checkpoint"]["source_dataset_name"] == (
        "model_weights_20241213"
    )
    assert config["method"]["checkpoint"]["source_md5"] == (
        "56a3d78f689ae7f1190906b975700311"
    )
    assert config["smoke"]["split"] == "dev"
    assert config["smoke"]["allow_test_split"] is False
    assert config["smoke"]["run_training"] is False
    assert config["smoke"]["run_evaluation"] is True
    assert config["evaluation"]["iou_threshold"] == 0.5
    assert config["evaluation"]["postprocessing_selection_permitted"] is False
    assert "for_instance_smokes" in config["paths"]["predictions_root"]
    assert "one_plot_smokes" in config["paths"]["tables_root"]


def test_treelearn_adapter_writes_evaluator_ready_fields(tmp_path: Path) -> None:
    runner = load_runner()
    source_path = tmp_path / "source.las"
    write_source_las(source_path)
    source = laspy.read(source_path)
    arrays = {
        "source": source,
        "pred_tree_id": np.asarray([0, 7, 7, -1, 8], dtype=np.int64),
        "target_tree_id": np.asarray(source["treeID"], dtype=np.int64),
        "classification": np.asarray(source.classification, dtype=np.int64),
        "source_row_index": np.arange(5, dtype=np.int64),
    }
    adapted_npz = tmp_path / "adapted.npz"
    adapted_las = tmp_path / "adapted.las"

    summary = runner.write_adapted_outputs(arrays, adapted_npz, adapted_las)

    assert summary["predicted_tree_count"] == 2
    with np.load(adapted_npz) as data:
        assert set(data.files) == {
            "pred_tree_id",
            "target_tree_id",
            "classification",
            "pred_classification",
            "source_row_index",
        }
        assert data["pred_classification"].tolist() == [0, 4, 4, 0, 4]
        assert np.array_equal(data["source_row_index"], np.arange(5))
    cloud = laspy.read(adapted_las)
    assert "pred_treeID" in cloud.point_format.dimension_names
    assert "pred_classification" in cloud.point_format.dimension_names
    assert np.asarray(cloud["pred_classification"]).tolist() == [0, 4, 4, 0, 4]


def test_treelearn_coordinate_alignment_gate_is_strict() -> None:
    runner = load_runner()
    runner.validate_row_alignment(0.005, 0.005)
    with pytest.raises(ValueError, match="not row-aligned"):
        runner.validate_row_alignment(0.005001, 0.005)


def test_treelearn_namespace_package_location_uses_search_locations(
    tmp_path: Path,
) -> None:
    validator = load_environment_validator()
    repo = tmp_path / "TreeLearn"
    package = repo / "tree_learn"
    package.mkdir(parents=True)

    locations = validator.validate_package_locations(repo, [str(package)])

    assert locations == [package.resolve()]
    with pytest.raises(ValueError, match="not under pinned repo"):
        validator.validate_package_locations(
            repo,
            [str(tmp_path / "another_checkout/tree_learn")],
        )


def test_treelearn_checkpoint_identity_is_not_self_derived(tmp_path: Path) -> None:
    runner = load_runner()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic released checkpoint")
    expected_md5 = hashlib.md5(
        checkpoint.read_bytes(), usedforsecurity=False
    ).hexdigest()

    identity = runner.validate_checkpoint_identity(checkpoint, expected_md5)

    assert identity["md5"] == expected_md5
    assert len(identity["sha256"]) == 64
    with pytest.raises(ValueError, match="does not match official"):
        runner.validate_checkpoint_identity(checkpoint, "0" * 32)


def test_treelearn_runner_records_preflight_failure_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    dataset = tmp_path / "dataset"
    write_source_las(dataset / "CULS/plot_1_annotated.las")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic checkpoint")
    metadata_base = tmp_path / "metadata"
    run_id = "synthetic_preflight_failure"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            run_id,
            "--dataset-root",
            str(dataset),
            "--treelearn-repo",
            str(tmp_path / "missing-treelearn"),
            "--checkpoint",
            str(checkpoint),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--predictions-root",
            str(tmp_path / "predictions"),
            "--metadata-root",
            str(metadata_base),
            "--tables-root",
            str(tmp_path / "tables"),
        ],
    )
    monkeypatch.setattr(
        runner,
        "benchmark_repository_state",
        lambda *_args, **_kwargs: {
            "commit": "1" * 40,
            "branch": "synthetic",
            "dirty": False,
        },
    )

    with pytest.raises(FileNotFoundError, match="missing-treelearn"):
        runner.main()

    metadata = metadata_base / run_id / "CULS_plot_1_annotated_inference.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"]["type"] == "FileNotFoundError"
    assert payload["retention"]["raw_full_forest_output_retained"] is False
    assert all(not entry["exists"] for entry in payload["retention"]["files"])


def test_treelearn_runner_refuses_existing_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    run_id = "synthetic_collision"
    runtime_base = tmp_path / "runtime"
    (runtime_base / run_id / "CULS_plot_1_annotated").mkdir(parents=True)
    metadata_base = tmp_path / "metadata"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--run-id",
            run_id,
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--treelearn-repo",
            str(tmp_path / "treelearn"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pth"),
            "--runtime-root",
            str(runtime_base),
            "--predictions-root",
            str(tmp_path / "predictions"),
            "--metadata-root",
            str(metadata_base),
            "--tables-root",
            str(tmp_path / "tables"),
        ],
    )

    with pytest.raises(FileExistsError, match="use a new run ID"):
        runner.main()

    assert not metadata_base.exists()


def test_treelearn_dataset_validation_uses_real_split_manifest(tmp_path: Path) -> None:
    runner = load_runner()
    config = yaml.safe_load(
        (
            ROOT / "methods/treelearn/configs/for_instance_one_plot_smoke.yml"
        ).read_text(encoding="utf-8")
    )
    config["smoke"]["expected_point_count"] = 5
    config["smoke"]["expected_reference_tree_count"] = 2
    dataset = tmp_path / "FORinstance_dataset"
    source = dataset / "CULS/plot_1_annotated.las"
    write_source_las(source)
    split = dataset / "data_split_metadata.csv"
    split.write_text(
        "relative_path,split\nCULS/plot_1_annotated.las,dev\n",
        encoding="utf-8",
    )

    record = runner.validate_dataset_source(config, dataset, source)

    assert record["split"] == "dev"
    assert record["point_count"] == 5
    assert record["reference_tree_count"] == 2
    assert record["dropped_points"] == 0
    split.write_text(
        "relative_path,split\nCULS/plot_1_annotated.las,test\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="development split"):
        runner.validate_dataset_source(config, dataset, source)

    split.write_text(
        "relative_path,split\nNIBIO/plot_1_annotated.las,dev\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="found 0"):
        runner.validate_dataset_source(config, dataset, source)

    split.write_text(
        "relative_path,split\n/archive/CULS/plot_1_annotated.las,dev\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="found 0"):
        runner.validate_dataset_source(config, dataset, source)


def test_treelearn_slurm_chain_is_guarded_and_development_only() -> None:
    scripts = {
        "setup": "methods/treelearn/slurm/setup_treelearn_environment.sbatch",
        "inference": "methods/treelearn/slurm/run_for_instance_one_plot_smoke.sbatch",
        "evaluation": "methods/treelearn/slurm/evaluate_for_instance_one_plot_smoke.sbatch",
        "submitter": "methods/treelearn/slurm/submit_for_instance_one_plot_smoke.sh",
        "monitor": "methods/treelearn/slurm/monitor_for_instance_one_plot_smoke.sh",
    }
    for relative_path in scripts.values():
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / relative_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (relative_path, completed.stderr)

    setup = (ROOT / scripts["setup"]).read_text(encoding="utf-8")
    inference = (ROOT / scripts["inference"]).read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    validator = ENVIRONMENT_VALIDATOR_PATH.read_text(encoding="utf-8")
    evaluation = (ROOT / scripts["evaluation"]).read_text(encoding="utf-8")
    submitter = (ROOT / scripts["submitter"]).read_text(encoding="utf-8")
    monitor = (ROOT / scripts["monitor"]).read_text(encoding="utf-8")
    assert "TREELEARN_SETUP_CONFIRMED" in setup
    assert "fd240ce7caa4c444fe3418aca454dc578bc557d4" in setup
    assert "model_weights_20241213" in setup
    assert "56a3d78f689ae7f1190906b975700311" in setup
    assert ".treelearn_setup_complete" in setup
    assert "TREELEARN_SETUP_RESUME_PARTIAL" in setup
    assert "reusing_complete_unmarked_treelearn_env" in setup
    assert '"setuptools==80.9.0"' in setup
    assert "validate_treelearn_environment.py" in setup
    assert "conda env remove" not in setup
    assert "--untracked-files=no" not in setup
    assert "TREELEARN_RUN_ID" in inference
    assert "TREELEARN_EXPECTED_CHECKPOINT_MD5" in inference
    assert "validate_treelearn_environment.py" in inference
    assert "tree_learn.__file__" not in inference
    assert "submodule_search_locations" in validator
    assert 'EXPECTED_SETUPTOOLS = "80.9.0"' in validator
    assert "import pkg_resources" in validator
    assert "sys.version_info[:2] != EXPECTED_PYTHON" in validator
    assert 'torch.__version__.split("+")[0] != EXPECTED_TORCH' in validator
    assert "torch.version.cuda != EXPECTED_TORCH_CUDA" in validator
    assert "TREELEARN_EXPECTED_BENCHMARK_COMMIT" in inference
    assert 'conda activate "$TREELEARN_ENV"' in inference
    assert '$TREELEARN_ENV/bin/activate' not in inference
    assert "--overwrite" not in inference
    assert 'parser.add_argument("--overwrite"' not in runner
    assert 'parser.add_argument("--dry-run"' not in runner
    assert "--split dev" in evaluation
    assert "PARTIAL_ROOT" in evaluation
    assert 'mv "$PARTIAL_ROOT" "$TABLE_ROOT"' in evaluation
    assert "afterok:$INFERENCE_JOB" in submitter
    assert "validate_treelearn_environment.py" in submitter
    assert "--kill-on-invalid-dep=yes" in submitter
    assert "TREELEARN_BENCHMARK_COMMIT" in submitter
    assert "56a3d78f689ae7f1190906b975700311" in submitter
    assert 'scancel "$INFERENCE_JOB"' in submitter
    assert "evaluation_submission_failed_inference_cancelled" in submitter
    assert "No training, full development array or held-out test job was submitted" in submitter
    assert "--array" not in submitter
    assert "tail " not in monitor
    assert "unmatched_predictions.csv" in monitor
    assert "unmatched_references.csv" in monitor
    assert 'EVALUATION_STATE" == COMPLETED' in monitor
    assert "development-smoke-inference-failed" in monitor
