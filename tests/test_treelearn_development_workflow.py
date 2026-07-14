from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "methods/treelearn/scripts"
CONFIG = ROOT / "methods/treelearn/configs/for_instance_development.yml"
EVALUATOR = SCRIPTS / "evaluate_for_instance_one_plot_smoke.py"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


SITE_COUNTS = {
    "CULS": 2,
    "NIBIO": 14,
    "RMIT": 1,
    "SCION": 3,
    "TUWIEN": 1,
}
SITE_POINTS = {
    "CULS": 4_901_588,
    "NIBIO": 79_435_164,
    "RMIT": 1_483_208,
    "SCION": 8_380_233,
    "TUWIEN": 7_568_844,
}
SITE_REFERENCES = {
    "CULS": 27,
    "NIBIO": 414,
    "RMIT": 159,
    "SCION": 92,
    "TUWIEN": 115,
}
EXPECTED_PATHS = (
    "CULS/plot_1_annotated.las",
    "CULS/plot_3_annotated.las",
    "NIBIO/plot_10_annotated.las",
    "NIBIO/plot_11_annotated.las",
    "NIBIO/plot_12_annotated.las",
    "NIBIO/plot_13_annotated.las",
    "NIBIO/plot_16_annotated.las",
    "NIBIO/plot_19_annotated.las",
    "NIBIO/plot_21_annotated.las",
    "NIBIO/plot_2_annotated.las",
    "NIBIO/plot_3_annotated.las",
    "NIBIO/plot_4_annotated.las",
    "NIBIO/plot_6_annotated.las",
    "NIBIO/plot_7_annotated.las",
    "NIBIO/plot_8_annotated.las",
    "NIBIO/plot_9_annotated.las",
    "RMIT/train.las",
    "SCION/plot_35_annotated.las",
    "SCION/plot_39_annotated.las",
    "SCION/plot_87_annotated.las",
    "TUWIEN/train.las",
)


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inventory_paths() -> list[str]:
    return list(EXPECTED_PATHS)


def inventory_contract() -> dict[str, tuple[int, int]]:
    contract: dict[str, tuple[int, int]] = {}
    for site, count in SITE_COUNTS.items():
        paths = [path for path in EXPECTED_PATHS if path.startswith(f"{site}/")]
        assert len(paths) == count
        point_counts = [SITE_POINTS[site] - 2 * (count - 1)] + [2] * (count - 1)
        if site == "CULS":
            reference_counts = [1, 26]
        else:
            reference_counts = [SITE_REFERENCES[site] - (count - 1)] + [1] * (
                count - 1
            )
        for path, points, references in zip(
            paths, point_counts, reference_counts, strict=True
        ):
            contract[path] = (points, references)
    return contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_inventory(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    dataset_root = tmp_path / "FORinstance_dataset"
    paths = inventory_paths()
    rows = []
    for index, relative_path in enumerate(reversed(paths), start=1):
        source = dataset_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"synthetic development source {index}".encode())
        rows.append(
            {
                "path": relative_path,
                "folder": relative_path.split("/", 1)[0],
                "split": "dev",
            }
        )
    # A held-out row is deliberately absent on disk. Selection must skip it
    # before checking availability and must never include it in either list.
    rows.append(
        {
            "path": "CULS/held_out_test_plot.las",
            "folder": "CULS",
            "split": "test",
        }
    )
    metadata = dataset_root / "data_split_metadata.csv"
    write_csv(metadata, ["path", "folder", "split"], rows)
    return dataset_root, metadata, paths


def manifest_rows(tmp_path: Path) -> list[dict[str, Any]]:
    common = load_script(
        "for_instance_development_common",
        "for_instance_development_common.py",
    )
    split_metadata = (tmp_path / "data_split_metadata.csv").resolve()
    rows: list[dict[str, Any]] = []
    for task_index, relative_path in enumerate(inventory_paths()):
        identifier = common.plot_id(relative_path)
        point_count, reference_tree_count = inventory_contract()[relative_path]
        rows.append(
            {
                "task_index": task_index,
                "plot_id": identifier,
                "safe_plot_id": common.safe_plot_id(identifier),
                "relative_path": relative_path,
                "collection": relative_path.split("/", 1)[0],
                "split": "dev",
                "input_las": str((tmp_path / "dataset" / relative_path).resolve()),
                "point_count": point_count,
                "reference_tree_count": reference_tree_count,
                "input_sha256": f"{task_index + 1:064x}",
                "split_metadata": str(split_metadata),
                "split_metadata_sha256": "a" * 64,
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen_exact_path_development_manifest",
                "dataset_split": "dev",
                "held_out_test_accessed": False,
                "plots": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_manifest_builder_freezes_exact_21_plot_five_site_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = load_script(
        "for_instance_development_common",
        "for_instance_development_common.py",
    )
    builder = load_script(
        "treelearn_development_manifest",
        "prepare_for_instance_development_manifest.py",
    )
    dataset_root, metadata, expected_paths = make_inventory(tmp_path)
    contract = inventory_contract()
    monkeypatch.setattr(
        builder,
        "inspect_las",
        lambda path: contract[path.relative_to(dataset_root).as_posix()],
    )

    available, unavailable = builder.read_available_development_rows(
        dataset_root.resolve(), metadata.resolve()
    )
    payload = builder.build_manifest(dataset_root, metadata)

    assert len(available) == 21
    assert unavailable == []
    assert payload["held_out_test_accessed"] is False
    assert payload["mapping_rule"] == "exact_metadata_path_only"
    assert payload["available_site_counts"] == SITE_COUNTS
    assert [row["task_index"] for row in payload["plots"]] == list(range(21))
    assert [row["relative_path"] for row in payload["plots"]] == expected_paths
    assert Counter(row["collection"] for row in payload["plots"]) == SITE_COUNTS
    assert all(row["split"] == "dev" for row in payload["plots"])
    assert all(
        row["safe_plot_id"] == common.safe_plot_id(row["plot_id"])
        for row in payload["plots"]
    )
    assert all(len(row["input_sha256"]) == 64 for row in payload["plots"])


def test_manifest_builder_rejects_missing_duplicate_and_path_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = load_script(
        "treelearn_development_manifest_rejections",
        "prepare_for_instance_development_manifest.py",
    )
    dataset_root, metadata, paths = make_inventory(tmp_path)
    contract = inventory_contract()
    monkeypatch.setattr(
        builder,
        "inspect_las",
        lambda path: contract[path.relative_to(dataset_root).as_posix()],
    )
    (dataset_root / paths[0]).unlink()
    with pytest.raises(ValueError, match="exactly 21 available development plots"):
        builder.build_manifest(dataset_root, metadata)

    dataset_root, metadata, _ = make_inventory(tmp_path / "duplicate")
    with metadata.open("a", encoding="utf-8") as handle:
        handle.write(f"{EXPECTED_PATHS[0]},CULS,dev\n")
    with pytest.raises(ValueError, match="Duplicate development metadata path"):
        builder.read_available_development_rows(dataset_root.resolve(), metadata.resolve())

    dataset_root, metadata, _ = make_inventory(tmp_path / "mismatch")
    text = metadata.read_text(encoding="utf-8")
    metadata.write_text(
        text.replace(
            f"{EXPECTED_PATHS[0]},CULS,dev",
            f"{EXPECTED_PATHS[0]},NIBIO,dev",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="folder/path mismatch"):
        builder.read_available_development_rows(dataset_root.resolve(), metadata.resolve())


def test_manifest_contract_rejects_test_duplicate_and_unsafe_rows(tmp_path: Path) -> None:
    common = load_script(
        "for_instance_development_common_contract",
        "for_instance_development_common.py",
    )
    rows = manifest_rows(tmp_path)

    test_rows = [dict(row) for row in rows]
    test_rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="Non-development"):
        common.validate_manifest_rows(test_rows)

    duplicate_rows = [dict(row) for row in rows]
    duplicate_rows[1].update(
        {
            "relative_path": duplicate_rows[0]["relative_path"],
            "plot_id": duplicate_rows[0]["plot_id"],
            "safe_plot_id": duplicate_rows[0]["safe_plot_id"],
            "collection": duplicate_rows[0]["collection"],
        }
    )
    with pytest.raises(ValueError, match="duplicate relative paths"):
        common.validate_manifest_rows(duplicate_rows)

    with pytest.raises(ValueError, match="Unsafe|non-canonical"):
        common.strict_relative_path("../test/plot.las")
    with pytest.raises(ValueError, match="LAS file"):
        common.strict_relative_path("CULS/plot.laz")

    relative_metadata_rows = [dict(row) for row in rows]
    relative_metadata_rows[0]["split_metadata"] = "data_split_metadata.csv"
    with pytest.raises(ValueError, match="split metadata is not absolute"):
        common.validate_manifest_rows(relative_metadata_rows)

    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rows)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["held_out_test_accessed"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lock held-out test"):
        common.load_manifest(manifest)


def test_development_tasks_reject_traversal_run_ids_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = load_script(
        "treelearn_development_task_run_id",
        "run_for_instance_development_task.py",
    )
    evaluator = load_script(
        "treelearn_development_evaluator_run_id",
        "evaluate_for_instance_development_task.py",
    )

    with pytest.raises(ValueError, match="Unsafe TreeLearn run ID"):
        task.write_preflight_failure(
            "missing.yml",
            "../../escape",
            0,
            None,
            ValueError("synthetic failure"),
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_for_instance_development_task.py",
            "--config",
            "missing.yml",
            "--manifest",
            "missing.json",
            "--task-index",
            "0",
            "--run-id",
            str(tmp_path / "absolute_escape"),
        ],
    )
    with pytest.raises(ValueError, match="Unsafe TreeLearn run ID"):
        evaluator.main()
    assert not (tmp_path / "absolute_escape").exists()


def test_development_task_selection_is_deterministic_and_strict(tmp_path: Path) -> None:
    common = load_script(
        "for_instance_development_common",
        "for_instance_development_common.py",
    )
    task = load_script(
        "treelearn_development_task_selection",
        "run_for_instance_development_task.py",
    )
    rows = manifest_rows(tmp_path)
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, common.MANIFEST_FIELDS, rows)

    selected = task.read_manifest_row(manifest, 7)
    assert selected["task_index"] == 7
    assert selected["relative_path"] == rows[7]["relative_path"]
    with pytest.raises(ValueError, match="found 0"):
        task.read_manifest_row(manifest, 99)

    duplicated = [dict(row) for row in rows]
    duplicated[8]["task_index"] = 7
    write_csv(tmp_path / "duplicate.csv", common.MANIFEST_FIELDS, duplicated)
    with pytest.raises(ValueError, match="contiguous"):
        task.read_manifest_row(tmp_path / "duplicate.csv", 7)

    non_dev = [dict(row) for row in rows]
    non_dev[7]["split"] = "test"
    write_csv(tmp_path / "test.csv", common.MANIFEST_FIELDS, non_dev)
    with pytest.raises(ValueError, match="Non-development"):
        task.read_manifest_row(tmp_path / "test.csv", 7)


def test_development_task_rejects_manifest_dataset_path_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = load_script(
        "for_instance_development_common",
        "for_instance_development_common.py",
    )
    task = load_script(
        "treelearn_development_task_path_gate",
        "run_for_instance_development_task.py",
    )
    dataset_root = tmp_path / "dataset"
    rows = manifest_rows(tmp_path)
    row = rows[0]
    relative_path = row["relative_path"]
    source = dataset_root / relative_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source")
    row["input_las"] = str((tmp_path / "wrong" / relative_path).resolve())
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, common.MANIFEST_FIELDS, rows)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    metadata_root = tmp_path / "metadata"
    config["paths"]["metadata_root"] = str(metadata_root)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "run_for_instance_development_task.py"),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest),
            "--task-index",
            "0",
            "--run-id",
            "synthetic_path_mismatch",
            "--dataset-root",
            str(dataset_root),
            "--treelearn-repo",
            str(tmp_path / "TreeLearn"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pth"),
        ],
    )

    with pytest.raises(ValueError, match="does not match the dataset root"):
        task.main()

    failure = json.loads(
        (
            metadata_root
            / "synthetic_path_mismatch"
            / f"{row['safe_plot_id']}_inference.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_preflight"
    assert failure["held_out_test_accessed"] is False


def test_development_task_builds_string_only_subprocess_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = load_script(
        "treelearn_development_task_command_types",
        "run_for_instance_development_task.py",
    )
    row = manifest_rows(tmp_path)[0]
    dataset_root = tmp_path / "dataset"
    source = dataset_root / row["relative_path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic source")
    row["input_las"] = str(source.resolve())
    captured: list[str] = []

    monkeypatch.setattr(task, "read_manifest_row", lambda _path, _index: row)

    def capture(command, *, cwd, check):
        assert cwd == task.runner.ROOT
        assert check is True
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(task.subprocess, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPTS / "run_for_instance_development_task.py"),
            "--config",
            str(CONFIG),
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--task-index",
            "0",
            "--run-id",
            "synthetic_command_types",
            "--dataset-root",
            str(dataset_root),
            "--treelearn-repo",
            str(tmp_path / "TreeLearn"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pth"),
        ],
    )

    assert task.main() == 0
    assert captured
    assert all(isinstance(value, str) for value in captured)
    assert captured[captured.index("--expected-point-count") + 1] == str(
        row["point_count"]
    )


def synthetic_smoke_evidence(tmp_path: Path, config: dict[str, Any]):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"synthetic published checkpoint")
    checkpoint_md5 = hashlib.md5(
        checkpoint.read_bytes(), usedforsecurity=False
    ).hexdigest()
    checkpoint_sha = sha256(checkpoint)
    config["method"]["checkpoint"]["source_md5"] = checkpoint_md5
    config["method"]["checkpoint"]["sha256"] = checkpoint_sha
    roles = (
        "raw_prediction_laz",
        "raw_prediction_npz",
        "raw_pointwise_npz",
        "adapted_npz",
        "adapted_las",
    )
    retained = []
    outputs = {}
    retained_contract = {}
    for index, role in enumerate(roles):
        artifact = tmp_path / f"retained_{index}.bin"
        artifact.write_bytes(f"retained prediction {index}".encode())
        outputs[role] = str(artifact.resolve())
        retained_contract[role] = {
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        }
        retained.append(
            {
                "path": str(artifact.resolve()),
                "exists": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
        )
    expected = config["accepted_smoke"]
    expected["benchmark_commit"] = "b" * 40
    expected["retained_artifacts"] = retained_contract
    inference = {
        "status": "completed",
        "run_id": expected["run_id"],
        "plot": {
            "split": "dev",
            "relative_path": expected["relative_path"],
        },
        "validation": {
            "row_count_match": True,
            "row_order_preserved": True,
            "source_point_count": expected["point_count"],
            "max_abs_coordinate_delta_m": expected[
                "max_abs_coordinate_delta_m"
            ],
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "md5": checkpoint_md5,
            "sha256": checkpoint_sha,
        },
        "environment": {
            "treelearn_repository": {
                "commit": config["method"]["upstream_commit"],
                "dirty": False,
            },
            "benchmark_repository": {"commit": "b" * 40, "dirty": False},
        },
        "outputs": outputs,
        "retention": {"files": retained},
    }
    metrics = {
        "status": "completed_development_smoke_evaluation",
        "run_id": expected["run_id"],
        "reference_instance_count": expected["reference_instances"],
        "prediction_instance_count": expected["predicted_instances"],
        "true_positives": expected["true_positives"],
        "false_positives": expected["false_positives"],
        "false_negatives": expected["false_negatives"],
        "precision": expected["true_positives"] / expected["predicted_instances"],
        "recall": 1.0,
        "f1": expected["f1"],
    }
    inference_path = tmp_path / "smoke_inference.json"
    metrics_path = tmp_path / "smoke_metrics.json"
    inference_path.write_text(json.dumps(inference), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return inference_path, metrics_path, retained


def test_accepted_smoke_identity_and_five_retained_files_are_refrozen(
    tmp_path: Path,
) -> None:
    load_script(
        "run_for_instance_one_plot_smoke",
        "run_for_instance_one_plot_smoke.py",
    )
    acceptance = load_script(
        "treelearn_development_acceptance",
        "prepare_for_instance_development_acceptance.py",
    )
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inference_path, metrics_path, retained = synthetic_smoke_evidence(tmp_path, config)
    output = tmp_path / "accepted.json"

    with pytest.raises(ValueError, match="alignment review"):
        acceptance.prepare_acceptance(
            config, inference_path, metrics_path, output, False
        )
    record = acceptance.prepare_acceptance(
        config, inference_path, metrics_path, output, True
    )

    assert record["status"] == "accepted_for_full_development_only"
    assert record["held_out_test_accessed"] is False
    assert record["manual_alignment_review_confirmed"] is True
    assert len(record["verified_retained_files"]) == 5
    assert output.is_file()

    first = Path(retained[0]["path"])
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="failed verification"):
        acceptance.prepare_acceptance(
            config,
            inference_path,
            metrics_path,
            tmp_path / "should_not_exist.json",
            True,
        )


def evaluator_arrays() -> dict[str, np.ndarray]:
    pred = np.zeros(4, dtype=np.int64)
    return {
        "pred_tree_id": pred,
        "target_tree_id": np.asarray([10, 10, 0, 20], dtype=np.int64),
        "classification": np.asarray([4, 4, 2, 5], dtype=np.int64),
        "pred_classification": np.where(pred > 0, 4, 0).astype(np.uint8),
        "source_row_index": np.arange(4, dtype=np.int64),
    }


def test_full_scope_evaluator_accepts_zero_predictions_and_marks_development(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, **evaluator_arrays())
    run_id = "treelearn_development_synthetic"
    metadata = {
        "status": "completed",
        "run_id": run_id,
        "plot": {"split": "dev", "relative_path": "CULS/plot.las"},
        "checkpoint": {
            "md5": "56a3d78f689ae7f1190906b975700311",
            "source_md5": "56a3d78f689ae7f1190906b975700311",
        },
        "environment": {
            "treelearn_repository": {
                "commit": "fd240ce7caa4c444fe3418aca454dc578bc557d4",
                "dirty": False,
            },
            "benchmark_repository": {"commit": "c" * 40, "dirty": False},
        },
        "outputs": {"adapted_npz": str(prediction.resolve())},
        "retention": {
            "files": [
                {
                    "path": str(prediction.resolve()),
                    "exists": True,
                    "size_bytes": prediction.stat().st_size,
                    "sha256": sha256(prediction),
                }
            ]
        },
    }
    metadata_path = tmp_path / "inference.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "evaluation"
    command = [
        sys.executable,
        str(EVALUATOR),
        "--prediction-npz",
        str(prediction),
        "--inference-metadata",
        str(metadata_path),
        "--run-id",
        run_id,
        "--plot-id",
        "CULS/plot",
        "--relative-path",
        "CULS/plot.las",
        "--split",
        "dev",
        "--metrics-json",
        str(output / "metrics.json"),
        "--harmonized-matches-csv",
        str(output / "matches.csv"),
        "--unmatched-predictions-csv",
        str(output / "unmatched_predictions.csv"),
        "--unmatched-references-csv",
        str(output / "unmatched_references.csv"),
        "--evaluation-scope",
        "development_full",
    ]

    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "completed_aligned_pointwise_development_plot"
    assert metrics["evaluation_scope"] == "development_full"
    assert metrics["dataset_split"] == "dev"
    assert metrics["prediction_instance_count"] == 0
    assert metrics["f1"] == 0.0
    assert metrics["next_gate"] == (
        "aggregate_full_development_results_before_any_test_route"
    )


def metric_counts(task_index: int, reference_count: int) -> tuple[int, int, int]:
    if task_index == 0:
        return 1, 9, reference_count - 1
    if task_index == 1:
        return 9, 0, reference_count - 9
    return 1, 0, reference_count - 1


def build_synthetic_completed_run(
    tmp_path: Path,
    *,
    failed_task: int | None = None,
) -> tuple[Path, Path, Path, Path, list[dict[str, Any]]]:
    summarizer = load_script(
        "treelearn_development_summary_helpers",
        "summarise_for_instance_development.py",
    )
    rows = manifest_rows(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rows)
    evaluation_root = tmp_path / "per_plot"
    metadata_root = tmp_path / "metadata"
    output_root = tmp_path / "summary"
    run_id = "treelearn_development_21_plot_synthetic"

    for row in rows:
        task_index = row["task_index"]
        per_plot = evaluation_root / row["safe_plot_id"]
        if task_index == failed_task:
            failure = {
                "method": "TreeLearn",
                "dataset": "FOR-instance",
                "run_id": run_id,
                "task_index": task_index,
                "plot_id": row["plot_id"],
                "relative_path": row["relative_path"],
                "collection": row["collection"],
                "split": "dev",
                "dataset_split": "dev",
                "status": "documented_inference_failure",
                "held_out_test_accessed": False,
                "inference_status": "failed_preflight",
                "inference_error": {"type": "RuntimeError", "message": "synthetic"},
                "error": None,
            }
            per_plot.mkdir(parents=True, exist_ok=True)
            (per_plot / "status.json").write_text(
                json.dumps(failure), encoding="utf-8"
            )
            continue

        retained_entries = []
        outputs = {}
        roles = (
            "raw_prediction_laz",
            "raw_prediction_npz",
            "raw_pointwise_npz",
            "adapted_npz",
            "adapted_las",
        )
        prediction_root = tmp_path / "predictions" / row["safe_plot_id"]
        for role in roles:
            artifact = prediction_root / f"{role}.bin"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(f"{task_index}:{role}".encode())
            outputs[role] = str(artifact.resolve())
            retained_entries.append(
                {
                    "path": str(artifact.resolve()),
                    "exists": True,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": sha256(artifact),
                }
            )
        metadata_path = metadata_root / f"{row['safe_plot_id']}_inference.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "method": "treelearn",
            "dataset": "for-instance",
            "dataset_split": "dev",
            "training_mode": "published_pretrained",
            "status": "completed",
            "evaluation_scope": "development_full",
            "run_id": run_id,
            "held_out_test_accessed": False,
            "checkpoint": {
                "md5": "56a3d78f689ae7f1190906b975700311",
                "sha256": (
                    "5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"
                ),
            },
            "environment": {
                "treelearn_repository": {
                    "commit": "fd240ce7caa4c444fe3418aca454dc578bc557d4",
                    "dirty": False,
                },
                "benchmark_repository": {
                    "commit": "d" * 40,
                    "dirty": False,
                },
            },
            "plot": {
                field: row[field]
                for field in (
                    "plot_id",
                    "safe_plot_id",
                    "relative_path",
                    "collection",
                    "split",
                )
            },
            "outputs": outputs,
            "retention": {
                "raw_pointwise_output_retained": True,
                "raw_full_forest_output_retained": True,
                "adapted_point_aligned_output_retained": True,
                "files": retained_entries,
            },
            "dataset_validation": {
                "input_sha256": row["input_sha256"],
                "split_metadata_sha256": row["split_metadata_sha256"],
                "point_count": row["point_count"],
                "reference_tree_count": row["reference_tree_count"],
            },
            "validation": {
                "row_count_match": True,
                "row_order_preserved": True,
                "source_point_count": row["point_count"],
                "prediction_point_count": row["point_count"],
                "max_abs_coordinate_delta_m": 0.0002,
                "row_coordinate_tolerance_m": 0.005,
            },
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        tp, fp, fn = metric_counts(task_index, row["reference_tree_count"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        mean_iou = 0.8 if tp else 0.0
        adapted = Path(outputs["adapted_npz"])
        metrics = {
            "status": "completed_aligned_pointwise_development_plot",
            "evaluation_scope": "development_full",
            "held_out_test_accessed": False,
            "run_id": run_id,
            "plot_id": row["plot_id"],
            "relative_path": row["relative_path"],
            "split": "dev",
            "dataset_split": "dev",
            "evaluation_protocol": "for_instance_pointwise_v1",
            "matching_policy": "maximum_cardinality_one_to_one",
            "evaluation_mask": "union_of_reference_tree_and_predicted_tree_points",
            "iou_threshold": 0.5,
            "iou_threshold_operator": ">=",
            "point_correspondence": "source_row_index",
            "prediction_semantic_mapping": "pred_tree_id > 0 -> class 4; else 0",
            "reference_tree_classes": [4, 5, 6],
            "prediction_tree_classes": [4],
            "ignored_instance_labels": [-1, 0],
            "tuned_prediction_filtering": False,
            "min_predicted_instance_points": 0,
            "min_predicted_tree_fraction": 0.0,
            "inference_metadata": str(metadata_path.resolve()),
            "prediction_npz": str(adapted.resolve()),
            "prediction_npz_sha256": sha256(adapted),
            "prediction_npz_size_bytes": adapted.stat().st_size,
            "point_count": row["point_count"],
            "evaluated_point_count": row["point_count"] - 1,
            "prediction_instance_count": tp + fp,
            "reference_instance_count": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_iou,
            "mean_unweighted_coverage": 0.7,
            "mean_weighted_coverage": 0.75,
            "harmonized": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": mean_iou,
            },
        }
        per_plot.mkdir(parents=True, exist_ok=True)
        (per_plot / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        matches = [
            {
                "plot_id": row["plot_id"],
                "pred_tree_id": index + 1,
                "target_tree_id": index + 101,
                "intersection_points": 8,
                "predicted_points": 10,
                "reference_points": 8,
                "union_points": 10,
                "iou": 0.8,
            }
            for index in range(tp)
        ]
        unmatched_predictions = [
            {
                "plot_id": row["plot_id"],
                "pred_tree_id": tp + index + 1,
                "predicted_points": 5,
                "best_target_tree_id": "",
                "best_iou": 0.0,
            }
            for index in range(fp)
        ]
        unmatched_references = [
            {
                "plot_id": row["plot_id"],
                "target_tree_id": tp + index + 101,
                "reference_points": 5,
                "best_pred_tree_id": "",
                "best_iou": 0.0,
            }
            for index in range(fn)
        ]
        write_csv(per_plot / "matches.csv", summarizer.MATCH_FIELDS, matches)
        write_csv(
            per_plot / "unmatched_predictions.csv",
            summarizer.UNMATCHED_PREDICTION_FIELDS,
            unmatched_predictions,
        )
        write_csv(
            per_plot / "unmatched_references.csv",
            summarizer.UNMATCHED_REFERENCE_FIELDS,
            unmatched_references,
        )
    return manifest, evaluation_root, metadata_root, output_root, rows


def test_development_summary_uses_count_first_micro_metrics_and_retains_outputs(
    tmp_path: Path,
) -> None:
    summarizer = load_script(
        "treelearn_development_summary_complete",
        "summarise_for_instance_development.py",
    )
    manifest, evaluation_root, metadata_root, output_root, rows = (
        build_synthetic_completed_run(tmp_path)
    )
    run_id = "treelearn_development_21_plot_synthetic"

    result = summarizer.summarise(
        manifest, run_id, evaluation_root, metadata_root, output_root
    )

    assert result["status"] == "completed_aligned_pointwise_development"
    assert result["completed_plots"] == 21
    assert result["documented_failures"] == 0
    assert result["held_out_test_accessed"] is False
    gate = load_script(
        "treelearn_development_final_gate",
        "validate_for_instance_development_run.py",
    )
    validated = gate.validate(output_root / "run_summary.json", "d" * 40)
    assert validated["status"] == "completed_aligned_pointwise_development"
    with pytest.raises(ValueError, match="submitted commit"):
        gate.validate(output_root / "run_summary.json", "e" * 40)
    with (output_root / "site_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        site_rows = {row["site"]: row for row in csv.DictReader(handle)}
    culs = site_rows["CULS"]
    assert int(culs["true_positives"]) == 10
    assert int(culs["false_positives"]) == 9
    assert int(culs["false_negatives"]) == 17
    assert math.isclose(float(culs["micro_f1"]), 20 / 46)
    assert not math.isclose(
        float(culs["micro_f1"]), float(culs["mean_plot_f1"])
    )
    with (output_root / "development_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        development = next(csv.DictReader(handle))
    assert development["site"] == "ALL"
    assert int(development["completed_plots"]) == 21
    assert int(development["true_positives"]) == 29
    assert int(development["false_positives"]) == 9
    assert int(development["false_negatives"]) == 778
    assert math.isclose(float(development["micro_f1"]), 58 / 845)

    retention = json.loads(
        (output_root / "retention_manifest.json").read_text(encoding="utf-8")
    )
    assert retention["status"] == "retention_verified"
    assert retention["complete_development_prediction_set_retained"] is True
    assert retention["verified_prediction_file_count"] == 21 * 5
    assert len(retention["plots"]) == 21
    assert all(plot["retention_verified"] is True for plot in retention["plots"])

    adapted = tmp_path / "predictions" / rows[0]["safe_plot_id"] / "adapted_npz.bin"
    adapted.write_bytes(adapted.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size changed|SHA-256 changed"):
        summarizer.summarise(
            manifest,
            run_id,
            evaluation_root,
            metadata_root,
            tmp_path / "tampered_summary",
        )


def test_development_summary_accounts_for_documented_failure(
    tmp_path: Path,
) -> None:
    summarizer = load_script(
        "treelearn_development_summary_failure",
        "summarise_for_instance_development.py",
    )
    manifest, evaluation_root, metadata_root, output_root, _rows = (
        build_synthetic_completed_run(tmp_path, failed_task=20)
    )
    result = summarizer.summarise(
        manifest,
        "treelearn_development_21_plot_synthetic",
        evaluation_root,
        metadata_root,
        output_root,
    )

    assert result["status"] == "development_with_documented_failures"
    assert result["completed_plots"] == 20
    assert result["documented_failures"] == 1
    assert result["next_gate"] == (
        "resolve_documented_development_failures_before_any_test_route"
    )
    with (output_root / "failures.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 1
    assert failures[0]["status"] == "documented_inference_failure"
    assert failures[0]["split"] == "dev"
    retention = json.loads(
        (output_root / "retention_manifest.json").read_text(encoding="utf-8")
    )
    assert retention["documented_failures"] == 1
    assert retention["complete_development_prediction_set_retained"] is False
    assert retention["verified_prediction_file_count"] == 20 * 5


def test_development_summary_accounts_for_missing_array_task_output(
    tmp_path: Path,
) -> None:
    summarizer = load_script(
        "treelearn_development_summary_missing_output",
        "summarise_for_instance_development.py",
    )
    manifest, evaluation_root, metadata_root, output_root, rows = (
        build_synthetic_completed_run(tmp_path)
    )
    missing_root = evaluation_root / rows[-1]["safe_plot_id"]
    shutil.rmtree(missing_root)
    partial_name = f".{rows[-1]['safe_plot_id']}.partial.12345"
    (evaluation_root / partial_name).mkdir()

    result = summarizer.summarise(
        manifest,
        "treelearn_development_21_plot_synthetic",
        evaluation_root,
        metadata_root,
        output_root,
    )

    assert result["status"] == "development_with_documented_failures"
    assert result["completed_plots"] == 20
    assert result["documented_failures"] == 1
    with (output_root / "failures.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        failures = list(csv.DictReader(handle))
    assert failures[0]["status"] == "documented_missing_task_output"
    assert failures[0]["inference_status"] == "completed"
    assert partial_name in failures[0]["reason"]
    retention = json.loads(
        (output_root / "retention_manifest.json").read_text(encoding="utf-8")
    )
    assert retention["inference_outputs_retained"] == 21
    assert retention["complete_development_prediction_set_retained"] is True


def test_development_summary_accounts_for_entirely_absent_evaluation_root(
    tmp_path: Path,
) -> None:
    summarizer = load_script(
        "treelearn_development_summary_absent_root",
        "summarise_for_instance_development.py",
    )
    rows = manifest_rows(tmp_path)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, rows)
    evaluation_root = tmp_path / "absent_per_plot"
    metadata_root = tmp_path / "empty_metadata"
    metadata_root.mkdir()
    output_root = tmp_path / "summary"

    result = summarizer.summarise(
        manifest,
        "treelearn_development_wholly_missing_synthetic",
        evaluation_root,
        metadata_root,
        output_root,
    )

    assert evaluation_root.is_dir()
    assert result["completed_plots"] == 0
    assert result["documented_failures"] == 21
    with (output_root / "failures.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        failures = list(csv.DictReader(handle))
    assert len(failures) == 21
    assert {row["status"] for row in failures} == {
        "documented_missing_task_output"
    }
    assert {row["inference_status"] for row in failures} == {"missing"}


def test_development_config_freezes_inventory_and_forbids_test_or_training() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["project"]["status"] == "completed_aligned_pointwise_development"
    assert config["accepted_smoke"]["status"] == "accepted"
    assert config["accepted_smoke"]["benchmark_commit"] == (
        "6fda72efb311873bb2beb0d4360109682710eeaa"
    )
    assert config["accepted_smoke"]["split"] == "dev"
    assert config["accepted_smoke"]["row_count_match"] is True
    assert config["accepted_smoke"]["row_order_preserved"] is True
    assert config["development"]["expected_available_plots"] == 21
    assert config["development"]["expected_site_counts"] == SITE_COUNTS
    assert config["development"]["expected_total_points"] == 101_769_037
    assert config["development"]["expected_reference_trees"] == 807
    assert config["development"]["allow_test_split"] is False
    assert config["development"]["run_training"] is False
    assert config["development"]["tune_checkpoint"] is False
    completed = config["completed_development"]
    assert completed["run_id"] == (
        "treelearn_for-instance_published_pretrained_development_20260712_150030"
    )
    assert completed["completed_plots"] == 21
    assert completed["documented_failures"] == 0
    assert completed["held_out_test_accessed"] is False
    assert math.isclose(completed["metrics"]["mean_plot_f1"], 0.5155705605170436)
    assert math.isclose(completed["metrics"]["micro_f1"], 0.5107604017216643)
    assert completed["retention"]["verified_prediction_file_count"] == 105
    assert config["evaluation"]["postprocessing_selection_permitted"] is False
    assert "development_runs" in config["paths"]["predictions_root"] or (
        "for_instance_development" in config["paths"]["predictions_root"]
    )


def test_completed_development_public_evidence_reconciles() -> None:
    examples = ROOT / "methods/treelearn/examples"
    with (
        examples / "treelearn_completed_development_results_20260712.csv"
    ).open(encoding="utf-8", newline="") as handle:
        overall_rows = list(csv.DictReader(handle))
    with (
        examples / "treelearn_completed_development_site_results_20260712.csv"
    ).open(encoding="utf-8", newline="") as handle:
        sites = list(csv.DictReader(handle))
    provenance = json.loads(
        (
            examples / "treelearn_completed_development_provenance_20260712.json"
        ).read_text(encoding="utf-8")
    )

    assert len(overall_rows) == 1
    overall = overall_rows[0]
    assert overall["dataset_split"] == "dev"
    assert overall["held_out_test_accessed"] == "false"
    assert int(overall["plots"]) == 21
    assert {row["site"] for row in sites} == set(SITE_COUNTS)
    assert all(row["dataset_split"] == "dev" for row in sites)
    assert all(row["held_out_test_accessed"] == "false" for row in sites)

    for overall_field, site_field in (
        ("point_count", "point_count"),
        ("evaluated_point_count", "evaluated_point_count"),
        ("predicted_instances", "predicted_instances"),
        ("reference_instances", "reference_instances"),
        ("true_positives", "true_positives"),
        ("false_positives", "false_positives"),
        ("false_negatives", "false_negatives"),
    ):
        assert int(overall[overall_field]) == sum(
            int(row[site_field]) for row in sites
        )

    tp = int(overall["true_positives"])
    fp = int(overall["false_positives"])
    fn = int(overall["false_negatives"])
    assert math.isclose(float(overall["micro_precision"]), tp / (tp + fp))
    assert math.isclose(float(overall["micro_recall"]), tp / (tp + fn))
    assert math.isclose(float(overall["micro_f1"]), 2 * tp / (2 * tp + fp + fn))
    assert math.isclose(
        float(overall["mean_plot_f1"]),
        sum(float(row["mean_plot_f1"]) * int(row["completed_plots"]) for row in sites)
        / 21,
    )

    assert provenance["run_id"] == overall["run_id"]
    assert provenance["result_status"] == overall["result_status"]
    assert provenance["held_out_test_accessed"] is False
    assert provenance["test_result_exists"] is False
    assert provenance["test_submission_route_exists"] is False
    assert provenance["retention"]["verified_prediction_file_count"] == 105
    assert provenance["retention"]["verified_prediction_size_bytes"] == 9645423654

    public_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "BENCHMARKS.md",
            ROOT / "methods/treelearn/README.md",
            ROOT / "methods/treelearn/docs/development_evaluation.md",
        )
    )
    assert "development_full_route_ready_not_run" not in public_docs
    assert "full development route is ready but" not in public_docs.casefold()


def test_development_slurm_route_is_guarded_combined_and_log_free_to_monitor() -> None:
    slurm = ROOT / "methods/treelearn/slurm"
    run_path = slurm / "run_for_instance_development.sbatch"
    run = run_path.read_text(encoding="utf-8")
    checked_paths = [run_path]
    for optional in (
        "summarise_for_instance_development.sbatch",
        "gate_for_instance_development.sbatch",
        "submit_for_instance_development.sh",
        "monitor_for_instance_development.sh",
    ):
        path = slurm / optional
        if path.is_file():
            checked_paths.append(path)
    for path in checked_paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (path, completed.stderr)

    assert "TREELEARN_FULL_DEV_CONFIRMED" in run
    assert "TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED" in run
    assert "run_for_instance_development_task.py" in run
    assert "evaluate_for_instance_development_task.py" in run
    assert "inference_return_code=" in run
    assert "--split test" not in run
    assert "train" not in run.casefold()

    runner = (
        ROOT / "methods/treelearn/scripts/run_for_instance_one_plot_smoke.py"
    ).read_text(encoding="utf-8")
    zero_prediction_gate = runner.index(
        'prediction_summary["positive_prediction_count"] <= 0'
    )
    assert 'args.evaluation_scope == "development_smoke"' in runner[
        zero_prediction_gate - 120 : zero_prediction_gate
    ]
    assert "development_full" in runner

    submitter = slurm / "submit_for_instance_development.sh"
    if submitter.is_file():
        text = submitter.read_text(encoding="utf-8")
        assert "TREELEARN_FULL_DEV_CONFIRMED" in text
        assert "TREELEARN_DEV_ALIGNMENT_REVIEW_CONFIRMED" in text
        assert "--array" in text
        assert "afterany" in text
        assert "held-out test" in text or "held_out_test" in text
        assert "--split test" not in text

    monitor = slurm / "monitor_for_instance_development.sh"
    if monitor.is_file():
        text = monitor.read_text(encoding="utf-8").casefold()
        assert "tail " not in text
        assert "less " not in text
        assert "more " not in text
        assert "cat logs/" not in text
        assert "squeue" in text
        assert "sacct" in text
        assert "time_left" in text or "timeleft" in text or "%.9l" in text
