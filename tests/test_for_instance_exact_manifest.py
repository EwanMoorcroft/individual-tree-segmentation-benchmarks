from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "methods/tls2trees/scripts/data/prepare_for_instance_manifest.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import (  # noqa: E402
    EXPECTED_PATHS,
    EXPECTED_SITE_COUNTS,
    STAGE0_SELECTION_RULE,
    build_exact_split_manifest,
    load_and_verify_manifest_plot,
    read_split_metadata,
    validate_manifest_payload,
)


def write_metadata(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "folder", "split"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("test", "dev"):
        for relative in reversed(EXPECTED_PATHS[split]):
            rows.append(
                {
                    "path": relative,
                    "folder": relative.split("/", 1)[0],
                    "split": split,
                }
            )
    return rows


def make_split_files(root: Path, split: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    per_site_positions: dict[str, int] = {}
    for relative in EXPECTED_PATHS[split]:
        site = relative.split("/", 1)[0]
        position = per_site_positions.get(site, 0) + 1
        per_site_positions[site] = position
        counts[relative] = position * 100
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {split} {relative}".encode())
    return counts


def build_synthetic(
    tmp_path: Path,
    *,
    split: str = "development",
    allow_held_out_test: bool = False,
) -> tuple[dict[str, Any], Path, Path, dict[str, int]]:
    dataset_root = tmp_path / "FORinstance_dataset"
    dataset_root.mkdir()
    split_code = "dev" if split == "development" else "test"
    counts = make_split_files(dataset_root, split_code)
    metadata = dataset_root / "data_split_metadata.csv"
    metadata_sha256 = write_metadata(metadata, metadata_rows())
    payload = build_exact_split_manifest(
        dataset_root,
        metadata,
        split=split,
        allow_held_out_test=allow_held_out_test,
        expected_metadata_sha256=metadata_sha256,
        inventory_reader=lambda path: (
            counts[path.relative_to(dataset_root).as_posix()],
            counts[path.relative_to(dataset_root).as_posix()] // 100,
        ),
    )
    return payload, dataset_root, metadata, counts


def load_cli():
    spec = importlib.util.spec_from_file_location("tls2trees_manifest_cli", CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_development_manifest_is_exact_and_never_touches_test_files(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "FORinstance_dataset"
    dataset_root.mkdir()
    counts = make_split_files(dataset_root, "dev")
    metadata = dataset_root / "data_split_metadata.csv"
    metadata_sha256 = write_metadata(metadata, metadata_rows())
    touched: list[str] = []

    def development_only_exists(path: Path) -> bool:
        relative = path.relative_to(dataset_root).as_posix()
        assert relative not in EXPECTED_PATHS["test"]
        touched.append(relative)
        return path.is_file()

    payload = build_exact_split_manifest(
        dataset_root,
        metadata,
        split="development",
        expected_metadata_sha256=metadata_sha256,
        file_exists=development_only_exists,
        inventory_reader=lambda path: (
            counts[path.relative_to(dataset_root).as_posix()],
            counts[path.relative_to(dataset_root).as_posix()] // 100,
        ),
    )

    assert payload["dataset_split"] == "development"
    assert payload["held_out_test_accessed"] is False
    assert payload["held_out_metrics_computed"] is False
    assert payload["tuning_eligible"] is True
    assert touched == list(EXPECTED_PATHS["dev"])
    assert [row["relative_path"] for row in payload["plots"]] == list(
        EXPECTED_PATHS["dev"]
    )
    assert all(row["split"] == "development" for row in payload["plots"])
    assert all(row["reference_tree_count"] > 0 for row in payload["plots"])
    assert all(len(row["input_sha256"]) == 64 for row in payload["plots"])


def test_stage0_is_one_per_site_and_uses_frozen_median_rule(tmp_path: Path) -> None:
    payload, _, _, _ = build_synthetic(tmp_path)

    assert payload["stage0_selection_rule"] == STAGE0_SELECTION_RULE
    selected = payload["stage0_selection"]
    assert [row["collection"] for row in selected] == list(
        EXPECTED_SITE_COUNTS["dev"]
    )
    assert [row["relative_path"] for row in selected] == [
        "CULS/plot_1_annotated.las",
        "NIBIO/plot_21_annotated.las",
        "RMIT/train.las",
        "SCION/plot_39_annotated.las",
        "TUWIEN/train.las",
    ]
    assert sum(row["stage0_selected"] for row in payload["plots"]) == 5


def test_metadata_rejects_hash_duplicates_missing_contract_and_split_leakage(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "data_split_metadata.csv"
    metadata_sha256 = write_metadata(metadata, metadata_rows())
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_split_metadata(metadata)

    duplicates = metadata_rows() + [metadata_rows()[0]]
    duplicate_sha256 = write_metadata(metadata, duplicates)
    with pytest.raises(ValueError, match="Duplicate split metadata path"):
        read_split_metadata(metadata, expected_sha256=duplicate_sha256)

    missing = [
        row for row in metadata_rows() if row["path"] != EXPECTED_PATHS["dev"][0]
    ]
    missing_sha256 = write_metadata(metadata, missing)
    with pytest.raises(ValueError, match="absent from split metadata"):
        read_split_metadata(metadata, expected_sha256=missing_sha256)

    leaking = metadata_rows()
    for row in leaking:
        if row["path"] == EXPECTED_PATHS["dev"][0]:
            row["split"] = "test"
            break
    leaking_sha256 = write_metadata(metadata, leaking)
    with pytest.raises(ValueError, match="Split leakage"):
        read_split_metadata(metadata, expected_sha256=leaking_sha256)


def test_builder_rejects_missing_requested_file(tmp_path: Path) -> None:
    dataset_root = tmp_path / "FORinstance_dataset"
    dataset_root.mkdir()
    counts = make_split_files(dataset_root, "dev")
    missing = dataset_root / EXPECTED_PATHS["dev"][3]
    missing.unlink()
    metadata = dataset_root / "data_split_metadata.csv"
    metadata_sha256 = write_metadata(metadata, metadata_rows())

    with pytest.raises(FileNotFoundError, match=EXPECTED_PATHS["dev"][3]):
        build_exact_split_manifest(
            dataset_root,
            metadata,
            expected_metadata_sha256=metadata_sha256,
            inventory_reader=lambda path: (
                counts[path.relative_to(dataset_root).as_posix()],
                counts[path.relative_to(dataset_root).as_posix()] // 100,
            ),
        )


def test_held_out_manifest_requires_explicit_opt_in_and_contains_no_stage0(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "FORinstance_dataset"
    dataset_root.mkdir()
    counts = make_split_files(dataset_root, "test")
    metadata = dataset_root / "data_split_metadata.csv"
    metadata_sha256 = write_metadata(metadata, metadata_rows())

    with pytest.raises(PermissionError, match="allow_held_out_test"):
        build_exact_split_manifest(
            dataset_root,
            metadata,
            split="test",
            expected_metadata_sha256=metadata_sha256,
        )

    payload = build_exact_split_manifest(
        dataset_root,
        metadata,
        split="test",
        allow_held_out_test=True,
        expected_metadata_sha256=metadata_sha256,
        inventory_reader=lambda path: (
            counts[path.relative_to(dataset_root).as_posix()],
            counts[path.relative_to(dataset_root).as_posix()] // 100,
        ),
    )
    assert payload["held_out_test_accessed"] is True
    assert payload["held_out_metrics_computed"] is False
    assert payload["tuning_eligible"] is False
    assert payload["stage0_selection"] == []
    assert not any(row["stage0_selected"] for row in payload["plots"])


def test_validator_rejects_manifest_tampering(tmp_path: Path) -> None:
    payload, _, metadata, _ = build_synthetic(tmp_path)
    metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()

    duplicate = copy.deepcopy(payload)
    duplicate["plots"][1]["relative_path"] = duplicate["plots"][0]["relative_path"]
    with pytest.raises(ValueError, match="paths or order"):
        validate_manifest_payload(
            duplicate,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )

    leaked = copy.deepcopy(payload)
    leaked["plots"][0]["split"] = "test"
    with pytest.raises(ValueError, match="Split leakage"):
        validate_manifest_payload(
            leaked,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )

    changed_selection = copy.deepcopy(payload)
    changed_selection["plots"][0]["stage0_selected"] = False
    with pytest.raises(ValueError, match="Stage 0 flags"):
        validate_manifest_payload(
            changed_selection,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )


def test_selected_plot_verification_rechecks_metadata_path_and_input_hash(
    tmp_path: Path,
) -> None:
    payload, dataset_root, metadata, _ = build_synthetic(tmp_path)
    metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    _, verified = load_and_verify_manifest_plot(
        manifest,
        task_index=0,
        expected_split="development",
        expected_metadata_sha256=metadata_sha256,
    )
    assert verified["relative_path"] == EXPECTED_PATHS["dev"][0]
    assert verified["observed_input_sha256"] == verified["input_sha256"]

    redirected = copy.deepcopy(payload)
    held_out = dataset_root / EXPECTED_PATHS["test"][0]
    held_out.parent.mkdir(parents=True, exist_ok=True)
    held_out.write_bytes(b"held-out sentinel")
    redirected["plots"][0]["input_las"] = str(held_out)
    manifest.write_text(json.dumps(redirected), encoding="utf-8")
    with pytest.raises(ValueError, match="input path mismatch"):
        load_and_verify_manifest_plot(
            manifest,
            task_index=0,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )

    manifest.write_text(json.dumps(payload), encoding="utf-8")
    selected = Path(payload["plots"][0]["input_las"])
    original_input = selected.read_bytes()
    selected.write_bytes(original_input + b"changed")
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        load_and_verify_manifest_plot(
            manifest,
            task_index=0,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )
    selected.write_bytes(original_input)

    metadata.write_bytes(metadata.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metadata SHA-256 mismatch"):
        load_and_verify_manifest_plot(
            manifest,
            task_index=0,
            expected_split="development",
            expected_metadata_sha256=metadata_sha256,
        )


def test_stage0_rows_retain_original_task_indexes_and_resolve_fields(
    tmp_path: Path,
) -> None:
    payload, _, _, _ = build_synthetic(tmp_path)
    cli = load_cli()

    rows = cli.stage0_rows(payload)
    assert [row["stage0_index"] for row in rows] == list(range(5))
    assert [row["task_index"] for row in rows] == [0, 8, 16, 18, 20]
    assert all(row["split"] == "development" for row in rows)
    assert cli.resolve_task(payload, 18, "safe_plot_id") == "SCION_plot_39_annotated"
    assert cli.resolve_task(payload, 18, "reference_tree_count") == 2
    assert cli.resolve_stage0_task(payload, 1, "task_index") == 8
    assert (
        cli.resolve_stage0_task(payload, 1, "safe_plot_id")
        == "NIBIO_plot_21_annotated"
    )
    with pytest.raises(ValueError, match="no unique task index"):
        cli.resolve_task(payload, 99, "safe_plot_id")
    with pytest.raises(ValueError, match="no unique index"):
        cli.resolve_stage0_task(payload, 99, "safe_plot_id")


def test_tls2trees_manifest_cli_help_is_available() -> None:
    for command in (
        [],
        ["build", "--help"],
        ["validate", "--help"],
        ["select-stage0", "--help"],
        ["resolve", "--help"],
        ["resolve-stage0", "--help"],
    ):
        completed = subprocess.run(
            [sys.executable, str(CLI), *command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        expected_returncode = 2 if not command else 0
        assert completed.returncode == expected_returncode
        stream = completed.stderr if not command else completed.stdout
        assert "usage:" in stream
