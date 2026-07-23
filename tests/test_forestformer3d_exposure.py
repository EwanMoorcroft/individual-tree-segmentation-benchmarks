from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from methods.forestformer3d.scripts.data import audit_checkpoint_exposure as audit


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forestformer3d"
INVENTORY = METHOD / "examples/zenodo_v2_archive_inventory_20260723.csv"
AUDIT_CSV = METHOD / "examples/checkpoint_exposure_audit_20260723.csv"
SUMMARY = METHOD / "examples/checkpoint_exposure_summary_20260723.json"


def test_exact_member_normalisation_is_reviewable_and_not_fuzzy() -> None:
    assert audit.expected_member_name(
        "NIBIO/plot_17_annotated.las", "test"
    ) == "NIBIO_NIBIO_plot_17_annotated_test.ply"
    assert audit.expected_member_name(
        "RMIT/train.las", "train"
    ) == "RMIT_RMIT_train_train.ply"
    with pytest.raises(ValueError, match="Unsupported official role"):
        audit.expected_member_name("RMIT/train.las", "development")
    with pytest.raises(ValueError, match="Non-canonical"):
        audit.expected_member_name("../RMIT/train.las", "train")


def test_committed_inventory_produces_exact_eligible_32_row_audit() -> None:
    inventory, digest = audit.read_inventory(INVENTORY)
    rows = audit.build_exposure_rows(inventory)
    summary = audit.summarise(
        rows,
        inventory_sha256=digest,
        inventory_path=INVENTORY.relative_to(ROOT),
        manual_review_status="reviewed_20260723",
    )

    assert len(rows) == 32
    assert sum(row.original_split == "dev" for row in rows) == 21
    assert sum(row.original_split == "test" for row in rows) == 11
    assert sum(row.checkpoint_train_val_exposed for row in rows) == 21
    assert not any(
        row.checkpoint_train_val_exposed
        for row in rows
        if row.original_split == "test"
    )
    assert summary["official_role_counts"] == {"test": 11, "train": 13, "val": 8}
    assert summary["held_out_train_val_match_count"] == 0
    assert summary["unmatched_or_multiple"] == []
    assert summary["decision"] == "eligible_exposure_gate_passed"


def test_committed_outputs_reconcile_with_fresh_generation(tmp_path: Path) -> None:
    output_csv = tmp_path / "audit.csv"
    output_summary = tmp_path / "audit.json"
    assert audit.main(
        [
            "--inventory",
            str(INVENTORY),
            "--output-csv",
            str(output_csv),
            "--output-summary",
            str(output_summary),
            "--manual-review-status",
            "reviewed_20260723",
        ]
    ) == 0
    assert output_csv.read_bytes() == AUDIT_CSV.read_bytes()
    expected_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    actual_summary = json.loads(output_summary.read_text(encoding="utf-8"))
    actual_summary["inventory_snapshot"] = expected_summary["inventory_snapshot"]
    assert actual_summary == expected_summary

    with pytest.raises(FileExistsError, match="must not overwrite"):
        audit.main(
            [
                "--inventory",
                str(INVENTORY),
                "--output-csv",
                str(output_csv),
                "--output-summary",
                str(output_summary),
            ]
        )


def test_audit_fails_closed_on_duplicate_or_held_out_training_match(
    tmp_path: Path,
) -> None:
    rows = list(csv.DictReader(INVENTORY.read_text(encoding="utf-8").splitlines()))
    held_out = next(
        row for row in rows if row["member_path"] == "CULS_CULS_plot_2_annotated_test.ply"
    )
    duplicate_training = dict(held_out)
    duplicate_training.update(
        {
            "archive": "train_val_data.zip",
            "member_path": "CULS_CULS_plot_2_annotated_train.ply",
            "official_role": "train",
            "source_url": (
                "https://zenodo.org/records/16742708/preview/"
                "train_val_data.zip?include_deleted=0"
            ),
        }
    )
    rows.append(duplicate_training)
    inventory_path = tmp_path / "inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    inventory, _ = audit.read_inventory(inventory_path)
    with pytest.raises(ValueError, match="exactly one official match"):
        audit.build_exposure_rows(inventory)


def test_public_summary_records_manual_review_and_snapshot_hash() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["manual_review_status"] == "reviewed_20260723"
    assert summary["inventory_snapshot_sha256"] == audit.sha256_file(INVENTORY)
    assert summary["decision"] == "eligible_exposure_gate_passed"
