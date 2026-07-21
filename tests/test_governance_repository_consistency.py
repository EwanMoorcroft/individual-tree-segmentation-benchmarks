from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.governance import (
    BUDGET_REQUIRED_FIELDS,
    EXPOSURE_REQUIRED_FIELDS,
    GOVERNANCE_REQUIRED_FIELDS,
    PROVENANCE_REQUIRED_FIELDS,
    validate_budget_row,
    validate_exposure_row,
    validate_governance_row,
    parse_future_run_id,
    validate_provenance_row,
)


OUTPUT_ROOT = ROOT / "outputs/for_instance_benchmark_metrics"
HEADLINE_PATH = OUTPUT_ROOT / "for_instance_method_benchmark_results.csv"
GOVERNANCE_PATH = OUTPUT_ROOT / "benchmark_result_registry.csv"
DIAGNOSTICS_PATH = OUTPUT_ROOT / "for_instance_method_development_diagnostics.csv"
EXPOSURE_PATH = OUTPUT_ROOT / "test_exposure_ledger.csv"
BUDGET_PATH = OUTPUT_ROOT / "method_development_budget.csv"
PROVENANCE_PATH = OUTPUT_ROOT / "method_environment_provenance.csv"

DIAGNOSTIC_GOVERNANCE_RECORDS = {
    (
        "treelearn",
        "treelearn_for-instance_published_pretrained_development_20260712_150030",
        "treelearn_full_development_diagnostic",
    ): "treelearn-overlap-development",
    (
        "treelearn",
        "treelearn_for-instance_published_pretrained_development_20260712_150030",
        "treelearn_matched_internal_validation",
    ): "treelearn-overlap-matched-baseline",
    (
        "treelearn",
        "treelearn_for-instance_fine_tuned_on_dev_20260712_164057_epoch_70_validation_20260712_203249",
        "treelearn_matched_internal_validation",
    ): "treelearn-rejected-finetune-epoch70",
    (
        "tls2trees",
        "tls2trees_for-instance_development_tuned_held_out_test_20260719_110219",
        "tls2trees_leaf_off_class3_ignore_diagnostic",
    ): "tls2trees-tuned-leafoff",
    (
        "tls2trees",
        "tls2trees_for-instance_published_default_held_out_test_20260721_122448",
        "tls2trees_leaf_off_class3_ignore_diagnostic",
    ): "tls2trees-published-leafoff",
    (
        "treex",
        "treex_for_instance_exact_path_subset",
        "treex_reference_labelled_mask_diagnostic",
    ): "treex-reference-mask-diagnostic",
}

Validator = Callable[[dict[str, str]], dict[str, Any]]
SCHEMA_TABLES: tuple[tuple[Path, Validator, tuple[str, ...]], ...] = (
    (GOVERNANCE_PATH, validate_governance_row, GOVERNANCE_REQUIRED_FIELDS),
    (EXPOSURE_PATH, validate_exposure_row, EXPOSURE_REQUIRED_FIELDS),
    (BUDGET_PATH, validate_budget_row, BUDGET_REQUIRED_FIELDS),
    (PROVENANCE_PATH, validate_provenance_row, PROVENANCE_REQUIRED_FIELDS),
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    assert headers, f"Canonical CSV has no header: {path.relative_to(ROOT)}"
    assert rows, f"Canonical CSV has no data rows: {path.relative_to(ROOT)}"
    return headers, rows


def _validate_rows(path: Path, validator: Validator) -> list[dict[str, str]]:
    _, rows = _read_csv(path)
    for row_number, row in enumerate(rows, start=2):
        try:
            validated = validator(row)
        except (TypeError, ValueError) as exc:
            pytest.fail(
                f"{path.relative_to(ROOT)}:{row_number} violates governance schema: "
                f"{exc}"
            )
        assert set(row) <= set(validated), (
            f"{path.relative_to(ROOT)}:{row_number} lost canonical or legacy fields"
        )
    return rows


@pytest.mark.parametrize(
    ("path", "validator", "required_fields"),
    SCHEMA_TABLES,
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_canonical_governance_csvs_validate_through_shared_schema(
    path: Path,
    validator: Validator,
    required_fields: tuple[str, ...],
) -> None:
    headers, _ = _read_csv(path)

    assert set(required_fields) <= set(headers)
    _validate_rows(path, validator)


def test_every_canonical_governance_evidence_path_exists_in_repository() -> None:
    for path, validator, _ in SCHEMA_TABLES:
        rows = _validate_rows(path, validator)
        assert "evidence_path" in rows[0], path.relative_to(ROOT)
        for row_number, row in enumerate(rows, start=2):
            value = row["evidence_path"]
            assert value, f"{path.relative_to(ROOT)}:{row_number} has no evidence path"
            evidence = Path(value)
            assert not evidence.is_absolute(), (
                f"{path.relative_to(ROOT)}:{row_number} evidence must be repository-relative"
            )
            resolved = (ROOT / evidence).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), (
                f"{path.relative_to(ROOT)}:{row_number} evidence escapes repository"
            )
            assert resolved.is_file(), (
                f"{path.relative_to(ROOT)}:{row_number} missing evidence: {value}"
            )


def _headline_identity(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["dataset_slug"],
        row["method_slug"],
        row["variant"],
        row["run_id"],
    )


def test_accepted_headlines_join_once_to_governance_and_rank_by_scoring_mask() -> None:
    _, headlines = _read_csv(HEADLINE_PATH)
    governance = _validate_rows(GOVERNANCE_PATH, validate_governance_row)
    assert len(headlines) == 7

    joined: list[tuple[dict[str, str], dict[str, str]]] = []
    for headline in headlines:
        identity = _headline_identity(headline)
        matches = [
            row
            for row in governance
            if _headline_identity(row) == identity
            and row["completion_state"] == "completed"
            and row["result_status"] in {"primary", "baseline"}
        ]
        assert len(matches) == 1, (
            f"Accepted headline {identity} must join to exactly one completed "
            f"primary/baseline governance record; found {len(matches)}"
        )
        joined.append((headline, matches[0]))

    eligible = [item for item in joined if item[1]["ranking_eligible"] == "true"]
    excluded = [item for item in joined if item[1]["ranking_eligible"] == "false"]
    assert len(eligible) == 5
    assert len(excluded) == 2
    assert all(
        governance_row["reference_scoring_mask"] == "classes_4_5_6"
        and governance_row["exclusion_reason"] == ""
        for _, governance_row in eligible
    )
    assert {
        (governance_row["method_slug"], governance_row["variant"])
        for _, governance_row in excluded
    } == {
        ("tls2trees", "development_tuned"),
        ("tls2trees", "published_default"),
    }
    assert all(
        governance_row["reference_scoring_mask"] == "class_3_ignored"
        and governance_row["exclusion_reason"]
        == "different_reference_scoring_mask"
        for _, governance_row in excluded
    )


def test_every_recorded_canonical_alias_uses_the_future_run_id_contract() -> None:
    governance = _validate_rows(GOVERNANCE_PATH, validate_governance_row)
    aliases = [
        row["canonical_run_id_alias"]
        for row in governance
        if row["canonical_run_id_alias"] not in {"unknown", "not_recorded"}
    ]

    assert aliases
    assert all(str(parse_future_run_id(alias)) == alias for alias in aliases)


def test_every_published_development_diagnostic_has_explicit_governance() -> None:
    _, diagnostics = _read_csv(DIAGNOSTICS_PATH)
    governance = _validate_rows(GOVERNANCE_PATH, validate_governance_row)
    governance_by_id = {row["result_record_id"]: row for row in governance}
    identities = {
        (row["method_slug"], row["run_id"], row["comparable_group"])
        for row in diagnostics
    }

    assert identities == set(DIAGNOSTIC_GOVERNANCE_RECORDS)
    for identity, record_id in DIAGNOSTIC_GOVERNANCE_RECORDS.items():
        record = governance_by_id[record_id]
        assert (record["method_slug"], record["run_id"]) == identity[:2]
        assert record["result_status"] in {"diagnostic", "rejected"}
        assert record["completion_state"] == "completed"
        assert record["ranking_eligible"] == "false"


def test_every_accepted_headline_has_budget_environment_and_exposure_records() -> None:
    _, headlines = _read_csv(HEADLINE_PATH)
    budgets = _validate_rows(BUDGET_PATH, validate_budget_row)
    provenance = _validate_rows(PROVENANCE_PATH, validate_provenance_row)
    exposures = _validate_rows(EXPOSURE_PATH, validate_exposure_row)

    for headline in headlines:
        identity = (
            headline["method_slug"],
            headline["variant"],
            headline["run_id"],
        )
        budget_matches = [
            row
            for row in budgets
            if (row["method"], row["variant"], row["run_id"]) == identity
        ]
        provenance_matches = [
            row
            for row in provenance
            if (row["method"], row["variant"], row["run_id"]) == identity
        ]
        exposure_matches = [
            row
            for row in exposures
            if (row["method"], row["run_id"])
            == (headline["method_slug"], headline["run_id"])
        ]

        assert len(budget_matches) == 1, identity
        assert len(provenance_matches) == 1, identity
        assert exposure_matches, identity
