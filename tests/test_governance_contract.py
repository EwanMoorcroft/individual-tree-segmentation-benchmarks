from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.governance import (
    COMPLETION_STATE_VALUES,
    DATASET_EXPOSURE_VALUES,
    LEAF_ATTACHMENT_VALUES,
    LEARNING_REGIME_VALUES,
    PREDICTION_MATERIAL_VALUES,
    REFERENCE_SCORING_MASK_VALUES,
    RESULT_STATUS_VALUES,
    build_future_run_id,
    parse_future_run_id,
    validate_budget_row,
    validate_exposure_row,
    validate_governance_row,
    validate_historical_run_id,
    validate_provenance_row,
)


VALID_FUTURE_RUN_ID = (
    "segmentanytree__for-instance__fine-tuned-dev__best-validation__test__"
    "20260711T002931"
)


def _governance_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "result_status": "primary",
        "completion_state": "completed",
        "ranking_eligible": True,
        "exclusion_reason": "",
        "learning_regime": "supervised",
        "dataset_exposure": "published_checkpoint",
        "prediction_material": "full_tree_material",
        "reference_scoring_mask": "classes_4_5_6",
        "leaf_attachment": "enabled",
        "status_description": "Primary harmonised held-out test result",
        "legacy_result_role": "primary benchmark",
        "legacy_result_status": "complete",
        "training_mode": "external_training_only",
    }
    row.update(overrides)
    return row


def _exposure_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2026-07-11",
        "method": "segmentanytree",
        "run_id": VALID_FUTURE_RUN_ID,
        "test_job_executed": "true",
        "metrics_viewed": "true",
        "predictions_visualised": "false",
        "configuration_changed_afterwards": "false",
        "decision_or_change": "Frozen configuration retained",
        "notes": "Aggregate metrics were recorded",
    }
    row.update(overrides)
    return row


def _budget_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "method": "segmentanytree",
        "variant": "published_default",
        "configurations_attempted": 1,
        "validation_evaluations": "3",
        "checkpoints_evaluated": "unknown",
        "training_epochs": 50,
        "optimizer_steps": "not_applicable",
        "gpu_hours": "12.5",
        "cpu_hours": 0,
        "manual_validation_inspection": "true",
        "hyperparameter_source": "Published configuration",
        "notes": "Only evidenced effort is recorded",
    }
    row.update(overrides)
    return row


def _provenance_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "upstream_repository": "https://github.com/example/segmenter",
        "upstream_commit": "a" * 40,
        "upstream_dirty_state": "clean",
        "container_uri": "docker://ghcr.io/example/segmenter:1.0",
        "container_digest": "sha256:" + "b" * 64,
        "python_version": "3.10.13",
        "cuda_version": "11.8",
        "pytorch_version": "2.0.1",
        "minkowski_engine_version": "0.5.4",
        "checkpoint_source": "https://example.org/checkpoints/model.pt",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_training_datasets": ["external-dataset-a"],
        "method": "segmentanytree",
        "run_id": VALID_FUTURE_RUN_ID,
    }
    row.update(overrides)
    return row


def test_controlled_taxonomies_are_exact() -> None:
    assert RESULT_STATUS_VALUES == {
        "primary",
        "baseline",
        "diagnostic",
        "historical",
        "rejected",
        "operational_only",
        "candidate",
        "failed",
    }
    assert COMPLETION_STATE_VALUES == {"completed", "partial", "pending"}
    assert LEARNING_REGIME_VALUES == {
        "supervised",
        "self_supervised",
        "unsupervised",
        "deterministic",
        "rule_based",
    }
    assert DATASET_EXPOSURE_VALUES == {
        "published_checkpoint",
        "external_only",
        "development_tuned",
        "development_trained",
        "none",
    }
    assert PREDICTION_MATERIAL_VALUES == {
        "woody_only",
        "woody_plus_leaf",
        "full_tree_material",
    }
    assert REFERENCE_SCORING_MASK_VALUES == {
        "classes_4_5_6",
        "class_3_ignored",
        "custom",
    }
    assert LEAF_ATTACHMENT_VALUES == {"enabled", "disabled", "not_applicable"}


def test_future_run_id_parses_exact_six_component_contract() -> None:
    parsed = parse_future_run_id(VALID_FUTURE_RUN_ID)

    assert parsed.method == "segmentanytree"
    assert parsed.dataset == "for-instance"
    assert parsed.training_mode == "fine-tuned-dev"
    assert parsed.selection_mode == "best-validation"
    assert parsed.split == "test"
    assert parsed.timestamp == "20260711T002931"
    assert str(parsed) == VALID_FUTURE_RUN_ID


def test_future_run_id_builder_normalizes_aware_timestamp_to_utc() -> None:
    timestamp = datetime(
        2026, 7, 11, 1, 29, 31, tzinfo=timezone(timedelta(hours=1))
    )

    assert (
        build_future_run_id(
            method="segmentanytree",
            dataset="for-instance",
            training_mode="fine-tuned-dev",
            selection_mode="best-validation",
            split="test",
            timestamp=timestamp,
        )
        == VALID_FUTURE_RUN_ID
    )


def test_future_run_id_builder_requires_datetime_and_timezone() -> None:
    with pytest.raises(TypeError, match="datetime"):
        build_future_run_id(
            method="segmentanytree",
            dataset="for-instance",
            training_mode="fine-tuned-dev",
            selection_mode="best-validation",
            split="test",
            timestamp="20260711T002931",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        build_future_run_id(
            method="segmentanytree",
            dataset="for-instance",
            training_mode="fine-tuned-dev",
            selection_mode="best-validation",
            split="test",
            timestamp=datetime(2026, 7, 11, 0, 29, 31),
        )


@pytest.mark.parametrize(
    "run_id",
    [
        "sat__for-instance__fine-tuned-dev__best-validation__test__20260711T002931",
        "SegmentAnyTree__for-instance__fine-tuned-dev__best-validation__test__20260711T002931",
        "segment_any_tree__for-instance__fine-tuned-dev__best-validation__test__20260711T002931",
        "segmentanytree__for--instance__fine-tuned-dev__best-validation__test__20260711T002931",
        "segmentanytree__for-instance__fine-tuned-dev__test__20260711T002931",
        VALID_FUTURE_RUN_ID + "__final",
        "segmentanytree__for-instance__long__best-validation__test__20260711T002931",
        "segmentanytree__for-instance__fine-tuned-long__best-validation__test__20260711T002931",
        "segmentanytree__for-instance__fine-tuned-dev__best__test__20260711T002931",
        "segmentanytree__for-instance__fine-tuned-dev__checkpoint-final__test__20260711T002931",
        "segmentanytree__for-instance__fine-tuned-dev__best-validation__test__20260711_002931",
        "segmentanytree__for-instance__fine-tuned-dev__best-validation__test__20260230T002931",
        "segmentanytree__for-instance__fine-tuned-dev__best-validation__test__20260711T242931",
        "segmentanytree__for-instance__fine-tuned-dev__best-validation__test __20260711T002931",
    ],
)
def test_future_run_id_rejects_legacy_or_malformed_forms(run_id: str) -> None:
    with pytest.raises(ValueError):
        parse_future_run_id(run_id)


def test_future_run_id_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        parse_future_run_id(123)


@pytest.mark.parametrize(
    "run_id",
    [
        "sat_for_quicktune_to49_20260706_140730",
        "segmentanytree_for-instance_test_20260711_002931",
        "tls2trees_for_instance_published_default__semantic_cache",
        "treex_for_instance_exact_path_subset",
        VALID_FUTURE_RUN_ID,
    ],
)
def test_historical_run_ids_remain_supported_without_renaming(run_id: str) -> None:
    assert validate_historical_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../run", "folder/run", "run name", "run:latest", "a" * 256],
)
def test_historical_run_id_rejects_unsafe_path_components(run_id: str) -> None:
    with pytest.raises(ValueError):
        validate_historical_run_id(run_id)


def test_historical_run_id_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="string"):
        validate_historical_run_id(None)


@pytest.mark.parametrize("status", ["primary", "baseline"])
def test_only_completed_primary_or_baseline_rows_can_rank(status: str) -> None:
    row = _governance_row(result_status=status, ranking_eligible="true")

    validated = validate_governance_row(row)

    assert validated["ranking_eligible"] is True
    assert validated["exclusion_reason"] == ""
    assert validated["training_mode"] == "external_training_only"
    assert row["ranking_eligible"] == "true"


def test_ineligible_governance_row_requires_specific_reason_and_retains_legacy() -> None:
    validated = validate_governance_row(
        _governance_row(
            result_status="diagnostic",
            ranking_eligible=False,
            exclusion_reason="different_reference_scoring_mask",
            learning_regime="unsupervised",
            dataset_exposure="development_tuned",
            legacy_result_role="diagnostic",
            legacy_result_status="completed diagnostic",
            method_specific_note="retained",
        )
    )

    assert validated["ranking_eligible"] is False
    assert validated["method_specific_note"] == "retained"
    assert validated["legacy_result_status"] == "completed diagnostic"


def test_pending_candidate_accepts_explicit_unknown_metadata_without_legacy_fields() -> None:
    row = _governance_row(
        result_status="candidate",
        completion_state="pending",
        ranking_eligible=False,
        exclusion_reason="not_implemented",
        learning_regime="unknown",
        dataset_exposure="unknown",
        prediction_material="unknown",
        reference_scoring_mask="custom",
        leaf_attachment="unknown",
        status_description="Candidate TLS accuracy benchmark",
    )
    del row["legacy_result_role"]
    del row["legacy_result_status"]

    validated = validate_governance_row(row)

    assert validated["learning_regime"] == "unknown"
    assert validated["dataset_exposure"] == "unknown"
    assert validated["prediction_material"] == "unknown"
    assert validated["leaf_attachment"] == "unknown"
    assert "legacy_result_role" not in validated


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("result_status", "leaderboard"),
        ("completion_state", "complete"),
        ("learning_regime", "external_training_only"),
        ("dataset_exposure", "test_tuned"),
        ("prediction_material", "leaf_on"),
        ("reference_scoring_mask", "all_tree_classes"),
        ("leaf_attachment", "off"),
    ],
)
def test_governance_rejects_values_outside_controlled_taxonomy(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_governance_row(_governance_row(**{field: value}))


@pytest.mark.parametrize("status", sorted(RESULT_STATUS_VALUES - {"primary", "baseline"}))
def test_nonranking_status_cannot_be_marked_eligible(status: str) -> None:
    with pytest.raises(ValueError, match="Only completed"):
        validate_governance_row(_governance_row(result_status=status))


@pytest.mark.parametrize("completion", ["partial", "pending"])
def test_incomplete_row_cannot_be_marked_eligible(completion: str) -> None:
    with pytest.raises(ValueError, match="Only completed"):
        validate_governance_row(_governance_row(completion_state=completion))


@pytest.mark.parametrize("reason", [None, "", " ", "unknown", "not_applicable"])
def test_ineligible_row_rejects_missing_or_nonspecific_reason(reason: object) -> None:
    with pytest.raises(ValueError, match="exclusion_reason"):
        validate_governance_row(
            _governance_row(ranking_eligible=False, exclusion_reason=reason)
        )


@pytest.mark.parametrize(
    "reason",
    ["Different reference scoring mask", "UPPER_CASE", "path/reason"],
)
def test_ineligible_row_requires_machine_readable_reason_slug(reason: str) -> None:
    with pytest.raises(ValueError, match="lower-case"):
        validate_governance_row(
            _governance_row(ranking_eligible=False, exclusion_reason=reason)
        )


@pytest.mark.parametrize("reason", ["Should not be present", ["not scalar"]])
def test_eligible_row_rejects_any_exclusion_reason(reason: object) -> None:
    with pytest.raises(ValueError, match="must not contain"):
        validate_governance_row(_governance_row(exclusion_reason=reason))


@pytest.mark.parametrize("value", [1, 0, "True", "FALSE", "yes", None])
def test_ranking_eligible_is_a_strict_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="true or false"):
        validate_governance_row(_governance_row(ranking_eligible=value))


def test_governance_requires_status_description_and_mapping_input() -> None:
    row = _governance_row()
    del row["status_description"]
    with pytest.raises(ValueError, match="status_description"):
        validate_governance_row(row)
    with pytest.raises(TypeError, match="mapping"):
        validate_governance_row([])  # type: ignore[arg-type]


def test_exposure_ledger_accepts_explicit_states_and_normalizes_booleans() -> None:
    source = _exposure_row(
        test_job_executed=True,
        metrics_viewed=False,
        predictions_visualised="unknown",
        configuration_changed_afterwards="not_applicable",
        evidence_path="docs/evidence/test-exposure.md",
    )

    validated = validate_exposure_row(source)

    assert validated["test_job_executed"] == "true"
    assert validated["metrics_viewed"] == "false"
    assert validated["predictions_visualised"] == "unknown"
    assert validated["configuration_changed_afterwards"] == "not_applicable"
    assert source["test_job_executed"] is True


@pytest.mark.parametrize("date", ["2026/07/11", "2026-02-30", "not_recorded", ""])
def test_exposure_ledger_rejects_invalid_or_implicit_dates(date: str) -> None:
    with pytest.raises(ValueError, match="date"):
        validate_exposure_row(_exposure_row(date=date))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_job_executed", "yes"),
        ("metrics_viewed", "TRUE"),
        ("predictions_visualised", 1),
        ("configuration_changed_afterwards", None),
    ],
)
def test_exposure_ledger_rejects_ambiguous_states(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        validate_exposure_row(_exposure_row(**{field: value}))


@pytest.mark.parametrize("decision", ["unknown", "not_applicable"])
def test_changed_configuration_requires_recorded_decision(decision: str) -> None:
    with pytest.raises(ValueError, match="recorded decision"):
        validate_exposure_row(
            _exposure_row(
                configuration_changed_afterwards="true",
                decision_or_change=decision,
            )
        )


def test_exposure_ledger_rejects_bad_method_run_id_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="method"):
        validate_exposure_row(_exposure_row(method="segment_any_tree"))
    with pytest.raises(ValueError, match="run_id"):
        validate_exposure_row(_exposure_row(run_id="private/run"))
    row = _exposure_row()
    del row["notes"]
    with pytest.raises(ValueError, match="notes"):
        validate_exposure_row(row)


def test_budget_schema_normalizes_counts_and_hours_without_guessing() -> None:
    validated = validate_budget_row(
        _budget_row(
            run_id="unknown",
            evidence_path="outputs/public-safe/slurm-summary.csv",
        )
    )

    assert validated["validation_evaluations"] == 3
    assert validated["gpu_hours"] == 12.5
    assert validated["cpu_hours"] == 0.0
    assert validated["checkpoints_evaluated"] == "unknown"
    assert validated["optimizer_steps"] == "not_applicable"
    assert validated["manual_validation_inspection"] is True
    assert validated["run_id"] == "unknown"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configurations_attempted", -1),
        ("validation_evaluations", 1.5),
        ("checkpoints_evaluated", True),
        ("training_epochs", "01"),
        ("optimizer_steps", "unrecorded"),
        ("gpu_hours", -0.1),
        ("gpu_hours", float("nan")),
        ("cpu_hours", float("inf")),
        ("cpu_hours", "many"),
        ("manual_validation_inspection", "yes"),
    ],
)
def test_budget_schema_rejects_invalid_quantities(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        validate_budget_row(_budget_row(**{field: value}))


def test_budget_schema_rejects_bad_identifiers_and_implicit_text() -> None:
    with pytest.raises(ValueError, match="method"):
        validate_budget_row(_budget_row(method="TreeLearn"))
    with pytest.raises(ValueError, match="variant"):
        validate_budget_row(_budget_row(variant="published default"))
    with pytest.raises(ValueError, match="notes"):
        validate_budget_row(_budget_row(notes=""))
    with pytest.raises(ValueError, match="historical run_id"):
        validate_budget_row(_budget_row(run_id="../run"))


def test_provenance_schema_accepts_known_deep_learning_environment() -> None:
    source = _provenance_row(evidence_path="methods/example/provenance.yml")

    validated = validate_provenance_row(source)

    assert validated["container_digest"] == "sha256:" + "b" * 64
    assert validated["checkpoint_training_datasets"] == ("external-dataset-a",)
    assert source["checkpoint_training_datasets"] == ["external-dataset-a"]


def test_provenance_schema_accepts_explicit_not_applicable_values() -> None:
    validated = validate_provenance_row(
        _provenance_row(
            container_uri="not_applicable",
            container_digest="not_applicable",
            cuda_version="not_applicable",
            pytorch_version="not_applicable",
            minkowski_engine_version="not_applicable",
            checkpoint_source="not_applicable",
            checkpoint_sha256="not_applicable",
            checkpoint_training_datasets="not_applicable",
        )
    )

    assert validated["container_uri"] == "not_applicable"
    assert validated["checkpoint_training_datasets"] == "not_applicable"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("upstream_repository", "http://github.com/example/repo"),
        ("upstream_repository", "github.com/example/repo"),
        ("upstream_commit", "abc123"),
        ("upstream_commit", "A" * 40),
        ("upstream_dirty_state", "modified"),
        ("container_uri", "ghcr.io/example/image:latest"),
        ("container_digest", "b" * 63),
        ("container_digest", "B" * 64),
        ("checkpoint_sha256", "c" * 63),
        ("checkpoint_sha256", "C" * 64),
        ("checkpoint_training_datasets", []),
        ("checkpoint_training_datasets", [""]),
        ("python_version", ""),
        ("cuda_version", None),
    ],
)
def test_provenance_schema_rejects_malformed_or_implicit_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        validate_provenance_row(_provenance_row(**{field: value}))


def test_provenance_schema_enforces_not_applicable_consistency() -> None:
    with pytest.raises(ValueError, match="container_digest"):
        validate_provenance_row(
            _provenance_row(
                container_uri="not_applicable", container_digest="unknown"
            )
        )
    with pytest.raises(ValueError, match="checkpoint hash"):
        validate_provenance_row(
            _provenance_row(
                checkpoint_source="not_applicable",
                checkpoint_sha256="unknown",
                checkpoint_training_datasets="not_applicable",
            )
        )


def test_provenance_schema_validates_optional_identifiers() -> None:
    with pytest.raises(ValueError, match="method"):
        validate_provenance_row(_provenance_row(method="segment_any_tree"))
    with pytest.raises(ValueError, match="run_id"):
        validate_provenance_row(_provenance_row(run_id="../../private"))
    with pytest.raises(ValueError, match="evidence_path"):
        validate_provenance_row(_provenance_row(evidence_path=""))
