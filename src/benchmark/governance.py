"""Controlled governance values and lightweight row validation.

The validators in this module deliberately avoid a schema-framework
dependency.  They are suitable for both dictionaries created in Python and
rows read from CSV, and return normalized copies without dropping legacy or
method-specific fields.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


RESULT_STATUS_VALUES = frozenset(
    {
        "primary",
        "baseline",
        "diagnostic",
        "historical",
        "rejected",
        "operational_only",
        "candidate",
        "failed",
    }
)
COMPLETION_STATE_VALUES = frozenset({"completed", "partial", "pending"})
LEARNING_REGIME_VALUES = frozenset(
    {"supervised", "self_supervised", "unsupervised", "deterministic", "rule_based"}
)
DATASET_EXPOSURE_VALUES = frozenset(
    {
        "published_checkpoint",
        "external_only",
        "development_tuned",
        "development_trained",
        "none",
    }
)
PREDICTION_MATERIAL_VALUES = frozenset(
    {"woody_only", "woody_plus_leaf", "full_tree_material"}
)
REFERENCE_SCORING_MASK_VALUES = frozenset(
    {"classes_4_5_6", "class_3_ignored", "custom"}
)
LEAF_ATTACHMENT_VALUES = frozenset({"enabled", "disabled", "not_applicable"})

UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
MISSING_VALUE_SENTINELS = frozenset({UNKNOWN, NOT_APPLICABLE})

GOVERNANCE_REQUIRED_FIELDS = (
    "result_status",
    "completion_state",
    "ranking_eligible",
    "exclusion_reason",
    "learning_regime",
    "dataset_exposure",
    "prediction_material",
    "reference_scoring_mask",
    "leaf_attachment",
    "status_description",
)
EXPOSURE_REQUIRED_FIELDS = (
    "date",
    "method",
    "run_id",
    "test_job_executed",
    "metrics_viewed",
    "predictions_visualised",
    "configuration_changed_afterwards",
    "decision_or_change",
    "notes",
)
BUDGET_REQUIRED_FIELDS = (
    "method",
    "variant",
    "configurations_attempted",
    "validation_evaluations",
    "checkpoints_evaluated",
    "training_epochs",
    "optimizer_steps",
    "gpu_hours",
    "cpu_hours",
    "manual_validation_inspection",
    "hyperparameter_source",
    "notes",
)
PROVENANCE_REQUIRED_FIELDS = (
    "upstream_repository",
    "upstream_commit",
    "upstream_dirty_state",
    "container_uri",
    "container_digest",
    "python_version",
    "cuda_version",
    "pytorch_version",
    "minkowski_engine_version",
    "checkpoint_source",
    "checkpoint_sha256",
    "checkpoint_training_datasets",
)

_TOKEN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LEGACY_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VARIANT_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
_TIMESTAMP = re.compile(r"^[0-9]{8}T[0-9]{6}$")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_NONNEGATIVE_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_INFORMAL_RUN_COMPONENTS = frozenset(
    {"long", "full", "quicktune", "final", "best", "new"}
)
_EXPOSURE_STATE_VALUES = frozenset(
    {"true", "false", UNKNOWN, NOT_APPLICABLE}
)
_DIRTY_STATE_VALUES = frozenset({"clean", "dirty", UNKNOWN, NOT_APPLICABLE})


@dataclass(frozen=True)
class FutureRunId:
    """Parsed fields from one canonical future root-run identifier."""

    method: str
    dataset: str
    training_mode: str
    selection_mode: str
    split: str
    timestamp: str

    def __str__(self) -> str:
        return "__".join(
            (
                self.method,
                self.dataset,
                self.training_mode,
                self.selection_mode,
                self.split,
                self.timestamp,
            )
        )


def _copy_with_required_fields(
    row: Mapping[str, Any], required: Sequence[str], label: str
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    return dict(row)


def _controlled(value: Any, values: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in values:
        raise ValueError(f"{field} must be one of {sorted(values)}")
    return value


def _controlled_or_unknown(
    value: Any, values: frozenset[str], field: str
) -> str:
    if value == UNKNOWN:
        return UNKNOWN
    return _controlled(value, values, field)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field} must be non-empty; use {UNKNOWN!r} or "
            f"{NOT_APPLICABLE!r} when appropriate"
        )
    return value.strip()


def _strict_boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def _exposure_state(value: Any, field: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _controlled(value, _EXPOSURE_STATE_VALUES, field)


def _method_slug(value: Any, field: str = "method") -> str:
    text = _required_text(value, field)
    if not _TOKEN.fullmatch(text):
        raise ValueError(f"{field} must be a lower-case hyphenated slug")
    return text


def parse_future_run_id(value: Any) -> FutureRunId:
    """Validate and parse the canonical six-component future run-ID format."""

    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    parts = value.split("__")
    if len(parts) != 6:
        raise ValueError("Future run_id must contain exactly six __ components")
    method, dataset, training_mode, selection_mode, split, timestamp = parts
    semantic_components = (method, dataset, training_mode, selection_mode, split)
    if any(not _TOKEN.fullmatch(component) for component in semantic_components):
        raise ValueError(
            "Future run_id components must be lower-case hyphenated tokens"
        )
    if method == "sat":
        raise ValueError("Future run_id must use method 'segmentanytree', not 'sat'")
    informal: list[str] = []
    for field, component in zip(
        ("method", "dataset", "training_mode", "selection_mode", "split"),
        semantic_components,
        strict=True,
    ):
        tokens = set(component.split("-"))
        disallowed = tokens & _INFORMAL_RUN_COMPONENTS
        if field == "selection_mode" and component != "best":
            disallowed.discard("best")
        informal.extend(f"{field}:{token}" for token in sorted(disallowed))
    if informal:
        raise ValueError(
            "Future run_id contains an informal token: "
            + ", ".join(informal)
        )
    if not _TIMESTAMP.fullmatch(timestamp):
        raise ValueError("Future run_id timestamp must use YYYYMMDDTHHMMSS")
    try:
        datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
    except ValueError as exc:
        raise ValueError("Future run_id timestamp is not a valid UTC date/time") from exc
    return FutureRunId(
        method=method,
        dataset=dataset,
        training_mode=training_mode,
        selection_mode=selection_mode,
        split=split,
        timestamp=timestamp,
    )


def build_future_run_id(
    *,
    method: str,
    dataset: str,
    training_mode: str,
    selection_mode: str,
    split: str,
    timestamp: datetime,
) -> str:
    """Build a canonical ID from an aware timestamp, normalized to UTC."""

    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    candidate = "__".join(
        (method, dataset, training_mode, selection_mode, split, stamp)
    )
    return str(parse_future_run_id(candidate))


def validate_historical_run_id(value: Any) -> str:
    """Accept existing path-safe IDs without imposing the future convention."""

    if not isinstance(value, str):
        raise TypeError("historical run_id must be a string")
    if len(value) > 255:
        raise ValueError("historical run_id exceeds 255 characters")
    if value in {".", ".."} or not _LEGACY_IDENTIFIER.fullmatch(value):
        raise ValueError("historical run_id must be one safe path component")
    return value


def validate_governance_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate controlled result metadata while retaining legacy fields."""

    result = _copy_with_required_fields(
        row, GOVERNANCE_REQUIRED_FIELDS, "governance row"
    )
    result["result_status"] = _controlled(
        result["result_status"], RESULT_STATUS_VALUES, "result_status"
    )
    result["completion_state"] = _controlled(
        result["completion_state"], COMPLETION_STATE_VALUES, "completion_state"
    )
    result["learning_regime"] = _controlled_or_unknown(
        result["learning_regime"], LEARNING_REGIME_VALUES, "learning_regime"
    )
    result["dataset_exposure"] = _controlled_or_unknown(
        result["dataset_exposure"], DATASET_EXPOSURE_VALUES, "dataset_exposure"
    )
    result["prediction_material"] = _controlled_or_unknown(
        result["prediction_material"],
        PREDICTION_MATERIAL_VALUES,
        "prediction_material",
    )
    result["reference_scoring_mask"] = _controlled_or_unknown(
        result["reference_scoring_mask"],
        REFERENCE_SCORING_MASK_VALUES,
        "reference_scoring_mask",
    )
    result["leaf_attachment"] = _controlled_or_unknown(
        result["leaf_attachment"], LEAF_ATTACHMENT_VALUES, "leaf_attachment"
    )
    result["status_description"] = _required_text(
        result["status_description"], "status_description"
    )
    for legacy_field in ("legacy_result_role", "legacy_result_status"):
        if legacy_field in result:
            result[legacy_field] = _required_text(result[legacy_field], legacy_field)

    eligible = _strict_boolean(result["ranking_eligible"], "ranking_eligible")
    result["ranking_eligible"] = eligible
    reason = result["exclusion_reason"]
    if eligible:
        if result["completion_state"] != "completed" or result[
            "result_status"
        ] not in {"primary", "baseline"}:
            raise ValueError(
                "Only completed primary or baseline results may be ranking eligible"
            )
        if reason is not None and reason != "":
            raise ValueError(
                "ranking-eligible results must not contain an exclusion_reason"
            )
        result["exclusion_reason"] = ""
    else:
        reason_text = _required_text(reason, "exclusion_reason")
        if reason_text in MISSING_VALUE_SENTINELS:
            raise ValueError(
                "ranking-ineligible results require a specific exclusion_reason"
            )
        if not _VARIANT_IDENTIFIER.fullmatch(reason_text):
            raise ValueError(
                "exclusion_reason must be a lower-case underscore or hyphen slug"
            )
        result["exclusion_reason"] = reason_text
    return result


def validate_exposure_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one public-safe held-out test-exposure ledger event."""

    result = _copy_with_required_fields(
        row, EXPOSURE_REQUIRED_FIELDS, "test-exposure row"
    )
    date = _required_text(result["date"], "date")
    if date not in MISSING_VALUE_SENTINELS:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
            raise ValueError("date must use YYYY-MM-DD, unknown, or not_applicable")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date is not a valid calendar date") from exc
    result["date"] = date
    result["method"] = _method_slug(result["method"])
    result["run_id"] = validate_historical_run_id(result["run_id"])
    for field in (
        "test_job_executed",
        "metrics_viewed",
        "predictions_visualised",
        "configuration_changed_afterwards",
    ):
        result[field] = _exposure_state(result[field], field)
    result["decision_or_change"] = _required_text(
        result["decision_or_change"], "decision_or_change"
    )
    result["notes"] = _required_text(result["notes"], "notes")
    if result["configuration_changed_afterwards"] == "true" and result[
        "decision_or_change"
    ] in MISSING_VALUE_SENTINELS:
        raise ValueError(
            "configuration_changed_afterwards=true requires a recorded decision_or_change"
        )
    if "evidence_path" in result:
        result["evidence_path"] = _required_text(
            result["evidence_path"], "evidence_path"
        )
    return result


def _nonnegative_integer_or_sentinel(value: Any, field: str) -> int | str:
    if isinstance(value, str) and value in MISSING_VALUE_SENTINELS:
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer or sentinel")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} cannot be negative")
        return value
    if isinstance(value, str) and _NONNEGATIVE_INTEGER.fullmatch(value):
        return int(value)
    raise ValueError(
        f"{field} must be a non-negative integer, {UNKNOWN}, or {NOT_APPLICABLE}"
    )


def _nonnegative_number_or_sentinel(value: Any, field: str) -> float | str:
    if isinstance(value, str) and value in MISSING_VALUE_SENTINELS:
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative number or sentinel")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a non-negative number, {UNKNOWN}, or "
            f"{NOT_APPLICABLE}"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def validate_budget_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one method-development and tuning-budget record."""

    result = _copy_with_required_fields(row, BUDGET_REQUIRED_FIELDS, "budget row")
    result["method"] = _method_slug(result["method"])
    variant = _required_text(result["variant"], "variant")
    if not _VARIANT_IDENTIFIER.fullmatch(variant):
        raise ValueError("variant must be a lower-case underscore or hyphen slug")
    result["variant"] = variant
    for field in (
        "configurations_attempted",
        "validation_evaluations",
        "checkpoints_evaluated",
        "training_epochs",
        "optimizer_steps",
    ):
        result[field] = _nonnegative_integer_or_sentinel(result[field], field)
    for field in ("gpu_hours", "cpu_hours"):
        result[field] = _nonnegative_number_or_sentinel(result[field], field)
    manual = result["manual_validation_inspection"]
    if isinstance(manual, str) and manual in MISSING_VALUE_SENTINELS:
        result["manual_validation_inspection"] = manual
    else:
        result["manual_validation_inspection"] = _strict_boolean(
            manual, "manual_validation_inspection"
        )
    result["hyperparameter_source"] = _required_text(
        result["hyperparameter_source"], "hyperparameter_source"
    )
    result["notes"] = _required_text(result["notes"], "notes")
    if "run_id" in result and result["run_id"] not in MISSING_VALUE_SENTINELS:
        result["run_id"] = validate_historical_run_id(result["run_id"])
    if "evidence_path" in result:
        result["evidence_path"] = _required_text(
            result["evidence_path"], "evidence_path"
        )
    return result


def _explicit_text_or_sentinel(value: Any, field: str) -> str:
    return _required_text(value, field)


def _sha256_or_sentinel(value: Any, field: str) -> str:
    text = _explicit_text_or_sentinel(value, field)
    if text in MISSING_VALUE_SENTINELS:
        return text
    digest = text.removeprefix("sha256:")
    if not _HEX_64.fullmatch(digest):
        raise ValueError(f"{field} must be a SHA-256 digest or explicit sentinel")
    return text


def validate_provenance_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the common environment, upstream and checkpoint provenance."""

    result = _copy_with_required_fields(
        row, PROVENANCE_REQUIRED_FIELDS, "provenance row"
    )
    repository = _explicit_text_or_sentinel(
        result["upstream_repository"], "upstream_repository"
    )
    if repository not in MISSING_VALUE_SENTINELS:
        parsed = urlparse(repository)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("upstream_repository must be a public HTTPS URL")
    result["upstream_repository"] = repository

    commit = _explicit_text_or_sentinel(result["upstream_commit"], "upstream_commit")
    if commit not in MISSING_VALUE_SENTINELS and not _HEX_40.fullmatch(commit):
        raise ValueError("upstream_commit must be a 40-character lower-case Git hash")
    result["upstream_commit"] = commit
    result["upstream_dirty_state"] = _controlled(
        result["upstream_dirty_state"], _DIRTY_STATE_VALUES, "upstream_dirty_state"
    )

    container_uri = _explicit_text_or_sentinel(result["container_uri"], "container_uri")
    if container_uri not in MISSING_VALUE_SENTINELS and not re.match(
        r"^[a-z][a-z0-9+.-]*://[^\s]+$", container_uri
    ):
        raise ValueError("container_uri must be a URI or explicit sentinel")
    result["container_uri"] = container_uri
    result["container_digest"] = _sha256_or_sentinel(
        result["container_digest"], "container_digest"
    )
    if container_uri == NOT_APPLICABLE and result["container_digest"] != NOT_APPLICABLE:
        raise ValueError(
            "container_digest must be not_applicable when container_uri is not_applicable"
        )

    for field in (
        "python_version",
        "cuda_version",
        "pytorch_version",
        "minkowski_engine_version",
    ):
        result[field] = _explicit_text_or_sentinel(result[field], field)

    result["checkpoint_source"] = _explicit_text_or_sentinel(
        result["checkpoint_source"], "checkpoint_source"
    )
    result["checkpoint_sha256"] = _sha256_or_sentinel(
        result["checkpoint_sha256"], "checkpoint_sha256"
    )
    datasets = result["checkpoint_training_datasets"]
    if isinstance(datasets, str):
        result["checkpoint_training_datasets"] = _explicit_text_or_sentinel(
            datasets, "checkpoint_training_datasets"
        )
    elif isinstance(datasets, Sequence) and not isinstance(
        datasets, (bytes, bytearray)
    ):
        if not datasets:
            raise ValueError(
                "checkpoint_training_datasets must not be empty; use an explicit sentinel"
            )
        result["checkpoint_training_datasets"] = tuple(
            _required_text(value, "checkpoint_training_datasets item")
            for value in datasets
        )
    else:
        raise ValueError(
            "checkpoint_training_datasets must be a non-empty sequence or sentinel"
        )
    if result["checkpoint_source"] == NOT_APPLICABLE and (
        result["checkpoint_sha256"] != NOT_APPLICABLE
        or result["checkpoint_training_datasets"] != NOT_APPLICABLE
    ):
        raise ValueError(
            "checkpoint hash and training datasets must be not_applicable when "
            "checkpoint_source is not_applicable"
        )

    if "method" in result:
        result["method"] = _method_slug(result["method"])
    if "run_id" in result and result["run_id"] not in MISSING_VALUE_SENTINELS:
        result["run_id"] = validate_historical_run_id(result["run_id"])
    if "evidence_path" in result:
        result["evidence_path"] = _required_text(
            result["evidence_path"], "evidence_path"
        )
    return result
