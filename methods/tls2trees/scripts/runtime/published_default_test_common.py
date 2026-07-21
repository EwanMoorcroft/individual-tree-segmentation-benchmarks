"""Fail-closed helpers for the fixed TLS2trees published-default test route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_VARIANT = "published_default"
EXPECTED_SPLIT = "test"
TARGETS = ("leaf_off", "leaf_on")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_yaml(path_text: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_path(path_text)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload, path


def load_json(path_text: str | Path) -> tuple[dict[str, Any], Path]:
    path = resolve_path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload, path


def validate_frozen_configuration(
    workflow_config_path: str | Path,
    published_config_path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    """Prove the test configuration is the unselected published parameter set."""

    workflow, workflow_path = load_yaml(workflow_config_path)
    published, published_path = load_yaml(published_config_path)
    if (
        workflow.get("schema_version") != 1
        or workflow.get("project", {}).get("status")
        != "published_default_configuration_frozen"
        or workflow.get("dataset", {}).get("split") != EXPECTED_SPLIT
        or workflow.get("method", {}).get("variant") != EXPECTED_VARIANT
        or workflow.get("method", {}).get("selected_from_for_instance_metrics")
        is not False
        or workflow.get("method", {}).get(
            "configuration_changes_after_test_permitted"
        )
        is not False
        or workflow.get("targets") != list(TARGETS)
    ):
        raise ValueError("Published-default test workflow is not cleanly frozen")
    configured_source = resolve_path(workflow["method"]["source_config"])
    if configured_source != published_path:
        raise ValueError("Workflow source_config does not match the submitted config")
    if published.get("method", {}).get("variant") != EXPECTED_VARIANT:
        raise ValueError("Submitted method config is not published_default")
    if workflow.get("frozen_semantic_parameters") != published.get(
        "semantic_parameters"
    ):
        raise ValueError("Published semantic parameters differ from the frozen test")
    if workflow.get("frozen_instance_parameters") != published.get(
        "instance_parameters"
    ):
        raise ValueError("Published instance parameters differ from the frozen test")
    expected = workflow.get("dataset", {})
    paths = expected.get("exact_relative_paths")
    if (
        int(expected.get("expected_plot_count", -1)) != 11
        or int(expected.get("expected_point_count", -1)) != 49_709_922
        or int(expected.get("expected_reference_tree_count", -1)) != 323
        or not isinstance(paths, list)
        or len(paths) != 11
        or len(set(paths)) != 11
    ):
        raise ValueError("Published-default test config lacks the exact 11-plot contract")
    evaluation = workflow.get("evaluation", {})
    if (
        float(evaluation.get("iou_threshold", -1)) != 0.5
        or int(evaluation.get("expected_metric_count", -1)) != 22
        or evaluation.get("valid_empty_predictions_are_evaluated_as_zero")
        is not True
        or evaluation.get("configuration_changed_after_test") is not False
    ):
        raise ValueError("Published-default evaluation contract changed")
    return workflow, workflow_path, published, published_path


def validate_exact_manifest(
    manifest: dict[str, Any], workflow: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = workflow["dataset"]
    plots = manifest.get("plots")
    if not isinstance(plots, list):
        raise ValueError("Test manifest has no plot list")
    if (
        manifest.get("dataset_split") != EXPECTED_SPLIT
        or len(plots) != int(expected["expected_plot_count"])
        or [int(row.get("task_index", -1)) for row in plots]
        != list(range(int(expected["expected_plot_count"])))
        or [row.get("relative_path") for row in plots]
        != expected["exact_relative_paths"]
        or sum(int(row.get("point_count", -1)) for row in plots)
        != int(expected["expected_point_count"])
        or sum(int(row.get("reference_tree_count", -1)) for row in plots)
        != int(expected["expected_reference_tree_count"])
    ):
        raise ValueError("Workflow requires the exact ordered 11-plot test manifest")
    return plots
