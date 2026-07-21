"""Publish the completed TLS2trees development leaf-screen evidence safely."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml


EVALUATION = Path(__file__).resolve().parent
if str(EVALUATION) not in sys.path:
    sys.path.insert(0, str(EVALUATION))

from tls2trees_publication import (  # noqa: E402
    publication_lock,
    publication_path,
    require_git_head,
    validate_git_worktree,
)


EXPECTED_EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
EXPECTED_EVALUATION_MASK = (
    "union_of_reference_target_and_predicted_target_points_excluding_class3_outpoints"
)
EXPECTED_COLLECTIONS = {"CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"}
EXPECTED_CANDIDATE_COUNT = 9
EXPECTED_PLOTS_PER_CANDIDATE = 5
EXPECTED_METRIC_COUNT = EXPECTED_CANDIDATE_COUNT * EXPECTED_PLOTS_PER_CANDIDATE

PUBLIC_PLOT_NAME = "tls2trees_development_leaf_screen_plot_results.csv"
PUBLIC_CANDIDATE_NAME = "tls2trees_development_leaf_screen_candidate_results.csv"
PUBLIC_PROVENANCE_NAME = "tls2trees_development_leaf_screen_provenance.json"
PUBLICATION_STAGE_NAME = ".tls2trees_development_leaf_screen_publication.staging"

SOURCE_PLOT_FIELDS = (
    "candidate_index",
    "candidate_id",
    "add_leaves_voxel_length",
    "add_leaves_edge_length",
    "stage0_index",
    "collection",
    "safe_plot_id",
    "relative_path",
    "target",
    "status",
    "safe_for_scoring",
    "prediction_instance_count",
    "reference_instance_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "oversegmented_reference_count",
    "undersegmented_prediction_count",
    "ignored_class3_predicted_point_count",
    "instance_runtime_seconds",
    "instance_peak_rss_gb",
    "adapter_runtime_seconds",
    "metrics_path",
    "metrics_sha256",
    "error",
)
PUBLIC_PLOT_FIELDS = tuple(
    field for field in SOURCE_PLOT_FIELDS if field != "metrics_path"
)
PUBLIC_CANDIDATE_FIELDS = (
    "candidate_id",
    "target",
    "expected_plot_count",
    "evaluated_plot_count",
    "failed_or_invalid_plot_count",
    "prediction_instance_count",
    "reference_instance_count",
    "true_positives",
    "false_positives",
    "false_negatives",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_plot_f1",
    "per_site_f1",
    "oversegmented_reference_count",
    "undersegmented_prediction_count",
    "total_instance_runtime_seconds",
    "maximum_instance_peak_rss_gb",
)
HEX_64 = set("0123456789abcdef")
HEX_40 = HEX_64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= HEX_64
    )


def is_git_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX_40


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_csv(fields: Iterable[str], rows: list[dict[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    fieldnames = list(fields)
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row[field]) for field in fieldnames})
    return handle.getvalue().encode("utf-8")


def render_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validate_staged_bundle(
    *,
    staged: dict[str, Path],
    expected_payloads: dict[str, bytes],
    provenance: dict[str, Any],
) -> None:
    """Validate the complete on-disk bundle before publishing any target."""

    if set(staged) != set(expected_payloads):
        raise ValueError("Leaf-screen staged bundle is incomplete")
    for name, path in staged.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
        ):
            raise ValueError(f"Leaf-screen staged artifact is invalid: {path}")
        if path.read_bytes() != expected_payloads[name]:
            raise ValueError(f"Leaf-screen staged artifact differs after write: {path}")

    plot_fields, plot_rows = read_csv(staged[PUBLIC_PLOT_NAME])
    if (
        plot_fields != list(PUBLIC_PLOT_FIELDS)
        or len(plot_rows) != EXPECTED_METRIC_COUNT
    ):
        raise ValueError("Leaf-screen staged plot evidence is invalid")
    candidate_fields, candidate_rows = read_csv(staged[PUBLIC_CANDIDATE_NAME])
    if (
        candidate_fields != list(PUBLIC_CANDIDATE_FIELDS)
        or len(candidate_rows) != EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError("Leaf-screen staged candidate evidence is invalid")
    staged_provenance = json.loads(
        staged[PUBLIC_PROVENANCE_NAME].read_text(encoding="utf-8")
    )
    if staged_provenance != provenance:
        raise ValueError("Leaf-screen staged provenance differs after write")
    public_artifacts = staged_provenance.get("public_artifacts", {})
    for key, name in (
        ("plot_results", PUBLIC_PLOT_NAME),
        ("candidate_results", PUBLIC_CANDIDATE_NAME),
    ):
        recorded = public_artifacts.get(key, {})
        if recorded.get("sha256") != sha256(staged[name]):
            raise ValueError(f"Leaf-screen staged {key} hash does not reconcile")


def existing_output_is_exact(path: Path, expected: bytes) -> bool:
    """Accept only a regular, byte-identical prior publication output."""

    if not path.exists() and not path.is_symlink():
        return False
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_nlink != 1
        or path.read_bytes() != expected
    ):
        raise FileExistsError(
            "Existing leaf-screen public evidence conflicts with rendered bundle: "
            f"{path}"
        )
    return True


def publish_bundle(
    *,
    output_dir: Path,
    payloads: dict[str, bytes],
    provenance: dict[str, Any],
    project_root: Path | None = None,
    expected_head: str | None = None,
) -> tuple[int, int]:
    """Stage, validate, and atomically create missing public bundle members.

    A previous interrupted invocation may have created any prefix of the bundle.
    Byte-identical members are retained, while any conflict is rejected before a
    missing target is created. Hard-link creation provides atomic no-overwrite
    publication because the staging directory resides beside the targets.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / PUBLICATION_STAGE_NAME
    if stage_dir.is_symlink() or (stage_dir.exists() and not stage_dir.is_dir()):
        raise FileExistsError(f"Invalid leaf-screen publication stage: {stage_dir}")
    stage_dir.mkdir(exist_ok=True)
    expected_names = set(payloads)
    unexpected = [
        path for path in stage_dir.iterdir() if path.name not in expected_names
    ]
    if unexpected:
        raise FileExistsError(
            "Leaf-screen publication stage contains unexpected artifacts: "
            + ", ".join(str(path) for path in unexpected)
        )

    staged = {name: stage_dir / name for name in payloads}
    invalid_staged = [
        path
        for path in staged.values()
        if path.is_symlink() or (path.exists() and not path.is_file())
    ]
    if invalid_staged:
        raise FileExistsError(
            "Leaf-screen publication stage contains invalid artifacts: "
            + ", ".join(str(path) for path in invalid_staged)
        )
    created_staging: set[Path] = set()
    try:
        # Accept only exact standalone stage files or the one recognizable
        # two-link state left by a crash between linking and unlinking a target.
        for name, path in staged.items():
            if path.exists():
                target = output_dir / name
                if path.read_bytes() != payloads[name]:
                    raise FileExistsError(
                        f"Leaf-screen staging content conflicts: {path}"
                    )
                if path.stat().st_nlink != 1:
                    if (
                        path.stat().st_nlink != 2
                        or not target.is_file()
                        or target.is_symlink()
                        or not os.path.samefile(path, target)
                    ):
                        raise FileExistsError(
                            f"Leaf-screen staging hard link is unsafe: {path}"
                        )
                path.unlink()
        for name, payload in payloads.items():
            staged[name].write_bytes(payload)
            created_staging.add(staged[name])
        validate_staged_bundle(
            staged=staged,
            expected_payloads=payloads,
            provenance=provenance,
        )

        targets = {name: output_dir / name for name in payloads}
        existing = {
            name: existing_output_is_exact(targets[name], payload)
            for name, payload in payloads.items()
        }
        created = 0
        retained = 0
        if project_root is not None and expected_head is not None:
            require_git_head(project_root, expected_head)
        for name, payload in payloads.items():
            if existing_output_is_exact(targets[name], payload) != existing[name]:
                raise RuntimeError(
                    f"Leaf-screen public target changed during staging: {targets[name]}"
                )
        for name, payload in payloads.items():
            if project_root is not None and expected_head is not None:
                require_git_head(project_root, expected_head)
            if existing_output_is_exact(targets[name], payload) != existing[name]:
                raise RuntimeError(
                    "Leaf-screen public target changed before publication: "
                    f"{targets[name]}"
                )
            if existing[name]:
                retained += 1
                if project_root is not None and expected_head is not None:
                    require_git_head(project_root, expected_head)
                continue
            try:
                os.link(staged[name], targets[name])
                staged[name].unlink()
                created_staging.discard(staged[name])
                existing_output_is_exact(targets[name], payload)
                if project_root is not None and expected_head is not None:
                    require_git_head(project_root, expected_head)
                created += 1
            except FileExistsError:
                # A concurrent invocation may have won the no-overwrite race.
                if existing_output_is_exact(targets[name], payload):
                    retained += 1
                    continue
                raise
        for name, payload in payloads.items():
            if not existing_output_is_exact(targets[name], payload):
                raise RuntimeError(
                    f"Leaf-screen published bundle is incomplete: {targets[name]}"
                )
        if project_root is not None and expected_head is not None:
            require_git_head(project_root, expected_head)
        return created, retained
    finally:
        for path in created_staging:
            if path.exists() or path.is_symlink():
                path.unlink()
        try:
            stage_dir.rmdir()
        except OSError:
            # Preserve unexpected concurrent content rather than deleting it.
            pass


def validate_source_csv(
    *,
    path: Path,
    expected_fields: tuple[str, ...],
    expected_rows: list[dict[str, Any]],
) -> None:
    fields, observed_rows = read_csv(path)
    if fields != list(expected_fields):
        raise ValueError(f"Unexpected source CSV fields: {path}")
    if len(observed_rows) != len(expected_rows):
        raise ValueError(f"Source CSV row count differs from summary: {path}")
    for index, (observed, expected) in enumerate(zip(observed_rows, expected_rows)):
        expected_cells = {
            field: csv_cell(expected[field]) for field in expected_fields
        }
        if observed != expected_cells:
            raise ValueError(
                f"Source CSV row {index} differs from embedded summary: {path}"
            )


def normalize_source_evaluator(value: Any) -> str:
    """Accept the semantic protocol and remove an optional numeric tag."""

    if not isinstance(value, str):
        raise ValueError("Leaf-screen evaluator is missing")
    parts = value.split("_")
    prefix = ["for", "instance", "tls2trees", "source", "row"]
    suffix = ["class3", "ignore"]
    if parts[: len(prefix)] != prefix or parts[-len(suffix) :] != suffix:
        raise ValueError("Leaf-screen evaluator is not the class-3-ignore protocol")
    middle = parts[len(prefix) : -len(suffix)]
    if middle:
        if not (
            len(middle) == 1
            and len(middle[0]) > 1
            and middle[0][0] == "v"
            and middle[0][1:].isdigit()
        ):
            raise ValueError("Leaf-screen evaluator has an unsupported tag")
    return EXPECTED_EVALUATOR


def assert_relative_dataset_path(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("Plot relative_path is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError(f"Unsafe plot relative_path: {value!r}")


def assert_public_safe(value: Any, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_public_safe(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_safe(item, f"{location}[{index}]")
        return
    if not isinstance(value, str):
        return
    lowered = value.casefold()
    forbidden = (
        "/users/",
        "/home/",
        "/mnt/",
        "fastscratch",
        "sgemoorc",
        "barkla2.liv",
    )
    if Path(value).is_absolute() or any(token in lowered for token in forbidden):
        raise ValueError(f"Private or absolute path at {location}")


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def assert_close(observed: Any, expected: float, field: str) -> None:
    if not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Candidate aggregate {field} does not reconcile")


def reconcile_candidate(
    rows: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    if len(rows) != EXPECTED_PLOTS_PER_CANDIDATE:
        raise ValueError("Candidate does not contain five plot metrics")
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision, recall, f1 = prf(tp, fp, fn)
    exact = {
        "target": "leaf_on",
        "expected_plot_count": EXPECTED_PLOTS_PER_CANDIDATE,
        "evaluated_plot_count": EXPECTED_PLOTS_PER_CANDIDATE,
        "failed_or_invalid_plot_count": 0,
        "prediction_instance_count": sum(
            int(row["prediction_instance_count"]) for row in rows
        ),
        "reference_instance_count": sum(
            int(row["reference_instance_count"]) for row in rows
        ),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "oversegmented_reference_count": sum(
            int(row["oversegmented_reference_count"]) for row in rows
        ),
        "undersegmented_prediction_count": sum(
            int(row["undersegmented_prediction_count"]) for row in rows
        ),
    }
    for field, expected in exact.items():
        if aggregate.get(field) != expected:
            raise ValueError(f"Candidate aggregate {field} does not reconcile")
    floats = {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_plot_f1": sum(float(row["f1"]) for row in rows) / len(rows),
        "total_instance_runtime_seconds": sum(
            float(row["instance_runtime_seconds"] or 0.0) for row in rows
        ),
        "maximum_instance_peak_rss_gb": max(
            float(row["instance_peak_rss_gb"] or 0.0) for row in rows
        ),
    }
    for field, expected in floats.items():
        assert_close(aggregate.get(field), expected, field)
    expected_site_f1 = {
        str(row["collection"]): float(row["f1"]) for row in rows
    }
    if aggregate.get("per_site_f1") != expected_site_f1:
        raise ValueError("Candidate aggregate per_site_f1 does not reconcile")


def validate_summary(
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    expected_run_id: str,
    expected_source_run_id: str,
    expected_manifest_sha256: str,
    expected_source_config_sha256: str,
    expected_development_evidence_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "status": "development_leaf_screen_completed",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "split": "development",
        "target": "leaf_on",
        "workflow_run_id": expected_run_id,
        "manifest_sha256": expected_manifest_sha256,
        "candidate_config_sha256": expected_source_config_sha256,
        "development_evidence_sha256": expected_development_evidence_sha256,
        "development_evidence_run_id": expected_source_run_id,
        "evaluation_mask": EXPECTED_EVALUATION_MASK,
        "expected_metric_count": EXPECTED_METRIC_COUNT,
        "valid_metric_count": EXPECTED_METRIC_COUNT,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "development_reference_labels_accessed": True,
        "development_accuracy_metrics_computed": True,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"Leaf-screen summary has invalid {key}")
    if summary.get("incomplete_tasks") != []:
        raise ValueError("Leaf-screen summary contains incomplete tasks")
    normalize_source_evaluator(summary.get("evaluator"))
    if summary.get("ignored_semantic_classes") != [3]:
        raise ValueError("Leaf-screen summary must ignore class 3")

    candidates = config.get("candidates", [])
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Publication config must define nine candidates")
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(set(candidate_ids)) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Publication config has duplicate candidate IDs")
    if [int(candidate["candidate_index"]) for candidate in candidates] != list(
        range(EXPECTED_CANDIDATE_COUNT)
    ):
        raise ValueError("Publication config candidate indices changed")
    if config.get("evaluation", {}).get("evaluator") != EXPECTED_EVALUATOR:
        raise ValueError("Publication config does not use the neutral evaluator")
    if config.get("evaluation", {}).get("evaluation_mask") != EXPECTED_EVALUATION_MASK:
        raise ValueError("Publication config evaluation mask changed")
    if config.get("evaluation", {}).get("ignored_semantic_classes") != [3]:
        raise ValueError("Publication config must ignore class 3")
    if config.get("scope", {}).get("held_out_test_accessed") is not False:
        raise ValueError("Publication config crossed the held-out boundary")
    if config.get("scope", {}).get("final_configuration_selected") is not False:
        raise ValueError("Leaf screen cannot select a final configuration")
    expected_parameters = {
        str(candidate["candidate_id"]): candidate["parameters"]
        for candidate in candidates
    }
    if summary.get("candidate_parameters") != expected_parameters:
        raise ValueError("Publication config no longer matches executed candidates")

    rows = summary.get("plot_metrics")
    aggregates = summary.get("aggregates")
    if not isinstance(rows, list) or len(rows) != EXPECTED_METRIC_COUNT:
        raise ValueError("Leaf-screen summary must contain 45 plot metrics")
    if not isinstance(aggregates, list) or len(aggregates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Leaf-screen summary must contain nine candidate results")
    if any(set(row) != set(SOURCE_PLOT_FIELDS) for row in rows):
        raise ValueError("Leaf-screen plot schema changed")
    if any(set(row) != set(PUBLIC_CANDIDATE_FIELDS) for row in aggregates):
        raise ValueError("Leaf-screen candidate schema changed")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_pairs: set[tuple[str, int]] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        stage0_index = int(row["stage0_index"])
        if candidate_id not in expected_parameters:
            raise ValueError("Plot metric has an unknown candidate")
        pair = (candidate_id, stage0_index)
        if pair in observed_pairs:
            raise ValueError("Plot metric candidate/stage index is duplicated")
        observed_pairs.add(pair)
        if stage0_index not in range(EXPECTED_PLOTS_PER_CANDIDATE):
            raise ValueError("Plot metric stage index is outside the frozen screen")
        if row["status"] != "evaluated" or row["safe_for_scoring"] is not True:
            raise ValueError("Leaf-screen plot is not valid for scoring")
        if row["target"] != "leaf_on" or row["error"] not in (None, ""):
            raise ValueError("Leaf-screen plot target or error is invalid")
        if row["collection"] not in EXPECTED_COLLECTIONS:
            raise ValueError("Leaf-screen plot collection is invalid")
        assert_relative_dataset_path(row["relative_path"])
        if not is_sha256(row["metrics_sha256"]):
            raise ValueError("Leaf-screen plot has no metric hash")
        if int(row["prediction_instance_count"]) != (
            int(row["true_positives"]) + int(row["false_positives"])
        ):
            raise ValueError("Plot prediction counts do not reconcile")
        if int(row["reference_instance_count"]) != (
            int(row["true_positives"]) + int(row["false_negatives"])
        ):
            raise ValueError("Plot reference counts do not reconcile")
        grouped[candidate_id].append(row)
    if set(grouped) != set(candidate_ids):
        raise ValueError("Leaf-screen candidate coverage is incomplete")
    if any(
        {row["collection"] for row in group} != EXPECTED_COLLECTIONS
        for group in grouped.values()
    ):
        raise ValueError("Each candidate must cover all five collections")

    by_candidate = {str(row["candidate_id"]): row for row in aggregates}
    if set(by_candidate) != set(candidate_ids):
        raise ValueError("Candidate result coverage is incomplete")
    for candidate_id in candidate_ids:
        reconcile_candidate(grouped[candidate_id], by_candidate[candidate_id])
    ranking = summary.get("candidate_ranking_for_review")
    top_three = summary.get("top_three_candidate_ids_for_review")
    if not isinstance(ranking, list) or set(ranking) != set(candidate_ids):
        raise ValueError("Leaf-screen ranking is incomplete")
    if top_three != ranking[:3]:
        raise ValueError("Leaf-screen top-three evidence differs from ranking")
    return rows, aggregates


def _finalise_locked(
    *,
    summary_path: Path,
    source_plot_csv: Path,
    source_candidate_csv: Path,
    candidate_config_path: Path,
    output_dir: Path,
    source_state_sha256: str,
    expected_run_id: str,
    expected_source_run_id: str,
    expected_semantic_cache_run_id: str,
    expected_manifest_sha256: str,
    expected_source_config_sha256: str,
    expected_development_evidence_sha256: str,
    source_benchmark_commit: str,
    publication_benchmark_commit: str,
    publication_project_root: Path | None,
    expected_publication_head: str | None,
) -> dict[str, Any]:
    paths = (summary_path, source_plot_csv, source_candidate_csv, candidate_config_path)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Leaf-screen publication input is missing")
    for value, label in (
        (source_state_sha256, "source state"),
        (expected_manifest_sha256, "manifest"),
        (expected_source_config_sha256, "source candidate config"),
        (expected_development_evidence_sha256, "development evidence"),
    ):
        if not is_sha256(value):
            raise ValueError(f"Invalid {label} SHA-256")
    for value, label in (
        (source_benchmark_commit, "source benchmark commit"),
        (publication_benchmark_commit, "publication benchmark commit"),
    ):
        if not is_git_commit(value):
            raise ValueError(f"Invalid {label}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    rows, aggregates = validate_summary(
        summary=summary,
        config=config,
        expected_run_id=expected_run_id,
        expected_source_run_id=expected_source_run_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_config_sha256=expected_source_config_sha256,
        expected_development_evidence_sha256=expected_development_evidence_sha256,
    )
    validate_source_csv(
        path=source_plot_csv,
        expected_fields=SOURCE_PLOT_FIELDS,
        expected_rows=rows,
    )
    validate_source_csv(
        path=source_candidate_csv,
        expected_fields=PUBLIC_CANDIDATE_FIELDS,
        expected_rows=aggregates,
    )

    public_rows = [
        {field: row[field] for field in PUBLIC_PLOT_FIELDS} for row in rows
    ]
    public_aggregates = [
        {field: row[field] for field in PUBLIC_CANDIDATE_FIELDS}
        for row in aggregates
    ]
    plot_payload = render_csv(PUBLIC_PLOT_FIELDS, public_rows)
    candidate_payload = render_csv(PUBLIC_CANDIDATE_FIELDS, public_aggregates)
    provenance = {
        "schema_version": 1,
        "status": "development_leaf_screen_publication_completed",
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": "development_tuned",
        "training_mode": "external_training_only",
        "split": "development",
        "target": "leaf_on",
        "workflow_run_id": expected_run_id,
        "source_stage1_run_id": expected_source_run_id,
        "source_semantic_cache_run_id": expected_semantic_cache_run_id,
        "source_benchmark_commit": source_benchmark_commit,
        "publication_benchmark_commit": publication_benchmark_commit,
        "evaluation_protocol": EXPECTED_EVALUATOR,
        "evaluation_mask": EXPECTED_EVALUATION_MASK,
        "ignored_semantic_classes": [3],
        "expected_metric_count": EXPECTED_METRIC_COUNT,
        "valid_metric_count": EXPECTED_METRIC_COUNT,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "plots_per_candidate": EXPECTED_PLOTS_PER_CANDIDATE,
        "development_reference_labels_accessed": True,
        "development_accuracy_metrics_computed": True,
        "held_out_test_accessed": False,
        "final_configuration_selected": False,
        "candidate_ranking_for_review": summary["candidate_ranking_for_review"],
        "top_three_candidate_ids_for_review": summary[
            "top_three_candidate_ids_for_review"
        ],
        "candidate_parameters": summary["candidate_parameters"],
        "source_artifact_hashes": {
            "state_sha256": source_state_sha256,
            "summary_sha256": sha256(summary_path),
            "plot_csv_sha256": sha256(source_plot_csv),
            "candidate_csv_sha256": sha256(source_candidate_csv),
            "manifest_sha256": expected_manifest_sha256,
            "execution_candidate_config_sha256": expected_source_config_sha256,
            "development_evidence_sha256": expected_development_evidence_sha256,
        },
        "publication_candidate_config": (
            "methods/tls2trees/configs/"
            "for_instance_development_tuned_leaf_screen.yml"
        ),
        "publication_candidate_config_sha256": sha256(candidate_config_path),
        "public_artifacts": {
            "plot_results": {
                "path": f"methods/tls2trees/examples/{PUBLIC_PLOT_NAME}",
                "row_count": EXPECTED_METRIC_COUNT,
                "sha256": sha256_bytes(plot_payload),
            },
            "candidate_results": {
                "path": f"methods/tls2trees/examples/{PUBLIC_CANDIDATE_NAME}",
                "row_count": EXPECTED_CANDIDATE_COUNT,
                "sha256": sha256_bytes(candidate_payload),
            },
        },
    }
    assert_public_safe(public_rows, "plot_results")
    assert_public_safe(public_aggregates, "candidate_results")
    assert_public_safe(provenance, "provenance")

    plot_output = output_dir / PUBLIC_PLOT_NAME
    candidate_output = output_dir / PUBLIC_CANDIDATE_NAME
    provenance_output = output_dir / PUBLIC_PROVENANCE_NAME
    payloads = {
        PUBLIC_PLOT_NAME: plot_payload,
        PUBLIC_CANDIDATE_NAME: candidate_payload,
        PUBLIC_PROVENANCE_NAME: render_json(provenance),
    }
    created, retained = publish_bundle(
        output_dir=output_dir,
        payloads=payloads,
        provenance=provenance,
        project_root=publication_project_root,
        expected_head=expected_publication_head,
    )
    print("status=TLS2TREES_DEVELOPMENT_LEAF_SCREEN_PUBLICATION_READY")
    print(f"plot_results={plot_output}")
    print(f"candidate_results={candidate_output}")
    print(f"provenance={provenance_output}")
    print(f"plot_rows={EXPECTED_METRIC_COUNT}")
    print(f"candidate_rows={EXPECTED_CANDIDATE_COUNT}")
    print(f"publication_outputs_created={created}")
    print(f"publication_outputs_retained={retained}")
    print("held_out_test_accessed=false")
    print("final_configuration_selected=false")
    return provenance


def finalise(
    *,
    summary_path: Path,
    source_plot_csv: Path,
    source_candidate_csv: Path,
    candidate_config_path: Path,
    output_dir: Path,
    project_root: Path,
    source_state_sha256: str,
    expected_run_id: str,
    expected_source_run_id: str,
    expected_semantic_cache_run_id: str,
    expected_manifest_sha256: str,
    expected_source_config_sha256: str,
    expected_development_evidence_sha256: str,
    source_benchmark_commit: str,
    publication_benchmark_commit: str,
    recovery_confirmed: bool = False,
    validate_worktree: bool = False,
) -> dict[str, Any]:
    """Validate and publish the leaf-screen bundle under the shared lock."""

    root = project_root.expanduser().resolve()
    for name in (
        PUBLIC_PLOT_NAME,
        PUBLIC_CANDIDATE_NAME,
        PUBLIC_PROVENANCE_NAME,
    ):
        publication_path(output_dir / name, root)
    with publication_lock(root):
        if validate_worktree:
            public_paths = {
                f"methods/tls2trees/examples/{PUBLIC_PLOT_NAME}",
                f"methods/tls2trees/examples/{PUBLIC_CANDIDATE_NAME}",
                f"methods/tls2trees/examples/{PUBLIC_PROVENANCE_NAME}",
            }
            recovery_paths = set(public_paths)
            for path in public_paths:
                _, name = path.rsplit("/", 1)
                recovery_paths.add(
                    "methods/tls2trees/examples/"
                    f"{PUBLICATION_STAGE_NAME}/{name}"
                )
            validate_git_worktree(
                root,
                recovery_confirmed=recovery_confirmed,
                recovery_paths=recovery_paths,
                expected_head=publication_benchmark_commit,
            )
        return _finalise_locked(
            summary_path=summary_path,
            source_plot_csv=source_plot_csv,
            source_candidate_csv=source_candidate_csv,
            candidate_config_path=candidate_config_path,
            output_dir=output_dir,
            source_state_sha256=source_state_sha256,
            expected_run_id=expected_run_id,
            expected_source_run_id=expected_source_run_id,
            expected_semantic_cache_run_id=expected_semantic_cache_run_id,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_source_config_sha256=expected_source_config_sha256,
            expected_development_evidence_sha256=(
                expected_development_evidence_sha256
            ),
            source_benchmark_commit=source_benchmark_commit,
            publication_benchmark_commit=publication_benchmark_commit,
            publication_project_root=(root if validate_worktree else None),
            expected_publication_head=(
                publication_benchmark_commit if validate_worktree else None
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--source-plot-csv", required=True, type=Path)
    parser.add_argument("--source-candidate-csv", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--source-state-sha256", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-source-run-id", required=True)
    parser.add_argument("--expected-semantic-cache-run-id", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-config-sha256", required=True)
    parser.add_argument("--expected-development-evidence-sha256", required=True)
    parser.add_argument("--source-benchmark-commit", required=True)
    parser.add_argument("--publication-benchmark-commit", required=True)
    parser.add_argument("--recovery-confirmed", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    finalise(
        summary_path=args.summary_json,
        source_plot_csv=args.source_plot_csv,
        source_candidate_csv=args.source_candidate_csv,
        candidate_config_path=args.candidate_config,
        output_dir=args.output_dir,
        project_root=args.project_root,
        source_state_sha256=args.source_state_sha256,
        expected_run_id=args.expected_run_id,
        expected_source_run_id=args.expected_source_run_id,
        expected_semantic_cache_run_id=args.expected_semantic_cache_run_id,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_source_config_sha256=args.expected_source_config_sha256,
        expected_development_evidence_sha256=(
            args.expected_development_evidence_sha256
        ),
        source_benchmark_commit=args.source_benchmark_commit,
        publication_benchmark_commit=args.publication_benchmark_commit,
        recovery_confirmed=bool(args.recovery_confirmed),
        validate_worktree=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
