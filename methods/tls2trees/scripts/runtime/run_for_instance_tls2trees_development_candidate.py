"""Run one frozen TLS2trees development candidate on one Stage 0 plot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path(__file__).resolve().parent
SRC = ROOT / "src"
for entry in (RUNTIME, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from for_instance_published_common import (
    load_config,
    peak_rss_gb,
    resolve_development_plot_context,
    resolve_held_out_test_plot_context,
    sha256,
    utc_now,
    verify_upstream,
    write_json,
)
from run_for_instance_tls2trees_instance import prediction_inventory
from run_tls2trees_instance_for_plot import build_command


PATCH_WRAPPER = RUNTIME / "patches" / "instance_patched.py"
NO_PREDICTIONS_PATTERN = "*.tls2trees_no_predictions.txt"
NO_PREDICTIONS_REASONS = {
    "no_clustered_wood_convex_hulls",
    "no_graph_connected_stem_bases",
    "no_in_tile_stem_predictions",
}
EXPECTED_VARIANT = "development_tuned"
EXPECTED_SPLIT = "development"


def load_stage1_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage 1 config must contain a YAML mapping")
    if payload.get("method", {}).get("variant") != EXPECTED_VARIANT:
        raise ValueError("Stage 1 config is not development_tuned")
    scope = payload.get("scope", {})
    if scope.get("held_out_test_accessed") is not False:
        raise ValueError("Stage 1 config crossed the held-out-test boundary")
    if scope.get("final_configuration_selected") is not False:
        raise ValueError("Stage 1 config must not preselect a final configuration")
    if payload.get("candidate_generation", {}).get(
        "ordering_frozen_before_stage1_metrics"
    ) is not True:
        raise ValueError("Stage 1 candidate ordering is not frozen")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Stage 1 config has no candidates")
    for index, candidate in enumerate(candidates):
        if candidate.get("candidate_index") != index:
            raise ValueError("Stage 1 candidate indexes must be contiguous")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", candidate_id):
            raise ValueError(f"Unsafe Stage 1 candidate ID: {candidate_id!r}")
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("add_leaves") is not True:
            raise ValueError(f"Stage 1 candidate {candidate_id} must enable leaves")
        build_command(Path("instance.py"), Path("tile.ply"), Path("index.dat"), Path("out"), parameters)
    return payload, path


def verify_probe_evidence(
    probe_summary_path: Path,
    expected_sha256: str,
    required_ids: list[str],
) -> dict[str, Any]:
    probe_summary_path = probe_summary_path.expanduser().resolve()
    if not probe_summary_path.is_file() or sha256(probe_summary_path) != expected_sha256:
        raise RuntimeError("Probe summary is missing or its checksum changed")
    payload = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "viable_candidates_found":
        raise ValueError("Probe did not complete with viable candidates")
    if payload.get("viable_candidate_ids") != required_ids:
        raise ValueError("Probe viable candidates differ from the frozen Stage 1 promotion")
    if payload.get("held_out_test_accessed") is not False:
        raise ValueError("Probe summary crossed the held-out-test boundary")
    return payload


LEAF_ATTACHMENT_PARAMETER_NAMES = {
    "add_leaves_voxel_length",
    "add_leaves_edge_length",
}


def verify_development_stage1_evidence(
    summary_path: Path,
    expected_sha256: str,
    candidate_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a development-only source summary and the derived leaf grid.

    The source Stage 1 accuracy is not used to invent this grid.  Its summary is
    retained here only as immutable evidence for the p02 stem parameters and the
    semantic-cache run which the leaf screen reuses.
    """

    summary_path = summary_path.expanduser().resolve()
    if not summary_path.is_file() or sha256(summary_path) != expected_sha256:
        raise RuntimeError(
            "Development Stage 1 summary is missing or its checksum changed"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = candidate_config.get("development_evidence", {})
    required_valid_metrics = int(evidence.get("required_valid_metric_count", -1))
    if (
        payload.get("status") != evidence.get("required_summary_status")
        or int(payload.get("valid_metric_count", -1)) != required_valid_metrics
        or int(payload.get("expected_metric_count", -1)) != required_valid_metrics
        or payload.get("held_out_test_accessed") is not False
        or payload.get("final_configuration_selected") is not False
        or payload.get("split") != "development"
    ):
        raise ValueError(
            "Development Stage 1 evidence is incomplete or crossed the test boundary"
        )

    source_candidate_id = evidence.get("source_candidate_id")
    source_parameters = payload.get("candidate_parameters", {}).get(
        source_candidate_id
    )
    if not isinstance(source_parameters, dict):
        raise ValueError(
            "Development Stage 1 evidence has no source-candidate parameters"
        )
    if source_parameters.get("add_leaves") is not True:
        raise ValueError("Leaf-screen source candidate did not enable leaf attachment")

    grid = candidate_config.get("leaf_attachment_grid", {})
    voxel_values = [float(value) for value in grid.get("voxel_length_m", [])]
    edge_values = [float(value) for value in grid.get("edge_length_m", [])]
    expected_pairs = {
        (voxel_length, edge_length)
        for voxel_length in voxel_values
        for edge_length in edge_values
    }
    observed_pairs: set[tuple[float, float]] = set()
    source_non_leaf = {
        key: value
        for key, value in source_parameters.items()
        if key not in LEAF_ATTACHMENT_PARAMETER_NAMES
    }
    for candidate in candidate_config["candidates"]:
        parameters = candidate["parameters"]
        candidate_non_leaf = {
            key: value
            for key, value in parameters.items()
            if key not in LEAF_ATTACHMENT_PARAMETER_NAMES
        }
        if candidate_non_leaf != source_non_leaf:
            raise ValueError(
                f"Leaf-screen candidate {candidate['candidate_id']} changed a p02 stem parameter"
            )
        observed_pairs.add(
            (
                float(parameters["add_leaves_voxel_length"]),
                float(parameters["add_leaves_edge_length"]),
            )
        )
    if not voxel_values or not edge_values or observed_pairs != expected_pairs:
        raise ValueError("Leaf-screen candidates do not cover the exact declared grid")
    if len(candidate_config["candidates"]) != len(expected_pairs):
        raise ValueError("Leaf-screen grid contains duplicate candidates")
    return payload


def verified_semantic_cache(
    source_plot_root: Path,
    row: dict[str, Any],
    *,
    expected_split: str = EXPECTED_SPLIT,
) -> dict[str, Any]:
    root = source_plot_root.expanduser().resolve()
    if root.name != row["safe_plot_id"]:
        raise ValueError("Semantic-cache plot ID differs from the manifest row")
    conversion_path = root / "converted" / "conversion_metadata.json"
    semantic_path = root / "metadata" / "semantic_run.json"
    if not conversion_path.is_file() or not semantic_path.is_file():
        raise FileNotFoundError(f"Incomplete semantic cache: {root}")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    if (
        conversion.get("variant") != EXPECTED_VARIANT
        or conversion.get("split") != expected_split
        or conversion.get("labels_stripped") is not True
        or int(conversion.get("task_index", -1)) != int(row["task_index"])
    ):
        raise ValueError("Semantic-cache conversion has the wrong provenance")
    if (
        semantic.get("status") != "completed"
        or semantic.get("variant") != EXPECTED_VARIANT
        or semantic.get("split") != expected_split
        or semantic.get("held_out_test_accessed") != (expected_split == "test")
        or int(semantic.get("task_index", -1)) != int(row["task_index"])
    ):
        raise ValueError("Semantic cache is not a completed development_tuned run")
    tile_index = Path(conversion["tile_index"])
    if not tile_index.is_file() or sha256(tile_index) != conversion["tile_index_sha256"]:
        raise RuntimeError("Semantic-cache tile index changed")
    outputs: list[Path] = []
    for record in semantic.get("outputs", []):
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Semantic-cache output changed: {path}")
        outputs.append(path)
    if not outputs:
        raise ValueError("Semantic cache contains no outputs")
    return {
        "root": root,
        "conversion_path": conversion_path,
        "semantic_path": semantic_path,
        "tile_index": tile_index,
        "outputs": outputs,
    }


AUDITED_INSTANCE_FAILURES = {
    "small_graph": "Expected n_neighbors <= n_samples",
    "empty_groupby": "Cannot restore clstr: groupby.apply did not return a grouped index",
    "empty_in_tile_stems": "cannot set a frame with no defined index and a scalar",
}


def archive_failed_audited_instance_attempt(
    plot_root: Path,
    *,
    expected_failure_kind: str | None = None,
) -> dict[str, Any]:
    metadata_path = plot_root / "metadata" / "instance_run.json"
    raw_root = plot_root / "predictions" / "raw"
    logs_root = plot_root / "logs" / "instance"
    if not metadata_path.is_file() or not raw_root.is_dir() or not logs_root.is_dir():
        raise FileNotFoundError(
            "Audited recovery requires failed instance metadata, raw root, and logs"
        )
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    if previous.get("status") != "failed" or "Instance tile" not in str(previous.get("error", "")):
        raise ValueError("Audited recovery requires failed instance-tile metadata")
    tile_errors = sorted(logs_root.glob("tile_*.stderr.log"))
    combined_errors = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in tile_errors
    )
    failure_kinds = [
        kind
        for kind, signature in AUDITED_INSTANCE_FAILURES.items()
        if signature in combined_errors
    ]
    if len(failure_kinds) != 1:
        raise ValueError(
            "Recovery requires exactly one audited instance failure signature"
        )
    failure_kind = failure_kinds[0]
    if expected_failure_kind is not None and failure_kind != expected_failure_kind:
        raise ValueError(
            f"Expected {expected_failure_kind} recovery, found {failure_kind}"
        )
    if (plot_root / "predictions" / "aligned").exists() or (plot_root / "evaluation").exists():
        raise RuntimeError("Refusing recovery after alignment or evaluation output exists")
    archive_parent = plot_root / "recovery"
    attempt = 1
    archive = archive_parent / f"instance_{failure_kind}_attempt_{attempt}"
    while archive.exists():
        attempt += 1
        archive = archive_parent / f"instance_{failure_kind}_attempt_{attempt}"
    archive.mkdir(parents=True)
    previous_sha256 = sha256(metadata_path)
    shutil.move(str(raw_root), str(archive / "raw"))
    shutil.move(str(logs_root), str(archive / "logs"))
    shutil.move(str(metadata_path), str(archive / "instance_run.json"))
    return {
        "status": f"failed_{failure_kind}_attempt_archived",
        "failure_kind": failure_kind,
        "attempt": attempt,
        "archive_root": str(archive),
        "previous_metadata_sha256": previous_sha256,
        "previous_error": previous.get("error"),
    }


def archive_failed_small_graph_attempt(plot_root: Path) -> dict[str, Any]:
    return archive_failed_audited_instance_attempt(
        plot_root,
        expected_failure_kind="small_graph",
    )


def run_candidate(
    *,
    manifest_path: Path,
    task_index: int,
    source_plot_root: Path,
    output_root: Path,
    workflow_run_id: str,
    candidate_run_id: str,
    candidate_index: int,
    stage1_config_path: str,
    tls2trees_repo: Path,
    probe_summary_path: Path | None = None,
    probe_summary_sha256: str | None = None,
    development_evidence_path: Path | None = None,
    development_evidence_sha256: str | None = None,
    resume_failed_small_graph: bool = False,
    resume_failed_audited_instance: bool = False,
    split: str = EXPECTED_SPLIT,
    target: str | None = None,
    final_selection_path: Path | None = None,
    final_selection_sha256: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stage1, resolved_stage1 = load_stage1_config(stage1_config_path)
    candidates = stage1["candidates"]
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError(f"Candidate index must be in 0..{len(candidates) - 1}")
    candidate = candidates[candidate_index]
    development_evidence: dict[str, Any] | None = None
    if "development_evidence" in stage1:
        if (
            split != EXPECTED_SPLIT
            or stage1.get("dataset", {}).get("allowed_split") != EXPECTED_SPLIT
        ):
            raise PermissionError(
                "Development-derived candidate config cannot access the held-out test"
            )
        if probe_summary_path is not None or probe_summary_sha256 is not None:
            raise ValueError(
                "Development-derived candidate config cannot use compatibility-probe evidence"
            )
        if development_evidence_path is None or development_evidence_sha256 is None:
            raise ValueError(
                "Development-derived candidate config requires source Stage 1 evidence"
            )
        development_evidence = verify_development_stage1_evidence(
            development_evidence_path,
            development_evidence_sha256,
            stage1,
        )
    elif "probe_promotion" in stage1:
        if probe_summary_path is None or probe_summary_sha256 is None:
            raise ValueError("Stage 1 candidate requires compatibility-probe evidence")
        if development_evidence_path is not None or development_evidence_sha256 is not None:
            raise ValueError(
                "Probe-promoted candidate config cannot use development-summary evidence"
            )
        required_ids = stage1["probe_promotion"]["required_viable_candidate_ids"]
        verify_probe_evidence(probe_summary_path, probe_summary_sha256, required_ids)
        if candidate["candidate_id"] not in required_ids:
            raise ValueError("Candidate was not promoted by the compatibility probe")
    else:
        raise ValueError("Candidate config has no accepted provenance evidence")
    is_held_out_test = split == "test"
    if is_held_out_test:
        if target not in {"leaf_off", "leaf_on"}:
            raise ValueError("Held-out candidate requires leaf_off or leaf_on target")
        if final_selection_path is None or final_selection_sha256 is None:
            raise ValueError("Held-out candidate requires the reviewed final selection")
        final_selection_path = final_selection_path.expanduser().resolve()
        if (
            not final_selection_path.is_file()
            or sha256(final_selection_path) != final_selection_sha256
        ):
            raise RuntimeError("Final selection is missing or its checksum changed")
        final_selection = json.loads(final_selection_path.read_text(encoding="utf-8"))
        selected = final_selection.get("selected_by_target", {}).get(target, {})
        if (
            final_selection.get("status") != "development_tuned_configuration_frozen"
            or final_selection.get("held_out_test_accessed") is not False
            or final_selection.get("final_configuration_selected") is not True
            or final_selection.get("review_required_before_held_out_test") is not True
            or selected.get("candidate_id") != candidate["candidate_id"]
            or int(selected.get("stage1_candidate_index", -1)) != candidate_index
            or selected.get("parameters") != candidate["parameters"]
            or final_selection.get("source_stage1_config_sha256") != sha256(resolved_stage1)
        ):
            raise ValueError("Candidate does not match the reviewed target-specific freeze")
        plot_root, row = resolve_held_out_test_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=candidate_run_id,
            variant=EXPECTED_VARIANT,
        )
    elif split == EXPECTED_SPLIT:
        plot_root, row = resolve_development_plot_context(
            manifest_path=manifest_path,
            task_index=task_index,
            output_root=output_root,
            run_id=candidate_run_id,
            variant=EXPECTED_VARIANT,
            allowed_variants={EXPECTED_VARIANT},
        )
    else:
        raise ValueError(f"Unsupported split: {split!r}")
    cache = verified_semantic_cache(
        source_plot_root, row, expected_split=split
    )
    published_config, resolved_published = load_config(stage1["method"]["semantic_config"])
    upstream = verify_upstream(published_config, tls2trees_repo)

    recovery: dict[str, Any] | None = None
    if resume_failed_small_graph and resume_failed_audited_instance:
        raise ValueError("Choose only one audited instance recovery mode")
    if resume_failed_small_graph or resume_failed_audited_instance:
        if not plot_root.is_dir():
            raise FileNotFoundError(f"Failed candidate plot root does not exist: {plot_root}")
        if (plot_root / "converted").resolve() != (cache["root"] / "converted").resolve():
            raise RuntimeError("Candidate conversion link differs from the semantic cache")
        if (plot_root / "semantic").resolve() != (cache["root"] / "semantic").resolve():
            raise RuntimeError("Candidate semantic link differs from the semantic cache")
        recovery = (
            archive_failed_small_graph_attempt(plot_root)
            if resume_failed_small_graph
            else archive_failed_audited_instance_attempt(plot_root)
        )
    else:
        if plot_root.exists():
            raise FileExistsError(f"Candidate plot root already exists: {plot_root}")
        plot_root.mkdir(parents=True)
        (plot_root / "converted").symlink_to(cache["root"] / "converted", target_is_directory=True)
        (plot_root / "semantic").symlink_to(cache["root"] / "semantic", target_is_directory=True)
        metadata_root = plot_root / "metadata"
        metadata_root.mkdir()
        (metadata_root / "semantic_run.json").symlink_to(cache["semantic_path"])
    metadata_root = plot_root / "metadata"
    raw_root = plot_root / "predictions" / "raw"
    logs_root = plot_root / "logs" / "instance"
    raw_root.mkdir(parents=True)
    logs_root.mkdir(parents=True)

    parameters = dict(candidate["parameters"])
    commands = [
        build_command(PATCH_WRAPPER, tile, cache["tile_index"], raw_root, parameters)
        for tile in cache["outputs"]
    ]
    metadata_path = metadata_root / "instance_run.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": EXPECTED_VARIANT,
        "split": split,
        "workflow_run_id": workflow_run_id,
        "run_id": candidate_run_id,
        "candidate_index": candidate_index,
        "candidate_id": candidate["candidate_id"],
        "target": target,
        "task_index": task_index,
        "relative_path": row["relative_path"],
        "safe_plot_id": row["safe_plot_id"],
        "collection": row["collection"],
        "hostname": platform.node(),
        "stage1_config": str(resolved_stage1),
        "stage1_config_sha256": sha256(resolved_stage1),
        "probe_summary": (
            str(probe_summary_path.resolve()) if probe_summary_path else None
        ),
        "probe_summary_sha256": probe_summary_sha256,
        "development_evidence": (
            str(development_evidence_path.resolve())
            if development_evidence_path
            else None
        ),
        "development_evidence_sha256": development_evidence_sha256,
        "development_evidence_run_id": (
            development_evidence.get("workflow_run_id")
            if development_evidence
            else None
        ),
        "semantic_cache_plot_root": str(cache["root"]),
        "semantic_metadata_sha256": sha256(cache["semantic_path"]),
        "published_semantic_config": str(resolved_published),
        "tls2trees": upstream,
        "compatibility_patches": [
            "pandas_groupby_apply_clstr_restore",
            "empty_groupby_apply_recorded_as_no_predictions",
            "parsed_leaf_graph_edge_length",
            "empty_graph_sources_recorded_as_no_predictions",
            "empty_in_tile_stems_recorded_as_no_predictions",
            "empty_leaf_tip_graph_preserves_stem_only_leaf_on_predictions",
            "small_wood_graph_neighbours_capped_to_available_samples",
            "deterministic_numpy_and_python_seed",
        ],
        "resolved_instance_parameters": parameters,
        "reproducibility_seed": stage1["candidate_generation"]["seed"],
        "commands": commands,
        "raw_prediction_root": str(raw_root),
        "prediction_inventory": {"leaf_off": [], "leaf_on": []},
        "no_prediction_evidence": [],
        "return_code": None,
        "runtime_seconds": None,
        "peak_rss_gb": None,
        "development_accuracy_metrics_accessed_by_instance_stage": False,
        "held_out_test_accessed": is_held_out_test,
        "final_selection": str(final_selection_path) if final_selection_path else None,
        "final_selection_sha256": final_selection_sha256,
        "recovery": recovery,
    }
    environment = os.environ.copy()
    environment["TLS2TREES_REPO"] = upstream["repo"]
    environment["TLS2TREES_SEED"] = str(payload["reproducibility_seed"])
    environment["PYTHONHASHSEED"] = str(payload["reproducibility_seed"])
    python_paths = [upstream["repo"], str(Path(upstream["repo"]) / "tls2trees")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    try:
        for index, command in enumerate(commands):
            stdout_path = logs_root / f"tile_{index:06d}.stdout.log"
            stderr_path = logs_root / f"tile_{index:06d}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=upstream["repo"],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    shell=False,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Instance tile {index} failed with return code {completed.returncode}; see {stderr_path}"
                )
        inventory = prediction_inventory(raw_root)
        evidence: list[dict[str, str]] = []
        for path in sorted(raw_root.rglob(NO_PREDICTIONS_PATTERN)):
            reason = path.read_text(encoding="utf-8").strip()
            if reason not in NO_PREDICTIONS_REASONS:
                raise ValueError(f"Unexpected no-predictions reason: {reason}")
            evidence.append({"path": str(path), "reason": reason, "sha256": sha256(path)})
        if len(inventory["leaf_off"]) != len(inventory["leaf_on"]):
            raise ValueError("Leaf-off and leaf-on prediction counts differ with add_leaves=true")
        if not inventory["leaf_off"] and len(evidence) != len(commands):
            raise RuntimeError("No predictions without one audited empty-graph record per tile")
        payload["prediction_inventory"] = inventory
        payload["no_prediction_evidence"] = evidence
        payload.update(
            {
                "status": "completed" if inventory["leaf_off"] else "completed_no_predictions",
                "return_code": 0,
            }
        )
    except Exception as exc:
        payload.update(
            {"status": "failed", "return_code": 1, "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
    finally:
        payload["ended_at_utc"] = utc_now()
        payload["runtime_seconds"] = round(time.perf_counter() - started, 6)
        payload["peak_rss_gb"] = peak_rss_gb()
        write_json(metadata_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--source-plot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--candidate-index", required=True, type=int)
    parser.add_argument(
        "--stage1-config", "--candidate-config", dest="stage1_config", required=True
    )
    parser.add_argument("--probe-summary-json")
    parser.add_argument("--probe-summary-sha256")
    parser.add_argument("--development-evidence-json")
    parser.add_argument("--development-evidence-sha256")
    parser.add_argument("--tls2trees-repo", required=True)
    parser.add_argument("--resume-failed-small-graph", action="store_true")
    parser.add_argument("--resume-failed-audited-instance", action="store_true")
    parser.add_argument("--split", choices=("development", "test"), default=EXPECTED_SPLIT)
    parser.add_argument("--target", choices=("leaf_off", "leaf_on"))
    parser.add_argument("--final-selection-json")
    parser.add_argument("--final-selection-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = run_candidate(
            manifest_path=Path(args.manifest_json),
            task_index=args.task_index,
            source_plot_root=Path(args.source_plot_root),
            output_root=Path(args.output_root),
            workflow_run_id=args.workflow_run_id,
            candidate_run_id=args.candidate_run_id,
            candidate_index=args.candidate_index,
            stage1_config_path=args.stage1_config,
            probe_summary_path=(
                Path(args.probe_summary_json) if args.probe_summary_json else None
            ),
            probe_summary_sha256=args.probe_summary_sha256,
            development_evidence_path=(
                Path(args.development_evidence_json)
                if args.development_evidence_json
                else None
            ),
            development_evidence_sha256=args.development_evidence_sha256,
            tls2trees_repo=Path(args.tls2trees_repo),
            resume_failed_small_graph=args.resume_failed_small_graph,
            resume_failed_audited_instance=args.resume_failed_audited_instance,
            split=args.split,
            target=args.target,
            final_selection_path=(
                Path(args.final_selection_json) if args.final_selection_json else None
            ),
            final_selection_sha256=args.final_selection_sha256,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print(f"candidate_id={payload['candidate_id']}")
    print(f"leaf_off_predictions={len(payload['prediction_inventory']['leaf_off'])}")
    print(f"leaf_on_predictions={len(payload['prediction_inventory']['leaf_on'])}")
    print(
        "held_out_test_accessed="
        + str(bool(payload["held_out_test_accessed"])).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
