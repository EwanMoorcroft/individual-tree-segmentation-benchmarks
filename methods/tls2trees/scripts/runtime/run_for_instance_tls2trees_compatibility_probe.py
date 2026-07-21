"""Run a label-free TLS2trees development compatibility probe.

This entrypoint reuses checksum-verified published-default semantic tiles and
only asks whether a frozen instance configuration emits any leaf-off trees.
It never opens FOR-instance reference labels or evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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
    sha256,
    utc_now,
    verify_upstream,
    write_json,
)
from run_for_instance_tls2trees_instance import prediction_inventory
from run_tls2trees_instance_for_plot import build_command


PATCH_WRAPPER = RUNTIME / "patches" / "instance_patched.py"
NO_PREDICTIONS_PATTERN = "*.tls2trees_no_predictions.txt"
EXPECTED_VARIANT = "development_tuned"
EXPECTED_SPLIT = "development"
RUN_ID_PATTERN = re.compile(
    r"^tls2trees_for-instance_development_tuned_compatibility_probe_[0-9]{8}_[0-9]{6}$"
)


def load_probe_manifest(path_text: str) -> tuple[dict[str, Any], Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Compatibility-probe manifest does not exist: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Compatibility-probe manifest must contain a YAML mapping")
    scope = manifest.get("scope", {})
    required_false = (
        "held_out_test_accessed",
        "reference_labels_accessed",
        "accuracy_metrics_accessed",
        "selection_uses_accuracy_metrics",
        "benchmark_result",
    )
    if scope.get("variant") != EXPECTED_VARIANT or scope.get("split") != EXPECTED_SPLIT:
        raise ValueError("Compatibility probe only permits development_tuned/development")
    if scope.get("probe_target") != "leaf_off":
        raise ValueError("Compatibility probe must remain leaf_off-only")
    if any(scope.get(key) is not False for key in required_false):
        raise ValueError("Compatibility probe must not use labels, metrics, or held-out test data")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Compatibility-probe manifest has no candidates")
    ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if candidate.get("candidate_index") != index:
            raise ValueError("Candidate indexes must be contiguous and ordered from zero")
        candidate_id = str(candidate.get("candidate_id", ""))
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", candidate_id) or candidate_id in ids:
            raise ValueError(f"Unsafe or duplicate candidate_id: {candidate_id!r}")
        ids.add(candidate_id)
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("add_leaves") is not False:
            raise ValueError(f"Candidate {candidate_id} must set add_leaves=false")
        # build_command owns the complete parameter-name validation.
        build_command(Path("instance.py"), Path("tile.ply"), Path("index.dat"), Path("out"), parameters)
    return manifest, path


def verified_source(source_plot_root: Path) -> dict[str, Any]:
    source_plot_root = source_plot_root.expanduser().resolve()
    semantic_path = source_plot_root / "metadata" / "semantic_run.json"
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Published semantic metadata does not exist: {semantic_path}")
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    if (
        semantic.get("status") != "completed"
        or semantic.get("variant") != "published_default"
        or semantic.get("split") != EXPECTED_SPLIT
        or semantic.get("held_out_test_accessed") is not False
    ):
        raise ValueError("Source is not a completed published-default development semantic run")
    safe_plot_id = str(semantic.get("safe_plot_id", ""))
    if source_plot_root.name != safe_plot_id or Path(safe_plot_id).name != safe_plot_id:
        raise ValueError("Source semantic safe_plot_id does not match its immutable plot root")

    conversion_path = Path(str(semantic.get("conversion_metadata", ""))).resolve()
    if conversion_path != source_plot_root / "converted" / "conversion_metadata.json":
        raise ValueError("Source semantic metadata points outside its converted plot root")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    tile_index = Path(str(conversion.get("tile_index", ""))).resolve()
    if tile_index != source_plot_root / "converted" / "tile_index.dat":
        raise ValueError("Source tile index points outside its converted plot root")
    if not tile_index.is_file() or sha256(tile_index) != conversion.get("tile_index_sha256"):
        raise RuntimeError("Source tile index is missing or its checksum changed")

    output_records = semantic.get("outputs")
    if not isinstance(output_records, list) or not output_records:
        raise ValueError("Source semantic metadata has no outputs")
    outputs: list[Path] = []
    for record in output_records:
        path = Path(str(record.get("path", ""))).resolve()
        if path.parent != source_plot_root / "semantic":
            raise ValueError(f"Source semantic tile points outside its semantic root: {path}")
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Source semantic tile is missing or changed: {path}")
        outputs.append(path)
    return {
        "plot_root": source_plot_root,
        "semantic_path": semantic_path,
        "semantic": semantic,
        "conversion_path": conversion_path,
        "tile_index": tile_index,
        "outputs": outputs,
    }


def run_probe(
    *,
    source_plot_root: Path,
    output_root: Path,
    run_id: str,
    candidate_manifest: str,
    candidate_index: int,
    tls2trees_repo: Path,
    published_config: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Unsafe compatibility-probe run_id: {run_id!r}")
    manifest, manifest_path = load_probe_manifest(candidate_manifest)
    candidates = manifest["candidates"]
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise IndexError(f"Candidate index must be in 0..{len(candidates) - 1}")
    candidate = candidates[candidate_index]
    candidate_id = candidate["candidate_id"]
    parameters = dict(candidate["parameters"])
    source = verified_source(source_plot_root)
    semantic = source["semantic"]

    config, resolved_published_config = load_config(published_config)
    upstream = verify_upstream(config, tls2trees_repo)
    if upstream["actual_commit"] != manifest["source"]["upstream_commit"]:
        raise RuntimeError("Probe manifest and published config upstream commits differ")
    if upstream["model_sha256"] != manifest["source"]["model_sha256"]:
        raise RuntimeError("Probe manifest and published config model checksums differ")

    run_root = (
        output_root.expanduser().resolve()
        / "tls2trees"
        / "for_instance"
        / EXPECTED_VARIANT
        / EXPECTED_SPLIT
        / run_id
    )
    candidate_root = run_root / semantic["safe_plot_id"] / "compatibility_probe" / candidate_id
    if candidate_root.exists():
        raise FileExistsError(f"Compatibility-probe candidate already exists: {candidate_root}")
    raw_root = candidate_root / "predictions" / "raw"
    logs_root = candidate_root / "logs" / "instance"
    metadata_path = candidate_root / "metadata" / (
        "probe_dry_run.json" if dry_run else "probe_run.json"
    )
    commands = [
        build_command(PATCH_WRAPPER, tile, source["tile_index"], raw_root, parameters)
        for tile in source["outputs"]
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "running",
        "started_at_utc": utc_now(),
        "ended_at_utc": None,
        "dataset": "FOR-instance",
        "method": "TLS2trees",
        "variant": EXPECTED_VARIANT,
        "split": EXPECTED_SPLIT,
        "run_id": run_id,
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "safe_plot_id": semantic["safe_plot_id"],
        "relative_path": semantic["relative_path"],
        "hostname": platform.node(),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256(manifest_path),
        "published_config": str(resolved_published_config),
        "published_config_sha256": sha256(resolved_published_config),
        "source_plot_root": str(source["plot_root"]),
        "source_semantic_metadata": str(source["semantic_path"]),
        "source_semantic_metadata_sha256": sha256(source["semantic_path"]),
        "source_conversion_metadata": str(source["conversion_path"]),
        "source_conversion_metadata_sha256": sha256(source["conversion_path"]),
        "source_semantic_output_count": len(source["outputs"]),
        "tls2trees": upstream,
        "patch_wrapper": str(PATCH_WRAPPER),
        "patch_wrapper_sha256": sha256(PATCH_WRAPPER),
        "resolved_instance_parameters": parameters,
        "reproducibility_seed": manifest["candidate_generation"]["seed"],
        "commands": commands,
        "prediction_inventory": {"leaf_off": [], "leaf_on": []},
        "no_prediction_evidence": [],
        "runtime_seconds": None,
        "peak_rss_gb": None,
        "return_code": None,
        "reference_labels_accessed": False,
        "accuracy_metrics_accessed": False,
        "selection_uses_accuracy_metrics": False,
        "held_out_test_accessed": False,
        "benchmark_result": False,
    }
    if dry_run:
        payload.update(
            {
                "ended_at_utc": utc_now(),
                "runtime_seconds": round(time.perf_counter() - started, 6),
                "peak_rss_gb": peak_rss_gb(),
                "return_code": 0,
            }
        )
        write_json(metadata_path, payload)
        return payload

    raw_root.mkdir(parents=True)
    logs_root.mkdir(parents=True)
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
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Probe tile {index} failed with return code {completed.returncode}; see {stderr_path}"
                )
        inventory = prediction_inventory(raw_root)
        if inventory["leaf_on"]:
            raise RuntimeError("Leaf-on output appeared despite add_leaves=false")
        evidence: list[dict[str, str]] = []
        for path in sorted(raw_root.rglob(NO_PREDICTIONS_PATTERN)):
            reason = path.read_text(encoding="utf-8").strip()
            if reason != "no_graph_connected_stem_bases":
                raise ValueError(f"Unexpected no-predictions reason in {path}: {reason}")
            evidence.append({"path": str(path), "reason": reason, "sha256": sha256(path)})
        if not inventory["leaf_off"] and len(evidence) != len(commands):
            raise RuntimeError(
                "Probe emitted no predictions without one audited empty-graph record per tile"
            )
        payload["prediction_inventory"] = inventory
        payload["no_prediction_evidence"] = evidence
        payload.update(
            {
                "status": "viable_nonempty" if inventory["leaf_off"] else "completed_no_predictions",
                "return_code": 0,
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "return_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
            }
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
    parser.add_argument("--source-plot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-index", required=True, type=int)
    parser.add_argument("--tls2trees-repo", required=True)
    parser.add_argument(
        "--published-config",
        default="methods/tls2trees/configs/for_instance_published_default.yml",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = run_probe(
            source_plot_root=Path(args.source_plot_root),
            output_root=Path(args.output_root),
            run_id=args.run_id,
            candidate_manifest=args.candidate_manifest,
            candidate_index=args.candidate_index,
            tls2trees_repo=Path(args.tls2trees_repo),
            published_config=args.published_config,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"status={payload['status']}")
    print(f"candidate_id={payload['candidate_id']}")
    print(f"leaf_off_predictions={len(payload['prediction_inventory']['leaf_off'])}")
    print("accuracy_metrics_accessed=false")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
