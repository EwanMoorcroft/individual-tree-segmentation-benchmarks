"""Reuse one held-out semantic cache only after exact provenance verification.

Exit status 3 means the cache is not reusable and the caller must run the
dedicated conversion and semantic path.  No destination is created in that
case.  Other non-zero statuses are workflow errors and must fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path(__file__).resolve().parent
for entry in (ROOT, RUNTIME):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from for_instance_published_common import (  # noqa: E402
    resolve_held_out_test_plot_context,
    sha256,
    utc_now,
    verify_upstream,
    write_json,
)
from published_default_test_common import (  # noqa: E402
    load_json,
    validate_exact_manifest,
    validate_frozen_configuration,
)
from shared.for_instance_manifest import load_and_verify_manifest_plot  # noqa: E402


class CacheNotReusable(RuntimeError):
    """Expected cache mismatch which authorises dedicated semantic inference."""


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise CacheNotReusable(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise CacheNotReusable(
            f"{label} checksum mismatch: expected {expected}, found {actual}"
        )


def verify_cache(
    *,
    manifest_path: Path,
    source_manifest_path: Path,
    task_index: int,
    source_output_root: Path,
    source_run_id: str,
    source_variant: str,
    output_root: Path,
    run_id: str,
    workflow_config_path: Path,
    published_config_path: Path,
    tls2trees_repo: Path,
    source_state_path: Path,
    source_state_sha256: str,
) -> dict[str, Any]:
    workflow, workflow_path, published, published_path = (
        validate_frozen_configuration(
            workflow_config_path, published_config_path
        )
    )
    upstream = verify_upstream(published, tls2trees_repo)
    manifest_path = manifest_path.expanduser().resolve()
    source_manifest_path = source_manifest_path.expanduser().resolve()
    source_state_path = source_state_path.expanduser().resolve()
    require_file_hash(source_state_path, source_state_sha256, "source state")
    if sha256(manifest_path) != sha256(source_manifest_path):
        raise CacheNotReusable("current and source manifests are not byte-identical")
    manifest, row = load_and_verify_manifest_plot(
        manifest_path,
        task_index=task_index,
        expected_split="test",
        allow_held_out_test=True,
    )
    source_manifest, source_row = load_and_verify_manifest_plot(
        source_manifest_path,
        task_index=task_index,
        expected_split="test",
        allow_held_out_test=True,
    )
    validate_exact_manifest(manifest, workflow)
    validate_exact_manifest(source_manifest, workflow)
    row_keys = (
        "task_index",
        "safe_plot_id",
        "relative_path",
        "input_las",
        "input_sha256",
        "point_count",
        "reference_tree_count",
    )
    if any(row.get(key) != source_row.get(key) for key in row_keys):
        raise CacheNotReusable("source cache plot differs from the current manifest")
    if source_variant not in {"development_tuned", "published_default"}:
        raise CacheNotReusable(f"unsupported semantic-cache variant: {source_variant}")
    source_plot_root = (
        source_output_root.expanduser().resolve()
        / "tls2trees"
        / "for_instance"
        / source_variant
        / "test"
        / source_run_id
        / row["safe_plot_id"]
    )
    conversion_path = source_plot_root / "converted" / "conversion_metadata.json"
    semantic_path = source_plot_root / "metadata" / "semantic_run.json"
    if not conversion_path.is_file() or not semantic_path.is_file():
        raise CacheNotReusable("source cache conversion or semantic metadata is missing")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    published_sha256 = sha256(published_path)
    source_manifest_sha256 = sha256(source_manifest_path)
    expected_conversion = {
        "split": "test",
        "task_index": task_index,
        "safe_plot_id": row["safe_plot_id"],
        "relative_path": row["relative_path"],
        "input_sha256": row["input_sha256"],
        "manifest_sha256": source_manifest_sha256,
        "labels_stripped": True,
    }
    for key, expected in expected_conversion.items():
        if conversion.get(key) != expected:
            raise CacheNotReusable(f"conversion metadata mismatch for {key}")
    if (
        float(conversion.get("tile_size_m", -1))
        != float(published["published_preprocessing"]["tile_edge_length_m"])
        or float(conversion.get("downsample_voxel_size_m", -1))
        != float(
            published["published_preprocessing"]["downsample_voxel_length_m"]
        )
    ):
        raise CacheNotReusable("source conversion parameters are not published defaults")
    for key, label in (
        ("source_map", "source map"),
        ("tile_index", "tile index"),
    ):
        require_file_hash(
            Path(conversion[key]), conversion[f"{key}_sha256"], label
        )
    for record in conversion.get("tiles", []):
        require_file_hash(Path(record["path"]), record["sha256"], "converted tile")
    if not conversion.get("tiles"):
        raise CacheNotReusable("source conversion contains no tiles")
    if (
        semantic.get("status") != "completed"
        or semantic.get("split") != "test"
        or int(semantic.get("task_index", -1)) != task_index
        or semantic.get("safe_plot_id") != row["safe_plot_id"]
        or semantic.get("relative_path") != row["relative_path"]
        or semantic.get("config_sha256") != published_sha256
        or semantic.get("held_out_test_accessed") is not True
        or semantic.get("tls2trees", {}).get("actual_commit")
        != upstream["actual_commit"]
        or semantic.get("tls2trees", {}).get("model_sha256")
        != upstream["model_sha256"]
    ):
        raise CacheNotReusable("semantic metadata provenance is not an exact match")
    outputs = semantic.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise CacheNotReusable("semantic cache contains no outputs")
    for record in outputs:
        require_file_hash(Path(record["path"]), record["sha256"], "semantic output")

    destination, destination_row = resolve_held_out_test_plot_context(
        manifest_path=manifest_path,
        task_index=task_index,
        output_root=output_root,
        run_id=run_id,
        variant="published_default",
    )
    if destination_row["input_sha256"] != row["input_sha256"]:
        raise CacheNotReusable("destination input hash differs from source cache")
    if destination.exists():
        raise FileExistsError(f"Immutable destination already exists: {destination}")

    destination.mkdir(parents=True)
    (destination / "converted").symlink_to(
        source_plot_root / "converted", target_is_directory=True
    )
    (destination / "semantic").symlink_to(
        source_plot_root / "semantic", target_is_directory=True
    )
    metadata_root = destination / "metadata"
    metadata_root.mkdir()
    (metadata_root / "semantic_run.json").symlink_to(semantic_path)
    evidence = {
        "schema_version": 1,
        "status": "semantic_cache_reused",
        "created_at_utc": utc_now(),
        "dataset": "FOR-instance",
        "variant": "published_default",
        "split": "test",
        "run_id": run_id,
        "task_index": task_index,
        "safe_plot_id": row["safe_plot_id"],
        "relative_path": row["relative_path"],
        "source_variant": source_variant,
        "source_run_id": source_run_id,
        "source_plot_root": str(source_plot_root),
        "source_state": str(source_state_path),
        "source_state_sha256": source_state_sha256,
        "manifest_sha256": sha256(manifest_path),
        "input_las_sha256": row["input_sha256"],
        "workflow_config": str(workflow_path),
        "workflow_config_sha256": sha256(workflow_path),
        "published_config": str(published_path),
        "published_config_sha256": published_sha256,
        "bundled_model_sha256": upstream["model_sha256"],
        "conversion_metadata_sha256": sha256(conversion_path),
        "conversion_tile_index_sha256": conversion["tile_index_sha256"],
        "conversion_source_map_sha256": conversion["source_map_sha256"],
        "semantic_metadata_sha256": sha256(semantic_path),
        "semantic_output_sha256": [record["sha256"] for record in outputs],
        "held_out_test_accessed": True,
        "inference_rerun": False,
    }
    write_json(metadata_root / "semantic_cache_reuse.json", evidence)
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--source-manifest-json", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-variant", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-config", required=True)
    parser.add_argument("--published-config", required=True)
    parser.add_argument("--tls2trees-repo", required=True)
    parser.add_argument("--source-state-file", required=True)
    parser.add_argument("--source-state-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = verify_cache(
            manifest_path=Path(args.manifest_json),
            source_manifest_path=Path(args.source_manifest_json),
            task_index=args.task_index,
            source_output_root=Path(args.source_output_root),
            source_run_id=args.source_run_id,
            source_variant=args.source_variant,
            output_root=Path(args.output_root),
            run_id=args.run_id,
            workflow_config_path=Path(args.workflow_config),
            published_config_path=Path(args.published_config),
            tls2trees_repo=Path(args.tls2trees_repo),
            source_state_path=Path(args.source_state_file),
            source_state_sha256=args.source_state_sha256,
        )
    except CacheNotReusable as exc:
        print(f"cache_reusable=false reason={exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"cache_reusable=false reason={type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("status=published_default_semantic_cache_reused")
    print(f"task_index={evidence['task_index']}")
    print(f"safe_plot_id={evidence['safe_plot_id']}")
    print("inference_rerun=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
