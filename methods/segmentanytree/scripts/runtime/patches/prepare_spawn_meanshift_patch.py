"""Prepare a SegmentAnyTree MeanShift helper using a persistent spawn pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMPORT_ANCHOR = "from functools import partial\n"
POOL_EXPRESSION = "multiprocessing.Pool(processes=processes)"
MEANSHIFT_EXPRESSION = "MeanShift(bandwidth=bandwidth,bin_seeding=True)"
SPAWN_POOL = """

_spawn_pool = None
_spawn_pool_size = 0


class _PersistentSpawnPool:
    def __init__(self, processes):
        global _spawn_pool
        global _spawn_pool_size

        requested_size = max(1, int(processes))
        if _spawn_pool is None or requested_size > _spawn_pool_size:
            if _spawn_pool is not None:
                _spawn_pool.close()
                _spawn_pool.join()
            context = multiprocessing.get_context("spawn")
            _spawn_pool = context.Pool(
                processes=requested_size,
                maxtasksperchild=1000,
            )
            _spawn_pool_size = requested_size

    def __enter__(self):
        return _spawn_pool

    def __exit__(self, exc_type, exc_value, traceback):
        return False
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def patch_source(source: str) -> str:
    if source.count(IMPORT_ANCHOR) != 1:
        raise ValueError("Expected the pinned MeanShift import anchor.")
    if source.count(MEANSHIFT_EXPRESSION) != 1:
        raise ValueError("Expected the pinned MeanShift constructor.")
    pool_count = source.count(POOL_EXPRESSION)
    if pool_count != 2:
        raise ValueError(
            f"Expected two nested process pools in the pinned helper, found {pool_count}"
        )
    patched = source.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR + "import os\n" + SPAWN_POOL,
    )
    patched = patched.replace(
        POOL_EXPRESSION,
        "_PersistentSpawnPool(processes)",
    )
    return patched.replace(
        MEANSHIFT_EXPRESSION,
        (
            "MeanShift("
            "bandwidth=bandwidth,"
            "bin_seeding=True,"
            'n_jobs=int(os.environ.get("SEGMENTANYTREE_MEANSHIFT_JOBS", "2"))'
            ")"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace nested MeanShift fork pools with a persistent spawn pool."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    source = source_path.read_text(encoding="utf-8")
    patched = patch_source(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")

    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "source_sha256": sha256_text(source),
                    "output": str(output_path),
                    "output_sha256": sha256_text(patched),
                    "nested_process_pools_disabled": True,
                    "persistent_spawn_pool_enabled": True,
                    "meanshift_jobs_from_environment": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
