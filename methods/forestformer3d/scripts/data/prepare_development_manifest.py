"""Freeze the exact ForestFormer3D 21-plot development manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import (  # noqa: E402
    build_exact_split_manifest,
    sha256_file,
)

SMOKE_POINT_COUNT = 1_816_672
SMOKE_WALL_SECONDS = 143.053
CONSERVATIVE_BYTES_PER_POINT = 128


def build(dataset_root: Path) -> dict[str, object]:
    manifest = build_exact_split_manifest(
        dataset_root,
        dataset_root / "data_split_metadata.csv",
        split="development",
        allow_held_out_test=False,
    )
    total_points = sum(int(row["point_count"]) for row in manifest["plots"])
    estimated_gpu_seconds = (
        total_points * SMOKE_WALL_SECONDS / SMOKE_POINT_COUNT
    )
    manifest.update(
        {
            "schema": "forestformer3d_development_manifest_v1",
            "method": "ForestFormer3D",
            "training_mode": "published_pretrained",
            "held_out_access": False,
            "total_point_count": total_points,
            "total_reference_tree_count": sum(
                int(row["reference_tree_count"]) for row in manifest["plots"]
            ),
            "resource_estimate": {
                "basis": (
                    "successful CULS/plot_1 official inference: "
                    f"{SMOKE_POINT_COUNT} points in {SMOKE_WALL_SECONDS} seconds"
                ),
                "estimated_total_gpu_seconds_by_point_scaling": estimated_gpu_seconds,
                "estimated_serial_gpu_hours": estimated_gpu_seconds / 3600,
                "estimated_two_way_array_hours_excluding_queue": (
                    estimated_gpu_seconds / 7200
                ),
                "conservative_retained_bytes_per_point": (
                    CONSERVATIVE_BYTES_PER_POINT
                ),
                "conservative_retained_bytes": (
                    total_points * CONSERVATIVE_BYTES_PER_POINT
                ),
                "conservative_retained_gib": (
                    total_points * CONSERVATIVE_BYTES_PER_POINT / 1024**3
                ),
                "array_task_time_limit_hours": 2,
                "maximum_concurrent_gpu_tasks": 2,
            },
        }
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing manifest: {output}")
    payload = build(args.dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest={output}")
    print(f"manifest_sha256={sha256_file(output)}")
    print(f"plots={len(payload['plots'])}")
    print(f"points={payload['total_point_count']}")
    print(
        "estimated_retained_gib="
        f"{math.ceil(payload['resource_estimate']['conservative_retained_gib'] * 10) / 10}"
    )
    print("held_out_access=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
