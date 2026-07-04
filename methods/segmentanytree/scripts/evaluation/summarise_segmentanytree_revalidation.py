"""Summarise SegmentAnyTree export audits and internal-output inventories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
FIELDS = [
    "collection",
    "plot_name",
    "export_status",
    "safe_for_final_accuracy_evaluation",
    "point_count_delta",
    "coordinate_multiset_equal",
    "instance_candidate_count",
    "semantic_candidate_count",
    "checkpoint_count",
    "checkpoint_sha256",
    "instance_candidates",
    "semantic_candidates",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def collect(metadata_root: Path) -> list[dict[str, Any]]:
    audit_root = metadata_root / "export_validation"
    inventory_root = metadata_root / "internal_output_inventory"
    keys = {
        path.relative_to(root).with_suffix("")
        for root in (audit_root, inventory_root)
        if root.is_dir()
        for path in root.rglob("*.json")
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(keys):
        audit_path = audit_root / key.with_suffix(".json")
        inventory_path = inventory_root / key.with_suffix(".json")
        audit = read_json(audit_path) if audit_path.is_file() else {}
        inventory = read_json(inventory_path) if inventory_path.is_file() else {}
        instance_candidates = inventory.get("instance_candidates", [])
        semantic_candidates = inventory.get("semantic_candidates", [])
        checkpoints = inventory.get("checkpoint_files", [])
        rows.append(
            {
                "collection": key.parts[0] if len(key.parts) > 1 else "",
                "plot_name": key.name,
                "export_status": audit.get("status", "missing"),
                "safe_for_final_accuracy_evaluation": audit.get(
                    "safe_for_final_accuracy_evaluation", ""
                ),
                "point_count_delta": audit.get("point_count_delta", ""),
                "coordinate_multiset_equal": audit.get(
                    "coordinate_multiset_equal", ""
                ),
                "instance_candidate_count": len(instance_candidates),
                "semantic_candidate_count": len(semantic_candidates),
                "checkpoint_count": len(checkpoints),
                "checkpoint_sha256": ";".join(
                    sorted(
                        {
                            str(checkpoint.get("sha256", ""))
                            for checkpoint in checkpoints
                            if checkpoint.get("sha256")
                        }
                    )
                ),
                "instance_candidates": ";".join(instance_candidates),
                "semantic_candidates": ";".join(semantic_candidates),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise SegmentAnyTree export validation and internal aligned "
            "output discovery."
        )
    )
    parser.add_argument(
        "--metadata-root",
        default="results/metadata/segmentanytree_for_instance",
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "results/tables/segmentanytree_for_instance/"
            "revalidation_diagnostics.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect(resolve_path(args.metadata_root))
    if not rows:
        raise SystemExit("No audit or internal-output inventory JSON files found.")
    output_path = resolve_path(args.output_csv)
    write_csv(output_path, rows)
    safe = sum(row["safe_for_final_accuracy_evaluation"] is True for row in rows)
    paired = sum(
        row["instance_candidate_count"] == 1
        and row["semantic_candidate_count"] == 1
        for row in rows
    )
    checkpoint_hashes = {
        row["checkpoint_sha256"]
        for row in rows
        if row["checkpoint_sha256"]
    }
    print(f"Plots inspected: {len(rows)}")
    print(f"Exports safe for direct evaluation: {safe}")
    print(f"Plots with one internal instance/semantic pair: {paired}")
    print(f"Distinct recorded checkpoint hashes: {len(checkpoint_hashes)}")
    print(f"Output: {output_path}")
    for row in rows:
        print(
            f"{row['collection']}/{row['plot_name']}: "
            f"export={row['export_status']} "
            f"delta={row['point_count_delta']} "
            f"internal={row['instance_candidate_count']}/"
            f"{row['semantic_candidate_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
