#!/usr/bin/env python3
"""Audit ForestFormer3D checkpoint exposure against the frozen 32-plot subset.

The official inventory is a public filename-only snapshot obtained from the
Zenodo archive previews. No FOR-instanceV2 point-cloud payload is required.
Matching is deliberately exact: every original relative path has exactly three
possible official member names, one for each of ``train``, ``val`` and ``test``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import EXPECTED_PATHS


ZENODO_RECORD = "https://doi.org/10.5281/zenodo.16742708"
NORMALISATION_RULE = (
    "exact:<collection>/<stem>.las->"
    "<collection>_<collection>_<stem>_<official-role>.ply"
)
OFFICIAL_ROLES = ("train", "val", "test")
INVENTORY_COLUMNS = {
    "archive",
    "member_path",
    "official_role",
    "retrieved_at",
    "source_url",
}
OUTPUT_COLUMNS = (
    "original_relative_path",
    "original_split",
    "official_archive",
    "official_member_path",
    "official_role",
    "normalisation_rule",
    "match_count",
    "checkpoint_train_val_exposed",
    "held_out_absent_from_train_val",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class InventoryRow:
    archive: str
    member_path: str
    official_role: str
    retrieved_at: str
    source_url: str


@dataclass(frozen=True)
class ExposureRow:
    original_relative_path: str
    original_split: str
    official_archive: str
    official_member_path: str
    official_role: str
    normalisation_rule: str
    match_count: int
    checkpoint_train_val_exposed: bool
    held_out_absent_from_train_val: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_member_name(relative_path: str, official_role: str) -> str:
    """Return the only permitted V2 member name for a frozen original path."""

    if official_role not in OFFICIAL_ROLES:
        raise ValueError(f"Unsupported official role: {official_role!r}")
    path = Path(relative_path)
    if (
        len(path.parts) != 2
        or path.suffix.casefold() != ".las"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Non-canonical original relative path: {relative_path!r}")
    collection = path.parts[0]
    return f"{collection}_{collection}_{path.stem}_{official_role}.ply"


def read_inventory(path: Path) -> tuple[list[InventoryRow], str]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Inventory snapshot does not exist: {source}")
    digest = sha256_file(source)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = INVENTORY_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
        rows = [
            InventoryRow(
                archive=(row["archive"] or "").strip(),
                member_path=(row["member_path"] or "").strip(),
                official_role=(row["official_role"] or "").strip(),
                retrieved_at=(row["retrieved_at"] or "").strip(),
                source_url=(row["source_url"] or "").strip(),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError("Inventory snapshot contains no rows")
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if row.official_role not in OFFICIAL_ROLES:
            raise ValueError(f"Invalid official role: {row}")
        if not row.member_path.endswith(f"_{row.official_role}.ply"):
            raise ValueError(f"Role/member mismatch: {row}")
        if row.archive not in {"train_val_data.zip", "test_data.zip"}:
            raise ValueError(f"Unexpected archive: {row.archive!r}")
        expected_archive = (
            "test_data.zip" if row.official_role == "test" else "train_val_data.zip"
        )
        if row.archive != expected_archive:
            raise ValueError(f"Archive/role mismatch: {row}")
        if not row.retrieved_at or not row.source_url.startswith("https://zenodo.org/"):
            raise ValueError(f"Missing public source metadata: {row}")
        identity = (row.archive, row.member_path)
        if identity in identities:
            raise ValueError(f"Duplicate official inventory member: {identity}")
        identities.add(identity)
    return rows, digest


def build_exposure_rows(inventory: Sequence[InventoryRow]) -> list[ExposureRow]:
    by_member: dict[str, list[InventoryRow]] = {}
    for row in inventory:
        by_member.setdefault(row.member_path, []).append(row)

    output: list[ExposureRow] = []
    for original_split in ("dev", "test"):
        for relative_path in EXPECTED_PATHS[original_split]:
            candidates: list[InventoryRow] = []
            for role in OFFICIAL_ROLES:
                candidates.extend(
                    by_member.get(expected_member_name(relative_path, role), [])
                )
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one official match for {relative_path}, "
                    f"found {len(candidates)}"
                )
            match = candidates[0]
            exposed = match.official_role in {"train", "val"}
            output.append(
                ExposureRow(
                    original_relative_path=relative_path,
                    original_split=original_split,
                    official_archive=match.archive,
                    official_member_path=match.member_path,
                    official_role=match.official_role,
                    normalisation_rule=NORMALISATION_RULE,
                    match_count=1,
                    checkpoint_train_val_exposed=exposed,
                    held_out_absent_from_train_val=(
                        original_split != "test" or not exposed
                    ),
                )
            )
    if len(output) != 32:
        raise RuntimeError(f"Frozen audit produced {len(output)} rows, expected 32")
    if len({row.original_relative_path for row in output}) != 32:
        raise RuntimeError("Frozen audit output contains duplicate original paths")
    return output


def summarise(
    rows: Sequence[ExposureRow],
    *,
    inventory_sha256: str,
    inventory_path: Path,
    manual_review_status: str,
) -> dict[str, object]:
    if not _SHA256_RE.fullmatch(inventory_sha256):
        raise ValueError("Inventory SHA-256 is invalid")
    role_counts = Counter(row.official_role for row in rows)
    held_out = [row for row in rows if row.original_split == "test"]
    development = [row for row in rows if row.original_split == "dev"]
    held_out_train_val_matches = [
        row.original_relative_path
        for row in held_out
        if row.checkpoint_train_val_exposed
    ]
    unmatched_or_multiple = [
        row.original_relative_path for row in rows if row.match_count != 1
    ]
    decision = (
        "eligible_exposure_gate_passed"
        if not held_out_train_val_matches
        and not unmatched_or_multiple
        and len(held_out) == 11
        and len(development) == 21
        else "ineligible_or_unresolved"
    )
    return {
        "protocol_id": "forestformer3d_checkpoint_exposure_v1",
        "checkpoint_record": ZENODO_RECORD,
        "inventory_snapshot": inventory_path.as_posix(),
        "inventory_snapshot_sha256": inventory_sha256,
        "normalisation_rule": NORMALISATION_RULE,
        "row_count": len(rows),
        "development_row_count": len(development),
        "held_out_row_count": len(held_out),
        "official_role_counts": dict(sorted(role_counts.items())),
        "development_train_val_exposed_count": sum(
            row.checkpoint_train_val_exposed for row in development
        ),
        "held_out_train_val_match_count": len(held_out_train_val_matches),
        "held_out_train_val_matches": held_out_train_val_matches,
        "unmatched_or_multiple": unmatched_or_multiple,
        "manual_review_status": manual_review_status,
        "decision": decision,
    }


def write_csv(path: Path, rows: Iterable[ExposureRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow(
                {
                    key: (
                        str(payload[key]).lower()
                        if isinstance(payload[key], bool)
                        else payload[key]
                    )
                    for key in OUTPUT_COLUMNS
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument(
        "--manual-review-status",
        default="not_reviewed",
        choices=("not_reviewed", "reviewed_20260723"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_csv.exists() or args.output_summary.exists():
        raise FileExistsError("Exposure audit outputs must not overwrite existing files")
    inventory, inventory_sha256 = read_inventory(args.inventory)
    rows = build_exposure_rows(inventory)
    summary = summarise(
        rows,
        inventory_sha256=inventory_sha256,
        inventory_path=args.inventory,
        manual_review_status=args.manual_review_status,
    )
    write_csv(args.output_csv, rows)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if summary["decision"] != "eligible_exposure_gate_passed":
        raise RuntimeError("ForestFormer3D checkpoint exposure gate did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
