"""Stage the official training YAML for the pinned Hydra 1.0.7 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_OFFICIAL_CONFIG_SHA256 = (
    "7f491d8e4060974fafba1401ac38cec23c3476160decefd64ce60099c230ae96"
)
REFERENCE_LINE = (
    "# Ref: https://github.com/chrischoy/"
    "SpatioTemporalSegmentation/blob/master/config.py"
)
PACKAGE_LINE = "# @package training"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(
    official_config: Path,
    output_yaml: Path,
    metadata_json: Path,
) -> dict[str, object]:
    if output_yaml.exists() or metadata_json.exists():
        raise FileExistsError("refusing to overwrite staged training configuration")
    if sha256(official_config) != EXPECTED_OFFICIAL_CONFIG_SHA256:
        raise ValueError("official training configuration identity changed")
    source = official_config.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    if (
        len(lines) < 3
        or lines[0].rstrip("\r\n") != REFERENCE_LINE
        or lines[1].rstrip("\r\n") != PACKAGE_LINE
    ):
        raise ValueError("official training configuration header changed")
    staged = "".join([lines[1], lines[0], *lines[2:]])
    source_lines = source.splitlines()
    staged_lines = staged.splitlines()
    differences = [
        index + 1
        for index, (before, after) in enumerate(zip(source_lines, staged_lines))
        if before != after
    ]
    if differences != [1, 2] or len(source_lines) != len(staged_lines):
        raise ValueError(
            f"staged training configuration changed unexpected lines: {differences}"
        )
    if staged_lines[0] != PACKAGE_LINE:
        raise ValueError("Hydra package directive was not moved to line one")
    if staged_lines[2:] != source_lines[2:]:
        raise ValueError("staged training values differ from the official file")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(staged, encoding="utf-8")
    payload = {
        "schema": "forainet_finetune_training_config_stage_v1",
        "status": "verified",
        "official_config_sha256": EXPECTED_OFFICIAL_CONFIG_SHA256,
        "staged_config_sha256": sha256(output_yaml),
        "changed_lines": [1, 2],
        "change": "hydra_package_directive_moved_to_line_one_only",
        "training_values_changed": False,
        "runtime_hydra_version": "1.0.7",
        "held_out_access": False,
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-config", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()
    payload = stage(
        args.official_config,
        args.output_yaml,
        args.metadata_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
