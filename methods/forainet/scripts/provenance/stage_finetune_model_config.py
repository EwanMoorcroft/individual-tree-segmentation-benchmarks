"""Stage the official model YAML with only its author-local checkpoint path changed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_OFFICIAL_CONFIG_SHA256 = (
    "cfa698af22a4f545436aaf7285d46bc1d3690044c4d4de9db40bcac8ed90c2f5"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
)
PATH_PATTERN = re.compile(r'(?m)^  path_pretrained: "[^"\r\n]*"$')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(
    official_config: Path,
    checkpoint: Path,
    output_yaml: Path,
    metadata_json: Path,
) -> dict[str, object]:
    if output_yaml.exists() or metadata_json.exists():
        raise FileExistsError("refusing to overwrite staged model configuration")
    if sha256(official_config) != EXPECTED_OFFICIAL_CONFIG_SHA256:
        raise ValueError("official model configuration identity changed")
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("official initial checkpoint identity changed")
    source = official_config.read_text(encoding="utf-8")
    matches = PATH_PATTERN.findall(source)
    if len(matches) != 1:
        raise ValueError("official model configuration path is not unique")
    replacement = f"  path_pretrained: {json.dumps(str(checkpoint.resolve()))}"
    staged = PATH_PATTERN.sub(replacement, source, count=1)
    if staged == source:
        raise ValueError("staged model configuration did not change")
    source_lines = source.splitlines()
    staged_lines = staged.splitlines()
    differences = [
        index + 1
        for index, (before, after) in enumerate(zip(source_lines, staged_lines))
        if before != after
    ]
    if differences != [177] or len(source_lines) != len(staged_lines):
        raise ValueError(
            f"staged model configuration changed unexpected lines: {differences}"
        )
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(staged, encoding="utf-8")
    payload = {
        "schema": "forainet_finetune_model_config_stage_v1",
        "status": "verified",
        "official_config_sha256": EXPECTED_OFFICIAL_CONFIG_SHA256,
        "staged_config_sha256": sha256(output_yaml),
        "changed_lines": [177],
        "change": "models.PointGroup-PAPER.path_pretrained_only",
        "initial_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "initial_weight_name": "latest",
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-yaml", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    args = parser.parse_args()
    payload = stage(
        args.official_config,
        args.checkpoint,
        args.output_yaml,
        args.metadata_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
