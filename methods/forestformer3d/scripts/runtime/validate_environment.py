#!/usr/bin/env python3
"""Validate the pinned ForestFormer3D image, source and checkpoint on a GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SOURCE_COMMIT = "6a75c3735e4a4108d02ee944a8b93177f2360a4f"
CHECKPOINT_SHA256 = (
    "01037a648596832238ac72ea2f5eef87ceaf5aeb399e56ff4b760ba1ed1c777e"
)
SOURCE_HASHES = {
    "Dockerfile": "ffe98b014c3dfac69a1ecc939aca475180921569766fb6ad4691a6621277efd6",
    "configs/oneformer3d_qs_radius16_qp300_2many.py": (
        "cdf6bff5269dfbd73f4b4c7fe30deffdbfed7fd73668ca2f0bc5bb792f04ec1f"
    ),
    "replace_mmdetection_files/loops.py": (
        "df3b0d6688ae4f911fa6cbe8b1afb90520b1d147b3da800ee22075993a0bae27"
    ),
    "replace_mmdetection_files/base_model.py": (
        "9fb88239dd8eeddadbe6c909dc6bd5d613d3bbd487e272592972c856e56e233d"
    ),
    "replace_mmdetection_files/transforms_3d.py": (
        "c1a34b5a2ce006739fd1b810fdbe9cfc12f4b443acf389df71c6228aad690be9"
    ),
    "tools/test.py": (
        "e05fd7a449d4fd4f2c1bb091e29d09fba21e2942b64790aed63b49f0bba51c96"
    ),
    "tools/train.py": (
        "abb05dd13bf249638695427b91ea72dc1916858764c88f87a7b3e6dee928b9a0"
    ),
}
INSTALLED_REPLACEMENTS = {
    "venv/lib/python3.10/site-packages/mmengine/runner/loops.py": (
        SOURCE_HASHES["replace_mmdetection_files/loops.py"]
    ),
    "venv/lib/python3.10/site-packages/mmengine/model/base_model/base_model.py": (
        SOURCE_HASHES["replace_mmdetection_files/base_model.py"]
    ),
    "venv/lib/python3.10/site-packages/mmdet3d/datasets/transforms/transforms_3d.py": (
        SOURCE_HASHES["replace_mmdetection_files/transforms_3d.py"]
    ),
}
IMPORTS = (
    "torch",
    "mmengine",
    "mmdet",
    "mmseg",
    "mmdet3d",
    "spconv",
    "MinkowskiEngine",
    "torch_points_kernels",
    "torch_cluster",
    "oneformer3d",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hashes(root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected_digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
        digest = sha256_file(path)
        if digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_digest}, found {digest}"
            )
        observed[relative] = digest
    return observed


def module_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in IMPORTS:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "imported_no_version"))
    return versions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    environment_root = args.environment_root.resolve()
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Official ForestFormer3D checkpoint SHA-256 mismatch")

    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError(f"Unexpected ForestFormer3D source state: {commit}, dirty={bool(dirty)}")

    source_hashes = validate_hashes(source_root, SOURCE_HASHES)
    if Path(sys.prefix).resolve() != (environment_root / "venv").resolve():
        raise ValueError(
            f"Unexpected Python prefix: expected {environment_root / 'venv'}, "
            f"found {sys.prefix}"
        )
    if not (environment_root / "environment_build.complete").is_file():
        raise FileNotFoundError("Rootless environment completion marker is missing")

    installed_hashes = validate_hashes(environment_root, INSTALLED_REPLACEMENTS)
    versions = module_versions()

    import torch

    cuda_available = bool(torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
    device = None
    capability = None
    if cuda_available:
        device = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
        if args.require_cuda and capability != [8, 0]:
            raise RuntimeError(f"Expected A100 compute capability [8, 0], found {capability}")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, dict):
        raise TypeError("Official checkpoint did not load as a mapping")

    record = {
        "status": "forestformer3d_environment_validated",
        "source_commit": commit,
        "source_clean": True,
        "environment_root": str(environment_root),
        "python_prefix": sys.prefix,
        "source_hashes": source_hashes,
        "installed_replacement_hashes": installed_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_top_level_keys": sorted(str(key) for key in checkpoint_payload),
        "versions": versions,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": device,
        "compute_capability": capability,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
