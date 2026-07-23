"""Run pinned upstream tools/test.py without modifying the upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import runpy
import subprocess
import sys
from pathlib import Path


SOURCE_COMMIT = "6a75c3735e4a4108d02ee944a8b93177f2360a4f"
TEST_SHA256 = "e05fd7a449d4fd4f2c1bb091e29d09fba21e2942b64790aed63b49f0bba51c96"
CONFIG_SHA256 = "cdf6bff5269dfbd73f4b4c7fe30deffdbfed7fd73668ca2f0bc5bb792f04ec1f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_root: Path, config: Path) -> Path:
    source_root = source_root.resolve()
    test_py = source_root / "tools/test.py"
    if subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT:
        raise ValueError("ForestFormer3D source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("ForestFormer3D source checkout is dirty")
    if sha256_file(test_py) != TEST_SHA256:
        raise ValueError("Official tools/test.py SHA-256 mismatch")
    if sha256_file(config.resolve()) != CONFIG_SHA256:
        raise ValueError("Official config SHA-256 mismatch")
    return test_py


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_py = verify_source(args.source_root, args.config)
    for path in (args.checkpoint, args.data_root / args.ann_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.work_dir.mkdir(parents=True, exist_ok=False)

    # The pinned official file uses torch.load/torch.save but omits import torch.
    # Supplying the missing global preserves the exact upstream file and model
    # path; no model logic, checkpoint tensor or configuration is patched.
    import torch

    sys.path.insert(0, str(args.source_root.resolve()))
    sys.argv = [
        str(test_py),
        str(args.config.resolve()),
        str(args.checkpoint.resolve()),
        "--work-dir",
        str(args.work_dir.resolve()),
        "--cfg-options",
        f"test_dataloader.dataset.data_root={args.data_root.resolve()}",
        f"test_dataloader.dataset.ann_file={args.ann_file}",
        "test_dataloader.batch_size=1",
        "test_dataloader.num_workers=0",
        "test_dataloader.persistent_workers=False",
        "randomness.seed=3407",
        "randomness.deterministic=True",
    ]
    os.chdir(args.source_root.resolve())
    runpy.run_path(
        str(test_py), run_name="__main__", init_globals={"torch": torch}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
