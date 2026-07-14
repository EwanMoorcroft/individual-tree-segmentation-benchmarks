"""Validate the pinned TreeLearn environment and editable checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Iterable


EXPECTED_PYTHON = (3, 10)
EXPECTED_SETUPTOOLS = "80.9.0"
EXPECTED_TORCH = "2.0.0"
EXPECTED_TORCH_CUDA = "11.8"


def validate_package_locations(repo: Path, locations: Iterable[str]) -> list[Path]:
    """Accept regular or namespace packages only when resolved under the pin."""

    repo = repo.resolve()
    resolved = [Path(location).resolve() for location in locations]
    if not resolved or not all(location.is_relative_to(repo) for location in resolved):
        raise ValueError(
            f"tree_learn package locations {resolved} are not under pinned repo {repo}"
        )
    return resolved


def treelearn_package_locations(repo: Path) -> list[Path]:
    spec = importlib.util.find_spec("tree_learn")
    if spec is None:
        raise ModuleNotFoundError("tree_learn is not importable")
    return validate_package_locations(repo, spec.submodule_search_locations or ())


def validate_environment(repo: Path, require_cuda: bool) -> dict[str, object]:
    setuptools_version = version("setuptools")
    if setuptools_version != EXPECTED_SETUPTOOLS:
        raise RuntimeError(
            f"Expected setuptools 80.9.0, found {setuptools_version}"
        )
    import pkg_resources  # noqa: F401
    import open3d
    import spconv
    import torch
    import tree_learn  # noqa: F401

    locations = treelearn_package_locations(repo)
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python 3.10, found {sys.version.split()[0]}"
        )
    if torch.__version__.split("+")[0] != EXPECTED_TORCH:
        raise RuntimeError(f"Expected PyTorch 2.0.0, found {torch.__version__}")
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"Expected PyTorch CUDA 11.8, found {torch.version.cuda}"
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active TreeLearn environment")
    return {
        "python": sys.version.split()[0],
        "setuptools": setuptools_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "spconv": getattr(spconv, "__version__", "unknown"),
        "open3d": open3d.__version__,
        "treelearn_package_locations": [str(location) for location in locations],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treelearn-repo", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate_environment(
        Path(args.treelearn_repo).expanduser(),
        require_cuda=args.require_cuda,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
