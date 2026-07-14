"""Launch pinned upstream TreeLearn training after fixing process RNG seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import runpy
import sys
from pathlib import Path

from consolidate_for_instance_finetune_long_crops import verify_consolidated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--environment-record", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--crop-inventory", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    crop_integrity = verify_consolidated(
        args.freeze.expanduser().resolve(),
        args.crop_inventory.expanduser().resolve(),
    )
    repo = args.treelearn_repo.expanduser().resolve()
    entrypoint = repo / "tools" / "training" / "train.py"
    if not entrypoint.is_file() or not args.config.is_file():
        raise FileNotFoundError(entrypoint if not entrypoint.is_file() else args.config)

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    original_listdir = os.listdir
    os.listdir = lambda path: sorted(original_listdir(path))
    sys.path.insert(0, str(repo))
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    environment_record = args.environment_record.expanduser().resolve()
    if environment_record.exists():
        raise FileExistsError(environment_record)
    try:
        import spconv

        spconv_version = getattr(spconv, "__version__", "unknown")
    except ImportError:
        spconv_version = "unavailable"
    environment_record.parent.mkdir(parents=True, exist_ok=True)
    environment_record.write_text(json.dumps({
        "schema_version": 1,
        "status": "treelearn_seeded_training_environment_frozen",
        "seed": args.seed,
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "spconv": spconv_version,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "training_config": str(args.config.resolve()),
        "training_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "crop_inventory": crop_integrity["inventory"],
        "crop_inventory_sha256": crop_integrity["inventory_sha256"],
        "crop_entries_aggregate_sha256": crop_integrity[
            "entries_aggregate_sha256"
        ],
        "crop_count_verified": crop_integrity["crop_count"],
        "crop_referenced_size_bytes_verified": crop_integrity[
            "referenced_size_bytes"
        ],
        "bitwise_determinism_guaranteed": False,
        "determinism_note": (
            "Seeds, sorted input paths and deterministic cuDNN settings are fixed; "
            "pinned sparse CUDA kernels may remain nondeterministic."
        ),
    }, indent=2, sort_keys=True) + "\n")
    sys.argv = [
        str(entrypoint),
        "--config", str(args.config.resolve()),
        "--work_dir", str(work_dir),
    ]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
