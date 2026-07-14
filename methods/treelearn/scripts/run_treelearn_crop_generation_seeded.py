"""Launch upstream TreeLearn crop generation with a frozen per-plot seed."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treelearn-repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    repo = args.treelearn_repo.expanduser().resolve()
    entrypoint = repo / "tools" / "data_gen" / "gen_train_data.py"
    if not entrypoint.is_file() or not args.config.is_file():
        raise FileNotFoundError(entrypoint if not entrypoint.is_file() else args.config)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)
    original_listdir = os.listdir
    os.listdir = lambda path: sorted(original_listdir(path))
    sys.path.insert(0, str(repo))
    sys.argv = [str(entrypoint), "--config", str(args.config.resolve())]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
