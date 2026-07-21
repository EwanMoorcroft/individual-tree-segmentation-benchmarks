"""Run pinned TLS2trees semantic inference with documented compatibility fixes."""

from __future__ import annotations

import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
LOCAL_SHIFT_TARGET = (
    "np.asarray(data.local_shift.cpu())[3 * batch:3 + (3 * batch)]"
)
LOCAL_SHIFT_REPLACEMENT = (
    "np.asarray(data.local_shift.cpu()).reshape(-1, 3)[int(batch)]"
)


def upstream_repo() -> Path:
    value = os.environ.get("TLS2TREES_REPO")
    return (
        Path(value).expanduser().resolve()
        if value
        else PROJECT_ROOT / "external" / "TLS2trees"
    )


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        # The published PyG stack requires CUDA scatter_add, which has no
        # deterministic implementation under PyTorch 1.9. Keep the recorded
        # seeds without imposing a non-upstream runtime constraint.
        torch.use_deterministic_algorithms(False)


def patch_pandas_append() -> None:
    import pandas as pd

    if not hasattr(pd.DataFrame, "append"):
        def append(frame, other, *args, **kwargs):  # type: ignore[no-untyped-def]
            ignore_index = bool(kwargs.get("ignore_index", False))
            return pd.concat([frame, other], ignore_index=ignore_index)

        pd.DataFrame.append = append  # type: ignore[attr-defined,method-assign]


def patched_inference_source(source: str) -> str:
    if source.count(LOCAL_SHIFT_TARGET) != 1:
        raise RuntimeError(
            "Expected one TLS2trees semantic local-shift patch target; verify the pinned commit"
        )
    return source.replace(LOCAL_SHIFT_TARGET, LOCAL_SHIFT_REPLACEMENT, 1)


def patch_inference(repo: Path) -> None:
    import tls2trees.fsct.inference as inference

    path = repo / "tls2trees" / "fsct" / "inference.py"
    patched = patched_inference_source(path.read_text(encoding="utf-8"))
    exec(compile(patched, str(path), "exec"), inference.__dict__)


def main() -> None:
    repo = upstream_repo()
    script = repo / "tls2trees" / "semantic.py"
    if not script.is_file():
        raise FileNotFoundError(f"Upstream TLS2trees semantic script not found: {script}")
    for entry in (repo, repo / "tls2trees"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    seed = int(os.environ.get("TLS2TREES_SEED", "42"))
    seed_everything(seed)
    patch_pandas_append()
    patch_inference(repo)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
