"""Shared helpers for immutable TLS2trees FOR-instance development runs."""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
EXPECTED_VARIANT = "published_default"
EXPECTED_SPLIT = "development"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.for_instance_manifest import load_and_verify_manifest_plot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def peak_rss_gb() -> float:
    self_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    child_rss = float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    # RUSAGE_CHILDREN captures the heavyweight inference subprocesses.  Taking
    # the maximum avoids double-counting parent and child peaks that need not
    # have occurred at the same time.
    rss = max(self_rss, child_rss)
    return rss / (1_000_000_000 if sys.platform == "darwin" else 1_000_000)


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TLS2trees config does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"TLS2trees config must contain a YAML mapping: {path}")
    if config.get("method", {}).get("variant") != EXPECTED_VARIANT:
        raise ValueError(f"Config is not the {EXPECTED_VARIANT} variant: {path}")
    return config, path


def resolve_plot_context(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str,
    split: str,
) -> tuple[Path, dict[str, Any]]:
    if variant != EXPECTED_VARIANT or split != EXPECTED_SPLIT:
        raise ValueError(
            "This runtime entrypoint only permits published_default development runs"
        )
    return resolve_development_plot_context(
        manifest_path=manifest_path,
        task_index=task_index,
        output_root=output_root,
        run_id=run_id,
        variant=variant,
        allowed_variants={EXPECTED_VARIANT},
    )


def resolve_development_plot_context(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str,
    allowed_variants: set[str],
) -> tuple[Path, dict[str, Any]]:
    """Resolve an immutable development plot under an explicit variant allow-list."""

    if variant not in allowed_variants:
        raise ValueError(
            f"Variant {variant!r} is not permitted; expected one of {sorted(allowed_variants)}"
        )
    manifest_path = manifest_path.expanduser().resolve()
    _, row = load_and_verify_manifest_plot(
        manifest_path,
        task_index=task_index,
        expected_split=EXPECTED_SPLIT,
    )
    safe_plot_id = str(row.get("safe_plot_id", ""))
    if not safe_plot_id or Path(safe_plot_id).name != safe_plot_id:
        raise ValueError(f"Unsafe or missing safe_plot_id: {safe_plot_id!r}")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"Unsafe run_id: {run_id!r}")
    plot_root = (
        output_root.expanduser().resolve()
        / "tls2trees"
        / "for_instance"
        / variant
        / EXPECTED_SPLIT
        / run_id
        / safe_plot_id
    )
    return plot_root, row


def resolve_held_out_test_plot_context(
    *,
    manifest_path: Path,
    task_index: int,
    output_root: Path,
    run_id: str,
    variant: str,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a plot only after a caller's explicit held-out-test gate."""

    allowed_variants = {"development_tuned", "published_default"}
    if variant not in allowed_variants:
        raise ValueError(
            "Held-out TLS2trees inference requires one of "
            f"{sorted(allowed_variants)}, received {variant!r}"
        )
    manifest_path = manifest_path.expanduser().resolve()
    _, row = load_and_verify_manifest_plot(
        manifest_path,
        task_index=task_index,
        expected_split="test",
        allow_held_out_test=True,
    )
    safe_plot_id = str(row.get("safe_plot_id", ""))
    if not safe_plot_id or Path(safe_plot_id).name != safe_plot_id:
        raise ValueError(f"Unsafe or missing safe_plot_id: {safe_plot_id!r}")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"Unsafe run_id: {run_id!r}")
    plot_root = (
        output_root.expanduser().resolve()
        / "tls2trees"
        / "for_instance"
        / variant
        / "test"
        / run_id
        / safe_plot_id
    )
    return plot_root, row


def verify_upstream(config: dict[str, Any], repo: Path) -> dict[str, str]:
    repo = repo.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"TLS2trees repository does not exist: {repo}")
    expected = str(config["method"]["executable_pin"]["commit"])
    actual = git_commit(repo)
    if actual != expected:
        raise RuntimeError(
            f"TLS2trees commit mismatch: expected {expected}, found {actual}"
        )
    model = repo / str(config["method"]["bundled_fsct_model"]["relative_path"])
    if not model.is_file():
        raise FileNotFoundError(f"Bundled TLS2trees model does not exist: {model}")
    expected_model = str(config["method"]["bundled_fsct_model"]["sha256"])
    actual_model = sha256(model)
    if actual_model != expected_model:
        raise RuntimeError(
            f"TLS2trees model checksum mismatch: expected {expected_model}, found {actual_model}"
        )
    return {
        "repo": str(repo),
        "expected_commit": expected,
        "actual_commit": actual,
        "model": str(model),
        "model_sha256": actual_model,
    }
