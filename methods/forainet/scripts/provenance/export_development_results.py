"""Export a completed ForAINet development result through the public-safe gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


EXPECTED_PLOTS = 21
EXPECTED_PROTOCOL = "for_instance_pointwise_v1"
EXPECTED_UPSTREAM_COMMIT = "5fe600ae8f2fe913ae8740f475f0261a702f2a72"
EXPECTED_CHECKPOINT_SHA256 = (
    "97c03ce81621dc4193e55d2ca2294861b1f4421c94d192799e5fe031f9d35861"
)
EXPECTED_IMAGE_SHA256 = (
    "ad0df684209014c52421dc213cd0e15ddbb47214c00fac264e829f68dc17812d"
)
PRIVATE_MARKERS = (
    str(Path("/", "users")) + "/",
    str(Path("/", "mnt")) + "/",
    str(Path("/", "home")) + "/",
    "fastscratch",
    "barkla",
    "slurm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_public_text(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    if marker := next((value for value in PRIVATE_MARKERS if value in text), None):
        raise ValueError(f"private marker {marker!r} found in {path.name}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export(run_root: Path, output_root: Path) -> dict[str, Any]:
    final_gate_path = run_root / "final_gate.json"
    metrics_path = run_root / "summary" / "metrics.json"
    plots_path = run_root / "summary" / "plots.csv"
    sites_path = run_root / "summary" / "sites.csv"
    retention_path = run_root / "retention" / "manifest.json"
    sources = (
        final_gate_path,
        metrics_path,
        plots_path,
        sites_path,
        retention_path,
    )
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        ensure_public_text(path)

    final_gate = load_json(final_gate_path)
    metrics = load_json(metrics_path)
    retention = load_json(retention_path)
    run_id = str(final_gate.get("run_id", ""))
    benchmark_commit = str(final_gate.get("benchmark_commit", ""))
    if (
        final_gate.get("schema") != "forainet_development_final_gate_v1"
        or final_gate.get("status") != "complete"
        or final_gate.get("variant") != "published_pretrained"
        or final_gate.get("expected_plots") != EXPECTED_PLOTS
        or final_gate.get("completed_plots") != EXPECTED_PLOTS
        or final_gate.get("protocol_id") != EXPECTED_PROTOCOL
        or final_gate.get("held_out_access") is not False
        or final_gate.get("summary_metrics_sha256") != sha256(metrics_path)
        or final_gate.get("retention_manifest_sha256") != sha256(retention_path)
    ):
        raise ValueError("development final gate is incomplete or inconsistent")
    if (
        metrics.get("schema") != "forainet_development_summary_v1"
        or metrics.get("status") != "complete"
        or metrics.get("run_id") != run_id
        or metrics.get("benchmark_commit") != benchmark_commit
        or metrics.get("variant") != "published_pretrained"
        or metrics.get("split") != "dev"
        or metrics.get("protocol_id") != EXPECTED_PROTOCOL
        or metrics.get("expected_plots") != EXPECTED_PLOTS
        or metrics.get("completed_plots") != EXPECTED_PLOTS
        or metrics.get("held_out_access") is not False
    ):
        raise ValueError("development summary is incomplete or inconsistent")
    plot_rows = csv_rows(plots_path)
    site_rows = csv_rows(sites_path)
    if (
        len(plot_rows) != EXPECTED_PLOTS
        or any(Path(row["relative_path"]).is_absolute() for row in plot_rows)
        or len({int(row["task_index"]) for row in plot_rows}) != EXPECTED_PLOTS
        or not site_rows
    ):
        raise ValueError("public development tables are incomplete or unsafe")
    if (
        retention.get("schema") != "forainet_development_retention_v1"
        or retention.get("status") != "complete"
        or retention.get("run_id") != run_id
        or retention.get("held_out_access") is not False
        or retention.get("plot_summary_sha256") != sha256(plots_path)
        or retention.get("site_summary_sha256") != sha256(sites_path)
        or retention.get("metrics_summary_sha256") != sha256(metrics_path)
        or len(retention.get("child_manifests", [])) != EXPECTED_PLOTS
    ):
        raise ValueError("development retention summary is incomplete")

    outputs = {
        "plots": output_root / "forainet_development_plot_results.csv",
        "sites": output_root / "forainet_development_site_results.csv",
        "metrics": output_root / "forainet_development_results.json",
        "provenance": output_root / "forainet_development_provenance.json",
    }
    if existing := [str(path) for path in outputs.values() if path.exists()]:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plots_path, outputs["plots"])
    shutil.copyfile(sites_path, outputs["sites"])
    shutil.copyfile(metrics_path, outputs["metrics"])
    provenance = {
        "schema": "forainet_public_development_provenance_v1",
        "status": "complete",
        "run_id": run_id,
        "benchmark_commit": benchmark_commit,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "image_sha256": EXPECTED_IMAGE_SHA256,
        "protocol_id": EXPECTED_PROTOCOL,
        "split": "dev",
        "plot_count": EXPECTED_PLOTS,
        "held_out_access": False,
        "ranking_eligible": False,
        "exclusion_reason": (
            "development_only_and_checkpoint_training_or_validation_overlap"
        ),
        "source_summary_sha256": sha256(metrics_path),
        "source_retention_manifest_sha256": sha256(retention_path),
        "exported_files": {
            role: {
                "filename": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in outputs.items()
            if role != "provenance"
        },
    }
    outputs["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ensure_public_text(outputs["provenance"])
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    payload = export(args.run_root, args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
