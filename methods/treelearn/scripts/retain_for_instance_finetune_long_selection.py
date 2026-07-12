"""Copy selected TreeLearn control evidence and checkpoint from fastscratch to scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-freeze", required=True, type=Path)
    parser.add_argument("--crop-inventory", required=True, type=Path)
    parser.add_argument("--selection-freeze", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection_freeze.read_text())
    long_freeze = json.loads(args.long_freeze.read_text())
    if selection.get("status") != "frozen_comparable_development_selected_checkpoint":
        raise ValueError("Comparable checkpoint selection is not frozen")
    if selection.get("held_out_test_accessed") is not False:
        raise ValueError("Selection does not lock held-out test access")
    if args.output_root.exists() or args.checkpoint_output.exists() or args.manifest_output.exists():
        raise FileExistsError("Durable long-run selection destination already exists")
    source_checkpoint = Path(selection["selected_checkpoint"]).resolve()
    if sha256(source_checkpoint) != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected source checkpoint changed")

    args.output_root.mkdir(parents=True)
    sources = {
        "long_finetune_freeze.json": args.long_freeze,
        "crop_inventory.json": args.crop_inventory,
        "selection_freeze.json": args.selection_freeze,
        "candidate_selection.csv": args.selection_freeze.parent / "candidate_selection.csv",
        "validation_diagnostics.csv": args.selection_freeze.parent / "validation_diagnostics.csv",
        "evaluation_config.yml": Path(long_freeze["evaluation_config"]),
    }
    for trial in long_freeze["trials"]:
        sources[f'trial_{int(trial["trial_index"])}_training_config.yaml'] = Path(
            trial["training_config"]
        )
    for row in long_freeze["plots"]:
        if row.get("training_role") != "train":
            continue
        safe = row["safe_plot_id"]
        sources[f"{safe}_crop_config.yaml"] = Path(row["crop_config"])
        sources[f"{safe}_crop_inventory.json"] = Path(row["crop_inventory"])
        sources[f"{safe}_normalisation.json"] = Path(row["normalisation_metadata"])
    for trial_index in range(8):
        sources[f"trial_{trial_index}_environment.json"] = (
            args.long_freeze.parent / "trial_completions"
            / f"trial_{trial_index}_environment.json"
        )
        sources[f"trial_{trial_index}_completion.json"] = (
            args.long_freeze.parent / "trial_completions"
            / f"trial_{trial_index}.json"
        )
    retained = []
    for name, source in sources.items():
        destination = args.output_root / name
        shutil.copy2(source, destination)
        retained.append({
            "path": str(destination.resolve()),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, args.checkpoint_output)
    if sha256(args.checkpoint_output) != selection["selected_checkpoint_sha256"]:
        raise ValueError("Durable checkpoint copy hash mismatch")
    retained.append({
        "path": str(args.checkpoint_output.resolve()),
        "size_bytes": args.checkpoint_output.stat().st_size,
        "sha256": sha256(args.checkpoint_output),
    })
    payload = {
        "schema_version": 1,
        "status": "long_finetune_selection_retained_on_scratch",
        "run_id": selection["source_long_run_id"],
        "held_out_test_accessed": False,
        "selected_epoch": 35,
        "selected_training_plots": 16,
        "retained_checkpoint": str(args.checkpoint_output.resolve()),
        "retained_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "retained_files": retained,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"retention_manifest={args.manifest_output}")
    print(f"retained_checkpoint={args.checkpoint_output}")
    print("held_out_test_accessed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
