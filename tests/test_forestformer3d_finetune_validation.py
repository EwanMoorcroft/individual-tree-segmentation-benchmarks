from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from methods.forestformer3d.scripts.runtime.resolve_finetune_validation_task import (
    EPOCHS,
    resolve,
)
from methods.forestformer3d.scripts.runtime.run_official_test import (
    prepare_entrypoint_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forestformer3d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_resolver_maps_five_checkpoints_to_five_frozen_plots(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(21):
        rows.append(
            {
                "task_index": index,
                "plot_id": f"site/plot_{index}",
                "safe_plot_id": f"site__plot_{index}",
                "relative_path": f"site/plot_{index}.las",
                "fine_tune_role": (
                    "validation" if index in {0, 3, 7, 8, 20} else "train"
                ),
                "point_count": 10,
                "input_sha256": "a" * 64,
            }
        )
    with (tmp_path / "fine_tune_split.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "fine_tune_freeze.json").write_text(
        json.dumps(
            {
                "schema": "forestformer3d_fine_tune_freeze_v1",
                "split": {"held_out_access": False},
                "selection": {"evaluated_checkpoint_epochs": list(EPOCHS)},
            }
        ),
        encoding="utf-8",
    )
    checkpoints = []
    training = tmp_path / "training_full"
    training.mkdir()
    for epoch in EPOCHS:
        path = training / f"epoch_{epoch}.pth"
        path.write_bytes(f"checkpoint-{epoch}".encode())
        checkpoints.append(
            {
                "epoch": epoch,
                "relative_path": f"training_full/epoch_{epoch}.pth",
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    (tmp_path / "checkpoint_inventory.json").write_text(
        json.dumps(
            {
                "schema": "forestformer3d_finetune_checkpoint_inventory_v1",
                "status": "complete",
                "held_out_access": False,
                "epochs": list(EPOCHS),
                "checkpoints": checkpoints,
            }
        ),
        encoding="utf-8",
    )

    first = resolve(tmp_path, 0)
    last = resolve(tmp_path, 24)
    assert (first["epoch"], first["manifest_task_index"]) == (7, 0)
    assert (last["epoch"], last["manifest_task_index"]) == (35, 20)
    assert first["held_out_access"] is False
    assert last["checkpoint_sha256"] == checkpoints[-1]["sha256"]


def test_validation_submission_is_development_only_and_frozen() -> None:
    submit = (METHOD / "slurm/submit_finetune_validation.sh").read_text()
    task = (METHOD / "slurm/run_finetune_validation.sbatch").read_text()
    summary = (
        METHOD / "scripts/evaluation/summarise_finetune_validation.py"
    ).read_text()
    assert "--array=0-24%2" in submit
    assert 're.fullmatch(r"[0-9a-f]{40}", f["benchmark_commit"])' in submit
    assert "5 checkpoints x 5 frozen development-validation plots" in submit
    assert "held-out access false" in submit
    assert "--checkpoint-sha256" in task
    assert "--checkpoint-layout runtime_saved" in task
    assert "--training-mode fine_tuned_on_dev" in task
    assert "maximum_mean_plot_f1" in summary
    assert "maximum_micro_f1" in summary
    assert "earliest_checkpoint_epoch" in summary
    assert '"held_out_access": False' in summary


def test_runtime_saved_checkpoint_is_not_preconditioned_twice(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    checkpoint = tmp_path / "epoch_7.pth"
    state = {f"unet.block_{index}.weight": tensor for index in range(49)}
    torch.save({"state_dict": state}, checkpoint)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    prepared = prepare_entrypoint_checkpoint(
        checkpoint,
        evidence,
        expected_sha256=_sha(checkpoint),
        checkpoint_layout="runtime_saved",
    )
    try:
        observed = torch.load(prepared, map_location="cpu")["state_dict"][
            "unet.block_0.weight"
        ]
        assert torch.equal(observed, tensor)
        adapter = json.loads(
            (evidence / "checkpoint_entrypoint_adapter.json").read_text()
        )
        assert adapter["precondition_operation"] == "identity"
        assert adapter["source_layout"] == "runtime_saved"
    finally:
        prepared.unlink(missing_ok=True)
