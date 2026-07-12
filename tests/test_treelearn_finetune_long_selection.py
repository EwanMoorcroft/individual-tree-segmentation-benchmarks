from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "methods/treelearn/scripts"


def load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolver = load("resolve_for_instance_finetune_long_validation_task")
selector = load("select_for_instance_finetune_long")


def frozen_trials(tmp_path: Path) -> dict:
    trials = []
    seeds = (42, 31415, 2022, 2026, 2718, 1618, 1729, 123456)
    for trial_index in range(8):
        seed = seeds[trial_index]
        trials.append({
            "trial_index": trial_index,
            "config_id": "full_lr_1e-5",
            "seed": seed,
            "checkpoint_root": str(tmp_path / f"trial_{trial_index}"),
        })
    return {
        "run_id": "treelearn_long_unit",
        "evaluation_config": str(tmp_path / "evaluation.yml"),
        "evaluation_config_sha256": "e" * 64,
        "trials": trials,
    }


def test_long_validation_array_maps_all_checkpoints_to_five_plots(tmp_path: Path) -> None:
    freeze = frozen_trials(tmp_path)
    first = resolver.resolve_task(freeze, 0)
    assert first == {
        "trial_index": 0,
        "config_id": "full_lr_1e-5",
        "seed": 42,
        "epoch": 7,
        "manifest_task_indices": [0, 3, 7, 8, 20],
        "checkpoint": str((tmp_path / "trial_0/epoch_7.pth").resolve()),
        "validation_run_id": (
            "treelearn_long_unit_trial_00_full_lr_1e-5_seed_42_epoch_7_validation"
        ),
        "evaluation_config": str((tmp_path / "evaluation.yml").resolve()),
        "evaluation_config_sha256": "e" * 64,
        "training_mode": "fine_tuned_on_dev",
    }
    last = resolver.resolve_task(freeze, 39)
    assert (last["trial_index"], last["seed"], last["epoch"]) == (7, 123456, 35)
    assert last["manifest_task_indices"] == [0, 3, 7, 8, 20]
    freeze["initial_checkpoint"] = str(tmp_path / "clean.pth")
    baseline = resolver.resolve_task(freeze, 40)
    assert baseline["config_id"] == "clean_pretrained"
    assert baseline["manifest_task_indices"] == [0, 3, 7, 8, 20]
    assert baseline["training_mode"] == "published_pretrained"


def test_selection_rule_uses_mean_then_micro_then_earlier_epoch() -> None:
    candidates = [
        {"config_id": "a", "epoch": 500, "average_mean_plot_f1": 0.6,
         "average_micro_f1": 0.7},
        {"config_id": "b", "epoch": 250, "average_mean_plot_f1": 0.6,
         "average_micro_f1": 0.7},
        {"config_id": "c", "epoch": 100, "average_mean_plot_f1": 0.59,
         "average_micro_f1": 0.9},
    ]
    assert selector.choose_candidate(candidates)["config_id"] == "b"


def test_long_jobs_lock_test_and_use_expected_resources() -> None:
    validation = (
        ROOT / "methods/treelearn/slurm/run_for_instance_finetune_long_validation.sbatch"
    ).read_text()
    gate = (
        ROOT / "methods/treelearn/slurm/gate_for_instance_finetune_long.sbatch"
    ).read_text()
    submit = (
        ROOT / "methods/treelearn/slurm/submit_for_instance_finetune_long.sh"
    ).read_text()
    crop = (
        ROOT / "methods/treelearn/slurm/generate_for_instance_finetune_long_crops.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-40%8" in validation
    assert 'for MANIFEST_TASK_INDEX in "${PLOT_TASKS[@]}"' in validation
    assert "No held-out test job was submitted" in gate
    assert "--array=0-7%8" in submit
    assert "--array=0-40%8" in submit
    assert "TREELEARN_LONG_CROPS_PER_PLOT=32" in submit
    assert "106a80de2991c5f23484a3f9d03e3b16" in submit
    assert "TREELEARN_DEV_MANIFEST_JSON=$TREELEARN_LONG_DEV_MANIFEST" in submit
    assert "No held-out test job was submitted" in submit
    assert "run_treelearn_crop_generation_seeded.py" in crop
    assert 'export PYTHONHASHSEED="$CROP_SEED"' in crop
