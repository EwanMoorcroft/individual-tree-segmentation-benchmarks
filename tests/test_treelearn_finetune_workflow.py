from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "methods/treelearn/scripts/prepare_for_instance_finetune.py"


def load_module():
    spec = importlib.util.spec_from_file_location("treelearn_finetune_prepare", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finetune_preparation_is_seeded_and_development_only(tmp_path: Path) -> None:
    module = load_module()
    dataset = tmp_path / "dataset"
    plots = []
    for index in range(21):
        site = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")[index % 5]
        source = dataset / site / f"plot_{index}.las"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic")
        plots.append({"task_index": index, "split": "dev", "collection": site,
                      "relative_path": f"{site}/plot_{index}.las", "input_las": str(source)})
    manifest = tmp_path / "development.json"
    manifest.write_text(json.dumps({"held_out_test_accessed": False, "plots": plots}))
    repo = tmp_path / "TreeLearn"
    (repo / "configs/_modular").mkdir(parents=True)
    checkpoint = tmp_path / "published.pth"
    checkpoint.write_bytes(b"weights")

    assert module.assign_roles(plots, 42) == module.assign_roles(plots, 42)
    roles = module.assign_roles(plots, 42)
    assert sum(row["training_role"] == "train" for row in roles) == 16
    assert sum(row["training_role"] == "validation" for row in roles) == 5

    run_root = tmp_path / "run"
    frozen = module.prepare(manifest, run_root, repo, checkpoint, 42, 64)
    assert frozen["training_mode"] == "fine_tuned_on_dev"
    assert frozen["held_out_test_accessed"] is False
    assert (frozen["training_plots"], frozen["validation_plots"]) == (16, 5)
    assert len(list((run_root / "data/train/forests").iterdir())) == 16
    full = json.loads((run_root / "configs/finetune_full.yaml").read_text())
    assert full["pretrain"] == str(checkpoint.resolve())
    assert full["epochs"] == 100
    assert full["optimizer"]["lr"] == 0.0003
    assert isinstance(full["scheduler"]["lr_min"], float)
    assert isinstance(full["scheduler"]["warmup_lr_init"], float)


def test_finetune_submission_has_smoke_gate_and_no_test_route() -> None:
    submitter = (ROOT / "methods/treelearn/slurm/submit_for_instance_finetune.sh").read_text()
    monitor = (ROOT / "methods/treelearn/slurm/monitor_for_instance_finetune.sh").read_text()
    assert 'afterok:$PREP' in submitter
    assert 'afterok:$SMOKE' in submitter
    assert "TREELEARN_FINETUNE_DEV_CONFIRMED" in submitter
    assert "No held-out test job was submitted" in submitter
    assert "checkpoint_ready" in monitor


def test_finetune_validation_is_five_plot_test_locked() -> None:
    submitter = (ROOT / "methods/treelearn/slurm/submit_for_instance_finetune_validation.sh").read_text()
    task = (ROOT / "methods/treelearn/slurm/run_for_instance_finetune_validation.sbatch").read_text()
    summary = (ROOT / "methods/treelearn/scripts/summarise_for_instance_finetune_validation.py").read_text()
    assert "--array=0,3,7,8,20%2" in submitter
    assert 'VALIDATION_RUN_ID="${TRAINING_RUN_ID}_validation_$STAMP"' in submitter
    assert "--training-mode fine_tuned_on_dev" in task
    assert "held_out_test_accessed=false" in task
    assert 'len(rows) != 5' in summary
    assert '"retention_status": "retention_verified"' in summary
    assert "No held-out test job was submitted" in submitter
