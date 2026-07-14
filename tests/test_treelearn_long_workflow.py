from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_long_label_normalisation_is_explicit() -> None:
    module = load(
        "treelearn_long_normalise",
        "methods/treelearn/scripts/normalise_for_instance_finetune_long_plot.py",
    )
    classification = np.array([1, 2, 3, 4, 5, 6, 4, 0])
    tree_id = np.array([91, 0, 7, 11, 12, -1, 0, 13], dtype=np.float64)
    result = module.normalise_instance_labels(classification, tree_id)
    assert result.tolist() == [0, 0, -1, 11, 12, -1, -1, -1]
    assert module.encode_tree_ids(result, np.dtype("float32")).dtype == np.float32


def test_long_label_normalisation_rejects_non_integral_tree_ids() -> None:
    module = load(
        "treelearn_long_normalise_invalid",
        "methods/treelearn/scripts/normalise_for_instance_finetune_long_plot.py",
    )
    with pytest.raises(ValueError, match="non-integral"):
        module.normalise_instance_labels(
            np.array([4, 4]), np.array([1.0, 2.5], dtype=np.float64)
        )


def test_long_freeze_has_fixed_split_matrix_budget_and_clean_checkpoint(
    tmp_path: Path, monkeypatch,
) -> None:
    module = load(
        "treelearn_long_prepare",
        "methods/treelearn/scripts/prepare_for_instance_finetune_long.py",
    )
    dataset = tmp_path / "dataset"
    plots = []
    metadata = tmp_path / "data_split_metadata.csv"
    metadata_rows = ["path,folder,split"]
    for index in range(21):
        site = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")[index % 5]
        source = dataset / site / f"plot_{index}.las"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"synthetic")
        metadata_rows.append(f"{site}\\plot_{index}.las,{site},dev")
        plots.append({
            "task_index": index, "split": "dev", "collection": site,
            "relative_path": f"{site}/plot_{index}.las", "input_las": str(source),
            "safe_plot_id": f"{site}_plot_{index}",
        })
    for index in range(11):
        metadata_rows.append(f"TEST\\test_{index}.las,TEST,test")
    for index in range(35):
        metadata_rows.append(f"NIBIO2\\dev_{index}.las,NIBIO2,dev")
    for index in range(15):
        metadata_rows.append(f"NIBIO2\\test_{index}.las,NIBIO2,test")
    metadata.write_text("\n".join(metadata_rows) + "\n")
    metadata_sha256 = hashlib.sha256(metadata.read_bytes()).hexdigest()
    for row in plots:
        row["split_metadata_sha256"] = metadata_sha256
    manifest = tmp_path / "development.json"
    manifest.write_text(json.dumps({
        "status": "frozen_exact_path_development_manifest",
        "dataset_split": "dev",
        "mapping_rule": "exact_metadata_path_only",
        "held_out_test_accessed": False,
        "split_metadata": str(metadata),
        "split_metadata_sha256": metadata_sha256,
        "plots": plots,
    }))
    repo = tmp_path / "TreeLearn"
    (repo / "configs/_modular").mkdir(parents=True)
    checkpoint = tmp_path / "model_weights_finetuned.pth"
    checkpoint.write_bytes(b"clean weights")
    monkeypatch.setattr(
        module, "CLEAN_CHECKPOINT_MD5", hashlib.md5(b"clean weights").hexdigest()
    )
    template = tmp_path / "evaluation.yml"
    template.write_text(
        "filename: model_weights_20241213.pth\n"
        'default_path: "~/fastscratch/treelearn_checkpoints/model_weights_20241213.pth"\n'
        "source_dataset_name: model_weights_20241213\n"
        'source: "TreeLearn upstream default December 2024 model weights"\n'
        'source_url: "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/IMHF3G"\n'
        'source_md5: "56a3d78f689ae7f1190906b975700311"\n'
        'training_data_provenance: "Authors\' diverse-data checkpoint subsequently fine-tuned on manually labelled data including FOR-instance validation/test"\n'
        'released_weight_test_overlap_status: "documented_for_instance_validation_test_training_overlap; exact_plot_manifest_not_bundled"\n'
        "eligible_for_leakage_free_primary_ranking: false\n"
        'sha256: "5df2f92828f92755bc12e114eaebe83f7ecea94a74c25a6170b68844cc5e19bb"\n'
        "sha256_status: frozen_from_accepted_development_smoke\n"
    )

    freeze = module.prepare(manifest, tmp_path / "run", repo, checkpoint, template)
    assert (freeze["training_plots"], freeze["validation_plots"]) == (16, 5)
    assert [row["task_index"] for row in freeze["plots"] if row["training_role"] == "validation"] == [0, 3, 7, 8, 20]
    assert freeze["initial_checkpoint_role"] == (
        "authors_released_l1w_finetuned_for_instance_clean"
    )
    assert freeze["initial_checkpoint_persistent_id"] == "doi:10.25625/VPMPID/8CIIW0"
    assert freeze["initial_checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert freeze["held_out_test_accessed"] is False
    assert freeze["supplied_split_contract"]["metadata_development_rows"] == 56
    assert freeze["supplied_split_contract"]["metadata_test_rows"] == 26
    assert freeze["supplied_split_contract"]["benchmark_development_rows"] == 21
    assert freeze["supplied_split_contract"]["benchmark_expected_test_rows"] == 11
    assert freeze["supplied_split_contract"]["held_out_test_files_opened"] is False
    verifier = load(
        "treelearn_long_verify",
        "methods/treelearn/scripts/verify_for_instance_finetune_long_contract.py",
    )
    verifier.verify_supplied_split(freeze, manifest)
    metadata.write_text(metadata.read_text().replace(",dev\n", ",test\n", 1))
    with pytest.raises(ValueError, match="split metadata changed"):
        verifier.verify_supplied_split(freeze, manifest)
    assert freeze["crops_per_plot"] == 32
    assert freeze["crop_generation_attempts_per_plot"] == 48
    assert freeze["tuning_crop_count"] == 512
    assert all(row["crops_generate_requested"] == 48 for row in freeze["plots"])
    assert freeze["training_budget"]["epochs"] == 35
    assert freeze["training_budget"]["examples_per_epoch"] == 714
    assert freeze["training_budget"]["examples_seen"] == 24_990
    assert freeze["training_budget"]["batch_size"] == 2
    assert freeze["training_budget"]["optimizer_steps"] == 12_495
    assert freeze["training_budget"]["tuning_trials"] == 8
    assert freeze["training_budget"]["checkpoint_epochs"] == [7, 14, 21, 28, 35]
    assert len(freeze["trials"]) == 8
    assert {trial["seed"] for trial in freeze["trials"]} == {
        42, 31415, 2022, 2026, 2718, 1618, 1729, 123456,
    }
    assert {trial["config_id"] for trial in freeze["trials"]} == {"full_lr_1e-5"}
    assert all(
        hashlib.sha256(Path(trial["training_config"]).read_bytes()).hexdigest()
        == trial["training_config_sha256"]
        for trial in freeze["trials"]
    )
    configs = [json.loads(Path(trial["training_config"]).read_text()) for trial in freeze["trials"]]
    yaml_configs = [yaml.safe_load(Path(trial["training_config"]).read_text()) for trial in freeze["trials"]]
    assert all(config["epochs"] == 35 for config in configs)
    assert all(config["examples_per_epoch"] == 714 for config in configs)
    assert all(config["save_frequency"] == 7 for config in configs)
    assert all(config["optimizer"]["lr"] == 1e-5 for config in configs)
    assert all(
        isinstance(config["optimizer"]["lr"], float)
        and config["optimizer"]["lr"] == 1e-5
        for config in yaml_configs
    )
    assert all(
        '"lr": 0.00001' in Path(trial["training_config"]).read_text()
        for trial in freeze["trials"]
    )
    assert all(config["model"]["fixed_modules"] == [] for config in configs)
    evaluation = Path(freeze["evaluation_config"]).read_text()
    assert "model_weights_finetuned.pth" in evaluation
    assert module.CLEAN_CHECKPOINT_MD5 in evaluation
    assert "model_weights_20241213.pth" not in evaluation
    assert "no_for_instance_training_reported_by_authors" in evaluation
    assert "eligible_for_leakage_free_primary_ranking: true" in evaluation


def test_long_crop_views_keep_validation_out_of_tuning(tmp_path: Path) -> None:
    module = load(
        "treelearn_long_consolidate",
        "methods/treelearn/scripts/consolidate_for_instance_finetune_long_crops.py",
    )
    rows = []
    for index in range(21):
        crop_root = tmp_path / "by_plot" / str(index)
        crop_root.mkdir(parents=True)
        for crop in range(2):
            (crop_root / f"{crop}.npz").write_bytes(b"crop")
        crop_inventory = tmp_path / "by_plot" / str(index) / "crop_inventory.json"
        entries = [
            {
                "name": f"{crop}.npz",
                "size_bytes": 4,
                "sha256": hashlib.sha256(b"crop").hexdigest(),
            }
            for crop in range(2)
        ]
        aggregate = hashlib.sha256()
        for entry in entries:
            aggregate.update(
                f'{entry["name"]}\0{entry["size_bytes"]}\0{entry["sha256"]}\n'.encode()
            )
        crop_inventory.write_text(json.dumps({
            "status": "treelearn_plot_crops_sha256_inventoried",
            "held_out_test_accessed": False,
            "safe_plot_id": f"plot_{index}",
            "crop_seed": 42000 + index,
            "crop_count": 2,
            "total_size_bytes": 8,
            "crop_root": str(crop_root.resolve()),
            "entries_aggregate_sha256": aggregate.hexdigest(),
            "files": entries,
        }))
        rows.append({
            "task_index": index, "safe_plot_id": f"plot_{index}", "split": "dev",
            "training_role": "train" if index < 16 else "validation",
            "crop_root": str(crop_root),
            "crop_seed": 42000 + index,
            "crop_inventory": str(crop_inventory),
        })
    tuning = tmp_path / "views/tuning/npz"
    all_dev = tmp_path / "views/all_development/npz"
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps({
        "held_out_test_accessed": False, "crops_per_plot": 2, "plots": rows,
        "tuning_data_root": str(tuning),
    }))
    inventory_path = tmp_path / "inventory.json"
    result = module.consolidate(freeze_path, inventory_path)
    assert result["tuning"]["crop_count"] == 32
    assert len(result["tuning"]["entries_aggregate_sha256"]) == 64
    assert not any(path.name.startswith("plot_20__") for path in tuning.iterdir())
    assert not all_dev.exists()
    assert result["plot_crop_inventory_count"] == 16
    verified = module.verify_consolidated(freeze_path, inventory_path)
    assert verified["crop_count"] == 32
    legacy_inventory_path = tmp_path / "inventory_schema_v1.json"
    legacy_inventory = json.loads(inventory_path.read_text())
    legacy_inventory["schema_version"] = 1
    for field in (
        "entries_aggregate_sha256",
        "aggregate_fields",
        "symlink_targets_verified",
    ):
        legacy_inventory["tuning"].pop(field)
    legacy_inventory_path.write_text(
        json.dumps(legacy_inventory, indent=2, sort_keys=True) + "\n"
    )
    legacy_bytes = legacy_inventory_path.read_bytes()
    link_targets = {
        path.name: path.readlink() for path in sorted(tuning.iterdir())
    }
    legacy_verified = module.verify_consolidated(
        freeze_path, legacy_inventory_path
    )
    assert legacy_verified["inventory_schema_version"] == 1
    assert legacy_verified["entries_aggregate_sha256"] == (
        verified["entries_aggregate_sha256"]
    )
    assert legacy_inventory_path.read_bytes() == legacy_bytes
    assert {
        path.name: path.readlink() for path in sorted(tuning.iterdir())
    } == link_targets
    source = tmp_path / "by_plot/0/0.npz"
    source.write_bytes(b"xxxx")
    with pytest.raises(ValueError, match="hashes changed"):
        module.verify_consolidated(freeze_path, legacy_inventory_path)
    source.write_bytes(b"crop")
    changed = tuning / "plot_0__0000.npz"
    changed.unlink()
    changed.symlink_to(tmp_path / "by_plot/1/0.npz")
    with pytest.raises(ValueError, match="symlink target changed"):
        module.verify_consolidated(freeze_path, inventory_path)


def test_long_crop_inventory_deterministically_prunes_excess(tmp_path: Path) -> None:
    module = load(
        "treelearn_long_crop_inventory",
        "methods/treelearn/scripts/inventory_for_instance_finetune_long_crops.py",
    )
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    for index in range(35):
        (crop_root / f"{index:04d}.npz").write_bytes(f"crop-{index}".encode())
    normalised = tmp_path / "normalised.las"
    normalised.write_bytes(b"las")
    config = tmp_path / "crops.json"
    config.write_text("{}")
    inventory = tmp_path / "crop_inventory.json"
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "held_out_test_accessed": False,
        "plots": [{
            "task_index": 1,
            "split": "dev",
            "safe_plot_id": "plot_1",
            "crop_inventory": str(inventory),
            "crop_root": str(crop_root),
            "crops_expected": 32,
            "crops_generate_requested": 48,
            "crop_seed": 42001,
            "normalised_las": str(normalised),
            "crop_config": str(config),
        }],
    }))
    result = module.inventory(freeze, 1)
    assert result["crop_generation_outputs_created"] == 35
    assert result["pruned_crop_count"] == 3
    assert result["pruned_crop_names"] == ["0032.npz", "0033.npz", "0034.npz"]
    assert result["crop_count"] == 32
    assert len(list(crop_root.glob("*.npz"))) == 32


def test_long_crop_inventory_rejects_unmanaged_and_symlinked_roots(
    tmp_path: Path,
) -> None:
    module = load(
        "treelearn_long_crop_inventory_containment",
        "methods/treelearn/scripts/inventory_for_instance_finetune_long_crops.py",
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    for index in range(35):
        (outside_root / f"{index:04d}.npz").write_bytes(f"crop-{index}".encode())
    normalised = run_root / "normalised.las"
    normalised.write_bytes(b"las")
    config = run_root / "crops.json"
    config.write_text("{}")

    def write_freeze(crop_root: Path, inventory_name: str) -> Path:
        freeze = run_root / f"{inventory_name}.json"
        freeze.write_text(
            json.dumps(
                {
                    "held_out_test_accessed": False,
                    "plots": [
                        {
                            "task_index": 1,
                            "split": "dev",
                            "safe_plot_id": "plot_1",
                            "crop_inventory": str(
                                run_root / f"{inventory_name}_inventory.json"
                            ),
                            "crop_root": str(crop_root),
                            "crops_expected": 32,
                            "crops_generate_requested": 48,
                            "crop_seed": 42001,
                            "normalised_las": str(normalised),
                            "crop_config": str(config),
                        }
                    ],
                }
            )
        )
        return freeze

    with pytest.raises(ValueError, match="escapes the long-run root"):
        module.inventory(write_freeze(outside_root, "outside"), 1)
    assert len(list(outside_root.glob("*.npz"))) == 35

    linked_root = run_root / "linked_crops"
    linked_root.symlink_to(outside_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked path component"):
        module.inventory(write_freeze(linked_root, "symlink"), 1)
    assert len(list(outside_root.glob("*.npz"))) == 35


def test_long_slurm_training_half_is_guarded_and_uses_eight_gpus() -> None:
    paths = [
        "methods/treelearn/slurm/prepare_for_instance_finetune_long.sbatch",
        "methods/treelearn/slurm/generate_for_instance_finetune_long_crops.sbatch",
        "methods/treelearn/slurm/consolidate_for_instance_finetune_long_crops.sbatch",
        "methods/treelearn/slurm/run_for_instance_finetune_long_trial.sbatch",
    ]
    for relative in paths:
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / relative)], capture_output=True, text=True
        )
        assert completed.returncode == 0, (relative, completed.stderr)
        text = (ROOT / relative).read_text()
        assert "TREELEARN_FINETUNE_LONG_CONFIRMED" in text
        assert "held_out_test_accessed=false" in text
    crop = (ROOT / paths[1]).read_text()
    train = (ROOT / paths[3]).read_text()
    training_runner = (
        ROOT / "methods/treelearn/scripts/run_treelearn_training_seeded.py"
    ).read_text()
    assert "#SBATCH --partition=nodes" in crop
    assert "#SBATCH --gres=gpu:1" not in crop
    assert 'EXPECTED_CROPS="${TREELEARN_LONG_CROPS_PER_PLOT:-32}"' in crop
    assert "#SBATCH --partition=gpu-l40s" in train
    assert "#SBATCH --time=36:00:00" in train
    assert "run_treelearn_training_seeded.py" in train
    assert '--work-dir "$WORK_DIR"' in train
    assert '--crop-inventory "$TREELEARN_LONG_ROOT/crop_inventory.json"' in train
    assert '--freeze "$TREELEARN_LONG_FREEZE"' in train
    assert training_runner.index("crop_integrity = verify_consolidated") < (
        training_runner.index("import numpy as np")
    )


def test_long_slurm_chain_honours_project_root_override() -> None:
    paths = [
        "methods/treelearn/slurm/prepare_for_instance_finetune_long.sbatch",
        "methods/treelearn/slurm/generate_for_instance_finetune_long_crops.sbatch",
        "methods/treelearn/slurm/consolidate_for_instance_finetune_long_crops.sbatch",
        "methods/treelearn/slurm/run_for_instance_finetune_long_trial.sbatch",
        "methods/treelearn/slurm/run_for_instance_finetune_long_validation.sbatch",
        "methods/treelearn/slurm/select_for_instance_finetune_long.sbatch",
        "methods/treelearn/slurm/gate_for_instance_finetune_long.sbatch",
    ]
    root_assignment = (
        'TREELEARN_PROJECT_ROOT="${TREELEARN_PROJECT_ROOT:-'
        '$HOME/scratch/tree-seg-benchmark}"'
    )
    for relative in paths:
        text = (ROOT / relative).read_text()
        assert root_assignment in text
        assert 'cd "$TREELEARN_PROJECT_ROOT"' in text
        completed = subprocess.run(
            ["bash", "-n", str(ROOT / relative)], capture_output=True, text=True
        )
        assert completed.returncode == 0, (relative, completed.stderr)

    crop = (ROOT / paths[1]).read_text()
    assert (
        'python "$TREELEARN_PROJECT_ROOT/methods/treelearn/scripts/'
        'run_treelearn_crop_generation_seeded.py"'
    ) in crop
    submit = (
        ROOT / "methods/treelearn/slurm/submit_for_instance_finetune_long.sh"
    ).read_text()
    assert root_assignment in submit
    assert "export TREELEARN_PROJECT_ROOT" in submit
    assert 'PROJECT_ROOT="$TREELEARN_PROJECT_ROOT"' in submit
