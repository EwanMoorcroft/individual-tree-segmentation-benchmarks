from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "methods/tls2trees/scripts/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))


def load_script(relative_path: str, name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_command_resolves_every_published_operational_value() -> None:
    semantic = load_script(
        "methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_semantic.py",
        "published_semantic_runner",
    )
    common = load_script(
        "methods/tls2trees/scripts/runtime/for_instance_published_common.py",
        "published_common_semantic",
    )
    config, _ = common.load_config(
        "methods/tls2trees/configs/for_instance_published_default.yml"
    )
    command = semantic.build_semantic_command(
        input_tile=Path("input/000000.downsample.ply"),
        tile_index=Path("input/tile_index.dat"),
        output_dir=Path("semantic"),
        model_path=Path("TLS2trees/tls2trees/fsct/model/model.pth"),
        config=config,
    )

    assert command[command.index("--buffer") + 1] == "5.0"
    assert command[command.index("--batch_size") + 1] == "10"
    assert command[command.index("--num_procs") + 1] == "10"
    assert command[command.index("--is-wood") + 1] == "1.0"
    assert command[command.index("--step") + 1] == "3"
    assert "--keep-npy" not in command
    assert "--verbose" in command
    reproducibility = config["reproducibility_controls"]
    assert reproducibility["deterministic_algorithms"] is False
    assert "scatter_add" in reproducibility["deterministic_algorithms_reason"]
    assert reproducibility["determinism_policy"].endswith(
        "nondeterministic_cuda_scatter"
    )


def test_instance_command_uses_published_values_and_both_targets() -> None:
    instance = load_script(
        "methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py",
        "published_instance_runner",
    )
    common = load_script(
        "methods/tls2trees/scripts/runtime/for_instance_published_common.py",
        "published_common_instance",
    )
    config, _ = common.load_config(
        "methods/tls2trees/configs/for_instance_published_default.yml"
    )
    parameters = instance.resolved_instance_parameters(config)
    command = instance.build_command(
        Path("instance_patched.py"),
        Path("000000.downsample.segmented.ply"),
        Path("tile_index.dat"),
        Path("predictions"),
        parameters,
    )

    assert command[command.index("--n-tiles") + 1] == "5"
    assert command[command.index("--slice-thickness") + 1] == "0.5"
    boundary = command.index("--find-stems-boundary")
    assert command[boundary + 1 : boundary + 3] == ["2.0", "2.5"]
    assert command[command.index("--graph-edge-length") + 1] == "2.0"
    assert command[command.index("--graph-maximum-cumulative-gap") + 1] == "3.0"
    assert command[command.index("--min-points-per-tree") + 1] == "200"
    assert command[command.index("--add-leaves-voxel-length") + 1] == "0.5"
    assert command[command.index("--add-leaves-edge-length") + 1] == "1.0"
    assert "--add-leaves" in command


def test_runtime_patch_sources_are_exact_and_fail_closed() -> None:
    instance_patch = load_script(
        "methods/tls2trees/scripts/runtime/patches/instance_patched.py",
        "published_instance_patch",
    )
    semantic_patch = load_script(
        "methods/tls2trees/scripts/runtime/patches/semantic_patched.py",
        "published_semantic_patch",
    )
    instance_source = (
        "before\n"
        + instance_patch.PATCH_TARGET
        + "\nmiddle\n"
        + instance_patch.EMPTY_ORIGINS_TARGET
        + "\nmiddle2\n"
        + instance_patch.EMPTY_WOOD_PATH_TARGET
        + "\nmiddle3\n"
        + instance_patch.SMALL_GRAPH_TARGET
        + "\nmiddle4\n"
        + instance_patch.NO_STEMS_TARGET
        + "\nmiddle5\n"
        + instance_patch.EMPTY_LEAF_TIPS_TARGET
        + "\nmiddle6\n"
        + instance_patch.LEAF_EDGE_TARGET
        + "\nafter\n"
    )
    patched_instance = instance_patch.patched_source(
        instance_source,
        require_leaf_edge=True,
        require_empty_graph_guard=True,
        require_small_graph_guard=True,
        require_no_stems_guard=True,
        require_empty_leaf_tips_guard=True,
    )
    assert "Cannot restore clstr" in patched_instance
    assert "params.add_leaves_edge_length" in patched_instance
    assert "no_graph_connected_stem_bases" in patched_instance
    assert "raise SystemExit(0)" in patched_instance
    assert "n_neighbours = min(n_neighbours, sample_count - 1)" in patched_instance
    assert "no_clustered_wood_convex_hulls" in patched_instance
    assert "no_in_tile_stem_predictions" in patched_instance
    assert "stem-only leaf-on predictions written" in patched_instance
    inference_source = "x = " + semantic_patch.LOCAL_SHIFT_TARGET
    patched_inference = semantic_patch.patched_inference_source(inference_source)
    assert semantic_patch.LOCAL_SHIFT_REPLACEMENT in patched_inference

    with pytest.raises(RuntimeError, match="leaf-edge"):
        instance_patch.patched_source(
            instance_patch.PATCH_TARGET, require_leaf_edge=True
        )
    with pytest.raises(RuntimeError, match="empty-graph"):
        instance_patch.patched_source(
            instance_patch.PATCH_TARGET,
            require_empty_graph_guard=True,
        )
    with pytest.raises(RuntimeError, match="small-graph"):
        instance_patch.patched_source(
            instance_patch.PATCH_TARGET,
            require_small_graph_guard=True,
        )
    with pytest.raises(RuntimeError, match="no-stems"):
        instance_patch.patched_source(
            instance_patch.PATCH_TARGET,
            require_no_stems_guard=True,
        )
    with pytest.raises(RuntimeError, match="empty-leaf-tips"):
        instance_patch.patched_source(
            instance_patch.PATCH_TARGET,
            require_empty_leaf_tips_guard=True,
        )
    with pytest.raises(RuntimeError, match="local-shift"):
        semantic_patch.patched_inference_source("no matching source")

    semantic_patch_source = (
        RUNTIME / "patches" / "semantic_patched.py"
    ).read_text(encoding="utf-8")
    assert "patch_pandas_append" in semantic_patch_source
    assert "CappedNearestNeighbors" not in semantic_patch_source
    assert "torch.use_deterministic_algorithms(False)" in semantic_patch_source
    assert "torch.use_deterministic_algorithms(True)" not in semantic_patch_source


def test_runtime_metadata_includes_child_memory_and_forces_hash_seed() -> None:
    common_source = (RUNTIME / "for_instance_published_common.py").read_text(
        encoding="utf-8"
    )
    assert "resource.RUSAGE_CHILDREN" in common_source

    for runner in (
        "run_for_instance_tls2trees_semantic.py",
        "run_for_instance_tls2trees_instance.py",
    ):
        source = (RUNTIME / runner).read_text(encoding="utf-8")
        assert 'environment["PYTHONHASHSEED"] = ' in source
        assert 'environment.setdefault("PYTHONHASHSEED"' not in source

    semantic_source = (
        RUNTIME / "run_for_instance_tls2trees_semantic.py"
    ).read_text(encoding="utf-8")
    assert "No neighbour-count fallback is permitted" in semantic_source
    assert "header.vertex_count <= 0" in semantic_source
    instance_source = (
        RUNTIME / "run_for_instance_tls2trees_instance.py"
    ).read_text(encoding="utf-8")
    assert 'conversion["tile_index_sha256"]' in instance_source
    assert "empty_graph_sources_recorded_as_no_predictions" in instance_source
    assert "small_wood_graph_neighbours_capped_to_available_samples" in instance_source


def test_failed_empty_instance_attempt_is_archived_without_deletion(
    tmp_path: Path,
) -> None:
    instance = load_script(
        "methods/tls2trees/scripts/runtime/run_for_instance_tls2trees_instance.py",
        "published_instance_recovery",
    )
    plot_root = tmp_path / "plot"
    raw_root = plot_root / "predictions/raw"
    logs_root = plot_root / "logs/instance"
    metadata_path = plot_root / "metadata/instance_run.json"
    raw_root.mkdir(parents=True)
    logs_root.mkdir(parents=True)
    metadata_path.parent.mkdir(parents=True)
    (logs_root / "tile_000000.stderr.log").write_text(
        "ValueError: sources must not be empty\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "error": "RuntimeError: Instance tile 0 failed with return code 1",
            }
        ),
        encoding="utf-8",
    )

    evidence = instance.archive_failed_instance_attempt(
        plot_root=plot_root,
        raw_root=raw_root,
        metadata_path=metadata_path,
    )

    archive = plot_root / "recovery/instance_failed_attempt_1"
    assert evidence["status"] == "failed_attempt_archived"
    assert (archive / "instance_run.json").is_file()
    assert (archive / "logs/tile_000000.stderr.log").is_file()
    assert (archive / "raw").is_dir()
    assert not raw_root.exists()
    assert not metadata_path.exists()


def test_runtime_plot_resolution_blocks_test_rows(tmp_path: Path) -> None:
    common = load_script(
        "methods/tls2trees/scripts/runtime/for_instance_published_common.py",
        "published_common_split_gate",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_split": "test",
                "plots": [
                    {
                        "task_index": 0,
                        "split": "test",
                        "safe_plot_id": "CULS_plot_2_annotated",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Manifest has unexpected"):
        common.resolve_plot_context(
            manifest_path=manifest,
            task_index=0,
            output_root=tmp_path,
            run_id="smoke-001",
            variant="published_default",
            split="development",
        )


def test_isolated_environment_version_contract_and_gpu_evidence_are_explicit() -> None:
    validator = load_script(
        "methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py",
        "tls2trees_environment_validator",
    )
    installed = dict(validator.EXPECTED_DISTRIBUTIONS)
    assert validator.version_contract_errors(installed, (3, 9, 19)) == []

    installed["torch"] = "2.0.0"
    errors = validator.version_contract_errors(installed, (3, 10, 0))
    assert any("Python version mismatch" in error for error in errors)
    assert any("Distribution mismatch for torch" in error for error in errors)

    conda_packages = {
        "cudatoolkit": dict(validator.EXPECTED_CONDA_PACKAGES["cudatoolkit"])
    }
    assert validator.conda_package_contract_errors(conda_packages) == []
    conda_packages["cudatoolkit"]["build"] = "hb139c0e_13"
    assert any(
        "Conda package mismatch for cudatoolkit build" in error
        for error in validator.conda_package_contract_errors(conda_packages)
    )

    expected_library_dir = str(Path(sys.executable).resolve().parent.parent / "lib")

    marker = {
        "schema_version": validator.EXPECTED_MARKER_SCHEMA,
        "status": "passed",
        "python_version": "3.9.23",
        "validated_at_utc": "2026-07-16T00:00:00+00:00",
        "python_user_site_disabled": True,
        "python_executable": sys.executable,
        "tls2trees_commit": validator.EXPECTED_UPSTREAM_COMMIT,
        "model_sha256": validator.EXPECTED_MODEL_SHA256,
        "cuda_required": True,
        "cpu_model_load": "passed",
        "compatibility_classification": (
            "historical_runtime_reproduction_not_for_instance_parameter_tuning"
        ),
        "distributions": dict(validator.EXPECTED_DISTRIBUTIONS),
        "conda_packages": {
            "cudatoolkit": dict(
                validator.EXPECTED_CONDA_PACKAGES["cudatoolkit"]
            )
        },
        "cuda_runtime": {
            "library_dir": expected_library_dir,
            "ld_library_path_contains_prefix_lib": True,
            "libraries": {
                name: f"{expected_library_dir}/{name}"
                for name in validator.EXPECTED_CUDA_RUNTIME_LIBRARIES
            },
        },
        "cuda": {
            "available": True,
            "compiled_pyg_operations": "passed",
            "bundled_model_forward": "passed",
            "deterministic_algorithms_enabled": False,
            "determinism_policy": validator.EXPECTED_DETERMINISM_POLICY,
            "repeat_forward_all_finite": True,
            "repeat_forward_exact_equal": False,
            "repeat_forward_max_abs_delta": 1.0e-7,
            "seed": 42,
            "torch_cuda_build": "11.1",
            "model_missing_keys": [],
            "model_unexpected_keys": [],
            "device_name": "synthetic GPU",
        },
    }
    assert validator.setup_marker_errors(marker, Path(sys.executable)) == []
    marker["cuda"]["bundled_model_forward"] = "not_run"
    assert any(
        "bundled_model_forward" in error
        for error in validator.setup_marker_errors(marker, Path(sys.executable))
    )
    marker["cuda"]["bundled_model_forward"] = "passed"
    marker["cuda"]["repeat_forward_max_abs_delta"] = float("nan")
    assert any(
        "repeat_forward_max_abs_delta" in error
        for error in validator.setup_marker_errors(marker, Path(sys.executable))
    )

    source = (
        RUNTIME / "validate_tls2trees_environment.py"
    ).read_text(encoding="utf-8")
    for operation in ("fps", "radius", "knn_interpolate", "global_max_pool"):
        assert operation in source
    assert "torch.use_deterministic_algorithms(False)" in source
    assert "torch.use_deterministic_algorithms(True)" not in source
    assert "repeat_forward_max_abs_delta" in source
    assert "bundled_model_forward" in source
    assert "state dict does not exactly match" in source
    assert "setup_marker_errors" in source
    assert "cuda_runtime_library_evidence" in source
    assert "--skip-model-load cannot be combined with --require-cuda" in source


def test_cuda_runtime_conda_record_and_library_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = load_script(
        "methods/tls2trees/scripts/runtime/validate_tls2trees_environment.py",
        "tls2trees_cuda_runtime_validator",
    )
    prefix = tmp_path / "tls2trees"
    conda_meta = prefix / "conda-meta"
    library_dir = prefix / "lib"
    conda_meta.mkdir(parents=True)
    library_dir.mkdir()
    (conda_meta / "cudatoolkit-11.1.1-h6406543_8.json").write_text(
        json.dumps(
            {
                "name": "cudatoolkit",
                "version": "11.1.1",
                "build": "h6406543_8",
                "build_number": 8,
                "subdir": "linux-64",
                "md5": "4851e7f19b684e517dc8e6b5b375dda0",
                "url": (
                    "https://prefix.dev/conda-forge/linux-64/"
                    "cudatoolkit-11.1.1-h6406543_8.tar.bz2"
                ),
            }
        ),
        encoding="utf-8",
    )
    for name in validator.EXPECTED_CUDA_RUNTIME_LIBRARIES:
        (library_dir / name).touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", str(library_dir))

    records = validator.conda_package_records(prefix)
    assert validator.conda_package_contract_errors(records) == []
    evidence = validator.cuda_runtime_library_evidence(prefix)
    assert evidence["library_dir"] == str(library_dir.resolve())
    assert evidence["ld_library_path_contains_prefix_lib"] is True
    assert set(evidence["libraries"]) == set(
        validator.EXPECTED_CUDA_RUNTIME_LIBRARIES
    )
