from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from methods.forestformer3d.scripts.provenance import inventory_artifacts
from methods.forestformer3d.scripts.runtime import validate_environment


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "methods/forestformer3d"


def test_all_method_configs_expose_matching_slugs() -> None:
    configs = sorted((METHOD / "configs").glob("*.yml"))
    assert {path.name for path in configs} == {
        "fine_tuned_on_dev.yml",
        "fine_tuned_test.yml",
        "for_instance_contract.yml",
        "one_plot_smoke.yml",
        "published_pretrained_development.yml",
        "published_pretrained_test.yml",
        "retention.yml",
        "upstream.yml",
    }
    for path in configs:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["project"]["dataset_slug"] == "for-instance", path
        assert payload["project"]["method_slug"] == "forestformer3d", path
        assert payload["dataset"]["slug"] == "for-instance", path
        assert payload["method"]["slug"] == "forestformer3d", path


def test_upstream_config_freezes_observed_identities_and_remaining_gates() -> None:
    config = yaml.safe_load(
        (METHOD / "configs/upstream.yml").read_text(encoding="utf-8")
    )
    source = config["method"]["source"]
    checkpoint = config["method"]["checkpoint"]
    container = config["container"]

    assert source["commit"] == validate_environment.SOURCE_COMMIT
    assert checkpoint["sha256"] == validate_environment.CHECKPOINT_SHA256
    assert checkpoint["archive_md5"] == "553d67379331966509076f3fbb409e57"
    assert container["base_manifest_digest"] == (
        "sha256:58d848c38665fd3ed20bee65918255cb083637c860eb4fae67face2fb2ff5702"
    )
    assert container["qualified_base_sif_sha256"] == (
        "4a35d5a57c1d57061f899b514329ad8ec2bf74a9ff31d103c0a53a289e07c84f"
    )
    assert container["full_image_status"].startswith("blocked_barkla_")
    assert container["barkla_runtime"]["rootless_probe_status"] == "passed"
    assert config["eligibility"]["exposure_gate"] == "passed"
    assert config["eligibility"]["overall_admission"].startswith("conditional_")


def test_apptainer_recipe_pins_base_source_and_official_replacements() -> None:
    recipe = (METHOD / "container/forestformer3d.def").read_text(encoding="utf-8")

    assert (
        "From: pytorch/pytorch@sha256:"
        "58d848c38665fd3ed20bee65918255cb083637c860eb4fae67face2fb2ff5702"
        in recipe
    )
    assert validate_environment.SOURCE_COMMIT in recipe
    for digest in validate_environment.SOURCE_HASHES.values():
        assert digest in recipe or digest in {
            validate_environment.SOURCE_HASHES["Dockerfile"],
            validate_environment.SOURCE_HASHES[
                "configs/oneformer3d_qs_radius16_qp300_2many.py"
            ],
            validate_environment.SOURCE_HASHES["tools/test.py"],
            validate_environment.SOURCE_HASHES["tools/train.py"],
        }
    for digest in validate_environment.INSTALLED_REPLACEMENTS.values():
        assert digest in recipe
    assert "train_val_data.zip" not in recipe
    assert "test_data.zip" not in recipe
    assert "epoch_3000_fix.pth" not in recipe
    assert "TORCH_CUDA_ARCH_LIST=\"8.0\"" in recipe


def test_rootless_builder_preserves_official_dependency_and_source_pins() -> None:
    builder = (
        METHOD / "scripts/runtime/build_rootless_environment.sh"
    ).read_text(encoding="utf-8")

    assert "apt-get" not in builder
    assert "fakeroot" not in builder
    assert "gcc_linux-64=9" in builder
    assert "openblas=0.3.21" in builder
    assert "libgl=1.7.0" in builder
    assert "libglx=1.7.0" in builder
    assert '--install-option="--blas_include_dirs=$TOOLCHAIN/include"' in builder
    assert '--install-option="--blas_library_dirs=$TOOLCHAIN/lib"' in builder
    assert 'test -f "$TOOLCHAIN/include/cblas.h"' in builder
    assert 'test -e "$TOOLCHAIN/lib/libGL.so.1"' in builder
    assert 'test -e "$TOOLCHAIN/lib/libGLX.so.0"' in builder
    assert "torch.utils.cmake_prefix_path)')" in builder
    assert "torch.utils.cmake_prefix_path())" not in builder
    assert "CONDA_PKGS_DIRS" in builder
    assert "PIP_CACHE_DIR" in builder
    assert 'export HOME="$BUILD_HOME"' in builder
    assert "set +u" in builder
    assert "conda activate" in builder
    assert builder.index("set +u") < builder.index("conda activate") < builder.index(
        "set -u", builder.index("conda activate")
    )
    assert "mmengine==0.7.3" in builder
    assert "mmcv==2.0.0" in builder
    assert validate_environment.SOURCE_COMMIT in builder
    for digest in validate_environment.INSTALLED_REPLACEMENTS.values():
        assert digest in builder
    assert "conda_explicit.txt" in builder
    assert "pip_freeze.txt" in builder


def test_hash_validator_and_retention_inventory_are_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    artifact = root / "prediction.npz"
    artifact.write_bytes(b"retained prediction")
    digest = inventory_artifacts.sha256_file(artifact)

    assert validate_environment.validate_hashes(
        root, {"prediction.npz": digest}
    ) == {"prediction.npz": digest}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_environment.validate_hashes(root, {"prediction.npz": "0" * 64})

    rows = inventory_artifacts.inventory(root, ["prediction.npz"])
    assert rows == [
        {
            "relative_path": "prediction.npz",
            "size_bytes": len(b"retained prediction"),
            "sha256": digest,
        }
    ]
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        inventory_artifacts.inventory(root, ["../outside"])
    with pytest.raises(ValueError, match="duplicates"):
        inventory_artifacts.inventory(root, ["prediction.npz", "prediction.npz"])


def test_retention_cli_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("one", encoding="utf-8")
    output = tmp_path / "manifest.json"
    assert inventory_artifacts.main(
        ["--root", str(root), "--output", str(output), "one.txt"]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "forestformer3d_retention_manifest_v1"
    assert payload["artifact_count"] == 1
    with pytest.raises(FileExistsError, match="already exists"):
        inventory_artifacts.main(
            ["--root", str(root), "--output", str(output), "one.txt"]
        )
