from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_METHOD_DIRS = {"configs", "docs", "examples", "scripts", "slurm"}
REQUIRED_README_SECTIONS = {
    "Method Summary",
    "Upstream Repository And Citation",
    "Training Mode Support",
    "Input Requirements",
    "Output Contract",
    "FOR-instance Compatibility",
    "Barkla Environment",
    "Slurm Workflow",
    "Evaluation Route",
    "Known Limitations",
    "Current Benchmark Status",
}
KNOWN_DATASET_SLUGS = {"for-instance", "frdr-treeiso", "wytham-woods"}
KNOWN_METHOD_SLUGS = {
    "segmentanytree",
    "tls2trees",
    "treex",
    "treelearn",
    "forainet",
    "pointgroup",
    "softgroup",
    "hais",
    "mask3d",
    "randlanet",
    "pointnetpp",
    "forestformer3d",
    "segmentanytreev2",
}
REGISTRY_COLUMNS = (
    "| Dataset slug | Method slug | Run label | Training mode | "
    "Evaluation mode | Status | Evidence |"
)
STALE_STATUS_TEXT = "full development-only training running and validation queued"


def load_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))


def test_registry_uses_explicit_method_agnostic_schema() -> None:
    registry = (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")

    assert REGISTRY_COLUMNS in registry
    assert "for-instance | segmentanytree" in registry
    assert "for-instance | treex" in registry
    assert "for-instance | treelearn | pending" in registry
    assert "Candidate accuracy benchmark" in registry
    assert "Prediction benchmark completed" in registry


def test_method_folders_have_public_shape_and_readme_contract() -> None:
    for method_dir in sorted((ROOT / "methods").iterdir()):
        if not method_dir.is_dir():
            continue

        present_dirs = {path.name for path in method_dir.iterdir() if path.is_dir()}
        assert REQUIRED_METHOD_DIRS <= present_dirs, method_dir

        readme = method_dir / "README.md"
        assert readme.is_file(), method_dir
        text = readme.read_text(encoding="utf-8")
        for section in REQUIRED_README_SECTIONS:
            assert f"## {section}" in text, f"{readme}: missing {section}"


def test_dataset_configs_use_stable_slugs_and_candidate_method_slugs() -> None:
    for relative_path in (
        "datasets/for-instance/benchmark.yml",
        "datasets/wytham-woods/benchmark.yml",
    ):
        payload = load_yaml(relative_path)
        dataset_slug = payload["dataset"]["slug"]
        candidate_slugs = set(payload["candidate_method_slugs"])

        assert dataset_slug in KNOWN_DATASET_SLUGS
        assert candidate_slugs <= KNOWN_METHOD_SLUGS
        assert "method_candidates" not in payload
        assert "segment_any_tree" not in candidate_slugs
        assert "treelearn_or_other_deep_learning" not in candidate_slugs


def test_method_configs_expose_matching_project_dataset_and_method_slugs() -> None:
    config_paths = sorted((ROOT / "methods").glob("*/configs/*.yml"))
    assert config_paths

    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        project = payload["project"]
        dataset = payload["dataset"]
        method = payload["method"]

        assert project["dataset_slug"] == dataset["slug"], path
        assert project["method_slug"] == method["slug"], path
        assert dataset["slug"] in KNOWN_DATASET_SLUGS, path
        assert method["slug"] in KNOWN_METHOD_SLUGS, path
        assert method["slug"] == path.relative_to(ROOT).parts[1], path


def test_dataset_feasibility_no_longer_reports_stale_sat_training_status() -> None:
    text = (ROOT / "docs/dataset_feasibility.md").read_text(encoding="utf-8")

    assert STALE_STATUS_TEXT not in text
    assert "sat_for_quicktune_to49_20260706_140730" in text
    assert "TreeX deterministic baseline" in text
