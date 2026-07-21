from __future__ import annotations

import csv
import json
import math
import re
import subprocess
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
CANONICAL_RESULT_VARIANTS = {
    "development_tuned",
    "unsupervised_parameterised",
    "published_default",
    "published_pretrained",
    "fine_tuned_on_dev",
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
    assert (
        "for-instance | treelearn | "
        "treelearn_for-instance_published_pretrained_development_20260712_150030"
        in registry
    )
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


def test_accepted_sat_summary_and_provenance_reconcile() -> None:
    examples = ROOT / "methods/segmentanytree/examples"
    summary_path = examples / (
        "sat_final_test_aligned_summary_"
        "sat_for_quicktune_to49_20260706_140730.csv"
    )
    provenance_path = examples / (
        "sat_final_test_aligned_provenance_"
        "sat_for_quicktune_to49_20260706_140730.json"
    )
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    tp, fp, fn = (int(row[field]) for field in ("total_tp", "total_fp", "total_fn"))
    assert math.isclose(
        float(row["micro_precision"]), tp / (tp + fp)
    )
    assert math.isclose(float(row["micro_recall"]), tp / (tp + fn))
    assert math.isclose(
        float(row["micro_f1"]), 2 * tp / ((2 * tp) + fp + fn)
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["run_id"] == row["run_id"]
    assert provenance["final_evaluation_id"] == row["final_id"]
    assert provenance["checkpoint"]["sha256"] == row["checkpoint_sha256"]
    assert provenance["evaluation"]["micro_f1"] == float(row["micro_f1"])
    assert provenance["diagnostic_snapshot"]["status"] == (
        "pre_final_evaluation_id_not_canonical"
    )
    assert provenance["public_evidence"]["final_per_plot_tables"].startswith(
        "not_available_in_local_checkout"
    )


def test_repository_publication_metadata_is_present() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == "Individual Tree Segmentation Benchmarks"
    assert citation["repository-code"].endswith(
        "/individual-tree-segmentation-benchmarks"
    )
    assert (ROOT / "docs/README.md").is_file()
    assert (ROOT / "outputs/README.md").is_file()


def test_cross_method_outputs_use_method_neutral_paths_and_canonical_variants() -> None:
    output_root = ROOT / "outputs/for_instance_benchmark_metrics"
    assert output_root.is_dir()
    assert not (ROOT / "outputs/sat_treex_benchmark_metrics").exists()

    with (output_root / "for_instance_method_development_diagnostics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        diagnostics = list(csv.DictReader(handle))
    assert diagnostics
    assert {row["variant"] for row in diagnostics} <= CANONICAL_RESULT_VARIANTS
    assert all(
        row["variant"] == row["training_mode"]
        or (
            row["method_slug"] == "tls2trees"
            and row["variant"] in {"development_tuned", "published_default"}
            and row["training_mode"] == "external_training_only"
        )
        for row in diagnostics
    )

    with (output_root / "for_instance_prediction_retention_registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        retention = list(csv.DictReader(handle))
    assert retention
    assert {row["variant"] for row in retention} <= CANONICAL_RESULT_VARIANTS
    assert all(row["retention_profile"] for row in retention)
    assert len(
        {
            (row["method_slug"], row["retention_profile"], row["run_id"])
            for row in retention
        }
    ) == len(retention)


def test_segmentanytree_result_registry_is_method_scoped() -> None:
    path = ROOT / "methods/segmentanytree/examples/segmentanytree_result_registry.csv"
    assert path.is_file()
    assert not (
        ROOT / "methods/segmentanytree/examples/for_instance_result_registry.csv"
    ).exists()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["method_slug"] for row in rows} == {"segmentanytree"}


def test_tracked_repository_excludes_private_or_large_runtime_artifacts() -> None:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [path for path in completed.stdout.splitlines() if path]
    forbidden_suffixes = (
        ".las",
        ".laz",
        ".ply",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".ckpt",
        ".sif",
        ".zip",
        ".tar",
        ".tar.gz",
        ".log",
        ".err",
        ".out",
    )
    assert not [path for path in tracked if path.lower().endswith(forbidden_suffixes)]
    assert not [path for path in tracked if Path(path).name.startswith("~$")]

    text_suffixes = {
        ".cff",
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sbatch",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    public_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8", errors="strict")
        for path in tracked
        if (ROOT / path).is_file() and (ROOT / path).suffix in text_suffixes
    )
    assert re.search(r"/Users/[A-Za-z0-9._-]+/", public_text) is None
    assert "pending_pointwise_" + "revalidation" not in public_text


def test_tls2trees_public_surface_uses_neutral_current_terminology() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    repository_paths = [
        Path(value) for value in completed.stdout.splitlines() if value
    ]
    text_suffixes = {
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sbatch",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    tls2trees_paths = [
        path
        for path in repository_paths
        if path.parts[:2] == ("methods", "tls2trees")
        and (ROOT / path).is_file()
    ]
    assert tls2trees_paths

    migration_path_pattern = re.compile(
        r"(?:^|[/_.-])(?:v[0-9]+|rescor(?:e|ed|ing)?|correct(?:ed|ion)|"
        r"supersed(?:e|ed|es|ing)|invalidat(?:e|ed|ion|ing))(?:[/_.-]|$)",
        flags=re.IGNORECASE,
    )
    assert not [
        path.as_posix()
        for path in tls2trees_paths
        if migration_path_pattern.search(
            re.sub(
                r"\bleaf_v[0-9]+_e[0-9]+\b",
                "leaf_parameter_candidate",
                path.as_posix(),
                flags=re.IGNORECASE,
            )
        )
    ]

    forbidden_content_patterns = (
        re.compile(
            r"for_instance_(?:pointwise|tls2trees_source_row)_v[0-9]+",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:rescor(?:e|ed|ing)?|corrected|correction|"
            r"supersed(?:e|ed|es|ing)|invalidat(?:e|ed|ion|ing))\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:protocol|evaluator|evaluation|result|retention|alignment|"
            r"metric|prediction)[-_ ]*v[0-9]+\b",
            flags=re.IGNORECASE,
        ),
    )
    private_path_patterns = (
        re.compile(r"/(?:Users|users|home)/[A-Za-z0-9._-]+/"),
        re.compile(r"/mnt/(?:scratch/)?users/[A-Za-z0-9._-]+/"),
    )

    violations: list[str] = []
    for relative_path in repository_paths:
        path = ROOT / relative_path
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if relative_path.parts[:2] == ("methods", "tls2trees"):
            scoped_text = text
        else:
            is_public_result_or_documentation = (
                relative_path.as_posix() in {"README.md", "BENCHMARKS.md"}
                or relative_path.parts[:1] in {("datasets",), ("docs",), ("outputs",)}
            )
            if not is_public_result_or_documentation:
                continue
            scoped_text = "\n".join(
                line for line in text.splitlines() if "tls2trees" in line.lower()
            )
            if not scoped_text:
                continue

        # Candidate IDs encode voxel and edge lengths, not protocol generations.
        scoped_text = re.sub(
            r"\bleaf_v[0-9]+_e[0-9]+\b",
            "leaf_parameter_candidate",
            scoped_text,
            flags=re.IGNORECASE,
        )
        for pattern in (*forbidden_content_patterns, *private_path_patterns):
            match = pattern.search(scoped_text)
            if match:
                violations.append(
                    f"{relative_path.as_posix()}: {match.group(0)!r}"
                )
    assert not violations, "TLS2trees public-surface violations:\n" + "\n".join(
        violations
    )
