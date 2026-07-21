from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "methods/tls2trees/examples"
OUTPUTS = ROOT / "outputs/for_instance_benchmark_metrics"

DOCUMENTS = {
    "root": ROOT / "README.md",
    "registry": ROOT / "BENCHMARKS.md",
    "outputs": ROOT / "outputs/README.md",
    "method": ROOT / "methods/tls2trees/README.md",
    "examples": EXAMPLES / "README.md",
    "benchmark": ROOT / "methods/tls2trees/docs/for_instance_benchmark.md",
    "published_runbook": (
        ROOT
        / "methods/tls2trees/docs/for_instance_published_default_smoke.md"
    ),
}

PUBLISHED_SUFFIXES = (
    "test_plot_results.csv",
    "test_site_results.csv",
    "test_results.csv",
    "test_provenance.json",
    "prediction_retention_manifest.json",
    "leaf_off_test_plot_diagnostic.csv",
    "leaf_off_test_site_diagnostic.csv",
    "leaf_off_test_diagnostic.csv",
)

LEAF_SCREEN_FILES = (
    "tls2trees_development_leaf_screen_plot_results.csv",
    "tls2trees_development_leaf_screen_candidate_results.csv",
    "tls2trees_development_leaf_screen_provenance.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def document_text(name: str) -> str:
    return DOCUMENTS[name].read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_result(variant: str, suffix: str) -> dict[str, str]:
    rows = read_csv(EXAMPLES / f"tls2trees_{variant}_{suffix}")
    assert len(rows) == 1
    return rows[0]


def assert_contains(text: str, *tokens: str) -> None:
    missing = [token for token in tokens if token not in text]
    assert not missing, f"Documentation is missing: {missing}"


def test_development_tuned_documentation_tracks_the_public_result() -> None:
    result = one_result("development_tuned", "test_results.csv")
    mean_f1 = f'{float(result["mean_plot_f1"]):.6f}'
    micro_f1 = f'{float(result["micro_f1"]):.6f}'

    for name in ("root", "method", "examples", "benchmark"):
        assert_contains(document_text(name), mean_f1, micro_f1)

    assert_contains(
        document_text("registry"),
        result["run_id"],
        f'{float(result["mean_plot_f1"]):.4f}',
        f'{float(result["micro_f1"]):.4f}',
    )


def test_published_default_completion_updates_every_public_index() -> None:
    paths = [
        EXAMPLES / f"tls2trees_published_default_{suffix}"
        for suffix in PUBLISHED_SUFFIXES
    ]
    if not any(path.exists() for path in paths):
        return
    assert all(path.is_file() for path in paths)

    result = one_result("published_default", "test_results.csv")
    provenance = json.loads(
        (
            EXAMPLES / "tls2trees_published_default_test_provenance.json"
        ).read_text(encoding="utf-8")
    )
    retention = json.loads(
        (
            EXAMPLES
            / "tls2trees_published_default_prediction_retention_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert provenance["configuration_selected_from_for_instance_metrics"] is False
    assert provenance["configuration_changed_after_test"] is False
    for field, relative_path in {
        "published_config_sha256": (
            "methods/tls2trees/configs/for_instance_published_default.yml"
        ),
        "workflow_config_sha256": (
            "methods/tls2trees/configs/for_instance_published_default_test.yml"
        ),
        "benchmark_config_sha256": (
            "methods/tls2trees/configs/for_instance_benchmark.yml"
        ),
    }.items():
        assert provenance[field] == sha256(ROOT / relative_path)
    assert retention["status"] == "retention_verified"
    assert retention["verified_prediction_files"] == 22

    run_id = result["run_id"]
    mean_f1 = f'{float(result["mean_plot_f1"]):.6f}'
    micro_f1 = f'{float(result["micro_f1"]):.6f}'
    root_text = document_text("root")
    assert_contains(root_text, "seven completed", mean_f1, micro_f1)
    assert "frozen and ready for its separate Barkla execution" not in root_text

    artifact_names = [path.name for path in paths]
    registry_text = document_text("registry")
    assert_contains(
        registry_text,
        run_id,
        f'{float(result["mean_plot_f1"]):.4f}',
        f'{float(result["micro_f1"]):.4f}',
        *artifact_names,
    )
    assert "result pending separate Barkla execution" not in registry_text

    assert_contains(document_text("outputs"), *artifact_names)
    for name in ("method", "benchmark"):
        assert_contains(
            document_text(name),
            run_id,
            mean_f1,
            micro_f1,
            "tls2trees_published_default_test_results.csv",
            "tls2trees_published_default_test_provenance.json",
            "tls2trees_published_default_prediction_retention_manifest.json",
        )
    assert "its separate full-test workflow is ready" not in document_text(
        "benchmark"
    )
    for name in ("examples", "published_runbook"):
        assert_contains(
            document_text(name), run_id, mean_f1, micro_f1, *artifact_names
        )

    assert "after the separate Barkla run completes" not in document_text(
        "examples"
    )


def test_leaf_screen_completion_updates_its_public_indexes() -> None:
    paths = [EXAMPLES / name for name in LEAF_SCREEN_FILES]
    if not any(path.exists() for path in paths):
        return
    assert all(path.is_file() for path in paths)

    provenance = json.loads(paths[-1].read_text(encoding="utf-8"))
    assert provenance["status"] == "development_leaf_screen_publication_completed"
    assert provenance["valid_metric_count"] == 45
    assert provenance["held_out_test_accessed"] is False
    assert provenance["final_configuration_selected"] is False
    leaf_config = (
        ROOT
        / "methods/tls2trees/configs/"
        "for_instance_development_tuned_leaf_screen.yml"
    )
    assert provenance["publication_candidate_config"] == (
        "methods/tls2trees/configs/"
        "for_instance_development_tuned_leaf_screen.yml"
    )
    assert provenance["publication_candidate_config_sha256"] == sha256(
        leaf_config
    )

    artifact_names = [path.name for path in paths]
    for name in ("registry", "outputs", "method", "examples", "benchmark"):
        assert_contains(document_text(name), *artifact_names)
    assert_contains(document_text("registry"), provenance["workflow_run_id"])
    assert "public-safe export ready" not in document_text("registry")
    assert "will write public leaf-screen" not in document_text("examples")
