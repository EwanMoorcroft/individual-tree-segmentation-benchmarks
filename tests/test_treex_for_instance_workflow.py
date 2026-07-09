from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / relative_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_treex_config_records_exact_path_subset_and_unsupervised_mode() -> None:
    config = yaml.safe_load(
        (ROOT / "methods/treex/configs/for_instance_benchmark.yml").read_text(
            encoding="utf-8"
        )
    )

    assert config["project"]["benchmark_name"] == "for_instance_treex"
    assert config["dataset"]["exact_path_only"] is True
    assert config["dataset"]["allow_basename_fallback"] is False
    assert config["dataset"]["allow_nibio2_remap"] is False
    assert config["dataset"]["metadata_dev_plot_count"] == 56
    assert config["dataset"]["metadata_test_plot_count"] == 26
    assert config["dataset"]["local_dev_plot_count"] == 21
    assert config["dataset"]["local_test_plot_count"] == 11
    assert config["dataset"]["tree_classes"] == [4, 5, 6]
    assert config["method"]["algorithm"] == "TreeXAlgorithm"
    assert config["method"]["preset"] == "TreeXPresetULS"
    assert config["method"]["supervised"] is False
    assert config["method"]["package_version_status"] == (
        "not_captured_in_completed_exact_path_run"
    )
    assert "complete for current reporting" in config["method"]["package_version_note"]
    assert config["method"]["params"]["stem_search_min_cluster_intensity"] is None
    assert config["evaluation"]["primary_reporting_protocol"] == "strict"


def test_treex_required_files_are_present_and_tracked() -> None:
    relative_paths = [
        "methods/treex/README.md",
        "methods/treex/docs/for_instance_benchmark.md",
        "methods/treex/configs/for_instance_benchmark.yml",
        "methods/treex/scripts/run_treex_for_instance_plot.py",
        "methods/treex/scripts/evaluate_treex_for_instance_plot.py",
        "methods/treex/scripts/make_treex_for_instance_exact_split_lists.py",
        "methods/treex/scripts/create_treex_split_summary.py",
        "methods/treex/scripts/create_treex_final_summaries.py",
        "methods/treex/scripts/rsync_treex_predictions_from_barkla.sh",
        "methods/treex/scripts/rsync_treex_public_results_from_barkla.sh",
        "methods/treex/slurm/run_treex_for_instance_dev_array.sbatch",
        "methods/treex/slurm/run_treex_for_instance_test_array.sbatch",
        "methods/treex/slurm/evaluate_treex_for_instance_array.sbatch",
        "methods/treex/slurm/evaluate_treex_for_instance_test_array.sbatch",
        "methods/treex/examples/README.md",
        "methods/treex/examples/treex_combined_dev_test_summary.csv",
        "methods/treex/examples/treex_split_summary.csv",
        "methods/treex/examples/treex_site_summary.csv",
        "methods/treex/examples/treex_dev_full_summary.csv",
        "methods/treex/examples/treex_test_full_summary.csv",
        "methods/treex/examples/treex_best_plots_by_strict_f1.csv",
        "methods/treex/examples/treex_worst_plots_by_strict_f1.csv",
        "methods/treex/plots/README.md",
        "methods/treex/plots/treex_labelled_mask_f1_by_plot.png",
        "methods/treex/plots/treex_predicted_vs_reference_counts.png",
        "methods/treex/plots/treex_runtime_vs_strict_f1.png",
        "methods/treex/plots/treex_strict_f1_by_plot.png",
    ]
    assert all((ROOT / relative_path).is_file() for relative_path in relative_paths)


def test_treex_committed_result_summaries_match_recorded_aggregate_scores() -> None:
    combined_summary = (
        ROOT / "methods/treex/examples/treex_combined_dev_test_summary.csv"
    )
    split_summary = ROOT / "methods/treex/examples/treex_split_summary.csv"
    site_summary = ROOT / "methods/treex/examples/treex_site_summary.csv"

    with combined_summary.open("r", encoding="utf-8", newline="") as handle:
        combined_rows = list(csv.DictReader(handle))
    with split_summary.open("r", encoding="utf-8", newline="") as handle:
        split_rows = {row["split"]: row for row in csv.DictReader(handle)}
    with site_summary.open("r", encoding="utf-8", newline="") as handle:
        site_rows = list(csv.DictReader(handle))

    assert split_rows["dev"]["n_plots"] == "21"
    assert split_rows["dev"]["mean_f1_strict"] == "0.3408525717572193"
    assert split_rows["test"]["n_plots"] == "11"
    assert split_rows["test"]["mean_f1_strict"] == "0.4021745538352326"
    assert split_rows["test"]["mean_f1_labelled"] == "0.5221866540710387"
    assert len(combined_rows) == 32
    assert len({row["plot_id"] for row in combined_rows}) == 32
    assert sum(row["split"] == "dev" for row in combined_rows) == 21
    assert sum(row["split"] == "test" for row in combined_rows) == 11
    for row in combined_rows:
        true_positives = int(row["true_positives"])
        false_negatives = int(row["false_negatives"])
        false_positives_labelled = int(
            row["false_positives_labelled_mask"]
        )
        false_positives_strict = int(row["false_positives_strict"])
        assert true_positives + false_negatives == int(
            row["reference_trees"]
        )
        assert true_positives + false_positives_labelled == int(
            row["predicted_trees_on_labelled_mask"]
        )
        assert true_positives + false_positives_strict == int(
            row["predicted_trees_all"]
        )
        expected_precision_labelled = (
            true_positives
            / (true_positives + false_positives_labelled)
            if true_positives + false_positives_labelled
            else 0.0
        )
        expected_recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        )
        expected_f1_labelled = (
            2
            * expected_precision_labelled
            * expected_recall
            / (expected_precision_labelled + expected_recall)
            if expected_precision_labelled + expected_recall
            else 0.0
        )
        expected_precision_strict = (
            true_positives
            / (true_positives + false_positives_strict)
            if true_positives + false_positives_strict
            else 0.0
        )
        expected_f1_strict = (
            2
            * expected_precision_strict
            * expected_recall
            / (expected_precision_strict + expected_recall)
            if expected_precision_strict + expected_recall
            else 0.0
        )
        assert math.isclose(
            float(row["precision_labelled_mask"]),
            expected_precision_labelled,
        )
        assert math.isclose(float(row["recall_labelled_mask"]), expected_recall)
        assert math.isclose(float(row["f1_labelled_mask"]), expected_f1_labelled)
        assert math.isclose(
            float(row["precision_strict"]),
            expected_precision_strict,
        )
        assert math.isclose(float(row["recall_strict"]), expected_recall)
        assert math.isclose(float(row["f1_strict"]), expected_f1_strict)
    for split, aggregate in split_rows.items():
        rows = [row for row in combined_rows if row["split"] == split]
        assert int(aggregate["n_plots"]) == len(rows)
        assert int(aggregate["total_reference_trees"]) == sum(
            int(row["reference_trees"]) for row in rows
        )
        assert int(aggregate["total_predicted_trees_all"]) == sum(
            int(row["predicted_trees_all"]) for row in rows
        )
        assert math.isclose(
            float(aggregate["mean_f1_strict"]),
            sum(float(row["f1_strict"]) for row in rows) / len(rows),
        )
    assert any(
        row["split"] == "test"
        and row["site"] == "SCION"
        and row["mean_f1_strict"] == "0.5645889792231256"
        for row in site_rows
    )


def test_treex_scripts_show_help() -> None:
    for relative_path in (
        "methods/treex/scripts/run_treex_for_instance_plot.py",
        "methods/treex/scripts/evaluate_treex_for_instance_plot.py",
        "methods/treex/scripts/make_treex_for_instance_exact_split_lists.py",
        "methods/treex/scripts/make_treex_eval_list.py",
        "methods/treex/scripts/make_treex_test_eval_list.py",
        "methods/treex/scripts/create_treex_split_summary.py",
        "methods/treex/scripts/create_treex_final_summaries.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative_path), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout


def test_treex_slurm_calls_public_method_paths_and_scripts_parse() -> None:
    scripts = sorted((ROOT / "methods/treex/slurm").glob("*.sbatch"))
    assert len(scripts) == 4
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "scripts/methods/treex" not in text
        assert "methods/treex/scripts/" in text
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_treex_exact_split_and_evaluation_lists_match_runtime_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "CULS").mkdir(parents=True)
    (dataset_root / "NIBIO").mkdir(parents=True)
    (dataset_root / "CULS/plot_1.las").touch()
    (dataset_root / "NIBIO/plot_2.las").touch()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "path,folder,split\n"
        "CULS/plot_1.las,CULS,dev\n"
        "NIBIO/plot_2.las,NIBIO,test\n"
        "NIBIO/missing.las,NIBIO,dev\n",
        encoding="utf-8",
    )
    existing = tmp_path / "existing.csv"
    missing = tmp_path / "missing.csv"
    development = tmp_path / "development.csv"
    test = tmp_path / "test.csv"
    split_script = load_script(
        "methods/treex/scripts/make_treex_for_instance_exact_split_lists.py",
        "treex_split_lists",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_treex_for_instance_exact_split_lists.py",
            "--dataset-root",
            str(dataset_root),
            "--metadata-csv",
            str(metadata),
            "--existing-output",
            str(existing),
            "--missing-output",
            str(missing),
            "--dev-output",
            str(development),
            "--test-output",
            str(test),
        ],
    )
    assert split_script.main() == 0

    with development.open("r", encoding="utf-8", newline="") as handle:
        development_rows = list(csv.DictReader(handle))
    with test.open("r", encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    assert development_rows[0]["plot_id"] == "CULS/plot_1"
    assert Path(development_rows[0]["input_las"]).is_absolute()
    assert development_rows[0]["mapping_rule"] == "exact"
    assert test_rows[0]["plot_id"] == "NIBIO/plot_2"

    evaluation_list = tmp_path / "evaluation.csv"
    list_script = load_script(
        "methods/treex/scripts/make_treex_eval_list.py",
        "treex_eval_list",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_treex_eval_list.py",
            "--input-csv",
            str(development),
            "--output-csv",
            str(evaluation_list),
        ],
    )
    assert list_script.main() == 0
    with evaluation_list.open("r", encoding="utf-8", newline="") as handle:
        evaluation_rows = list(csv.DictReader(handle))
    assert evaluation_rows == [
        {
            "plot_id": "CULS/plot_1",
            "safe_plot_id": "CULS_plot_1",
            "summary_json": (
                "results/treex_for_instance/CULS_plot_1_treex_summary.json"
            ),
            "prediction_npz": (
                "data/predictions/treex/for_instance/CULS_plot_1/"
                "plot_1_treex_predictions.npz"
            ),
        }
    ]


def test_treex_evaluator_reproduces_labelled_and_strict_protocols(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prediction = tmp_path / "prediction.npz"
    np.savez(
        prediction,
        pred_tree_id=np.array([-1, 10, 10, 20, 20, 30]),
        target_tree_id=np.array([0, 1, 1, 2, 2, 0]),
        classification=np.array([1, 4, 4, 5, 5, 1]),
    )
    metrics_json = tmp_path / "metrics.json"
    metrics_csv = tmp_path / "metrics.csv"
    matches_csv = tmp_path / "matches.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    evaluator = load_script(
        "methods/treex/scripts/evaluate_treex_for_instance_plot.py",
        "treex_evaluator",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_treex_for_instance_plot.py",
            "--prediction-npz",
            str(prediction),
            "--plot-id",
            "synthetic/plot",
            "--metrics-json",
            str(metrics_json),
            "--metrics-csv",
            str(metrics_csv),
            "--matches-csv",
            str(matches_csv),
            "--diagnostics-csv",
            str(diagnostics_csv),
        ],
    )
    assert evaluator.main() == 0
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert metrics["reference_trees"] == 2
    assert metrics["predicted_trees_on_labelled_mask"] == 2
    assert metrics["predicted_trees_all"] == 3
    assert metrics["true_positives"] == 2
    assert metrics["f1_labelled_mask"] == 1.0
    assert metrics["f1_strict"] == 0.8


def test_treex_split_summary_combines_validated_per_plot_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plot_list = tmp_path / "plots.csv"
    plot_list.write_text(
        "plot_id,input_las\nCULS/plot_1,/dataset/CULS/plot_1.las\n",
        encoding="utf-8",
    )
    results_root = tmp_path / "results"
    results_root.mkdir()
    summary = {
        "plot_id": "CULS/plot_1",
        "total_points": 100,
        "tree_class_points": 80,
        "reference_tree_count_tree_classes": 2,
        "predicted_instances": 3,
        "elapsed_seconds": 4.5,
    }
    metrics = {
        "plot_id": "CULS/plot_1",
        "reference_trees": 2,
        "predicted_trees_all": 3,
        "predicted_trees_on_labelled_mask": 2,
        "true_positives": 2,
        "false_positives_labelled_mask": 0,
        "false_positives_strict": 1,
        "false_negatives": 0,
        "precision_labelled_mask": 1.0,
        "recall_labelled_mask": 1.0,
        "f1_labelled_mask": 1.0,
        "precision_strict": 2 / 3,
        "recall_strict": 1.0,
        "f1_strict": 0.8,
        "mean_matched_iou": 0.75,
    }
    (results_root / "CULS_plot_1_treex_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (results_root / "CULS_plot_1_treex_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    output = tmp_path / "summary.csv"
    collector = load_script(
        "methods/treex/scripts/create_treex_split_summary.py",
        "treex_split_summary",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_treex_split_summary.py",
            "--plot-list",
            str(plot_list),
            "--results-root",
            str(results_root),
            "--output-csv",
            str(output),
            "--split",
            "dev",
        ],
    )
    assert collector.main() == 0
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["plot_id"] == "CULS/plot_1"
    assert rows[0]["reference_trees"] == "2"
    assert rows[0]["f1_strict"] == "0.8"


def test_treex_public_examples_exclude_machine_paths_and_intermediates() -> None:
    examples = ROOT / "methods/treex/examples"
    published = "\n".join(
        path.read_text(encoding="utf-8")
        for path in examples.glob("*")
        if path.suffix in {".csv", ".md"}
    )
    assert "/users/" not in published
    assert "/mnt/" not in published
    assert "/Users/" not in published
    assert "prediction_npz" not in published
    assert not any("pilot" in path.name for path in examples.iterdir())
