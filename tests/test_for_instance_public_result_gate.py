from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "for_instance_pointwise_v1"
MATCHING = "maximum_cardinality_one_to_one"
MASK = "union_of_reference_tree_and_predicted_tree_points"
IOU_THRESHOLD = 0.5
HEX_64 = re.compile(r"[0-9a-f]{64}")

EXPECTED_SITE_PLOTS = {
    "CULS": 1,
    "NIBIO": 6,
    "RMIT": 1,
    "SCION": 2,
    "TUWIEN": 1,
}
EXPECTED_SITE_REFERENCES = {
    "CULS": 20,
    "NIBIO": 161,
    "RMIT": 64,
    "SCION": 43,
    "TUWIEN": 35,
}

SPECS = {
    ("treex", "external_training_only"): {
        "run_id": "treex_for_instance_exact_path_subset",
        "expected": (653, 323, 177, 476, 146),
        "mean_plot_f1": 0.38310765715157086,
        "micro_f1": 0.36270491803278687,
        "overall_path": "methods/treex/examples/treex_split_summary.csv",
        "site_path": "methods/treex/examples/treex_site_summary.csv",
        "summary_kind": "treex",
    },
    ("segmentanytree", "published_pretrained"): {
        "run_id": "segmentanytree_for-instance_published_pretrained_20260710_231601",
        "expected": (789, 323, 247, 542, 76),
        "mean_plot_f1": 0.45340897930934665,
        "micro_f1": 0.4442446043165468,
        "overall_path": (
            "methods/segmentanytree/examples/"
            "sat_completed_target_results_20260711.csv"
        ),
        "site_path": (
            "methods/segmentanytree/examples/"
            "sat_completed_target_site_results_20260711.csv"
        ),
        "summary_kind": "segmentanytree",
    },
    ("segmentanytree", "fine_tuned_on_dev"): {
        "run_id": "segmentanytree_for-instance_fine_tuned_on_dev_20260711_002931",
        "expected": (568, 323, 237, 331, 86),
        "mean_plot_f1": 0.5446789009405125,
        "micro_f1": 0.531986531986532,
        "overall_path": (
            "methods/segmentanytree/examples/"
            "sat_completed_target_results_20260711.csv"
        ),
        "site_path": (
            "methods/segmentanytree/examples/"
            "sat_completed_target_site_results_20260711.csv"
        ),
        "summary_kind": "segmentanytree",
    },
    ("treelearn", "published_pretrained"): {
        "run_id": "treelearn_for-instance_published_pretrained_20260714_134109",
        "expected": (366, 323, 34, 332, 289),
        "mean_plot_f1": 0.07894350692791946,
        "micro_f1": 0.09869375907111756,
        "overall_path": (
            "methods/treelearn/examples/"
            "treelearn_pretrained_test_results_20260714.csv"
        ),
        "site_path": (
            "methods/treelearn/examples/"
            "treelearn_pretrained_test_site_results_20260714.csv"
        ),
        "summary_kind": "treelearn",
    },
    ("treelearn", "fine_tuned_on_dev"): {
        "run_id": "treelearn_for-instance_fine_tuned_on_dev_long_20260712_233227",
        "expected": (623, 323, 157, 466, 166),
        "mean_plot_f1": 0.36468463620440583,
        "micro_f1": 0.33192389006342493,
        "overall_path": (
            "methods/treelearn/examples/"
            "treelearn_finetuned_test_results_20260713.csv"
        ),
        "site_path": (
            "methods/treelearn/examples/"
            "treelearn_finetuned_test_site_results_20260713.csv"
        ),
        "summary_kind": "treelearn",
    },
}

TLS2TREES_EXAMPLES = "methods/tls2trees/examples"
TLS2TREES_PROTOCOL = "for_instance_pointwise_class3_ignore"
TLS2TREES_EVALUATOR = "for_instance_tls2trees_source_row_class3_ignore"
TLS2TREES_MASK = (
    "classification_3_excluded_then_union_of_reference_tree_and_predicted_tree_points"
)
TLS2TREES_COMPARABLE_GROUP = "held_out_test_tls2trees_class3_ignore"
TLS2TREES_DIAGNOSTIC_GROUP = "tls2trees_leaf_off_class3_ignore_diagnostic"
TLS2TREES_RETENTION_PROFILE = "held_out_test_tls2trees_class3_ignore"
TLS2TREES_VARIANTS = ("development_tuned", "published_default")
TLS2TREES_BUNDLE_SUFFIXES = (
    "test_plot_results.csv",
    "test_site_results.csv",
    "test_results.csv",
    "test_provenance.json",
    "prediction_retention_manifest.json",
    "leaf_off_test_plot_diagnostic.csv",
    "leaf_off_test_site_diagnostic.csv",
    "leaf_off_test_diagnostic.csv",
)


def tls2trees_prefix(variant: str) -> str:
    assert variant in TLS2TREES_VARIANTS
    return f"{TLS2TREES_EXAMPLES}/tls2trees_{variant}"


def tls2trees_bundle_paths(variant: str) -> list[Path]:
    prefix = tls2trees_prefix(variant)
    return [ROOT / f"{prefix}_{suffix}" for suffix in TLS2TREES_BUNDLE_SUFFIXES]


def published_default_bundle_present() -> bool:
    """Allow the baseline to be absent, but never partially published."""
    artifact_flags = [path.is_file() for path in tls2trees_bundle_paths("published_default")]
    headline = [
        row
        for row in read_csv(
            "outputs/for_instance_benchmark_metrics/"
            "for_instance_method_benchmark_results.csv"
        )
        if row["method_slug"] == "tls2trees"
        and row["variant"] == "published_default"
    ]
    diagnostic = [
        row
        for row in read_csv(
            "outputs/for_instance_benchmark_metrics/"
            "for_instance_method_development_diagnostics.csv"
        )
        if row["method_slug"] == "tls2trees"
        and row["variant"] == "published_default"
    ]
    retention = [
        row
        for row in read_csv(
            "outputs/for_instance_benchmark_metrics/"
            "for_instance_prediction_retention_registry.csv"
        )
        if row["method_slug"] == "tls2trees"
        and row["variant"] == "published_default"
    ]
    signals = artifact_flags + [bool(headline), bool(diagnostic), bool(retention)]
    if not any(signals):
        return False
    assert all(artifact_flags), "TLS2trees published-default artifact bundle is partial"
    assert len(headline) == len(diagnostic) == len(retention) == 1, (
        "TLS2trees published-default rows must be published atomically"
    )
    return True


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_plot_id(relative_path: str) -> str:
    return Path(relative_path).with_suffix("").as_posix()


def normalize_aligned_plot(row: dict[str, str]) -> dict[str, object]:
    return {
        "plot_id": as_plot_id(row["relative_path"]),
        "site": row["collection"],
        "predicted_instances": int(row["predicted_instances"]),
        "reference_instances": int(row["reference_instances"]),
        "true_positives": int(row["true_positives"]),
        "false_positives": int(row["false_positives"]),
        "false_negatives": int(row["false_negatives"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def normalize_treex_plot(row: dict[str, str]) -> dict[str, object]:
    return {
        "plot_id": row["plot_id"],
        "site": row["site"],
        "predicted_instances": int(
            row["predicted_trees_harmonized_union_mask"]
        ),
        "reference_instances": int(row["reference_trees"]),
        "true_positives": int(row["true_positives_harmonized"]),
        "false_positives": int(row["false_positives_harmonized"]),
        "false_negatives": int(row["false_negatives_harmonized"]),
        "precision": float(row["precision_harmonized"]),
        "recall": float(row["recall_harmonized"]),
        "f1": float(row["f1_harmonized"]),
    }


def load_plot_sources() -> tuple[
    dict[tuple[str, str], list[dict[str, object]]], list[dict[str, str]]
]:
    source_rows: list[dict[str, str]] = []
    results: dict[tuple[str, str], list[dict[str, object]]] = {}

    sat_rows = read_csv(
        "methods/segmentanytree/examples/"
        "sat_completed_target_plot_results_20260711.csv"
    )
    source_rows.extend(sat_rows)
    assert Counter(row["variant"] for row in sat_rows) == {
        "published_pretrained": 11,
        "fine_tuned_on_dev": 11,
    }
    for variant in ("published_pretrained", "fine_tuned_on_dev"):
        rows = [row for row in sat_rows if row["variant"] == variant]
        key = ("segmentanytree", variant)
        assert [int(row["plot_index"]) for row in rows] == list(range(11))
        assert {row["method_slug"] for row in rows} == {key[0]}
        assert {row["run_id"] for row in rows} == {SPECS[key]["run_id"]}
        results[key] = [
            normalize_aligned_plot(row) for row in rows
        ]

    treelearn_sources = (
        (
            "published_pretrained",
            "methods/treelearn/examples/"
            "treelearn_pretrained_test_plot_results_20260714.csv",
        ),
        (
            "fine_tuned_on_dev",
            "methods/treelearn/examples/"
            "treelearn_finetuned_test_plot_results_20260713.csv",
        ),
    )
    for variant, relative_path in treelearn_sources:
        rows = read_csv(relative_path)
        key = ("treelearn", variant)
        source_rows.extend(rows)
        assert len(rows) == 11
        assert {row["variant"] for row in rows} == {variant}
        assert {row["method_slug"] for row in rows} == {key[0]}
        assert {row["run_id"] for row in rows} == {SPECS[key]["run_id"]}
        assert [int(row["plot_index"]) for row in rows] == list(range(11))
        results[key] = [
            normalize_aligned_plot(row) for row in rows
        ]

    treex_rows = read_csv(
        "methods/treex/examples/treex_combined_dev_test_summary.csv"
    )
    assert len(treex_rows) == 32
    assert len({row["plot_id"] for row in treex_rows}) == 32
    results[("treex", "external_training_only")] = [
        normalize_treex_plot(row) for row in treex_rows if row["split"] == "test"
    ]
    return results, source_rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, int | float]:
    predicted = sum(int(row["predicted_instances"]) for row in rows)
    references = sum(int(row["reference_instances"]) for row in rows)
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        "plots": len(rows),
        "predicted_instances": predicted,
        "reference_instances": references,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_plot_f1": statistics.fmean(float(row["f1"]) for row in rows),
        "micro_precision": (
            tp / precision_denominator if precision_denominator else 0.0
        ),
        "micro_recall": tp / recall_denominator if recall_denominator else 0.0,
        "micro_f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
    }


def test_aggregate_accepts_a_site_with_no_predictions() -> None:
    calculated = aggregate(
        [
            {
                "predicted_instances": 0,
                "reference_instances": 64,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 64,
                "f1": 0.0,
            }
        ]
    )
    assert calculated["micro_precision"] == 0.0
    assert calculated["micro_recall"] == 0.0
    assert calculated["micro_f1"] == 0.0


def assert_plot_rows(rows: list[dict[str, object]]) -> None:
    assert len(rows) == 11
    assert len({str(row["plot_id"]) for row in rows}) == 11
    assert Counter(str(row["site"]) for row in rows) == EXPECTED_SITE_PLOTS
    for row in rows:
        predicted = int(row["predicted_instances"])
        references = int(row["reference_instances"])
        tp = int(row["true_positives"])
        fp = int(row["false_positives"])
        fn = int(row["false_negatives"])
        assert predicted == tp + fp
        assert references == tp + fn
        expected_precision = tp / predicted if predicted else 0.0
        assert float(row["precision"]) == pytest.approx(expected_precision)
        assert float(row["recall"]) == pytest.approx(tp / references)
        denominator = 2 * tp + fp + fn
        expected_f1 = 2 * tp / denominator if denominator else 0.0
        assert float(row["f1"]) == pytest.approx(expected_f1)


def normalize_overall(spec: dict[str, object]) -> dict[str, str]:
    rows = read_csv(str(spec["overall_path"]))
    if spec["summary_kind"] == "treex":
        row = next(row for row in rows if row["split"] == "test")
        return {
            "plots": row["n_plots"],
            "predicted_instances": row["total_predicted_trees_harmonized"],
            "reference_instances": row["total_reference_trees"],
            "true_positives": row["total_tp_harmonized"],
            "false_positives": row["total_fp_harmonized"],
            "false_negatives": row["total_fn_harmonized"],
            "mean_plot_f1": row["mean_plot_f1_harmonized"],
            "micro_precision": row["micro_precision_harmonized"],
            "micro_recall": row["micro_recall_harmonized"],
            "micro_f1": row["micro_f1_harmonized"],
        }
    row = next(row for row in rows if row["run_id"] == spec["run_id"])
    return row


def normalize_site_rows(spec: dict[str, object]) -> dict[str, dict[str, str]]:
    rows = read_csv(str(spec["site_path"]))
    kind = str(spec["summary_kind"])
    if kind == "treex":
        rows = [row for row in rows if row["split"] == "test"]
        normalized = {
            row["site"]: {
                "plots": row["n_plots"],
                "predicted_instances": row[
                    "total_predicted_trees_harmonized"
                ],
                "reference_instances": row["total_reference_trees"],
                "true_positives": row["total_tp_harmonized"],
                "false_positives": row["total_fp_harmonized"],
                "false_negatives": row["total_fn_harmonized"],
                "mean_plot_f1": row["mean_plot_f1_harmonized"],
                "micro_precision": row["micro_precision_harmonized"],
                "micro_recall": row["micro_recall_harmonized"],
                "micro_f1": row["micro_f1_harmonized"],
            }
            for row in rows
        }
    else:
        if kind == "segmentanytree":
            rows = [
                row
                for row in rows
                if row["variant"] == str(spec["training_mode"])
            ]
        else:
            rows = [row for row in rows if row["run_id"] == spec["run_id"]]
        normalized = {}
        for row in rows:
            if kind == "treelearn":
                assert row["expected_plots"] == row["completed_plots"]
                assert row["failed_plots"] == "0"
                plots = row["completed_plots"]
            else:
                plots = row["plots"]
            normalized[row["site"]] = {
                "plots": plots,
                "predicted_instances": row["predicted_instances"],
                "reference_instances": row["reference_instances"],
                "true_positives": row["true_positives"],
                "false_positives": row["false_positives"],
                "false_negatives": row["false_negatives"],
                "mean_plot_f1": row["mean_plot_f1"],
                "micro_precision": row["micro_precision"],
                "micro_recall": row["micro_recall"],
                "micro_f1": row["micro_f1"],
            }
    return normalized


def assert_summary_matches(
    calculated: dict[str, int | float], recorded: dict[str, str]
) -> None:
    for field in (
        "plots",
        "predicted_instances",
        "reference_instances",
        "true_positives",
        "false_positives",
        "false_negatives",
    ):
        assert int(recorded[field]) == calculated[field]
    for field in (
        "mean_plot_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ):
        assert float(recorded[field]) == pytest.approx(calculated[field])


def test_headline_rows_reconcile_from_committed_plot_sources() -> None:
    plot_sources, source_rows = load_plot_sources()
    assert set(plot_sources) == set(SPECS)

    aligned_protocol_fields = {
        (
            row["dataset_split"],
            row["evaluation_protocol"],
            row["matching_policy"],
            row["evaluation_mask"],
            float(row["iou_threshold"]),
        )
        for row in source_rows
    }
    assert aligned_protocol_fields == {
        ("test", PROTOCOL, MATCHING, MASK, IOU_THRESHOLD)
    }

    treex_config = yaml.safe_load(
        (ROOT / "methods/treex/configs/for_instance_benchmark.yml").read_text(
            encoding="utf-8"
        )
    )
    assert treex_config["project"]["protocol_id"] == PROTOCOL
    assert treex_config["evaluation"]["matching"] == MATCHING
    assert treex_config["evaluation"]["primary_evaluation_mask"] == MASK
    assert treex_config["evaluation"]["iou_threshold"] == IOU_THRESHOLD

    tracker_rows = read_csv(
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_benchmark_results.csv"
    )
    tracker = {
        (row["method_slug"], row["training_mode"]): row
        for row in tracker_rows
        if row["method_slug"] != "tls2trees"
    }
    tls2trees_rows = [row for row in tracker_rows if row["method_slug"] == "tls2trees"]
    expected_tls2trees_rows = 1 + int(published_default_bundle_present())
    assert len(tracker_rows) == len(SPECS) + expected_tls2trees_rows
    assert len(tls2trees_rows) == expected_tls2trees_rows
    assert set(tracker) == set(SPECS)

    reference_inventory: dict[str, int] | None = None
    for key, spec in SPECS.items():
        rows = plot_sources[key]
        assert_plot_rows(rows)
        current_inventory = {
            str(row["plot_id"]): int(row["reference_instances"]) for row in rows
        }
        if reference_inventory is None:
            reference_inventory = current_inventory
        else:
            assert current_inventory == reference_inventory

        calculated = aggregate(rows)
        predicted, references, tp, fp, fn = spec["expected"]
        assert calculated["plots"] == 11
        assert (
            calculated["predicted_instances"],
            calculated["reference_instances"],
            calculated["true_positives"],
            calculated["false_positives"],
            calculated["false_negatives"],
        ) == (predicted, references, tp, fp, fn)
        assert references == 323
        assert calculated["mean_plot_f1"] == pytest.approx(
            spec["mean_plot_f1"]
        )
        assert calculated["micro_f1"] == pytest.approx(spec["micro_f1"])

        assert_summary_matches(calculated, normalize_overall(spec))
        assert_summary_matches(calculated, tracker[key])
        assert tracker[key]["run_id"] == spec["run_id"]
        assert tracker[key]["evaluation_protocol"] == PROTOCOL
        assert tracker[key]["matching_policy"] == MATCHING
        assert tracker[key]["evaluation_mask"] == MASK
        assert tracker[key]["evaluation_split"] == "test"
        assert tracker[key]["comparable_group"] == "held_out_test_primary"

        site_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            site_rows[str(row["site"])].append(row)
        assert set(site_rows) == set(EXPECTED_SITE_PLOTS)
        recorded_sites = normalize_site_rows({**spec, "training_mode": key[1]})
        assert set(recorded_sites) == set(EXPECTED_SITE_PLOTS)
        for site, expected_plots in EXPECTED_SITE_PLOTS.items():
            site_calculated = aggregate(site_rows[site])
            assert site_calculated["plots"] == expected_plots
            assert (
                site_calculated["reference_instances"]
                == EXPECTED_SITE_REFERENCES[site]
            )
            assert_summary_matches(site_calculated, recorded_sites[site])


def test_public_source_hashes_and_treex_retention_manifest_reconcile() -> None:
    _, source_rows = load_plot_sources()
    source_hashes = [row["source_metrics_sha256"] for row in source_rows]
    expected_hashes = 44
    assert len(source_hashes) == expected_hashes
    assert len(set(source_hashes)) == expected_hashes
    assert all(HEX_64.fullmatch(value) for value in source_hashes)

    manifest = json.loads(
        (
            ROOT
            / "methods/treex/examples/treex_prediction_retention_manifest.json"
        ).read_text(encoding="utf-8")
    )
    files = manifest["files"]
    assert manifest["status"] == "retention_verified"
    assert manifest["expected_plots"] == 32
    assert manifest["verified_prediction_files"] == len(files) == 64
    assert manifest["verified_prediction_size_bytes"] == 6_745_673_181
    assert sum(int(row["size_bytes"]) for row in files) == 6_745_673_181
    assert len({row["relative_path"] for row in files}) == 64
    assert all(HEX_64.fullmatch(row["sha256"]) for row in files)

    by_plot: dict[str, set[str]] = defaultdict(set)
    for row in files:
        assert row["split"] in {"dev", "test"}
        assert row["format"] in {"las", "npz"}
        by_plot[row["plot_id"]].add(row["format"])
    assert len(by_plot) == 32
    assert all(formats == {"las", "npz"} for formats in by_plot.values())

    retention_rows = read_csv(
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_prediction_retention_registry.csv"
    )
    treex = next(row for row in retention_rows if row["method_slug"] == "treex")
    assert treex["run_id"] == "treex_for_instance_exact_path_subset"
    assert int(treex["retained_file_count"]) == len(files)
    assert int(treex["retained_size_bytes"]) == sum(
        int(row["size_bytes"]) for row in files
    )
    assert treex["hash_status"] == "sha256_verified"


def tls2trees_output_rows(
    relative_path: str, variant: str
) -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(relative_path)
        if row["method_slug"] == "tls2trees" and row["variant"] == variant
    ]


def assert_tls2trees_protocol_fields(rows: list[dict[str, str]]) -> None:
    assert rows
    assert {row["evaluation_protocol"] for row in rows} == {
        TLS2TREES_PROTOCOL
    }
    assert {row["matching_policy"] for row in rows} == {MATCHING}
    assert {row["evaluation_mask"] for row in rows} == {TLS2TREES_MASK}


def assert_tls2trees_sites(
    plot_rows: list[dict[str, str]], site_relative_path: str
) -> None:
    normalized = [normalize_aligned_plot(row) for row in plot_rows]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["site"])].append(row)
    assert {site: len(rows) for site, rows in grouped.items()} == (
        EXPECTED_SITE_PLOTS
    )
    recorded = {row["site"]: row for row in read_csv(site_relative_path)}
    assert set(recorded) == set(EXPECTED_SITE_PLOTS)
    for site, rows in grouped.items():
        calculated = aggregate(rows)
        assert calculated["reference_instances"] == EXPECTED_SITE_REFERENCES[site]
        assert_summary_matches(calculated, recorded[site])


def assert_tls2trees_bundle(variant: str) -> None:
    prefix = tls2trees_prefix(variant)
    paths = tls2trees_bundle_paths(variant)
    assert all(path.is_file() for path in paths), f"missing {variant} artifact bundle"

    public_plot_rows = read_csv(f"{prefix}_test_plot_results.csv")
    diagnostic_plot_rows = read_csv(
        f"{prefix}_leaf_off_test_plot_diagnostic.csv"
    )
    for target, rows in (
        ("leaf_on", public_plot_rows),
        ("leaf_off", diagnostic_plot_rows),
    ):
        assert len(rows) == 11
        assert {row["method_slug"] for row in rows} == {"tls2trees"}
        assert {row["variant"] for row in rows} == {variant}
        assert {row["training_mode"] for row in rows} == {
            "external_training_only"
        }
        assert {row["target"] for row in rows} == {target}
        assert {row["dataset_split"] for row in rows} == {"test"}
        assert {float(row["iou_threshold"]) for row in rows} == {
            IOU_THRESHOLD
        }
        assert_tls2trees_protocol_fields(rows)
        assert all(HEX_64.fullmatch(row["source_metrics_sha256"]) for row in rows)
        assert all(
            HEX_64.fullmatch(row["aligned_prediction_sha256"]) for row in rows
        )
        assert_plot_rows([normalize_aligned_plot(row) for row in rows])

    public_inventory = {
        (row["plot_index"], row["relative_path"], row["collection"]): int(
            row["reference_instances"]
        )
        for row in public_plot_rows
    }
    diagnostic_inventory = {
        (row["plot_index"], row["relative_path"], row["collection"]): int(
            row["reference_instances"]
        )
        for row in diagnostic_plot_rows
    }
    assert public_inventory == diagnostic_inventory

    run_ids = {row["run_id"] for row in public_plot_rows + diagnostic_plot_rows}
    assert len(run_ids) == 1
    run_id = run_ids.pop()

    public_calculated = aggregate(
        [normalize_aligned_plot(row) for row in public_plot_rows]
    )
    public_overall_rows = read_csv(f"{prefix}_test_results.csv")
    assert len(public_overall_rows) == 1
    public_overall = public_overall_rows[0]
    assert public_overall["target"] == "leaf_on"
    assert public_overall["run_id"] == run_id
    assert_tls2trees_protocol_fields(public_overall_rows)
    assert_summary_matches(public_calculated, public_overall)
    assert_tls2trees_sites(public_plot_rows, f"{prefix}_test_site_results.csv")

    diagnostic_calculated = aggregate(
        [normalize_aligned_plot(row) for row in diagnostic_plot_rows]
    )
    diagnostic_overall_rows = read_csv(
        f"{prefix}_leaf_off_test_diagnostic.csv"
    )
    assert len(diagnostic_overall_rows) == 1
    diagnostic_overall = diagnostic_overall_rows[0]
    assert diagnostic_overall["target"] == "leaf_off"
    assert diagnostic_overall["run_id"] == run_id
    assert_tls2trees_protocol_fields(diagnostic_overall_rows)
    assert_summary_matches(diagnostic_calculated, diagnostic_overall)
    assert_tls2trees_sites(
        diagnostic_plot_rows,
        f"{prefix}_leaf_off_test_site_diagnostic.csv",
    )

    headline = tls2trees_output_rows(
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_benchmark_results.csv",
        variant,
    )
    assert len(headline) == 1
    assert headline[0]["run_id"] == run_id
    assert headline[0]["comparable_group"] == TLS2TREES_COMPARABLE_GROUP
    assert headline[0]["result_status"] == "completed_aligned_pointwise_test"
    assert_tls2trees_protocol_fields(headline)
    assert_summary_matches(public_calculated, headline[0])

    diagnostic = tls2trees_output_rows(
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_method_development_diagnostics.csv",
        variant,
    )
    assert len(diagnostic) == 1
    assert diagnostic[0]["run_id"] == run_id
    assert diagnostic[0]["comparable_group"] == TLS2TREES_DIAGNOSTIC_GROUP
    assert diagnostic[0]["result_status"] == "completed_aligned_pointwise_test"
    assert_tls2trees_protocol_fields(diagnostic)
    assert_summary_matches(diagnostic_calculated, diagnostic[0])

    retention_path = ROOT / f"{prefix}_prediction_retention_manifest.json"
    retention_bytes = retention_path.read_bytes()
    retention_sha256 = hashlib.sha256(retention_bytes).hexdigest()
    retention = json.loads(retention_bytes)
    files = retention["files"]
    assert retention["status"] == "retention_verified"
    assert retention["run_id"] == run_id
    assert retention["variant"] == variant
    assert retention["dataset_split"] == "test"
    assert retention["targets"] == ["leaf_off", "leaf_on"]
    assert retention["prediction_contract"] == "one_instance_label_per_source_row"
    assert retention["future_metrics_without_inference"] is True
    assert retention["configuration_changed_after_test"] is False
    assert retention["verified_prediction_files"] == len(files) == 22
    assert retention["verified_prediction_size_bytes"] == sum(
        int(row["size_bytes"]) for row in files
    )
    assert Counter(row["target"] for row in files) == {
        "leaf_off": 11,
        "leaf_on": 11,
    }
    assert len({row["relative_path"] for row in files}) == 22
    assert all(row["point_correspondence"] == "source_row_index" for row in files)
    assert all(HEX_64.fullmatch(row["sha256"]) for row in files)

    retained_hashes = {
        (row["target"], int(row["plot_index"])): row["sha256"] for row in files
    }
    assert len(retained_hashes) == 22
    for row in public_plot_rows + diagnostic_plot_rows:
        assert retained_hashes[(row["target"], int(row["plot_index"]))] == (
            row["aligned_prediction_sha256"]
        )

    retention_registry = tls2trees_output_rows(
        "outputs/for_instance_benchmark_metrics/"
        "for_instance_prediction_retention_registry.csv",
        variant,
    )
    assert len(retention_registry) == 1
    registry = retention_registry[0]
    assert registry["retention_profile"] == TLS2TREES_RETENTION_PROFILE
    assert registry["run_id"] == run_id
    assert registry["evaluation_split"] == "test"
    assert registry["future_metrics_without_inference"] == "true"
    assert registry["hash_status"] == "sha256_verified"
    assert int(registry["retained_file_count"]) == 22
    assert int(registry["retained_size_bytes"]) == retention[
        "verified_prediction_size_bytes"
    ]
    assert registry["retention_manifest"] == (
        f"{prefix}_prediction_retention_manifest.json"
    )
    assert registry["retention_manifest_sha256"] == retention_sha256
    assert registry["evidence_path"] == f"{prefix}_test_plot_results.csv"

    provenance = json.loads(
        (ROOT / f"{prefix}_test_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["run_id"] == run_id
    assert provenance["variant"] == variant
    assert provenance["canonical_target"] == "leaf_on"
    assert provenance["diagnostic_target"] == "leaf_off"
    assert provenance["evaluation_evaluator"] == TLS2TREES_EVALUATOR
    assert provenance["configuration_changed_after_test"] is False
    assert provenance["inference_rerun"] is False
    assert provenance["verified_prediction_files"] == 22
    assert provenance["retention_manifest_sha256"] == retention_sha256
    assert provenance["public_result"]["run_id"] == run_id
    assert provenance["diagnostic_result"]["run_id"] == run_id
    assert_summary_matches(public_calculated, provenance["public_result"])
    assert_summary_matches(
        diagnostic_calculated, provenance["diagnostic_result"]
    )


def test_current_tls2trees_development_tuned_bundle_reconciles() -> None:
    assert_tls2trees_bundle("development_tuned")


def test_published_default_bundle_is_absent_or_complete_and_reconciled() -> None:
    if published_default_bundle_present():
        assert_tls2trees_bundle("published_default")
