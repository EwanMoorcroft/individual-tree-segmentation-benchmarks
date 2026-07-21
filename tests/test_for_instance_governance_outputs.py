from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import pytest

from scripts.reporting import build_for_instance_governance_outputs as builder


def _identity(row: dict[str, object]) -> tuple[str, str]:
    return str(row["method_slug"]), str(row["variant"])


def test_governance_rows_cover_all_results_sites_and_bootstrap_schemes() -> None:
    outputs = builder.build_governance_rows(bootstrap_iterations=12)

    site_rows = outputs[builder.SITE_OUTPUT]
    distribution_rows = outputs[builder.DISTRIBUTION_OUTPUT]
    bootstrap_rows = outputs[builder.BOOTSTRAP_OUTPUT]
    assert len(site_rows) == 35
    assert len(distribution_rows) == 7
    assert len(bootstrap_rows) == 70
    assert [_identity(row) for row in distribution_rows] == list(
        builder.RESULT_KEYS
    )

    assert Counter(_identity(row) for row in site_rows) == {
        key: 5 for key in builder.RESULT_KEYS
    }
    for key in builder.RESULT_KEYS:
        assert [
            row["site"] for row in site_rows if _identity(row) == key
        ] == list(builder.EXPECTED_SITES)

    for key in builder.RESULT_KEYS:
        selected = [row for row in bootstrap_rows if _identity(row) == key]
        assert Counter(row["resampling_scheme"] for row in selected) == {
            "plot": 5,
            "site_stratified_plot": 5,
        }
        assert Counter(row["metric"] for row in selected) == {
            metric: 2 for metric in builder.SUPPORTED_BOOTSTRAP_METRICS
        }
        assert {row["bootstrap_iterations"] for row in selected} == {12}
        assert {row["bootstrap_seed"] for row in selected} == {20260721}
        assert {row["confidence_level"] for row in selected} == {0.95}
        assert {row["ranking_eligible"] for row in selected} == {"false"}
        assert {row["selection_eligible"] for row in selected} == {"false"}


def test_distribution_rows_preserve_exact_canonical_point_estimates() -> None:
    outputs = builder.build_governance_rows(bootstrap_iterations=2)
    headlines = builder.load_headlines()

    for row in outputs[builder.DISTRIBUTION_OUTPUT]:
        headline = headlines[_identity(row)]
        for field in (
            "plots",
            "predicted_instances",
            "reference_instances",
            "true_positives",
            "false_positives",
            "false_negatives",
            "mean_plot_f1",
            "micro_precision",
            "micro_recall",
            "micro_f1",
        ):
            assert str(row[field]) == headline[field]
        assert row["site_count"] == 5
        assert 0 <= float(row["median_plot_f1"]) <= 1
        assert float(row["plot_f1_iqr"]) == pytest.approx(
            float(row["plot_f1_q3"]) - float(row["plot_f1_q1"])
        )
        assert row["result_status"] == "diagnostic"


def test_builder_reads_only_headline_and_committed_per_plot_csvs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_read_csv = builder.read_csv
    paths: list[Path] = []

    def recording_read_csv(path: Path) -> list[dict[str, str]]:
        paths.append(path)
        return actual_read_csv(path)

    monkeypatch.setattr(builder, "read_csv", recording_read_csv)

    builder.build_governance_rows(bootstrap_iterations=2)

    expected = {builder.HEADLINE_CSV.resolve()}
    expected.update(
        (builder.ROOT / source.relative_path).resolve()
        for source in builder.PLOT_SOURCES.values()
    )
    assert {path.resolve() for path in paths} == expected
    assert all(path.suffix == ".csv" for path in paths)
    assert not any("site_results" in path.name for path in paths)


def test_tree_x_loader_is_explicitly_harmonized_union_mask() -> None:
    key = ("treex", "unsupervised_parameterised")
    headline = builder.load_headlines()[key]

    rows = builder.load_plot_rows(key, headline)

    assert builder.PLOT_SOURCES[key].kind == "treex_harmonized_union_mask"
    assert sum(row["predicted_instances"] for row in rows) == int(
        headline["predicted_instances"]
    )
    assert sum(row["true_positives"] for row in rows) == int(
        headline["true_positives"]
    )


def test_headline_verification_fails_on_any_exact_point_estimate_change() -> None:
    key = ("segmentanytree", "fine_tuned_on_dev")
    headline = builder.load_headlines()[key]
    rows = builder.load_plot_rows(key, headline)
    summary = builder.summarise_plot_distribution(rows)
    builder.verify_headline_aggregate(headline, rows, summary)

    changed_count = {**headline, "true_positives": "238"}
    with pytest.raises(ValueError, match="true_positives disagrees"):
        builder.verify_headline_aggregate(changed_count, rows, summary)

    changed_mean = {
        **headline,
        "mean_plot_f1": str(float(headline["mean_plot_f1"]) + 1e-12),
    }
    with pytest.raises(ValueError, match="mean_plot_f1 disagrees exactly"):
        builder.verify_headline_aggregate(changed_mean, rows, summary)


def test_payloads_are_byte_deterministic_and_have_fixed_row_counts() -> None:
    first = builder.build_output_payloads(bootstrap_iterations=9)
    second = builder.build_output_payloads(bootstrap_iterations=9)

    assert first == second
    expected_counts = {
        builder.SITE_OUTPUT: 35,
        builder.DISTRIBUTION_OUTPUT: 7,
        builder.BOOTSTRAP_OUTPUT: 70,
    }
    for filename, expected_count in expected_counts.items():
        text = first[filename].decode("utf-8")
        assert "\r\n" not in text
        assert len(list(csv.DictReader(io.StringIO(text)))) == expected_count


def test_cli_write_and_check_detects_stale_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path),
        "--bootstrap-iterations",
        "7",
    ]
    assert builder.main(arguments) == 0
    assert "Wrote 3 governance CSVs" in capsys.readouterr().out
    assert builder.main([*arguments, "--check"]) == 0
    assert "outputs are current" in capsys.readouterr().out

    stale = tmp_path / builder.DISTRIBUTION_OUTPUT
    stale.write_bytes(stale.read_bytes() + b"stale\n")
    with pytest.raises(SystemExit, match=builder.DISTRIBUTION_OUTPUT):
        builder.main([*arguments, "--check"])


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_builder_rejects_invalid_iteration_counts(value: object) -> None:
    exception = TypeError if isinstance(value, (bool, float)) else ValueError
    with pytest.raises(exception, match="bootstrap_iterations"):
        builder.build_governance_rows(bootstrap_iterations=value)  # type: ignore[arg-type]


def test_committed_governance_outputs_are_current() -> None:
    payloads = builder.build_output_payloads()

    for filename, payload in payloads.items():
        assert (builder.OUTPUT_ROOT / filename).read_bytes() == payload
