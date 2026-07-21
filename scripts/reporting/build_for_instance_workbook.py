#!/usr/bin/env python3
"""Build the public FOR-instance review workbook from canonical CSV tables.

The workbook is a deterministic review artefact. CSV files remain the source
of truth, and this script neither reads predictions nor recalculates accepted
benchmark scores.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "for_instance_benchmark_metrics"
HEADLINE_CSV = OUTPUT_ROOT / "for_instance_method_benchmark_results.csv"
RETENTION_CSV = OUTPUT_ROOT / "for_instance_prediction_retention_registry.csv"
GOVERNANCE_CSV = OUTPUT_ROOT / "benchmark_result_registry.csv"
DEFAULT_OUTPUT = OUTPUT_ROOT / "for_instance_method_benchmark_tracker.xlsx"

METHOD_LABELS = {
    "segmentanytree": "SegmentAnyTree",
    "tls2trees": "TLS2trees",
    "treelearn": "TreeLearn",
    "treex": "TreeX",
}
VARIANT_LABELS = {
    "development_tuned": "Development tuned",
    "fine_tuned_on_dev": "Fine-tuned on development data",
    "published_default": "Published default",
    "published_pretrained": "Published pretrained",
    "unsupervised_parameterised": "Unsupervised parameterised",
}
MATCHING_LABELS = {
    "maximum_cardinality_one_to_one": "Maximum-cardinality one-to-one"
}
EXPECTED_SITES = ("CULS", "NIBIO", "RMIT", "SCION", "TUWIEN")
AVAILABLE_DEVELOPMENT_PLOTS = 21

RESULT_HEADERS = [
    "Method",
    "Variant",
    "Training mode",
    "Split",
    "Plots",
    "Reference instances",
    "Predicted instances",
    "TP",
    "FP",
    "FN",
    "Mean plot F1",
    "Micro precision",
    "Micro recall",
    "Micro F1",
    "Δ mean F1 vs SAT published",
    "Evaluation protocol",
    "Retention",
]
SITE_HEADERS = [
    "Method",
    "Variant",
    "Split",
    "Site",
    "Plots",
    "Reference instances",
    "Predicted instances",
    "TP",
    "FP",
    "FN",
    "Mean plot F1",
    "Micro precision",
    "Micro recall",
    "Micro F1",
]
PROTOCOL_HEADERS = [
    "Method",
    "Variant",
    "Headline status",
    "Available dev plots",
    "Test plots",
    "References",
    "Evaluator",
    "Matching policy",
    "Selection control",
    "Interpretation",
]
RETENTION_HEADERS = [
    "Method",
    "Variant",
    "Run ID",
    "Split",
    "Prediction scope",
    "Files",
    "Bytes",
    "Hash status",
    "Storage status",
    "Reusable without inference",
]

PROTOCOL_TEXT = {
    ("treex", "unsupervised_parameterised"): (
        "Shared-protocol baseline",
        "No test-based selection",
        "Unsupervised method; development-parameterised and no fitted checkpoint.",
    ),
    ("segmentanytree", "published_pretrained"): (
        "Shared-protocol baseline",
        "No test-based selection",
        "Published weights; no FOR-instance fine-tuning.",
    ),
    ("segmentanytree", "fine_tuned_on_dev"): (
        "Shared-protocol primary",
        "Frozen before one test evaluation",
        "Fine-tuned on development only; frozen checkpoint evaluated once.",
    ),
    ("treelearn", "published_pretrained"): (
        "Shared-protocol baseline",
        "No test-based selection",
        "Clean authors-released checkpoint; documented detection scope is trees above 10 m.",
    ),
    ("treelearn", "fine_tuned_on_dev"): (
        "Shared-protocol primary",
        "Frozen before one test evaluation",
        "Fine-tuned on development only; frozen checkpoint evaluated once.",
    ),
    ("tls2trees", "development_tuned"): (
        "Primary; separate scoring domain",
        "Frozen on development before held-out test",
        "Leaf-on result excludes class-3 out-points and is not ranked with the shared mask.",
    ),
    ("tls2trees", "published_default"): (
        "Baseline; separate scoring domain",
        "No FOR-instance metric selection",
        "Publication-derived parameters; class-3-ignore result is reported separately.",
    ),
}


@dataclass(frozen=True)
class Cell:
    value: Any
    formula: str | None = None


@dataclass(frozen=True)
class SheetSpec:
    name: str
    title: str
    note: str
    table_name: str
    headers: list[str]
    rows: list[list[Any]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display_key(row: Mapping[str, str]) -> tuple[str, str]:
    return METHOD_LABELS[row["method_slug"]], VARIANT_LABELS[row["variant"]]


def as_number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def governance_by_result() -> dict[tuple[str, str, str, str], dict[str, str]]:
    rows = read_csv(GOVERNANCE_CSV)
    selected: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["dataset_slug"],
            row["method_slug"],
            row["variant"],
            row["run_id"],
        )
        if key in selected:
            current = selected[key]
            if row["ranking_eligible"] == "true" and current["ranking_eligible"] != "true":
                selected[key] = row
            elif current["ranking_eligible"] == "true" and row["ranking_eligible"] != "true":
                continue
            else:
                raise ValueError(f"Ambiguous governance result identity: {key}")
            continue
        selected[key] = row
    return selected


def headline_rows() -> list[dict[str, str]]:
    rows = read_csv(HEADLINE_CSV)
    if not rows:
        raise ValueError("Canonical held-out result table is empty")
    identities = [(r["method_slug"], r["variant"], r["run_id"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Canonical held-out result identities are not unique")
    return rows


def result_table_rows(
    headlines: list[dict[str, str]],
    *,
    sat_baseline: Mapping[str, str] | None = None,
    row_numbers: bool = False,
) -> list[list[Any]]:
    sat = sat_baseline
    if sat is None:
        sat = next(
            row
            for row in headlines
            if (row["method_slug"], row["variant"])
            == ("segmentanytree", "published_pretrained")
        )
    sat_row_number: int | None = None
    if row_numbers:
        sat_identity = (sat["method_slug"], sat["variant"], sat["run_id"])
        sat_row_number = next(
            index + 5
            for index, row in enumerate(headlines)
            if (row["method_slug"], row["variant"], row["run_id"])
            == sat_identity
        )
    output: list[list[Any]] = []
    for offset, row in enumerate(headlines, start=5):
        shares_protocol = (
            row["comparable_group"] == sat["comparable_group"]
            and row["evaluation_protocol"] == sat["evaluation_protocol"]
        )
        delta = (
            float(row["mean_plot_f1"]) - float(sat["mean_plot_f1"])
            if shares_protocol
            else None
        )
        delta_cell: Any = delta
        if row_numbers and delta is not None:
            delta_cell = Cell(delta, f"K{offset}-$K${sat_row_number}")
        output.append(
            [
                METHOD_LABELS[row["method_slug"]],
                VARIANT_LABELS[row["variant"]],
                row["training_mode"],
                row["evaluation_split"],
                int(row["plots"]),
                int(row["reference_instances"]),
                int(row["predicted_instances"]),
                int(row["true_positives"]),
                int(row["false_positives"]),
                int(row["false_negatives"]),
                float(row["mean_plot_f1"]),
                float(row["micro_precision"]),
                float(row["micro_recall"]),
                float(row["micro_f1"]),
                delta_cell,
                row["evaluation_protocol"],
                row["retention_status"],
            ]
        )
    return output


def site_source(headline: Mapping[str, str]) -> tuple[Path, dict[str, str], dict[str, str]]:
    common = {
        "split": "dataset_split",
        "site": "site",
        "plots": "plots",
        "references": "reference_instances",
        "predictions": "predicted_instances",
        "tp": "true_positives",
        "fp": "false_positives",
        "fn": "false_negatives",
        "mean_f1": "mean_plot_f1",
        "precision": "micro_precision",
        "recall": "micro_recall",
        "f1": "micro_f1",
    }
    key = (headline["method_slug"], headline["variant"])
    if key == ("treex", "unsupervised_parameterised"):
        return (
            ROOT / "methods/treex/examples/treex_site_summary.csv",
            {"split": headline["evaluation_split"]},
            {
                "split": "split",
                "site": "site",
                "plots": "n_plots",
                "references": "total_reference_trees",
                "predictions": "total_predicted_trees_harmonized",
                "tp": "total_tp_harmonized",
                "fp": "total_fp_harmonized",
                "fn": "total_fn_harmonized",
                "mean_f1": "mean_plot_f1_harmonized",
                "precision": "micro_precision_harmonized",
                "recall": "micro_recall_harmonized",
                "f1": "micro_f1_harmonized",
            },
        )
    if key[0] == "segmentanytree":
        path = ROOT / "methods/segmentanytree/examples/sat_completed_target_site_results_20260711.csv"
        return path, {"variant": key[1], "dataset_split": headline["evaluation_split"]}, common
    if key[0] == "treelearn":
        filename = (
            "treelearn_pretrained_test_site_results_20260714.csv"
            if key[1] == "published_pretrained"
            else "treelearn_finetuned_test_site_results_20260713.csv"
        )
        columns = dict(common)
        columns["plots"] = "completed_plots"
        return (
            ROOT / "methods/treelearn/examples" / filename,
            {
                "variant": key[1],
                "run_id": headline["run_id"],
                "dataset_split": headline["evaluation_split"],
            },
            columns,
        )
    if key[0] == "tls2trees":
        filename = f"tls2trees_{key[1]}_test_site_results.csv"
        return (
            ROOT / "methods/tls2trees/examples" / filename,
            {
                "variant": key[1],
                "run_id": headline["run_id"],
                "target": "leaf_on",
                "dataset_split": headline["evaluation_split"],
            },
            common,
        )
    raise ValueError(f"No site source for {key}")


def site_rows(headlines: list[dict[str, str]]) -> list[list[Any]]:
    output: list[list[Any]] = []
    for headline in headlines:
        path, where, columns = site_source(headline)
        rows = [
            row
            for row in read_csv(path)
            if all(row[field] == value for field, value in where.items())
        ]
        by_site = {row[columns["site"]]: row for row in rows}
        if set(by_site) != set(EXPECTED_SITES):
            raise ValueError(f"Site source for {display_key(headline)} is incomplete")
        for site in EXPECTED_SITES:
            row = by_site[site]
            output.append(
                [
                    METHOD_LABELS[headline["method_slug"]],
                    VARIANT_LABELS[headline["variant"]],
                    row[columns["split"]],
                    site,
                    int(row[columns["plots"]]),
                    int(row[columns["references"]]),
                    int(row[columns["predictions"]]),
                    int(row[columns["tp"]]),
                    int(row[columns["fp"]]),
                    int(row[columns["fn"]]),
                    float(row[columns["mean_f1"]]),
                    float(row[columns["precision"]]),
                    float(row[columns["recall"]]),
                    float(row[columns["f1"]]),
                ]
            )
    return output


def protocol_rows(headlines: list[dict[str, str]]) -> list[list[Any]]:
    rows = []
    for headline in headlines:
        status, control, interpretation = PROTOCOL_TEXT[
            (headline["method_slug"], headline["variant"])
        ]
        rows.append(
            [
                METHOD_LABELS[headline["method_slug"]],
                VARIANT_LABELS[headline["variant"]],
                status,
                AVAILABLE_DEVELOPMENT_PLOTS,
                int(headline["plots"]),
                int(headline["reference_instances"]),
                headline["evaluation_protocol"],
                MATCHING_LABELS[headline["matching_policy"]],
                control,
                interpretation,
            ]
        )
    return rows


def retention_rows(headlines: list[dict[str, str]]) -> list[list[Any]]:
    registry = read_csv(RETENTION_CSV)
    output = []
    for headline in headlines:
        matches = [
            row
            for row in registry
            if (row["method_slug"], row["variant"], row["run_id"])
            == (headline["method_slug"], headline["variant"], headline["run_id"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one retention row for {display_key(headline)}")
        row = matches[0]
        reusable = row["future_metrics_without_inference"].lower()
        if reusable not in {"true", "false"}:
            raise ValueError(
                f"Invalid retention reuse state for {display_key(headline)}"
            )
        output.append(
            [
                METHOD_LABELS[headline["method_slug"]],
                VARIANT_LABELS[headline["variant"]],
                row["run_id"],
                row["evaluation_split"],
                row["prediction_scope"],
                int(row["retained_file_count"]),
                int(row["retained_size_bytes"]),
                row["hash_status"],
                row["storage_status"],
                reusable == "true",
            ]
        )
    return output


def generic_csv_sheet(
    path: Path,
    *,
    name: str,
    title: str,
    note: str,
    table_name: str,
    numeric_columns: frozenset[str] = frozenset(),
) -> SheetSpec:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Canonical table is empty: {path}")
    headers = rows[0]
    unknown_numeric_columns = numeric_columns - set(headers)
    if unknown_numeric_columns:
        raise ValueError(
            f"Unknown numeric columns for {path}: "
            + ", ".join(sorted(unknown_numeric_columns))
        )
    values: list[list[Any]] = []
    for row in rows[1:]:
        if len(row) != len(headers):
            raise ValueError(f"Canonical table has a ragged row: {path}")
        converted = []
        for header, value in zip(headers, row, strict=True):
            if value == "":
                converted.append(None)
            elif header in numeric_columns and re.fullmatch(
                r"-?(?:0|[1-9][0-9]*)", value
            ):
                converted.append(int(value))
            elif header in numeric_columns and re.fullmatch(
                r"-?(?:0|[1-9][0-9]*)\.[0-9]+", value
            ):
                converted.append(float(value))
            else:
                converted.append(value)
        values.append(converted)
    return SheetSpec(name, title, note, table_name, headers, values)


def build_sheet_specs() -> list[SheetSpec]:
    headlines = headline_rows()
    sat_baseline = next(
        row
        for row in headlines
        if (row["method_slug"], row["variant"])
        == ("segmentanytree", "published_pretrained")
    )
    governance = governance_by_result()
    eligible = []
    separate = []
    for row in headlines:
        key = (
            row["dataset_slug"],
            row["method_slug"],
            row["variant"],
            row["run_id"],
        )
        if key not in governance:
            raise ValueError(f"Headline result is missing governance metadata: {key}")
        (eligible if governance[key]["ranking_eligible"] == "true" else separate).append(row)
    if len(eligible) != 5 or len(separate) != 2:
        raise ValueError("Expected five ranked shared-protocol rows and two separate rows")

    specs = [
        SheetSpec(
            "Ranked Leaderboard",
            "FOR-instance Shared-Protocol Leaderboard",
            "Only ranking-eligible rows using the classes 4/5/6 shared pointwise protocol are shown.",
            "RankedLeaderboard",
            RESULT_HEADERS,
            result_table_rows(eligible),
        ),
        SheetSpec(
            "Differently Scoped",
            "Differently Scoped Held-out Results",
            "These accepted TLS2trees results use the class-3-ignore scoring mask and are not ranked with the shared protocol.",
            "DifferentlyScopedResults",
            RESULT_HEADERS,
            result_table_rows(separate, sat_baseline=sat_baseline),
        ),
        SheetSpec(
            "Comparable Results",
            "FOR-instance All Held-out Results",
            "All seven accepted held-out rows are retained; use Ranked Leaderboard for the five shared-mask comparisons.",
            "ComparableResults",
            RESULT_HEADERS,
            result_table_rows(headlines, row_numbers=True),
        ),
        SheetSpec(
            "Site Breakdown",
            "FOR-instance Site Breakdown",
            "Site detail for all accepted held-out results; method, variant, mask and protocol context remain in the source tables.",
            "ComparableSiteResults",
            SITE_HEADERS,
            site_rows(headlines),
        ),
        SheetSpec(
            "Protocol Alignment",
            "Protocol Alignment",
            "Ranking and scoring-domain differences are explicit; cross-mask comparisons are not presented as a leaderboard.",
            "ProtocolAlignment",
            PROTOCOL_HEADERS,
            protocol_rows(headlines),
        ),
        SheetSpec(
            "Prediction Retention",
            "Prediction Retention",
            "Primary artefacts remain off Git; every accepted held-out row names a retained prediction set.",
            "ComparableRetention",
            RETENTION_HEADERS,
            retention_rows(headlines),
        ),
    ]
    generic = [
        (
            "benchmark_result_registry.csv",
            "Result Governance",
            "Controlled Result Governance",
            "Controlled taxonomy, eligibility, exposure and material fields. "
            "Legacy IDs remain unchanged.",
            "ResultGovernance",
            frozenset(),
        ),
        (
            "test_exposure_ledger.csv",
            "Test Exposure",
            "Held-out Test Exposure Ledger",
            "Execution, metric viewing, qualitative inspection and later "
            "changes are recorded separately; unknown means not evidenced.",
            "TestExposureLedger",
            frozenset(),
        ),
        (
            "method_development_budget.csv",
            "Development Budget",
            "Method Development Budget",
            "Observed or evidenced effort only; unknown values are not inferred from requested wall limits.",
            "MethodDevelopmentBudget",
            frozenset(
                {
                    "configurations_attempted",
                    "validation_evaluations",
                    "checkpoints_evaluated",
                    "training_epochs",
                    "optimizer_steps",
                    "gpu_hours",
                    "cpu_hours",
                }
            ),
        ),
        (
            "method_environment_provenance.csv",
            "Environment Provenance",
            "Environment and Upstream Provenance",
            "Each implemented method remains environment-specific; unknown "
            "values are explicit.",
            "EnvironmentProvenance",
            frozenset(),
        ),
        (
            "diagnostic_metric_availability.csv",
            "Diagnostic Availability",
            "Diagnostic Metric Availability",
            "Additional thresholds and error decomposition require retained "
            "artefacts and are never used for selection.",
            "DiagnosticAvailability",
            frozenset(),
        ),
    ]
    for filename, name, title, note, table_name, numeric_columns in generic:
        specs.append(
            generic_csv_sheet(
                OUTPUT_ROOT / filename,
                name=name,
                title=title,
                note=note,
                table_name=table_name,
                numeric_columns=numeric_columns,
            )
        )
    return specs


def column_letters(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def xml_text(value: Any) -> str:
    return escape(str(value), {'"': "&quot;", "'": "&apos;"})


def cell_xml(reference: str, raw: Any, style: int) -> str:
    cell = raw if isinstance(raw, Cell) else Cell(raw)
    value = cell.value
    if value is None:
        return ""
    if cell.formula is not None:
        return (
            f'<c r="{reference}" s="{style}"><f>{xml_text(cell.formula)}</f>'
            f"<v>{value!r}</v></c>"
        )
    if isinstance(value, bool):
        return f'<c r="{reference}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Workbook cannot contain non-finite value at {reference}")
        return f'<c r="{reference}" s="{style}"><v>{value!r}</v></c>'
    text = xml_text(value)
    preserve = ' xml:space="preserve"' if str(value) != str(value).strip() else ""
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr"><is><t{preserve}>'
        f"{text}</t></is></c>"
    )


def style_for(value: Any, header: bool = False) -> int:
    if header:
        return 3
    raw = value.value if isinstance(value, Cell) else value
    if isinstance(raw, bool):
        return 8
    if isinstance(raw, int):
        return 5
    if isinstance(raw, float):
        return 6
    return 4


def column_widths(spec: SheetSpec) -> list[float]:
    widths = []
    for index, header in enumerate(spec.headers):
        lengths = [len(header)]
        for row in spec.rows:
            raw = row[index].value if isinstance(row[index], Cell) else row[index]
            if raw is not None:
                lengths.append(len(str(raw)))
        width = min(max(max(lengths) + 2, 10), 55)
        if any(token in header.lower() for token in ("note", "reason", "source", "path", "interpretation")):
            width = min(max(width, 24), 55)
        widths.append(float(width))
    return widths


def data_row_height(row: list[Any], widths: list[float]) -> int:
    """Choose enough height for wrapped identifiers, notes, and evidence paths."""

    wrapped_lines = 1
    for value, width in zip(row, widths, strict=True):
        raw = value.value if isinstance(value, Cell) else value
        if raw is None or isinstance(raw, (bool, int, float)):
            continue
        characters_per_line = max(int(width) - 2, 8)
        lines = sum(
            max(1, math.ceil(len(part) / characters_per_line))
            for part in str(raw).splitlines() or [""]
        )
        wrapped_lines = max(wrapped_lines, lines)
    return min(max(21, wrapped_lines * 15 + 4), 64)


def worksheet_xml(spec: SheetSpec, table_index: int) -> str:
    column_count = len(spec.headers)
    end_row = 4 + len(spec.rows)
    end_column = column_letters(column_count)
    width_values = column_widths(spec)
    widths = "".join(
        f'<col min="{index}" max="{index}" width="{width:.2f}" customWidth="1"/>'
        for index, width in enumerate(width_values, start=1)
    )
    title_cells = cell_xml("A1", spec.title, 1)
    note_cells = cell_xml("A2", spec.note, 2)
    header_cells = "".join(
        cell_xml(f"{column_letters(index)}4", header, 3)
        for index, header in enumerate(spec.headers, start=1)
    )
    data_rows = []
    for row_number, row in enumerate(spec.rows, start=5):
        cells = "".join(
            cell_xml(
                f"{column_letters(index)}{row_number}",
                value,
                style_for(value),
            )
            for index, value in enumerate(row, start=1)
        )
        height = data_row_height(row, width_values)
        data_rows.append(
            f'<row r="{row_number}" ht="{height}" customHeight="1">{cells}</row>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="A1:{end_column}{end_row}"/>'
        '<sheetViews><sheetView showGridLines="0" workbookViewId="0">'
        '<pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{widths}</cols>"
        '<sheetData>'
        f'<row r="1" ht="28" customHeight="1">{title_cells}</row>'
        f'<row r="2" ht="34" customHeight="1">{note_cells}</row>'
        '<row r="3" ht="8" customHeight="1"/>'
        f'<row r="4" ht="36" customHeight="1">{header_cells}</row>'
        + "".join(data_rows)
        + '</sheetData>'
        f'<mergeCells count="2"><mergeCell ref="A1:{end_column}1"/>'
        f'<mergeCell ref="A2:{end_column}2"/></mergeCells>'
        f'<autoFilter ref="A4:{end_column}{end_row}"/>'
        '<pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
        f'<tableParts count="1"><tablePart r:id="rId1"/></tableParts>'
        '</worksheet>'
    )


def worksheet_rels_xml(table_index: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/table" '
        f'Target="../tables/table{table_index}.xml"/>'
        '</Relationships>'
    )


def table_xml(spec: SheetSpec, table_index: int) -> str:
    end_column = column_letters(len(spec.headers))
    end_row = 4 + len(spec.rows)
    columns = "".join(
        f'<tableColumn id="{index}" name="{xml_text(header)}"/>'
        for index, header in enumerate(spec.headers, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<table xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'id="{table_index}" name="{spec.table_name}" displayName="{spec.table_name}" '
        f'ref="A4:{end_column}{end_row}" totalsRowShown="0">'
        f'<autoFilter ref="A4:{end_column}{end_row}"/>'
        f'<tableColumns count="{len(spec.headers)}">{columns}</tableColumns>'
        '<tableStyleInfo name="TableStyleMedium2" showFirstColumn="0" '
        'showLastColumn="0" showRowStripes="1" showColumnStripes="0"/>'
        '</table>'
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="4">'
        '<font><sz val="10"/><name val="Aptos"/><family val="2"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="16"/><name val="Aptos Display"/></font>'
        '<font><color rgb="FF365F7D"/><sz val="10"/><name val="Aptos"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>'
        '</fonts>'
        '<fills count="4"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF17375E"/><bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF23869E"/><bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="2"><border/><border><bottom style="thin"><color rgb="FFB7DDE8"/></bottom></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="9">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="3" fillId="3" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="3" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="2" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>'
        '<xf numFmtId="10" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '<dxfs count="0"/><tableStyles count="1" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>'
        '</styleSheet>'
    )


def workbook_xml(specs: list[SheetSpec]) -> str:
    sheets = "".join(
        f'<sheet name="{xml_text(spec.name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, spec in enumerate(specs, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets}</sheets><calcPr calcId="191029" calcMode="auto" fullCalcOnLoad="1"/>'
        '</workbook>'
    )


def workbook_rels_xml(specs: list[SheetSpec]) -> str:
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(specs) + 1)
    )
    relationships += (
        '<Relationship '
        f'Id="rId{len(specs) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{relationships}</Relationships>'
    )


def content_types_xml(specs: list[SheetSpec]) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(specs) + 1)
    )
    overrides.extend(
        f'<Override PartName="/xl/tables/table{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"/>'
        for index in range(1, len(specs) + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + '</Types>'
    )


def package_parts(specs: list[SheetSpec]) -> dict[str, str]:
    parts = {
        "[Content_Types].xml": content_types_xml(specs),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>'
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:creator>Ewan Moorcroft</dc:creator><cp:lastModifiedBy>Ewan Moorcroft</cp:lastModifiedBy>'
            '<dc:title>FOR-instance benchmark governance tracker</dc:title>'
            '<dcterms:created xsi:type="dcterms:W3CDTF">2026-07-21T00:00:00Z</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-07-21T00:00:00Z</dcterms:modified>'
            '</cp:coreProperties>'
        ),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>tree-seg-benchmark deterministic builder</Application>'
            f'<TitlesOfParts><vt:vector size="{len(specs)}" baseType="lpstr">'
            + "".join(f'<vt:lpstr>{xml_text(spec.name)}</vt:lpstr>' for spec in specs)
            + '</vt:vector></TitlesOfParts></Properties>'
        ),
        "xl/workbook.xml": workbook_xml(specs),
        "xl/_rels/workbook.xml.rels": workbook_rels_xml(specs),
        "xl/styles.xml": styles_xml(),
    }
    for index, spec in enumerate(specs, start=1):
        parts[f"xl/worksheets/sheet{index}.xml"] = worksheet_xml(spec, index)
        parts[f"xl/worksheets/_rels/sheet{index}.xml.rels"] = worksheet_rels_xml(index)
        parts[f"xl/tables/table{index}.xml"] = table_xml(spec, index)
    return parts


def build_workbook_bytes() -> bytes:
    buffer = io.BytesIO()
    # Store parts without DEFLATE so byte identity does not depend on the zlib
    # version bundled with macOS, Linux, or the method-specific HPC runtime.
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(package_parts(build_sheet_specs()).items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.encode("utf-8"))
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if --output is not byte-identical to a fresh deterministic build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    payload = build_workbook_bytes()
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit(f"Workbook is stale; rebuild it with: {Path(__file__).name}")
        print(f"Workbook is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
