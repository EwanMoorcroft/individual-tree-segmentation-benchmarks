from __future__ import annotations

import csv
import math
import posixpath
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = (
    ROOT
    / "outputs/for_instance_benchmark_metrics/"
    "for_instance_method_benchmark_tracker.xlsx"
)
HEADLINE_CSV = (
    ROOT
    / "outputs/for_instance_benchmark_metrics/"
    "for_instance_method_benchmark_results.csv"
)
RETENTION_CSV = (
    ROOT
    / "outputs/for_instance_benchmark_metrics/"
    "for_instance_prediction_retention_registry.csv"
)

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"x": SPREADSHEET_NS}
RELATIONSHIP_ID = f"{{{OFFICE_REL_NS}}}id"
CELL_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*)$")
TABLE_REF = re.compile(r"^([A-Z]+)([1-9][0-9]*):([A-Z]+)([1-9][0-9]*)$")
PRIVATE_PATH_PATTERNS = {
    "POSIX user path": re.compile(
        r"(?:file:(?://+)?)?/(?:Users|users|home)/[A-Za-z0-9._-]+(?:[/\\]|$)",
        flags=re.IGNORECASE,
    ),
    "POSIX private mount": re.compile(
        r"(?:file:(?://+)?)?/(?:mnt|Volumes|private|scratch|fastscratch)(?:[/\\]|$)",
        flags=re.IGNORECASE,
    ),
    "Windows absolute path": re.compile(
        r"(?<![A-Za-z0-9])(?:file:(?://+)?)?[A-Z]:[/\\]",
        flags=re.IGNORECASE,
    ),
    "UNC path": re.compile(
        r"(?<![A-Za-z0-9])\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+",
        flags=re.IGNORECASE,
    ),
}
PRIVATE_WORKBOOK_TOKENS = {
    "Barkla filesystem": re.compile(r"\bfastscratch\b", flags=re.IGNORECASE),
    "Barkla hostname": re.compile(
        r"\b(?:barkla(?:[0-9]+|viz[0-9]*)|liv\.alces\.network)\b",
        flags=re.IGNORECASE,
    ),
}
PUBLIC_BARKLA_VALUES = {
    "barkla_run_scoped_retained",
    "local_and_barkla_retained",
}

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
MATCHING_POLICY_LABELS = {
    "maximum_cardinality_one_to_one": "Maximum-cardinality one-to-one",
}
PROTOCOL_ALIGNMENT_TEXT = {
    ("treex", "unsupervised_parameterised"): {
        "Headline status": "Shared-protocol baseline",
        "Selection control": "No test-based selection",
        "Interpretation": (
            "Unsupervised method; development-parameterised and no fitted checkpoint."
        ),
    },
    ("segmentanytree", "published_pretrained"): {
        "Headline status": "Shared-protocol baseline",
        "Selection control": "No test-based selection",
        "Interpretation": "Published weights; no FOR-instance fine-tuning.",
    },
    ("segmentanytree", "fine_tuned_on_dev"): {
        "Headline status": "Shared-protocol primary",
        "Selection control": "Frozen before one test evaluation",
        "Interpretation": (
            "Fine-tuned on development only; frozen checkpoint evaluated once."
        ),
    },
    ("treelearn", "published_pretrained"): {
        "Headline status": "Shared-protocol baseline",
        "Selection control": "No test-based selection",
        "Interpretation": (
            "Clean authors-released checkpoint; documented detection "
            "scope is trees above 10 m."
        ),
    },
    ("treelearn", "fine_tuned_on_dev"): {
        "Headline status": "Shared-protocol primary",
        "Selection control": "Frozen before one test evaluation",
        "Interpretation": (
            "Fine-tuned on development only; frozen checkpoint evaluated once."
        ),
    },
    ("tls2trees", "development_tuned"): {
        "Headline status": "Primary; separate scoring domain",
        "Selection control": "Frozen on development before held-out test",
        "Interpretation": (
            "Leaf-on result excludes class-3 out-points and is not ranked with "
            "the shared mask."
        ),
    },
}

OPTIONAL_PROTOCOL_ALIGNMENT_TEXT = {
    ("tls2trees", "published_default"): {
        "Headline status": "Baseline; separate scoring domain",
        "Selection control": "No FOR-instance metric selection",
        "Interpretation": (
            "Publication-derived parameters; class-3-ignore result is reported "
            "separately."
        ),
    },
}
EXPECTED_SITES = {"CULS", "NIBIO", "RMIT", "SCION", "TUWIEN"}
AVAILABLE_DEVELOPMENT_PLOTS = 21

TABLE_HEADERS = {
    "ComparableResults": [
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
    ],
    "ComparableSiteResults": [
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
    ],
    "ProtocolAlignment": [
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
    ],
    "ComparableRetention": [
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
    ],
}


@dataclass(frozen=True)
class WorkbookTable:
    name: str
    sheet_name: str
    reference: str
    bounds: tuple[int, int, int, int]
    headers: list[str]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SiteSource:
    path: str
    where: dict[str, str]
    columns: dict[str, str]


COMMON_SITE_COLUMNS = {
    "Split": "dataset_split",
    "Site": "site",
    "Plots": "plots",
    "Reference instances": "reference_instances",
    "Predicted instances": "predicted_instances",
    "TP": "true_positives",
    "FP": "false_positives",
    "FN": "false_negatives",
    "Mean plot F1": "mean_plot_f1",
    "Micro precision": "micro_precision",
    "Micro recall": "micro_recall",
    "Micro F1": "micro_f1",
}
TREELEARN_SITE_COLUMNS = {
    **COMMON_SITE_COLUMNS,
    "Plots": "completed_plots",
}
TREEX_SITE_COLUMNS = {
    "Split": "split",
    "Site": "site",
    "Plots": "n_plots",
    "Reference instances": "total_reference_trees",
    "Predicted instances": "total_predicted_trees_harmonized",
    "TP": "total_tp_harmonized",
    "FP": "total_fp_harmonized",
    "FN": "total_fn_harmonized",
    "Mean plot F1": "mean_plot_f1_harmonized",
    "Micro precision": "micro_precision_harmonized",
    "Micro recall": "micro_recall_harmonized",
    "Micro F1": "micro_f1_harmonized",
}
SITE_SOURCES = {
    ("treex", "unsupervised_parameterised"): SiteSource(
        path="methods/treex/examples/treex_site_summary.csv",
        where={"split": "$evaluation_split"},
        columns=TREEX_SITE_COLUMNS,
    ),
    ("segmentanytree", "published_pretrained"): SiteSource(
        path=(
            "methods/segmentanytree/examples/"
            "sat_completed_target_site_results_20260711.csv"
        ),
        where={
            "variant": "$variant",
            "dataset_split": "$evaluation_split",
        },
        columns=COMMON_SITE_COLUMNS,
    ),
    ("segmentanytree", "fine_tuned_on_dev"): SiteSource(
        path=(
            "methods/segmentanytree/examples/"
            "sat_completed_target_site_results_20260711.csv"
        ),
        where={
            "variant": "$variant",
            "dataset_split": "$evaluation_split",
        },
        columns=COMMON_SITE_COLUMNS,
    ),
    ("treelearn", "published_pretrained"): SiteSource(
        path=(
            "methods/treelearn/examples/"
            "treelearn_pretrained_test_site_results_20260714.csv"
        ),
        where={
            "run_id": "$run_id",
            "variant": "$variant",
            "dataset_split": "$evaluation_split",
        },
        columns=TREELEARN_SITE_COLUMNS,
    ),
    ("treelearn", "fine_tuned_on_dev"): SiteSource(
        path=(
            "methods/treelearn/examples/"
            "treelearn_finetuned_test_site_results_20260713.csv"
        ),
        where={
            "run_id": "$run_id",
            "variant": "$variant",
            "dataset_split": "$evaluation_split",
        },
        columns=TREELEARN_SITE_COLUMNS,
    ),
    ("tls2trees", "development_tuned"): SiteSource(
        path=(
            "methods/tls2trees/examples/"
            "tls2trees_development_tuned_test_site_results.csv"
        ),
        where={
            "run_id": "$run_id",
            "variant": "$variant",
            "target": "leaf_on",
            "dataset_split": "$evaluation_split",
        },
        columns=COMMON_SITE_COLUMNS,
    ),
    ("tls2trees", "published_default"): SiteSource(
        path=(
            "methods/tls2trees/examples/"
            "tls2trees_published_default_test_site_results.csv"
        ),
        where={
            "run_id": "$run_id",
            "variant": "$variant",
            "target": "leaf_on",
            "dataset_split": "$evaluation_split",
        },
        columns=COMMON_SITE_COLUMNS,
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _headlines() -> list[dict[str, str]]:
    rows = _read_csv(HEADLINE_CSV)
    assert rows, "The canonical headline registry must not be empty"
    assert {row["dataset_slug"] for row in rows} == {"for-instance"}
    identities = [
        (row["method_slug"], row["variant"], row["run_id"]) for row in rows
    ]
    assert len(identities) == len(set(identities)), (
        "Canonical headline method/variant/run_id identities must be unique"
    )
    display_identities = [_display_key(row) for row in rows]
    assert len(display_identities) == len(set(display_identities)), (
        "Workbook sheets require one headline run per method/variant display key"
    )
    return rows


def _display_key(row: dict[str, str]) -> tuple[str, str]:
    method_slug = row["method_slug"]
    variant = row["variant"]
    assert method_slug in METHOD_LABELS, f"Missing method display label: {method_slug}"
    assert variant in VARIANT_LABELS, f"Missing variant display label: {variant}"
    return METHOD_LABELS[method_slug], VARIANT_LABELS[variant]


def _column_number(letters: str) -> int:
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value


def _column_letters(number: int) -> str:
    letters = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _parse_cell_reference(reference: str) -> tuple[int, int]:
    match = CELL_REF.fullmatch(reference)
    assert match is not None, f"Unsupported XLSX cell reference: {reference}"
    return _column_number(match.group(1)), int(match.group(2))


def _parse_table_reference(reference: str) -> tuple[int, int, int, int]:
    match = TABLE_REF.fullmatch(reference)
    assert match is not None, f"Unsupported XLSX table reference: {reference}"
    return (
        _column_number(match.group(1)),
        int(match.group(2)),
        _column_number(match.group(3)),
        int(match.group(4)),
    )


def _resolved_part_path(source_part: str, target: str) -> str:
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(source_part), target)
    ).lstrip("/")


def _relationships(
    archive: zipfile.ZipFile, relationship_path: str, source_part: str
) -> dict[str, str]:
    root = ElementTree.fromstring(archive.read(relationship_path))
    return {
        relationship.attrib["Id"]: _resolved_part_path(
            source_part, relationship.attrib["Target"]
        )
        for relationship in root.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship"
        )
    }


def _xml_payload_strings(
    archive: zipfile.ZipFile, member: str
) -> tuple[list[str], str]:
    root = ElementTree.fromstring(archive.read(member))
    values: list[str] = []
    for element in root.iter():
        values.extend(element.attrib.values())
        if element.text:
            values.append(element.text)
        if element.tail:
            values.append(element.tail)
    return values, "".join(root.itertext())


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", NS))
        for item in root.findall("x:si", NS)
    ]


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(".//x:is/x:t", NS)
        )
    value_node = cell.find("x:v", NS)
    if value_node is None:
        return None
    raw = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    if re.fullmatch(r"-?[0-9]+", raw):
        return int(raw)
    return float(raw)


def _worksheet_cells(
    archive: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]
) -> dict[tuple[int, int], Any]:
    root = ElementTree.fromstring(archive.read(sheet_path))
    cells: dict[tuple[int, int], Any] = {}
    for cell in root.findall(".//x:sheetData/x:row/x:c", NS):
        column, row = _parse_cell_reference(cell.attrib["r"])
        cells[(column, row)] = _cell_value(cell, shared_strings)
    return cells


def _workbook_tables() -> dict[str, WorkbookTable]:
    with zipfile.ZipFile(WORKBOOK) as archive:
        shared_strings = _shared_strings(archive)
        workbook_path = "xl/workbook.xml"
        workbook = ElementTree.fromstring(archive.read(workbook_path))
        workbook_relationships = _relationships(
            archive, "xl/_rels/workbook.xml.rels", workbook_path
        )
        tables: dict[str, WorkbookTable] = {}
        for sheet in workbook.findall("x:sheets/x:sheet", NS):
            sheet_name = sheet.attrib["name"]
            sheet_path = workbook_relationships[sheet.attrib[RELATIONSHIP_ID]]
            sheet_root = ElementTree.fromstring(archive.read(sheet_path))
            table_parts = sheet_root.find("x:tableParts", NS)
            if table_parts is None:
                continue
            table_part_nodes = table_parts.findall("x:tablePart", NS)
            assert int(table_parts.attrib["count"]) == len(table_part_nodes)
            sheet_relationship_path = posixpath.join(
                posixpath.dirname(sheet_path),
                "_rels",
                f"{posixpath.basename(sheet_path)}.rels",
            )
            sheet_relationships = _relationships(
                archive, sheet_relationship_path, sheet_path
            )
            cells = _worksheet_cells(archive, sheet_path, shared_strings)
            for table_part in table_part_nodes:
                table_path = sheet_relationships[
                    table_part.attrib[RELATIONSHIP_ID]
                ]
                table_root = ElementTree.fromstring(archive.read(table_path))
                name = table_root.attrib["displayName"]
                assert name not in tables, f"Duplicate XLSX table name: {name}"
                reference = table_root.attrib["ref"]
                bounds = _parse_table_reference(reference)
                start_column, header_row, end_column, end_row = bounds
                column_nodes = table_root.findall(
                    "x:tableColumns/x:tableColumn", NS
                )
                declared_count = int(
                    table_root.find("x:tableColumns", NS).attrib["count"]
                )
                assert declared_count == len(column_nodes)
                headers = [node.attrib["name"] for node in column_nodes]
                assert end_column - start_column + 1 == len(headers)
                worksheet_headers = [
                    cells.get((column, header_row))
                    for column in range(start_column, end_column + 1)
                ]
                assert worksheet_headers == headers, (
                    f"{name} table metadata and worksheet headers differ"
                )
                rows = [
                    {
                        header: cells.get((start_column + offset, row_number))
                        for offset, header in enumerate(headers)
                    }
                    for row_number in range(header_row + 1, end_row + 1)
                ]
                assert all(any(value is not None for value in row.values()) for row in rows)
                tables[name] = WorkbookTable(
                    name=name,
                    sheet_name=sheet_name,
                    reference=reference,
                    bounds=bounds,
                    headers=headers,
                    rows=rows,
                )
    return tables


def _assert_integer(actual: Any, expected: str | int, context: str) -> None:
    assert not isinstance(actual, bool), f"{context}: expected integer, got bool"
    assert isinstance(actual, (int, float)), f"{context}: not numeric: {actual!r}"
    assert float(actual).is_integer(), f"{context}: not an integer: {actual!r}"
    assert int(actual) == int(expected), f"{context}: {actual!r} != {expected!r}"


def _assert_float(actual: Any, expected: str | float, context: str) -> None:
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
        f"{context}: not numeric: {actual!r}"
    )
    assert math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    ), f"{context}: {actual!r} != {expected!r}"


def _assert_text(actual: Any, expected: str, context: str) -> None:
    assert actual == expected, f"{context}: {actual!r} != {expected!r}"


def _table_by_display_key(
    table: WorkbookTable,
) -> dict[tuple[str, str], dict[str, Any]]:
    keys = [(row["Method"], row["Variant"]) for row in table.rows]
    assert len(keys) == len(set(keys)), f"{table.name} contains duplicate headlines"
    return dict(zip(keys, table.rows, strict=True))


def _site_source_rows(headline: dict[str, str]) -> list[dict[str, str]]:
    key = (headline["method_slug"], headline["variant"])
    assert key in SITE_SOURCES, f"No canonical site source declared for {key}"
    source = SITE_SOURCES[key]
    rows = _read_csv(ROOT / source.path)
    for source_field, expected in source.where.items():
        expected_value = (
            headline[expected[1:]] if expected.startswith("$") else expected
        )
        rows = [row for row in rows if row[source_field] == expected_value]
    assert len(rows) == 5, f"{key} must have exactly five canonical site rows"
    assert {row[source.columns["Site"]] for row in rows} == EXPECTED_SITES
    return rows


def _expected_table_reference(column_count: int, data_rows: int) -> str:
    return f"A4:{_column_letters(column_count)}{4 + data_rows}"


def test_workbook_archive_is_public_safe_and_self_contained() -> None:
    with zipfile.ZipFile(WORKBOOK) as archive:
        members = archive.namelist()
        violations: set[str] = set()

        external_parts = [
            member
            for member in members
            if "/externallinks/" in f"/{member.lower()}/"
        ]
        if external_parts:
            violations.add(
                "external workbook parts: " + ", ".join(sorted(external_parts))
            )

        xml_members = [
            member
            for member in members
            if member.lower().endswith((".xml", ".rels"))
        ]
        for member in xml_members:
            values, combined_text = _xml_payload_strings(archive, member)
            inspected_values = [member, combined_text, *values]
            for label, pattern in {
                **PRIVATE_PATH_PATTERNS,
                **PRIVATE_WORKBOOK_TOKENS,
            }.items():
                for value in inspected_values:
                    match = pattern.search(value)
                    if match:
                        violations.add(
                            f"{member}: {label}: {match.group(0)!r}"
                        )

            for value in values:
                if "barkla" not in value.lower():
                    continue
                if value.strip().lower() not in PUBLIC_BARKLA_VALUES:
                    violations.add(
                        f"{member}: non-public Barkla token in {value!r}"
                    )

            root = ElementTree.fromstring(archive.read(member))
            if member.lower().endswith(".rels"):
                for relationship in root.findall(
                    f"{{{PACKAGE_REL_NS}}}Relationship"
                ):
                    relationship_type = relationship.attrib.get("Type", "")
                    target = relationship.attrib.get("Target", "")
                    target_mode = relationship.attrib.get("TargetMode", "")
                    is_external = (
                        target_mode.lower() == "external"
                        or "externallink" in relationship_type.lower()
                        or re.match(
                            r"^(?:https?|ftp|file):",
                            target,
                            flags=re.IGNORECASE,
                        )
                        is not None
                        or target.startswith("\\\\")
                    )
                    if is_external:
                        violations.add(
                            f"{member}: external relationship "
                            f"{relationship.attrib.get('Id', '<missing id>')}: "
                            f"{target!r}"
                        )

            external_reference_tags = {
                element.tag.rsplit("}", 1)[-1]
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1]
                in {"externalBook", "externalReference", "externalReferences"}
            }
            if external_reference_tags:
                violations.add(
                    f"{member}: external-link XML tags: "
                    + ", ".join(sorted(external_reference_tags))
                )

    assert not violations, "Workbook public-safety violations:\n" + "\n".join(
        sorted(violations)
    )


def test_workbook_table_ranges_expand_with_the_canonical_headlines() -> None:
    headlines = _headlines()
    tables = _workbook_tables()
    expected_counts = {
        "ComparableResults": len(headlines),
        "ComparableSiteResults": len(headlines) * 5,
        "ProtocolAlignment": len(headlines),
        "ComparableRetention": len(headlines),
    }
    expected_sheets = {
        "ComparableResults": "Comparable Results",
        "ComparableSiteResults": "Site Breakdown",
        "ProtocolAlignment": "Protocol Alignment",
        "ComparableRetention": "Prediction Retention",
    }
    assert set(TABLE_HEADERS) <= set(tables)
    for name, expected_headers in TABLE_HEADERS.items():
        table = tables[name]
        assert table.sheet_name == expected_sheets[name]
        assert table.headers == expected_headers
        assert len(table.rows) == expected_counts[name]
        assert table.reference == _expected_table_reference(
            len(expected_headers), expected_counts[name]
        )


def test_comparable_results_match_the_canonical_headline_csv() -> None:
    headlines = _headlines()
    table = _workbook_tables()["ComparableResults"]
    workbook_rows = _table_by_display_key(table)
    expected_keys = [_display_key(row) for row in headlines]
    assert list(workbook_rows) == expected_keys

    integer_fields = {
        "Plots": "plots",
        "Reference instances": "reference_instances",
        "Predicted instances": "predicted_instances",
        "TP": "true_positives",
        "FP": "false_positives",
        "FN": "false_negatives",
    }
    float_fields = {
        "Mean plot F1": "mean_plot_f1",
        "Micro precision": "micro_precision",
        "Micro recall": "micro_recall",
        "Micro F1": "micro_f1",
    }
    text_fields = {
        "Training mode": "training_mode",
        "Split": "evaluation_split",
        "Evaluation protocol": "evaluation_protocol",
        "Retention": "retention_status",
    }
    sat_published = next(
        row
        for row in headlines
        if (row["method_slug"], row["variant"])
        == ("segmentanytree", "published_pretrained")
    )
    for headline in headlines:
        key = _display_key(headline)
        row = workbook_rows[key]
        for workbook_field, csv_field in integer_fields.items():
            _assert_integer(row[workbook_field], headline[csv_field], f"{key} {workbook_field}")
        for workbook_field, csv_field in float_fields.items():
            _assert_float(row[workbook_field], headline[csv_field], f"{key} {workbook_field}")
        for workbook_field, csv_field in text_fields.items():
            _assert_text(row[workbook_field], headline[csv_field], f"{key} {workbook_field}")

        delta = row["Δ mean F1 vs SAT published"]
        shares_sat_protocol = (
            headline["comparable_group"] == sat_published["comparable_group"]
            and headline["evaluation_protocol"]
            == sat_published["evaluation_protocol"]
        )
        if shares_sat_protocol:
            _assert_float(
                delta,
                float(headline["mean_plot_f1"])
                - float(sat_published["mean_plot_f1"]),
                f"{key} Δ mean F1 vs SAT published",
            )
        else:
            assert delta is None, (
                f"{key} must not show a cross-protocol SAT delta: {delta!r}"
            )


def test_site_breakdown_matches_sources_and_reconciles_to_headlines() -> None:
    headlines = _headlines()
    tables = _workbook_tables()
    headline_rows = _table_by_display_key(tables["ComparableResults"])
    site_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tables["ComparableSiteResults"].rows:
        site_rows[(row["Method"], row["Variant"])].append(row)
    assert set(site_rows) == {_display_key(row) for row in headlines}

    integer_fields = (
        "Plots",
        "Reference instances",
        "Predicted instances",
        "TP",
        "FP",
        "FN",
    )
    float_fields = (
        "Mean plot F1",
        "Micro precision",
        "Micro recall",
        "Micro F1",
    )
    aggregate_csv_fields = {
        "Plots": "plots",
        "Reference instances": "reference_instances",
        "Predicted instances": "predicted_instances",
        "TP": "true_positives",
        "FP": "false_positives",
        "FN": "false_negatives",
    }
    for headline in headlines:
        key = _display_key(headline)
        actual_rows = site_rows[key]
        assert len(actual_rows) == 5
        assert {row["Site"] for row in actual_rows} == EXPECTED_SITES
        assert {row["Split"] for row in actual_rows} == {
            headline["evaluation_split"]
        }
        actual_by_site = {row["Site"]: row for row in actual_rows}
        assert len(actual_by_site) == 5

        source = SITE_SOURCES[(headline["method_slug"], headline["variant"])]
        for source_row in _site_source_rows(headline):
            site = source_row[source.columns["Site"]]
            actual = actual_by_site[site]
            for workbook_field in integer_fields:
                _assert_integer(
                    actual[workbook_field],
                    source_row[source.columns[workbook_field]],
                    f"{key} {site} {workbook_field}",
                )
            for workbook_field in float_fields:
                _assert_float(
                    actual[workbook_field],
                    source_row[source.columns[workbook_field]],
                    f"{key} {site} {workbook_field}",
                )

            tp = int(actual["TP"])
            fp = int(actual["FP"])
            fn = int(actual["FN"])
            _assert_float(
                actual["Micro precision"],
                tp / (tp + fp) if tp + fp else 0.0,
                f"{key} {site} micro precision reconciliation",
            )
            _assert_float(
                actual["Micro recall"],
                tp / (tp + fn) if tp + fn else 0.0,
                f"{key} {site} micro recall reconciliation",
            )
            _assert_float(
                actual["Micro F1"],
                2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                f"{key} {site} micro F1 reconciliation",
            )

        aggregate = headline_rows[key]
        for workbook_field, csv_field in aggregate_csv_fields.items():
            total = sum(int(row[workbook_field]) for row in actual_rows)
            _assert_integer(total, headline[csv_field], f"{key} summed {workbook_field}")
            _assert_integer(total, aggregate[workbook_field], f"{key} workbook {workbook_field}")
        total_plots = sum(int(row["Plots"]) for row in actual_rows)
        weighted_mean_f1 = sum(
            int(row["Plots"]) * float(row["Mean plot F1"])
            for row in actual_rows
        ) / total_plots
        _assert_float(
            weighted_mean_f1,
            headline["mean_plot_f1"],
            f"{key} weighted mean plot F1",
        )
        tp = sum(int(row["TP"]) for row in actual_rows)
        fp = sum(int(row["FP"]) for row in actual_rows)
        fn = sum(int(row["FN"]) for row in actual_rows)
        pooled = {
            "Micro precision": tp / (tp + fp) if tp + fp else 0.0,
            "Micro recall": tp / (tp + fn) if tp + fn else 0.0,
            "Micro F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
        for field, value in pooled.items():
            _assert_float(value, aggregate[field], f"{key} aggregate {field}")


def test_protocol_alignment_has_one_canonical_row_per_headline() -> None:
    headlines = _headlines()
    table = _workbook_tables()["ProtocolAlignment"]
    workbook_rows = _table_by_display_key(table)
    expected_keys = [_display_key(row) for row in headlines]
    assert list(workbook_rows) == expected_keys
    canonical_keys = {
        (row["method_slug"], row["variant"]) for row in headlines
    }
    narratives = dict(PROTOCOL_ALIGNMENT_TEXT)
    narratives.update(
        {
            key: value
            for key, value in OPTIONAL_PROTOCOL_ALIGNMENT_TEXT.items()
            if key in canonical_keys
        }
    )
    assert set(narratives) == canonical_keys, (
        "Protocol-alignment narrative expectations must cover every headline "
        "method/variant exactly"
    )
    for headline in headlines:
        key = _display_key(headline)
        row = workbook_rows[key]
        narrative = narratives[
            (headline["method_slug"], headline["variant"])
        ]
        for field, expected in narrative.items():
            _assert_text(row[field], expected, f"{key} {field}")
        _assert_integer(
            row["Available dev plots"],
            AVAILABLE_DEVELOPMENT_PLOTS,
            f"{key} available development plots",
        )
        _assert_integer(row["Test plots"], headline["plots"], f"{key} test plots")
        _assert_integer(
            row["References"],
            headline["reference_instances"],
            f"{key} references",
        )
        _assert_text(
            row["Evaluator"],
            headline["evaluation_protocol"],
            f"{key} evaluator",
        )
        matching_policy = headline["matching_policy"]
        assert matching_policy in MATCHING_POLICY_LABELS, (
            f"Missing matching-policy display label: {matching_policy}"
        )
        _assert_text(
            row["Matching policy"],
            MATCHING_POLICY_LABELS[matching_policy],
            f"{key} matching policy",
        )


def test_prediction_retention_matches_registry_by_full_run_identity() -> None:
    headlines = _headlines()
    retention_rows = _read_csv(RETENTION_CSV)
    table = _workbook_tables()["ComparableRetention"]
    workbook_rows = _table_by_display_key(table)
    expected_keys = [_display_key(row) for row in headlines]
    assert list(workbook_rows) == expected_keys

    for headline in headlines:
        identity = (
            headline["method_slug"],
            headline["variant"],
            headline["run_id"],
        )
        matches = [
            row
            for row in retention_rows
            if (row["method_slug"], row["variant"], row["run_id"])
            == identity
        ]
        assert len(matches) == 1, (
            f"Headline {identity} requires exactly one retention-registry row"
        )
        registry = matches[0]
        key = _display_key(headline)
        row = workbook_rows[key]
        _assert_text(row["Run ID"], registry["run_id"], f"{key} run ID")
        _assert_text(
            row["Split"], registry["evaluation_split"], f"{key} retention split"
        )
        _assert_text(
            row["Prediction scope"],
            registry["prediction_scope"],
            f"{key} prediction scope",
        )
        _assert_integer(
            row["Files"], registry["retained_file_count"], f"{key} retained files"
        )
        _assert_integer(
            row["Bytes"], registry["retained_size_bytes"], f"{key} retained bytes"
        )
        _assert_text(
            row["Hash status"], registry["hash_status"], f"{key} hash status"
        )
        _assert_text(
            row["Storage status"],
            registry["storage_status"],
            f"{key} storage status",
        )
        expected_reusable = registry["future_metrics_without_inference"].lower()
        assert expected_reusable in {"true", "false"}
        assert row["Reusable without inference"] is (expected_reusable == "true")
