from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/reporting/build_for_instance_workbook.py"
WORKBOOK_PATH = (
    ROOT
    / "outputs/for_instance_benchmark_metrics/"
    "for_instance_method_benchmark_tracker.xlsx"
)
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
EXPECTED_SHEET_TABLES = {
    "Ranked Leaderboard": "RankedLeaderboard",
    "Differently Scoped": "DifferentlyScopedResults",
    "Comparable Results": "ComparableResults",
    "Site Breakdown": "ComparableSiteResults",
    "Protocol Alignment": "ProtocolAlignment",
    "Prediction Retention": "ComparableRetention",
    "Result Governance": "ResultGovernance",
    "Test Exposure": "TestExposureLedger",
    "Development Budget": "MethodDevelopmentBudget",
    "Environment Provenance": "EnvironmentProvenance",
    "Diagnostic Availability": "DiagnosticAvailability",
}


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "for_instance_deterministic_workbook_builder", BUILDER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _numbered_part_key(path: str) -> int:
    return int(Path(path).stem.removeprefix("table"))


def test_future_workbook_build_is_byte_deterministic_and_canonical() -> None:
    builder = _load_builder()

    first = builder.build_workbook_bytes()
    second = builder.build_workbook_bytes()

    assert first == second
    assert first == WORKBOOK_PATH.read_bytes(), (
        "Canonical workbook is stale; regenerate it from the canonical CSV tables"
    )


def test_canonical_workbook_passes_builder_check_mode() -> None:
    completed = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Workbook is current:" in completed.stdout


def test_canonical_workbook_contains_expected_eleven_sheets_and_tables() -> None:
    with zipfile.ZipFile(WORKBOOK_PATH) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [
            sheet.attrib["name"]
            for sheet in workbook.findall(
                f"{{{SPREADSHEET_NS}}}sheets/"
                f"{{{SPREADSHEET_NS}}}sheet"
            )
        ]
        table_paths = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("xl/tables/table") and name.endswith(".xml")
            ),
            key=_numbered_part_key,
        )
        table_names = [
            ElementTree.fromstring(archive.read(path)).attrib["displayName"]
            for path in table_paths
        ]

    assert len(sheet_names) == len(table_names) == 11
    assert dict(zip(sheet_names, table_names, strict=True)) == EXPECTED_SHEET_TABLES


def test_generic_csv_sheet_preserves_text_and_converts_only_declared_numbers(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    source = tmp_path / "typed_values.csv"
    source.write_text(
        "python_version,run_id,gpu_hours,configurations_attempted\n"
        "3.10,000123,12.5,4\n",
        encoding="utf-8",
    )

    spec = builder.generic_csv_sheet(
        source,
        name="Types",
        title="Types",
        note="Types",
        table_name="Types",
        numeric_columns=frozenset({"gpu_hours", "configurations_attempted"}),
    )

    assert spec.rows == [["3.10", "000123", 12.5, 4]]
    assert isinstance(spec.rows[0][0], str)
    assert isinstance(spec.rows[0][1], str)
    assert isinstance(spec.rows[0][2], float)
    assert isinstance(spec.rows[0][3], int)
