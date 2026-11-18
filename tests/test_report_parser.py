"""Tests for report text extraction and parameter parsing."""

import importlib.util
import shutil

import pytest

from src.report_parser import ReportParsingError, extract_text, extract_text_from_path, parse_parameters, parse_report
from src.schema import validate_input

EXPECTED_SAMPLE = {"N": 40.0, "P": 27.0, "K": 45.0, "temperature": 21.66, "humidity": 94.79,
                   "ph": 5.89, "rainfall": 112.43, "soil": "sandy loam"}
EXPECTED_TABLE_REPORT = {"N": 5.0, "P": 25.0, "K": 6.0, "temperature": 30.72, "humidity": 94.01,
                         "ph": 6.01, "rainfall": 106.81, "soil": "loamy"}  # docx / pdf / doc fixtures

needs_pdfplumber = pytest.mark.skipif(importlib.util.find_spec("pdfplumber") is None, reason="pdfplumber not installed")
needs_docx = pytest.mark.skipif(importlib.util.find_spec("docx") is None, reason="python-docx not installed")
needs_doc_tool = pytest.mark.skipif(not any(shutil.which(t) for t in ("antiword", "catdoc", "soffice")),
                                    reason="no .doc converter installed")


def test_txt_report(fixtures_dir):
    result = parse_parameters(extract_text_from_path(fixtures_dir / "sample_report.txt"))
    assert result.found_all
    assert result.values == EXPECTED_SAMPLE
    assert validate_input(result.values).is_valid


def test_unit_conversions(fixtures_dir):
    result = parse_parameters(extract_text_from_path(fixtures_dir / "report_us_units.txt"))
    assert result.found_all
    v = result.values
    assert v["temperature"] == pytest.approx(24.25, abs=0.01)   # 75.65 °F
    assert v["rainfall"] == pytest.approx(54.86, abs=0.01)      # 2.16 inches
    assert v["P"] == pytest.approx(7.0, abs=0.01)               # 16.04 P2O5
    assert v["K"] == pytest.approx(46.98, abs=0.01)             # 56.6 K2O
    assert v["soil"] == "loamy clay"                            # "clay loam"
    assert set(result.notes) == {"temperature", "rainfall", "P", "K"}


def test_incomplete_report_lists_missing(fixtures_dir):
    result = parse_parameters(extract_text_from_path(fixtures_dir / "report_incomplete.txt"))
    assert result.values == {"N": 102.0, "K": 47.0, "ph": 6.5}
    assert set(result.missing) == {"P", "temperature", "humidity", "rainfall", "soil"}
    assert not result.found_all


@pytest.mark.parametrize("text,feature,expected", [
    ("pH (H2O): 6.2", "ph", 6.2),                 # formula digits ignored
    ("Nitrate test pending\nNitrogen - 55", "N", 55.0),  # "Nitrate" is not "N"
    ("Phosphorus | 33 | mg/kg", "P", 33.0),       # table row
    ("Temperature: -3.5 °C", "temperature", -3.5),
    ("Rainfall 1,2 cm", "rainfall", 12.0),        # decimal comma + cm
    ("Rainfall: 1,250 mm", "rainfall", 1250.0),   # thousands separator
    ("Humidity (RH) 77%", "humidity", 77.0),
])
def test_pattern_edge_cases(text, feature, expected):
    assert parse_parameters(text).values[feature] == pytest.approx(expected)


def test_ph_not_confused_with_phosphorus():
    values = parse_parameters("pH 6.8\nP: 30").values
    assert values["ph"] == 6.8 and values["P"] == 30.0


@needs_docx
def test_docx_report(fixtures_dir):
    result, _ = parse_report((fixtures_dir / "sample_report.docx").read_bytes(), "sample_report.docx")
    assert result.values == EXPECTED_TABLE_REPORT


@needs_pdfplumber
def test_pdf_report(fixtures_dir):
    result, _ = parse_report((fixtures_dir / "sample_report.pdf").read_bytes(), "sample_report.pdf")
    assert result.values == EXPECTED_TABLE_REPORT


@needs_doc_tool
def test_doc_report(fixtures_dir):
    result, _ = parse_report((fixtures_dir / "sample_report.doc").read_bytes(), "sample_report.doc")
    assert result.values == EXPECTED_TABLE_REPORT


@pytest.mark.parametrize("data,name,message", [
    (b"abc", "report.xlsx", "Unsupported file type"),
    (b"", "report.txt", "empty"),
    (b"   \n  ", "report.txt", "No readable text"),
])
def test_extraction_errors(data, name, message):
    with pytest.raises(ReportParsingError, match=message):
        extract_text(data, name)


@needs_pdfplumber
def test_corrupt_pdf():
    with pytest.raises(ReportParsingError, match="Could not read"):
        extract_text(b"%PDF-1.4 not really a pdf", "broken.pdf")
