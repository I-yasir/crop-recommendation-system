"""Soil/lab report parsing: extract text from PDF, DOCX, DOC or TXT files, then find the 8 inputs.

Pipeline::

    file bytes --extract_text()--> plain text --parse_parameters()--> ParseResult(values, missing, ...)

Text extraction
    * TXT:  decoded as UTF-8, falling back to Latin-1.
    * PDF:  ``pdfplumber`` (text-based PDFs only; scanned images need OCR, which is out of scope).
    * DOCX: ``python-docx`` (paragraphs and table cells).
    * DOC:  legacy Word, via the ``antiword`` or ``catdoc`` command-line tools, or LibreOffice as a last resort.

Parameter extraction (regex and keyword matching)
    For each input, a list of synonyms (e.g. "Nitrogen", "N", "Total N") is searched. The first number that
    follows a synonym on the same line (within a short gap that may contain units, colons, brackets or table
    separators) is taken. Units are detected and converted where needed (°F to °C; cm or inches of rain to mm).
    Every extracted value is then checked by :func:`src.schema.validate_input`, so implausible numbers are
    reported instead of silently used.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.schema import normalize_soil

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class ReportParsingError(ValueError):
    """Raised when a file cannot be read or contains no usable text."""


# --------------------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------------------
def extract_text(data: bytes, filename: str) -> str:
    """Extract plain text from a report file.

    Args:
        data: Raw file bytes (as uploaded).
        filename: Original file name; the extension selects the extractor.

    Raises:
        ReportParsingError: Unsupported type, file too large, unreadable, or no text found.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ReportParsingError(f"Unsupported file type {ext!r}. Upload a PDF, DOCX, DOC or TXT file.")
    if len(data) > MAX_FILE_BYTES:
        raise ReportParsingError("File is larger than 10 MB.")
    if not data:
        raise ReportParsingError("The file is empty.")

    extractors = {".txt": _text_from_txt, ".pdf": _text_from_pdf, ".docx": _text_from_docx, ".doc": _text_from_doc}
    try:
        text = extractors[ext](data)
    except ReportParsingError:
        raise
    except Exception as exc:  # corrupt files raise many library-specific errors
        raise ReportParsingError(f"Could not read the {ext.upper().lstrip('.')} file: {exc}") from exc

    if not text or not text.strip():
        hint = " It may be a scanned image; please enter the values manually." if ext == ".pdf" else ""
        raise ReportParsingError("No readable text found in the file." + hint)
    return text


def extract_text_from_path(path: Path | str) -> str:
    """Convenience wrapper around :func:`extract_text` for files on disk."""
    path = Path(path)
    return extract_text(path.read_bytes(), path.name)


def _text_from_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _text_from_pdf(data: bytes) -> str:
    import pdfplumber  # imported lazily so the rest of the package works without it

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            for table in page.extract_tables():
                pages.extend(" | ".join(cell or "" for cell in row) for row in table)
    return "\n".join(pages)


def _text_from_docx(data: bytes) -> str:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(lines)


def _text_from_doc(data: bytes) -> str:
    """Legacy .doc: try antiword, then catdoc, then a LibreOffice conversion."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.doc"
        src.write_bytes(data)
        for tool in ("antiword", "catdoc"):
            if shutil.which(tool):
                out = subprocess.run([tool, str(src)], capture_output=True, timeout=30)
                if out.returncode == 0 and out.stdout.strip():
                    return _text_from_txt(out.stdout)
        office = shutil.which("soffice") or shutil.which("libreoffice")
        if office:
            subprocess.run([office, "--headless", "--convert-to", "txt:Text", "--outdir", tmp, str(src)],
                           capture_output=True, timeout=60)
            txt = Path(tmp) / "report.txt"
            if txt.exists():
                return _text_from_txt(txt.read_bytes())
    raise ReportParsingError(
        "Reading legacy .doc files needs 'antiword', 'catdoc' or LibreOffice on the server. "
        "Please save the report as DOCX or PDF and upload it again."
    )


# --------------------------------------------------------------------------------------
# Parameter extraction
# --------------------------------------------------------------------------------------
# Synonyms per feature. Longer phrases come first so "Relative Humidity" wins over "RH".
SYNONYMS: dict[str, list[str]] = {
    "N": ["available nitrogen", "total nitrogen", "nitrogen", "avail. n", "n"],
    "P": ["available phosphorus", "available phosphorous", "phosphorus", "phosphorous", "phosphate oxide",
          "avail. p", "p"],
    "K": ["available potassium", "exchangeable potassium", "potassium", "potash oxide", "potash", "avail. k", "k"],
    "temperature": ["average temperature", "mean temperature", "air temperature", "temperature", "temp"],
    "humidity": ["relative humidity", "humidity", "rh"],
    "ph": ["soil ph", "ph value", "ph level", "ph"],
    "rainfall": ["annual rainfall", "average rainfall", "rainfall", "precipitation", "rain"],
}

# Thousands separators ("1,250") or a decimal point/comma ("6.5", "6,5").
_NUMBER = r"(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?)"
_UNIT = r"\s*(?P<unit>°\s?[CF]\b|deg(?:rees)?\s?[CF]\b|[CF]\b(?!\w)|%|mm\b|cm\b|in(?:ches)?\b|ppm\b|mg\s?/\s?kg\b|kg\s?/\s?ha\b)?"
# Gap between label and number: same line, no digits, at most 40 characters (units, ':', '(', '|', etc.).
_GAP = r"[^\d\n]{0,40}?"

SOIL_LABEL_RE = re.compile(r"(?i)\b(?:soil\s*(?:type|texture|class)|texture(?:\s*class)?)\b[^\w\n]{0,10}(?P<val>[A-Za-z][A-Za-z \-]{2,40})")


@dataclass
class ParseResult:
    """Outcome of :func:`parse_parameters`."""

    values: dict[str, float | str] = field(default_factory=dict)
    matches: dict[str, str] = field(default_factory=dict)  # feature -> matched text snippet
    notes: dict[str, str] = field(default_factory=dict)    # feature -> unit conversion notes
    missing: list[str] = field(default_factory=list)

    @property
    def found_all(self) -> bool:
        return not self.missing


def _label_pattern(label: str) -> re.Pattern:
    # Letters must not touch the label, so "N" never matches inside "Nitrate" and "P" never matches "pH".
    escaped = re.escape(label).replace(r"\ ", r"\s+")
    return re.compile(rf"(?i)(?<![A-Za-z]){escaped}(?![A-Za-z])" + _GAP + _NUMBER + _UNIT)


_PATTERNS: dict[str, list[re.Pattern]] = {f: [_label_pattern(s) for s in syns] for f, syns in SYNONYMS.items()}


# Oxide forms reported by many labs, converted to elemental P and K (standard factors).
OXIDE_FACTORS = {"P": 0.4364, "K": 0.8301}  # P = P2O5 x 0.4364 ; K = K2O x 0.8301


def preprocess_text(text: str) -> str:
    """Normalise text before matching.

    * ``P2O5``/``K2O`` become "phosphate oxide"/"potash oxide" (converted to P/K later).
    * Digits inside chemical formulas are removed (``H2O`` becomes ``HO``, ``CaCl2)`` becomes ``CaCl)``),
      so e.g. "pH (H2O) 6.5" yields 6.5, not 2.
    """
    text = re.sub(r"(?i)\bp\s?2\s?o\s?5\b", "phosphate oxide", text)
    text = re.sub(r"(?i)\bk\s?2\s?o\b", "potash oxide", text)
    return re.sub(r"(?<=[A-Za-z])\d+(?=[A-Za-z)\]])", "", text)


def _convert(feature: str, value: float, unit: str | None, snippet: str = "") -> tuple[float, str | None]:
    """Convert units to those used in training. Returns the value and an optional note."""
    if feature in OXIDE_FACTORS and "oxide" in snippet.lower():
        oxide = "P2O5" if feature == "P" else "K2O"
        return round(value * OXIDE_FACTORS[feature], 2), f"converted from {value:g} {oxide}"
    unit = (unit or "").lower().replace(" ", "").replace("°", "").replace("degrees", "").replace("deg", "")
    if feature == "temperature" and unit == "f":
        return round((value - 32) * 5 / 9, 2), f"converted from {value:g} °F"
    if feature == "rainfall":
        if unit == "cm":
            return value * 10, f"converted from {value:g} cm"
        if unit in ("in", "inches"):
            return round(value * 25.4, 2), f"converted from {value:g} inches"
    return value, None


def _to_number(raw: str) -> float:
    """Parse "1,250" as 1250 (thousands) but "6,5" as 6.5 (decimal comma)."""
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", raw):
        return float(raw.replace(",", ""))
    return float(raw.replace(",", "."))


def _find_feature(feature: str, text: str) -> tuple[float, str, str | None] | None:
    """Return (value, snippet, note) for the earliest-ranked synonym that matches, else None."""
    for pattern in _PATTERNS[feature]:
        match = pattern.search(text)
        if match:
            value = _to_number(match.group("num"))
            value, note = _convert(feature, value, match.group("unit"), match.group(0))
            return value, match.group(0).strip(), note
    return None


def _find_soil(text: str) -> tuple[str, str] | None:
    """Find the soil texture from a labelled field, else from any known soil phrase in the text."""
    for match in SOIL_LABEL_RE.finditer(text):
        soil = normalize_soil(match.group("val"))
        if soil:
            return soil, match.group(0).strip()
    soil = normalize_soil(text)  # longest known soil phrase anywhere in the text
    if soil:
        return soil, f"(found '{soil}' in text)"
    return None


def parse_parameters(text: str) -> ParseResult:
    """Extract the 8 model inputs from report text.

    Values are returned as found (after unit conversion). Use :func:`src.schema.validate_input` to check
    them. The web app shows them to the user for review before predicting.
    """
    result = ParseResult()
    text = preprocess_text(text)
    for feature in config.NUMERIC_FEATURES:
        found = _find_feature(feature, text)
        if found is None:
            result.missing.append(feature)
            continue
        value, snippet, note = found
        result.values[feature] = value
        result.matches[feature] = snippet
        if note:
            result.notes[feature] = note

    soil = _find_soil(text)
    if soil is None:
        result.missing.append("soil")
    else:
        result.values["soil"], result.matches["soil"] = soil
    return result


def parse_report(data: bytes, filename: str) -> tuple[ParseResult, str]:
    """Extract text from a file and parse it. Returns ``(parse_result, extracted_text)``."""
    text = extract_text(data, filename)
    return parse_parameters(text), text
