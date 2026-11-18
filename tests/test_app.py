"""Headless tests of the Streamlit app using Streamlit's AppTest framework.

AppTest cannot simulate a real file upload, so the upload-review path is tested by placing a real
parse result (from a fixture report) into session state, which is what the app does after an upload.
"""

import importlib.util

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import config  # noqa: E402
from src.report_parser import parse_report  # noqa: E402

APP = str(config.PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _require_model():
    if not (config.MODELS_DIR / config.MODEL_FILENAME).exists():
        pytest.skip("models/ is empty; run `python -m src.train` first")


def _run(at: AppTest) -> AppTest:
    at.run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads():
    at = _run(AppTest.from_file(APP))
    assert at.title[0].value.endswith("Crop Recommendation")
    assert len(at.slider) == 7
    assert len(at.selectbox) == 1


def test_manual_prediction_requires_soil():
    at = _run(AppTest.from_file(APP))
    at.button[0].click()
    _run(at)
    assert any("Soil texture" in e.value for e in at.error)


def test_manual_prediction_flow():
    at = _run(AppTest.from_file(APP))
    values = {"N": 40, "P": 27, "K": 45, "temperature": 21.7, "humidity": 94.8, "ph": 5.89, "rainfall": 112.4}
    for slider in at.slider:
        key = slider.key.replace("manual_", "")
        slider.set_value(float(values[key]))
    at.selectbox(key="manual_soil").set_value("sandy loam")
    at.button[0].click()
    _run(at)
    result = at.session_state["result"]
    assert [r.rank for r in result.recommendations] == [1, 2, 3]
    assert result.top.crop == "guinea_corn"
    assert any("Recommended crops" in s.value for s in at.subheader)


@pytest.mark.skipif(importlib.util.find_spec("pdfplumber") is None, reason="pdfplumber not installed")
def test_upload_review_flow(fixtures_dir):
    parsed, text = parse_report((fixtures_dir / "sample_report.pdf").read_bytes(), "sample_report.pdf")
    at = AppTest.from_file(APP)
    at.session_state["parsed"] = parsed
    at.session_state["parsed_text"] = text
    at.session_state["parsed_name"] = "sample_report.pdf"
    at.session_state["parsed_file_id"] = "test"
    _run(at)
    # Review form is pre-filled with the parsed values.
    assert at.number_input(key="review_test_N").value == 5.0
    assert at.selectbox(key="review_test_soil").value == "loamy"
    at.button[1].click()  # the review form's submit button
    _run(at)
    assert at.session_state["result"].top.crop == "orange"  # this report is an 'orange' row of the dataset


def test_upload_missing_values_are_flagged(fixtures_dir):
    parsed, text = parse_report((fixtures_dir / "report_incomplete.txt").read_bytes(), "report_incomplete.txt")
    at = AppTest.from_file(APP)
    for k, v in {"parsed": parsed, "parsed_text": text, "parsed_name": "report_incomplete.txt",
                 "parsed_file_id": "inc"}.items():
        at.session_state[k] = v
    _run(at)
    assert any("Please fill in" in w.value for w in at.warning)
    at.button[1].click()
    _run(at)
    assert any("Please fix these inputs" in e.value for e in at.error)
    assert "result" not in at.session_state
