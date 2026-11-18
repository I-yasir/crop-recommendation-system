"""Streamlit web app: data-driven crop recommendation.

Run from the project root::

    streamlit run app/streamlit_app.py

The app holds UI code only. Validation, parsing, prediction and tips all come from the ``src``
package (`docs/project_overview.md` §4).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file from app/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.inference import CropPredictor, InvalidInputError, PredictionResult  # noqa: E402
from src.recommendations import build_tips  # noqa: E402
from src.report_parser import ReportParsingError, parse_report  # noqa: E402
from src.schema import validate_input  # noqa: E402

st.set_page_config(page_title="Crop Recommender", page_icon="🌱", layout="wide")

# Slider bounds: wide enough for real-world values, inside the hard limits enforced by src.schema.
SLIDER_SPEC: dict[str, tuple[float, float, float, float]] = {
    # feature: (min, max, default, step)
    "N": (0.0, 200.0, 50.0, 1.0),
    "P": (0.0, 150.0, 40.0, 1.0),
    "K": (0.0, 200.0, 35.0, 1.0),
    "temperature": (0.0, 50.0, 24.0, 0.1),
    "humidity": (0.0, 100.0, 70.0, 0.1),
    "ph": (3.0, 10.0, 6.5, 0.01),
    "rainfall": (0.0, 500.0, 90.0, 0.5),
}
HELP = {
    "N": "Soil nitrogen, as reported by your lab.",
    "P": "Soil phosphorus (elemental P).",
    "K": "Soil potassium (elemental K).",
    "temperature": "Average air temperature for the growing season, °C.",
    "humidity": "Average relative humidity, %.",
    "ph": "Soil pH (0 to 14; 7 is neutral).",
    "rainfall": "Rainfall for the growing season, mm.",
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.crop-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 1rem 1.1rem; height: 100%;}
.crop-rank {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}
.crop-name {font-size: 1.5rem; font-weight: 700; margin: .1rem 0 .4rem 0;}
.crop-conf {font-size: .95rem; margin-bottom: .3rem;}
.tip-src {font-size: .72rem; padding: 1px 7px; border-radius: 8px; margin-right: 6px;
          border: 1px solid rgba(128,128,128,.4); opacity: .85;}
</style>
"""


# --------------------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> CropPredictor:
    """Load the model once per server process."""
    return CropPredictor(config.MODELS_DIR)


def get_predictor() -> CropPredictor | None:
    try:
        return load_predictor()
    except FileNotFoundError:
        st.error("No trained model found. Run `python -m src.train` from the project root, then reload this page.")
        return None


# --------------------------------------------------------------------------------------
# Input widgets
# --------------------------------------------------------------------------------------
def input_widgets(prefix: str, defaults: dict | None = None, missing: list[str] | None = None,
                  use_sliders: bool = True) -> dict:
    """Render the 8 input controls and return their values. ``prefix`` keeps widget keys unique."""
    defaults = defaults or {}
    missing = missing or []
    values: dict = {}
    soil_options = config.SOIL_CATEGORIES

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Soil nutrients and pH**")
        for f in ["N", "P", "K", "ph"]:
            values[f] = _numeric_widget(prefix, f, defaults.get(f), f in missing, use_sliders)
    with col2:
        st.markdown("**Climate and soil texture**")
        for f in ["temperature", "humidity", "rainfall"]:
            values[f] = _numeric_widget(prefix, f, defaults.get(f), f in missing, use_sliders)
        default_soil = defaults.get("soil")
        label = config.FEATURE_LABELS["soil"] + (" ⚠️ not found in report" if "soil" in missing else "")
        values["soil"] = st.selectbox(
            label, soil_options,
            index=soil_options.index(default_soil) if default_soil in soil_options else None,
            placeholder="Choose soil texture", key=f"{prefix}_soil",
            format_func=lambda s: s.title(),
        )
    return values


def _numeric_widget(prefix: str, feature: str, default, is_missing: bool, use_slider: bool):
    lo, hi, fallback, step = SLIDER_SPEC[feature]
    label = config.FEATURE_LABELS[feature] + (" ⚠️ not found in report" if is_missing else "")
    key = f"{prefix}_{feature}"
    if use_slider:
        value = float(default) if default is not None else fallback
        return st.slider(label, lo, hi, min(max(value, lo), hi), step, key=key, help=HELP[feature])
    # Number inputs for reviewing parsed values: accept anything within the hard limits, allow blanks.
    h_lo, h_hi = config.HARD_LIMITS[feature]
    return st.number_input(label, min_value=h_lo, max_value=h_hi,
                           value=float(default) if default is not None else None,
                           step=step, key=key, help=HELP[feature], placeholder="Enter a value")


# --------------------------------------------------------------------------------------
# Prediction and results
# --------------------------------------------------------------------------------------
def run_prediction(predictor: CropPredictor, values: dict, source: str) -> None:
    """Validate and predict. Store the outcome in session state, or show errors."""
    validation = validate_input(values)
    if not validation.is_valid:
        st.error("Please fix these inputs before predicting:\n\n" +
                 "\n".join(f"- {msg}" for msg in validation.errors.values()))
        return
    try:
        result = predictor.predict(values)
    except InvalidInputError as exc:  # defensive; validation above should catch everything
        st.error(str(exc))
        return
    st.session_state["result"] = result
    st.session_state["result_source"] = source


def render_results(predictor: CropPredictor, result: PredictionResult, source: str) -> None:
    st.divider()
    st.subheader("🌾 Recommended crops")
    st.caption(f"Based on {source}.")

    if result.low_confidence:
        st.warning(
            f"**Low confidence.** The top recommendation has only {result.top.confidence:.0%} probability. "
            "These conditions don't closely match any crop in the training data, so treat the ranking as a "
            "rough guide and check the tips below."
        )
    if result.warnings:
        with st.expander(f"⚠️ {len(result.warnings)} input(s) outside the training range", expanded=True):
            for msg in result.warnings.values():
                st.markdown(f"- {msg}")

    cols = st.columns(len(result.recommendations), gap="medium")
    medals = ["🥇", "🥈", "🥉"]
    for col, rec in zip(cols, result.recommendations):
        with col:
            st.markdown(
                f"<div class='crop-card'><div class='crop-rank'>{medals[rec.rank - 1]} Rank {rec.rank}</div>"
                f"<div class='crop-name'>{rec.display_name}</div>"
                f"<div class='crop-conf'>Confidence: <b>{rec.confidence:.1%}</b></div></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(rec.confidence, 0.0), 1.0))

    st.markdown("#### 💡 Farming tips")
    profiles = predictor.metadata.get("crop_profiles", {})
    tabs = st.tabs([f"{medals[r.rank - 1]} {r.display_name}" for r in result.recommendations])
    for tab, rec in zip(tabs, result.recommendations):
        with tab:
            for tip in build_tips(rec.crop, result.inputs, profiles):
                badge = "your data vs. training data" if tip.source == "data" else "general advice"
                st.markdown(f"<span class='tip-src'>{badge}</span>{tip.text}", unsafe_allow_html=True)

    with st.expander("📊 Probability for all crops"):
        proba = predictor.predict_proba_frame(pd.DataFrame([result.inputs]))
        chart = proba.T.rename(columns={0: "probability"})
        chart.index = [c.replace("_", " ").title() for c in chart.index]
        st.bar_chart(chart.sort_values("probability", ascending=False), horizontal=True)

    st.download_button("⬇️ Download summary (.txt)", data=summary_text(result, source),
                       file_name="crop_recommendation.txt", mime="text/plain")


def summary_text(result: PredictionResult, source: str) -> str:
    lines = [f"Crop recommendation, {datetime.now():%Y-%m-%d %H:%M}", f"Source: {source}", "", "Inputs:"]
    lines += [f"  {config.FEATURE_LABELS[k]}: {v}" for k, v in result.inputs.items()]
    lines += ["", "Top recommendations:"]
    lines += [f"  {r.rank}. {r.display_name}: {r.confidence:.1%}" for r in result.recommendations]
    if result.low_confidence:
        lines += ["", "Note: low confidence; conditions differ from the training data."]
    if result.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in result.warnings.values()]
    lines += ["", "Advisory only. Confirm with local agricultural extension services."]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def manual_tab(predictor: CropPredictor) -> None:
    st.markdown("Adjust the sliders to match your field, then click **Recommend crops**.")
    with st.form("manual_form", border=True):
        values = input_widgets("manual")
        submitted = st.form_submit_button("🌱 Recommend crops", type="primary", use_container_width=True)
    if submitted:
        run_prediction(predictor, values, "manual input")


def _clear_parsed_report() -> None:
    """Called when the uploaded file changes or is removed: forget the previous report and result."""
    for key in ("parsed", "parsed_file_id", "parse_error", "parsed_text", "parsed_name", "result"):
        st.session_state.pop(key, None)


def upload_tab(predictor: CropPredictor) -> None:
    st.markdown("Upload a soil or lab report (**PDF, DOCX, DOC or TXT**). The values found are filled in below "
                "for you to review and correct before predicting.")
    uploaded = st.file_uploader("Soil / lab report", type=["pdf", "docx", "doc", "txt"],
                                help="Text-based files only. Scanned images can't be read.",
                                on_change=_clear_parsed_report)

    if uploaded is not None:
        file_id = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("parsed_file_id") != file_id:  # parse each new file once
            try:
                with st.spinner(f"Reading {uploaded.name}..."):
                    parsed, text = parse_report(uploaded.getvalue(), uploaded.name)
                st.session_state.update(parsed=parsed, parsed_text=text, parsed_file_id=file_id,
                                        parsed_name=uploaded.name, parse_error=None)
            except ReportParsingError as exc:
                st.session_state.update(parsed=None, parsed_file_id=file_id, parse_error=str(exc))
            st.session_state.pop("result", None)

    if st.session_state.get("parse_error"):
        st.error(f"Couldn't read this file: {st.session_state['parse_error']}")
        return
    parsed = st.session_state.get("parsed")
    if parsed is None:
        return

    found = len(config.FEATURES) - len(parsed.missing)
    if parsed.found_all:
        st.success(f"Found all {found} values in **{st.session_state['parsed_name']}**. Review them, then predict.")
    else:
        missing_labels = ", ".join(config.FEATURE_LABELS[f] for f in parsed.missing)
        st.warning(f"Found {found} of {len(config.FEATURES)} values. Please fill in: **{missing_labels}**.")
    for feature, note in parsed.notes.items():
        st.info(f"{config.FEATURE_LABELS[feature]}: {note}.")

    with st.expander("🔎 What was extracted (matched text)"):
        st.table(pd.DataFrame(
            [{"Parameter": config.FEATURE_LABELS[f], "Value": parsed.values.get(f, "—"),
              "Matched text": parsed.matches.get(f, "not found")} for f in config.FEATURES]
        ))
        st.text_area("Extracted report text", st.session_state.get("parsed_text", ""), height=200, disabled=True)

    with st.form("review_form", border=True):
        values = input_widgets(f"review_{st.session_state['parsed_file_id']}", defaults=parsed.values,
                               missing=parsed.missing, use_sliders=False)
        submitted = st.form_submit_button("🌱 Recommend crops", type="primary", use_container_width=True)
    if submitted:
        run_prediction(predictor, values, f"report '{st.session_state['parsed_name']}'")


def sidebar(predictor: CropPredictor | None) -> None:
    with st.sidebar:
        st.header("🌱 About")
        st.markdown(
            "This app recommends the **top 3 crops** for your soil and climate using a machine-learning model "
            "trained on soil nutrient, pH, climate and soil-texture data for 10 crops."
        )
        st.markdown("**Crops covered:** " + ", ".join(c.replace("_", " ") for c in config.CROP_CLASSES) + ".")
        if predictor and predictor.metadata:
            meta = predictor.metadata
            with st.expander("Model details"):
                st.markdown(f"- Model: **{meta.get('model_name')}**")
                st.markdown(f"- Trained on {meta.get('training_data', {}).get('rows', '?')} rows")
                tm = meta.get("test_metrics") or {}
                if tm:
                    st.markdown(f"- Test accuracy: {tm.get('accuracy', 0):.1%}, top-3 accuracy: "
                                f"{tm.get('top3_accuracy', 0):.1%}")
        st.subheader("⚠️ Limitations")
        st.markdown(
            "- In the training data each crop appears on **only one soil texture**, so soil strongly limits "
            "which crops can be recommended.\n"
            "- The data is very clean (possibly synthetic). Real-world accuracy will be lower than the test scores.\n"
            "- Nutrient values are used as reported; units are not converted.\n"
            "- Recommendations and tips are **advisory**. Confirm with a local agronomist or extension officer."
        )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    predictor = get_predictor()
    sidebar(predictor)

    st.title("🌱 Crop Recommendation")
    st.markdown("Get data-driven crop suggestions from your soil test and local climate.")
    if predictor is None:
        st.stop()

    manual, upload = st.tabs(["✍️ Manual input", "📄 Upload a report"])
    with manual:
        manual_tab(predictor)
    with upload:
        upload_tab(predictor)

    if "result" in st.session_state:
        render_results(predictor, st.session_state["result"], st.session_state.get("result_source", ""))


if __name__ == "__main__":
    main()
