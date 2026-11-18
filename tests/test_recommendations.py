"""Tests for farming tips."""

from src import config
from src.recommendations import GENERAL_TIPS, build_tips, input_comparison_tips, soil_health_tips

PROFILES = {"maize": {f: [10.0, 20.0, 30.0] for f in config.NUMERIC_FEATURES}}


def test_every_crop_has_general_tips():
    assert set(GENERAL_TIPS) == set(config.CROP_CLASSES)
    assert all(len(tips) >= 3 for tips in GENERAL_TIPS.values())


def test_input_comparison_flags_low_and_high():
    inputs = {f: 20.0 for f in config.NUMERIC_FEATURES} | {"N": 5.0, "rainfall": 50.0}
    tips = input_comparison_tips("maize", inputs, PROFILES)
    texts = " ".join(t.text for t in tips)
    assert "below the range" in texts and "Nitrogen" in texts
    assert "above the range" in texts and "Rainfall" in texts
    assert all(t.source == "data" for t in tips)


def test_input_comparison_all_within_range():
    tips = input_comparison_tips("maize", {f: 20.0 for f in config.NUMERIC_FEATURES}, PROFILES)
    assert len(tips) == 1 and "within the range" in tips[0].text


def test_soil_health_rules():
    texts = [t.text for t in soil_health_tips({"ph": 5.0, "humidity": 95, "rainfall": 20})]
    assert any("acidic" in t for t in texts)
    assert any("humidity" in t.lower() for t in texts)
    assert any("irrigation" in t for t in texts)


def test_build_tips_with_trained_profiles(predictor, valid_input):
    result = predictor.predict(valid_input)
    tips = build_tips(result.top.crop, result.inputs, predictor.metadata["crop_profiles"])
    assert tips and {t.source for t in tips} >= {"data", "general"}
