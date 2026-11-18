"""Tests for training reproducibility and the inference contract."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.inference import CropPredictor, InvalidInputError


def test_train_writes_artifacts(trained_model_dir):
    for name in (config.MODEL_FILENAME, config.ENCODER_FILENAME, config.METADATA_FILENAME):
        assert (trained_model_dir / name).exists()
    meta = json.loads((trained_model_dir / config.METADATA_FILENAME).read_text())
    assert meta["model_name"] == "LogisticRegression"
    assert meta["classes"] == config.CROP_CLASSES
    assert meta["training_data"]["rows"] == 800
    assert meta["test_metrics"]["f1_macro"] == pytest.approx(1.0)
    assert set(meta["crop_profiles"]) == set(config.CROP_CLASSES)


def test_train_reproduces_phase3_split(trained_model_dir):
    """The model from src.train scores identically to the Phase 3 test split saved in data/processed/."""
    model = joblib.load(trained_model_dir / config.MODEL_FILENAME)
    encoder = joblib.load(trained_model_dir / config.ENCODER_FILENAME)
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    pred = encoder.inverse_transform(model.predict(test[config.FEATURES]))
    assert (pred == test[config.TARGET]).mean() == pytest.approx(1.0)


def test_predict_contract(predictor, valid_input):
    result = predictor.predict(valid_input)
    recs = result.recommendations
    assert len(recs) == config.TOP_K
    assert [r.rank for r in recs] == [1, 2, 3]
    assert all(0.0 <= r.confidence <= 1.0 for r in recs)
    assert recs[0].confidence >= recs[1].confidence >= recs[2].confidence
    assert len({r.crop for r in recs}) == 3
    assert all(r.crop in config.CROP_CLASSES for r in recs)
    assert recs[0].crop == "guinea_corn"  # this input is a guinea_corn row from the dataset
    assert not result.low_confidence


def test_predict_normalises_soil(predictor, valid_input):
    valid_input["soil"] = "Sandy-Loam "
    assert predictor.predict(valid_input).inputs["soil"] == "sandy loam"


def test_predict_invalid_input_raises(predictor, valid_input):
    valid_input["ph"] = 20
    with pytest.raises(InvalidInputError) as exc:
        predictor.predict(valid_input)
    assert "ph" in exc.value.validation.errors


def test_predict_warns_out_of_range(predictor, valid_input):
    valid_input["rainfall"] = 900
    result = predictor.predict(valid_input)
    assert "rainfall" in result.warnings


def test_probabilities_sum_to_one(predictor):
    df = pd.read_csv(config.PROCESSED_DIR / "test.csv").head(20)
    proba = predictor.predict_proba_frame(df)
    assert list(proba.columns) == config.CROP_CLASSES
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        CropPredictor(tmp_path)
