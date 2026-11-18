"""Tests for ingestion, cleaning and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_cleaning import clean_dataset, count_label_conflicts, normalize_categories
from src.data_ingestion import SchemaError, load_raw_data, validate_schema
from src.feature_engineering import build_preprocessor


@pytest.fixture(scope="module")
def raw():
    return load_raw_data()


def test_raw_schema(raw):
    assert raw.shape == (5000, 9)
    assert list(raw.columns) == config.FEATURES + [config.TARGET]


def test_schema_alias_and_missing_column(tmp_path):
    df = pd.DataFrame({c: [1.0] for c in config.NUMERIC_FEATURES})
    df["soil_texture"], df[config.TARGET] = "loamy", "maize"
    path = tmp_path / "aliased.csv"
    df.to_csv(path, index=False)
    assert "soil" in load_raw_data(path).columns  # alias applied

    with pytest.raises(SchemaError, match="missing columns"):
        validate_schema(df.drop(columns=["soil_texture", "ph"]))


def test_schema_rejects_non_numeric():
    df = pd.DataFrame({c: [1.0] for c in config.NUMERIC_FEATURES} | {"soil": ["clay"], config.TARGET: ["rice"]})
    df["N"] = ["high"]
    with pytest.raises(SchemaError, match="not numeric"):
        validate_schema(df)


def test_normalize_categories_strips_whitespace():
    df = pd.DataFrame({"soil": ["loamy clay ", " Clay"], "x": [1, 2]})
    assert normalize_categories(df)["soil"].tolist() == ["loamy clay", "clay"]


def test_clean_dataset_matches_phase2(raw):
    clean, report = clean_dataset(raw)
    assert report.rows_in == 5000 and report.rows_out == 1000
    assert report.duplicates_removed == 4000
    assert report.label_conflicts == 0
    assert sorted(clean["soil"].unique()) == config.SOIL_CATEGORIES
    assert clean[config.TARGET].value_counts().eq(100).all()
    assert not clean.duplicated().any()


def test_label_conflicts_detected(raw):
    clean, _ = clean_dataset(raw)
    conflict = clean.iloc[[0]].copy()
    conflict[config.TARGET] = "rice" if clean.iloc[0][config.TARGET] != "rice" else "maize"
    bad = pd.concat([clean, conflict], ignore_index=True)
    assert count_label_conflicts(bad) == 1
    with pytest.raises(ValueError, match="more than one crop"):
        clean_dataset(bad)


def test_clean_drops_missing_and_impossible_rows(raw):
    df = raw.head(20).copy()
    df.loc[0, "N"] = np.nan
    df.loc[1, "ph"] = 20  # impossible pH
    clean, report = clean_dataset(df)
    assert report.rows_dropped_missing == 1
    assert report.out_of_limit_values == {"ph": 1}
    assert clean["ph"].max() <= 14


def test_preprocessor_output_shape(raw):
    clean, _ = clean_dataset(raw)
    pre = build_preprocessor().fit(clean[config.FEATURES])
    out = pre.transform(clean[config.FEATURES])
    assert out.shape == (1000, len(config.NUMERIC_FEATURES) + len(config.SOIL_CATEGORIES))
    assert np.allclose(out[:, :7].mean(axis=0), 0, atol=1e-9)


def test_preprocessor_handles_missing_and_unknown_soil(raw):
    clean, _ = clean_dataset(raw)
    pre = build_preprocessor().fit(clean[config.FEATURES])
    probe = pd.DataFrame([{**{c: np.nan for c in config.NUMERIC_FEATURES}, "soil": "peat"}])
    out = pre.transform(probe)
    assert not np.isnan(out).any()
    assert out[0, 7:].sum() == 0  # unknown soil -> all-zero one-hot
