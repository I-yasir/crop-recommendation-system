"""Tests for input validation (shared by the app, the parser and inference)."""

import pytest

from src.schema import normalize_soil, validate_input


def test_valid_input(valid_input):
    result = validate_input(valid_input)
    assert result.is_valid and not result.warnings
    assert result.values["soil"] == "sandy loam"
    assert all(isinstance(result.values[f], float) for f in ["N", "ph", "rainfall"])


def test_numeric_strings_are_accepted(valid_input):
    valid_input |= {"N": " 40 ", "rainfall": "1,112.4"}
    result = validate_input(valid_input)
    assert result.values["N"] == 40.0
    assert result.values["rainfall"] == 1112.4
    assert "rainfall" in result.warnings  # above training max (298.56)


@pytest.mark.parametrize("field,value,expected", [
    ("N", None, "Missing"),
    ("N", "", "Missing"),
    ("ph", "acidic", "must be a number"),
    ("ph", 15, "plausible range"),
    ("humidity", 120, "plausible range"),
    ("soil", "volcanic ash", "Unrecognised soil"),
])
def test_errors(valid_input, field, value, expected):
    valid_input[field] = value
    result = validate_input(valid_input)
    assert not result.is_valid
    assert expected in result.errors[field]


def test_missing_list(valid_input):
    del valid_input["P"], valid_input["soil"]
    assert set(validate_input(valid_input).missing) == {"P", "soil"}


def test_out_of_training_range_is_warning_not_error(valid_input):
    valid_input["temperature"] = 40  # plausible but above training max 34.95
    result = validate_input(valid_input)
    assert result.is_valid
    assert "temperature" in result.warnings


@pytest.mark.parametrize("text,expected", [
    ("Sandy Loam", "sandy loam"),
    ("sandy-loam", "sandy loam"),
    ("loamy clay ", "loamy clay"),
    ("Clay Loam", "loamy clay"),
    ("loam", "loamy"),
    ("CLAY", "clay"),
    ("fine sandy loam soil", "sandy loam"),
    ("peat", None),
    (None, None),
])
def test_normalize_soil(text, expected):
    assert normalize_soil(text) == expected
