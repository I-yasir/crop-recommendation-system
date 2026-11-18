"""Shared pytest fixtures.

A model is trained once per test session into a temporary folder with ``src.train``, so the
tests never depend on (or overwrite) the artifacts in ``models/``.
"""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

VALID_INPUT = {"N": 40, "P": 27, "K": 45, "temperature": 21.66, "humidity": 94.79,
               "ph": 5.89, "rainfall": 112.43, "soil": "sandy loam"}  # a guinea_corn row from the data


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory) -> Path:
    from src.train import train

    out = tmp_path_factory.mktemp("models")
    train(output_dir=out, measure_cost=False)
    return out


@pytest.fixture(scope="session")
def predictor(trained_model_dir):
    from src.inference import CropPredictor

    return CropPredictor(trained_model_dir)


@pytest.fixture
def valid_input() -> dict:
    return dict(VALID_INPUT)
