"""Data ingestion: load the raw CSV and verify it matches the expected schema."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Raised when a dataset does not have the expected columns or types."""


def load_raw_data(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset, apply column aliases and validate the schema.

    Args:
        path: CSV file to read. Defaults to ``data/original_dataset.csv``.

    Returns:
        The raw DataFrame (not cleaned or de-duplicated).

    Raises:
        FileNotFoundError: If the file does not exist.
        SchemaError: If required columns are missing or numeric columns are not numeric.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path).rename(columns=config.COLUMN_ALIASES)
    validate_schema(df)
    logger.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], path)
    return df


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Check that all expected columns exist and numeric features are numeric.

    Raises:
        SchemaError: describing every problem found.
    """
    expected = config.FEATURES + ([config.TARGET] if require_target else [])
    problems: list[str] = []

    missing = [c for c in expected if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    for col in config.NUMERIC_FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"column {col!r} is not numeric (dtype {df[col].dtype})")

    if problems:
        raise SchemaError("Dataset schema check failed: " + "; ".join(problems))
