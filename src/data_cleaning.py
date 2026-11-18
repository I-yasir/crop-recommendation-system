"""Data cleaning: normalise categories, remove duplicates, check label conflicts and ranges.

Implements the Phase 2 decisions (`docs/phase_logs.md`):

* strip and lower-case all string columns (fixes ``"loamy clay "``);
* drop exact duplicates **before** any train/test split (leakage guard);
* fail loudly on label conflicts (identical features with different crops);
* keep outliers, but report values outside the physically plausible limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Summary of what :func:`clean_dataset` changed."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_values: int = 0
    rows_dropped_missing: int = 0
    label_conflicts: int = 0
    out_of_limit_values: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every string column stripped and lower-cased."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip().str.lower().astype(object)
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence (and its original order)."""
    return df.drop_duplicates().reset_index(drop=True)


def count_label_conflicts(df: pd.DataFrame) -> int:
    """Number of distinct feature rows that map to more than one crop."""
    return int((df.groupby(config.FEATURES)[config.TARGET].nunique() > 1).sum())


def out_of_limit_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count values outside :data:`config.HARD_LIMITS` per numeric feature (non-zero only)."""
    counts = {}
    for col, (lo, hi) in config.HARD_LIMITS.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            counts[col] = n
    return counts


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning procedure.

    Rows with missing values are dropped (the raw data has none; this protects against future
    data). Rows outside the hard limits are dropped as data errors.

    Returns:
        ``(clean_df, report)``.

    Raises:
        ValueError: If label conflicts are found. They would need a human decision.
    """
    report = CleaningReport(rows_in=len(df))
    out = normalize_categories(df[config.FEATURES + [config.TARGET]])

    report.missing_values = int(out.isna().sum().sum())
    if report.missing_values:
        before = len(out)
        out = out.dropna()
        report.rows_dropped_missing = before - len(out)

    before = len(out)
    out = remove_duplicates(out)
    report.duplicates_removed = before - len(out)

    report.out_of_limit_values = out_of_limit_counts(out)
    if report.out_of_limit_values:
        mask = pd.Series(True, index=out.index)
        for col, (lo, hi) in config.HARD_LIMITS.items():
            mask &= out[col].between(lo, hi)
        out = out[mask].reset_index(drop=True)

    report.label_conflicts = count_label_conflicts(out)
    if report.label_conflicts:
        raise ValueError(f"{report.label_conflicts} feature rows map to more than one crop; resolve before training.")

    report.rows_out = len(out)
    logger.info("Cleaning: %s -> %s rows (%s duplicates removed)", report.rows_in, report.rows_out,
                report.duplicates_removed)
    return out, report
