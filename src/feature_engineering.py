"""Feature transformation: the leakage-safe preprocessing pipeline designed in Phase 2.

Engineered features (NPK ratios, totals, temperature-humidity interaction) were evaluated in
Phase 2 and gave 0 pp CV gain, so they are intentionally **not** used. The model consumes the
8 raw inputs only.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor(
    numeric: list[str] | None = None, categorical: list[str] | None = None
) -> ColumnTransformer:
    """Create the preprocessing ``ColumnTransformer``.

    * numeric: median imputation, then standard scaling;
    * categorical: most-frequent imputation, then one-hot encoding (unknown categories become all zeros).

    Must be fitted on training data only. Wrapping it in :func:`build_model_pipeline` ensures that.
    """
    numeric = numeric or config.NUMERIC_FEATURES
    categorical = categorical or config.CATEGORICAL_FEATURES
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def build_model_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Bundle the preprocessor and an estimator into one serialisable pipeline."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
