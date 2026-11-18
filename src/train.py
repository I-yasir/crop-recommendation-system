"""Train, evaluate and save the crop recommendation model.

Reproduces the Phase 3 selected model end to end, starting from the raw CSV:

    raw CSV -> schema check -> clean + de-duplicate -> stratified split (seed 42)
            -> preprocessing + LogisticRegression pipeline -> evaluate -> save artifacts

Usage::

    python -m src.train                    # reproduce the Phase 3 model (train on the 80 % split)
    python -m src.train --full-data        # retrain on all 1,000 unique rows (no held-out test)
    python -m src.train --output-dir /tmp/models
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src import config
from src.data_cleaning import clean_dataset
from src.data_ingestion import load_raw_data
from src.evaluate import confusion, cost_metrics, performance_metrics
from src.feature_engineering import build_model_pipeline

logger = logging.getLogger(__name__)


def build_estimator() -> LogisticRegression:
    """The estimator selected in Phase 3."""
    return LogisticRegression(**config.MODEL_PARAMS)


def split_data(df: pd.DataFrame, encoder: LabelEncoder):
    """Stratified train/test split identical to Phase 2 (same seed, same row order)."""
    X = df[config.FEATURES]
    y = encoder.transform(df[config.TARGET])
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)


def crop_profiles(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Observed [min, mean, max] of each numeric feature per crop.

    Saved in the metadata and used by :mod:`src.recommendations` to compare a user's inputs with the
    conditions under which each crop appears in the training data.
    """
    grouped = df.groupby(config.TARGET)[config.NUMERIC_FEATURES]
    mins, means, maxs = grouped.min(), grouped.mean(), grouped.max()
    return {
        crop: {f: [round(float(mins.loc[crop, f]), 2), round(float(means.loc[crop, f]), 2),
                   round(float(maxs.loc[crop, f]), 2)] for f in config.NUMERIC_FEATURES}
        for crop in mins.index
    }


def train(output_dir: Path = config.MODELS_DIR, raw_path: Path = config.RAW_DATA_PATH,
          full_data: bool = False, measure_cost: bool = True) -> dict:
    """Run the full training pipeline and write artifacts to ``output_dir``.

    Returns:
        The metadata dictionary that was saved to ``metadata.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    clean, cleaning_report = clean_dataset(raw)

    encoder = LabelEncoder().fit(config.CROP_CLASSES)
    unknown = set(clean[config.TARGET]) - set(config.CROP_CLASSES)
    if unknown:
        raise ValueError(f"Unknown crop labels in data: {sorted(unknown)}")

    X_train, X_test, y_train, y_test = split_data(clean, encoder)
    if full_data:
        X_train = clean[config.FEATURES]
        y_train = encoder.transform(clean[config.TARGET])

    model: Pipeline = build_model_pipeline(build_estimator())
    model.fit(X_train, y_train)

    model_path = output_dir / config.MODEL_FILENAME
    joblib.dump(model, model_path, compress=3)
    joblib.dump(encoder, output_dir / config.ENCODER_FILENAME)

    metadata: dict = {
        "model_name": config.MODEL_NAME,
        "estimator_class": type(model.named_steps["model"]).__name__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_by": "src.train",
        "artifacts": {"model": config.MODEL_FILENAME, "label_encoder": config.ENCODER_FILENAME},
        "features": {"numeric": config.NUMERIC_FEATURES, "categorical": config.CATEGORICAL_FEATURES,
                     "order": config.FEATURES},
        "soil_categories": config.SOIL_CATEGORIES,
        "classes": encoder.classes_.tolist(),
        "feature_ranges": {k: list(v) for k, v in config.TRAINING_RANGES.items()},
        "crop_profiles": crop_profiles(clean),
        "cleaning": cleaning_report.as_dict(),
        "training_data": {"source": str(Path(raw_path).name), "full_data": full_data,
                          "rows": int(len(X_train)), "random_state": config.RANDOM_STATE},
        "model_params": config.MODEL_PARAMS,
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }

    if full_data:
        metadata["test_metrics"] = None
        metadata["note"] = "Trained on all rows; no held-out test. See the Phase 3 notebook for unbiased metrics."
    else:
        proba = model.predict_proba(X_test)
        n_classes = len(encoder.classes_)
        metadata["test_metrics"] = performance_metrics(y_test, proba, n_classes, k=config.TOP_K)
        metadata["confusion_matrix_test"] = confusion(y_test, proba, n_classes)
        metadata["training_data"]["test_rows"] = int(len(X_test))

    if measure_cost:
        metadata["cost"] = cost_metrics(model, model_path, X_test)

    (output_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    logger.info("Saved model, encoder and metadata to %s", output_dir)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the crop recommendation model.")
    parser.add_argument("--output-dir", type=Path, default=config.MODELS_DIR, help="Where to write artifacts.")
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH, help="Raw CSV to train on.")
    parser.add_argument("--full-data", action="store_true", help="Train on all unique rows (no test split).")
    parser.add_argument("--skip-cost", action="store_true", help="Skip latency/size/memory measurement.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    meta = train(args.output_dir, args.data, full_data=args.full_data, measure_cost=not args.skip_cost)

    print(f"\nModel: {meta['model_name']}  |  trained on {meta['training_data']['rows']} rows")
    print(f"Cleaning: {meta['cleaning']['rows_in']} -> {meta['cleaning']['rows_out']} rows "
          f"({meta['cleaning']['duplicates_removed']} duplicates removed)")
    if meta.get("test_metrics"):
        print("Test metrics:", {k: round(v, 4) for k, v in meta["test_metrics"].items()})
    if meta.get("cost"):
        print("Cost:", {k: round(v, 3) for k, v in meta["cost"].items()})
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
