"""Inference: load the trained pipeline once and return top-k crop recommendations.

Example::

    from src.inference import CropPredictor
    predictor = CropPredictor()                     # loads models/ once
    result = predictor.predict({"N": 90, "P": 42, "K": 43, "temperature": 26.5,
                                "humidity": 80, "ph": 6.5, "rainfall": 120, "soil": "loamy"})
    for rec in result.recommendations:
        print(rec.crop, rec.confidence)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src import config
from src.schema import ValidationResult, validate_input


class InvalidInputError(ValueError):
    """Raised when inputs fail validation. ``.validation`` holds the details."""

    def __init__(self, validation: ValidationResult):
        self.validation = validation
        super().__init__("; ".join(validation.errors.values()))


@dataclass(frozen=True)
class CropRecommendation:
    """One recommended crop."""

    rank: int
    crop: str
    confidence: float  # predicted probability, 0 to 1

    @property
    def display_name(self) -> str:
        """Human-friendly crop name, e.g. ``guinea_corn`` becomes ``Guinea Corn``."""
        return self.crop.replace("_", " ").title()


@dataclass
class PredictionResult:
    """Output of :meth:`CropPredictor.predict`."""

    recommendations: list[CropRecommendation]
    inputs: dict[str, Any]
    warnings: dict[str, str] = field(default_factory=dict)
    low_confidence: bool = False

    @property
    def top(self) -> CropRecommendation:
        return self.recommendations[0]


class CropPredictor:
    """Thin, validated wrapper around the serialised scikit-learn pipeline."""

    def __init__(self, model_dir: Path | str = config.MODELS_DIR):
        model_dir = Path(model_dir)
        model_path = model_dir / config.MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m src.train` first."
            )
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(model_dir / config.ENCODER_FILENAME)
        meta_path = model_dir / config.METADATA_FILENAME
        self.metadata: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.classes: list[str] = list(self.encoder.classes_)

    def predict(self, inputs: Mapping[str, Any], top_k: int = config.TOP_K) -> PredictionResult:
        """Validate one set of inputs and return the ``top_k`` crops by probability.

        Raises:
            InvalidInputError: If any input is missing, non-numeric or physically implausible.
        """
        validation = validate_input(inputs)
        if not validation.is_valid:
            raise InvalidInputError(validation)

        frame = pd.DataFrame([validation.values], columns=config.FEATURES)
        proba = self.model.predict_proba(frame)[0]
        order = np.argsort(proba)[::-1][:top_k]
        recs = [
            CropRecommendation(rank=i + 1, crop=str(self.encoder.inverse_transform([idx])[0]),
                               confidence=float(proba[idx]))
            for i, idx in enumerate(order)
        ]
        return PredictionResult(
            recommendations=recs,
            inputs=validation.values,
            warnings=validation.warnings,
            low_confidence=recs[0].confidence < config.LOW_CONFIDENCE_THRESHOLD,
        )

    def predict_proba_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch helper: class probabilities for an already-clean DataFrame with :data:`config.FEATURES`."""
        proba = self.model.predict_proba(df[config.FEATURES])
        return pd.DataFrame(proba, columns=self.classes, index=df.index)
