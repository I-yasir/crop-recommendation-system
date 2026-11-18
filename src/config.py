"""Central configuration: paths, feature definitions, valid ranges and the random seed.

Every other module imports from here, so no paths or magic numbers are hard-coded elsewhere
(`docs/workflow.md` §4). Values are grounded in the Phase 2 analysis (`docs/phase_logs.md`).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "original_dataset.csv"
DATA_DICTIONARY_PATH: Path = DATA_DIR / "data_dictionary.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

MODEL_FILENAME = "crop_model.joblib"
ENCODER_FILENAME = "label_encoder.joblib"
METADATA_FILENAME = "metadata.json"

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CATEGORICAL_FEATURES: list[str] = ["soil"]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = "crop_label"

# The data dictionary calls the soil column `soil_texture`; the CSV calls it `soil`.
COLUMN_ALIASES: dict[str, str] = {"soil_texture": "soil"}

SOIL_CATEGORIES: list[str] = ["clay", "loamy", "loamy clay", "sandy loam"]
CROP_CLASSES: list[str] = [
    "beans", "cassava", "guinea_corn", "maize", "orange",
    "pepper", "rice", "soybean", "tomatoes", "yam",
]

# Human-friendly labels and units (used by the app and the report parser).
FEATURE_LABELS: dict[str, str] = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
    "soil": "Soil texture",
}

# --------------------------------------------------------------------------------------
# Valid ranges
# --------------------------------------------------------------------------------------
# TRAINING_RANGES: min/max observed in the de-duplicated training data (Phase 2).
# Inputs outside these ranges are accepted but produce a *warning*: the model has not seen them.
TRAINING_RANGES: dict[str, tuple[float, float]] = {
    "N": (0.0, 120.0),
    "P": (5.0, 80.0),
    "K": (5.0, 85.0),
    "temperature": (10.01, 34.95),
    "humidity": (14.26, 95.0),
    "ph": (5.01, 8.87),
    "rainfall": (20.21, 298.56),
}

# HARD_LIMITS: physically plausible bounds. Values outside these are rejected as *errors*
# (almost certainly a typo or a mis-parsed number).
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "N": (0.0, 500.0),
    "P": (0.0, 500.0),
    "K": (0.0, 1000.0),
    "temperature": (-20.0, 60.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 5000.0),
}

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
# Selected in Phase 3 (`reports/evaluation_log.json`): LogisticRegression with default C.
MODEL_NAME = "LogisticRegression"
MODEL_PARAMS: dict = {"max_iter": 5000, "random_state": RANDOM_STATE}

TOP_K: int = 3
# Below this top-1 probability the app flags the recommendation as low confidence.
LOW_CONFIDENCE_THRESHOLD: float = 0.60
