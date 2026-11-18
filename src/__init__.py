"""Crop recommendation ML pipeline package.

Modules
-------
config               Paths, feature names, valid ranges, random seed.
schema               Input validation shared by the app, the report parser and tests.
data_ingestion       Load the raw CSV and verify its schema.
data_cleaning        Normalise categories, de-duplicate, check label conflicts and ranges.
feature_engineering  The leakage-safe preprocessing ``ColumnTransformer``.
train                CLI that retrains and saves the model reproducibly (``python -m src.train``).
evaluate             Performance metrics and cost measurement.
inference            ``CropPredictor``: load once, validate, return the top-3 crops.
report_parser        Extract text from PDF/DOCX/DOC/TXT reports and parse the 8 inputs.
recommendations      Farming tips per crop plus input-driven advice.
"""

__version__ = "1.0.0"
