"""Input validation shared by the web app, the report parser and the tests.

A single function, :func:`validate_input`, turns a loose mapping of user/parsed values into a
clean, typed record and separates problems into:

* **errors**: the value is missing, not a number, or physically impossible. Prediction is blocked.
* **warnings**: the value is valid but outside the training range. Prediction proceeds, and the user
  is told the model has not seen such values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from src import config

# Free-text soil descriptions mapped to the 4 training categories. Keys are normalised
# (lower case, single spaces, hyphens/underscores turned into spaces).
SOIL_ALIASES: dict[str, str] = {
    "sandy loam": "sandy loam",
    "sandy loamy": "sandy loam",
    "loamy sand": "sandy loam",
    "loam": "loamy",
    "loamy": "loamy",
    "loamy soil": "loamy",
    "loamy clay": "loamy clay",
    "clay loam": "loamy clay",
    "clayey loam": "loamy clay",
    "clay": "clay",
    "clayey": "clay",
    "clay soil": "clay",
}


def normalize_soil(value: Any) -> str | None:
    """Map a free-text soil description to one of :data:`config.SOIL_CATEGORIES`.

    Returns ``None`` when the text cannot be mapped.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = re.sub(r"[\s_\-]+", " ", str(value).strip().lower())
    if text in SOIL_ALIASES:
        return SOIL_ALIASES[text]
    # Fall back to the longest alias contained in the text, e.g. "fine sandy loam soil".
    for alias in sorted(SOIL_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return SOIL_ALIASES[alias]
    return None


def _to_float(value: Any) -> float | None:
    """Convert to float, returning None for blanks or non-numeric text."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


@dataclass
class ValidationResult:
    """Outcome of :func:`validate_input`."""

    values: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors

    @property
    def missing(self) -> list[str]:
        """Features that were not supplied at all."""
        return [f for f, msg in self.errors.items() if msg.startswith("Missing")]


def validate_input(raw: Mapping[str, Any]) -> ValidationResult:
    """Validate and normalise one set of inputs.

    Args:
        raw: Mapping with any subset of :data:`config.FEATURES`. Numbers may be strings.

    Returns:
        A :class:`ValidationResult` with cleaned ``values`` (floats and a normalised soil string),
        plus per-feature ``errors`` and ``warnings``.
    """
    result = ValidationResult()

    for feature in config.NUMERIC_FEATURES:
        label = config.FEATURE_LABELS[feature]
        if feature not in raw or raw[feature] is None or str(raw[feature]).strip() == "":
            result.errors[feature] = f"Missing value for {label}."
            continue
        number = _to_float(raw[feature])
        if number is None:
            result.errors[feature] = f"{label} must be a number (got {raw[feature]!r})."
            continue
        lo, hi = config.HARD_LIMITS[feature]
        if not lo <= number <= hi:
            result.errors[feature] = f"{label} = {number:g} is outside the plausible range {lo:g} to {hi:g}."
            continue
        t_lo, t_hi = config.TRAINING_RANGES[feature]
        if not t_lo <= number <= t_hi:
            result.warnings[feature] = (
                f"{label} = {number:g} is outside the training range ({t_lo:g} to {t_hi:g}); "
                "the prediction is less reliable."
            )
        result.values[feature] = number

    soil_raw = raw.get("soil")
    if soil_raw is None or str(soil_raw).strip() == "":
        result.errors["soil"] = "Missing value for Soil texture."
    else:
        soil = normalize_soil(soil_raw)
        if soil is None:
            result.errors["soil"] = (
                f"Unrecognised soil texture {soil_raw!r}. Use one of: {', '.join(config.SOIL_CATEGORIES)}."
            )
        else:
            result.values["soil"] = soil

    return result
