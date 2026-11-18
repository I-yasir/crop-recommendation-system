"""Actionable farming tips: general guidance per crop, plus advice driven by the user's inputs.

Two kinds of tips, labelled clearly (`docs/workflow.md` §2):

* **General crop tips** are conservative, widely accepted agronomic practices. They are advisory and not
  derived from this dataset.
* **Input comparison tips** are data-derived. They compare each input with the range in which the
  crop appears in the training data (``crop_profiles`` in ``models/metadata.json``).

These are not a substitute for local extension services or a professional soil test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src import config

GENERAL_TIPS: dict[str, list[str]] = {
    "beans": [
        "Inoculate seed with Rhizobium if beans have not been grown in the field recently; "
        "nodulation reduces nitrogen needs.",
        "Avoid waterlogging: plant on raised beds or ridges in heavy soils.",
        "Rotate with cereals to break disease cycles such as anthracnose and root rot.",
    ],
    "cassava": [
        "Plant healthy, disease-free stem cuttings (20 to 25 cm, 5 to 7 nodes) at the start of the rains.",
        "Keep the field weed-free for the first 3 to 4 months, when cassava competes poorly.",
        "Cassava removes a lot of potassium: replace it with potash or organic manure in following seasons.",
    ],
    "guinea_corn": [
        "Guinea corn (sorghum) tolerates drought, but sow when soil moisture is adequate for establishment.",
        "Thin to the recommended spacing 2 to 3 weeks after emergence to reduce competition.",
        "Watch for stem borers and bird damage near maturity; harvest promptly.",
    ],
    "maize": [
        "Split nitrogen application: part at planting, the rest at knee height (about 4 to 6 weeks).",
        "Ensure good moisture at tasseling and silking; stress then cuts yield the most.",
        "Scout for fall armyworm early and act at the first signs of leaf damage.",
    ],
    "orange": [
        "Use certified, grafted seedlings and plant on well-drained sites; citrus dislikes waterlogging.",
        "Mulch around the drip line (not touching the trunk) to conserve moisture.",
        "Apply balanced fertiliser in split doses through the year and monitor for citrus pests.",
    ],
    "pepper": [
        "Raise seedlings in a nursery and transplant at 4 to 6 true leaves.",
        "Pepper is a heavy feeder: supply nitrogen and potassium in split doses after flowering begins.",
        "Rotate away from tomato, eggplant and other Solanaceae to limit soil-borne diseases.",
    ],
    "rice": [
        "Level the field well so water depth is uniform; maintain shallow flooding during tillering.",
        "Apply nitrogen in splits (basal, tillering, panicle initiation) to reduce losses.",
        "Control weeds early, within the first 30 to 40 days, when competition is most damaging.",
    ],
    "soybean": [
        "Inoculate seed with Bradyrhizobium in fields new to soybean.",
        "Sow at the onset of reliable rains; avoid deep planting (keep it at 2.5 to 5 cm).",
        "Harvest promptly at maturity to avoid pod shattering losses.",
    ],
    "tomatoes": [
        "Stake or trellis plants and prune lower leaves to improve airflow and reduce leaf diseases.",
        "Water consistently at the base. Irregular watering causes blossom-end rot and fruit cracking.",
        "Rotate away from Solanaceae for at least 2 to 3 seasons.",
    ],
    "yam": [
        "Use healthy seed yams or minisetts treated against rot before planting.",
        "Plant on mounds or ridges and provide stakes for the vines to climb.",
        "Store harvested tubers in a cool, shaded, well-ventilated place.",
    ],
}


@dataclass(frozen=True)
class Tip:
    """One piece of advice. ``source`` is "general" (advisory) or "data" (compared with training data)."""

    text: str
    source: str


def general_tips(crop: str) -> list[Tip]:
    """General advisory tips for a crop (empty for unknown crops)."""
    return [Tip(t, "general") for t in GENERAL_TIPS.get(crop, [])]


def input_comparison_tips(crop: str, inputs: Mapping[str, float], crop_profiles: Mapping[str, dict]) -> list[Tip]:
    """Compare inputs with the [min, mean, max] range observed for ``crop`` in the training data."""
    profile = crop_profiles.get(crop)
    if not profile:
        return []
    tips: list[Tip] = []
    name = crop.replace("_", " ")
    for feature in config.NUMERIC_FEATURES:
        if feature not in inputs or feature not in profile:
            continue
        lo, mean, hi = profile[feature]
        value = float(inputs[feature])
        label = config.FEATURE_LABELS[feature]
        if value < lo:
            tips.append(Tip(f"{label} is {value:g}, below the range where {name} appears in the training data "
                            f"({lo:g} to {hi:g}). {_low_action(feature)}", "data"))
        elif value > hi:
            tips.append(Tip(f"{label} is {value:g}, above the range where {name} appears in the training data "
                            f"({lo:g} to {hi:g}). {_high_action(feature)}", "data"))
    if not tips:
        tips.append(Tip(f"All your measurements fall within the range where {name} appears in the training data.",
                        "data"))
    return tips


def _low_action(feature: str) -> str:
    return {
        "N": "Consider nitrogen fertiliser, compost or a legume rotation.",
        "P": "Consider phosphate fertiliser (e.g. SSP/TSP) or well-rotted manure.",
        "K": "Consider potash (e.g. MOP) or wood ash.",
        "temperature": "Plant in the warmer season or use mulch to warm the soil.",
        "humidity": "Expect higher water demand; mulching and irrigation help.",
        "ph": "The soil is more acidic than typical; agricultural lime can raise pH.",
        "rainfall": "Plan for supplementary irrigation.",
    }[feature]


def _high_action(feature: str) -> str:
    return {
        "N": "Reduce nitrogen applications; excess N favours leaves over yield.",
        "P": "Skip phosphate this season; excess P can lock up micronutrients.",
        "K": "Reduce potash applications.",
        "temperature": "Consider shade, mulching or planting in a cooler season.",
        "humidity": "High humidity raises fungal disease risk; improve spacing and airflow.",
        "ph": "The soil is more alkaline than typical; organic matter or elemental sulphur can lower pH.",
        "rainfall": "Ensure good drainage (ridges, raised beds) to prevent waterlogging.",
    }[feature]


def soil_health_tips(inputs: Mapping[str, float]) -> list[Tip]:
    """Crop-independent advice from widely used soil and climate thresholds."""
    tips: list[Tip] = []
    ph = inputs.get("ph")
    if ph is not None:
        if ph < 5.5:
            tips.append(Tip(f"Soil pH {ph:g} is strongly acidic; liming is usually recommended below 5.5.", "general"))
        elif ph > 7.8:
            tips.append(Tip(f"Soil pH {ph:g} is alkaline; watch for iron and zinc deficiency.", "general"))
    if inputs.get("humidity", 0) > 90:
        tips.append(Tip("Very high humidity: scout regularly for fungal diseases.", "general"))
    if inputs.get("rainfall", 1e9) < 40:
        tips.append(Tip("Low rainfall: irrigation or water harvesting will likely be needed.", "general"))
    return tips


def build_tips(crop: str, inputs: Mapping[str, float], crop_profiles: Mapping[str, dict] | None = None) -> list[Tip]:
    """All tips for one recommended crop: data comparison first, then general crop and soil tips."""
    return (input_comparison_tips(crop, inputs, crop_profiles or {})
            + general_tips(crop)
            + soil_health_tips(inputs))
