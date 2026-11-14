"""Paths and constants shared by the whole project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, default))


# The folders can be moved with environment variables (used by the tests).
DATA_DIR = _path("AIRQ_DATA_DIR", ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = _path("AIRQ_RESULTS_DIR", ROOT / "results")
MODELS_DIR = _path("AIRQ_MODELS_DIR", ROOT / "models")

# The global CAMS air-quality series on Open-Meteo starts in August 2022.
HISTORY_START = "2022-09-01"

# Forecast horizons in days, and the quantiles the interval models learn.
HORIZONS = tuple(range(1, 8))
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)

# Daily mean PM2.5 (ug/m3) at which an episode starts: the lower edges of the
# "Unhealthy for Sensitive Groups" and "Unhealthy" categories of the US EPA scale.
THRESHOLDS = {"usg": 35.5, "unhealthy": 55.5}
# An episode alert covers the next two daily means (48 hours).
ALERT_DAYS = 2
# Probability cut-offs for the two alert levels (fixed in advance, not tuned on test data).
ALERT_CUTOFFS = {"watch": 0.30, "likely": 0.60}

# Last days of every training window that are held back to calibrate intervals/probabilities.
CAL_DAYS = 90
# Days of history a city needs before its first usable forecast origin.
MIN_HISTORY_DAYS = 30

# WHO 2021 air quality guideline for the 24-hour mean of PM2.5 (ug/m3).
WHO_PM25_24H = 15.0
