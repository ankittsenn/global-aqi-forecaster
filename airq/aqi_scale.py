"""The US EPA Air Quality Index for PM2.5, and plain-language health guidance.

Breakpoints follow 40 CFR Part 58 Appendix G as revised in 2024 (24-hour PM2.5):
Good 0.0-9.0, Moderate 9.1-35.4, Unhealthy for Sensitive Groups 35.5-55.4,
Unhealthy 55.5-125.4, Very Unhealthy 125.5-225.4, Hazardous 225.5 and above
(AQI 500 corresponds to 325.4 ug/m3; anything higher is reported as 500).
"""
import numpy as np

# (concentration, AQI) knots; between two knots the AQI is linear in the concentration.
_KNOTS_C = [0.0, 9.0, 9.1, 35.4, 35.5, 55.4, 55.5, 125.4, 125.5, 225.4, 225.5, 325.4]
_KNOTS_I = [0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500]

# name, first AQI value, last AQI value, colour, text colour on that colour
CATEGORIES = [
    ("Good", 0, 50, "#00e400", "#10151f"),
    ("Moderate", 51, 100, "#ffff00", "#10151f"),
    ("Unhealthy for Sensitive Groups", 101, 150, "#ff7e00", "#10151f"),
    ("Unhealthy", 151, 200, "#ff0000", "#ffffff"),
    ("Very Unhealthy", 201, 300, "#8f3f97", "#ffffff"),
    ("Hazardous", 301, 500, "#7e0023", "#ffffff"),
]
# Concentration (ug/m3) at which each category starts, with midpoints in the one-decimal
# gaps so that they can be used as plot band edges.
PM25_BAND_EDGES = [0.0, 9.05, 35.45, 55.45, 125.45, 225.45, 2000.0]


def pm25_to_aqi(conc):
    """Convert 24-hour PM2.5 (ug/m3) to the US AQI. Works on numbers and arrays."""
    c = np.asarray(conc, dtype=float)
    truncated = np.floor(c * 10 + 1e-9) / 10          # EPA truncates to one decimal
    truncated = np.maximum(truncated, 0.0)
    aqi = np.floor(np.interp(truncated, _KNOTS_C, _KNOTS_I) + 0.5)
    aqi = np.where(np.isnan(c), np.nan, aqi)
    return float(aqi) if aqi.ndim == 0 else aqi


def category_index(aqi) -> int:
    """0 for Good ... 5 for Hazardous."""
    value = float(aqi)
    for i, (_, lo, hi, _, _) in enumerate(CATEGORIES):
        if value <= hi:
            return i
    return len(CATEGORIES) - 1


def category(aqi) -> dict:
    name, lo, hi, colour, text = CATEGORIES[category_index(aqi)]
    return {"name": name, "color": colour, "text_color": text, "low": lo, "high": hi,
            "index": category_index(aqi)}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def alert_level(p_usg: float, p_unhealthy: float, watch: float = 0.30, likely: float = 0.60) -> dict:
    """Turn the two episode probabilities into one alert for the next 48 hours."""
    if p_unhealthy >= likely:
        return {"level": "likely", "severity": "severe",
                "text": "High-pollution episode likely within 48 hours: PM2.5 may reach "
                        "unhealthy levels (AQI above 150)."}
    if p_unhealthy >= watch:
        return {"level": "watch", "severity": "severe",
                "text": "Watch: an unhealthy pollution episode (AQI above 150) is possible "
                        "within 48 hours."}
    if p_usg >= likely:
        return {"level": "likely", "severity": "elevated",
                "text": "Elevated pollution likely within 48 hours: unhealthy for sensitive "
                        "groups (AQI above 100)."}
    if p_usg >= watch:
        return {"level": "watch", "severity": "elevated",
                "text": "Watch: elevated pollution (AQI above 100) is possible within 48 hours."}
    return {"level": "none", "severity": "none",
            "text": "No high-pollution episode expected in the next 48 hours."}


# ---------------------------------------------------------------------------
# Health guidance (general public information in the spirit of US EPA and WHO advice)
# ---------------------------------------------------------------------------
CATEGORY_ADVICE = {
    "Good": "Air quality is good. Normal outdoor activity is fine for everyone.",
    "Moderate": "Air quality is acceptable. Unusually sensitive people should consider "
                "shortening prolonged or heavy exertion outdoors.",
    "Unhealthy for Sensitive Groups": "People with asthma, heart or lung disease, children, "
                "older adults and pregnant people should reduce prolonged or heavy exertion "
                "outdoors. Everyone else can stay active but should watch for symptoms.",
    "Unhealthy": "Everyone may start to feel effects and sensitive groups may feel more "
                "serious ones. Cut back on prolonged or heavy exertion outdoors; sensitive "
                "groups should avoid it and move activity indoors or to a cleaner time.",
    "Very Unhealthy": "Health alert: the risk is higher for everyone. Avoid outdoor exertion; "
                "sensitive groups should stay indoors.",
    "Hazardous": "Emergency conditions. Everyone should avoid outdoor physical activity and "
                "stay indoors with windows closed.",
}

GROUPS = [
    "General public",
    "Asthma, lung or heart disease",
    "Children and teenagers",
    "Older adults",
    "Pregnancy",
    "Outdoor workers and athletes",
]

_GROUP_ADVICE = {
    "General public": [],
    "Asthma, lung or heart disease": [
        "Keep your quick-relief medicine with you and follow the action plan your doctor gave you.",
        "Start reducing outdoor exertion at AQI 101; do not wait for symptoms.",
        "Wheezing, chest tightness, unusual breathlessness, palpitations or chest pain: "
        "follow your plan and seek medical care if they do not settle quickly.",
    ],
    "Children and teenagers": [
        "Growing lungs are more sensitive: shorten outdoor play and sport from AQI 101.",
        "Move sport and recess indoors when the air is Unhealthy or worse.",
    ],
    "Older adults": [
        "Heart and lung conditions are more common and fine particles can aggravate them: "
        "limit outdoor exertion from AQI 101.",
        "Run errands and walks in the cleaner window suggested in the day planner.",
    ],
    "Pregnancy": [
        "Fine particles are linked to pregnancy complications: follow the sensitive-group "
        "advice and limit time outdoors on poor-air days.",
        "Talk to your clinician if you are worried or have symptoms.",
    ],
    "Outdoor workers and athletes": [
        "Breathing harder means inhaling more particles: lower the intensity or move training "
        "indoors from AQI 101.",
        "Schedule hard sessions in the cleanest window and take more breaks.",
        "For long outdoor exposure at Unhealthy levels or worse, a well-fitted N95, KN95 or "
        "FFP2 respirator reduces particle intake. Employers may have their own rules.",
    ],
}

_HOME_TIPS = [
    "Keep windows and doors closed while outdoor air is poor; air the home in the cleaner "
    "window of the day.",
    "A HEPA air purifier sized for the room helps. A box fan with a furnace filter is a "
    "low-cost alternative.",
    "Avoid adding indoor smoke: no smoking, incense or candles, and ventilate when frying or "
    "using a gas stove.",
    "Avoid exercising next to busy roads; keep car windows closed and use recirculation.",
]

_MASK_TIP = ("Masks: a well-fitted N95, KN95 or FFP2 respirator reduces particle exposure. "
             "Cloth and surgical masks do little against fine particles.")

DISCLAIMER = ("General information, not medical advice. If you feel unwell or have a heart or "
              "lung condition, follow your doctor's advice.")


def advice_for(category_name: str, group: str = "General public") -> list:
    """Bullet points for one air-quality category and one group of people."""
    idx = [c[0] for c in CATEGORIES].index(category_name)
    bullets = [CATEGORY_ADVICE[category_name]]
    if idx >= 2:                                    # Unhealthy for Sensitive Groups or worse
        bullets += _GROUP_ADVICE.get(group, [])
        bullets += _HOME_TIPS
        bullets.append(_MASK_TIP)
    elif idx == 1 and group != "General public":
        bullets += _GROUP_ADVICE.get(group, [])[:1]
    return bullets
