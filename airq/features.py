"""Feature engineering. Every feature of a row dated t uses information up to day t only."""
import numpy as np
import pandas as pd

from .config import MIN_HISTORY_DAYS

PM_FEATURES = ["pm_t", "pm_l1", "pm_l2", "pm_l3", "pm_l6", "pm_l7", "pm_roll3", "pm_roll7",
               "pm_roll14", "pm_roll30", "pm_std7", "pm_max7", "pm_diff1", "pm_ratio7"]
LEVEL_FEATURES = ["pm_roll90"]
OTHER_FEATURES = ["pm10", "no2", "so2", "o3", "co", "dust", "aod", "pm_ratio", "aod3", "dust3"]
WIND_FEATURES = ["wind", "wind3", "wind_u", "wind_v", "stag_days"]
TEMP_FEATURES = ["temp", "dtemp1"]
MOIST_FEATURES = ["rh", "precip", "precip3", "precip7", "cloud"]
PRESSURE_FEATURES = ["pressure", "dpress1", "dpress3"]
SEASON_FEATURES = ["doy_sin", "doy_cos"]
BURN_FEATURES = ["season_flag", "season_flag_target"]
LOCATION_FEATURES = ["lat", "lon"]

FEATURES = (PM_FEATURES + LEVEL_FEATURES + OTHER_FEATURES + WIND_FEATURES + TEMP_FEATURES
            + MOIST_FEATURES + PRESSURE_FEATURES + SEASON_FEATURES + BURN_FEATURES
            + LOCATION_FEATURES)

# Weather on the day being forecast. A real forecast would only know this approximately,
# so a model that uses it is an upper bound ("oracle"), never a deployable model.
ORACLE_RAW = ["temp", "rh", "precip", "pressure", "wind", "cloud"]
ORACLE_FEATURES = [f"{c}_tgt" for c in ORACLE_RAW]

# Driver families used in explanations (latitude and longitude are fixed, so not listed).
GROUPS = {
    "Recent pollution": PM_FEATURES,
    "Pollution level of the last 3 months": LEVEL_FEATURES,
    "Other pollutants and dust": OTHER_FEATURES,
    "Wind and stagnation": WIND_FEATURES,
    "Temperature": TEMP_FEATURES,
    "Humidity, cloud and rain": MOIST_FEATURES,
    "Air pressure": PRESSURE_FEATURES,
    "Time of year": SEASON_FEATURES,
    "Burning or heating season": BURN_FEATURES,
}
ORACLE_GROUP = "Weather on the forecast day (oracle)"


def feature_list(oracle: bool = False) -> list:
    return FEATURES + (ORACLE_FEATURES if oracle else [])


def city_features(daily: pd.DataFrame, lat: float, lon: float, months) -> pd.DataFrame:
    """Features for every day of one city. `daily` is indexed by date (see dataset.DAILY_COLS)."""
    d = daily.sort_index().asfreq("D")
    pm = d["pm25"]
    f = pd.DataFrame(index=d.index)

    f["pm_t"] = pm
    for k in (1, 2, 3, 6, 7):
        f[f"pm_l{k}"] = pm.shift(k)
    f["pm_roll3"] = pm.rolling(3, min_periods=2).mean()
    f["pm_roll7"] = pm.rolling(7, min_periods=4).mean()
    f["pm_roll14"] = pm.rolling(14, min_periods=7).mean()
    f["pm_roll30"] = pm.rolling(30, min_periods=15).mean()
    f["pm_roll90"] = pm.rolling(90, min_periods=60).mean()
    f["pm_std7"] = pm.rolling(7, min_periods=4).std()
    f["pm_max7"] = pm.rolling(7, min_periods=4).max()
    f["pm_diff1"] = pm - pm.shift(1)
    f["pm_ratio7"] = pm / (f["pm_roll7"] + 1.0)

    for col in ("pm10", "no2", "so2", "o3", "co", "dust", "aod"):
        f[col] = d[col]
    f["pm_ratio"] = (pm / d["pm10"]).where(d["pm10"] > 1.0)   # fine-particle share (smoke vs dust)
    f["aod3"] = d["aod"].rolling(3, min_periods=2).mean()
    f["dust3"] = d["dust"].rolling(3, min_periods=2).mean()

    f["wind"] = d["wind"]
    f["wind3"] = d["wind"].rolling(3, min_periods=2).mean()
    f["wind_u"], f["wind_v"] = d["wind_u"], d["wind_v"]
    calm = d["wind"] < 2.0                                    # consecutive calm days
    f["stag_days"] = (calm.astype(int).groupby((~calm).cumsum()).cumsum()
                      .where(d["wind"].notna()))

    f["temp"] = d["temp"]
    f["dtemp1"] = d["temp"] - d["temp"].shift(1)
    f["rh"], f["cloud"], f["precip"] = d["rh"], d["cloud"], d["precip"]
    f["precip3"] = d["precip"].rolling(3, min_periods=2).sum()
    f["precip7"] = d["precip"].rolling(7, min_periods=4).sum()
    f["pressure"] = d["pressure"]
    f["dpress1"] = d["pressure"] - d["pressure"].shift(1)
    f["dpress3"] = d["pressure"] - d["pressure"].shift(3)

    doy = f.index.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["season_flag"] = np.isin(f.index.month, list(months)).astype(float)
    f["lat"], f["lon"] = float(lat), float(lon)
    f["y"] = pm
    return f


def build_panel(daily: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Stack the features of all cities. `daily` has columns city, date and dataset.DAILY_COLS."""
    parts = []
    by_city = {c: g.set_index("date") for c, g in daily.groupby("city")}
    for row in catalogue.itertuples():
        if row.city not in by_city:
            continue
        f = city_features(by_city[row.city], row.lat, row.lon, row.season_months)
        f = f.iloc[MIN_HISTORY_DAYS:].copy()
        f["city"] = row.city
        f["date"] = f.index
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    return panel


def make_xy(panel: pd.DataFrame, h: int, season_months: dict, oracle: bool = False) -> pd.DataFrame:
    """Add the horizon-specific columns: the target h days ahead and what depends on h.

    Returns a copy of `panel` with target, target_date, snaive (value one week before the
    target day), season_flag_target and, for the oracle variant, the weather on the target day.
    """
    frame = panel.copy()
    grouped = frame.groupby("city", sort=False)
    frame["target"] = grouped["y"].shift(-h)
    frame["target_date"] = frame["date"] + pd.Timedelta(days=h)
    frame["snaive"] = grouped["y"].shift(7 - h)
    flag = np.zeros(len(frame))
    target_month = frame["target_date"].dt.month
    for city, months in season_months.items():
        hit = ((frame["city"] == city) & target_month.isin(months)).to_numpy()
        flag[hit] = 1.0
    frame["season_flag_target"] = flag
    if oracle:
        for col in ORACLE_RAW:
            frame[f"{col}_tgt"] = grouped[col].shift(-h)
    return frame


def typical_table(panel: pd.DataFrame) -> dict:
    """Median of every feature for each city and calendar month: {city: {month: {feature: value}}}."""
    columns = [c for c in FEATURES if c in panel.columns]
    months = panel["date"].dt.month.rename("month")
    table = panel.groupby([panel["city"], months])[columns].median()
    out: dict = {}
    for (city, month), row in table.iterrows():
        out.setdefault(city, {})[int(month)] = {
            k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    return out
