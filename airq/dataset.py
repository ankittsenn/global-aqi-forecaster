"""Turn hourly API data into the daily table and diurnal profiles used by the models."""
import numpy as np
import pandas as pd

from . import config

DAILY_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co", "dust", "aod", "temp", "rh",
              "precip", "pressure", "wind", "wind_u", "wind_v", "cloud"]
MIN_HOURS = 18   # a day needs at least this many valid hours to get a daily value

_AQ_RENAME = {"pm2_5": "pm25", "pm10": "pm10", "nitrogen_dioxide": "no2",
              "sulphur_dioxide": "so2", "ozone": "o3", "carbon_monoxide": "co",
              "dust": "dust", "aerosol_optical_depth": "aod"}
_WX_RENAME = {"temperature_2m": "temp", "relative_humidity_2m": "rh",
              "precipitation": "precip", "surface_pressure": "pressure",
              "wind_speed_10m": "wind", "wind_direction_10m": "wdir", "cloud_cover": "cloud"}


def daily_mean(series: pd.Series, how: str = "mean") -> pd.Series:
    """Daily mean (or sum) of an hourly series; days with too few valid hours are NaN."""
    resampled = series.resample("D")
    values = resampled.sum(min_count=1) if how == "sum" else resampled.mean()
    return values.where(resampled.count() >= MIN_HOURS)


def hourly_to_daily(aq: pd.DataFrame, wx: pd.DataFrame = None) -> pd.DataFrame:
    """Daily table (local calendar days) from hourly air quality and weather frames.

    Short gaps (up to 3 hours) are interpolated. Wind direction is averaged through its
    east-west and north-south components, because the mean of angles is meaningless.
    """
    out = {}
    aq = aq.rename(columns=_AQ_RENAME).interpolate(limit=3, limit_area="inside")
    for col in _AQ_RENAME.values():
        if col in aq:
            out[col] = daily_mean(aq[col])

    if wx is not None:
        wx = wx.rename(columns=_WX_RENAME).interpolate(limit=3, limit_area="inside")
        for col in ("temp", "rh", "pressure", "wind", "cloud"):
            if col in wx:
                out[col] = daily_mean(wx[col])
        if "precip" in wx:
            out["precip"] = daily_mean(wx["precip"], how="sum")
        if "wdir" in wx and "wind" in wx:
            radians = np.deg2rad(wx["wdir"])
            out["wind_u"] = daily_mean(-wx["wind"] * np.sin(radians))   # eastward component
            out["wind_v"] = daily_mean(-wx["wind"] * np.cos(radians))   # northward component

    daily = pd.DataFrame(out)
    daily.index.name = "date"
    for col in DAILY_COLS:
        if col not in daily:
            daily[col] = np.nan
    return daily[DAILY_COLS]


def diurnal_profile(aq: pd.DataFrame) -> pd.DataFrame:
    """Typical shape of the day: hourly PM2.5 divided by the day's mean, by month.

    Returns columns month, hour, ratio. The ratios of one month average to 1. Days with a
    daily mean below 1 ug/m3 are ignored because their ratios are dominated by noise.
    """
    column = "pm2_5" if "pm2_5" in aq else "pm25"
    series = aq[column].interpolate(limit=3, limit_area="inside")
    frame = pd.DataFrame({"v": series})
    frame["date"] = frame.index.normalize()
    frame["hour"] = frame.index.hour
    grouped = frame.groupby("date")["v"]
    count, mean = grouped.transform("count"), grouped.transform("mean")
    keep = (count >= MIN_HOURS) & (mean >= 1.0)
    frame = frame[keep].copy()
    frame["ratio"] = frame["v"] / mean[keep]
    frame["month"] = frame["date"].dt.month
    table = frame.groupby(["month", "hour"])["ratio"].median().unstack("hour")
    table = table.reindex(columns=range(24))
    table = table.div(table.mean(axis=1), axis=0)
    long = table.stack().rename("ratio").reset_index()
    return long[["month", "hour", "ratio"]]


def quality_report(daily_by_city: dict) -> dict:
    """Coverage of the daily PM2.5 series per city."""
    report = {}
    for city, daily in daily_by_city.items():
        series = daily["pm25"]
        full = pd.date_range(series.index.min(), series.index.max(), freq="D")
        report[city] = {
            "first_day": str(full[0].date()),
            "last_day": str(full[-1].date()),
            "days": int(len(full)),
            "missing_pm25_pct": round(float(series.reindex(full).isna().mean() * 100), 2),
        }
    return report


def load_daily(path=None) -> pd.DataFrame:
    path = path or (config.PROCESSED_DIR / "daily.csv.gz")
    return pd.read_csv(path, parse_dates=["date"])


def load_diurnal(path=None) -> pd.DataFrame:
    path = path or (config.PROCESSED_DIR / "diurnal.csv.gz")
    return pd.read_csv(path)
