"""Best time of day to be outdoors.

Within a day, PM2.5 follows a typical shape that depends on the city and the month (traffic,
heating, how deep the boundary layer gets). The forecast gives the level of the day; the shape
says when it is lowest. The air-quality model has 3-hourly steps, so the shape is coarse and
windows are three hours long. `evaluate_windows` measures on unseen days whether following
the recommendation really lowers exposure.
"""
import numpy as np
import pandas as pd

from . import dataset

WINDOW_HOURS = 3
FIRST_HOUR, LAST_HOUR = 6, 21          # people are only outdoors between 06:00 and 22:00


def profile_vector(diurnal: pd.DataFrame, month: int) -> np.ndarray:
    """24 hourly ratios for one city (the table must contain only that city) and month."""
    sub = diurnal[diurnal["month"] == month].set_index("hour")["ratio"].reindex(range(24))
    if sub.isna().all():
        return np.ones(24)
    return sub.interpolate(limit_direction="both").to_numpy()


def profile_from_hourly(hourly) -> np.ndarray:
    """Typical shape of the day from recent hourly PM2.5 of any place (all days pooled).

    Used for places outside the catalogue, where a per-month profile is not available.
    Returns 24 ratios of the hourly value to its day's mean; all ones if there is too little data.
    """
    s = pd.Series(hourly).dropna()
    s = s[s > 0]
    if len(s) < 24 * 14:
        return np.ones(24)
    daily = s.groupby(s.index.normalize()).transform("mean")
    ratio = (s / daily).groupby(s.index.hour).median().reindex(range(24))
    ratio = ratio.interpolate(limit_direction="both")
    return (ratio / ratio.mean()).to_numpy()


def window_means(profile: np.ndarray, length: int = WINDOW_HOURS) -> dict:
    return {s: float(np.mean(profile[s:s + length]))
            for s in range(FIRST_HOUR, LAST_HOUR - length + 2)}


def best_window(profile: np.ndarray, length: int = WINDOW_HOURS) -> dict:
    """Start hour of the cleanest window and how it compares with an average window."""
    means = window_means(profile, length)
    start = min(means, key=means.get)
    average = float(np.mean(list(means.values())))
    return {"start": start, "end": start + length, "ratio": means[start],
            "average_ratio": average, "worst_start": max(means, key=means.get),
            "reduction": 1.0 - means[start] / average if average else 0.0}


def hourly_plan(profile: np.ndarray, daily_mean: float) -> np.ndarray:
    """Expected hourly PM2.5 for a day: its typical shape scaled to the forecast daily mean."""
    return profile * daily_mean


def evaluate_windows(hourly: dict, train_end, test_start, test_end,
                     length: int = WINDOW_HOURS) -> dict:
    """Does the recommended window really have lower PM2.5 on days the shape was not fitted on?

    `hourly` maps city -> hourly PM2.5 Series. Profiles are built from data up to `train_end`
    only. Returns the pooled and per-city exposure reduction against a randomly chosen window
    within the allowed hours, and the share of days on which the recommendation was better.
    """
    rows = []
    for city, series in hourly.items():
        series = series.interpolate(limit=3, limit_area="inside")
        train = series[:train_end]
        if train.count() < 24 * 200:
            continue
        diurnal = dataset.diurnal_profile(train.to_frame("pm2_5"))
        profiles = {m: profile_vector(diurnal, m) for m in range(1, 13)}
        test = series[test_start:pd.Timestamp(test_end) + pd.Timedelta(hours=23)]
        for day, group in test.groupby(test.index.normalize()):
            if group.count() < 18:
                continue
            actual = np.full(24, np.nan)
            actual[group.index.hour] = group.to_numpy()
            starts = range(FIRST_HOUR, LAST_HOUR - length + 2)
            windows = {s: np.nanmean(actual[s:s + length]) for s in starts}
            choice = best_window(profiles[day.month], length)["start"]
            rows.append((city, day, windows[choice], float(np.mean(list(windows.values()))),
                         float(max(windows.values()))))
    if not rows:
        return {"days": 0}
    table = pd.DataFrame(rows, columns=["city", "day", "recommended", "average", "worst"])
    per_city = table.groupby("city")[["recommended", "average"]].sum()
    per_city["reduction"] = 1.0 - per_city["recommended"] / per_city["average"]
    return {
        "days": int(len(table)), "cities": int(table["city"].nunique()),
        "pooled_reduction": float(1.0 - table["recommended"].sum() / table["average"].sum()),
        "median_city_reduction": float(per_city["reduction"].median()),
        "share_cities_better": float((per_city["reduction"] > 0).mean()),
        "share_days_better": float((table["recommended"] <= table["average"]).mean()),
        "per_city": per_city["reduction"].round(4).to_dict(),
    }
