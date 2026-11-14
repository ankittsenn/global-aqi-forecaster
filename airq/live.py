"""Live forecasting for any place on Earth, and an offline mode that works without internet."""
import numpy as np
import pandas as pd

from . import aqi_scale, config, dataset, explain, features as ft, openmeteo
from .models import Bundle


def local_now(meta: dict) -> pd.Timestamp:
    """Current local time at the location, as a naive timestamp."""
    utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return utc + pd.Timedelta(seconds=int(meta.get("utc_offset_seconds", 0)))


def load_live(lat: float, lon: float):
    """Recent complete days (daily table) plus the CAMS model's own forecast for comparison.

    Returns (daily, cams, meta). `cams` is a Series of the model's daily mean PM2.5 for today
    and the next days, or None.
    """
    aq, wx, meta = openmeteo.fetch_live(lat, lon)
    today = local_now(meta).normalize()
    daily = dataset.hourly_to_daily(aq[aq.index < today], wx[wx.index < today])
    future = aq.loc[aq.index >= today, "pm2_5"]
    cams = dataset.daily_mean(future).dropna() if len(future) else None
    meta = dict(meta, hourly_pm25=aq.loc[aq.index < today, "pm2_5"])
    return daily, cams, meta


def percentile_context(history: pd.Series, month: int, value: float):
    """Share of days (%) in the same calendar month with lower PM2.5 than `value`."""
    if history is None or len(history) == 0:
        return None
    same = history[history.index.month == month].dropna()
    if len(same) < 30:
        return None
    return float((same < value).mean() * 100)


def _typical_for(bundle: Bundle, city, origin: pd.Timestamp, feats: pd.DataFrame) -> dict:
    known = bundle.typical.get(city, {}).get(int(origin.month)) if city else None
    if known:
        return known
    recent = feats.iloc[-90:].median(numeric_only=True)     # this place's last three months
    return {k: (None if pd.isna(v) else float(v)) for k, v in recent.items()}


def forecast_location(bundle: Bundle, name: str, lat: float, lon: float, months,
                      daily: pd.DataFrame, cams=None, history: pd.Series = None,
                      season_label: str = "", known_city: str = None,
                      source: str = "live") -> dict:
    """Run the trained models from the last observed day of `daily`."""
    daily = daily.dropna(subset=["pm25"]).sort_index()
    if len(daily) < 45:
        raise ValueError("Not enough recent data for this location (need 45 days).")
    origin = daily.index.max()
    feats = ft.city_features(daily, lat, lon, months)
    row = feats.loc[[origin]].copy()
    months = list(months)

    rows, inputs = [], {}
    for h in config.HORIZONS:
        x = row.copy()
        target = origin + pd.Timedelta(days=h)
        x["season_flag_target"] = float(target.month in months)
        inputs[h] = x[bundle.features]
        pred = bundle.horizon_models[h].predict(inputs[h]).iloc[0]
        rows.append({"h": h, "date": target, "median": pred["median"], "lo80": pred["lo80"],
                     "hi80": pred["hi80"], "lo95": pred["lo95"], "hi95": pred["hi95"]})
    forecast = pd.DataFrame(rows)
    for col in ("median", "lo80", "hi80", "lo95", "hi95"):
        forecast[f"aqi_{col}"] = aqi_scale.pm25_to_aqi(forecast[col].to_numpy())
    forecast["category"] = [aqi_scale.category(a)["name"] for a in forecast["aqi_median"]]

    p = bundle.episode.predict_proba(inputs[1]).iloc[0]
    alert = aqi_scale.alert_level(float(p["usg"]), float(p["unhealthy"]),
                                  config.ALERT_CUTOFFS["watch"], config.ALERT_CUTOFFS["likely"])
    latest = float(daily["pm25"].iloc[-1])
    latest_aqi = float(aqi_scale.pm25_to_aqi(latest))
    return {
        "name": name, "lat": lat, "lon": lon, "origin": origin, "source": source,
        "latest_pm25": latest, "latest_aqi": latest_aqi,
        "latest_category": aqi_scale.category(latest_aqi),
        "observed": daily["pm25"].iloc[-30:],
        "forecast": forecast,
        "episode": {"usg": float(p["usg"]), "unhealthy": float(p["unhealthy"])},
        "alert": alert,
        "percentile": percentile_context(history, int(origin.month), latest),
        "cams": cams, "months": months, "season_label": season_label,
        "known_city": known_city, "_inputs": inputs,
        "_typical": _typical_for(bundle, known_city, origin, feats),
    }


def drivers_for(bundle: Bundle, result: dict, h: int):
    """(effects table, sentences) explaining the forecast h days ahead."""
    x = result["_inputs"][h]
    effects = explain.explain_forecast(bundle.horizon_models[h], x, result["_typical"])
    lines = explain.describe(effects, x.iloc[0], result["_typical"], result["season_label"])
    return effects, lines


def forecast_catalogue_city(bundle: Bundle, city_row, daily_all: pd.DataFrame = None,
                            live: bool = True) -> dict:
    """Forecast one catalogue city, live if possible, otherwise from the stored dataset."""
    history = None
    if daily_all is not None:
        h = daily_all[daily_all["city"] == city_row.city].set_index("date")["pm25"]
        history = h
    if live:
        daily, cams, _ = load_live(city_row.lat, city_row.lon)
        source = "live"
    else:
        sub = daily_all[daily_all["city"] == city_row.city].set_index("date")
        daily, cams = sub[dataset.DAILY_COLS].tail(120), None
        source = f"stored dataset (up to {daily.index.max():%d %b %Y})"
    return forecast_location(bundle, city_row.city, city_row.lat, city_row.lon,
                             city_row.season_months, daily, cams, history,
                             city_row.season_label, city_row.city, source)
