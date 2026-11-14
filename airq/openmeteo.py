"""Small client for the free Open-Meteo APIs (air quality, weather archive, geocoding).

Air quality comes from the CAMS global atmospheric composition model (about 45 km grid,
3-hourly, interpolated to hourly by Open-Meteo, available from August 2022). Weather is
ERA5 reanalysis (archive) or the forecast API's recent past (live). Both are model products,
not station measurements.

Free use is non-commercial, under 10,000 calls per day and 600 per minute; the data is
CC BY 4.0. Requests are throttled and retried politely.
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WX_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
HEADERS = {"User-Agent": "global-aqi-forecaster/1.0 (+https://github.com/ankittsenn)"}

AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
           "carbon_monoxide", "dust", "aerosol_optical_depth"]
WX_VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
           "wind_speed_10m", "wind_direction_10m", "cloud_cover"]

CHUNK_DAYS = 366          # one request per year of hourly data
PAUSE_SECONDS = 2.5       # keeps a full download well under the per-minute limit


class ApiError(RuntimeError):
    """The API refused the request or could not be reached."""


def _get(url: str, params: dict, retries: int = 6, timeout: int = 90) -> dict:
    last = "no attempt made"
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException as err:
            last = str(err)
            time.sleep(4 * (attempt + 1))
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:               # rate limit: wait for the minute to pass
            last = "rate limit (HTTP 429)"
            time.sleep(min(75, 20 * (attempt + 1)))
            continue
        if response.status_code >= 500:
            last = f"server error (HTTP {response.status_code})"
            time.sleep(4 * (attempt + 1))
            continue
        try:
            reason = response.json().get("reason", response.text[:200])
        except ValueError:
            reason = response.text[:200]
        raise ApiError(f"HTTP {response.status_code}: {reason}")
    raise ApiError(f"request failed after {retries} tries ({last})")


def _hourly_frame(payload: dict):
    """Open-Meteo JSON -> (DataFrame indexed by local time, metadata)."""
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time").sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    meta = {"timezone": payload.get("timezone", "UTC"),
            "utc_offset_seconds": int(payload.get("utc_offset_seconds", 0))}
    return frame, meta


def date_chunks(start: str, end: str, days: int = CHUNK_DAYS):
    """Split [start, end] into consecutive (start, end) date strings of at most `days` days."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    chunks = []
    while first <= last:
        stop = min(first + timedelta(days=days - 1), last)
        chunks.append((first.isoformat(), stop.isoformat()))
        first = stop + timedelta(days=1)
    return chunks


def fetch_air_quality(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(AQ_VARS),
              "start_date": start, "end_date": end, "timezone": "auto",
              "domains": "cams_global", "cell_selection": "land"}
    frame, _ = _hourly_frame(_get(AQ_URL, params))
    return frame


def fetch_weather_archive(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WX_VARS),
              "start_date": start, "end_date": end, "timezone": "auto",
              "wind_speed_unit": "ms"}
    frame, _ = _hourly_frame(_get(WX_ARCHIVE_URL, params))
    return frame


def fetch_history(lat: float, lon: float, start: str, end: str):
    """Hourly air quality and weather for a long period, one request per year."""
    aq_parts, wx_parts = [], []
    for chunk_start, chunk_end in date_chunks(start, end):
        aq_parts.append(fetch_air_quality(lat, lon, chunk_start, chunk_end))
        time.sleep(PAUSE_SECONDS)
        wx_parts.append(fetch_weather_archive(lat, lon, chunk_start, chunk_end))
        time.sleep(PAUSE_SECONDS)
    aq = pd.concat(aq_parts)
    wx = pd.concat(wx_parts)
    return aq[~aq.index.duplicated(keep="last")], wx[~wx.index.duplicated(keep="last")]


def fetch_live(lat: float, lon: float):
    """The last 92 days plus the CAMS model's own 5-day forecast, and recent weather.

    Returns (air_quality, weather, meta). Both frames use local time; `meta` holds the
    time zone name and the UTC offset in seconds.
    """
    aq_params = {"latitude": lat, "longitude": lon, "hourly": ",".join(AQ_VARS),
                 "past_days": 92, "forecast_days": 5, "timezone": "auto",
                 "domains": "cams_global", "cell_selection": "land"}
    aq, meta = _hourly_frame(_get(AQ_URL, aq_params))
    wx_params = {"latitude": lat, "longitude": lon, "hourly": ",".join(WX_VARS),
                 "past_days": 92, "forecast_days": 1, "timezone": "auto",
                 "wind_speed_unit": "ms"}
    wx, _ = _hourly_frame(_get(WX_FORECAST_URL, wx_params))
    return aq, wx, meta


def geocode(query: str, count: int = 6) -> list:
    """Look up a place name; returns dicts with label, lat and lon."""
    payload = _get(GEO_URL, {"name": query, "count": count, "language": "en", "format": "json"})
    places = []
    for item in payload.get("results") or []:
        parts = [item.get("name", ""), item.get("admin1", ""), item.get("country", "")]
        places.append({"name": item.get("name", query), "country": item.get("country", ""),
                       "lat": float(item["latitude"]), "lon": float(item["longitude"]),
                       "label": ", ".join(p for p in parts if p)})
    return places
