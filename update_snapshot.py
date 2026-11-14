"""Refresh data/processed/world_snapshot.csv, the table behind the dashboard's world view.

    python update_snapshot.py

Each city is forecast live from the latest data. If one city's request fails it falls back to
the stored dataset and is marked so. The GitHub workflow in .github/workflows runs this daily.
"""
import time
from datetime import datetime, timezone

import pandas as pd

from airq import aqi_scale, config, dataset, live, openmeteo
from airq.cities import catalogue
from airq.models import Bundle


def main() -> None:
    bundle = Bundle.load(config.MODELS_DIR / "bundle.pkl")
    daily_all = dataset.load_daily()
    cat = catalogue()
    rows = []
    for i, city in enumerate(cat.itertuples(), start=1):
        try:
            result = live.forecast_catalogue_city(bundle, city, daily_all, live=True)
        except (openmeteo.ApiError, ValueError) as err:
            print(f"[{i}/{len(cat)}] {city.city}: live data failed ({str(err)[:70]}...); using stored dataset")
            result = live.forecast_catalogue_city(bundle, city, daily_all, live=False)
        fc = result["forecast"]
        row = {"city": city.city, "country": city.country, "region": city.region,
               "lat": city.lat, "lon": city.lon, "source": "live" if result["source"] == "live" else "stored",
               "origin": result["origin"].date().isoformat(),
               "latest_pm25": round(result["latest_pm25"], 1),
               "latest_aqi": int(result["latest_aqi"]),
               "p_usg": round(result["episode"]["usg"], 3),
               "p_unhealthy": round(result["episode"]["unhealthy"], 3),
               "alert_level": result["alert"]["level"], "alert_severity": result["alert"]["severity"]}
        for h, value in zip(fc["h"], fc["median"]):
            row[f"pm25_d{h}"] = round(float(value), 1)
            row[f"aqi_d{h}"] = int(aqi_scale.pm25_to_aqi(value))
        rows.append(row)
        print(f"[{i}/{len(cat)}] {city.city}: AQI {row['latest_aqi']} now, "
              f"{row['aqi_d1']} tomorrow, alert {row['alert_level']}")
        time.sleep(1.0)

    snapshot = pd.DataFrame(rows)
    snapshot["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(config.PROCESSED_DIR / "world_snapshot.csv", index=False)
    print(f"Saved world_snapshot.csv with {len(snapshot)} cities.")


if __name__ == "__main__":
    main()
