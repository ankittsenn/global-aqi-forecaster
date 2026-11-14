"""The logic behind the API, kept free of web-framework code so it can be tested directly."""
import pandas as pd

from . import aqi_scale, config, dataset, live, openmeteo
from .cities import catalogue
from .models import Bundle


class Service:
    def __init__(self):
        self.bundle = Bundle.load(config.MODELS_DIR / "bundle.pkl")
        self.daily = dataset.load_daily()
        cat = catalogue()
        self.cat = cat[cat["city"].isin(self.daily["city"].unique())].reset_index(drop=True)

    def city_row(self, name: str):
        hit = self.cat[self.cat["city"].str.lower() == name.lower()]
        if hit.empty:
            raise KeyError(name)
        return hit.iloc[0]

    def cities(self) -> list:
        return self.cat[["city", "country", "region", "lat", "lon"]].to_dict("records")

    def forecast(self, name: str, use_live: bool = True) -> dict:
        row = self.city_row(name)
        source_note = None
        if use_live:
            try:
                result = live.forecast_catalogue_city(self.bundle, row, self.daily, live=True)
            except (openmeteo.ApiError, ValueError) as err:
                source_note = f"live data unavailable ({str(err)[:80]}); used stored dataset"
                result = live.forecast_catalogue_city(self.bundle, row, self.daily, live=False)
        else:
            result = live.forecast_catalogue_city(self.bundle, row, self.daily, live=False)
        fc = result["forecast"]
        days = [{"day": int(r.h), "date": r.date.date().isoformat(),
                 "pm25_median": round(float(r.median), 1),
                 "pm25_80": [round(float(r.lo80), 1), round(float(r.hi80), 1)],
                 "pm25_95": [round(float(r.lo95), 1), round(float(r.hi95), 1)],
                 "aqi": int(r.aqi_median), "category": r.category} for r in fc.itertuples()]
        return {"city": row.city, "country": row.country,
                "origin": result["origin"].date().isoformat(), "source": result["source"],
                "note": source_note,
                "latest": {"pm25": round(result["latest_pm25"], 1), "aqi": int(result["latest_aqi"]),
                           "category": result["latest_category"]["name"]},
                "alert": {**result["alert"], "p_aqi_over_100": round(result["episode"]["usg"], 3),
                          "p_aqi_over_150": round(result["episode"]["unhealthy"], 3)},
                "forecast": days, "disclaimer": aqi_scale.DISCLAIMER}

    def alerts(self) -> list:
        path = config.PROCESSED_DIR / "world_snapshot.csv"
        if not path.exists():
            return []
        snap = pd.read_csv(path)
        snap = snap[snap["alert_level"] != "none"].sort_values("p_unhealthy", ascending=False)
        return snap[["city", "country", "alert_level", "p_usg", "p_unhealthy", "aqi_d1", "aqi_d2",
                     "updated_utc"]].to_dict("records")

    @staticmethod
    def advice(aqi: float, group: str = "General public") -> dict:
        if group not in aqi_scale.GROUPS:
            raise ValueError(f"group must be one of {aqi_scale.GROUPS}")
        cat = aqi_scale.category(aqi)
        return {"aqi": aqi, "category": cat["name"], "group": group,
                "advice": aqi_scale.advice_for(cat["name"], group),
                "disclaimer": aqi_scale.DISCLAIMER}
