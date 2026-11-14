"""Explaining a forecast with counterfactuals.

For each driver family (recent pollution, wind, temperature, ...) the model is asked again with
that family set to what is typical for the city and month. The difference is how much the
family moves this forecast. Effects are computed one family at a time and interact, so they do
not add up exactly to the total. For the burning/heating season the contrast is "out of
season", and for the time of year it is "the opposite season, six months away".
"""
import numpy as np
import pandas as pd

from . import features as ft

TIME_OF_YEAR = "Time of year"
BURN = "Burning or heating season"


def typical_frame(frame: pd.DataFrame, typical: dict, features: list) -> pd.DataFrame:
    """Typical feature values for every row, by its city and calendar month of the origin date."""
    out = pd.DataFrame(np.nan, index=frame.index, columns=features)
    months = frame["date"].dt.month
    for (city, month), index in frame.groupby([frame["city"], months]).groups.items():
        row = typical.get(city, {}).get(int(month))
        if row:
            out.loc[index, features] = [np.nan if row.get(f) is None else row[f] for f in features]
    return out


def ablate(X: pd.DataFrame, typical_df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Copy of X with one driver family replaced by its counterfactual values."""
    alt = X.copy()
    columns = [c for c in ft.GROUPS.get(group, []) if c in alt.columns]
    if group == BURN:
        for c in columns:
            alt[c] = 0.0
    elif group == TIME_OF_YEAR:
        for c in columns:
            alt[c] = -alt[c]                       # six months away on the seasonal circle
    else:
        for c in columns:
            if c in typical_df.columns:
                alt[c] = X[c].where(typical_df[c].isna(), typical_df[c])
    return alt


def explain_rows(model, X: pd.DataFrame, typical_df: pd.DataFrame):
    """Median forecast and the effect (ug/m3) of each driver family, for every row of X."""
    base = model.median(X)
    effects = {}
    for group in ft.GROUPS:
        effects[group] = base - model.median(ablate(X, typical_df, group))
    return base, pd.DataFrame(effects, index=X.index)


def driver_importance(model, X: pd.DataFrame, typical_df: pd.DataFrame) -> pd.DataFrame:
    """Average absolute and signed effect of each driver family over many forecasts."""
    _, effects = explain_rows(model, X, typical_df)
    table = pd.DataFrame({"mean_abs_effect": effects.abs().mean(), "mean_effect": effects.mean()})
    table.index.name = "driver"
    return table.sort_values("mean_abs_effect", ascending=False).reset_index()


def explain_forecast(model, row: pd.DataFrame, typical: dict) -> pd.DataFrame:
    """Effects for a single forecast (row is a one-row frame of model features)."""
    typical_df = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in typical.items()}],
                              index=row.index).reindex(columns=row.columns)
    _, effects = explain_rows(model, row, typical_df)
    out = effects.iloc[0].rename("effect").reset_index()
    out.columns = ["driver", "effect"]
    out["size"] = out["effect"].abs()
    return out.sort_values("size", ascending=False).reset_index(drop=True)


def _verb(effect: float) -> str:
    return "raises" if effect > 0 else "lowers"


def describe(effects: pd.DataFrame, row: pd.Series, typical: dict, season_label: str = "",
             limit: int = 5) -> list:
    """Plain-language sentences for the biggest drivers (only those that matter)."""
    lines = []
    for item in effects.itertuples():
        if len(lines) >= limit:
            break
        e = float(item.effect)
        if abs(e) < 1.0:                       # under 1 ug/m3 is not worth a sentence
            continue
        amount = f"about {abs(e):.0f} µg/m³"
        name = item.driver
        if name == "Recent pollution":
            lines.append(f"Recent pollution ({row['pm_t']:.0f} µg/m³ on the last observed "
                         f"day, typically {typical.get('pm_t', float('nan')):.0f} for this time of "
                         f"year) {_verb(e)} the forecast by {amount}.")
        elif name == "Pollution level of the last 3 months":
            word = "higher" if e > 0 else "lower"
            lines.append(f"The last three months were {word} than usual for this time of year, "
                         f"which {_verb(e)} the forecast by {amount}.")
        elif name == "Wind and stagnation":
            wind, typ = row["wind"], typical.get("wind", float("nan"))
            word = "Light" if e > 0 else "Strong"
            lines.append(f"{word} wind ({wind:.1f} m/s against a typical {typ:.1f}) {_verb(e)} "
                         f"the forecast by {amount}.")
        elif name == "Temperature":
            lines.append(f"Temperature ({row['temp']:.0f}°C against a typical "
                         f"{typical.get('temp', float('nan')):.0f}°C) {_verb(e)} the forecast "
                         f"by {amount}.")
        elif name == "Humidity, cloud and rain":
            lines.append(f"Humidity, cloud and recent rain {_verb(e)} the forecast by {amount}.")
        elif name == "Air pressure":
            lines.append(f"The air-pressure pattern {_verb(e)} the forecast by {amount}.")
        elif name == "Other pollutants and dust":
            lines.append(f"Dust and other pollutants {_verb(e)} the forecast by {amount}.")
        elif name == TIME_OF_YEAR:
            lines.append(f"The time of year (compared with the opposite season) {_verb(e)} the "
                         f"forecast by {amount}.")
        elif name == BURN:
            label = season_label or "high-pollution season"
            lines.append(f"It is currently the {label}, which {_verb(e)} the forecast by {amount}.")
    return lines
