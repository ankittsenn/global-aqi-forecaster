"""Rolling-origin backtesting. Nothing is ever split at random.

The last `n_folds` blocks of `block_days` forecast origins (about one year in total) are test
data. For each fold every model is trained only on days before the block starts, then makes
7-day forecasts from every origin day in the block. Targets of training rows always fall
before the test block, so no future information leaks in.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, features as ft, models as md, training

MODEL_LABELS = {
    "persistence": "Persistence (tomorrow = today)",
    "snaive": "Seasonal naive (same weekday last week)",
    "climatology": "Monthly climatology (city median)",
    "sarima": "SARIMA(1,0,1)(1,0,1)7",
    "lgbm": "LightGBM (pooled, no future weather)",
    "lgbm_oracle": "LightGBM + perfect weather forecast (upper bound)",
}


@dataclass
class Fold:
    index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        return self.test_start - pd.Timedelta(days=1)


def make_folds(panel: pd.DataFrame, n_folds: int = 4, block_days: int = 91) -> list:
    """Consecutive test blocks ending at the last origin that still has a 7-day-ahead target."""
    last_origin = panel["date"].max() - pd.Timedelta(days=max(config.HORIZONS))
    folds = []
    for i in range(n_folds):
        test_end = last_origin - pd.Timedelta(days=(n_folds - 1 - i) * block_days)
        folds.append(Fold(i, test_end - pd.Timedelta(days=block_days - 1), test_end))
    return folds


def build_frames(panel: pd.DataFrame, season_months: dict, oracle: bool = True) -> dict:
    return {h: ft.make_xy(panel, h, season_months, oracle=oracle) for h in config.HORIZONS}


def run_fold(panel: pd.DataFrame, frames: dict, episode_frame: pd.DataFrame, fold: Fold,
             params: dict = None, oracle: bool = True, progress=print) -> dict:
    """Fit the baselines and gradient-boosting models for one fold and forecast its test block."""
    features, oracle_features = ft.feature_list(False), ft.feature_list(True)
    climatology = md.fit_climatology(panel[panel["date"] <= fold.train_end])
    forecasts, horizon_models = [], {}

    for h in config.HORIZONS:
        frame = frames[h]
        in_block = (frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)
        test = frame[in_block & frame["target"].notna()]
        model = training.fit_horizon(frame, h, fold.train_end, features, params)
        horizon_models[h] = model
        pred = model.predict(test)
        table = pd.DataFrame({
            "fold": fold.index, "city": test["city"].to_numpy(), "origin": test["date"].to_numpy(),
            "h": h, "target_date": test["target_date"].to_numpy(), "y": test["target"].to_numpy(),
            "pm_t": test["pm_t"].to_numpy(),
            "persistence": md.predict_persistence(test), "snaive": md.predict_seasonal_naive(test),
            "climatology": md.predict_climatology(test, climatology),
            "lgbm": pred["median"].to_numpy(),
            "lo80": pred["lo80"].to_numpy(), "hi80": pred["hi80"].to_numpy(),
            "lo95": pred["lo95"].to_numpy(), "hi95": pred["hi95"].to_numpy(),
            "raw_lo80": pred["raw_lo80"].to_numpy(), "raw_hi80": pred["raw_hi80"].to_numpy(),
            "raw_lo95": pred["raw_lo95"].to_numpy(), "raw_hi95": pred["raw_hi95"].to_numpy(),
        })
        if oracle:
            oracle_model = training.fit_horizon(frame, h, fold.train_end, oracle_features, params)
            table["lgbm_oracle"] = oracle_model.predict(test)["median"].to_numpy()
        forecasts.append(table)
        progress(f"    fold {fold.index + 1}: horizon {h}d done")

    # 48-hour episode alerts
    episode = training.fit_episode(episode_frame, fold.train_end, features, params)
    in_block = ((episode_frame["date"] >= fold.test_start)
                & (episode_frame["date"] <= fold.test_end) & episode_frame["valid"])
    test_e = episode_frame[in_block]
    probs = episode.predict_proba(test_e)
    alerts = pd.DataFrame({
        "fold": fold.index, "city": test_e["city"].to_numpy(), "origin": test_e["date"].to_numpy(),
        "pm_t": test_e["pm_t"].to_numpy(),
        "peak": np.fmax(test_e["y1"].to_numpy(), test_e["y2"].to_numpy()),
    })
    for key in config.THRESHOLDS:
        alerts[f"label_{key}"] = test_e[f"label_{key}"].to_numpy()
        alerts[f"p_{key}"] = probs[key].to_numpy()

    return {"forecasts": pd.concat(forecasts, ignore_index=True), "alerts": alerts,
            "models": horizon_models, "episode": episode}


def sarima_fold(panel: pd.DataFrame, fold: Fold, order=(1, 0, 1), seasonal=(1, 0, 1, 7),
                progress=print) -> pd.DataFrame:
    """SARIMA forecasts for every city and origin of the fold.

    The model is estimated once per city on data before the block, then updated day by day
    with the newly observed values (without re-estimating) and asked for 7 steps ahead.
    Fitted on log(1 + PM2.5); forecasts are transformed back.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX      # imported lazily

    rows = []
    for city, group in panel.groupby("city", sort=False):
        series = group.set_index("date")["y"].asfreq("D")
        filled = series.interpolate(limit_direction="both").fillna(series.median())
        if filled.isna().any() or series.isna().mean() > 0.2:
            continue
        index = filled.index
        values = np.log1p(filled.clip(lower=0).to_numpy())
        n_train = int((index <= fold.train_end).sum())
        if n_train < 200:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(values[:n_train], order=order, seasonal_order=seasonal,
                                trend="c", enforce_stationarity=False,
                                enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=60)
            except Exception as err:                              # noqa: BLE001
                progress(f"    SARIMA failed for {city}: {err}")
                continue
            seen = n_train
            for origin in pd.date_range(fold.test_start, fold.test_end):
                if origin not in index:
                    continue
                position = index.get_loc(origin) + 1              # observations up to the origin
                if position > seen:
                    result = result.extend(values[seen:position])
                    seen = position
                forecast = np.asarray(result.forecast(steps=max(config.HORIZONS)))
                for h in config.HORIZONS:
                    rows.append((city, origin, h, max(float(np.expm1(forecast[h - 1])), 0.0)))
        progress(f"    fold {fold.index + 1}: SARIMA {city} done")
    return pd.DataFrame(rows, columns=["city", "origin", "h", "sarima"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def common_sample(forecasts: pd.DataFrame, models: list) -> pd.DataFrame:
    """Rows where every compared model produced a forecast, so MAEs are like for like."""
    return forecasts.dropna(subset=["y"] + list(models))


def mae_table(forecasts: pd.DataFrame, models: list, minimum_y: float = None) -> pd.DataFrame:
    """MAE (ug/m3) by horizon and over all horizons, for each model, on a common sample."""
    data = common_sample(forecasts, models)
    if minimum_y is not None:
        data = data[data["y"] >= minimum_y]
    rows = {}
    for h, group in data.groupby("h"):
        rows[h] = {m: mae(group["y"], group[m]) for m in models}
    rows["all"] = {m: mae(data["y"], data[m]) for m in models}
    table = pd.DataFrame(rows).T
    table.index = table.index.astype(str)
    table["n"] = [len(data[data["h"] == int(i)]) if i != "all" else len(data) for i in table.index]
    return table


def skill_table(mae_frame: pd.DataFrame, models: list, reference: str = "persistence") -> pd.DataFrame:
    """Relative MAE improvement over a reference model (positive = better than reference)."""
    out = pd.DataFrame(index=mae_frame.index)
    for m in models:
        out[m] = 1.0 - mae_frame[m] / mae_frame[reference]
    return out


def interval_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Coverage and width of the 80% and 95% intervals, before and after calibration."""
    data = forecasts.dropna(subset=["y", "lo80", "hi80", "lo95", "hi95"])
    rows = []
    for level, lo, hi in (("80%", "lo80", "hi80"), ("95%", "lo95", "hi95")):
        for variant, prefix in (("raw quantile models", "raw_"), ("calibrated (conformal)", "")):
            lo_c, hi_c = prefix + lo, prefix + hi
            covered = (data["y"] >= data[lo_c]) & (data["y"] <= data[hi_c])
            rows.append({"interval": level, "variant": variant,
                         "coverage": float(covered.mean()),
                         "mean_width": float((data[hi_c] - data[lo_c]).mean())})
    return pd.DataFrame(rows)


def alert_table(alerts: pd.DataFrame) -> pd.DataFrame:
    """Quality of the 48-hour episode alerts for each threshold, against a persistence rule."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    rows = []
    for key, threshold in config.THRESHOLDS.items():
        y = alerts[f"label_{key}"].to_numpy()
        p = alerts[f"p_{key}"].to_numpy()
        onset = (alerts["pm_t"] < threshold).to_numpy()          # not yet in an episode today
        base = {"threshold": key, "pm25_threshold": threshold, "rows": int(len(y)),
                "episode_rate": float(y.mean())}
        if 0 < y.sum() < len(y):
            base.update(roc_auc=float(roc_auc_score(y, p)),
                        pr_auc=float(average_precision_score(y, p)),
                        brier=float(brier_score_loss(y, p)))
        rules = {f"model, p >= {cut:.2f} ({name})": (p >= cut)
                 for name, cut in config.ALERT_CUTOFFS.items()}
        rules["rule: today's value is already above the threshold"] = (
            alerts["pm_t"].to_numpy() >= threshold)
        for name, flagged in rules.items():
            tp = int((flagged & (y == 1)).sum())
            fp = int((flagged & (y == 0)).sum())
            fn = int((~flagged & (y == 1)).sum())
            onset_events = int(((y == 1) & onset).sum())
            rows.append({**base, "alert_rule": name,
                         "precision": tp / (tp + fp) if tp + fp else np.nan,
                         "recall": tp / (tp + fn) if tp + fn else np.nan,
                         "new_episodes": onset_events,
                         "new_episode_recall": (float((flagged & (y == 1) & onset).sum()
                                                      / onset_events) if onset_events else np.nan)})
    return pd.DataFrame(rows)


def reliability_table(alerts: pd.DataFrame, key: str, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, for a reliability diagram."""
    p = alerts[f"p_{key}"].to_numpy()
    y = alerts[f"label_{key}"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        if sel.sum() >= 20:
            rows.append({"mean_predicted": float(p[sel].mean()),
                         "observed_rate": float(y[sel].mean()), "n": int(sel.sum())})
    return pd.DataFrame(rows)


def region_table(forecasts: pd.DataFrame, catalogue: pd.DataFrame, models: list) -> pd.DataFrame:
    """MAE by world region on the 1-day and 3-day horizons."""
    data = common_sample(forecasts, models).merge(catalogue[["city", "region"]], on="city")
    rows = []
    for (region, h), group in data[data["h"].isin([1, 3])].groupby(["region", "h"]):
        rows.append({"region": region, "h": int(h), "cities": int(group["city"].nunique()),
                     **{m: mae(group["y"], group[m]) for m in models}})
    return pd.DataFrame(rows)
