"""Shared fitting code, used by the backtest (one fit per fold) and by the final training."""
import numpy as np
import pandas as pd

from . import config, features as ft
from .models import (Bundle, EpisodeModel, HorizonModel, default_params, episode_labels,
                     fit_climatology)


def split_windows(frame: pd.DataFrame, last_origin: pd.Timestamp, cal_days: int):
    """Training origins up to `last_origin`; the final `cal_days` of them are for calibration."""
    usable = frame[(frame["date"] <= last_origin) & frame["target"].notna() & frame["pm_t"].notna()]
    cutoff = last_origin - pd.Timedelta(days=cal_days)
    return usable[usable["date"] <= cutoff], usable[usable["date"] > cutoff]


def fit_horizon(frame: pd.DataFrame, h: int, train_end: pd.Timestamp, features: list,
                params: dict = None, cal_days: int = config.CAL_DAYS) -> HorizonModel:
    """Fit the model for horizon h using only targets that fall on or before `train_end`."""
    proper, cal = split_windows(frame, train_end - pd.Timedelta(days=h), cal_days)
    return HorizonModel(h, features, params).fit(proper[features], proper["target"],
                                                 cal[features], cal["target"])


def episode_frame(panel: pd.DataFrame, season_months: dict) -> pd.DataFrame:
    """Horizon-1 features plus the 48-hour labels (needs the days t+1 and t+2)."""
    one = ft.make_xy(panel, 1, season_months)
    two = ft.make_xy(panel, 2, season_months)
    one["y1"], one["y2"] = one["target"], two["target"]
    one["valid"] = one["y1"].notna() & one["y2"].notna() & one["pm_t"].notna()
    for key, label in episode_labels(one["y1"], one["y2"]).items():
        one[f"label_{key}"] = label
    return one


def fit_episode(frame: pd.DataFrame, train_end: pd.Timestamp, features: list,
                params: dict = None, cal_days: int = config.CAL_DAYS) -> EpisodeModel:
    last_origin = train_end - pd.Timedelta(days=config.ALERT_DAYS)
    usable = frame[(frame["date"] <= last_origin) & frame["valid"]]
    cutoff = last_origin - pd.Timedelta(days=cal_days)
    proper, cal = usable[usable["date"] <= cutoff], usable[usable["date"] > cutoff]
    labels = {k: proper[f"label_{k}"].to_numpy() for k in config.THRESHOLDS}
    labels_cal = {k: cal[f"label_{k}"].to_numpy() for k in config.THRESHOLDS}
    return EpisodeModel(features, params).fit(proper, labels, cal, labels_cal)


def train_bundle(panel: pd.DataFrame, season_months: dict, params: dict = None,
                 progress=print) -> Bundle:
    """Fit every model on all the data and return the deployable bundle."""
    params = params or default_params()
    features = ft.feature_list(oracle=False)
    train_end = panel["date"].max()
    horizon_models = {}
    for h in config.HORIZONS:
        frame = ft.make_xy(panel, h, season_months)
        horizon_models[h] = fit_horizon(frame, h, train_end, features, params)
        progress(f"  horizon {h}d fitted")
    episode = fit_episode(episode_frame(panel, season_months), train_end, features, params)
    progress("  episode alert models fitted")

    from .models import backend_name
    meta = {"trained_until": str(train_end.date()), "n_rows": int(len(panel)),
            "n_cities": int(panel["city"].nunique()), "backend": backend_name(),
            "n_features": len(features), "cal_days": config.CAL_DAYS}
    return Bundle(horizon_models, episode, features, ft.typical_table(panel),
                  fit_climatology(panel), meta)
