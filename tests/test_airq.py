"""Unit tests. Run with:  python -m pytest -q
They use small synthetic series built inside the tests; no downloads, no real data."""
import os
import sys
import types

os.environ.setdefault("AIRQ_FAST", "1")          # small models keep the tests quick
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from airq import (aqi_scale, backtest, cities, config, dataset, explain, exposure,
                  features as ft, live, models, training)
from airq.readme import replace_block


# --------------------------------------------------------------------------- helpers
def synthetic_daily(city_seed: int, days: int = 520, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(city_seed)
    idx = pd.date_range(start, periods=days, freq="D", name="date")
    doy = idx.dayofyear.to_numpy()
    wind = np.clip(rng.normal(3, 1.2, days), 0.3, None)
    ar = np.zeros(days)
    for i in range(1, days):
        ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 0.4)
    pm = np.clip((30 + 25 * np.isin(idx.month, [10, 11, 12])) * np.exp(ar) / (0.5 + wind / 3), 1, None)
    frame = pd.DataFrame({"pm25": pm, "pm10": pm * 1.5, "no2": 20.0, "so2": 5.0, "o3": 50.0,
                          "co": 300.0, "dust": 5.0, "aod": 0.4,
                          "temp": 20 + 10 * np.sin(2 * np.pi * (doy - 100) / 365),
                          "rh": 60.0, "precip": 0.0, "pressure": 1000.0, "wind": wind,
                          "wind_u": rng.normal(0, 2, days), "wind_v": rng.normal(0, 2, days),
                          "cloud": 50.0}, index=idx)
    return frame[dataset.DAILY_COLS]


@pytest.fixture(scope="module")
def mini():
    names = ["Delhi", "Lima", "London"]
    cat = cities.catalogue()
    cat = cat[cat["city"].isin(names)].reset_index(drop=True)
    daily = []
    for i, name in enumerate(names):
        d = synthetic_daily(i + 1).reset_index()
        d.insert(0, "city", name)
        daily.append(d)
    daily = pd.concat(daily, ignore_index=True)
    panel = ft.build_panel(daily, cat)
    return {"cat": cat, "daily": daily, "panel": panel, "months": cities.season_months()}


# --------------------------------------------------------------------------- AQI scale
@pytest.mark.parametrize("conc,expected", [
    (0.0, 0), (9.0, 50), (9.1, 51), (12.0, 56), (35.4, 100), (35.5, 101), (55.4, 150),
    (55.5, 151), (125.4, 200), (125.5, 201), (225.4, 300), (225.5, 301), (325.4, 500), (900, 500)])
def test_pm25_to_aqi_breakpoints(conc, expected):
    assert int(aqi_scale.pm25_to_aqi(conc)) == expected


def test_aqi_truncates_to_one_decimal():
    assert int(aqi_scale.pm25_to_aqi(35.46)) == int(aqi_scale.pm25_to_aqi(35.4))


def test_categories_follow_the_official_bands():
    names = [aqi_scale.category(a)["name"] for a in (25, 75, 125, 175, 250, 400)]
    assert names == ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy",
                     "Very Unhealthy", "Hazardous"]
    assert aqi_scale.category_index(50) == 0 and aqi_scale.category_index(51) == 1


def test_alert_levels():
    assert aqi_scale.alert_level(0.05, 0.01)["level"] == "none"
    assert aqi_scale.alert_level(0.35, 0.05) == {**aqi_scale.alert_level(0.35, 0.05),
                                                 "level": "watch", "severity": "elevated"}
    assert aqi_scale.alert_level(0.70, 0.10)["level"] == "likely"
    severe = aqi_scale.alert_level(0.2, 0.65)
    assert severe["level"] == "likely" and severe["severity"] == "severe"
    assert aqi_scale.alert_level(0.2, 0.35)["severity"] == "severe"


def test_advice_exists_for_every_category_and_group():
    for cat in aqi_scale.CATEGORIES:
        for group in aqi_scale.GROUPS:
            assert aqi_scale.advice_for(cat[0], group)


# --------------------------------------------------------------------------- data
def test_daily_mean_needs_enough_hours():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    s = pd.Series(10.0, index=idx)
    s.iloc[10:30] = np.nan                    # day 1 keeps 10 hours, day 2 keeps 18
    out = dataset.daily_mean(s)
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 10.0


def test_wind_direction_is_averaged_as_vectors():
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    aq = pd.DataFrame({"pm2_5": 10.0}, index=idx)
    wx = pd.DataFrame({"wind_speed_10m": 5.0, "wind_direction_10m": [350.0, 10.0] * 12}, index=idx)
    daily = dataset.hourly_to_daily(aq, wx)
    assert abs(daily["wind_u"].iloc[0]) < 0.1          # 350 and 10 degrees cancel east-west
    assert daily["wind_v"].iloc[0] < -4.5              # wind from the north blows southward


# --------------------------------------------------------------------------- features
def test_features_use_only_the_past(mini):
    """Changing a future PM2.5 value must not change features of earlier days."""
    daily = mini["daily"]
    city = daily[daily["city"] == "Delhi"].set_index("date").drop(columns="city")
    row = mini["cat"][mini["cat"]["city"] == "Delhi"].iloc[0]
    base = ft.city_features(city, row.lat, row.lon, row.season_months)
    altered = city.copy()
    cut = altered.index[300]
    altered.loc[altered.index > cut, ["pm25", "wind", "temp"]] *= 5
    other = ft.city_features(altered, row.lat, row.lon, row.season_months)
    before = base.index <= cut
    cols = [c for c in ft.FEATURES if c in base.columns]
    pd.testing.assert_frame_equal(base[before][cols], other[before][cols])


def test_target_alignment_and_seasonal_naive(mini):
    frame = ft.make_xy(mini["panel"], 3, mini["months"])
    g = frame[frame["city"] == "Lima"].reset_index(drop=True)
    i = 100
    assert g.loc[i, "target"] == g.loc[i + 3, "y"]
    assert g.loc[i, "target_date"] == g.loc[i + 3, "date"]
    assert g.loc[i, "snaive"] == g.loc[i - 4, "y"]      # same weekday as the target day, a week earlier


def test_feature_list_oracle_adds_target_day_weather():
    assert set(ft.feature_list(True)) > set(ft.feature_list(False))


# --------------------------------------------------------------------------- uncertainty
def test_conformal_adjustment_hand_cases():
    y = np.arange(1, 20, dtype=float)
    lo, hi = np.zeros(19), np.zeros(19)
    assert models.conformal_adjustment(lo, hi, y, 0.1) == 0.0     # fewer than 20 points: no change
    y = np.arange(1, 21, dtype=float)
    assert models.conformal_adjustment(np.zeros(20), np.zeros(20), y, 0.1) == 20.0
    lo, hi, y = np.full(25, -1.0), np.full(25, 1.0), np.zeros(25)
    assert models.conformal_adjustment(lo, hi, y, 0.2) == -1.0     # interval too wide: can shrink


def test_horizon_model_quantiles_cross_free_and_cover(mini):
    frame = ft.make_xy(mini["panel"], 1, mini["months"]).dropna(subset=["target"])
    train, cal = frame.iloc[:-300], frame.iloc[-300:-150]
    test = frame.iloc[-150:]
    model = models.HorizonModel(1, ft.FEATURES).fit(train[ft.FEATURES], train["target"],
                                                    cal[ft.FEATURES], cal["target"])
    pred = model.predict(test[ft.FEATURES])
    assert (pred["lo95"] <= pred["lo80"]).all() and (pred["lo80"] <= pred["median"]).all()
    assert (pred["median"] <= pred["hi80"]).all() and (pred["hi80"] <= pred["hi95"]).all()
    inside = ((test["target"].to_numpy() >= pred["lo95"].to_numpy())
              & (test["target"].to_numpy() <= pred["hi95"].to_numpy())).mean()
    assert inside > 0.8


def test_episode_model_falls_back_when_no_positives(mini):
    ef = training.episode_frame(mini["panel"], mini["months"])
    ef = ef[ef["valid"]]
    X = ef[ft.FEATURES]
    labels = {"usg": np.zeros(len(ef)), "unhealthy": np.zeros(len(ef))}
    model = models.EpisodeModel(ft.FEATURES).fit(X, labels, X, labels)
    p = model.predict_proba(X)
    assert (p.to_numpy() < 0.05).all() and np.isfinite(p.to_numpy()).all()


def test_episode_labels_look_two_days_ahead():
    out = models.episode_labels(np.array([10, 40, 10.0]), np.array([10, 10, 60.0]))
    assert list(out["usg"]) == [0, 1, 1] and list(out["unhealthy"]) == [0, 0, 1]


# --------------------------------------------------------------------------- backtest
def test_folds_never_leak(mini):
    folds = backtest.make_folds(mini["panel"], 2, 60)
    assert folds[0].test_end < folds[1].test_start
    frames = backtest.build_frames(mini["panel"], mini["months"], oracle=False)
    ef = training.episode_frame(mini["panel"], mini["months"])
    out = backtest.run_fold(mini["panel"], frames, ef, folds[0], oracle=False)
    fc = out["forecasts"]
    assert fc["origin"].min() >= folds[0].test_start and fc["origin"].max() <= folds[0].test_end
    assert (fc["target_date"] > folds[0].train_end).all()
    assert {"persistence", "snaive", "climatology", "lgbm", "lo95", "hi95"} <= set(fc.columns)


def test_mae_table_uses_one_common_sample():
    fc = pd.DataFrame({"h": [1, 1, 1], "y": [10.0, 20.0, 30.0],
                       "a": [10.0, 22.0, np.nan], "b": [12.0, 20.0, 30.0]})
    t = backtest.mae_table(fc, ["a", "b"])
    assert t.loc["1", "a"] == 1.0 and t.loc["1", "b"] == 1.0   # row with a missing forecast dropped for both


def test_sarima_plumbing_with_fake_statsmodels(mini, monkeypatch):
    class Result:
        def extend(self, values):
            return self

        def forecast(self, steps):
            return np.full(steps, np.log1p(42.0))

    class Model:
        def __init__(self, *a, **k):
            pass

        def fit(self, **k):
            return Result()

    pkg = types.ModuleType("statsmodels")
    tsa = types.ModuleType("statsmodels.tsa")
    ss = types.ModuleType("statsmodels.tsa.statespace")
    sx = types.ModuleType("statsmodels.tsa.statespace.sarimax")
    sx.SARIMAX = Model
    for name, mod in (("statsmodels", pkg), ("statsmodels.tsa", tsa),
                      ("statsmodels.tsa.statespace", ss), ("statsmodels.tsa.statespace.sarimax", sx)):
        monkeypatch.setitem(sys.modules, name, mod)
    fold = backtest.make_folds(mini["panel"], 1, 30)[0]
    out = backtest.sarima_fold(mini["panel"], fold, progress=lambda *_: None)
    assert out["sarima"].round(3).eq(42.0).all()
    assert set(out["h"]) == set(config.HORIZONS) and out["city"].nunique() == 3


# --------------------------------------------------------------------------- planner / drivers
def test_best_window_picks_the_cleanest_hours():
    profile = np.ones(24)
    profile[13:16] = 0.4
    w = exposure.best_window(profile)
    assert w["start"] == 13 and w["reduction"] > 0.2


def test_ablation_changes_only_the_chosen_family(mini):
    frame = ft.make_xy(mini["panel"], 1, mini["months"]).dropna(subset=["target"]).iloc[-200:]
    typical = ft.typical_table(mini["panel"])
    typ = explain.typical_frame(frame, typical, ft.FEATURES)
    out = explain.ablate(frame[ft.FEATURES], typ, "Temperature")
    original = frame[ft.FEATURES]
    changed = [c for c in ft.FEATURES if not np.allclose(out[c], original[c], equal_nan=True)]
    assert changed and set(changed) <= set(ft.GROUPS["Temperature"])


# --------------------------------------------------------------------------- bundle and live
@pytest.fixture(scope="module")
def bundle(mini):
    return training.train_bundle(mini["panel"], mini["months"], progress=lambda *_: None)


def test_bundle_roundtrip(bundle, mini, tmp_path):
    path = tmp_path / "b.pkl"
    bundle.save(path)
    again = models.Bundle.load(path)
    X = ft.make_xy(mini["panel"], 2, mini["months"]).dropna(subset=["target"]).iloc[-50:][ft.FEATURES]
    pd.testing.assert_frame_equal(bundle.horizon_models[2].predict(X), again.horizon_models[2].predict(X))


def test_offline_forecast_for_a_catalogue_city(bundle, mini):
    row = mini["cat"][mini["cat"]["city"] == "Delhi"].iloc[0]
    res = live.forecast_catalogue_city(bundle, row, mini["daily"], live=False)
    fc = res["forecast"]
    assert list(fc["h"]) == list(config.HORIZONS)
    assert (fc["lo95"] <= fc["median"]).all() and (fc["median"] <= fc["hi95"]).all()
    assert 0 <= res["episode"]["usg"] <= 1 and res["alert"]["level"]
    assert res["latest_aqi"] >= 0


def test_percentile_context_ranks_against_the_same_month():
    hist = pd.Series(np.arange(1, 101, dtype=float),
                     index=pd.date_range("2023-01-01", periods=100, freq="D"))
    out = live.percentile_context(hist, 1, 90.0)
    assert out is not None


# --------------------------------------------------------------------------- misc
def test_replace_block():
    text = "a\n<!-- X:START -->\nold\n<!-- X:END -->\nb"
    assert replace_block(text, "X", "new") == "a\n<!-- X:START -->\nnew\n<!-- X:END -->\nb"
    with pytest.raises(ValueError):
        replace_block("nothing", "X", "new")


def test_city_catalogue_is_sane():
    cat = cities.catalogue()
    assert cat["city"].is_unique and cat["slug"].is_unique
    assert cat["lat"].between(-90, 90).all() and cat["lon"].between(-180, 180).all()
    assert cat["region"].nunique() >= 8 and len(cat) >= 40
    assert set(cities.season_months()) <= set(cat["city"])


# --------------------------------------------------------------------------- service (API logic)
def test_service_forecast_alerts_and_advice(bundle, mini, tmp_path, monkeypatch):
    from airq import service as svc
    s = svc.Service.__new__(svc.Service)
    s.bundle, s.daily, s.cat = bundle, mini["daily"], mini["cat"]
    out = s.forecast("delhi", use_live=False)
    assert out["city"] == "Delhi" and len(out["forecast"]) == 7
    assert out["forecast"][0]["pm25_95"][0] <= out["forecast"][0]["pm25_median"] <= out["forecast"][0]["pm25_95"][1]
    with pytest.raises(KeyError):
        s.forecast("Atlantis", use_live=False)
    assert s.advice(160)["category"] == "Unhealthy"
    with pytest.raises(ValueError):
        s.advice(160, "Aliens")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    assert s.alerts() == []


def test_profile_from_hourly_finds_the_cleanest_hours():
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    values = np.where((idx.hour >= 13) & (idx.hour < 16), 20.0, 50.0)
    p = exposure.profile_from_hourly(pd.Series(values, index=idx))
    assert abs(p.mean() - 1) < 1e-9 and exposure.best_window(p)["start"] == 13
    assert np.allclose(exposure.profile_from_hourly(pd.Series([5.0], index=idx[:1])), 1)
