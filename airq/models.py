"""Models: baselines, pooled gradient-boosting quantile forecasters, episode classifiers.

The gradient-boosting backend is LightGBM. If LightGBM is not installed (or AIRQ_BACKEND=sklearn
is set) scikit-learn's HistGradientBoosting is used instead, so the code and tests run anywhere.
Models are stored as plain text or bytes inside a pickle of plain Python types, which stays
loadable when library versions change.
"""
import gzip
import os
import pickle

import numpy as np
import pandas as pd

from . import config

DEFAULT_PARAMS = {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 31,
                  "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8,
                  "reg_lambda": 1.0, "seed": 7}
FAST_PARAMS = {**DEFAULT_PARAMS, "n_estimators": 60, "learning_rate": 0.10}


def default_params() -> dict:
    return dict(FAST_PARAMS if os.environ.get("AIRQ_FAST") == "1" else DEFAULT_PARAMS)


def backend_name() -> str:
    forced = os.environ.get("AIRQ_BACKEND", "").lower()
    if forced == "sklearn":
        return "sklearn"
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except Exception:
        if forced == "lightgbm":
            raise
        return "sklearn"


# ---------------------------------------------------------------------------
# Gradient boosting wrapper
# ---------------------------------------------------------------------------
class GBM:
    """One gradient-boosting model: kind 'quantile' (needs alpha) or 'binary'."""

    def __init__(self, kind: str, alpha: float = None, params: dict = None):
        self.kind, self.alpha = kind, alpha
        self.params = params or default_params()
        self.backend = backend_name()
        self.columns = None
        self._model = None
        self._booster = None

    def fit(self, X: pd.DataFrame, y):
        self.columns = list(X.columns)
        data = X[self.columns].to_numpy(dtype=float)
        y = np.asarray(y)
        y = y.astype(int) if self.kind == "binary" else y.astype(float)
        p = self.params
        if self.backend == "lightgbm":
            import lightgbm as lgb
            common = dict(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                          num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                          subsample=p["subsample"], subsample_freq=1,
                          colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                          random_state=p["seed"], verbose=-1)
            if self.kind == "quantile":
                model = lgb.LGBMRegressor(objective="quantile", alpha=self.alpha, **common)
            else:
                model = lgb.LGBMClassifier(objective="binary", **common)
            model.fit(data, y)
            self._booster = model.booster_
        else:
            from sklearn.ensemble import (HistGradientBoostingClassifier,
                                          HistGradientBoostingRegressor)
            common = dict(max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
                          max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"],
                          l2_regularization=p["reg_lambda"], random_state=p["seed"])
            if self.kind == "quantile":
                model = HistGradientBoostingRegressor(loss="quantile", quantile=self.alpha, **common)
            else:
                model = HistGradientBoostingClassifier(**common)
            model.fit(data, y)
            self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        data = X[self.columns].to_numpy(dtype=float)
        if self.backend == "lightgbm":
            return np.asarray(self._booster.predict(data))
        if self.kind == "quantile":
            return self._model.predict(data)
        return self._model.predict_proba(data)[:, 1]

    def to_state(self) -> dict:
        model = (self._booster.model_to_string() if self.backend == "lightgbm"
                 else pickle.dumps(self._model))
        return {"kind": self.kind, "alpha": self.alpha, "backend": self.backend,
                "columns": self.columns, "model": model}

    @classmethod
    def from_state(cls, state: dict) -> "GBM":
        gbm = cls(state["kind"], state["alpha"])
        gbm.backend, gbm.columns = state["backend"], state["columns"]
        if gbm.backend == "lightgbm":
            import lightgbm as lgb
            gbm._booster = lgb.Booster(model_str=state["model"])
        else:
            gbm._model = pickle.loads(state["model"])
        return gbm


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def predict_persistence(frame: pd.DataFrame) -> np.ndarray:
    """Tomorrow looks like today."""
    return frame["pm_t"].to_numpy(dtype=float)


def predict_seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    """The target day looks like the same weekday one week earlier."""
    return frame["snaive"].to_numpy(dtype=float)


def fit_climatology(panel: pd.DataFrame) -> dict:
    """Median PM2.5 for each city and calendar month, from the given (training) days."""
    valid = panel[panel["y"].notna()]
    month = valid["date"].dt.month
    table = valid.groupby([valid["city"], month])["y"].median()
    return {(city, int(m)): float(v) for (city, m), v in table.items()}


def predict_climatology(frame: pd.DataFrame, climatology: dict) -> np.ndarray:
    months = frame["target_date"].dt.month.to_numpy()
    cities = frame["city"].to_numpy()
    return np.array([climatology.get((c, int(m)), np.nan) for c, m in zip(cities, months)])


# ---------------------------------------------------------------------------
# Conformal calibration of the intervals
# ---------------------------------------------------------------------------
def conformal_adjustment(lo, hi, y, alpha: float) -> float:
    """Split-conformal correction for a quantile interval (Romano et al., CQR).

    Returns how much the interval must be widened (or may be narrowed, if negative) so that
    it covers about 1 - alpha of new observations. Inputs are on the same scale as y.
    """
    scores = np.maximum(np.asarray(lo) - np.asarray(y), np.asarray(y) - np.asarray(hi))
    n = len(scores)
    if n < 20:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------------------
# Forecaster for one horizon
# ---------------------------------------------------------------------------
class HorizonModel:
    """Pooled (all cities) model of daily PM2.5 h days ahead: median and two intervals.

    Quantile models are fitted on log(1 + PM2.5), which keeps forecasts positive and makes
    intervals multiplicative. Quantiles survive monotone transforms, so exponentiating the
    predicted quantiles gives valid quantiles of PM2.5 itself.
    """

    def __init__(self, h: int, features: list, params: dict = None):
        self.h, self.features = h, list(features)
        self.params = params or default_params()
        self.models: list = []
        self.adj80 = 0.0
        self.adj95 = 0.0

    def fit(self, X, y, X_cal, y_cal) -> "HorizonModel":
        ylog = np.log1p(np.asarray(y, dtype=float))
        self.models = [GBM("quantile", alpha=q, params=self.params).fit(X[self.features], ylog)
                       for q in config.QUANTILES]
        raw = self._raw(X_cal)
        ycal = np.log1p(np.asarray(y_cal, dtype=float))
        self.adj80 = conformal_adjustment(raw[:, 1], raw[:, 3], ycal, 0.20)
        self.adj95 = conformal_adjustment(raw[:, 0], raw[:, 4], ycal, 0.05)
        return self

    def _raw(self, X) -> np.ndarray:
        preds = np.column_stack([m.predict(X[self.features]) for m in self.models])
        return np.sort(preds, axis=1)            # no crossing quantiles

    def median(self, X) -> np.ndarray:
        """Median forecast only (cheaper; used for explanations)."""
        return np.clip(np.expm1(self.models[2].predict(X[self.features])), 0, None)

    def predict(self, X) -> pd.DataFrame:
        raw = self._raw(X)
        med = raw[:, 2]
        lo80 = np.minimum(raw[:, 1] - self.adj80, med)
        hi80 = np.maximum(raw[:, 3] + self.adj80, med)
        lo95 = np.minimum(raw[:, 0] - self.adj95, lo80)
        hi95 = np.maximum(raw[:, 4] + self.adj95, hi80)

        def back(a):
            return np.clip(np.expm1(a), 0, None)

        return pd.DataFrame({
            "median": back(med), "lo80": back(lo80), "hi80": back(hi80),
            "lo95": back(lo95), "hi95": back(hi95),
            "raw_lo80": back(raw[:, 1]), "raw_hi80": back(raw[:, 3]),
            "raw_lo95": back(raw[:, 0]), "raw_hi95": back(raw[:, 4]),
        }, index=X.index)

    def to_state(self) -> dict:
        return {"h": self.h, "features": self.features, "adj80": self.adj80, "adj95": self.adj95,
                "params": self.params, "models": [m.to_state() for m in self.models]}

    @classmethod
    def from_state(cls, state: dict) -> "HorizonModel":
        model = cls(state["h"], state["features"], state["params"])
        model.adj80, model.adj95 = state["adj80"], state["adj95"]
        model.models = [GBM.from_state(s) for s in state["models"]]
        return model


# ---------------------------------------------------------------------------
# Episode classifiers
# ---------------------------------------------------------------------------
def episode_labels(y1, y2, thresholds: dict = None) -> dict:
    """1 if the daily mean on day t+1 or t+2 reaches the threshold (the next 48 hours)."""
    thresholds = thresholds or config.THRESHOLDS
    peak = np.fmax(np.asarray(y1, dtype=float), np.asarray(y2, dtype=float))
    return {key: (peak >= thr).astype(int) for key, thr in thresholds.items()}


class EpisodeModel:
    """Calibrated probability that PM2.5 reaches each threshold within the next 48 hours."""

    def __init__(self, features: list, params: dict = None, thresholds: dict = None):
        self.features = list(features)
        self.params = params or default_params()
        self.thresholds = dict(thresholds or config.THRESHOLDS)
        self.models: dict = {}
        self.calibration: dict = {}

    def fit(self, X, labels: dict, X_cal, labels_cal: dict) -> "EpisodeModel":
        from sklearn.isotonic import IsotonicRegression
        for key in self.thresholds:
            y = np.asarray(labels[key]).astype(int)
            if y.sum() < 15 or (1 - y).sum() < 15:       # too few episodes to learn from
                self.models[key] = float(y.mean()) if len(y) else 0.0
                self.calibration[key] = None
                continue
            clf = GBM("binary", params=self.params).fit(X[self.features], y)
            self.models[key] = clf
            yc = np.asarray(labels_cal[key]).astype(int)
            if yc.sum() >= 15 and (1 - yc).sum() >= 15:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(clf.predict(X_cal[self.features]), yc)
                self.calibration[key] = (iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
            else:
                self.calibration[key] = None
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        out = {}
        for key, model in self.models.items():
            if isinstance(model, float):
                p = np.full(len(X), model)
            else:
                p = model.predict(X[self.features])
                if self.calibration.get(key):
                    xs, ys = self.calibration[key]
                    p = np.interp(p, xs, ys)
            out[key] = np.clip(p, 0.0, 1.0)
        return pd.DataFrame(out, index=X.index)

    def to_state(self) -> dict:
        models = {k: (m if isinstance(m, float) else m.to_state()) for k, m in self.models.items()}
        return {"features": self.features, "params": self.params, "thresholds": self.thresholds,
                "models": models, "calibration": self.calibration}

    @classmethod
    def from_state(cls, state: dict) -> "EpisodeModel":
        model = cls(state["features"], state["params"], state["thresholds"])
        model.models = {k: (m if isinstance(m, float) else GBM.from_state(m))
                        for k, m in state["models"].items()}
        model.calibration = state["calibration"]
        return model


# ---------------------------------------------------------------------------
# Everything the app needs, in one file
# ---------------------------------------------------------------------------
class Bundle:
    def __init__(self, horizon_models: dict, episode: EpisodeModel, features: list,
                 typical: dict, climatology: dict, meta: dict):
        self.horizon_models, self.episode, self.features = horizon_models, episode, features
        self.typical, self.climatology, self.meta = typical, climatology, meta

    def save(self, path) -> None:
        state = {
            "version": 1,
            "features": self.features,
            "horizon": {h: m.to_state() for h, m in self.horizon_models.items()},
            "episode": self.episode.to_state(),
            "typical": self.typical,
            "climatology": {f"{c}|{m}": v for (c, m), v in self.climatology.items()},
            "meta": self.meta,
        }
        with gzip.open(path, "wb", compresslevel=6) as handle:     # model text compresses ~5x
            pickle.dump(state, handle, protocol=4)

    @classmethod
    def load(cls, path) -> "Bundle":
        with open(path, "rb") as probe:
            zipped = probe.read(2) == b"\x1f\x8b"
        opener = gzip.open if zipped else open
        with opener(path, "rb") as handle:
            state = pickle.load(handle)
        clim = {}
        for key, value in state["climatology"].items():
            city, month = key.rsplit("|", 1)
            clim[(city, int(month))] = value
        return cls({int(h): HorizonModel.from_state(s) for h, s in state["horizon"].items()},
                   EpisodeModel.from_state(state["episode"]), state["features"],
                   state["typical"], clim, state["meta"])
