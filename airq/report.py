"""Figures and the generated results section of the README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from . import config                     # noqa: E402
from .backtest import MODEL_LABELS      # noqa: E402

INK, GRID, MUTED = "#0f172a", "#e2e8f0", "#64748b"
ACCENT, ACCENT_LIGHT, WARM = "#0369a1", "#7dd3fc", "#ea580c"
MODEL_STYLE = {
    "persistence": ("#94a3b8", "-", 1.6),
    "snaive": ("#a8b4c4", "--", 1.6),
    "climatology": ("#d6a85c", "-", 1.6),
    "sarima": ("#8b5cf6", "-", 1.8),
    "lgbm": (ACCENT, "-", 2.6),
    "lgbm_oracle": (ACCENT, ":", 2.0),
}


def _axes(figsize=(7.6, 4.3)):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_mae(mae_all: pd.DataFrame, models: list, path) -> None:
    fig, ax = _axes()
    horizons = [h for h in mae_all.index if h != "all"]
    for m in models:
        color, style, width = MODEL_STYLE[m]
        ax.plot([int(h) for h in horizons], mae_all.loc[horizons, m], style, color=color,
                linewidth=width, marker="o", markersize=3.5, label=MODEL_LABELS[m])
    ax.set_xlabel("Forecast horizon (days ahead)", color=MUTED, fontsize=9)
    ax.set_ylabel("Mean absolute error (µg/m³)", color=MUTED, fontsize=9)
    ax.set_title("Forecast error by horizon, rolling-origin backtest", color=INK, fontsize=11,
                 loc="left", fontweight="bold")
    ax.set_xticks([int(h) for h in horizons])
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    _save(fig, path)


def fig_intervals(intervals: pd.DataFrame, path) -> None:
    fig, ax = _axes((6.4, 4.0))
    levels = ["80%", "95%"]
    width = 0.34
    for j, (variant, color) in enumerate((("raw quantile models", ACCENT_LIGHT),
                                          ("calibrated (conformal)", ACCENT))):
        values = [float(intervals[(intervals["interval"] == l) & (intervals["variant"] == variant)]
                        ["coverage"].iloc[0]) for l in levels]
        bars = ax.bar(np.arange(2) + (j - 0.5) * width, values, width, color=color, label=variant)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center",
                    fontsize=9, color=INK)
    for i, nominal in enumerate((0.80, 0.95)):
        ax.hlines(nominal, i - 0.45, i + 0.45, colors=WARM, linestyles="--", linewidth=1.6,
                  label="nominal level" if i == 0 else None)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["80% interval", "95% interval"], color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of observations inside the interval", color=MUTED, fontsize=9)
    ax.set_title("Do the intervals cover what they promise?", color=INK, fontsize=11, loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, path)


def fig_reliability(tables: dict, path) -> None:
    fig, ax = _axes((5.4, 4.6))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="perfectly calibrated")
    names = {"usg": "Unhealthy for sensitive groups (≥ 35.5 µg/m³)",
             "unhealthy": "Unhealthy (≥ 55.5 µg/m³)"}
    for (key, table), color in zip(tables.items(), (ACCENT, WARM)):
        if len(table):
            ax.plot(table["mean_predicted"], table["observed_rate"], "-o", color=color,
                    markersize=4, linewidth=2, label=names.get(key, key))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of an episode within 48 h", color=MUTED, fontsize=9)
    ax.set_ylabel("Observed frequency", color=MUTED, fontsize=9)
    ax.set_title("Are the alert probabilities honest?", color=INK, fontsize=11, loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path)


def fig_drivers(importance: pd.DataFrame, path) -> None:
    """importance has columns horizon, driver, mean_abs_effect."""
    horizons = sorted(importance["horizon"].unique())
    order = (importance[importance["horizon"] == horizons[0]]
             .sort_values("mean_abs_effect")["driver"].tolist())
    fig, ax = _axes((7.6, 4.4))
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    height = 0.8 / len(horizons)
    colors = [ACCENT, ACCENT_LIGHT, "#bae6fd"]
    for j, h in enumerate(horizons):
        sub = importance[importance["horizon"] == h].set_index("driver").reindex(order)
        ax.barh(np.arange(len(order)) + (j - (len(horizons) - 1) / 2) * height,
                sub["mean_abs_effect"], height, color=colors[j % 3], label=f"{h}-day forecast")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9, color=INK)
    ax.set_xlabel("Average effect on the forecast (µg/m³), against a typical day",
                  color=MUTED, fontsize=9)
    ax.set_title("What moves the forecast", color=INK, fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def md_table(rows: list, header: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def _better(skill: float) -> str:
    return f"{abs(skill) * 100:.0f}% {'lower' if skill > 0 else 'higher'}"


def results_markdown(ctx: dict) -> str:
    info, models = ctx["info"], ctx["models"]
    mae_all, mae_high, skill = ctx["mae_all"], ctx["mae_high"], ctx["skill"]
    out = [f"_Generated by `python evaluate.py` on {info['generated']} from data up to "
           f"{info['data_end']}. Backtest: {info['n_folds']} consecutive blocks of "
           f"{info['block_days']} forecast days ({info['test_start']} to {info['test_end']}), "
           f"{info['n_cities']} cities, {info['n_h1']:,} one-day-ahead forecasts per model. Every "
           f"model is trained only on days before its block starts._", ""]

    cols = ["1", "2", "3", "5", "7", "all"]
    head = ["Model", "1 day", "2 days", "3 days", "5 days", "7 days", "All horizons"]
    out += ["#### Forecast error: MAE in µg/m³ (lower is better)", "",
            md_table([[MODEL_LABELS[m]] + [f"{mae_all.loc[c, m]:.1f}" for c in cols] for m in models],
                     head), ""]

    sentences = []
    if "persistence" in models and "lgbm" in models:
        s1, s7, sa = skill.loc["1", "lgbm"], skill.loc["7", "lgbm"], skill.loc["all", "lgbm"]
        sentences.append(f"Against persistence, the LightGBM model's MAE is {_better(s1)} at 1 day, "
                         f"{_better(s7)} at 7 days and {_better(sa)} over all horizons.")
    if "sarima" in models and "lgbm" in models:
        sentences.append("Against SARIMA, its MAE over all horizons is "
                         f"{_better(1 - mae_all.loc['all', 'lgbm'] / mae_all.loc['all', 'sarima'])}.")
    if "lgbm_oracle" in models and "lgbm" in models:
        gain = 1 - mae_all.loc["all", "lgbm_oracle"] / mae_all.loc["all", "lgbm"]
        sentences.append(f"Handing the model the real weather of the forecast day would make its MAE "
                         f"{_better(gain)}; that is the most a better weather forecast could add.")
    if sentences:
        out += [" ".join(sentences), ""]
    out += ["![MAE by horizon](results/figures/mae_by_horizon.png)", ""]

    out += [f"#### Error on high-pollution days (observed PM2.5 of at least {config.THRESHOLDS['usg']} "
            "µg/m³)", "",
            md_table([[MODEL_LABELS[m]] + [f"{mae_high.loc[c, m]:.1f}" for c in cols] for m in models],
                     head), ""]

    iv = ctx["intervals"]

    def cov(level, variant):
        return float(iv[(iv["interval"] == level) & (iv["variant"] == variant)]["coverage"].iloc[0])

    out += ["#### Prediction intervals", "",
            f"Intervals come from quantile models and are then calibrated on the most recent "
            f"{config.CAL_DAYS} days before each block (split-conformal correction). The nominal "
            f"80% interval covered {cov('80%', 'calibrated (conformal)'):.0%} of observations "
            f"(raw: {cov('80%', 'raw quantile models'):.0%}); the nominal 95% interval covered "
            f"{cov('95%', 'calibrated (conformal)'):.0%} (raw: {cov('95%', 'raw quantile models'):.0%}).",
            "", "![Interval coverage](results/figures/interval_coverage.png)", ""]

    al = ctx["alerts"]
    rows = []
    for key in config.THRESHOLDS:
        sub = al[al["threshold"] == key]
        auc = sub["roc_auc"].iloc[0] if "roc_auc" in sub and sub["roc_auc"].notna().any() else np.nan
        for r in sub.itertuples():
            rows.append([key, f"{r.episode_rate:.1%}", "n/a" if pd.isna(auc) else f"{auc:.2f}",
                         r.alert_rule, _pct(r.precision), _pct(r.recall),
                         f"{_pct(r.new_episode_recall)} of {r.new_episodes}"])
    out += ["#### 48-hour episode alert", "",
            "An episode means a daily mean PM2.5 at or above the threshold on either of the next "
            "two days (`usg` = 35.5, `unhealthy` = 55.5 µg/m³). \"New episodes\" are those "
            "that start when today is still below the threshold, which a rule based on today's "
            "value can never catch.", "",
            md_table(rows, ["Threshold", "Episode rate", "ROC-AUC", "Alert rule", "Precision",
                            "Recall", "New episodes caught"]), "",
            "![Reliability of alert probabilities](results/figures/alert_reliability.png)", ""]

    drivers = ctx.get("drivers")
    if drivers is not None and len(drivers):
        top = (drivers[drivers["horizon"] == drivers["horizon"].min()]
               .sort_values("mean_abs_effect", ascending=False).head(3))
        names = ", ".join(f"{r.driver.lower()} ({r.mean_abs_effect:.1f} µg/m³)"
                          for r in top.itertuples())
        out += ["#### What drives the forecast", "",
                f"Averaged over test forecasts, the largest drivers of the 1-day forecast are {names}. "
                "Each effect is how much the forecast changes if that driver family is set to what is "
                "typical for the city and month. Effects are computed one family at a time, so they "
                "do not add up exactly.", "",
                "![Driver importance](results/figures/drivers.png)", ""]

    ex = ctx.get("exposure") or {}
    if ex.get("days"):
        out += ["#### Best time of day", "",
                f"On {ex['days']:,} unseen city-days, choosing the window the model recommends "
                f"(cleanest 3 hours between 06:00 and 22:00, using a typical daily shape learned "
                f"before the test period) lowered PM2.5 exposure by {ex['pooled_reduction']:.1%} "
                f"compared with a randomly chosen window. The median city saw "
                f"{ex['median_city_reduction']:.1%}, and the recommendation was better than average "
                f"in {ex['share_cities_better']:.0%} of cities. The air-quality model has 3-hourly "
                "steps, so the daily shape is coarse.", ""]
    return "\n".join(out).strip()
