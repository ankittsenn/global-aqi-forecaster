"""Global air-quality forecaster: dashboard.

    streamlit run app.py

Forecasts PM2.5 for 1 to 7 days ahead for 42 cities or any place you search for, with
uncertainty bands, a 48-hour episode alert, the reasons behind the forecast and a planner
for the cleanest hours of the day.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airq import aqi_scale, config, dataset, exposure, live, openmeteo
from airq.cities import catalogue
from airq.models import Bundle

AUTHOR_GITHUB = "https://github.com/ankittsenn"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/ankit-sen-aa90881b8"
REPO_URL = "https://github.com/ankittsenn/global-aqi-forecaster"

INK, MUTED, GRID = "#e6edf7", "#93a1b8", "#1f2a40"
ACCENT, WARM = "#38bdf8", "#f59e0b"
CARD, BG = "#111a2c", "#0a1120"

st.set_page_config(page_title="Global AQI Forecaster", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.card {{ background: {CARD}; border: 1px solid {GRID}; border-radius: 10px; padding: 18px 20px; }}
.kicker {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }}
.big {{ font-size: 3.4rem; font-weight: 700; line-height: 1.05; }}
.chip {{ display:inline-block; padding: 3px 11px; border-radius: 999px; font-weight: 600; font-size: .85rem; }}
.banner {{ border-radius: 10px; padding: 14px 18px; border: 1px solid {GRID}; border-left-width: 5px; margin: 6px 0 14px 0; }}
.muted {{ color: {MUTED}; }}
.small {{ font-size: .85rem; }}
div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def stretch(fn, *args, **kwargs):
    """Full-width elements across Streamlit versions (the keyword changed over time)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def chart(fig):
    return stretch(st.plotly_chart, fig, config={"displayModeBar": False})


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK, family="Inter, sans-serif"),
                      legend=dict(orientation="h", y=-0.18, font=dict(color=MUTED)),
                      hoverlabel=dict(bgcolor=CARD, font_color=INK))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


def chip(cat: dict) -> str:
    return (f"<span class='chip' style='background:{cat['color']};color:{cat['text_color']}'>"
            f"{cat['name']}</span>")


def hex_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


# --------------------------------------------------------------------------- cached data
@st.cache_resource(show_spinner=False)
def load_bundle():
    path = config.MODELS_DIR / "bundle.pkl"
    return Bundle.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_daily_table():
    path = config.PROCESSED_DIR / "daily.csv.gz"
    return dataset.load_daily(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_diurnal_table():
    path = config.PROCESSED_DIR / "diurnal.csv.gz"
    return dataset.load_diurnal(path) if path.exists() else None


@st.cache_data(show_spinner=False, ttl=900)
def load_snapshot():
    path = config.PROCESSED_DIR / "world_snapshot.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_results():
    out = {}
    for name in ("mae_by_horizon", "mae_high_pollution", "skill_vs_persistence", "interval_coverage",
                 "alert_metrics", "mae_by_region", "driver_importance", "reliability_usg",
                 "reliability_unhealthy"):
        path = config.RESULTS_DIR / f"{name}.csv"
        if path.exists():
            out[name] = pd.read_csv(path, index_col=0 if name.startswith("mae_by_h") or
                                    name.startswith("mae_high") or name.startswith("skill") else None)
    for name in ("summary", "exposure_planner"):
        path = config.RESULTS_DIR / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


@st.cache_data(show_spinner="Fetching the latest data ...", ttl=3600)
def fetch_live(lat: float, lon: float):
    daily, cams, meta = live.load_live(lat, lon)
    hourly = meta.pop("hourly_pm25", None)
    return daily, cams, hourly


@st.cache_data(show_spinner=False, ttl=1800)
def search_places(query: str):
    return openmeteo.geocode(query)


# --------------------------------------------------------------------------- forecast
def run_forecast(bundle, daily_all, cat, place: dict, offline: bool):
    """Return (result, hourly_series_or_None, notice)."""
    notice = None
    if place["kind"] == "city":
        row = cat[cat["city"] == place["city"]].iloc[0]
        if not offline:
            try:
                daily, cams, hourly = fetch_live(row.lat, row.lon)
                history = daily_all[daily_all["city"] == row.city].set_index("date")["pm25"]
                res = live.forecast_location(bundle, row.city, row.lat, row.lon, row.season_months,
                                             daily, cams, history, row.season_label, row.city, "live")
                return res, hourly, None
            except (openmeteo.ApiError, ValueError) as err:
                notice = f"Live data could not be loaded ({str(err)[:90]}). Showing the stored dataset instead."
        return live.forecast_catalogue_city(bundle, row, daily_all, live=False), None, notice
    daily, cams, hourly = fetch_live(place["lat"], place["lon"])
    res = live.forecast_location(bundle, place["label"], place["lat"], place["lon"], [], daily, cams,
                                 None, "", None, "live")
    return res, hourly, None


# --------------------------------------------------------------------------- sections
def alert_banner(res):
    a, p = res["alert"], res["episode"]
    colors = {"none": "#22c55e", "elevated": "#f59e0b", "severe": "#ef4444"}
    c = colors.get(a["severity"], MUTED)
    title = {"none": "No episode expected", "watch": "Watch", "likely": "Episode likely"}[a["level"]]
    st.markdown(
        f"<div class='banner' style='border-left-color:{c};background:{hex_rgba(c, 0.09)}'>"
        f"<b>48-hour alert: {title}.</b> {a['text']}<br>"
        f"<span class='muted small'>Chance of a daily mean at or above {config.THRESHOLDS['usg']} µg/m³ "
        f"(AQI 101) in the next two days: <b>{p['usg']:.0%}</b>. At or above "
        f"{config.THRESHOLDS['unhealthy']} µg/m³ (AQI 151): <b>{p['unhealthy']:.0%}</b>. "
        f"Probabilities are calibrated on past forecasts; see Track record.</span></div>",
        unsafe_allow_html=True)


def headline(res):
    cat = res["latest_category"]
    left, mid, right = st.columns([1.1, 1.2, 1.5])
    with left:
        st.markdown(f"<div class='card'><div class='kicker'>Latest complete day</div>"
                    f"<div class='big' style='color:{cat['color']}'>{res['latest_aqi']:.0f}</div>"
                    f"{chip(cat)}<div class='muted small' style='margin-top:8px'>"
                    f"{res['latest_pm25']:.1f} µg/m³ PM2.5 on {res['origin']:%d %b}</div></div>",
                    unsafe_allow_html=True)
    with mid:
        times = res["latest_pm25"] / config.WHO_PM25_24H
        lines = [f"<b>{times:.1f}x</b> the WHO 24-hour guideline ({config.WHO_PM25_24H:.0f} µg/m³)"]
        if res["percentile"] is not None:
            lines.append(f"Worse than <b>{res['percentile']:.0f}%</b> of days in {res['origin']:%B} "
                         "since 2022 for this city")
        tomorrow = res["forecast"].iloc[0]
        lines.append(f"Tomorrow: AQI <b>{tomorrow['aqi_median']:.0f}</b> "
                     f"({tomorrow['aqi_lo80']:.0f} to {tomorrow['aqi_hi80']:.0f}, 80% range)")
        st.markdown("<div class='card'><div class='kicker'>In context</div>"
                    + "".join(f"<div style='margin-top:8px'>{x}</div>" for x in lines) + "</div>",
                    unsafe_allow_html=True)
    with right:
        fc = res["forecast"]
        cells = "".join(
            f"<div style='flex:1;text-align:center'><div class='muted small'>{r.date:%a %d}</div>"
            f"<div style='font-weight:700;font-size:1.25rem;color:{aqi_scale.category(r.aqi_median)['color']}'>"
            f"{r.aqi_median:.0f}</div></div>" for r in fc.itertuples())
        st.markdown(f"<div class='card'><div class='kicker'>AQI outlook</div>"
                    f"<div style='display:flex;margin-top:10px'>{cells}</div></div>",
                    unsafe_allow_html=True)


def fan_chart(res):
    fc, obs = res["forecast"], res["observed"]
    top = float(max(fc["hi95"].max(), obs.max(), 60)) * 1.08
    fig = go.Figure()
    for i, cat in enumerate(aqi_scale.CATEGORIES):
        lo, hi = aqi_scale.PM25_BAND_EDGES[i], min(aqi_scale.PM25_BAND_EDGES[i + 1], top)
        if lo < top:
            fig.add_hrect(y0=lo, y1=hi, fillcolor=hex_rgba(cat[3], 0.10), line_width=0, layer="below")
    x0 = [res["origin"]] + list(fc["date"])
    for lo, hi, a, label in (("lo95", "hi95", 0.16, "95% range"), ("lo80", "hi80", 0.30, "80% range")):
        y_hi = [float(res["latest_pm25"])] + list(fc[hi])
        y_lo = [float(res["latest_pm25"])] + list(fc[lo])
        fig.add_trace(go.Scatter(x=x0 + x0[::-1], y=y_hi + y_lo[::-1], fill="toself", mode="none",
                                 fillcolor=hex_rgba(ACCENT, a), name=label, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=obs.index, y=obs.values, mode="lines", name="Observed (model analysis)",
                             line=dict(color=MUTED, width=2),
                             hovertemplate="%{x|%d %b}: %{y:.1f} µg/m³<extra></extra>"))
    fig.add_trace(go.Scatter(x=x0, y=[float(res["latest_pm25"])] + list(fc["median"]), mode="lines+markers",
                             name="Forecast (median)", line=dict(color=ACCENT, width=3),
                             marker=dict(size=7, line=dict(width=2, color=BG)),
                             customdata=np.c_[[res["latest_aqi"]] + list(fc["aqi_median"])],
                             hovertemplate="%{x|%d %b}: %{y:.1f} µg/m³ (AQI %{customdata[0]:.0f})<extra></extra>"))
    cams = res.get("cams")
    if cams is not None and len(cams):
        fig.add_trace(go.Scatter(x=cams.index, y=cams.values, mode="lines", name="CAMS physical model",
                                 line=dict(color=WARM, width=2, dash="dash"),
                                 hovertemplate="%{x|%d %b}: %{y:.1f} µg/m³<extra></extra>"))
    fig.add_vline(x=res["origin"], line_dash="dot", line_color=GRID)
    fig.update_yaxes(title_text="PM2.5 daily mean (µg/m³)", range=[0, top])
    chart(style(fig, 420))


def drivers_section(bundle, res):
    horizon = st.slider("Explain the forecast for", 1, 7, 1, format="%d day(s) ahead")
    effects, lines = live.drivers_for(bundle, res, horizon)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        eff = effects.sort_values("effect")
        fig = go.Figure(go.Bar(x=eff["effect"], y=eff["driver"], orientation="h",
                               marker_color=[WARM if v > 0 else ACCENT for v in eff["effect"]],
                               hovertemplate="%{y}: %{x:+.1f} µg/m³<extra></extra>"))
        fig.update_xaxes(title_text="Change in the forecast against a typical day (µg/m³)")
        chart(style(fig, 360))
    with c2:
        st.markdown("<div class='kicker'>In words</div>", unsafe_allow_html=True)
        for line in lines or ["No single factor moves this forecast by more than 1 µg/m³."]:
            st.markdown(f"- {line}")
        if res["season_label"]:
            st.markdown(f"<span class='muted small'>Seasonal driver for this city: "
                        f"{res['season_label']}.</span>", unsafe_allow_html=True)
    st.caption("Each bar replaces one driver family with what is typical for the city and month and "
               "measures how the forecast changes. Bars are computed one at a time and do not add up "
               "exactly.")


def precautions(res, group):
    fc = res["forecast"]
    worst = max([res["latest_aqi"], *list(fc["aqi_median"].iloc[:2])])
    if res["alert"]["severity"] == "severe":
        worst = max(worst, 151)
    elif res["alert"]["severity"] == "elevated":
        worst = max(worst, 101)
    cat = aqi_scale.category(worst)
    st.markdown(f"Guidance for the next two days, based on {chip(cat)} for **{group.lower()}**",
                unsafe_allow_html=True)
    for tip in aqi_scale.advice_for(cat["name"], group):
        st.markdown(f"- {tip}")
    st.caption(aqi_scale.DISCLAIMER)


def forecast_table(res):
    fc = res["forecast"]
    table = pd.DataFrame({
        "Day": fc["date"].dt.strftime("%a %d %b"),
        "AQI": fc["aqi_median"].round().astype(int),
        "AQI, 80% range": [f"{a:.0f} to {b:.0f}" for a, b in zip(fc["aqi_lo80"], fc["aqi_hi80"])],
        "PM2.5 (µg/m³)": fc["median"].round(1),
        "95% range (µg/m³)": [f"{a:.0f} to {b:.0f}" for a, b in zip(fc["lo95"], fc["hi95"])],
        "Category": fc["category"]})
    stretch(st.dataframe, table, hide_index=True)


# --------------------------------------------------------------------------- tabs
def tab_forecast(bundle, res, notice, group):
    if notice:
        st.warning(notice)
    st.markdown(f"### {res['name']}")
    if not res["known_city"]:
        st.caption("This place is not one of the 42 listed cities, so the model has no local seasonal-driver "
                   "flag or typical values for it; it relies on the last three months of this place's own data.")
    st.caption(f"Forecast made from {res['origin']:%A %d %B %Y} (last complete local day). "
               f"Data: {res['source']}.")
    alert_banner(res)
    headline(res)
    st.markdown("#### Forecast with uncertainty")
    fan_chart(res)
    st.caption("Shaded bands show where 80% and 95% of outcomes are expected to fall. Background colours "
               "are the US EPA categories. Dashed line: the CAMS atmospheric model's own forecast, for "
               "comparison.")
    forecast_table(res)
    st.markdown("#### What is driving this forecast")
    drivers_section(bundle, res)
    st.markdown("#### What to do")
    precautions(res, group)


def tab_plan(res, hourly, diurnal, group):
    st.markdown("### Plan your day")
    month = int(res["origin"].month)
    if res["known_city"] and diurnal is not None:
        sub = diurnal[diurnal["city"] == res["known_city"]]
        profile = exposure.profile_vector(sub, month)
    elif hourly is not None:
        profile = exposure.profile_from_hourly(hourly)
    else:
        profile = np.ones(24)
    if np.allclose(profile, 1):
        st.info("There is not enough hourly data to estimate a daily pattern for this place.")
        return
    day = st.radio("Day", list(res["forecast"]["h"]), format_func=lambda h: f"{res['forecast'].iloc[h - 1]['date']:%a %d %b}",
                   horizontal=True)
    row = res["forecast"].iloc[day - 1]
    win = exposure.best_window(profile)
    plan = exposure.hourly_plan(profile, float(row["median"]))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"<div class='card'><div class='kicker'>Cleanest 3 hours (06:00 to 22:00)</div>"
                    f"<div class='big'>{win['start']:02d}:00 to {win['end']:02d}:00</div>"
                    f"<div class='muted' style='margin-top:8px'>About <b>{win['reduction']:.0%}</b> lower than "
                    f"an average 3-hour window that day. Daily mean forecast: {row['median']:.0f} µg/m³.</div></div>",
                    unsafe_allow_html=True)
        planner = (load_results().get("exposure_planner") or {})
        if planner.get("days"):
            st.caption(f"Checked on {planner['days']:,} unseen city-days: following the recommendation "
                       f"lowered exposure by {planner['pooled_reduction']:.0%} on average.")
    with c2:
        hours = list(range(24))
        colors = [ACCENT if win["start"] <= h < win["end"] else GRID for h in hours]
        fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hours], y=plan, marker_color=colors,
                               hovertemplate="%{x}: about %{y:.0f} µg/m³<extra></extra>"))
        fig.update_yaxes(title_text="Expected PM2.5 (µg/m³)")
        chart(style(fig, 300))
    st.caption("Typical shape of the day for this place and season, scaled to the forecast daily mean. The "
               "air-quality model works in 3-hour steps, so this is a coarse guide, not a measurement.")
    cat = aqi_scale.category(float(row["aqi_median"]))
    st.markdown(f"Expected category: {chip(cat)}", unsafe_allow_html=True)
    for tip in aqi_scale.advice_for(cat["name"], group):
        st.markdown(f"- {tip}")
    if cat["name"] not in ("Good", "Moderate"):
        st.markdown(f"- If you must exercise outdoors, do it in the window above "
                    f"({win['start']:02d}:00 to {win['end']:02d}:00) and keep it light.")
    st.caption(aqi_scale.DISCLAIMER)


def tab_world(snapshot):
    st.markdown("### World view")
    if snapshot is None:
        st.info("No world snapshot yet. Run `python update_snapshot.py`.")
        return
    st.caption(f"Model-based forecasts for {len(snapshot)} cities, updated {snapshot['updated_utc'].iloc[0]} UTC.")
    options = {"Latest complete day": "latest_aqi", **{f"Day +{h}": f"aqi_d{h}" for h in range(1, 8)}}
    pick = st.select_slider("Show", list(options), value="Day +1")
    col = options[pick]
    regions = sorted(snapshot["region"].unique())
    chosen = st.multiselect("Regions", regions, default=regions)
    data = snapshot[snapshot["region"].isin(chosen)].copy()
    if data.empty:
        st.info("Select at least one region.")
        return
    data["category"] = [aqi_scale.category(a)["name"] for a in data[col]]
    fig = go.Figure()
    for cat in aqi_scale.CATEGORIES:
        part = data[data["category"] == cat[0]]
        if len(part):
            fig.add_trace(go.Scattergeo(lat=part["lat"], lon=part["lon"], name=cat[0], mode="markers",
                                        marker=dict(size=13, color=cat[3], line=dict(width=1.5, color=BG)),
                                        customdata=np.c_[part["city"], part["country"], part[col], part["alert_level"]],
                                        hovertemplate="<b>%{customdata[0]}</b>, %{customdata[1]}<br>AQI %{customdata[2]}"
                                                      "<br>48 h alert: %{customdata[3]}<extra></extra>"))
    fig.update_geos(projection_type="natural earth", showland=True, landcolor="#16223a",
                    showocean=True, oceancolor=BG, showcountries=True, countrycolor=GRID,
                    coastlinecolor=GRID, bgcolor="rgba(0,0,0,0)")
    chart(style(fig, 460))
    c1, c2 = st.columns(2)
    show = data.sort_values(col, ascending=False)
    cols = {"city": "City", "country": "Country", col: "AQI", "alert_level": "48 h alert"}
    with c1:
        st.markdown("**Most polluted**")
        stretch(st.dataframe, show.head(10)[list(cols)].rename(columns=cols), hide_index=True)
    with c2:
        st.markdown("**Cleanest**")
        stretch(st.dataframe, show.tail(10)[::-1][list(cols)].rename(columns=cols), hide_index=True)
    flagged = snapshot[snapshot["alert_level"] != "none"].sort_values("p_unhealthy", ascending=False)
    st.markdown(f"**Cities with a 48-hour alert ({len(flagged)})**")
    if len(flagged):
        stretch(st.dataframe, flagged[["city", "country", "alert_level", "p_usg", "p_unhealthy"]].rename(
            columns={"city": "City", "country": "Country", "alert_level": "Level",
                     "p_usg": "P(AQI > 100)", "p_unhealthy": "P(AQI > 150)"}), hide_index=True)
    st.markdown("**Compare cities**")
    names = st.multiselect("Cities", list(snapshot["city"]), default=list(show["city"].head(3)), max_selections=6)
    if names:
        fig = go.Figure()
        days = [f"Day +{h}" for h in range(1, 8)]
        for name in names:
            r = snapshot[snapshot["city"] == name].iloc[0]
            fig.add_trace(go.Scatter(x=days, y=[r[f"aqi_d{h}"] for h in range(1, 8)], mode="lines+markers",
                                     name=name, line=dict(width=2.5)))
        fig.update_yaxes(title_text="Forecast AQI")
        chart(style(fig, 340))


MODEL_NAMES = {"persistence": "Persistence", "snaive": "Seasonal naive", "climatology": "Monthly climatology",
               "sarima": "SARIMA", "lgbm": "LightGBM", "lgbm_oracle": "LightGBM + perfect weather (upper bound)"}


def tab_record(results):
    st.markdown("### Track record")
    if "mae_by_horizon" not in results:
        st.info("No backtest results yet. Run `python evaluate.py`.")
        return
    info = results.get("summary", {})
    st.caption(f"Rolling-origin backtest, {info.get('n_folds', '?')} blocks of {info.get('block_days', '?')} days "
               f"({info.get('test_start', '?')} to {info.get('test_end', '?')}), {info.get('n_cities', '?')} cities. "
               "Models are always trained on days before the block they forecast. No random splits.")
    mae = results["mae_by_horizon"]
    models = [m for m in MODEL_NAMES if m in mae.columns]
    horizons = [h for h in mae.index.astype(str) if h != "all"]
    fig = go.Figure()
    palette = {"persistence": "#94a3b8", "snaive": "#64748b", "climatology": WARM, "sarima": "#a78bfa",
               "lgbm": ACCENT, "lgbm_oracle": ACCENT}
    for m in models:
        fig.add_trace(go.Scatter(x=[int(h) for h in horizons], y=mae.loc[[h for h in mae.index if str(h) != "all"], m],
                                 mode="lines+markers", name=MODEL_NAMES[m],
                                 line=dict(color=palette[m], width=3.5 if m == "lgbm" else 2,
                                           dash="dot" if m == "lgbm_oracle" else "solid")))
    fig.update_xaxes(title_text="Forecast horizon (days ahead)", dtick=1)
    fig.update_yaxes(title_text="Mean absolute error (µg/m³)")
    chart(style(fig, 380))
    shown = mae[[m for m in models]].rename(columns=MODEL_NAMES).round(1)
    shown.index = shown.index.astype(str)
    stretch(st.dataframe, shown.loc[[i for i in ("1", "2", "3", "5", "7", "all") if i in shown.index]])
    st.caption("MAE in µg/m³, lower is better. All models are scored on the same forecasts.")
    if "mae_high_pollution" in results:
        st.markdown("**Error on high-pollution days only** (observed PM2.5 of at least 35.5 µg/m³)")
        high = results["mae_high_pollution"][models].rename(columns=MODEL_NAMES).round(1)
        high.index = high.index.astype(str)
        stretch(st.dataframe, high.loc[[i for i in ("1", "3", "7", "all") if i in high.index]])
    c1, c2 = st.columns(2)
    with c1:
        if "interval_coverage" in results:
            st.markdown("**Do the intervals cover what they promise?**")
            iv = results["interval_coverage"].copy()
            iv["coverage"] = (iv["coverage"] * 100).round(1).astype(str) + "%"
            stretch(st.dataframe, iv, hide_index=True)
    with c2:
        if "alert_metrics" in results:
            st.markdown("**48-hour alert**")
            al = results["alert_metrics"]
            keep = [c for c in ("threshold", "alert_rule", "precision", "recall", "new_episode_recall") if c in al]
            stretch(st.dataframe, al[keep].round(2), hide_index=True)
    rel = [(k, l) for k, l in (("reliability_usg", "Episode of AQI above 100"),
                               ("reliability_unhealthy", "Episode of AQI above 150")) if k in results]
    if rel:
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                   line=dict(color=MUTED, dash="dash")))
        for (k, l), color in zip(rel, (ACCENT, WARM)):
            t = results[k]
            fig.add_trace(go.Scatter(x=t["mean_predicted"], y=t["observed_rate"], mode="lines+markers",
                                     name=l, line=dict(color=color, width=2.5)))
        fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
        fig.update_yaxes(title_text="Observed frequency", range=[0, 1])
        chart(style(fig, 340))
    if "mae_by_region" in results:
        st.markdown("**Error by region** (LightGBM against persistence, µg/m³)")
        stretch(st.dataframe, results["mae_by_region"].round(1), hide_index=True)


def tab_about():
    st.markdown(f"""
### About this project

**Question.** Can fine-particle pollution (PM2.5) be forecast up to a week ahead for cities around the world,
with honest uncertainty and an early warning for pollution episodes?

**Data.** Hourly PM2.5 and other pollutants come from the CAMS global atmospheric model (Copernicus), and
weather from ERA5 reanalysis, both served by [Open-Meteo](https://open-meteo.com) (CC BY 4.0), from September 2022.
These are **model outputs on a grid of about 45 km, not readings from a street-level monitor**. A city
centre can be far dirtier than its grid cell. The dashboard describes the regional air, and it is not a
substitute for local monitoring.

**Models.** One pooled model learns from all cities. Compared in the backtest: persistence, same weekday last
week, monthly climatology, SARIMA, and LightGBM. Each horizon (1 to 7 days) has its own model. Intervals come
from quantile regression with split-conformal calibration. The 48-hour alert uses calibrated classifiers.
The weather of the forecast day is not an input, because a forecaster does not know it yet.

**Evaluation.** Rolling-origin backtest with expanding training windows, always trained on earlier days than
it forecasts. The Track record tab shows error by horizon, interval coverage and alert quality.

**Beyond a typical air-quality app.**
- A calibrated probability of a pollution episode in the next 48 hours, scored on episodes that begin after a clean day.
- Prediction intervals whose coverage is measured on unseen data.
- Counterfactual explanations: how much wind, temperature, season or the burning season moves each forecast.
- A cleanest-three-hours planner, checked on unseen days.
- The physical CAMS forecast as a reference line next to the statistical one.
- Percentile context: how unusual today is for this city and month.

**Limits.** Short forecasts lean on recent pollution; beyond three days they approach the seasonal average.
Sudden events (a fire, a dust storm) are missed until they show up in the data. Seasonal-driver months per
city are assumptions drawn from the literature, not fitted. Open-Meteo's free service is for non-commercial use.

**Attribution.** Contains modified Copernicus Atmosphere Monitoring Service information. Generated using
Copernicus Climate Change Service information. Weather and air-quality data by Open-Meteo.com (CC BY 4.0).
AQI follows the US EPA 2024 PM2.5 breakpoints; the 15 µg/m³ line is the WHO 2021 daily guideline.

**Health note.** {aqi_scale.DISCLAIMER}

Built by Ankit Kumar. [GitHub]({AUTHOR_GITHUB}) · [LinkedIn]({AUTHOR_LINKEDIN}) · [Source code]({REPO_URL})
""")


# --------------------------------------------------------------------------- main
def main():
    bundle, daily_all, diurnal = load_bundle(), load_daily_table(), load_diurnal_table()
    st.title("Global AQI Forecaster")
    st.markdown("<span class='muted'>Seven-day PM2.5 forecasts with uncertainty, a 48-hour episode alert "
                "and the reasons behind them.</span>", unsafe_allow_html=True)
    if bundle is None or daily_all is None:
        st.error("Models or data are missing. Run `python build_data.py` and `python train.py`, "
                 "then restart the app.")
        st.stop()
    cat = catalogue()
    cat = cat[cat["city"].isin(daily_all["city"].unique())].reset_index(drop=True)

    with st.sidebar:
        st.markdown("### Location")
        mode = st.radio("Choose by", ["City list", "Search any place"], horizontal=True, label_visibility="collapsed")
        place = None
        if mode == "City list":
            region = st.selectbox("Region", sorted(cat["region"].unique()))
            names = cat[cat["region"] == region]["city"].tolist()
            default = names.index("Delhi") if "Delhi" in names else 0
            place = {"kind": "city", "city": st.selectbox("City", names, index=default)}
        else:
            query = st.text_input("Town or city", placeholder="e.g. Pune, Quito, Reykjavik")
            if query.strip():
                try:
                    found = search_places(query.strip())
                except openmeteo.ApiError as err:
                    found = []
                    st.error(f"Search failed: {str(err)[:80]}")
                if found:
                    label = st.selectbox("Matches", range(len(found)), format_func=lambda i: found[i]["label"])
                    place = {"kind": "custom", **found[label]}
                elif query.strip():
                    st.caption("No place found.")
        st.markdown("### Who is this for")
        group = st.selectbox("Guidance for", aqi_scale.GROUPS, label_visibility="collapsed")
        offline = st.toggle("Use stored data only", value=False,
                            help="Skips live requests and forecasts from the saved dataset.")
        st.markdown("---")
        st.caption(f"Models trained on data up to {bundle.meta.get('trained_until', '?')[:10]}.")
        st.caption(f"[GitHub]({AUTHOR_GITHUB}) · [LinkedIn]({AUTHOR_LINKEDIN})")

    tabs = st.tabs(["Forecast", "Plan your day", "World", "Track record", "About"])
    res = hourly = notice = None
    if place is not None:
        try:
            if place["kind"] == "custom" and offline:
                st.warning("Stored data covers the 42 listed cities only. Turn off \"Use stored data only\" "
                           "to forecast a searched place.")
            else:
                res, hourly, notice = run_forecast(bundle, daily_all, cat, place, offline)
        except (openmeteo.ApiError, ValueError) as err:
            st.error(f"Could not forecast this place: {err}")
    with tabs[0]:
        if res is not None:
            tab_forecast(bundle, res, notice, group)
        elif place is None:
            st.info("Search for a place in the sidebar.")
    with tabs[1]:
        if res is not None:
            tab_plan(res, hourly, diurnal, group)
    with tabs[2]:
        tab_world(load_snapshot())
    with tabs[3]:
        tab_record(load_results())
    with tabs[4]:
        tab_about()


main()
