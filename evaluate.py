"""Backtest every model and write results/ (tables, figures) and the README results section.

    python evaluate.py                 # full run: 4 blocks of 91 days, all cities (30-60 minutes)
    python evaluate.py --fast          # quick check with small models (a few minutes)
    python evaluate.py --skip-sarima   # leave out the slow SARIMA model

Only time-based splits are used. For each block of forecast origins every model is trained
on days strictly before the block, then forecasts 1 to 7 days ahead from every origin day.
"""
import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from airq import backtest, config, dataset, explain, exposure, features as ft, report, training
from airq.cities import catalogue, season_months
from airq.readme import replace_block


def load_hourly(cat: pd.DataFrame) -> dict:
    hourly = {}
    for row in cat.itertuples():
        path = config.RAW_DIR / f"{row.slug}_aq.csv.gz"
        if path.exists():
            frame = pd.read_csv(path, index_col="time", parse_dates=True, usecols=["time", "pm2_5"])
            hourly[row.city] = frame["pm2_5"]
    return hourly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="small models, for a quick check")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--block-days", type=int, default=91)
    parser.add_argument("--skip-sarima", action="store_true")
    parser.add_argument("--no-oracle", action="store_true", help="skip the perfect-weather upper bound")
    parser.add_argument("--no-readme", action="store_true", help="do not touch README.md")
    args = parser.parse_args()
    if args.fast:
        os.environ["AIRQ_FAST"] = "1"

    started = time.time()
    path = config.PROCESSED_DIR / "daily.csv.gz"
    if not path.exists():
        sys.exit("data/processed/daily.csv.gz not found. Run `python build_data.py` first.")
    cat = catalogue()
    months = season_months()
    panel = ft.build_panel(dataset.load_daily(path), cat)
    folds = backtest.make_folds(panel, args.folds, args.block_days)
    span_days = (folds[0].train_end - panel["date"].min()).days
    if span_days < 365:
        sys.exit(f"Only {span_days} days of history before the first test block; "
                 "need at least 365. Use fewer folds (--folds) or download more data.")
    print(f"{panel['city'].nunique()} cities, {len(panel):,} city-days, "
          f"{len(folds)} folds of {args.block_days} days "
          f"({folds[0].test_start:%Y-%m-%d} to {folds[-1].test_end:%Y-%m-%d})")

    use_sarima = not args.skip_sarima
    if use_sarima:
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            print("statsmodels is not installed: SARIMA is skipped.")
            use_sarima = False

    frames = backtest.build_frames(panel, months, oracle=not args.no_oracle)
    episode_frame = training.episode_frame(panel, months)

    forecasts, alerts, sarima_parts, last = [], [], [], None
    for fold in folds:
        print(f"Fold {fold.index + 1}/{len(folds)}: test {fold.test_start:%Y-%m-%d} to "
              f"{fold.test_end:%Y-%m-%d}")
        out = backtest.run_fold(panel, frames, episode_frame, fold, oracle=not args.no_oracle)
        forecasts.append(out["forecasts"])
        alerts.append(out["alerts"])
        last = (fold, out)
        if use_sarima:
            sarima_parts.append(backtest.sarima_fold(panel, fold))

    fc = pd.concat(forecasts, ignore_index=True)
    if sarima_parts:
        fc = fc.merge(pd.concat(sarima_parts, ignore_index=True), on=["city", "origin", "h"], how="left")
    al = pd.concat(alerts, ignore_index=True)
    models = [m for m in backtest.MODEL_LABELS if m in fc and fc[m].notna().any()]

    results = config.RESULTS_DIR
    (results / "figures").mkdir(parents=True, exist_ok=True)
    fc.to_csv(results / "_predictions.csv.gz", index=False)        # large; not committed

    mae_all = backtest.mae_table(fc, models)
    mae_high = backtest.mae_table(fc, models, minimum_y=config.THRESHOLDS["usg"])
    skill = backtest.skill_table(mae_all, [m for m in models if m != "persistence"])
    intervals = backtest.interval_table(fc)
    alert_metrics = backtest.alert_table(al)
    reliability = {k: backtest.reliability_table(al, k) for k in config.THRESHOLDS}
    regions = backtest.region_table(fc, cat, models)

    # drivers: last fold's models on a sample of its test rows
    fold, out = last
    train_part = panel[panel["date"] <= fold.train_end]
    typical = ft.typical_table(train_part)
    rng = np.random.default_rng(7)
    importance = []
    for h in (1, 3, 7):
        frame = frames[h]
        block = frame[(frame["date"] >= fold.test_start) & (frame["date"] <= fold.test_end)
                      & frame["target"].notna() & frame["pm_t"].notna()]
        sample = block.iloc[rng.choice(len(block), size=min(3000, len(block)), replace=False)]
        typ = explain.typical_frame(sample, typical, ft.FEATURES)
        table = explain.driver_importance(out["models"][h], sample[ft.FEATURES], typ)
        table.insert(0, "horizon", h)
        importance.append(table)
    importance = pd.concat(importance, ignore_index=True)

    # best-time-of-day planner
    hourly = load_hourly(cat)
    planner = {"days": 0}
    if hourly:
        planner = exposure.evaluate_windows(hourly, folds[0].train_end, folds[0].test_start,
                                            folds[-1].test_end)
    else:
        print("Hourly raw files not found in data/raw: the time-of-day evaluation is skipped.")

    # write tables
    mae_all.round(3).to_csv(results / "mae_by_horizon.csv")
    mae_high.round(3).to_csv(results / "mae_high_pollution.csv")
    skill.round(4).to_csv(results / "skill_vs_persistence.csv")
    intervals.round(4).to_csv(results / "interval_coverage.csv", index=False)
    alert_metrics.round(4).to_csv(results / "alert_metrics.csv", index=False)
    for key, table in reliability.items():
        table.round(4).to_csv(results / f"reliability_{key}.csv", index=False)
    regions.round(3).to_csv(results / "mae_by_region.csv", index=False)
    importance.round(4).to_csv(results / "driver_importance.csv", index=False)
    planner_out = {k: v for k, v in planner.items() if k != "per_city"}
    (results / "exposure_planner.json").write_text(json.dumps(planner, indent=2), encoding="utf-8")

    # figures
    report.fig_mae(mae_all, models, results / "figures" / "mae_by_horizon.png")
    report.fig_intervals(intervals, results / "figures" / "interval_coverage.png")
    report.fig_reliability(reliability, results / "figures" / "alert_reliability.png")
    report.fig_drivers(importance, results / "figures" / "drivers.png")

    info = {"generated": date.today().isoformat(), "data_end": f"{panel['date'].max():%Y-%m-%d}",
            "n_folds": len(folds), "block_days": args.block_days,
            "test_start": f"{folds[0].test_start:%Y-%m-%d}", "test_end": f"{folds[-1].test_end:%Y-%m-%d}",
            "n_cities": int(panel["city"].nunique()),
            "n_h1": int(len(backtest.common_sample(fc[fc["h"] == 1], models))),
            "models": models, "backend": os.environ.get("AIRQ_BACKEND", "auto"),
            "fast": bool(args.fast)}
    (results / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    markdown = report.results_markdown({"info": info, "models": models, "mae_all": mae_all,
                                        "mae_high": mae_high, "skill": skill, "intervals": intervals,
                                        "alerts": alert_metrics, "drivers": importance,
                                        "exposure": planner_out})
    (results / "results.md").write_text(markdown, encoding="utf-8")
    readme = config.ROOT / "README.md"
    if readme.exists() and not args.no_readme:
        readme.write_text(replace_block(readme.read_text(encoding="utf-8"), "RESULTS", markdown),
                          encoding="utf-8")
        print("README.md results section updated.")

    print(f"\nDone in {(time.time() - started) / 60:.1f} minutes. Models compared: {', '.join(models)}")
    print(mae_all.round(1).to_string())
    if args.fast:
        print("\nNOTE: --fast uses small models; run without it for the numbers you publish.")


if __name__ == "__main__":
    main()
