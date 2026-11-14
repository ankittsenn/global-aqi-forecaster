"""Download hourly air quality and weather for every city and build the daily dataset.

    python build_data.py                      # all cities (about 15-25 minutes, resumable)
    python build_data.py --cities Delhi,Lima  # a few cities only
    python build_data.py --no-download        # rebuild the tables from files already downloaded

Raw hourly files are cached in data/raw/, so an interrupted run continues where it stopped.
Source: Open-Meteo (CC BY 4.0), CAMS global model data and ERA5 reanalysis. See the README.
"""
import argparse
import json
import sys
from datetime import date, timedelta

import pandas as pd

from airq import config, dataset, openmeteo
from airq.cities import catalogue


def raw_paths(slug: str):
    return config.RAW_DIR / f"{slug}_aq.csv.gz", config.RAW_DIR / f"{slug}_wx.csv.gz"


def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cities", help="comma-separated city names (default: all)")
    parser.add_argument("--start", default=config.HISTORY_START)
    parser.add_argument("--end", default=(date.today() - timedelta(days=6)).isoformat(),
                        help="last day (default: today minus 6, the weather archive's delay)")
    parser.add_argument("--refresh", action="store_true", help="download again even if cached")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cat = catalogue()
    if args.cities:
        wanted = {c.strip().lower() for c in args.cities.split(",")}
        cat = cat[cat["city"].str.lower().isin(wanted)]
        if cat.empty:
            sys.exit("None of those cities are in the catalogue (see airq/cities.py).")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, row in enumerate(cat.itertuples(), start=1):
        aq_path, wx_path = raw_paths(row.slug)
        if args.no_download or (aq_path.exists() and wx_path.exists() and not args.refresh):
            print(f"[{i}/{len(cat)}] {row.city}: using cached files")
            continue
        print(f"[{i}/{len(cat)}] {row.city}: downloading {args.start} to {args.end} ...", flush=True)
        try:
            aq, wx = openmeteo.fetch_history(row.lat, row.lon, args.start, args.end)
        except openmeteo.ApiError as err:
            print(f"    FAILED: {err}")
            failed.append(row.city)
            continue
        aq.to_csv(aq_path)
        wx.to_csv(wx_path)

    daily_parts, profiles, by_city = [], [], {}
    for row in cat.itertuples():
        aq_path, wx_path = raw_paths(row.slug)
        if not (aq_path.exists() and wx_path.exists()):
            continue
        aq, wx = read_raw(aq_path), read_raw(wx_path)
        daily = dataset.hourly_to_daily(aq, wx)
        daily = daily[daily["pm25"].notna() | daily["temp"].notna()]
        by_city[row.city] = daily
        part = daily.reset_index()
        part.insert(0, "city", row.city)
        daily_parts.append(part)
        profile = dataset.diurnal_profile(aq)
        profile.insert(0, "city", row.city)
        profiles.append(profile)

    if not daily_parts:
        sys.exit("No data available. Run without --no-download and check your internet connection.")

    all_daily = pd.concat(daily_parts, ignore_index=True)
    num = all_daily.select_dtypes("number").columns
    all_daily[num] = all_daily[num].round(3)
    all_daily.to_csv(config.PROCESSED_DIR / "daily.csv.gz", index=False)
    pd.concat(profiles, ignore_index=True).round(4).to_csv(
        config.PROCESSED_DIR / "diurnal.csv.gz", index=False)
    report = dataset.quality_report(by_city)
    (config.PROCESSED_DIR / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDone: {len(all_daily):,} city-days for {len(by_city)} cities "
          f"({all_daily['date'].min():%Y-%m-%d} to {all_daily['date'].max():%Y-%m-%d}).")
    worst = sorted(report.items(), key=lambda kv: -kv[1]["missing_pm25_pct"])[:3]
    print("Most missing PM2.5 days:", ", ".join(f"{c} {v['missing_pm25_pct']}%" for c, v in worst))
    if failed:
        print("Failed cities (run again to retry):", ", ".join(failed))


if __name__ == "__main__":
    main()
