"""Fit the final models on all data and save them to models/bundle.pkl.

    python train.py          # about 5-10 minutes with LightGBM

The dashboard and the API load this file. Run `python evaluate.py` first if you want the
honest backtest numbers; this script only produces the deployable models.
"""
import json
import time

from airq import config, dataset, features as ft, training
from airq.cities import catalogue, season_months


def main() -> None:
    path = config.PROCESSED_DIR / "daily.csv.gz"
    if not path.exists():
        raise SystemExit("data/processed/daily.csv.gz not found. Run `python build_data.py` first.")
    started = time.time()
    cat = catalogue()
    panel = ft.build_panel(dataset.load_daily(path), cat)
    print(f"Training on {len(panel):,} city-days from {panel['city'].nunique()} cities "
          f"(up to {panel['date'].max():%Y-%m-%d})")
    bundle = training.train_bundle(panel, season_months())

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle.save(config.MODELS_DIR / "bundle.pkl")
    (config.MODELS_DIR / "meta.json").write_text(json.dumps(bundle.meta, indent=2), encoding="utf-8")
    size = (config.MODELS_DIR / "bundle.pkl").stat().st_size / 1e6
    print(f"Saved models/bundle.pkl ({size:.1f} MB) in {time.time() - started:.0f} s. "
          f"Backend: {bundle.meta['backend']}.")


if __name__ == "__main__":
    main()
