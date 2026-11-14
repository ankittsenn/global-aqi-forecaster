# Step-by-step guide (Windows)

Everything below is typed into **Command Prompt** or **PowerShell**. Copy one block at a time and wait for
it to finish. Replace nothing unless a line says so.

## 0. What you need

1. **Python 3.11 or 3.12** from python.org. On the first installer screen tick **"Add python.exe to PATH"**.
   Check: `python --version`
2. **Git** from git-scm.com (you already used it for the Olist project).
3. A GitHub account (`ankittsenn`).

## 1. Open the project

Unzip the folder. Open the **inner** folder that contains `app.py` (the same nested-folder trap as before),
click the address bar in File Explorer, type `cmd` and press Enter. Check:

```
dir app.py
```

## 2. Install the packages

```
pip install -r requirements-dev.txt
```

It takes a few minutes. If `pip` is not found, use `python -m pip install -r requirements-dev.txt`.

## 3. Download the data (15 to 25 minutes)

```
python build_data.py
```

Internet is needed. It prints one line per city and **can be stopped and restarted**: finished cities are
kept. If some cities fail with a rate-limit message, wait 10 minutes and run the same command again.
Result: `data/processed/daily.csv.gz`.

## 4. Run the tests (1 minute)

```
python -m pytest -q
```

You should see "passed" and no "failed".

## 5. Backtest: compare the models (30 to 60 minutes)

```
python evaluate.py
```

This is the heart of the project: it compares persistence, seasonal naive, climatology, SARIMA and
LightGBM, always on future days it never trained on, and writes `results/` plus the Results section of
`README.md`. For a quick trial first, `python evaluate.py --fast` (a few minutes), then run the full one for
the numbers you publish. If SARIMA takes too long: `python evaluate.py --skip-sarima`.

Open `README.md` afterwards and read the Results section. **Do not edit numbers by hand.** If a number
surprises you, that is something to understand and mention honestly, not to fix.

## 6. Train the final models and build the world table

```
python train.py
python update_snapshot.py
```

`train.py` takes 5 to 10 minutes. `update_snapshot.py` takes about 5 minutes (it calls the live service for
each city and falls back to stored data if one fails).

## 7. Look at the dashboard

```
streamlit run app.py
```

It opens at http://localhost:8501. Click through all five tabs, try a city from each region, and search for
your own town in the sidebar. Press `Ctrl+C` in the window to stop.

Optional API: `uvicorn api.main:app` then open http://127.0.0.1:8000/docs.

## 8. Screenshots (optional but good)

Take two screenshots of the dashboard (Forecast tab, World tab). Make a folder `docs` and save them as
`docs/forecast.png` and `docs/world.png` (exact names, lower case). Add under the first paragraph of
`README.md`:

```
![Forecast tab](docs/forecast.png)
![World tab](docs/world.png)
```

## 9. Put it on GitHub

1. On github.com click **New repository**. Name it exactly `global-aqi-forecaster`, Public, **do not** add a
   README, .gitignore or licence.
2. Back in the project folder:

```
git init
git config user.name "Ankit Kumar"
git config user.email "ankit2472003@gmail.com"
git add .
git status
```

Read the `git status` list: you should see `models/bundle.pkl`, `data/processed/...`, `results/...` and
no `data/raw`. Then:

```
git commit -m "Global AQI forecaster"
git branch -M main
git remote add origin https://github.com/ankittsenn/global-aqi-forecaster.git
git push -u origin main
```

If `bundle.pkl` is above 100 MB GitHub will refuse the push. It is gzip-compressed and should be far smaller;
if it is not, tell me the size.

## 10. Deploy for free (Streamlit Community Cloud)

1. Go to share.streamlit.io and sign in with GitHub.
2. **Create app** > pick `ankittsenn/global-aqi-forecaster`, branch `main`, main file `app.py` > Deploy.
3. Wait a few minutes. Copy the address, then:

```
python tools/add_live_link.py https://YOUR-ADDRESS.streamlit.app
git add README.md
git commit -m "Add live link"
git push
```

(Use the real address you were given in place of `YOUR-ADDRESS`.)

The world view refreshes daily through `.github/workflows/refresh-snapshot.yml`. Open the **Actions** tab on
GitHub once and enable workflows if asked. Streamlit Cloud shares an address with many apps, so if the live
service rate-limits it the dashboard shows the stored dataset with a notice; that is intended.

## 11. How to talk about it (CV and interviews)

CV bullet, one possibility once you have your numbers:

> Built a global PM2.5 forecaster (42 cities, 1 to 7 days ahead) with rolling-origin backtesting, conformal
> prediction intervals and a calibrated 48-hour episode alert; Streamlit dashboard and FastAPI service.
> LightGBM cut MAE by X% against persistence (replace X with the number from your own Results table).

Be ready to explain, in your own words:

- **Why time-based splits.** Random splits let the model see tomorrow while predicting yesterday.
- **Why model data and not stations.** What CAMS and ERA5 are, what the 45 km grid means for accuracy.
- **What the baselines show.** If a model cannot beat persistence at 1 day, it is not adding much.
- **What "calibrated" means.** When it says 60%, about 60% of such cases happen. Point at the reliability plot.
- **What conformal calibration does.** It widens or narrows the raw intervals until measured coverage matches.
- **What the "perfect weather" row is.** An upper bound; it shows weather-forecast error is not the main limit.
- **Limits.** Short history, assumed seasonal months, no sudden-event detection.

## 12. If something goes wrong

| Message | Fix |
|---|---|
| `No module named ...` | `pip install -r requirements-dev.txt` again |
| `can't open file 'build_data.py'` | You are in the outer folder; `cd` into the inner one |
| Download says rate limit / 429 | Wait 10 minutes, run `python build_data.py` again |
| `data/processed/daily.csv.gz not found` | Step 3 did not finish |
| LightGBM will not install | `set AIRQ_BACKEND=sklearn` before the commands; results will say sklearn was used |
| Dashboard says models are missing | Run step 6 |
