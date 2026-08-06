# LSTM Trend

A Streamlit web app that pulls historical stock data from Yahoo Finance, trains a
small LSTM (Long Short-Term Memory) neural network on the closing prices, and
tries to forecast what the price might do over the next few business days.

**Live app:** <https://lstm-trend.streamlit.app>

---

## Table of contents

- [About this project](#about-this-project)
- [What the app does](#what-the-app-does)
- [Features](#features)
- [Tech stack](#tech-stack)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Running it locally](#running-it-locally)
- [GPU acceleration](#gpu-acceleration)
- [Tests](#tests)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Configuration](#configuration)
- [Project history](#project-history)
- [Honest limitations](#honest-limitations)
- [Ideas for the future](#ideas-for-the-future)
- [License](#license)

---

## About this project

This started as my final year project (FYP) around 2023. I wanted to learn how
deep learning models actually behave on real-world time series data, so I built a
small LSTM network that predicts stock prices and wrapped it in a Streamlit app
so that anyone (including my examiners) could try it without touching code.

The project has kept growing since then:

- the modelling stack now runs on **Keras 3 with a PyTorch backend** — the same
  model code, with CUDA support that just works instead of requiring a fragile
  `LD_LIBRARY_PATH` dance,
- the app grew into a **multipage tool**: a Dashboard for forecasting, an
  Analytics page for deep exploratory analysis, a Compare page for multi-ticker
  work, an empirical Findings page, and an About page,
- the analysis is grounded in **baselines**, so the LSTM's numbers are always
  shown next to 'repeat yesterday' and a moving average,
- the project has a proper **test suite** (unit tests + AppTest end-to-end
  checks) wired into **GitHub Actions CI**,
- everything runs on **uv** for a clean, reproducible environment.

One thing I want to be upfront about: this is a learning project, not a trading
tool. Stock prediction with a single LSTM on just the closing price is a hard,
humbling problem, and the results should be read with a lot of scepticism. More
on that in [Honest limitations](#honest-limitations).

## What the app does

In plain words:

1. You pick a stock ticker (for example `AAPL`) and a date range.
2. The app downloads the historical prices from Yahoo Finance and shows you
   charts — closing price, trading volume, and a candlestick chart.
3. It feeds the closing prices to an LSTM model, which learns from the past to
   predict the next day's price.
4. It trains the model, shows you how well it did (MSE, RMSE, MAE, MAPE, R²),
   compares it against naive and moving-average baselines, and then forecasts
   prices for the next few business days.
5. It draws everything on interactive charts so you can zoom in, hover, and
   explore.

## Features

**Data**

- Fetches daily open/high/low/close/volume data from Yahoo Finance via `yfinance`.
- Handles empty downloads, bad dates, missing values, and the awkward
  multi-level column layout that `yfinance` sometimes returns.

**Charts (all interactive, Plotly)**

- Closing price over time.
- Trading volume over time.
- Candlestick chart of open, high, low, and close — long date ranges are
  automatically aggregated into weekly bars so the chart stays readable.
- Training loss curves (training vs validation).
- Test predictions against actual prices, with the forecasted future prices.
- Bar chart of each model's RMSE, plus a full comparison chart of test
  predictions across LSTM, naive, and moving-average baselines.

**Model**

- LSTM built with Keras' Functional API: two LSTM layers with dropout in
  between, a dense layer, and an output layer.
- Prices are scaled to 0–1 before training (MinMaxScaler), and predictions are
  scaled back to real prices afterwards.
- Early stopping with patience, seeded for reproducible runs.
- A live progress bar and epoch-by-epoch loss readout while training.

**Evaluation**

- MSE, RMSE, MAE, MAPE and R², all calculated on real prices rather than scaled
  values, so they actually mean something.
- The Findings page puts those numbers in context: does the LSTM beat
  "repeat yesterday" (naive) or a 20-day moving average? The verdict is stated
  honestly either way.

**Interface**

- Six pages: Dashboard (run analyses, see charts and forecast), Analytics
  (deep EDA of any ticker), Compare (multi-ticker analysis), Findings
  (metrics, baseline comparison, caveats), Methodology (pipeline diagram,
  architecture, hyperparameters), About (project story).
- All controls live in the sidebar inside popovers: ticker (with suggestions),
  dates, lookback window, prediction horizon, epochs, and batch size.
- Forecast table with formatted columns and a CSV download button.
- Dark theme configured in `.streamlit/config.toml`, custom logo in the sidebar.
- The training device (GPU or CPU) is shown in the sidebar.

**Analytics (EDA)**

- Performance snapshot: last close, YTD/1M/6M/1Y returns, 52-week range
  position, total return, CAGR, annualized volatility.
- Risk metrics: Sharpe and Sortino ratios, max drawdown, historical VaR and
  CVaR, positive-day ratio, full drawdown events table.
- Returns analysis: histogram with KDE, Q-Q plot vs normal, rolling
  volatility, autocorrelation, weekday effects, skewness/kurtosis.
- Seasonality: year × month return heatmap, average return and hit rate by
  month and weekday.
- Stationarity: Augmented Dickey–Fuller test on log prices with a plain-English
  interpretation.
- Technical indicators: SMA 20/50/200, EMA 50, golden/death cross markers,
  RSI (14), MACD (12,26,9), Bollinger bands (20, 2σ), plus a human-readable
  "current signals" summary.
- Volume analysis: volume bars coloured by daily move (weekly bars on long
  histories), and a volume-vs-return scatter.

**Compare (multi-ticker)**

- Normalized price chart (all tickers rebased to 100) and cumulative returns.
- Correlation matrix of daily returns with a plain-English reading.
- Drawdown comparison, and a risk-vs-return scatter coloured by Sharpe ratio.
- Side-by-side metrics table (total return, CAGR, volatility, Sharpe, Sortino,
  max drawdown, VaR, CVaR, positive days) with CSV download.

## Tech stack

| Piece            | What it's used for                                        |
| ---------------- | --------------------------------------------------------- |
| Python 3.13      | The language everything is written in                     |
| Streamlit 1.61+  | The web app framework                                     |
| yfinance         | Downloading stock data from Yahoo Finance                 |
| pandas / numpy   | Data wrangling and numerical work                         |
| Keras 3 + PyTorch| Building and training the LSTM model                      |
| scikit-learn     | MinMaxScaler, MSE and R² metrics                          |
| scipy            | Distributions, KDE, Q-Q plots, moments                    |
| statsmodels      | Augmented Dickey–Fuller stationarity test                 |
| Plotly           | Interactive charts                                        |
| uv               | Environment and dependency management                     |
| pytest           | Unit tests and Streamlit AppTest end-to-end tests         |
| GitHub Actions   | CI: type-check, format check, tests                        |

## How it works

The pipeline is simple enough to describe in one breath:

```
download data → scale prices → build sequences → split train/test
             → train LSTM → evaluate → predict future → plot
```

A few details worth knowing:

- **Sequences.** The model doesn't see single days; it sees windows. With a
  lookback of 60, each training sample is the last 60 closing prices and the
  "answer" is the price on day 61. That window size is something you can change
  in the sidebar.
- **Train/test split.** 80% of the sequence samples are used for training, 20%
  for testing. The test part is data the model has never seen, which is where
  the metrics come from.
- **Baselines.** The test window is scored twice more: once with a naive model
  ("tomorrow equals today") and once with a 20-day moving average. The Findings
  page compares all three, which is the closest this toy project gets to
  honesty.
- **Forecasting.** The model predicts one day at a time. Each prediction gets
  appended to the window, the oldest day drops off, and the model predicts
  again. That means small errors can build up over the horizon — the further
  ahead you ask, the less reliable it gets. The forecast dates are business
  days only, since that's when markets are actually open.
- **LSTM in one sentence.** An LSTM is a neural network with a small internal
  memory, so it can hold onto patterns from earlier in a sequence instead of
  only seeing the most recent value. That makes it a natural fit for time
  series — but as with any model, it only learns patterns that exist in the
  training data.

## Project structure

```
lstm-trend/
├── streamlit_app.py          # Entry point: st.navigation + st.logo
├── src/
│   ├── constants.py          # Defaults and shared settings
│   ├── core/                 # Pure logic, no Streamlit imports
│   │   ├── data_loader.py    # yfinance download + column flattening
│   │   ├── preprocessing.py  # Scaling and sequence creation
│   │   ├── metrics.py        # MSE / RMSE / MAE / MAPE / R²
│   │   ├── baselines.py      # Naive and moving-average baselines
│   │   ├── analytics.py      # EDA: risk metrics, drawdowns, seasonality, indicators
│   │   ├── callbacks.py      # Keras callback driving progress callbacks
│   │   ├── lstm_model.py     # Model creation, training, forecasting
│   │   └── pipeline.py       # run_analysis(): the whole pipeline, typed
│   └── ui/                   # Streamlit-specific rendering
│       ├── charts.py         # All Plotly chart builders
│       ├── components.py     # Sidebar config, progress UI, result rendering
│       └── pages/            # dashboard, analytics, compare, findings, methodology, about
├── tests/                    # pytest unit tests + AppTest E2E
├── .github/workflows/ci.yml  # CI: uv sync, check, format, pytest
├── .streamlit/config.toml    # Dark theme
├── assets/logo.svg           # Sidebar logo
├── pyproject.toml            # Project metadata + dependencies (uv)
├── uv.lock                   # Locked dependency versions
├── run.sh                    # Launcher: sets KERAS_BACKEND, runs the app
└── README.md                 # This file
```

## Running it locally

I use **uv** for everything in this project — it keeps the environment
reproducible and fast. If you're not familiar with it, it's a modern replacement
for `pip` + `venv` and you can install it from <https://docs.astral.sh/uv/>.

### Prerequisites

- Python 3.13 (the project is configured for it via `.python-version`)
- [uv](https://docs.astral.sh/uv/)

### Steps

```bash
# 1. Get the code
git clone git@github.com:c0k0n/lstm-trend.git
cd lstm-trend

# 2. Create the virtual environment and install everything
uv sync

# 3. Run the app
./run.sh
```

Your browser should open on `http://localhost:8501`. From there, pick a ticker,
adjust the settings in the sidebar, and hit the **Run analysis** button.

A few notes:

- The first run downloads PyTorch, so be patient if `uv sync` takes a while.
- The first analysis downloads data and trains a model, which takes a bit of
  time too — the progress bar will keep you company.

## GPU acceleration

The modelling stack is Keras 3 running on a **PyTorch** backend. PyTorch ships
its CUDA libraries inside its own wheels, which means:

- on a machine with an NVIDIA GPU and working drivers, the app picks up the GPU
  automatically — no environment variables, no extra packages,
- the sidebar shows which device was used ("Training device: GPU"),
- on machines without a GPU (including Streamlit Community Cloud) it silently
  runs on CPU.

That's a deliberate change from the original TensorFlow setup, which needed
CUDA libraries installed into the virtualenv plus an `LD_LIBRARY_PATH` export
(`run.sh` used to do both). None of that is needed anymore.

## Tests

```bash
uv run pytest tests -q
```

The suite covers:

- unit tests for scaling/sequence creation, metrics, and the baselines
  (`tests/test_preprocessing.py`, `test_metrics.py`, `test_baselines.py`),
- a full suite for the analytics module: returns, risk metrics, drawdowns,
  seasonality, stationarity, and every technical indicator
  (`tests/test_analytics.py`),
- a full pipeline test on synthetic data with a tiny model
  (`tests/test_pipeline.py`),
- AppTest end-to-end tests that boot the real app, click through a full
  analysis (this one downloads live data from Yahoo Finance), and check the
  Analytics, Compare and Findings pages (`tests/test_app.py`).

CI runs on every push to `main` via GitHub Actions: `uv sync`, `uv check`
(type-check), `uv format --check`, then `pytest`.

## Deploying to Streamlit Community Cloud

The app is already live at <https://lstm-trend.streamlit.app>, deployed on
Streamlit's free Community Cloud tier straight from this GitHub repository. If
you ever need to redeploy it (or set up your own copy), here's the routine:

1. Push the repository to GitHub.
2. Go to <https://share.streamlit.io> and click **Create app**, then connect the
   repository (for me: `c0k0n/lstm-trend`, branch `main`, entrypoint
   `streamlit_app.py`).
3. **Important:** open **Advanced settings** and set the Python version to
   **3.13**. Community Cloud defaults to 3.12, and this project requires 3.13+.
4. Deploy and wait a few minutes — PyTorch makes the first build take longer
   than a typical Streamlit app.

How the cloud figures out dependencies: Community Cloud looks for dependency
files in order of priority, and **`uv.lock` wins** over `requirements.txt`,
`pyproject.toml`, and the rest. Since this repo ships a lock file, the cloud
installs the exact same versions you run locally.

One heads-up about the free tier: the app sleeps after a while of inactivity,
and the first visitor after a nap has to wait for it to wake up and rebuild the
model. That's normal Streamlit Community Cloud behaviour, not a bug.

## Configuration

Everything is controlled from the sidebar (inside popovers):

| Setting                 | What it does                            | Range          | Default |
| ----------------------- | --------------------------------------- | -------------- | ------- |
| Ticker                  | Stock to analyse (`AAPL`, `MSFT`, … or Custom) | —        | `AAPL`  |
| Start / End Date        | Historical data range                   | —              | 2020 → today |
| Lookback window         | Days of history the model sees per step | 10 – 120       | 60      |
| Prediction horizon      | Business days to forecast ahead         | 5 – 90         | 15      |
| Epochs                  | Training passes over the data           | 1 – 100        | 50      |
| Batch size              | Samples per training step               | 8 – 128        | 32      |

Some other fixed settings live in `src/constants.py`: 80/20 train-test split,
10% validation split, 100 LSTM units, 0.2 dropout, early-stopping patience,
the moving-average window, and the plot colours.

## Project history

- **2023 — FYP start.** First version of the app: Streamlit, Keras LSTM, a few
  notebooks worth of trial and error, and a README that was mostly notes to
  myself.
- **2026 — continued development.** Moved to uv with a proper `pyproject.toml`
  and lock file, updated every dependency to a current version, replaced
  deprecated Streamlit API calls, and deployed the app publicly on Community
  Cloud.
- **2026 — latest chapters.** Migrated to Keras 3 + PyTorch (GPU support
  without the `LD_LIBRARY_PATH` hacks), grew into a multipage app with deep
  analytics and comparison pages, added baseline comparisons and an empirical
  Findings page, and introduced a test suite with GitHub Actions CI.

The git history still contains all the earlier commits, if you ever want to see
how it evolved.

## Honest limitations

I don't want this README to oversell the project, so here are the things I know
are weak or missing:

- **This is not investment advice.** Markets are noisy and influenced by things
  no price history can tell a model — news, earnings, sentiment, policy. A
  single LSTM on closing prices is a toy model compared to what professional
  shops run, and it can easily be wrong.
- **Future forecasts drift.** Because predictions feed back into the window,
  errors compound. The 5-day forecast is more believable than the 90-day one.
- **Only the closing price is used.** No volume, no indicators like RSI or MACD,
  no fundamentals. That's a deliberate simplification, but it leaves a lot on
  the table.
- **Free-tier constraints.** Training happens on every run (nothing is saved
  between sessions), and the Community Cloud free tier is memory-limited, so
  large lookbacks or long histories can be slow.
- **Metrics on a trending stock look flattering.** R² on a strongly trending
  stock is easy to score well on; a flat, choppy stock is much harder. That's
  exactly why the Findings page also shows the naive and moving-average
  baselines — don't quote the LSTM's number without its context.

## Ideas for the future

In rough order of how useful I think they'd be:

- Save and reuse trained models (and cache predictions) so repeat visits don't
  retrain from scratch.
- Add more features to the model: trading volume, technical indicators, or
  returns instead of raw prices.
- Let users pick the target column (Open, High, Low) and the model architecture.
- Add confidence bands around the forecast.
- A proper backtest view: how would this model have performed on past periods?

## License

MIT — see [LICENSE](LICENSE). Use it freely; just don't blame me if the
prediction says "up" and the stock goes down.
