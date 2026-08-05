# LSTM Stock Trend Predictor

A Streamlit web app that pulls historical stock data from Yahoo Finance, trains a
small LSTM (Long Short-Term Memory) neural network on the closing prices, and tries
to forecast what the price might do over the next few business days.

**Live app:** https://lstm-trend.streamlit.app

---

## Table of contents

- [About this project](#about-this-project)
- [What the app does](#what-the-app-does)
- [Features](#features)
- [Tech stack](#tech-stack)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Running it locally](#running-it-locally)
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

The original code served its purpose, but a lot of things in it aged poorly —
old library versions, deprecated APIs, and setup steps that assumed a very
specific machine. So in 2026 I went back and rejuvenated it:

- everything now runs on **uv** for a clean, reproducible environment,
- all libraries were bumped to current versions (Streamlit 1.61, TensorFlow 2.21,
  pandas 3.x, and friends),
- deprecated Streamlit calls were replaced with their modern equivalents,
- the app is now live on free Streamlit Community Cloud.

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
4. It trains the model, shows you how well it did (with simple metrics), and then
   forecasts prices for the next few business days.
5. It draws everything on interactive charts so you can zoom in, hover, and
   explore.

## Features

**Data**

- Fetches daily open/high/low/close/volume data from Yahoo Finance via `yfinance`.
- Downloads are cached, so re-running with the same settings doesn't hit the
  network again.
- Handles empty downloads, bad dates, missing values, and the awkward
  multi-level column layout that `yfinance` sometimes returns.

**Charts (all interactive, Plotly)**

- Closing price over time.
- Trading volume over time.
- Candlestick chart of open, high, low, and close.
- Training loss curves (training vs validation).
- Final chart combining actual test prices, the model's test predictions, and the
  forecasted future prices.

**Model**

- LSTM built with Keras' Functional API: two LSTM layers with dropout in between,
  a dense layer, and an output layer.
- Prices are scaled to 0–1 before training (MinMaxScaler), and predictions are
  scaled back to real prices afterwards.
- Early stopping to avoid wasting time once the model stops improving.
- A live progress bar and epoch-by-epoch loss readout while training.

**Evaluation**

- Mean Squared Error (MSE) and R² score, both calculated on real prices rather
  than scaled values, so they actually mean something.

**Interface**

- All controls live in the sidebar: ticker, dates, lookback window, prediction
  horizon, epochs, and batch size.
- Dark theme configured in `.streamlit/config.toml`.

## Tech stack

| Piece            | What it's used for                                        |
| ---------------- | --------------------------------------------------------- |
| Python 3.13      | The language everything is written in                     |
| Streamlit 1.61+  | The web app framework                                     |
| yfinance         | Downloading stock data from Yahoo Finance                 |
| pandas / numpy   | Data wrangling and numerical work                         |
| TensorFlow/Keras | Building and training the LSTM model                      |
| scikit-learn     | MinMaxScaler, MSE and R² metrics                          |
| Plotly           | Interactive charts                                        |
| uv               | Environment and dependency management                     |

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
  for testing. The test part is data the model has never seen, which is where the
  MSE and R² come from.
- **Forecasting.** The model predicts one day at a time. Each prediction gets
  appended to the window, the oldest day drops off, and the model predicts again.
  That means small errors can build up over the horizon — the further ahead you
  ask, the less reliable it gets. I made the forecast dates business days only,
  since that's when markets are actually open.
- **LSTM in one sentence.** An LSTM is a neural network with a small internal
  memory, so it can hold onto patterns from earlier in a sequence instead of only
  seeing the most recent value. That makes it a natural fit for time series — but
  as with any model, it only learns patterns that exist in the training data.

## Project structure

```
lstm-trend/
├── streamlit_app.py          # Entry point for Streamlit (streamlit run this)
├── src/
│   ├── app.py                # Main app logic and UI layout
│   ├── constants.py          # Defaults and shared settings
│   ├── utils/
│   │   ├── data_loader.py    # yfinance download + caching
│   │   ├── preprocessing.py  # Scaling and sequence creation
│   │   └── plotting.py       # All Plotly charts
│   └── models/
│       ├── lstm_model.py     # Model creation, training, evaluation, forecasting
│       └── callbacks.py      # Keras callback that drives the Streamlit progress bar
├── .streamlit/config.toml    # Dark theme
├── pyproject.toml            # Project metadata + dependencies (uv)
├── uv.lock                   # Locked dependency versions
├── requirements.txt          # Unpinned deps, for people who prefer pip
└── README.md                 # This file
```

## Running it locally

I use **uv** for everything in this project — it keeps the environment
reproducible and fast. If you're not familiar with it, it's a modern replacement
for `pip` + `venv` and you can install it from https://docs.astral.sh/uv/.

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
uv run streamlit run streamlit_app.py
```

Your browser should open on `http://localhost:8501`. From there, pick a ticker,
adjust the settings in the sidebar, and hit the **Run Analysis & Prediction**
button.

A few notes:

- The first run downloads TensorFlow, so be patient if `uv sync` takes a while.
- The first analysis downloads data and trains a model, which takes a bit of time
  too — the progress bar will keep you company.
- If you'd rather use `pip`, `requirements.txt` lists the same dependencies, but
  honestly, `uv sync` is the path I test and recommend.

## Deploying to Streamlit Community Cloud

The app is already live at https://lstm-trend.streamlit.app, deployed on
Streamlit's free Community Cloud tier straight from this GitHub repository. If
you ever need to redeploy it (or set up your own copy), here's the routine:

1. Push the repository to GitHub.
2. Go to https://share.streamlit.io and click **Create app**, then connect the
   repository (for me: `c0k0n/lstm-trend`, branch `main`, entrypoint
   `streamlit_app.py`).
3. **Important:** open **Advanced settings** and set the Python version to
   **3.13**. Community Cloud defaults to 3.12, and this project requires 3.13+.
4. Deploy and wait a few minutes — TensorFlow makes the first build take longer
   than a typical Streamlit app.

How the cloud figures out dependencies: Community Cloud looks for dependency
files in order of priority, and `uv.lock` wins over `requirements.txt`. Since
this repo has both, the cloud uses the lock file, which is exactly what we want
for reproducibility. `requirements.txt` stays around only for people who prefer
pip locally.

One heads-up about the free tier: the app sleeps after a while of inactivity,
and the first visitor after a nap has to wait for it to wake up and rebuild the
model. That's normal Streamlit Community Cloud behaviour, not a bug.

## Configuration

Everything is controlled from the sidebar:

| Setting                 | What it does                            | Range          | Default |
| ----------------------- | --------------------------------------- | -------------- | ------- |
| Stock Symbol            | Ticker to analyse (e.g. `AAPL`)         | —              | `AAPL`  |
| Start / End Date        | Historical data range                   | —              | 2020 → today |
| Lookback Window (Days)  | Days of history the model sees per step | 10 – 120       | 60      |
| Prediction Horizon      | Business days to forecast ahead         | 5 – 90         | 15      |
| Epochs                  | Training passes over the data           | 1 – 500        | 5       |
| Batch Size              | Samples per training step               | 8 – 128        | 32      |

Some other fixed settings live in `src/constants.py`: 80/20 train-test split,
10% validation split, 100 LSTM units, and the plot colours.

## Project history

- **2023 — original FYP build.** First version of the app: Streamlit, Keras LSTM,
  a few notebooks worth of trial and error, and a README that was mostly notes to
  myself.
- **2026 — rejuvenation.** Moved to uv with a proper `pyproject.toml` and lock
  file, updated every dependency to a current version, replaced deprecated
  Streamlit API calls, cleaned up the project structure, refreshed this README,
  and deployed the app publicly on Community Cloud.

The git history still contains the old commits, if you ever want to see how it
evolved.

## Honest limitations

I don't want this README to oversell the project, so here are the things I know
are weak or missing:

- **This is not investment advice.** Markets are noisy and influenced by things
  no price history can tell a model — news, earnings, sentiment, policy. A single
  LSTM on closing prices is a toy model compared to what professional shops run,
  and it can easily be wrong.
- **Future forecasts drift.** Because predictions feed back into the window,
  errors compound. The 5-day forecast is more believable than the 90-day one.
- **Only the closing price is used.** No volume, no indicators like RSI or MACD,
  no fundamentals. That's a deliberate simplification, but it leaves a lot on the
  table.
- **Free-tier constraints.** Training happens on every run (nothing is saved
  between sessions), and the Community Cloud free tier is memory-limited, so
  large lookbacks or long histories can be slow.
- **Metrics on a trending stock look flattering.** R² on a strongly trending
  stock is easy to score well on; a flat, choppy stock is much harder. Don't
  quote the number without context.
- **No automated tests yet.** The app is exercised manually; a proper test suite
  is on the to-do list.

## Ideas for the future

In rough order of how useful I think they'd be:

- Save and reuse trained models (and cache predictions) so repeat visits don't
  retrain from scratch.
- Add more features to the model: trading volume, technical indicators, or
  returns instead of raw prices.
- Let users pick the target column (Open, High, Low) and the model architecture.
- Add confidence bands around the forecast.
- A small test suite with Streamlit's `AppTest` framework.
- A proper backtest view: how would this model have performed on past periods?

## License

MIT — see [LICENSE](LICENSE). Use it freely; just don't blame me if the
prediction says "up" and the stock goes down.
