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
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Running it locally](#running-it-locally)
- [GPU acceleration](#gpu-acceleration)
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
  work, an empirical Findings page, a Methodology page, and an About page,
- the analysis is grounded in **baselines**, so the LSTM's numbers are always
  shown next to 'repeat yesterday' and a moving average,
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
- Downloaded data is cached for one hour (`st.cache_data(ttl=3600)`) to avoid
  hitting Yahoo Finance rate limits on repeated visits.

**Charts (all interactive, Plotly)**

- Closing price over time.
- Trading volume over time — bars coloured green/red by daily return; on
  histories longer than one year, volume is aggregated to weekly bars.
- Candlestick chart of open, high, low, and close — long date ranges are
  automatically aggregated into weekly bars so the chart stays readable.
- Training loss curves (training vs validation).
- Test predictions against actual prices, with the forecasted future prices.
- Bar chart of each model's RMSE, plus a full comparison chart of test
  predictions across LSTM, naive, and moving-average baselines.

**Model**

- LSTM built with Keras' Functional API: two LSTM layers (100 then 50 units)
  with dropout after each, a dense layer (25 units, ReLU), and an output
  layer — compiled with Adam and mean absolute error.
- Prices are scaled to 0-1 before training (MinMaxScaler), and predictions are
  scaled back to real prices afterwards.
- Early stopping (patience 10) with `restore_best_weights`, and every run is
  seeded (`keras.utils.set_random_seed(42)`) so results are reproducible.
- A live progress bar and epoch-by-epoch loss readout while training, driven by
  a `ProgressReporterCallback` that reports through plain callables (no
  Streamlit imports in the core layer).

**Evaluation**

- MSE, RMSE, MAE, MAPE and R², all calculated on real prices rather than scaled
  values, so they actually mean something.
- The Findings page puts those numbers in context: does the LSTM beat
  'repeat yesterday' (naive) or a 20-day moving average? The verdict is stated
  honestly either way.

**Interface**

- Six pages: Dashboard (run analyses, see charts and forecast), Analytics
  (deep EDA of any ticker), Compare (multi-ticker analysis), Findings
  (metrics, baseline comparison, caveats), Methodology (pipeline diagram,
  architecture, hyperparameters), About (project story).
- All controls live in the sidebar: ticker and model parameters sit in
  popovers, and the model settings (lookback, horizon, epochs, batch size)
  are wrapped in a `st.form` — nothing takes effect until you press
  **Apply model settings**.
- Forecast horizon picks come as presets (5 / 15 / 30 days or Custom...)
  via a `st.segmented_control`. Custom picks a value between 5 and 90.
- The sidebar shows a **Current settings** summary (ticker, dates, horizon)
  below the Run button, and the training device (GPU or CPU) is shown at
  the top of the sidebar.
- Forecast table with formatted columns and a CSV download button.
- Dark theme configured in `.streamlit/config.toml`.
- Shareable deep links: after a run the ticker, dates and horizon are written
  back to the URL, so you can bookmark or share an exact analysis. Preset
  horizons (5/15/30) land on the matching preset; any other value (5-90)
  switches the control to Custom with the forecast length set accordingly.
- Loading states everywhere: skeletons while data downloads, a streaming
  verdict on the Findings page, a live "analysis run at ..." age indicator that
  refreshes itself every 60 seconds (via `@st.fragment(run_every=60)`), and a
  one-time confetti celebration after the first run.
- A "how to read the charts" dialog on the Findings page and a theme-aware
  app footer with the GitHub link.
- Rounded widget corners via Streamlit theming.

**Analytics (EDA)**

- Performance snapshot: last close, YTD/1M/6M/1Y returns, 52-week range
  position, total return, CAGR, annualized volatility.
- Risk metrics: Sharpe and Sortino ratios, max drawdown, historical VaR and
  CVaR, positive-day ratio, full drawdown events table — paginated 10 rows at
  a time, with a count caption.
- Returns analysis: histogram with KDE, Q-Q plot vs normal, rolling
  volatility, autocorrelation, weekday effects, skewness/kurtosis.
- Seasonality: year x month return heatmap, average return and hit rate by
  month and weekday.
- Stationarity: Augmented Dickey-Fuller test on log prices with a plain-English
  interpretation.
- Technical indicators: SMA 20/50/200, EMA 50, golden/death cross markers,
  RSI (14), MACD (12,26,9), Bollinger bands (20, 2 sigma), plus a human-readable
  "current signals" summary.
- Volume analysis: volume bars coloured by daily move (weekly bars on long
  histories), and a volume-vs-return scatter.
- Lazy tabs: only the tab you open actually computes — the other four stay
  idle until you click them.
- Skeleton placeholders on first data load, then cached for the session.

**Compare (multi-ticker)**

- Normalized price chart (all tickers rebased to 100) and cumulative returns.
- Correlation matrix of daily returns with a plain-English reading.
- Drawdown comparison, and a risk-vs-return scatter coloured by Sharpe ratio.
- Side-by-side metrics table (total return, CAGR, volatility, Sharpe, Sortino,
  max drawdown, VaR, CVaR, positive days) with CSV download. Each row carries
  a sparkline of the price trend.
- An editable watchlist (add/remove tickers in the Compare page) that feeds
  the comparison when enabled; requires at least 2 tickers with data.

## Tech stack

| Piece             | Version    | What it's used for                                        |
| ----------------- | ---------- | --------------------------------------------------------- |
| Python            | 3.13       | The language everything is written in                     |
| Streamlit         | 1.61.1     | The web app framework                                     |
| yfinance          | 1.5.2      | Downloading stock data from Yahoo Finance                 |
| pandas            | 3.0.5      | Data wrangling, time series, DataFrames                   |
| NumPy             | 2.5.1      | Numerical operations, array slicing                       |
| Keras             | 3.15.1     | Building and training the LSTM model (Functional API)     |
| PyTorch           | 2.13.0     | Deep learning backend (GPU via CUDA when available)       |
| scikit-learn      | 1.9.0      | MinMaxScaler, MSE, MAE, MAPE, R² metrics                  |
| scipy             | 1.18.0     | Distributions, KDE, Q-Q plots, moments, ADF test          |
| Plotly            | 6.9.0      | Interactive charts                                        |
| uv                | (latest)   | Environment and dependency management                     |

Versions are pinned in `pyproject.toml` and locked in `uv.lock`.

## How it works

The pipeline is simple enough to describe in one breath:

```
download data -> scale prices -> build sequences -> split train/test
             -> train LSTM -> evaluate -> predict future -> plot
```

A few details worth knowing:

- **Sequences.** The model doesn't see single days; it sees windows. With a
  lookback of 60, each training sample is the last 60 closing prices and the
  "answer" is the price on day 61. That window size is something you can change
  in the sidebar.
- **Train/test split.** 80% of the sequence samples are used for training, 20%
  for testing. The test part is data the model has never seen, which is where
  the metrics come from. The last 10% of the training slice is held out as the
  validation set for early stopping — so the data splits roughly 72 / 8 / 20
  across train / validation / test.
- **Baselines.** The test window is scored twice more: once with a naive model
  ("tomorrow equals today") and once with a 20-day moving average. The Findings
  page compares all three, which is the closest this toy project gets to
  honesty.
- **Forecasting.** The model predicts one day at a time. Each prediction gets
  appended to the window, the oldest day drops off, and the model predicts
  again. That means small errors can build up over the horizon — the further
  ahead you ask, the less reliable it gets. The forecast dates are business
  days only, since that's when markets are actually open. Each forecast row
  also carries a day-over-day change percentage.
- **LSTM in one sentence.** An LSTM is a neural network with a small internal
  memory, so it can hold onto patterns from earlier in a sequence instead of
  only seeing the most recent value. That makes it a natural fit for time
  series — but as with any model, it only learns patterns that exist in the
  training data.
- **A quiet terminal.** PyTorch's cuDNN LSTM path prints a "weights are not
  contiguous" hint on every forward pass; it is a performance note we can't act
  on (Keras calls the functional PyTorch API), so it is filtered out in
  `src/core/lstm_model.py`.

## Architecture

The codebase is split into two clean layers:

### `src/core/` — pure analysis logic

No Streamlit imports. Every function here is testable in isolation, which keeps
the business logic portable and debuggable:

| Module              | Responsibility                                              |
| ------------------- | ----------------------------------------------------------- |
| `data_loader.py`    | Download OHLCV from yfinance, flatten MultiIndex columns    |
| `preprocessing.py`  | MinMaxScaler wrapping, sliding-window sequence creation     |
| `callbacks.py`      | `ProgressReporterCallback` — reports epoch progress via plain callables |
| `lstm_model.py`     | Model definition (Functional API), training, evaluation, step-wise forecasting |
| `pipeline.py`       | `run_analysis()` orchestrator, `AnalysisResult` dataclass, `PipelineError` |
| `baselines.py`      | Naive (repeat yesterday) and moving-average forecasts        |
| `metrics.py`        | MSE / RMSE / MAE / MAPE / R² on original price scale       |
| `returns.py`        | Daily/cumulative/annualized returns, CAGR, trailing periods, 52-week position |
| `risk.py`           | Sharpe, Sortino, max drawdown, drawdown events, VaR, CVaR  |
| `seasonality.py`    | Weekday effects, monthly return heatmap, month effects       |
| `statistics.py`     | Return moments (skew/kurtosis), ADF test, autocorrelation   |
| `indicators.py`     | SMA, EMA, RSI, MACD, Bollinger bands, crossover detection   |
| `constants.py`      | All shared defaults, URL constants, plot colour palette      |

### `src/ui/` — Streamlit presentation layer

Streamlit-specific rendering. Each module has a single responsibility:

| Module                | Responsibility                                              |
| --------------------- | ----------------------------------------------------------- |
| `chart_theme.py`      | Shared Plotly theme (`DARK_THEME`), colour palette, `layout()` helper, `with_alpha()`, delegates `cumulative_returns()` to `returns.py` |
| `dashboard_charts.py` | Dashboard chart builders: candlestick, loss history, test predictions, forecast, baselines, metric bars |
| `analytics_charts.py` | Analytics page charts: cumulative returns, histogram+KDE, Q-Q, rolling volatility, ACF, weekday effects, monthly heatmap, underwater, indicators (price+MA, RSI, MACD, Bollinger), volume analysis |
| `compare_charts.py`   | Compare page charts: normalized prices, cumulative comparison, correlation heatmap, drawdown comparison, risk-vs-return scatter |
| `sidebar.py`          | Sidebar controls, symbol picker (pills + custom), model-settings form, session state helpers, progress UI, training device detection |
| `result_rendering.py` | Metric cards, forecast table with CSV download, quick stats, full dashboard result assembly, analysis age caption (`@st.fragment`) |
| `pages/nav.py`        | Page registry (`st.Page` objects) for `st.navigation`       |
| `pages/dashboard.py`  | Run analyses, deep-link query params, render results        |
| `pages/analytics.py`  | Deep EDA with lazy tabs: Overview, Returns, Seasonality, Technicals, Risk |
| `pages/compare.py`    | Multi-ticker comparison with editable watchlist             |
| `pages/findings.py`   | Empirical verdict: LSTM vs baselines with streaming text    |
| `pages/methodology.py`| Pipeline/architecture Mermaid diagrams, hyperparameter table, design trade-offs |
| `pages/about.py`      | Project story, FAQ, tech stack table, run instructions      |

### Key patterns

- **Session state sharing.** The `AnalysisResult` is stored in
  `st.session_state["analysis"]` so every page can read it without re-running
  the pipeline. The Findings page, for example, just calls `get_analysis()`.
- **Form-gated model settings.** Lookback, horizon, epochs, and batch size live
  inside a `st.form` — nothing takes effect until the user presses Apply. Date
  pickers take effect immediately.
- **Progress via callables.** The `ProgressReporterCallback` receives two
  callbacks (`on_epoch`, `on_finish`) and reports to them. The UI layer in
  `sidebar.py` maps these to an `st.progress` bar inside an `st.status` box.
- **Lazy tab loading.** The Analytics page uses `if tabs[i].open:` so each tab
  only computes when the user clicks it — no wasted work.
- **Fragment-based auto-refresh.** The "analysis run at ..." age caption uses
  `@st.fragment(run_every=60)` to refresh itself every minute without a full
  page rerun.
- **Cached data downloads.** `load_data_cached()` wraps yfinance in
  `st.cache_data(ttl=3600)` — one hour of caching to avoid rate limits.
- **Device detection.** `training_device()` uses `@lru_cache(maxsize=1)` and
  checks `torch.cuda.is_available()` once, then never recomputes.

## Project structure

```
lstm-trend/
├── streamlit_app.py            # Entry point: sets KERAS_BACKEND, st.navigation + HTML footer
├── .gitignore                  # Git ignore rules (excludes AGENTS.md, streamlitinfolinks.txt)
├── .python-version             # Pins Python 3.13 for uv
├── .streamlit/
│   └── config.toml             # Dark theme, widget borders, usage stats off
├── src/
│   ├── __init__.py
│   ├── constants.py            # All shared defaults, URLs, plot colour palette
│   ├── core/                   # Pure analysis logic — no Streamlit imports
│   │   ├── __init__.py
│   │   ├── data_loader.py      # yfinance download + MultiIndex column flattening
│   │   ├── preprocessing.py    # MinMaxScaler, sliding-window sequence creation
│   │   ├── callbacks.py        # ProgressReporterCallback (callable-based, no UI deps)
│   │   ├── lstm_model.py       # Model creation (Functional API), training, evaluation, forecasting
│   │   ├── pipeline.py         # run_analysis() orchestrator, AnalysisResult dataclass, PipelineError
│   │   ├── baselines.py        # Naive (repeat yesterday) and moving-average baselines
│   │   ├── metrics.py          # MSE / RMSE / MAE / MAPE / R² on original price scale
│   │   ├── returns.py          # Daily/cumulative/annualized returns, CAGR, trailing periods
│   │   ├── risk.py             # Sharpe, Sortino, drawdowns, VaR, CVaR, comparison_frame()
│   │   ├── seasonality.py      # Weekday effects, monthly return heatmap, month effects
│   │   ├── statistics.py       # Return moments, ADF test (MacKinnon p-value), autocorrelation
│   │   └── indicators.py       # SMA, EMA, RSI, MACD, Bollinger, crossover detection, latest_signals()
│   └── ui/                     # Streamlit-specific rendering
│       ├── __init__.py
│       ├── chart_theme.py      # DARK_THEME, PALETTE, layout(), with_alpha(), cumulative_returns()
│       ├── dashboard_charts.py # Candlestick, loss history, test predictions, forecast, baselines, RMSE bars
│       ├── analytics_charts.py # 15 chart builders for returns, risk, seasonality, indicators, volume
│       ├── compare_charts.py   # Normalized prices, cumulative comparison, correlation, drawdown, risk scatter
│       ├── sidebar.py          # Sidebar config, symbol picker, model-settings form, progress UI, session helpers
│       ├── result_rendering.py # Metric cards, forecast table + CSV, quick stats, full dashboard assembly
│       └── pages/              # One module per app page
│           ├── __init__.py
│           ├── nav.py          # Page registry (st.Page objects) for st.navigation
│           ├── dashboard.py    # Run analyses, query-param deep links, render results
│           ├── analytics.py    # Deep EDA with lazy tabs: Overview, Returns, Seasonality, Technicals, Risk
│           ├── compare.py      # Multi-ticker comparison with editable watchlist
│           ├── findings.py     # LSTM vs baselines: streaming verdict, metrics table, charts help dialog
│           ├── methodology.py  # Mermaid diagrams, hyperparameter table, design trade-offs
│           └── about.py        # Project story, FAQ, tech stack, run instructions, acknowledgements
├── streamlitinfolinks.txt      # Index of official Streamlit docs links (dev reference, gitignored)
├── pyproject.toml              # Project metadata + dependencies (uv)
├── uv.lock                     # Locked dependency versions
├── AGENTS.md                   # Instructions for AI coding tools (gitignored)
├── LICENSE                     # MIT
└── README.md                   # This file
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
uv run streamlit run streamlit_app.py
```

Your browser should open on `http://localhost:8501`. From there, pick a ticker,
adjust the settings in the sidebar, and hit the **Run analysis** button.

A few notes:

- The first run downloads PyTorch, so be patient if `uv sync` takes a while.
- The first analysis downloads data and trains a model, which takes a bit of
  time too — the progress bar will keep you company.
- The minimum data requirement is `sequence_length + 10` trading days (default:
  70). Shorter ranges fail with a friendly error instead of crashing.
- The Analytics page requires at least 60 trading days of data.

## GPU acceleration

The modelling stack is Keras 3 running on a **PyTorch** backend. PyTorch ships
its CUDA libraries inside its own wheels, which means:

- on a machine with an NVIDIA GPU and working drivers, the app picks up the GPU
  automatically — no environment variables, no extra packages,
- the sidebar shows which device was used ("Training device: GPU (NVIDIA ...)"),
- on machines without a GPU (including Streamlit Community Cloud) it silently
  runs on CPU.

That's a deliberate change from the original TensorFlow setup, which needed
CUDA libraries installed into the virtualenv plus an `LD_LIBRARY_PATH` export
(a launcher script used to do both). None of that is needed anymore — the
entry point sets `KERAS_BACKEND=torch` itself, so
`uv run streamlit run streamlit_app.py` is all it takes (any pre-set
`KERAS_BACKEND` value wins).

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
| Ticker                  | Stock to analyse (`AAPL`, `MSFT`, ... or Custom) | --        | `AAPL`  |
| Start / End Date        | Historical data range                   | --             | 2020 -> today |
| Lookback window         | Days of history the model sees per step | 10 - 120       | 60      |
| Prediction horizon      | Business days to forecast ahead (preset or Custom) | 5 - 90 | 15  |
| Epochs                  | Training passes over the data           | 1 - 100        | 50      |
| Batch size              | Samples per training step (preset steps) | 8 - 128       | 32      |

The model settings only apply once **Apply model settings** is pressed; the
date pickers take effect immediately. You can also deep-link any analysis:
`?ticker=MSFT&start=2024-01-01&end=2025-01-01&horizon=30` prefills the
sidebar and triggers a run. `horizon` accepts the presets (5, 15, 30) or any
integer from 5-90 — non-preset values switch the segmented control to
Custom with your number as the forecast length.

Some other fixed settings live in `src/constants.py`: 80/20 train-test split,
10% validation split, 100 LSTM units (layer 1), 50 units (layer 2),
25 dense units, 0.2 dropout, early-stopping patience of 10,
random seed 42, the 20-day moving-average window, and the plot colour palette.

## Project history

- **2023 — FYP start.** First version of the app: Streamlit, Keras LSTM, a few
  notebooks worth of trial and error, and a README that was mostly notes to
  myself.
- **Continued development.** Moved to uv with a proper `pyproject.toml`
  and lock file, updated every dependency to a current version, replaced
  deprecated Streamlit API calls, and deployed the app publicly on Community
  Cloud.
- **Keras 3 + PyTorch.** Migrated to Keras 3 with a PyTorch backend (GPU
  support without the `LD_LIBRARY_PATH` hacks), grew into a multipage app with
  deep analytics and comparison pages, and added baseline comparisons and an
  empirical Findings page.
- **Modern UI pass.** Lazy tab loading, dialogs, model-settings forms,
  deep-linkable URLs, skeletons and streaming status, and a theme-aware footer.
- **Quality pass.** A full audit of the codebase: fixed the moving average
  baseline (it was scored against the wrong window), a crash on weekday tables
  when a weekday is missing, a broken deep-link path for custom horizons, and a
  hard-coded footer link; removed dead code and leftover dependencies,
  deduplicated chart colours and page configuration, and modernized the typing.
- **Second quality pass.** Found and fixed the worst bug in the app: the
  drawdown detector silently dropped every drawdown event (a 10-day -50% crash
  reported "no drawdowns"). Also replaced magic calendar lookbacks with exact
  business-day offsets, fixed mislabelled monthly-returns heatmap columns and
  mismatched bar colours in the RMSE chart, made sequence creation zero-copy
  and baselines vectorized, seeded every training run (the docs always claimed
  it), added a minimum-data guard so tiny datasets fail with a friendly error
  instead of a Keras crash, adapted to Keras 3.15's string `verbose` API and
  pandas 3.0, and deleted the now-redundant `run.sh`.
- **Third quality pass.** Removed unused imports and dead parameters, replaced
  a fragile `id()`-based session key with a stable symbol-based key, added type
  annotations to chart and findings functions, fixed the lazy tab loading
  pattern, corrected the project structure in the README, and removed stale
  references to files that no longer exist.
- **Fourth quality pass.** Removed unnecessary `cast()` calls across the
  codebase (analytics, baselines, charts, compare, data loader, pipeline),
  extracted the magic millisecond-per-day constant in the volume chart, and
  simplified the drawdown event detector by dropping a redundant `pd.Timestamp()`
  wrapper that was masking a type-checker issue.
- **Module split.** Decomposed the three monolithic files (`analytics.py`,
  `charts.py`, `components.py`) into focused, single-responsibility modules:
  `returns.py`, `risk.py`, `seasonality.py`, `statistics.py`, `indicators.py`
  for the core layer; `chart_theme.py`, `dashboard_charts.py`,
  `analytics_charts.py`, `compare_charts.py` for chart builders; and
  `sidebar.py`, `result_rendering.py` for the UI layer.
- **Fifth quality pass.** Eliminated six redundant indicator computations in the
  Analytics Technicals tab (RSI, MACD, Bollinger now computed once in
  `_technicals` and passed to chart builders). Moved `cumulative_returns()` from
  `chart_theme.py` to `returns.py` where it belongs, and moved
  `_analysis_age_caption` from `sidebar.py` to its only consumer
  `result_rendering.py`. Added docstrings to all public functions in the core
  layer. Documented magic numbers in `pipeline.py` and removed an unused `risk`
  import.

The git history still contains all the earlier commits, if you ever want to see
how it evolved.

## Limitations

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
