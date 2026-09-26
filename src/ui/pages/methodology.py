"""Methodology page: how the model works and why it was built this way."""

import streamlit as st

from ...constants import (
    APP_TITLE,
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    LSTM_DROPOUT,
    LSTM_UNITS,
    MOVING_AVERAGE_WINDOW,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT_RATIO,
    VALIDATION_SPLIT,
)

PIPELINE_DIAGRAM = """
flowchart LR
    A[Yahoo Finance<br/>daily OHLCV] --> B[Log returns<br/>+ calendar covariates]
    B --> C[Scale on the train slice<br/>60-day windows]
    C --> D[Train LSTM<br/>Adam, MAE loss]
    D --> E[Score against naive<br/>skill, not dollars]
    E --> F[Recursive forecast<br/>in return space]
    D -. early stopping .-> D
"""

ARCHITECTURE_DIAGRAM = """
flowchart LR
    A[Input<br/>60 days x 5 features] --> B[LSTM 100<br/>layer norm, dropout]
    B --> C[LSTM 50<br/>layer norm, dropout]
    C --> D[Dense 25<br/>ReLU]
    D --> E[Output<br/>next log return]
"""


def render() -> None:
    st.set_page_config(
        page_title=f"Methodology — {APP_TITLE} | How the LSTM pipeline works",
        page_icon="📚",
    )
    st.title("📚 Methodology")
    st.text(
        "Everything behind the numbers: the data pipeline, the network "
        "architecture, how training and forecasting work, and why each "
        "design choice was made."
    )

    st.header("The problem")
    st.markdown(
        """
        Given the closing prices of a stock over time, predict the next day's
        price, then keep predicting forward. This is a classic *univariate
        time series forecasting* problem — one signal (closing price), one
        target (tomorrow's close).

        Why it is hard: prices are noisy, non-stationary, and influenced by
        things that are not in the data at all (news, sentiment, macro
        events). A good forecast model can still capture short-term patterns
        that simpler rules miss — that is the hypothesis this project tests.
        """
    )

    st.header("Why an LSTM?")
    st.markdown(
        """
        An **LSTM (Long Short-Term Memory)** network is a recurrent neural
        network with a gated internal state. Where a plain feed-forward
        network sees a fixed window of numbers, an LSTM carries a memory of
        what it has seen before and decides what to keep and what to forget —
        which makes it a natural fit for sequences like price history.

        The version used here is deliberately modest: two LSTM layers with
        dropout between them, one dense layer, and a single output. Small
        enough to train on a laptop or a free cloud instance in seconds, yet
        complex enough to learn trends and short-term momentum.
        """
    )

    st.header("The pipeline")
    st.mermaid_chart(PIPELINE_DIAGRAM)

    st.header("Data preparation")
    st.markdown(
        f"""
        Before anything touches the network, the raw download goes through
        six steps:

        1. **Clean & flatten.** Missing rows and the multi-level column layout
           that `yfinance` sometimes returns are normalised away; the result
           is a tidy OHLCV table indexed by trading day.
        2. **Log returns.** The target is the next day's *log return*, not the
           price. A price series carries a unit root — this app's own
           stationarity test will tell you so — and regressing on levels is
           asking the network to memorise a trend.
        3. **Covariates.** Weekday and month are added, cyclically encoded.
           They are the one class of covariate that is knowable in advance:
           when forecasting day H, its weekday is already a fact.
        4. **Scale.** The scaler is fitted on the training slice only, then
           applied to the rest. Fitting it on the whole series would leak the
           test window into training and flatter every number below.
        5. **Window.** Training is built from overlapping windows: with a
           lookback of 60, each sample is the last 60 days of features and the
           answer is the next day's return. Windows slide one day at a time.
        6. **Split.** Windows are split {f"{TRAIN_TEST_SPLIT_RATIO:.0%}"} /
           {f"{1 - TRAIN_TEST_SPLIT_RATIO:.0%}"} into train and test. The test
           slice is never seen during training — it exists only to score the
           model afterwards.
        """
    )

    st.header("Training details")
    st.markdown(
        f"""
        - **Optimizer / loss:** Adam with its default learning rate, training
          against mean absolute error — the same unit as the MAE you see on
          the dashboard.
        - **Validation:** {f"{VALIDATION_SPLIT:.0%}"} of the training windows
          are held out to monitor `val_loss`.
        - **Early stopping:** training stops if `val_loss` does not improve
          for {EARLY_STOPPING_PATIENCE} epochs, restoring the best weights —
          so more epochs are only used when they genuinely help.
        - **Reproducibility:** everything is seeded ({RANDOM_SEED}) so the
          same settings give the same run.
        - **Device:** Keras 3 on a PyTorch backend, which uses an NVIDIA GPU
          when PyTorch was installed with CUDA. Plain PyPI PyTorch on Windows
          is a CPU-only build, so the project pulls torch from PyTorch's own
          CUDA index — see the README. Where that is not available it falls
          back to CPU and says so in the sidebar.
        """
    )

    st.header("The architecture")
    st.mermaid_chart(ARCHITECTURE_DIAGRAM)

    st.header("Hyperparameters")
    st.dataframe(
        {
            "Setting": [
                "Lookback window",
                "Input features per day",
                "LSTM units (layer 1)",
                "LSTM units (layer 2)",
                "Dropout",
                "Dense layer",
                "Loss",
                "Optimizer / learning rate",
                "Train / validation split",
                "Early stopping patience",
                "Random seed",
                "Baseline moving-average window",
            ],
            "Value": [
                "60 trading days (adjustable in the sidebar)",
                "5 — log return plus cyclically encoded weekday and month",
                str(LSTM_UNITS),
                str(LSTM_UNITS // 2),
                f"{LSTM_DROPOUT:.0%}",
                f"{DENSE_UNITS} units, ReLU",
                "Mean absolute error",
                "Adam, default (1e-3)",
                f"80% train / 20% test (validation = 10% of the train slice)",
                f"{EARLY_STOPPING_PATIENCE} epochs without improvement",
                str(RANDOM_SEED),
                str(MOVING_AVERAGE_WINDOW),
            ],
        },
        hide_index=True,
        width="stretch",
    )

    st.header("Evaluation")
    st.markdown(
        f"""
        The model is scored on the **test window** — the last
        {f"{1 - TRAIN_TEST_SPLIT_RATIO:.0%}"} of the data, which it never saw
        during training.

        The metric that matters is **skill**: `1 − MAE(model) / MAE(naive)`.
        Zero means no better than repeating yesterday's price; negative means
        worse. It is used because it is the only metric here that transfers —
        "MAE $2.14" means nothing on another ticker or another year, "0.93×
        naive error" does.

        | Metric | What it measures | Standing |
        |---|---|---|
        | Skill vs naive | Error relative to repeating yesterday | **The headline** |
        | MSE / RMSE | Average squared error | Reported, penalises big misses |
        | MAE | Average absolute error in dollars | Reported, not comparable across tickers |
        | MAPE | Average percentage error | Decoration — asymmetric, unstable near zero |
        | R² | Variance explained | Flattered by any trend; do not read much into it |

        The LSTM is compared against two baselines: *naive* (predict
        yesterday's price) and *moving average* (predict the mean of the last
        {MOVING_AVERAGE_WINDOW} days). They are the best thing in this app and
        most forecasting projects quietly skip them.

        :orange[One test window cannot tell skill from luck.] A single split is
        one sample. The 🧪 **Evidence** page rolls the same test across up to
        20 tickers and 300 windows and reports whether anything survives.
        """
    )

    st.header("How the forecast is generated")
    st.markdown(
        """
        The model predicts **one day at a time**, in log-return space. Each
        predicted return is converted to a price, appended to the window, and
        the oldest day drops off — rolling the forecast forward as far as you
        asked. Calendar columns simply continue, because next Tuesday is
        already a known fact.

        **Why the line is nearly flat.** This is the part that surprises
        people. The model is trained with mean absolute error, and the
        error-minimising prediction for a near-random daily return is close to
        zero — so the forecast hugs the last close. That is not a bug and it
        is not a broken model: it is what an honest forecast of a series with
        no predictable signal actually looks like. A wandering, confident
        line is what an *overfitted* model looks like. The Evidence page is
        where "is there any signal at all" gets answered properly.

        Two other consequences worth knowing:

        - **Errors compound.** A slightly-off prediction feeds the next one,
          so the 5-day forecast is far more believable than the 90-day one.
        - **Business days only.** Forecast dates are generated on trading
          days, because that is when prices actually move.
        """
    )

    st.header("Design trade-offs")
    st.markdown(
        """
        | Choice | Why | What it costs |
        |---|---|---|
        | Small two-layer LSTM | Trains in seconds on a laptop or free cloud instance | Less capacity than deeper/transformer models |
        | Log returns, not price levels | Prices carry a unit root — this app's own stationarity test says so | The forecast then looks flat, as above |
        | Layer normalisation after each LSTM | Stabilises sequence training far better than dropout alone | Slightly slower per epoch |
        | Calendar covariates only | Weekday and month are known in advance, so they cannot leak | Ignores volume, news, sentiment, fundamentals |
        | Volume deliberately excluded | Tomorrow's volume does not exist yet | Loses a genuinely informative signal |
        | Scaling fitted on the train slice only | Fitting on the whole series leaks the test window into training | None |
        | Baselines always shown | A complex model is only worth it if it beats simple ones | None — this is the part that keeps the project honest |
        | Keras 3 + PyTorch | One model definition, GPU support without hacks | The default Windows/PyPI install is CPU-only; CUDA needs the PyTorch index |
        """
    )

    st.header("Beyond the dashboard")
    st.markdown(
        """
        The dashboard shows one ticker over one test window. Three things exist
        to answer the question that view cannot:

        - 🧪 **Evidence** — walk-forward testing. Instead of one 80/20 split,
          the forecast origin rolls through history and every window is
          scored, pooled across as many tickers as you enter. It also runs
          **LightGBM** challengers, an optional **panel model** (one model
          trained across all tickers at once), and optional **market
          context** (VIX, broad market, short rate, sector ETF).
        - **Conformal bands** — the forecast is drawn inside a range
          calibrated on the **LSTM's own** out-of-sample errors, widened along
          the path with √time, with realised coverage shown beside it so the
          range can be checked rather than trusted. It also *breathes*: the
          width is scaled by how volatile the market is right now versus how
          volatile it was across the calibration windows, measured from each
          day's high-low range. A constant band is really the average mistake
          — too wide in calm markets, too narrow when things break. This is
          the part of the app that carries real information, so it is drawn by
          default rather than offered as an option.
        - **The candle is deliberately held back from the point forecast.**
          Overnight gap, intraday range and Parkinson/Garman-Klass range
          volatility are implemented and available, but measured over 200
          windows they made the forecast *worse* (helping on 6 of 20 tickers),
          so they are off by default. Range-based volatility belongs to the
          band, not to the point.
        - **Forecast journal** — every forecast is written to `journal.db`
          with the price at the moment it was made, then graded when the date
          passes. This is what lets the app be wrong in public.
        """
    )

    st.header("References")
    st.markdown(
        """
        - Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.*
          Neural Computation 9(8).
        - TensorFlow Team. (2026). *Keras 3: multi-backend deep learning.*
          https://keras.io
        - PyTorch contributors. (2026). https://pytorch.org
        - Streamlit. (2026). *App dependencies for Community Cloud.*
          https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
        - Yahoo Finance via `yfinance`. https://pypi.org/project/yfinance/
        """
    )


if __name__ == "__main__":
    render()
