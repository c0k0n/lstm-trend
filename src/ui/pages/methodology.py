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
    A[Yahoo Finance<br/>daily OHLCV] --> B[Clean & flatten columns]
    B --> C[MinMax scale<br/>closing prices]
    C --> D[Sliding windows<br/>60 past days -> next day]
    D --> E{Train / test split<br/>80% / 20%}
    E --> F[Train LSTM<br/>Adam, MAE loss]
    F --> G[Evaluate on test window<br/>MSE RMSE MAE MAPE R2]
    G --> H[Iterative forecast<br/>next business days]
    F -. early stopping .-> F
"""

ARCHITECTURE_DIAGRAM = """
flowchart LR
    A[Input<br/>60 x 1] --> B[LSTM<br/>100 units]
    B --> C[Dropout 20%]
    C --> D[LSTM<br/>50 units]
    D --> E[Dropout 20%]
    E --> F[Dense 25<br/>ReLU]
    F --> G[Output<br/>1 value]
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
        four steps:

        1. **Clean & flatten.** Missing rows and the multi-level column layout
           that `yfinance` sometimes returns are normalised away; the result
           is a tidy OHLCV table indexed by trading day.
        2. **Scale.** Closing prices are min-max scaled to [0, 1]. Neural
           networks train far more reliably on bounded inputs than on raw
           dollar values.
        3. **Window.** Training is built from overlapping windows: with a
           lookback of 60, each sample is the last 60 closing prices and the
           answer is the price on day 61. Windows slide one day at a time, so
           the model sees a long history of small price moves.
        4. **Split.** Windows are split {f"{TRAIN_TEST_SPLIT_RATIO:.0%}"} /
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
        - **Device:** Keras 3 on a PyTorch backend — the app uses an NVIDIA
          GPU when available and silently falls back to CPU otherwise.
        """
    )

    st.header("The architecture")
    st.mermaid_chart(ARCHITECTURE_DIAGRAM)

    st.header("Hyperparameters")
    st.dataframe(
        {
            "Setting": [
                "Lookback window",
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
        The model is scored on the **test window** — the last 20% of the data,
        which it never saw during training. Five metrics are reported, all on
        the original price scale so they can be read directly:

        | Metric | What it measures |
        |---|---|
        | MSE / RMSE | Average squared / root squared error (penalises big misses) |
        | MAE | Average absolute error in dollars |
        | MAPE | Average percentage error |
        | R² | How much of the price variance the model explains (1 = perfect) |

        Because a complex model is only worth it if it beats a simple one, the
        LSTM is compared against two baselines: *naive* (predict yesterday's
        price) and *moving average* (predict the mean of the last
        {MOVING_AVERAGE_WINDOW} days).
        """
    )

    st.header("How the forecast is generated")
    st.markdown(
        """
        The model predicts **one day at a time**. Each prediction is appended
        to the window, the oldest day drops off, and the model predicts again
        — rolling the forecast forward as far as you asked.

        Two consequences worth knowing:

        - **Errors compound.** A slightly-off prediction feeds the next one,
          so the 5-day forecast is far more believable than the 90-day one.
          The chart's grey zone *is* the point: short horizons hug the recent
          trend, long horizons drift.
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
        | Dropout between layers | Regularises against overfitting a noisy signal | Slightly slower convergence |
        | Univariate (close only) | Keeps the experiment honest and the data pipeline simple | Ignores volume, news, sentiment, fundamentals |
        | MinMax scaling | Bounds inputs for stable training | Assumes a finite range — fine for a fixed window |
        | Baselines always shown | A complex model is only worth it if it beats simple ones | None — this is the part that keeps the project honest |
        | Keras 3 + PyTorch | One model definition, GPU support without hacks | PyTorch wheels make the first install heavy |
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
