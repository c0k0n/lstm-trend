"""Methodology page: how the model works and why it was built this way."""

import streamlit as st

from ...constants import (
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    LSTM_DROPOUT,
    LSTM_UNITS,
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
    st.title("📚 Methodology")

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
                "Optimizer",
                "Train / validation split",
                "Early stopping patience",
            ],
            "Value": [
                "60 trading days (adjustable in the sidebar)",
                str(LSTM_UNITS),
                str(LSTM_UNITS // 2),
                f"{LSTM_DROPOUT:.0%}",
                f"{DENSE_UNITS} units, ReLU",
                "Mean absolute error",
                "Adam",
                f"80% train, 10% validation, 10% test",
                f"{EARLY_STOPPING_PATIENCE} epochs without improvement",
            ],
        },
        hide_index=True,
        width="stretch",
    )

    st.header("Evaluation")
    st.markdown(
        """
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
        price) and *moving average* (predict the mean of the last N days).
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
