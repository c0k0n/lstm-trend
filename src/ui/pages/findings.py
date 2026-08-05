"""Findings page: metrics, baselines and the empirical story."""

import pandas as pd
import streamlit as st

from ...core.baselines import MOVING_AVERAGE, NAIVE
from ...core.metrics import METRIC_NAMES
from .. import charts
from ..components import get_analysis

METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "r2": "R²",
}


def _metrics_table(result) -> pd.DataFrame:
    rows = [{"Method": "LSTM", **result.lstm_metrics}]
    for name, metrics in result.baselines.items():
        rows.append({"Method": name, **metrics})
    return pd.DataFrame(rows)


def render() -> None:
    st.title("🔬 Findings")

    result = get_analysis()
    if result is None:
        st.info(
            "No analysis yet. Run one on the **Dashboard** first — the findings "
            "page then shows everything without retraining."
        )
        if st.button("Go to Dashboard", type="primary"):
            st.switch_page("src/ui/pages/dashboard.py")
        return

    st.caption(
        f"Based on the last analysis of **{result.symbol}** — "
        f"{len(result.data)} daily rows, "
        f"lookback {result.params['sequence_length']} days."
    )

    st.subheader("How the models compare")
    table = _metrics_table(result)

    def fmt(v: float, metric: str) -> str:
        if metric == "r2":
            return f"{v:.4f}"
        if metric == "mape":
            return f"{v:.2%}"
        return f"${v:,.2f}"

    display = table.rename(columns=METRIC_LABELS)
    for metric in METRIC_NAMES:
        display[METRIC_LABELS[metric]] = display[METRIC_LABELS[metric]].apply(
            lambda v, m=metric: fmt(v, m)
        )

    st.dataframe(display, hide_index=True, width="stretch")

    st.plotly_chart(charts.plot_metric_bars(result), width="stretch")
    st.plotly_chart(charts.plot_baseline_comparison(result), width="stretch")

    st.subheader("What this actually means")
    lstm_rmse = result.lstm_metrics["rmse"]
    naive_rmse = result.baselines[NAIVE]["rmse"]
    ma_rmse = result.baselines[MOVING_AVERAGE]["rmse"]

    if lstm_rmse < min(naive_rmse, ma_rmse):
        verdict = (
            f"The LSTM beat both baselines on this test window (RMSE "
            f"${lstm_rmse:,.2f} vs ${naive_rmse:,.2f} naive and "
            f"${ma_rmse:,.2f} moving average), so it did learn something "
            f"beyond 'repeat yesterday'."
        )
    else:
        verdict = (
            f"The LSTM did **not** beat the simpler baselines here "
            f"(RMSE ${lstm_rmse:,.2f} vs ${min(naive_rmse, ma_rmse):,.2f}). "
            f"That is a genuinely useful result too: it is a reminder that "
            f"stock prices are noisy and a simple rule is a strong opponent."
        )
    st.markdown(verdict)

    with st.expander("Read the caveats"):
        st.markdown(
            """
            - One run, one ticker, one test window. These numbers describe the
              last 20% of the downloaded history for this ticker and nothing more.
            - The model only sees closing prices — no volume, news, fundamentals,
              or market context.
            - Forecasts are iterative: each predicted day feeds the next window,
              so errors compound over the horizon.
            - Different date ranges and tickers will produce different results;
              that variance is part of the story.
            """
        )


if __name__ == "__main__":
    render()
