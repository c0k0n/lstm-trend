"""Findings page: metrics, baselines and the empirical story."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from ...core.baselines import MOVING_AVERAGE, NAIVE
from ...core.metrics import METRIC_NAMES
from ...constants import APP_TITLE
from .. import dashboard_charts as charts
from ..sidebar import get_analysis

if TYPE_CHECKING:
    from ...core.pipeline import AnalysisResult

METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE",
    "r2": "R²",
}


_CHARTS_GUIDE = """
- **Test RMSE by method** — a bar chart of the same three numbers;
  the shortest bar wins.
- **LSTM vs baselines** — the three predicted curves over the test
  window, overlaid on the actual prices. Look at where the curves
  hug the actual line: that is what a good forecast looks like.
  The moving average lags by construction — it is always a step
  behind.
"""


def _charts_help_dialog() -> None:
    """Modal explainer for the two charts above the verdict."""

    @st.dialog("How to read the charts", width="small")
    def _explain() -> None:
        st.markdown(_CHARTS_GUIDE)

    if st.button("💡 How to read the charts", key="charts_help_btn"):
        _explain()


def _verdict_stream(text: str) -> Iterator[str]:
    """Yield the verdict word by word, so st.write_stream can type it out."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def _render_verdict(text: str, result: AnalysisResult) -> None:
    """Stream the verdict once per result, then show it statically."""
    stream_key = f"verdict_streamed_{result.symbol}"
    if st.session_state.get(stream_key):
        st.markdown(text)
    else:
        st.write_stream(_verdict_stream(text))
        st.session_state[stream_key] = True


def _metrics_table(result: AnalysisResult) -> pd.DataFrame:
    """Build a DataFrame of metrics for LSTM and baseline methods."""
    rows = [{"Method": "LSTM", **result.lstm_metrics}]
    for name, metrics in result.baselines.items():
        rows.append({"Method": name, **metrics})
    return pd.DataFrame(rows)


def _format_metric(value: float, metric: str) -> str:
    """Render one metric the way it reads best: dollars, percent, or plain."""
    if metric == "r2":
        return f"{value:.4f}"
    if metric == "mape":
        return f"{value:.2%}"
    return f"${value:,.2f}"


def _metric_formatter(metric: str) -> Callable[[float], str]:
    """Bind a metric name to its formatter, for use with ``DataFrame.apply``."""
    return partial(_format_metric, metric=metric)


def render() -> None:
    st.set_page_config(
        page_title=f"Findings — {APP_TITLE} | LSTM vs baselines",
        page_icon="🔬",
    )
    st.title("🔬 Findings")
    st.text(
        "The empirical verdict: how the LSTM's test-window errors compare "
        "with 'repeat yesterday' and a 20-day moving average — on the last "
        "analysis you ran."
    )

    result = get_analysis()
    if result is None:
        st.info(
            "No analysis yet. Run one on the **Dashboard** first — the findings "
            "page then shows everything without retraining."
        )
        if st.button("Go to Dashboard", type="primary"):
            from .nav import get_pages

            st.switch_page(get_pages()["dashboard"])
        return

    st.caption(
        f"Based on the last analysis of **{result.symbol}** — "
        f"{len(result.data)} daily rows, "
        f"lookback {result.params['sequence_length']} days, "
        f"epochs {result.params['epochs']}, batch {result.params['batch_size']}."
    )
    st.space("small")

    test_data = result.data.iloc[result.test_start_index :]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Test window start",
        str(test_data.index[0].date()),
        help="First day of the held-out test period.",
    )
    col2.metric(
        "Test window end",
        str(test_data.index[-1].date()),
        help="Last day of the held-out test period.",
    )
    col3.metric(
        "Test days",
        f"{len(test_data):,}",
        help="Trading days the model was scored on (never seen in training).",
    )
    col4.metric(
        "Test price range",
        f"${test_data['Close'].min():,.2f} – ${test_data['Close'].max():,.2f}",
        help="Where the price moved during the test window — context for the errors.",
    )

    st.subheader("How the models compare")
    table = _metrics_table(result)

    display = table.rename(columns=METRIC_LABELS)
    for metric in METRIC_NAMES:
        display[METRIC_LABELS[metric]] = display[METRIC_LABELS[metric]].apply(
            _metric_formatter(metric)
        )

    st.dataframe(display, hide_index=True, width="stretch")

    st.plotly_chart(charts.plot_metric_bars(result), width="stretch")
    st.plotly_chart(charts.plot_baseline_comparison(result), width="stretch")
    st.space("small")
    _charts_help_dialog()
    st.space("small")

    st.subheader("What this actually means")
    lstm_rmse = result.lstm_metrics["rmse"]
    naive_rmse = result.baselines[NAIVE]["rmse"]
    ma_rmse = result.baselines[MOVING_AVERAGE]["rmse"]

    if lstm_rmse < min(naive_rmse, ma_rmse):
        verdict = (
            f"The LSTM beat both baselines on this test window (RMSE "
            f"**${lstm_rmse:,.2f}** vs ${naive_rmse:,.2f} naive and "
            f"${ma_rmse:,.2f} moving average), so it did learn something "
            f"beyond 'repeat yesterday'."
        )
        icon = "✅"
        verdict_badge = ("LSTM wins this test window", "green")
    else:
        verdict = (
            f"The LSTM did **not** beat the simpler baselines here "
            f"(RMSE **${lstm_rmse:,.2f}** vs ${min(naive_rmse, ma_rmse):,.2f}). "
            f"That is a genuinely useful result too: it is a reminder that "
            f"stock prices are noisy and a simple rule is a strong opponent."
        )
        icon = "⚠️"
        verdict_badge = ("Baselines win this time", "orange")
        with st.container(border=True):
            st.markdown(f"### {icon} Verdict")
            st.badge(verdict_badge[0], color=verdict_badge[1])
            _render_verdict(verdict, result)
            feedback = st.feedback("thumbs", key="verdict_feedback")
            if feedback is not None:
                st.session_state["verdict_feedback_value"] = feedback

    with st.expander("How to read the numbers"):
        st.markdown(
            """
            | Metric | Lower is better? | What it actually tells you |
            |---|---|---|
            | **RMSE** | Yes | Average error in dollars, with big misses punished. The headline number for this comparison. |
            | **MAE** | Yes | Average absolute error in dollars — easier to feel, ignores outliers. |
            | **MAPE** | Yes | The error as a percentage of price, so different stocks can be compared. |
            | **R²** | — | How much of the price variance the model explains. High on trending stocks, low on choppy ones. |

            RMSE is the comparison metric because it is sensitive to the same
            kind of error you care about in a forecast: consistently missing
            the next move by a lot.
            """
        )

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
