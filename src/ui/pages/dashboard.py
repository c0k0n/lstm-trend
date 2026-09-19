"""Dashboard page: run the analysis and explore the results."""

import datetime
from typing import Any

import streamlit as st

from ...constants import APP_TITLE, SUGGESTED_SYMBOLS
from ...core.pipeline import AnalysisResult
from ..sidebar import (
    get_analysis,
    render_hero,
    render_sidebar_config,
    run_analysis_with_ui,
)
from ..result_rendering import render_result

_GETTING_STARTED = """
1. **Pick a ticker** — use the quick picks or type any symbol in the sidebar.
2. **Choose a date range** — more history means more training data, but longer
   runs; the default (2020 → today) is a good balance.
3. **Tune the model** — lookback window, horizon, epochs and batch size live in
   *Model parameters*. The defaults are sane for a first run.
4. **Hit *Run analysis*** — the pipeline downloads data, trains the LSTM, and
   scores it against *repeat yesterday* and a 20-day moving average.
"""


def _apply_query_params() -> None:
    """Pre-fill the sidebar widgets from URL query params (once per session)."""
    if "query_params_applied" in st.session_state:
        return
    st.session_state["query_params_applied"] = True
    qp = st.query_params

    ticker = qp.get("ticker")
    if ticker:
        ticker = str(ticker).upper()
        st.session_state["ticker_pills"] = (
            ticker if ticker in SUGGESTED_SYMBOLS else "Custom…"
        )
        st.session_state["ticker_custom"] = ticker

    start = qp.get("start")
    if start:
        try:
            st.session_state["start_date"] = datetime.date.fromisoformat(str(start))
        except ValueError:
            pass

    end = qp.get("end")
    if end:
        try:
            st.session_state["end_date"] = datetime.date.fromisoformat(str(end))
        except ValueError:
            pass

    horizon = qp.get("horizon")
    if horizon and str(horizon).isdigit():
        value = int(str(horizon))
        if value in (5, 15, 30):
            st.session_state["horizon_preset"] = value
        else:
            st.session_state["horizon_preset"] = "Custom…"
            st.session_state["fut_steps"] = min(90, max(5, value))


def _write_query_params(params: dict[str, Any]) -> None:
    """Persist the completed run as a shareable deep link."""
    st.query_params["ticker"] = params["symbol"]
    st.query_params["start"] = str(params["start_date"])
    st.query_params["end"] = str(params["end_date"])
    st.query_params["horizon"] = str(params["future_steps"])


def render() -> None:
    st.set_page_config(
        page_title=f"Dashboard — {APP_TITLE} | LSTM stock price forecasting",
        page_icon="📈",
    )
    _apply_query_params()

    render_hero()
    st.space("small")

    with st.container(border=True):
        st.markdown("#### 👋 Welcome — how this dashboard works")
        st.markdown(_GETTING_STARTED)
        st.markdown(
            "**You'll get back:** a market snapshot, training curves, the test "
            "window actual-vs-predicted, the forecast table (with CSV download), "
            "and the same numbers for the two baselines."
        )

    st.space("medium")

    params = render_sidebar_config()

    sidebar_clicked = st.sidebar.button(
        "🚀 Run analysis", type="primary", width="stretch", key="run_button"
    )
    st.sidebar.caption(
        f"Current settings: **{params['symbol']}** · "
        f"{params['start_date']} → {params['end_date']} · "
        f"horizon {params['future_steps']} days"
    )

    if not sidebar_clicked:
        existing = get_analysis()
        if existing is not None:
            st.info(
                f"Showing the last analysis for **{existing.symbol}**. "
                "Adjust the settings and press *Run analysis* to start a new one."
            )
            render_result(existing)
        else:
            st.info(
                "Pick a ticker and date range in the sidebar, then press "
                "**🚀 Run analysis**. The pipeline downloads data, trains an "
                "LSTM, and compares it against simple baselines."
            )
        return

    result: AnalysisResult | None = run_analysis_with_ui(params)
    if result is None:
        return

    _write_query_params(params)
    render_result(result)


if __name__ == "__main__":
    render()
