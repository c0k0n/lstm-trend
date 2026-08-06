"""Dashboard page: run the analysis and explore the results."""

import streamlit as st

from ...constants import APP_TITLE
from ...core.pipeline import AnalysisResult
from ..components import (
    get_analysis,
    render_hero,
    render_result,
    render_sidebar_config,
    run_analysis_with_ui,
)

_GETTING_STARTED = """
1. **Pick a ticker** — use the quick picks or type any symbol in the sidebar.
2. **Choose a date range** — more history means more training data, but longer
   runs; the default (2020 → today) is a good balance.
3. **Tune the model** — lookback window, horizon, epochs and batch size live in
   *Model parameters*. The defaults are sane for a first run.
4. **Hit *Run analysis*** — the pipeline downloads data, trains the LSTM, and
   scores it against *repeat yesterday* and a 20-day moving average.
"""


def render() -> None:
    st.set_page_config(
        page_title=f"Dashboard — {APP_TITLE} | LSTM stock price forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_hero()

    with st.container(border=True):
        st.markdown("#### 👋 Welcome — how this dashboard works")
        st.markdown(_GETTING_STARTED)
        st.markdown(
            "**You'll get back:** a market snapshot, training curves, the test "
            "window actual-vs-predicted, the forecast table (with CSV download), "
            "and the same numbers for the two baselines."
        )

    with st.expander("📖 New to the project? Start here"):
        st.markdown(
            """
            - **Methodology** explains the pipeline, the architecture and every
              hyperparameter behind the numbers.
            - **Analytics** deep-dives into the same ticker: returns, risk
              metrics, seasonality, technical indicators.
            - **Compare** puts several tickers side by side.
            - **Findings** tells you, honestly, whether the LSTM beat the
              simple baselines — and when it didn't.
            """
        )

    params = render_sidebar_config()

    if not st.sidebar.button(
        "🚀 Run analysis", type="primary", width="stretch", key="run_button"
    ):
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

    render_result(result)


if __name__ == "__main__":
    render()
