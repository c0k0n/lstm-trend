"""Dashboard page: run the analysis and explore the results."""

import streamlit as st

from ...core.pipeline import AnalysisResult
from .. import charts
from ..components import (
    get_analysis,
    render_forecast_table,
    render_hero,
    render_metrics,
    render_result,
    render_sidebar_config,
    run_analysis_with_ui,
)


def render() -> None:
    render_hero()

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
