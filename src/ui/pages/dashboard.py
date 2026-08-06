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

_WELCOME_DIALOG = """
**Three things to try:**

1. **Run a forecast** — pick a ticker in the sidebar and press *Run analysis*.
2. **Deep-dive the data** — open *Analytics* or *Compare* for the same ticker.
3. **Read the verdict** — *Findings* honestly compares the LSTM against the
   simple baselines.
"""


def _jump_menu() -> None:
    """Menu button that jumps between pages."""
    from .nav import get_pages

    pages = get_pages()
    choices = {
        "📊 Analytics": pages["analytics"],
        "⚖️ Compare": pages["compare"],
        "🔬 Findings": pages["findings"],
        "📚 Methodology": pages["methodology"],
    }
    choice = st.menu_button(
        "More",
        options=list(choices),
        key="dash_menu",
        help="Jump straight to another page of the app.",
    )
    if choice is not None:
        st.session_state["dash_menu"] = None
        st.switch_page(choices[choice])


def _page_links() -> None:
    """Quick-navigation cards for first-time visitors."""
    from .nav import get_pages

    pages = get_pages()
    st.markdown("#### 📖 New to the project? Jump to a page")
    with st.container(horizontal=True):
        st.page_link(pages["methodology"], label="Methodology", icon="📚")
        st.page_link(pages["analytics"], label="Analytics", icon="📊")
        st.page_link(pages["compare"], label="Compare", icon="⚖️")
        st.page_link(pages["findings"], label="Findings", icon="🔬")


def _show_welcome_dialog() -> None:
    """One-time-per-session welcome dialog."""
    if st.session_state.get("welcome_seen"):
        return

    @st.dialog("👋 Welcome to LSTM Trend", width="small")
    def _welcome() -> None:
        st.markdown(_WELCOME_DIALOG)
        if st.button("Start exploring", type="primary", key="welcome_gotit"):
            st.session_state["welcome_seen"] = True
            st.rerun()

    _welcome()


def _bottom_bar(params: dict) -> bool:
    """Pinned bottom bar with a one-click rerun of the analysis."""
    with st.bottom:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.caption(
                f"Current settings: **{params['symbol']}** · "
                f"{params['start_date']} → {params['end_date']} · "
                f"horizon {params['future_steps']} days"
            )
        with col2:
            return st.button("🚀 Run analysis", type="primary", key="run_button_bottom")


def render() -> None:
    st.set_page_config(
        page_title=f"Dashboard — {APP_TITLE} | LSTM stock price forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _show_welcome_dialog()

    col_title, col_menu = st.columns([5, 1])
    with col_title:
        render_hero()
    with col_menu:
        st.space("large")
        _jump_menu()
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
    _page_links()
    st.space("medium")

    params = render_sidebar_config()

    sidebar_clicked = st.sidebar.button(
        "🚀 Run analysis", type="primary", width="stretch", key="run_button"
    )
    bottom_clicked = _bottom_bar(params)

    if not (sidebar_clicked or bottom_clicked):
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
