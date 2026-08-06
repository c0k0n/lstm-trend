"""App-level light/dark theme handling.

Streamlit's Settings menu can switch the whole app between the light and
dark themes defined in .streamlit/config.toml — that covers all native
widgets. The Python side never learns which one the user picked (the theme
lives in the browser), so charts need their own signal: a sidebar toggle
that flips the Plotly theme and re-tints the app chrome with CSS.
"""

import streamlit as st

from . import charts

_DARK_KEY = "theme_dark"

# Light-mode re-tint of the app chrome (backgrounds + text). Widget surfaces
# follow Streamlit's own theme tokens, so pairing this toggle with the
# Settings-menu theme gives the cleanest result.
_LIGHT_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #FFFFFF; }
[data-testid="stHeader"] { background: rgba(255,255,255,0.95); }
[data-testid="stSidebar"] { background-color: #F2F4F7; }
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] li { color: #31333F; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 { color: #17181C; }
[data-testid="stCaptionContainer"] { color: #6A6E79; }
[data-testid="stMetricValue"] { color: #17181C; }
[data-testid="stMetricLabel"] { color: #6A6E79; }
[data-testid="stMetric"] { background-color: #F8FAFC; }
[data-testid="stCodeBlock"] pre { background-color: #F4F6F8; color: #31333F; }
[data-testid="stExpander"] { border-color: #E6EAF0; }
[data-testid="stExpander"] summary { color: #31333F; }
hr { border-color: #E6EAF0; }
</style>
"""


def render_theme_toggle() -> bool:
    """Sidebar toggle that switches charts and the app chrome between themes."""
    st.session_state.setdefault(_DARK_KEY, True)
    if "theme_param_applied" not in st.session_state:
        st.session_state["theme_param_applied"] = True
        theme = st.query_params.get("theme")
        if theme in {"light", "dark"}:
            st.session_state[_DARK_KEY] = theme == "dark"

    dark = st.sidebar.toggle(
        "🌙 Dark mode",
        key=_DARK_KEY,
        help="Switches the charts and app colours. Streamlit's Settings menu "
        "has an equivalent switch for widget theming.",
    )
    charts.set_dark(dark)
    if not dark:
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)
    return dark
