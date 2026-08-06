"""LSTM Trend — multipage Streamlit entry point.

Run with:  ./run.sh   (or: uv run streamlit run streamlit_app.py)
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import streamlit as st  # noqa: E402

from src.constants import APP_TITLE, GITHUB_URL  # noqa: E402
from src.ui.pages.nav import get_pages  # noqa: E402


def _footer_html() -> str:
    """App-wide footer; CSS vars track the theme, with dark fallbacks."""
    return f"""
<footer
  style="
    margin-top: 2.5rem;
    padding: 0.9rem 1.25rem;
    border: 1px solid var(--border-color, #3D4452);
    border-radius: var(--radius-md, 0.5rem);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    color: var(--text-color, #9AA1AD);
    font-size: 0.8rem;
  "
>
  <span style="opacity: 0.8">
    LSTM Trend — an empirical study of LSTM forecasting, built as a final year
    project. Educational tool, <strong>not investment advice</strong>.
  </span>
  <a
    href="{GITHUB_URL}"
    target="_blank"
    rel="noopener noreferrer"
    style="color: var(--link-color, #4FB477); text-decoration: none"
  >
    Source on GitHub ↗
  </a>
</footer>
"""


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://docs.streamlit.io/",
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": (
            "LSTM Trend — an empirical study of LSTM forecasting on stock "
            "prices, built as a final year project. Educational tool, not "
            "investment advice."
        ),
    },
)

st.logo(
    "assets/logo.svg",
    icon_image="assets/logo.svg",
    link=GITHUB_URL,
)

pages = list(get_pages().values())

st.navigation(pages, position="sidebar").run()

st.html(_footer_html())
