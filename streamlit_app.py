"""LSTM Trend — multipage Streamlit entry point.

Run with:  ./run.sh   (or: uv run streamlit run streamlit_app.py)
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import streamlit as st  # noqa: E402

from src.constants import APP_TITLE, GITHUB_URL  # noqa: E402
from src.ui.pages.nav import get_pages  # noqa: E402
from src.ui.theme import render_theme_toggle  # noqa: E402

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(
    "assets/logo.svg",
    icon_image="assets/logo.svg",
    link=GITHUB_URL,
)

render_theme_toggle()

pages = list(get_pages().values())

st.navigation(pages, position="sidebar").run()
