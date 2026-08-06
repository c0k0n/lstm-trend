"""LSTM Trend — multipage Streamlit entry point.

Run with:  ./run.sh   (or: uv run streamlit run streamlit_app.py)
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import streamlit as st  # noqa: E402

from src.constants import APP_TITLE, GITHUB_URL  # noqa: E402
from src.ui.pages import (  # noqa: E402
    about,
    analytics,
    compare,
    dashboard,
    findings,
    methodology,
)

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

pages = [
    st.Page(
        dashboard.render,
        title="Dashboard",
        icon="📈",
        url_path="dashboard",
        default=True,
    ),
    st.Page(analytics.render, title="Analytics", icon="📊", url_path="analytics"),
    st.Page(compare.render, title="Compare", icon="⚖️", url_path="compare"),
    st.Page(findings.render, title="Findings", icon="🔬", url_path="findings"),
    st.Page(methodology.render, title="Methodology", icon="📚", url_path="methodology"),
    st.Page(about.render, title="About", icon="🎓", url_path="about"),
]

st.navigation(pages, position="sidebar").run()
