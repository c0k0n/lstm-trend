"""LSTM Trend — multipage Streamlit entry point.

Run with:  uv run streamlit run streamlit_app.py
"""

import logging
import os

os.environ.setdefault("KERAS_BACKEND", "torch")


def _quiet_third_party_noise() -> None:
    """Silence the one startup warning that comes from a dependency, not us.

    `torch.utils.flop_counter` logs "triton not found" at import time. Triton
    ships only on Linux; flop counting is a profiling utility this app never
    calls. Nothing to fix and nothing lost, but startup noise trains you to
    ignore the console, and a real error is easy to miss in a wall of
    known-irrelevant warnings. Targeted at that one logger, not a blanket
    filter — everything else still surfaces.

    A `torch.jit.script_method` FutureWarning used to be filtered here too, on
    the belief that Keras triggered it. That was wrong, so the filter is gone.
    Traced it: the decorator lives in `torch/utils/mkldnn.py`, inside an
    `lru_cache`'d class factory, so it only runs when MKLDNN conversion is
    actually requested. Measured across every path this app runs — the entry
    point, all seven page modules, a real analysis, and the walk-forward with
    the LSTM enabled — the count is zero with no filter installed.
    `torch.utils.mkldnn.to_mkldnn()` is the only call that fires it, and this
    project never makes it; Keras does not reach it either. See
    `tests/test_no_third_party_warnings.py`, which asserts the warning stays
    absent rather than merely silenced.
    """
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)


_quiet_third_party_noise()

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

pages = list(get_pages().values())

# `streamlit.navigation` is both a re-exported function and a sub-package, and
# type checkers bind the package. The public function is correct at runtime.
st.navigation(pages, position="sidebar").run()  # ty: ignore[call-non-callable]

st.html(_footer_html())
