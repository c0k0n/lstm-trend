"""About page: the project story, the stack, and how to run it."""

import streamlit as st

from ...constants import APP_TITLE, GITHUB_URL, LIVE_APP_URL


def render() -> None:
    st.title("🎓 About this project")

    st.markdown(
        f"""
        **{APP_TITLE}** started as my final year project in 2023: could a small
        LSTM network learn anything useful from raw stock closing prices, and
        could that be wrapped up in an app anyone can use without touching
        code?

        The answer the data keeps giving: *sometimes, and with caveats*. The
        LSTM regularly beats 'repeat yesterday', but not always — and that
        honest ambiguity is exactly what makes it an interesting project.

        The project has kept growing since then:

        - the modelling stack now runs on **Keras 3 with a PyTorch backend**
          (same model code, proper GPU support),
        - the app grew into a **multipage tool**: a Dashboard for forecasting,
          an Analytics page for deep exploratory analysis, a Compare page for
          multi-ticker work, an empirical Findings page, and this page,
        - the analysis is grounded in **baselines** — every LSTM result is
          shown next to 'repeat yesterday' and a moving average,
        - everything runs on **uv** with locked dependencies, unit and E2E
          tests, and GitHub Actions CI.
        """
    )

    st.header("Tech stack")
    st.dataframe(
        {
            "Layer": [
                "App framework",
                "Deep learning",
                "Backend",
                "Data",
                "Analytics",
                "Charts",
                "Environment",
                "Testing",
            ],
            "Choice": [
                "Streamlit 1.61 (multipage, st.navigation)",
                "Keras 3 (LSTM)",
                "PyTorch (torch.cuda optional)",
                "Yahoo Finance via yfinance + pandas",
                "scipy + statsmodels (statistics, ADF test, indicators)",
                "Plotly (interactive dark-theme charts)",
                "uv + uv.lock (Python 3.13)",
                "pytest + Streamlit AppTest, GitHub Actions",
            ],
        },
        hide_index=True,
        width="stretch",
    )

    st.header("Run it yourself")
    st.code(
        """git clone git@github.com:c0k0n/lstm-trend.git
cd lstm-trend
uv sync            # create the environment (takes a while the first time)
./run.sh           # start the app on http://localhost:8501""",
        language="bash",
    )

    col1, col2, col3 = st.columns(3)
    col1.link_button("🔗 Live app", LIVE_APP_URL)
    col2.link_button("📦 GitHub repo", GITHUB_URL)
    col3.link_button("📄 README", f"{GITHUB_URL}#readme")

    st.header("Disclaimer")
    st.markdown(
        """
        This is a learning project. It is not investment advice, and the
        forecasts it produces should not be used to make financial decisions.
        Markets are efficient and noisy; a model trained on closing prices
        alone is not a trading strategy.
        """
    )

    st.header("Was this useful?")
    st.markdown(
        "If you are an examiner or a curious visitor, a quick rating helps me improve it."
    )
    st.feedback("thumbs")


if __name__ == "__main__":
    render()
