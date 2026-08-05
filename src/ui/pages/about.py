"""About page: the project story, the stack, and how to run it."""

import streamlit as st

from ...constants import APP_TITLE, GITHUB_URL, LIVE_APP_URL


def render() -> None:
    st.title("🎓 About this project")

    st.markdown(
        f"""
        **{APP_TITLE}** started as a final year project in 2023: could a small
        LSTM network learn anything useful from raw stock closing prices, and
        could that be wrapped up in an app anyone can use without touching
        code?

        The answer the data keeps giving: *sometimes, and with caveats*. The
        LSTM regularly beats 'repeat yesterday', but not always — and that
        honest ambiguity is exactly what makes it an interesting project.

        In 2026 the project was rejuvenated end-to-end:

        - the modelling stack moved to **Keras 3 on a PyTorch backend**,
        - the app was rebuilt as a **multipage Streamlit app** with an
          empirical Findings page, a Methodology page, and polished UI,
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
                "Charts",
                "Environment",
                "Testing",
            ],
            "Choice": [
                "Streamlit 1.61 (multipage, st.navigation)",
                "Keras 3 (LSTM)",
                "PyTorch (torch.cuda optional)",
                "Yahoo Finance via yfinance + pandas",
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
