"""About page: the project story, the stack, and how to run it."""

import streamlit as st

from ...constants import APP_TITLE, GITHUB_URL, LIVE_APP_URL


def render() -> None:
    st.set_page_config(
        page_title=f"About — {APP_TITLE} | Project story, stack and FAQ",
        page_icon="🎓",
    )
    st.title("🎓 About this project")
    st.text(
        "The story, the stack, a tour of the app and answers to the questions "
        "people actually ask about a stock-prediction project."
    )
    st.space("medium")

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
        - everything runs on **uv** with locked dependencies.
        """
    )

    st.header("A tour of the app")
    st.markdown(
        """
        | Page | What you'll find there |
        |---|---|
        | **📈 Dashboard** | Run the pipeline: data → train → evaluate → forecast. Market snapshot, training curves, forecast table with CSV download. |
        | **📊 Analytics** | Deep EDA of any ticker: returns, risk metrics (Sharpe, drawdowns, VaR), seasonality, stationarity, technical indicators, volume analysis. |
        | **⚖️ Compare** | Several tickers side by side: normalized prices, correlations, drawdowns, risk-vs-return scatter, metrics table. |
        | **🔬 Findings** | The honest verdict: LSTM vs naive vs moving average on the last analysis, with test-window context. |
        | **📚 Methodology** | The full technical story: pipeline, architecture, hyperparameters, design trade-offs, references. |
        | **🎓 About** | You are here. |
        """
    )

    st.header("Frequently asked questions")
    with st.expander("Is this investment advice?"):
        st.markdown(
            "No. It is a learning project. Markets are noisy and driven by "
            "news, sentiment and policy that no price history can tell a "
            "model. Treat every forecast as an experiment, not a tip."
        )
    with st.expander("Why an LSTM, and not a Transformer or XGBoost?"):
        st.markdown(
            "An LSTM is the classic recurrent architecture for sequences, it "
            "trains fast on tiny data, and it keeps the project readable for "
            "an examiner. Transformers would overfit a few thousand prices; "
            "the trade-offs table on the Methodology page says more."
        )
    with st.expander("Why does the app retrain on every run?"):
        st.markdown(
            "Nothing is persisted between sessions — every run downloads "
            "fresh data and trains from scratch. That keeps the app simple "
            "and honest (no stale model), at the cost of a wait each time. "
            "Saving models is on the ideas list."
        )
    with st.expander("Why show baselines? Doesn't that make the LSTM look bad?"):
        st.markdown(
            "Sometimes it does — and that is exactly the point. A model is "
            "only worth its complexity if it beats a simple rule. Showing the "
            "naive and moving-average baselines next to every LSTM number is "
            "what keeps the project honest."
        )
    with st.expander("How do I run this on my own machine?"):
        st.markdown(
            "Clone the repo, `uv sync`, `uv run streamlit run streamlit_app.py`. "
            "The quick start below has the commands. Python 3.13 and uv are the "
            "only prerequisites."
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
            ],
            "Choice": [
                "Streamlit 1.61 (multipage, st.navigation)",
                "Keras 3 (LSTM)",
                "PyTorch (torch.cuda optional)",
                "Yahoo Finance via yfinance + pandas",
                "scipy (distributions, skew and kurtosis)",
                "Plotly (interactive theme-aware charts)",
                "uv + uv.lock (Python 3.13)",
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
uv run streamlit run streamlit_app.py   # start the app on http://localhost:8501""",
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

    st.header("Acknowledgements")
    st.markdown(
        """
        - **Hochreiter & Schmidhuber (1997)** — the LSTM paper that started
          it all. *Neural Computation* 9(8).
        - **Keras 3** — the multi-backend framework the model is written in.
        - **PyTorch** — the backend that makes GPU training just work.
        - **Streamlit** — the framework this whole app is built with.
        - **Yahoo Finance / `yfinance`** — free market data.
        - Every examiner, visitor and reader who pressed *Run analysis* and
          asked the obvious question: *"but does it beat just predicting "
          "yesterday?"* — that question shaped this project.
        """
    )

    st.header("Was this useful?")
    st.markdown(
        "If you are an examiner or a curious visitor, a quick rating helps me improve it."
    )
    st.feedback("thumbs")


if __name__ == "__main__":
    render()
