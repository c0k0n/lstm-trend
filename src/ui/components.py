"""Streamlit-facing building blocks: sidebar, hero, result widgets."""

import datetime
from functools import lru_cache
from typing import Any, Optional

import streamlit as st

from ..constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_END_DATE,
    DEFAULT_EPOCHS,
    DEFAULT_FUTURE_STEPS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    SUGGESTED_SYMBOLS,
)
from ..core.data_loader import download_stock_data
from ..core.pipeline import AnalysisResult, PipelineError, run_analysis
from . import charts

SESSION_KEY = "analysis"


@st.cache_data(ttl=3600, show_spinner="Downloading market data…")
def load_data_cached(symbol: str, start: datetime.date, end: datetime.date) -> Any:
    """Downloaded OHLCV data, cached for an hour (yfinance rate limits)."""
    return download_stock_data(symbol, start, end)


@lru_cache(maxsize=1)
def training_device() -> str:
    """Describe the device the deep learning stack is running on."""
    import torch

    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


def get_analysis() -> Optional[AnalysisResult]:
    """The most recent analysis, shared between pages via session state."""
    return st.session_state.get(SESSION_KEY)


def store_analysis(result: AnalysisResult) -> None:
    st.session_state[SESSION_KEY] = result
    st.session_state[f"{SESSION_KEY}_run_at"] = datetime.datetime.now()


def render_sidebar_config() -> dict[str, Any]:
    """Render the sidebar controls and return the analysis parameters."""
    st.sidebar.caption(f"Training device: {training_device()}")

    with st.sidebar.popover("⚙️ Model parameters", use_container_width=True):
        sequence_length = st.slider(
            "Lookback window (days)",
            10,
            120,
            DEFAULT_SEQUENCE_LENGTH,
            key="seq_len",
            help="How many past days the model sees before predicting the next.",
        )
        future_steps = st.slider(
            "Prediction horizon (days)",
            5,
            90,
            DEFAULT_FUTURE_STEPS,
            key="fut_steps",
            help="How many business days ahead to forecast.",
        )
        epochs = st.slider("Training epochs", 1, 100, DEFAULT_EPOCHS, key="epochs")
        batch_size = st.select_slider(
            "Batch size",
            options=[8, 16, 32, 64, 128],
            value=DEFAULT_BATCH_SIZE,
            key="batch_size",
        )

    symbol = _render_symbol_picker()
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", DEFAULT_START_DATE, key="start_date")
    end_date = col2.date_input("End", DEFAULT_END_DATE, key="end_date")

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "sequence_length": sequence_length,
        "future_steps": future_steps,
        "epochs": epochs,
        "batch_size": batch_size,
    }


def _render_symbol_picker() -> str:
    """Quick-pick pills for popular tickers with a free-text fallback."""
    with st.sidebar.popover("📈 Choose a ticker", use_container_width=True):
        choice = st.pills(
            "Popular tickers",
            options=[*SUGGESTED_SYMBOLS, "Custom…"],
            default=DEFAULT_SYMBOL,
            key="ticker_pills",
        )
        if choice is None:
            choice = DEFAULT_SYMBOL
        if choice == "Custom…":
            custom = st.text_input(
                "Ticker symbol",
                value=DEFAULT_SYMBOL,
                key="ticker_custom",
                help="e.g. AAPL, TSLA, ^GSPC, BTC-USD",
            )
            return custom.strip().upper()
        return choice


def render_hero() -> None:
    """App title bar with the project positioning."""
    st.title("📈 LSTM Trend")
    st.text(
        "Train a small LSTM on a stock's closing prices, evaluate it against "
        "simple baselines, and explore the forecast — all in your browser."
    )
    st.caption(
        "An empirical study of LSTM forecasting on stock prices — built as a "
        "final year project. This is a learning tool, not investment advice."
    )
    with st.container(horizontal=True):
        st.badge(
            "Final year project", color="violet", help="Built as a student project."
        )
        st.badge(
            "Keras 3 · LSTM",
            color="green",
            help="Two-layer LSTM trained in your browser session.",
        )
        st.badge(
            "PyTorch backend", color="orange", help="GPU acceleration when available."
        )
        st.badge("Streamlit", color="red", help="Built entirely with Streamlit.")
        st.badge(
            "Not investment advice",
            color="gray",
            help="An empirical study, not a trading tool.",
        )
    st.space("medium")


def run_analysis_with_ui(params: dict[str, Any]) -> Optional[AnalysisResult]:
    """Execute the pipeline inside an st.status stage box with progress."""
    from ..core.callbacks import ProgressReporterCallback

    with st.status("Running the analysis pipeline…", expanded=True) as status:
        status.write(
            f"Data: {params['symbol']}, {params['start_date']} → {params['end_date']}"
        )

        progress_bar = st.progress(0.0, text="Training…")

        def on_epoch(epoch: int, total: int, logs: dict[str, Any]) -> None:
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            label = f"Epoch {epoch}/{total}"
            if isinstance(loss, (int, float)):
                label += f" — loss {loss:.4f}"
            if isinstance(val_loss, (int, float)):
                label += f" | val {val_loss:.4f}"
            progress_bar.progress(epoch / total, text=label)

        def on_finish() -> None:
            progress_bar.progress(1.0, text="Training finished")

        callback = ProgressReporterCallback(
            params["epochs"], on_epoch=on_epoch, on_finish=on_finish
        )

        try:
            result = run_analysis(
                symbol=params["symbol"],
                start_date=params["start_date"],
                end_date=params["end_date"],
                sequence_length=params["sequence_length"],
                future_steps=params["future_steps"],
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                progress_callback=callback,
            )
        except PipelineError as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return None

        status.update(label="Analysis complete", state="complete", expanded=False)

    store_analysis(result)
    st.toast(f"Analysis complete for {result.symbol}", icon="✅")
    return result


def render_metrics(result: AnalysisResult) -> None:
    """Metric cards for the LSTM and the best baseline."""
    lstm = result.lstm_metrics
    best_baseline = min(result.baselines.items(), key=lambda kv: kv[1]["rmse"])
    best_rmse = best_baseline[1]["rmse"]
    delta = lstm["rmse"] - best_rmse

    cols = st.columns(4)
    cols[0].metric(
        "RMSE",
        f"${lstm['rmse']:,.2f}",
        delta=f"{delta:+,.2f} vs {best_baseline[0]}",
        help="Root mean squared error on the test window (lower is better).",
    )
    cols[1].metric("MAE", f"${lstm['mae']:,.2f}")
    cols[2].metric("MAPE", f"{lstm['mape']:.2%}")
    cols[3].metric("R²", f"{lstm['r2']:.4f}")


def render_forecast_table(result: AnalysisResult) -> None:
    """The forecast with formatted columns and a CSV download."""
    st.subheader(f"Forecast — next {len(result.future)} business days")
    table = result.forecast_with_change.rename(
        columns={
            "date": "Date",
            "predicted_close": "Predicted close",
            "change_pct": "Day-over-day",
        }
    )
    with st.container(height=480):
        st.dataframe(
            table,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Predicted close": st.column_config.NumberColumn(
                    "Predicted close", format="$%.2f"
                ),
                "Day-over-day": st.column_config.NumberColumn(
                    "Day-over-day", format="%+.2f%%"
                ),
            },
            hide_index=True,
            width="stretch",
        )

    st.space("small")
    csv = table.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download forecast (CSV)",
        data=csv,
        file_name=f"{result.symbol}_forecast.csv",
        mime="text/csv",
        type="primary",
    )


def render_quick_stats(result: AnalysisResult) -> None:
    """Quick context cards about the analysed ticker."""
    from ..core import analytics

    close = result.close
    pr = analytics.period_returns(close)
    pos = analytics.position_in_52w_range(close)

    cols = st.columns(6)
    cols[0].metric("Last close", f"${close.iloc[-1]:,.2f}")
    cols[1].metric("YTD", f"{pr.get('YTD', float('nan')):+.2f}%")
    cols[2].metric("1M", f"{pr.get('1M', float('nan')):+.2f}%")
    cols[3].metric("6M", f"{pr.get('6M', float('nan')):+.2f}%")
    cols[4].metric("1Y", f"{pr.get('1Y', float('nan')):+.2f}%")
    cols[5].metric("52w range position", f"{pos:.0%}")


def render_result(result: AnalysisResult) -> None:
    """Everything shown on the dashboard once an analysis exists."""
    st.subheader("Market snapshot")
    render_quick_stats(result)
    st.space("small")
    st.plotly_chart(
        charts.plot_candlestick(result.data),
        width="stretch",
    )

    st.space("medium")
    st.subheader("Model performance")
    render_metrics(result)
    st.space("small")
    st.plotly_chart(charts.plot_loss_history(result.history), width="stretch")
    st.space("small")
    st.plotly_chart(
        charts.plot_test_predictions(
            result.test_predictions,
            result.close,
            result.test_start_index,
            result.params["sequence_length"],
        ),
        width="stretch",
    )
    st.space("small")
    st.plotly_chart(charts.plot_forecast(result), width="stretch")
    st.space("medium")
    render_forecast_table(result)
    st.divider()
    st.caption(
        "Baseline comparison and a full breakdown of the metrics live on the "
        "**Findings** page."
    )
