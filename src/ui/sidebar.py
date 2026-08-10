"""Sidebar configuration, symbol picker, training progress, and shared session helpers."""

from __future__ import annotations

import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
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

SESSION_KEY = "analysis"


def format_percent(value: float | None) -> str:
    """Signed percent string, or an em dash when the value is missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2f}%"


@st.cache_data(ttl=3600, show_spinner="Downloading market data…")
def load_data_cached(
    symbol: str, start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    """Downloaded OHLCV data, cached for an hour (yfinance rate limits)."""
    return download_stock_data(symbol, start, end)


@lru_cache(maxsize=1)
def training_device() -> str:
    """Describe the device the deep learning stack is running on."""
    import torch

    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    return "CPU"


def get_analysis() -> AnalysisResult | None:
    """The most recent analysis, shared between pages via session state."""
    return st.session_state.get(SESSION_KEY)


def store_analysis(result: AnalysisResult) -> None:
    st.session_state[SESSION_KEY] = result
    st.session_state[f"{SESSION_KEY}_run_at"] = datetime.datetime.now()


def render_sidebar_config() -> dict[str, Any]:
    """Render the sidebar controls and return the analysis parameters."""
    st.sidebar.caption(f"Training device: {training_device()}")

    with st.sidebar.popover("⚙️ Model parameters", use_container_width=True):
        with st.form("model_settings_form"):
            sequence_length = st.slider(
                "Lookback window (days)",
                10,
                120,
                DEFAULT_SEQUENCE_LENGTH,
                key="seq_len",
                help="How many past days the model sees before predicting the next.",
            )
            st.session_state.setdefault("horizon_preset", DEFAULT_FUTURE_STEPS)
            future_steps = st.segmented_control(
                "Forecast horizon (days)",
                options=[5, 15, 30, "Custom…"],
                key="horizon_preset",
                help="How many business days ahead to forecast. Pick a preset or "
                "choose Custom… for any value between 5 and 90.",
            )
            if future_steps is None:
                future_steps = DEFAULT_FUTURE_STEPS
            if future_steps == "Custom…":
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
            st.form_submit_button(
                "Apply model settings",
                width="stretch",
                key="apply_model_settings",
                help="Changes above only take effect after you press Apply.",
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
    st.session_state.setdefault("ticker_pills", DEFAULT_SYMBOL)
    st.session_state.setdefault("ticker_custom", DEFAULT_SYMBOL)
    with st.sidebar.popover("📈 Choose a ticker", use_container_width=True):
        choice = st.pills(
            "Popular tickers",
            options=[*SUGGESTED_SYMBOLS, "Custom…"],
            key="ticker_pills",
        )
        if choice is None:
            choice = DEFAULT_SYMBOL
        if choice == "Custom…":
            custom = st.text_input(
                "Ticker symbol",
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


def run_analysis_with_ui(params: dict[str, Any]) -> AnalysisResult | None:
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
    if not st.session_state.get("balloons_done"):
        st.session_state["balloons_done"] = True
        st.balloons()
    st.toast(f"Analysis complete for {result.symbol}", icon="✅")
    return result


@st.fragment(run_every=60)
def _analysis_age_caption(result: AnalysisResult) -> None:
    """Live 'last run' caption; refreshes itself every minute."""
    run_at = st.session_state.get(f"{SESSION_KEY}_run_at")
    if run_at is None:
        return
    age = datetime.datetime.now() - run_at
    if age.total_seconds() < 60:
        when = "just now"
    elif age.total_seconds() < 3600:
        when = f"{int(age.total_seconds() // 60)} minutes ago"
    else:
        when = f"{age.total_seconds() / 3600:.1f} hours ago"
    st.caption(
        f"Analysis of **{result.symbol}** run at {run_at:%H:%M:%S} — {when}. "
        "Baseline comparison and a full breakdown of the metrics live on the "
        "**Findings** page."
    )
