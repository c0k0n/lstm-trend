"""Result rendering: metric cards, forecast table, quick stats, and full dashboard result."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import streamlit as st

from ..constants import APP_TITLE
from . import dashboard_charts as charts
from .sidebar import SESSION_KEY, format_percent, get_analysis

if TYPE_CHECKING:
    from ..core.pipeline import AnalysisResult


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
    from ..core import returns as ret_mod

    close = result.close
    pr = ret_mod.period_returns(close)
    pos = ret_mod.position_in_52w_range(close)

    cols = st.columns(6)
    cols[0].metric("Last close", f"${close.iloc[-1]:,.2f}")
    cols[1].metric("YTD", format_percent(pr.get("YTD")))
    cols[2].metric("1M", format_percent(pr.get("1M")))
    cols[3].metric("6M", format_percent(pr.get("6M")))
    cols[4].metric("1Y", format_percent(pr.get("1Y")))
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
    _analysis_age_caption(result)


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
