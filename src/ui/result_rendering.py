"""Result rendering: metric cards, forecast table, quick stats, and full dashboard result."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
import streamlit as st

from . import dashboard_charts as charts
from .sidebar import SESSION_KEY, format_percent

if TYPE_CHECKING:  # pragma: no cover
    from ..core.pipeline import AnalysisResult


class _BandSource(Protocol):
    """The part of `AnalysisResult` the band cache key is derived from.

    Naming the fields rather than taking the whole result keeps `_band_key`
    testable without building a trained model, and makes the key's
    dependencies explicit at the call site.
    """

    symbol: str
    data: pd.DataFrame
    future: pd.DataFrame
    params: dict[str, Any]


CALIBRATION_WINDOWS: int = 8
"""Origins used to calibrate the forecast range.

Conformal coverage is estimated from the residuals themselves, so too few
windows gives a width that is honest in method but noisy in value. Eight is
enough to read and cheap enough to run automatically.
"""

CALIBRATION_EPOCHS: int = 12
"""Epoch budget when retraining the network purely to measure its errors.

The calibration wants the size of the network's mistakes, not its best
possible weights, so it does not pay the full training budget eight times
over. Early stopping still applies.
"""


def _band_key(result: _BandSource) -> tuple[object, ...]:
    """Everything a calibrated band depends on, and nothing else.

    The band is drawn against `result.future` and calibrated from
    `result.data`, so both have to be in the key. The model settings are here
    because they change the network being measured. The date range and the
    forecast horizon were both missing, and both matter: two runs of the same
    ticker at the same settings but over different history produced the same
    key, so the second run silently drew the first run's range over its own
    forecast. The dates are taken from the data itself rather than from the
    request, so there is no way for the two to drift apart.
    """
    return (
        result.symbol,
        result.data.index[0],
        result.data.index[-1],
        len(result.future),
        tuple(sorted(result.params.items())),
    )


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

    # Two rows of three, not six across.
    #
    # Measured in a real browser with a real viewport: at 1262px these cards get
    # ~99px each, and "+25.80%" renders at 129px — so 3 of the 6 values already
    # overflow at full width, and 6 of 6 below 1000px. Streamlit only stacks the
    # row under ~600px, which is a phone.
    #
    # Three per row gives ~330px at the same width, so every value fits with
    # room to spare. Measured after the change: 0 overflow at 1262, 1000, 860
    # and 720px.
    top = st.columns(3)
    top[0].metric("Last close", f"${close.iloc[-1]:,.2f}")
    top[1].metric("YTD", format_percent(pr.get("YTD")))
    top[2].metric("1M", format_percent(pr.get("1M")))

    bottom = st.columns(3)
    bottom[0].metric("6M", format_percent(pr.get("6M")))
    bottom[1].metric("1Y", format_percent(pr.get("1Y")))
    bottom[2].metric("52w range position", f"{pos:.0%}")


def _calibrate_band(result: AnalysisResult) -> None:
    """Put the forecast inside a range, calibrated on the LSTM's own errors.

    This runs automatically rather than behind a button. The point forecast is
    close to flat because that is the honest answer, which makes the *range*
    the part of the chart worth reading — so hiding it behind an expander was
    exactly backwards. Training is GPU-backed where available, so calibrating
    on the network itself is affordable now.

    Cached per symbol and settings, so it costs one walk-forward run per
    distinct analysis rather than one per rerender.
    """
    from ..core.backtest import (
        LIGHTGBM,
        LSTM_UPGRADED,
        BacktestConfig,
        walk_forward,
    )
    from ..core.intervals import (
        conformal_width,
        coverage_of,
        horizon_band,
        parkinson_volatility,
        relative_residuals,
        volatility_scaled_width,
    )
    from ..core.skill import model_summary

    cal_horizon = min(5, len(result.future))
    key = _band_key(result)
    if st.session_state.get("band_key") == key:
        return

    with st.spinner("Calibrating the forecast range…"):
        try:
            frame = walk_forward(
                result.data,
                BacktestConfig(
                    horizon=cal_horizon,
                    n_origins=CALIBRATION_WINDOWS,
                    include_upgraded_lstm=True,
                    # Calibration measures how wrong the network is, which it
                    # can do from a shorter fit. Early stopping still guards
                    # the fit; this just does not pay for the last few epochs
                    # ten times over.
                    lstm_epochs=CALIBRATION_EPOCHS,
                ),
            )
            model = LSTM_UPGRADED
        except ValueError:
            # Too little history to retrain the network; the cheap challenger
            # still gives an honest, if less relevant, width.
            try:
                frame = walk_forward(
                    result.data,
                    BacktestConfig(horizon=cal_horizon, n_origins=10),
                )
                model = LIGHTGBM
            except ValueError as exc:
                st.warning(str(exc))
                return

        residuals = relative_residuals(frame, model)
        if not len(residuals):
            st.warning("Too few windows to calibrate a range.")
            return

        # Scale by how volatile the market is right now, measured from the
        # candle's range rather than close-to-close. A constant width is really
        # the average mistake; this one breathes.
        volatility = parkinson_volatility(result.data)
        at_origins = np.asarray(
            [
                volatility.asof(pd.Timestamp(date))
                for date in frame["origin"].to_numpy()
            ],
            dtype=float,
        )
        current = (
            float(volatility.dropna().iloc[-1])
            if volatility.notna().any()
            else float("nan")
        )

        width = volatility_scaled_width(residuals, at_origins, current)
        if not width:
            st.warning("Too few windows to calibrate a range.")
            return
        coverage = coverage_of(frame, model, width)
        flat_width = conformal_width(residuals)

    points = result.future["predicted_close"].to_numpy(dtype=float)
    lower, upper = horizon_band(points, width, cal_horizon)
    st.session_state["forecast_band"] = pd.DataFrame({"lower": lower, "upper": upper})
    st.session_state["band_info"] = {
        "model": model,
        "windows": len(frame),
        "horizon": cal_horizon,
        "width": width,
        "coverage": coverage,
        "flat_width": flat_width,
        "vol_ratio": (
            current / float(np.mean(at_origins[np.isfinite(at_origins)]))
            if np.isfinite(at_origins).any()
            and np.mean(at_origins[np.isfinite(at_origins)]) > 0
            else float("nan")
        ),
    }
    st.session_state["band_key"] = key
    # The calibration run already scored every model against naive. Reusing it
    # puts the walk-forward verdict on the dashboard instead of making the user
    # go to another page for the same numbers.
    st.session_state["calibration_summary"] = model_summary(frame)


def _band_caption() -> None:
    """State the range in words, and how often it actually held."""
    info = st.session_state.get("band_info")
    if not info:
        return
    ratio = info.get("vol_ratio")
    scaled = (
        f" Today's volatility is {ratio:.2f}× the calibration average, so the "
        f"range has been {'widened' if ratio and ratio > 1 else 'narrowed'} "
        f"from ±{info['flat_width']:.1%} to ±{info['width']:.1%}."
        if np.isfinite(ratio) and abs(ratio - 1.0) > 0.05
        else ""
    )
    st.caption(
        f"Range calibrated from the **{info['model']}**'s own out-of-sample "
        f"errors over {info['windows']} windows at a {info['horizon']}-day "
        f"horizon: ±{info['width']:.1%}, which contained "
        f"{info['coverage']:.0%} of realised outcomes. It widens along the "
        f"path with √time, and scales with the market's current volatility "
        f"measured from each day's high-low range."
        f"{scaled} The honest part of this forecast is how wide it is, not "
        f"where its centre sits."
    )


def _model_comparison() -> None:
    """The walk-forward verdict, shown where the forecast is shown.

    Same numbers as the Evidence page, because it is the same run — the
    dashboard no longer makes you go elsewhere to find out whether the model
    it just drew is any good.
    """
    summary = st.session_state.get("calibration_summary")
    if summary is None or summary.empty:
        return

    with st.expander("Does this model beat 'repeat yesterday'?", expanded=False):
        shown = summary.copy()
        shown["skill"] = shown["skill"].map(lambda v: f"{v:+.1%}")
        st.dataframe(
            shown[["model", "skill", "wins", "win_rate"]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Skill is `1 − MAE / MAE(naive)`. Zero means no better than "
            "repeating yesterday's price; negative means worse. This is the "
            "same test the Evidence page runs across more tickers and more "
            "windows — use that page when you want an answer you can trust "
            "rather than a first look."
        )


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
    _calibrate_band(result)
    st.plotly_chart(
        charts.plot_forecast(result, st.session_state.get("forecast_band")),
        width="stretch",
    )
    _band_caption()
    st.space("medium")
    render_forecast_table(result)
    _model_comparison()
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
