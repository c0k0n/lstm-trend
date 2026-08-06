"""Analytics page: deep exploratory analysis of a single ticker."""

import datetime

import pandas as pd
import streamlit as st

from ...constants import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    SUGGESTED_SYMBOLS,
)
from .. import charts
from ..components import load_data_cached

TABS = ["📈 Overview", "📊 Returns", "📅 Seasonality", "🧭 Technicals", "🕳️ Risk"]


def _controls() -> tuple[str, datetime.date, datetime.date]:
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        choice = st.selectbox(
            "Ticker",
            options=[*SUGGESTED_SYMBOLS, "Custom…"],
            index=SUGGESTED_SYMBOLS.index(DEFAULT_SYMBOL),
            key="analytics_ticker_choice",
        )
        symbol = (
            st.text_input(
                "Custom ticker",
                value="",
                placeholder="e.g. SPY",
                key="analytics_ticker_custom",
            )
            .strip()
            .upper()
            if choice == "Custom…"
            else choice
        )
    with col2:
        start = st.date_input("Start", DEFAULT_START_DATE, key="analytics_start")
    with col3:
        end = st.date_input("End", DEFAULT_END_DATE, key="analytics_end")
    return symbol, start, end


def _overview(data: pd.DataFrame) -> None:
    from ...core import analytics

    close = data["Close"]
    stats = analytics.comparison_frame(close)

    pr = analytics.period_returns(close)
    pos = analytics.position_in_52w_range(close)
    last = close.iloc[-1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last close", f"${last:,.2f}")
    m2.metric("YTD", f"{pr.get('YTD', float('nan')):.2f}%")
    m3.metric("1M", f"{pr.get('1M', float('nan')):.2f}%")
    m4.metric("6M", f"{pr.get('6M', float('nan')):.2f}%")
    m5.metric("1Y", f"{pr.get('1Y', float('nan')):.2f}%")
    m1.metric("52w range position", f"{pos:.0%}", delta=None)
    m2.metric("Max drawdown", f"{stats['Max drawdown']:.1%}")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(charts.plot_cumulative_returns(close), width="stretch")
    with col2:
        rows = pd.DataFrame(
            {
                "Metric": [
                    "Total return",
                    "CAGR",
                    "Annualized volatility",
                    "Sharpe ratio",
                    "Sortino ratio",
                    "Max drawdown",
                    "VaR 95% (daily)",
                    "CVaR 95% (daily)",
                    "Positive days",
                ],
                "Value": [
                    f"{stats['Total return']:.2%}",
                    f"{stats['CAGR']:.2%}",
                    f"{stats['Ann. volatility']:.2%}",
                    f"{stats['Sharpe']:.2f}",
                    f"{stats['Sortino']:.2f}",
                    f"{stats['Max drawdown']:.2%}",
                    f"{stats['VaR 95% (daily)']:.2%}",
                    f"{stats['CVaR 95% (daily)']:.2%}",
                    f"{stats['Positive days']:.1%}",
                ],
            }
        )
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Download stats (CSV)",
            data=pd.DataFrame({k: [v] for k, v in stats.items()})
            .to_csv(index=False)
            .encode(),
            file_name="analytics_stats.csv",
            mime="text/csv",
            type="primary",
        )


def _returns(data: pd.DataFrame) -> None:
    from ...core import analytics

    close = data["Close"]
    returns = analytics.daily_returns(close)
    moments = analytics.return_moments(returns)
    adf = analytics.adf_summary(close)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skewness", f"{moments['skewness']:.2f}")
    col2.metric("Kurtosis", f"{moments['kurtosis']:.2f}")
    col3.metric("Daily std dev", f"{moments['std']:.2%}")
    col4.metric("ADF p-value", f"{adf['pvalue']:.4f}")

    verdict = (
        "p < 0.05 — the price series is **stationary**, so patterns are more likely "
        "to persist."
        if adf["stationary"]
        else "p ≥ 0.05 — the price series looks **non-stationary** (typical for stocks): "
        "returns, not raw prices, are the safer thing to model."
    )
    st.info(f"**Augmented Dickey–Fuller test:** {verdict}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_return_histogram(returns), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_qq(returns), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_rolling_volatility(returns), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_acf(analytics.acf(returns)), width="stretch")

    st.plotly_chart(
        charts.plot_weekday_effects(analytics.weekday_effects(returns)), width="stretch"
    )


def _seasonality(data: pd.DataFrame) -> None:
    from ...core import analytics

    close = data["Close"]

    st.plotly_chart(
        charts.plot_monthly_heatmap(analytics.monthly_returns_matrix(close)),
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        months = analytics.month_effects(close)
        months["mean"] = months["mean"].map(lambda v: f"{v:.2%}")
        months["hit_rate"] = months["hit_rate"].map(lambda v: f"{v:.0%}")
        st.dataframe(
            months.rename(
                columns={
                    "mean": "Avg monthly return",
                    "hit_rate": "Up months",
                    "count": "Count",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with col2:
        st.markdown(
            """
            **Reading this tab**

            - **Monthly heatmap** — each cell is one calendar month's return.
              Red is down, green is up. Look for *patterns in time* (e.g. weak
              Septembers) before trusting them: with a few years of data, noise
              often mimics seasonality.
            - **Weekday effects** — some studies find Monday/Friday effects;
              this table shows whether this ticker actually exhibits them.
            - Everything here is descriptive, not predictive — and the sample
              sizes are small.
            """
        )


def _technicals(data: pd.DataFrame) -> None:
    from ...core import analytics

    close = data["Close"]
    signals = analytics.latest_signals(close)

    st.plotly_chart(
        charts.plot_price_with_indicators(
            close,
            analytics.crossover_dates(
                analytics.sma(close, 20), analytics.sma(close, 200)
            ),
        ),
        width="stretch",
    )

    with st.expander("Current signals", expanded=True):
        for name, signal in signals.items():
            st.markdown(f"- **{name}:** {signal}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_rsi(close), width="stretch")
        st.plotly_chart(charts.plot_bollinger(close), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_macd(close), width="stretch")


def _risk(data: pd.DataFrame) -> None:
    from ...core import analytics

    close = data["Close"]
    returns = analytics.daily_returns(close)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max drawdown", f"{analytics.max_drawdown(close):.1%}")
    col2.metric("VaR 95% (daily)", f"{analytics.value_at_risk(returns):.2%}")
    col3.metric("CVaR 95% (daily)", f"{analytics.conditional_var(returns):.2%}")
    col4.metric("Positive days", f"{analytics.positive_day_ratio(returns):.0%}")

    st.plotly_chart(charts.plot_underwater(close), width="stretch")

    events = analytics.drawdown_events(close)
    if len(events):
        events["Depth"] = events["Depth"].map(lambda v: f"{v:.1%}")
        events["Start"] = events["Start"].map(lambda d: d.date())
        events["Trough"] = events["Trough"].map(lambda d: d.date())
        events["End"] = events["End"].map(lambda d: d.date())
        st.dataframe(events, hide_index=True, width="stretch")
    else:
        st.caption("No drawdown of 5% or deeper in this window.")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_volume_analysis(data), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_volume_return_scatter(data), width="stretch")


def render() -> None:
    st.title("📊 Analytics")

    symbol, start, end = _controls()

    if not symbol:
        st.warning("Enter a ticker to analyse.")
        return

    data = load_data_cached(symbol, start, end)
    if data is None or len(data) < 60:
        st.error(
            f"Could not download enough data for **{symbol}** in this date range. "
            "Check the ticker and the range, then try again."
        )
        return

    close = data["Close"]
    st.caption(
        f"**{symbol}** · {len(data)} trading days · "
        f"{close.index[0].date()} → {close.index[-1].date()} · "
        f"range ${close.min():,.2f} – ${close.max():,.2f}"
    )

    tab_overview, tab_returns, tab_season, tab_tech, tab_risk = st.tabs(TABS)
    with tab_overview:
        _overview(data)
    with tab_returns:
        _returns(data)
    with tab_season:
        _seasonality(data)
    with tab_tech:
        _technicals(data)
    with tab_risk:
        _risk(data)


if __name__ == "__main__":
    render()
