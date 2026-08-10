"""Analytics page: deep exploratory analysis of a single ticker."""

import datetime

import pandas as pd
import streamlit as st

from ...constants import (
    APP_TITLE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    SUGGESTED_SYMBOLS,
)
from .. import analytics_charts as charts
from ..sidebar import format_percent, load_data_cached

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
    from ...core import risk, returns as ret_mod

    close = data["Close"]
    stats = risk.comparison_frame(close)

    pr = ret_mod.period_returns(close)
    pos = ret_mod.position_in_52w_range(close)
    last = close.iloc[-1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last close", f"${last:,.2f}")
    m2.metric("YTD", format_percent(pr.get("YTD")))
    m3.metric("1M", format_percent(pr.get("1M")))
    m4.metric("6M", format_percent(pr.get("6M")))
    m5.metric("1Y", format_percent(pr.get("1Y")))
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
    from ...core import (
        returns as ret_mod,
        statistics as stats_mod,
        seasonality as seas_mod,
    )

    close = data["Close"]
    returns = ret_mod.daily_returns(close)
    moments = stats_mod.return_moments(returns)
    adf = stats_mod.adf_summary(close)

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
        st.plotly_chart(charts.plot_acf(stats_mod.acf(returns)), width="stretch")

    st.plotly_chart(
        charts.plot_weekday_effects(seas_mod.weekday_effects(returns)), width="stretch"
    )


def _seasonality(data: pd.DataFrame) -> None:
    from ...core import seasonality as seas_mod

    close = data["Close"]

    st.plotly_chart(
        charts.plot_monthly_heatmap(seas_mod.monthly_returns_matrix(close)),
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        months = seas_mod.month_effects(close)
        months["mean"] = months["mean"].map(lambda v: "—" if pd.isna(v) else f"{v:.2%}")
        months["hit_rate"] = months["hit_rate"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.0%}"
        )
        st.dataframe(
            months.rename(
                columns={
                    "month": "Month",
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
    from ...core import indicators as ind_mod

    close = data["Close"]
    signals = ind_mod.latest_signals(close)

    st.plotly_chart(
        charts.plot_price_with_indicators(
            close,
            ind_mod.crossover_dates(ind_mod.sma(close, 20), ind_mod.sma(close, 200)),
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
    from ...core import returns as ret_mod, risk

    close = data["Close"]
    returns = ret_mod.daily_returns(close)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max drawdown", f"{risk.max_drawdown(close):.1%}")
    col2.metric("VaR 95% (daily)", f"{risk.value_at_risk(returns):.2%}")
    col3.metric("CVaR 95% (daily)", f"{risk.conditional_var(returns):.2%}")
    col4.metric("Positive days", f"{risk.positive_day_ratio(returns):.0%}")

    st.plotly_chart(charts.plot_underwater(close), width="stretch")

    events = risk.drawdown_events(close)
    if len(events):
        events["Depth"] = events["Depth"].map(lambda v: f"{v:.1%}")
        events["Start"] = events["Start"].map(lambda d: d.date())
        events["Trough"] = events["Trough"].map(lambda d: d.date())
        events["End"] = events["End"].map(lambda d: d.date())
        page_size = 10
        pages = max(1, (len(events) + page_size - 1) // page_size)
        page = st.pagination(
            pages,
            default=1,
            max_visible_pages=5,
            key="drawdown_pages",
        )
        start = (page - 1) * page_size
        st.caption(
            f"**{len(events)} drawdown events** — showing "
            f"{start + 1}–{min(start + page_size, len(events))}"
        )
        st.dataframe(
            events.iloc[start : start + page_size],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No drawdown of 5% or deeper in this window.")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_volume_analysis(data), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_volume_return_scatter(data), width="stretch")


def _load_data(
    symbol: str, start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    """Fetch OHLCV data, showing skeleton placeholders during a first download."""
    loaded = st.session_state.setdefault("analytics_loaded", {})
    loaded_key = f"{symbol}_{start}_{end}"
    if loaded.get(loaded_key):
        return load_data_cached(symbol, start, end)

    placeholder = st.empty()
    with placeholder.container():
        st.skeleton(height=20, width="stretch")
        st.skeleton(height=360, width="stretch")
    data = load_data_cached(symbol, start, end)
    placeholder.empty()
    loaded[loaded_key] = True
    return data


def render() -> None:
    st.set_page_config(
        page_title=f"Analytics — {APP_TITLE} | Stock EDA, risk metrics and indicators",
        page_icon="📊",
    )
    st.title("📊 Analytics")
    st.text(
        "Exploratory analysis for one ticker: returns and risk metrics, "
        "seasonality, stationarity, technical indicators and volume — "
        "no model training required."
    )

    symbol, start, end = _controls()

    if not symbol:
        st.warning("Enter a ticker to analyse.")
        return

    data = _load_data(symbol, start, end)
    if data is None:
        st.error(
            f"Could not download data for **{symbol}** in this date range. "
            "Check the ticker and the range, then try again."
        )
        return
    if len(data) < 60:
        st.error(
            f"Only {len(data)} trading days found for **{symbol}** — the "
            "analytics need at least 60. Extend the date range and try again."
        )
        return

    close = data["Close"]
    first_day = close.index[0].date()
    last_day = close.index[-1].date()
    st.caption(
        f"**{symbol}** · {len(data)} trading days · "
        f"{first_day} → {last_day} · "
        f"range ${close.min():,.2f} – ${close.max():,.2f}"
    )
    st.space("small")

    tabs = st.tabs(TABS, on_change="rerun")
    if tabs[0].open:
        with tabs[0]:
            _overview(data)
    if tabs[1].open:
        with tabs[1]:
            _returns(data)
    if tabs[2].open:
        with tabs[2]:
            _seasonality(data)
    if tabs[3].open:
        with tabs[3]:
            _technicals(data)
    if tabs[4].open:
        with tabs[4]:
            _risk(data)


if __name__ == "__main__":
    render()
