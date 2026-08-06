"""Compare page: multi-ticker side-by-side analysis."""

import datetime

import pandas as pd
import streamlit as st

from ...constants import (
    APP_TITLE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    SUGGESTED_SYMBOLS,
)
from .. import charts
from ..components import load_data_cached


def _controls() -> tuple[list[str], datetime.date, datetime.date]:
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        watch = _watchlist()
        if watch:
            symbols = watch
            st.caption(
                f"Using the watchlist ({len(watch)} tickers). Untick "
                "**Use watchlist** below to pick manually."
            )
        else:
            symbols = st.multiselect(
                "Tickers to compare",
                options=SUGGESTED_SYMBOLS,
                default=SUGGESTED_SYMBOLS[:3],
                key="compare_symbols",
                help="Pick 2–6 tickers. Suggestions plus a custom one below.",
            )
            custom = (
                st.text_input(
                    "Add custom ticker", placeholder="e.g. SPY", key="compare_custom"
                )
                .strip()
                .upper()
            )
            if custom and custom not in symbols:
                symbols.append(custom)
    with col2:
        start = st.date_input("Start", DEFAULT_START_DATE, key="compare_start")
    with col3:
        end = st.date_input("End", DEFAULT_END_DATE, key="compare_end")
    return symbols, start, end


def _watchlist() -> list[str]:
    """Editable watchlist that can replace the manual multiselect."""
    with st.expander("📋 Watchlist", expanded=False):
        st.caption(
            "Edit the rows to build your own list of tickers, then tick "
            "*Use watchlist* below to compare exactly these."
        )
        edited = st.data_editor(
            pd.DataFrame({"Symbol": SUGGESTED_SYMBOLS}),
            column_config={
                "Symbol": st.column_config.TextColumn(
                    "Symbol", help="Ticker symbol, e.g. AAPL or SPY."
                )
            },
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key="compare_watchlist",
        )
        if not st.checkbox("Use watchlist for comparison", key="compare_use_watchlist"):
            return []
        symbols = [str(s).strip().upper() for s in edited["Symbol"] if str(s).strip()]
        return [s for s in dict.fromkeys(symbols)]


def _metrics_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    from ...core import analytics

    rows = {}
    for name, close in series.items():
        stats = analytics.comparison_frame(close)
        rows[name] = {**stats, "Trend": _sparkline(close)}
    return pd.DataFrame(rows).T


def _sparkline(close: pd.Series) -> list[float]:
    """Downsampled normalized close, for the inline trend mini-chart."""
    sample = close.iloc[:: max(1, len(close) // 40)].to_numpy(dtype=float)
    lo, hi = sample.min(), sample.max()
    if hi - lo < 1e-12:
        return [0.5] * len(sample)
    return ((sample - lo) / (hi - lo)).round(4).tolist()


def render() -> None:
    st.set_page_config(
        page_title=f"Compare — {APP_TITLE} | Multi-ticker side-by-side analysis",
        page_icon="⚖️",
    )
    st.title("⚖️ Compare")
    st.text(
        "Put 2–6 tickers head to head: normalized prices, correlations, "
        "drawdowns, risk-vs-return trade-offs and a full metrics table."
    )
    st.space("medium")

    symbols, start, end = _controls()

    if len(symbols) < 2:
        st.info("Pick at least two tickers to compare them side by side.")
        return

    series: dict[str, pd.Series] = {}
    with st.status("Downloading data…", expanded=False) as status:
        for symbol in symbols:
            loaded_key = f"compare_loaded_{symbol}_{start}_{end}"
            if not st.session_state.get(loaded_key):
                st.skeleton(height=20, width="stretch")
            data = load_data_cached(symbol, start, end)
            st.session_state[loaded_key] = True
            if data is None or len(data) < 60:
                st.warning(f"Not enough data for **{symbol}** in this range — skipped.")
                continue
            series[symbol] = data["Close"]
        status.update(
            label=f"Downloaded {len(series)}/{len(symbols)} tickers", state="complete"
        )

    if len(series) < 2:
        st.error("Not enough tickers with usable data to compare.")
        return

    st.caption(
        f"{len(series)} tickers · {len(next(iter(series.values())))} trading days · "
        f"aligned on daily returns where required"
    )

    table = _metrics_table(series)

    st.subheader("Performance at a glance")
    st.dataframe(
        table.style.format(
            {
                "Total return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Ann. volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "VaR 95% (daily)": "{:.2%}",
                "CVaR 95% (daily)": "{:.2%}",
                "Positive days": "{:.1%}",
            }
        ),
        column_config={
            "Trend": st.column_config.LineChartColumn(
                "Price trend",
                width="medium",
                help="Normalized close over the window — a quick visual of the path.",
            ),
            "Positive days": st.column_config.ProgressColumn(
                "Positive days",
                min_value=0,
                max_value=1,
                format="%.0f%%",
                help="Share of trading days that ended up.",
            ),
        },
        hide_index=False,
        width="stretch",
    )
    st.download_button(
        "⬇️ Download comparison (CSV)",
        data=table.to_csv().encode(),
        file_name="comparison.csv",
        mime="text/csv",
        type="primary",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_normalized_prices(series), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_cumulative_comparison(series), width="stretch")

    # Correlation of daily returns (aligned on common dates)
    returns = pd.DataFrame(
        {name: s.pct_change() for name, s in series.items()}
    ).dropna()
    corr = returns.corr()
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(charts.plot_correlation_heatmap(corr), width="stretch")
    with col2:
        st.markdown(
            """
            **Reading the correlation matrix**

            Values near **+1** mean the tickers move together on the same days
            (they share market risk, so diversifying across them helps less
            than it looks). Values near **0** or negative mean their daily
            moves are largely independent — a genuine diversifier.

            Correlations are unstable over time: they spike in crises, which
            is exactly when you'd want them low.
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.plot_drawdown_comparison(series), width="stretch")
    with col2:
        st.plotly_chart(charts.plot_risk_return_scatter(table), width="stretch")

    with st.expander("What should I take from this?"):
        st.markdown(
            """
            - **CAGR vs volatility** is the classic trade-off: a ticker with a
              high CAGR but double the volatility may not actually reward you
              better per unit of risk — that's what **Sharpe** quantifies.
            - **Max drawdown** tells you the worst you'd have endured holding
              through this window. Big drawdowns are survivable emotionally
              only if you expect them.
            - **Correlation** matters most when you're combining tickers:
              picking two 0.95-correlated names is one bet, not two.
            - Everything here is *historical* — past relationships can and do
              break, so treat the numbers as context, not guidance.
            """
        )


if __name__ == "__main__":
    render()
