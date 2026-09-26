"""Evidence page: does any model actually beat 'repeat yesterday'?

This is the page that answers the question the rest of the app can only hint
at. Instead of one train/test split, it rolls the forecast origin forward
through history and scores every window, then states plainly whether an edge
survived.

Run it on several tickers at once and the windows pool into one verdict. A
single ticker gives about a dozen windows, which cannot resolve anything —
even seven wins out of ten is indistinguishable from a coin flip. Ten tickers
give over a hundred windows, which can.
"""

from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from ...constants import (
    APP_TITLE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    default_end_date,
)
from ...core.decision import decision_summary
from ...core.skill import BASELINE
from ...core.backtest import (
    LIGHTGBM,
    LSTM_UPGRADED,
    PANEL,
    BacktestConfig,
    walk_forward,
)
from ...core.intervals import (
    DEFAULT_COVERAGE,
    conformal_width,
    coverage_of,
    pooled_coverage,
    pooled_width,
    relative_residuals,
)
from ...core.journal import connect, pending, record_stats, settle, settled
from ...core.market_context import build_context, download_market_context
from ...core.skill import model_summary, pooled_summary, significance, verdict_for
from .. import evidence_charts as charts
from ..sidebar import load_data_cached

MIN_ROWS = 220
"""Roughly a year of trading days; below this the windows get too few to read."""


@st.cache_data(show_spinner=False, ttl=3600)
def _market_context_cached(
    start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    return download_market_context(start, end)


@st.cache_data(show_spinner=False, ttl=3600)
def _context_cached(
    symbol: str, start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    """Market context for one ticker, with the shared part downloaded once."""
    market = _market_context_cached(start, end)
    return None if market is None else build_context(symbol, start, end, market)


def _parse_symbols(raw: str) -> list[str]:
    """Split a comma-separated field into tickers, dropping blanks."""
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def _controls() -> tuple[
    list[str], datetime.date, datetime.date, int, int, bool, bool, bool
]:
    st.session_state.setdefault("evidence_symbol", DEFAULT_SYMBOL)

    with st.sidebar.popover("🧪 Test settings", use_container_width=True):
        with st.sidebar.form("evidence_settings_form"):
            raw = st.text_input(
                "Tickers",
                key="evidence_symbol",
                help="One symbol, or several separated by commas. Several "
                "gives a pooled verdict with far more windows behind it.",
            )

            col1, col2 = st.columns(2)
            start = col1.date_input("Start", DEFAULT_START_DATE, key="evidence_start")
            end = col2.date_input("End", default_end_date(), key="evidence_end")

            horizon = st.slider(
                "Horizon (trading days)",
                1,
                20,
                5,
                key="evidence_horizon",
                help="How far ahead each forecast reaches. Shorter is more believable.",
            )
            origins = st.slider(
                "Windows per ticker",
                4,
                20,
                12,
                key="evidence_origins",
                help="How many separate points in history to test. More windows "
                "means a sharper answer and a slower run.",
            )
            include_upgraded_lstm = st.toggle(
                "Include the LSTM (slow)",
                value=False,
                key="evidence_lstm",
                help="The upgraded network: log-return target, covariates, "
                "direct horizon, layer normalisation. It retrains at every "
                "window, so this is off by default. Measured over 200 windows "
                "it beats the original raw-price network by 26 points "
                "(p = 0.008), which is why the original is gone.",
            )
            include_panel = st.toggle(
                "Include a panel model",
                value=True,
                key="evidence_panel",
                help="Trains one model across every ticker at once instead of "
                "one per ticker. Needs at least two tickers. Slowest to train "
                "on the first window, then cheap.",
            )
            include_context = st.toggle(
                "Include market context",
                value=False,
                key="evidence_context",
                help="VIX, the broad market, the short rate and this ticker's "
                "sector ETF. Measured over 300 windows this did not help: it "
                "moved the panel model 4 points the wrong way and the "
                "per-ticker model 2 points the right way, with 10 of 20 "
                "tickers improving either way. Off by default — turn it on to "
                "re-test the lever yourself.",
            )
            st.form_submit_button("Apply test settings", width="stretch")

    return (
        _parse_symbols(raw),
        start,
        end,
        horizon,
        origins,
        include_upgraded_lstm,
        include_panel,
        include_context,
    )


def _verdict_block(summary: pd.DataFrame) -> None:
    """One bordered block per model, stating the outcome in plain words."""
    for _, row in summary.iterrows():
        verdict = verdict_for(row)
        icon = "✅" if verdict["edge"] else "⚠️"
        colour = "green" if verdict["edge"] else "orange"

        with st.container(border=True):
            st.markdown(f"#### {icon} {row['model']}")
            st.badge(verdict["label"], color=colour)
            st.space("small")
            st.markdown(verdict["detail"])
            st.caption(verdict["note"])


def _numbers_table(
    summary: pd.DataFrame,
    bands: dict[str, tuple[float, float]] | None = None,
) -> None:
    """`bands` maps a model to its calibrated half-width and realised coverage."""
    table = summary.copy()
    table["skill"] = table["skill"].map(lambda value: f"{value:+.1%}")
    table["win_rate"] = table["win_rate"].map(lambda value: f"{value:.0%}")
    table["p_value"] = table.apply(lambda row: f"{significance(row):.4f}", axis=1)

    columns = ["model", "skill", "win_rate", "p_value", "windows"]
    names = {
        "model": "Model",
        "skill": "Skill vs naive",
        "win_rate": "Windows won",
        "p_value": "Sign-test p",
        "windows": "Windows",
    }
    if "tickers" in table.columns:
        columns.insert(1, "tickers")
        names["tickers"] = "Tickers"

    if bands:
        table["band"] = table["model"].map(
            lambda name: f"±{bands[name][0]:.1%}" if bands[name][0] else "—"
        )
        table["coverage"] = table["model"].map(
            lambda name: f"{bands[name][1]:.0%}" if bands[name][0] else "—"
        )
        columns.insert(2, "band")
        columns.insert(3, "coverage")
        names["band"] = f"Band ({DEFAULT_COVERAGE:.0%})"
        names["coverage"] = "Realised"

    st.dataframe(
        table[columns].rename(columns=names),
        hide_index=True,
        width="stretch",
    )
    if bands:
        st.caption(
            "The band is the range that contained the realised price most "
            "often out of sample; **Realised** is how often it actually did. "
            "If the two disagree, the band is overconfident."
        )


def _decision_table(frames: dict[str, pd.DataFrame], horizon: int) -> None:
    """Score every model on what it would be worth *acting on*.

    Skill answers "did it predict the price better?". That is not the question
    a user has, which is "should I take this?" — and the two can disagree. A
    model with zero point skill and a well-calibrated range is useful for
    sizing risk; a model with positive point skill and no calibration is not
    tradeable, because you cannot tell which of its forecasts to believe.

    Costs are on. A strategy that only works at zero cost is not a strategy,
    and at 10 bps a trade the friction is what stops a 52%-accurate
    forecaster from looking like a money machine.
    """
    rows: list[dict[str, object]] = []
    for frame in frames.values():
        for model in frame.columns:
            if model in ("origin", "target", "actual") or model == BASELINE:
                continue
            # Typed all the way down, so no cast and no assert is needed here.
            summary = decision_summary(
                frame, model, reference=BASELINE, period_days=horizon
            )
            rows.append(
                {
                    "Model": model,
                    "Directional": summary["direction"]["accuracy"],
                    "vs coin flip": summary["direction"]["edge"],
                    "Ranking info": summary["information"]["model"],
                    "Net per window": summary["utility"]["net"],
                    "Annualised": summary["utility"]["annualized"],
                    "Sharpe": summary["sharpe"],
                }
            )

    if not rows:
        return

    table = pd.DataFrame(rows)
    st.subheader("Would it be worth acting on?")
    st.caption(
        "A different question from skill above. **Directional** is how often the "
        "sign of the call was right; **vs coin flip** is that minus what a "
        "coin flip would have scored, so it is the part that is skill. "
        f"**Net per window** is the return of going long when it says up and flat "
        f"when it says down, after 10 bps a trade. **Annualised** compounds that "
        f"over 252/{horizon} windows a year — it is a projection, not a result."
    )
    st.dataframe(
        table.style.format(
            {
                "Directional": "{:.1%}",
                "vs coin flip": "{:+.1%}",
                "Ranking info": "{:+.3f}",
                "Net per window": "{:+.4%}",
                "Annualised": "{:+.1%}",
                "Sharpe": "{:+.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _journal_section() -> None:
    """The forward record: forecasts written down, then graded when due.

    This is the only part of the app that can be wrong in public. Backtests are
    rehearsal; this is the performance.
    """
    with st.expander("📓 Forward record", expanded=False):
        connection = connect()
        try:
            stats = record_stats(connection)
            outstanding = pending(connection)
            graded = settled(connection)

            if not stats["recorded"]:
                st.caption(
                    "Nothing recorded yet. Run an analysis on the Dashboard "
                    "and its forecast gets written down here."
                )
                return

            col1, col2, col3 = st.columns(3)
            col1.metric("Recorded", f"{int(stats['recorded'])}")
            col2.metric("Graded", f"{int(stats['settled'])}")
            col3.metric("Awaiting their date", f"{int(stats['pending'])}")

            if stats["settled"]:
                col1, col2 = st.columns(2)
                if "skill" in stats:
                    col1.metric(
                        "Skill vs naive",
                        f"{stats['skill']:+.1%}",
                        help="On forecasts that were written down before the "
                        "outcome existed.",
                    )
                if "coverage" in stats:
                    col2.metric("Interval coverage", f"{stats['coverage']:.0%}")
                st.caption(
                    f"Beat 'repeat yesterday' on {stats.get('beat_naive', 0):.0%} "
                    "of graded forecasts."
                )

            if not outstanding.empty:
                st.caption(
                    f"{len(outstanding)} forecast(s) waiting for their date to pass."
                )
                if st.button("Grade forecasts that are due", key="settle_button"):
                    symbols = list(outstanding["symbol"].unique())
                    earliest = min(outstanding["target_date"])
                    start = earliest - datetime.timedelta(days=10)

                    prices: dict[str, pd.Series] = {}
                    for symbol in symbols:
                        data = load_data_cached(symbol, start, datetime.date.today())
                        if data is not None:
                            prices[symbol] = data["Close"]

                    graded_count = settle(connection, prices)
                    st.success(f"Graded {graded_count} forecast(s).")
                    st.rerun()

            if not graded.empty:
                st.dataframe(
                    graded[
                        ["symbol", "target_date", "model", "point", "actual"]
                    ].rename(
                        columns={
                            "symbol": "Ticker",
                            "target_date": "Target date",
                            "model": "Model",
                            "point": "Forecast",
                            "actual": "Actual",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        finally:
            connection.close()

        st.caption(
            "Stored in `journal.db`. On Streamlit Community Cloud the "
            "filesystem is ephemeral, so treat this as a session ledger unless "
            "you export it."
        )


def render() -> None:
    st.set_page_config(
        page_title=f"Evidence — {APP_TITLE} | Does the model beat a naive guess?",
        page_icon="🧪",
    )
    st.title("🧪 Evidence")
    st.text(
        "The same scorecard, run across many points in history instead of one. "
        "If a model cannot beat 'repeat yesterday' here, its test-window RMSE "
        "on the Findings page was luck."
    )

    (
        symbols,
        start,
        end,
        horizon,
        origins,
        include_upgraded_lstm,
        include_panel,
        include_context,
    ) = _controls()

    if not symbols:
        st.warning("Enter at least one ticker to test.")
        return

    st.caption(
        f"{len(symbols)} ticker(s) · {start} to {end} · horizon {horizon} days "
        f"· {origins} windows each"
    )

    _journal_section()
    st.space("small")

    if not st.button("▶️ Run the walk-forward test", type="primary"):
        st.info(
            "Press the button to roll the forecast origin through history and "
            "score every window."
        )
        return

    config = BacktestConfig(
        horizon=horizon,
        n_origins=origins,
        include_upgraded_lstm=include_upgraded_lstm,
        include_panel=include_panel,
    )

    # Every ticker is downloaded before any of them is tested, so the panel
    # model can train on all of them. Downloading and testing in one pass would
    # silently give the first ticker a panel model trained on nothing.
    datasets: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        with st.spinner(f"Downloading {symbol}…"):
            data = load_data_cached(symbol, start, end)

        if data is None:
            st.warning(f"No data for **{symbol}** — skipped.")
            continue
        if len(data) < MIN_ROWS:
            st.warning(
                f"**{symbol}** has only {len(data)} trading days, needs "
                f"{MIN_ROWS} — skipped."
            )
            continue
        datasets[symbol] = data

    # Context is all or nothing. A panel model fed a mixture of tickers with
    # and without context would be stacking frames of different widths, so a
    # single missing ticker drops the whole lever rather than half-applying it.
    contexts: dict[str, pd.DataFrame] | None = None
    if include_context:
        built: dict[str, pd.DataFrame] = {}
        for symbol in datasets:
            with st.spinner(f"Downloading market context for {symbol}…"):
                context = _context_cached(symbol, start, end)
            if context is None:
                st.warning(
                    f"No market context for **{symbol}** — dropping context for "
                    "the whole run so every ticker is scored the same way."
                )
                built = {}
                break
            built[symbol] = context
        contexts = built or None

    frames: dict[str, pd.DataFrame] = {}
    for symbol, data in datasets.items():
        with st.status(f"Testing {symbol}…", expanded=False) as status:
            progress = st.progress(0.0, text="Scoring windows…")

            def on_origin(done: int, total: int) -> None:
                progress.progress(done / total, text=f"Window {done}/{total}")

            try:
                frames[symbol] = walk_forward(
                    data,
                    config,
                    on_origin=on_origin,
                    panel_data=datasets,
                    symbol=symbol,
                    contexts=contexts,
                )
            except ValueError as exc:
                st.error(str(exc))
                continue
            status.update(label=f"{symbol} done", state="complete")

    if not frames:
        st.error("No ticker had enough data to test.")
        return

    if contexts is not None:
        # Label the run for what it was. Reporting a context run under the
        # plain model's name would make it look like the measured table above
        # had been reproduced when it had not.
        labelled = {
            LIGHTGBM: f"{LIGHTGBM} + context",
            PANEL: f"{PANEL} + context",
            LSTM_UPGRADED: f"{LSTM_UPGRADED} + context",
        }
        frames = {
            symbol: frame.rename(
                columns={old: new for old, new in labelled.items() if old in frame}
            )
            for symbol, frame in frames.items()
        }

    pooled = len(frames) > 1
    if pooled:
        summary = pooled_summary(frames)
    else:
        summary = model_summary(next(iter(frames.values())))
    total_windows = int(summary["windows"].max())

    st.subheader("The verdict")
    if pooled:
        st.caption(
            f"Pooled across {len(frames)} tickers — {total_windows} windows. "
            "One ticker's dozen windows cannot resolve anything; this can."
        )
    _verdict_block(summary)

    st.subheader("Skill by model")
    st.plotly_chart(charts.plot_model_skill(summary), width="stretch")

    if pooled:
        st.subheader("The spread behind that average")
        spread_model = st.selectbox(
            "Model",
            [m for m in summary["model"] if not m.startswith("Naive")],
            key="evidence_spread_model",
        )
        st.plotly_chart(
            charts.plot_ticker_spread(frames, spread_model), width="stretch"
        )
        st.caption(
            "Read this before quoting any single ticker. The same model can "
            "look strong on one name and be badly wrong on another, and "
            "picking the winner after seeing them all is how a demo turns "
            "into a claim nobody can reproduce."
        )

    if not pooled:
        st.subheader("Window by window")
        st.plotly_chart(
            charts.plot_window_errors(next(iter(frames.values()))), width="stretch"
        )
        st.caption(
            "A line that swings wildly from window to window is not a reliable "
            "edge, however good its average looks."
        )

    bands: dict[str, tuple[float, float]] = {}
    for model in summary["model"]:
        if pooled:
            width = pooled_width(frames, model)
            cover = pooled_coverage(frames, model, width) if width else 0.0
        else:
            only = next(iter(frames.values()))
            width = conformal_width(relative_residuals(only, model))
            cover = coverage_of(only, model, width) if width else 0.0
        bands[model] = (width, cover)

    st.subheader("The numbers")
    _numbers_table(summary, bands)

    _decision_table(frames, horizon)

    with st.expander("How to read this"):
        st.markdown(
            """
            **Skill** is `1 − MAE(model) / MAE(naive)`. Zero means the model is
            exactly as good as repeating yesterday's price; +10% means its
            average error was 10% smaller. Negative means worse.

            **Windows won** is how many individual windows the model beat the
            naive guess on. A model can show positive overall skill from one or
            two big wins while losing most windows — that is not an edge.

            **Sign-test p** asks how often a coin flip would produce a win
            record at least this lopsided. Above 0.05, the win rate is
            indistinguishable from luck.

            An edge is only reported when the margin, the consistency and the
            significance test all agree. Most of the time they do not — and
            that is the finding, not a failure of the test.

            With several tickers, skill is measured within each ticker and then
            averaged, one vote each. Dollar errors do not add across stocks,
            and averaging per-window ratios instead would let one quiet day
            swamp the result.
            """
        )

    with st.expander("What this test cannot tell you"):
        st.markdown(
            """
            - Every window is history. Even a clean forward-only test on past
              data can be undermined by having chosen the model after seeing
              that data.
            - Testing many tickers and then reporting the best one is
              selection, not evidence. The pooled verdict avoids that by
              reporting all of them together.
            - Costs, slippage and taxes are not modelled.
            - A model that beats naive on price may still be useless for
              anything you would actually do with it.
            """
        )

    st.caption(f"Last run {datetime.datetime.now():%H:%M:%S}.")


if __name__ == "__main__":
    render()
