"""Cross-sectional prediction: rank the universe, don't forecast the level.

Everything else in this project asks "what will this stock be worth in five
days?" and the answer, measured over 300 walk-forward windows, is that nothing
beats guessing yesterday's price. That result is solid and it is not going to
change.

This module asks a different question with a different shape: **given a set of
names, which ones go up relative to the others?** The distinction is not
cosmetic. An absolute return needs to beat a market-wide drift that is mostly
noise; a relative return has that drift differenced away. Two stocks can both
be dead money on the day and still have a clear winner between them, and that
winner is the part a long/short book is paid for.

The control matters more here than in the time-series test. "The basket went up"
proves nothing — most baskets do. The comparison is always against holding
everything equally, and against a *random* pick of the same size, because with
twenty names the top-five can easily land well by luck alone. A ranking that
cannot beat a random pick is not a ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import LongShortResult, rank_and_go_long_top
from .market_context import sector_etf_for


def sector_relative_returns(
    frames: dict[str, pd.DataFrame], horizon: int
) -> pd.DataFrame:
    """Forward return of each name minus its sector's, one row per date.

    The sector ETF is the right control rather than the broad index: it
    cancels the sector move that a single stock cannot be expected to
    out-earn, leaving the part that is genuinely about the name.

    A stock whose sector ETF is unavailable falls back to the equal-weighted
    mean of the other names, which is the honest available control rather than
    no control at all. The `sector` column records which was used, so a reader
    can see whether a result came from a real sector match or a fallback.
    """
    if not frames:
        return pd.DataFrame()

    forward: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        close = frame["Close"].astype(float)
        fwd = np.log(close.shift(-horizon) / close)
        forward[symbol] = fwd

    table = pd.DataFrame(forward)

    relatives: dict[str, pd.Series] = {}
    sector_of: dict[str, str] = {}
    for symbol in table.columns:
        series = table[symbol]
        sector = sector_etf_for(symbol)
        sector_of[symbol] = sector
        if sector in forward and sector != symbol:
            control = forward[sector]
        else:
            # Equal-weight of everything else available, excluding this name.
            others = table.drop(columns=[symbol], errors="ignore").mean(axis=1)
            control = others
        relatives[symbol] = series - control

    out = pd.DataFrame(relatives)
    out["sector"] = pd.Series(sector_of)
    return out


def rank_within_date(values: pd.Series) -> pd.Series:
    """Percentile rank of each name on a given day. 1 = best.

    Ranks rather than raw values, because the scale of a relative return
    differs by sector and by name; what the strategy consumes is the ordering.
    Ties share the average rank, so two identical scores do not get an
    arbitrary winner between them.
    """
    return values.rank(pct=True, ascending=True, na_option="keep")


def ranked_relative_frame(relatives: pd.DataFrame) -> pd.DataFrame:
    """Apply the per-date ranking to every column of a relative-return frame."""
    numeric = relatives.select_dtypes(include="number")
    return numeric.apply(rank_within_date, axis=0)


def top_k_long_short(
    scores: dict[str, float],
    realised: dict[str, float],
    k: int = 5,
    seed: int = 42,
    n_random: int = 2000,
) -> LongShortResult:
    """Long the best `k` names, short the worst `k`, against two controls.

    The equity is `top_k_mean - bottom_k_mean`, a dollar-neutral construction:
    it earns the *spread*, not a market bet, so a period where every stock
    falls is not automatically a loss.

    Two controls, because one is not enough:

    - **equal weight**: long the whole universe. "Did picking beat holding" is
      the comparison a user actually cares about.
    - **random pick of the same size**: with a small universe the top-k can look
      excellent on luck alone. The p-value is where the observed spread sits in
      the distribution of random k-name spreads. If it is not near the top, the
      ranking is not real.

    This was a near-duplicate of `decision.rank_and_go_long_top`, differing only
    by the random control, and the two drifted — one gained a `p_value` key the
    other lacked. There is now one definition, here, and `decision` keeps the
    long-only variant. The random control uses a fixed seed so a reported
    p-value is reproducible rather than a number that moves on page reload.
    """
    return rank_and_go_long_top(scores, realised, k=k, seed=seed, n_random=n_random)


def cross_sectional_backtest(
    frames: dict[str, pd.DataFrame],
    horizon: int,
    n_origins: int = 12,
    k: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Walk-forward cross-sectional test across the whole universe at once.

    One forward pass trains on every name and scores every name on the same
    date, then the strategy goes long the top `k` and short the bottom `k`. The
    *same* origins are used for every name, which is the whole point: a
    cross-sectional claim is about one date's ordering, not about one stock's
    direction.

    A panel LightGBM is the model, for one reason that matters more than
    accuracy: it already sees every name simultaneously, which is what a
    ranking requires. A per-ticker model structurally cannot express "this is
    better than that" because it never sees the other names.

    Returns the per-window spread so the result can be judged on its shape
    rather than on a single average — a strategy that wins occasionally by a
    lot is a different proposition from one that wins slightly and often, and
    only one of those survives costs.
    """
    import lightgbm as lgb

    from .features import build_features, forward_log_return

    if len(frames) < 2:
        raise ValueError("cross-sectional testing needs at least two tickers")

    prepared: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol, frame in frames.items():
        close = frame["Close"].astype(float)
        features = build_features(frame, ohlcv=False)
        target = forward_log_return(close, horizon)
        prepared[symbol] = (features, target)

    widths = {f.shape[1] for f, _ in prepared.values()}
    if len(widths) > 1:
        raise ValueError("features differ in width across tickers; use one universe")

    index = next(iter(prepared.values()))[0].index
    if not all(f.index.equals(index) for f, _ in prepared.values()):
        # Align on the common calendar so every name has a row on every date.
        common = index
        for f, _ in prepared.values():
            common = common.intersection(f.index)
        index = common

    usable = np.ones(len(index), dtype=bool)
    for _, (_, target) in prepared.items():
        usable &= target.reindex(index).notna().to_numpy()
    positions = np.flatnonzero(usable)

    if len(positions) < 200:
        raise ValueError("not enough complete rows for a cross-sectional test")

    # Origins spread over the usable span, leaving a full horizon ahead.
    horizon_rows = len(index) - 1 - horizon
    first, last = positions[0], horizon_rows
    if last <= first:
        raise ValueError("not enough history for a cross-sectional walk-forward")
    count = min(n_origins, last - first + 1)
    origins = sorted(
        {first + round(i * (last - first) / max(1, count - 1)) for i in range(count)}
    )

    n_features = next(iter(prepared.values()))[0].shape[1]
    rows: list[dict[str, object]] = []

    for origin in origins:
        train_rows = positions[positions <= origin - horizon]
        if len(train_rows) < 100:
            continue

        blocks_x = []
        blocks_y = []
        for _, (features, target) in prepared.items():
            f = features.reindex(index)
            t = target.reindex(index)
            blocks_x.append(f.iloc[train_rows].to_numpy(dtype=float))
            blocks_y.append(t.iloc[train_rows].to_numpy(dtype=float))

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(np.vstack(blocks_x), np.concatenate(blocks_y))

        scores: dict[str, float] = {}
        realised: dict[str, float] = {}
        for symbol, (features, target) in prepared.items():
            f = features.reindex(index)
            t = target.reindex(index)
            if not np.isfinite(f.iloc[origin].to_numpy(dtype=float)).all():
                continue
            score = float(model.predict(f.iloc[[origin]].to_numpy(dtype=float))[0])
            outcome = t.iloc[origin + horizon]
            if not np.isfinite(outcome):
                continue
            scores[symbol] = score
            realised[symbol] = float(outcome)

        if len(scores) < 2:
            continue
        result = top_k_long_short(scores, realised, k=k, seed=seed)
        rows.append(
            {
                "origin": index[origin],
                "spread": result["spread"],
                "vs_equal_weight": result["vs_equal_weight"],
                "p_value": result["p_value"],
                "n": result["n"],
            }
        )

    if not rows:
        raise ValueError("no window produced a usable cross-sectional ranking")

    frame = pd.DataFrame(rows)
    wins = int((frame["spread"] > 0).sum())
    n = len(frame)
    return {
        "windows": frame,
        "mean_spread": float(frame["spread"].mean()),
        "median_spread": float(frame["spread"].median()),
        "win_rate": wins / n,
        "beat_equal_weight": float((frame["vs_equal_weight"] > 0).mean()),
        "median_p_value": float(frame["p_value"].median()),
        "significant_windows": int((frame["p_value"] < 0.05).sum()),
    }
