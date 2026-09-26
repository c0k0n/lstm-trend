"""Forecast journal: write every forecast down, then grade it when the date comes.

The app's deepest flaw was that it kept no record of its own predictions, so it
could never be wrong in public. This is the fix.

Each entry stores the price at the moment of forecasting, which matters more
than it looks: it makes the naive guess recoverable later, so settled forecasts
can be scored as skill against "repeat yesterday" without re-running anything.

Storage is SQLite. One honest caveat: on Streamlit Community Cloud the
filesystem is ephemeral, so this journal persists within a session and may not
survive a reboot. Treated as a durable record, it is a local ledger; for a
public permanent one, export it or use a hosted forward-only service.
"""

from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass

import pandas as pd

DEFAULT_PATH = "journal.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    generated_at  TEXT NOT NULL,
    target_date   TEXT NOT NULL,
    model         TEXT NOT NULL,
    point         REAL NOT NULL,
    lower         REAL,
    upper         REAL,
    origin_price  REAL,
    actual        REAL,
    settled_at    TEXT,
    UNIQUE(symbol, generated_at, target_date, model)
);
"""


@dataclass
class ForecastEntry:
    """One forecast for one future trading day."""

    symbol: str
    generated_at: datetime.datetime
    target_date: datetime.date
    model: str
    point: float
    lower: float | None = None
    upper: float | None = None
    origin_price: float | None = None


def connect(path: str = DEFAULT_PATH) -> sqlite3.Connection:
    """Open the journal, creating it and its table if needed."""
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    return conn


def record(conn: sqlite3.Connection, entries: list[ForecastEntry]) -> int:
    """Store forecasts, ignoring exact duplicates. Returns rows inserted."""
    rows = [
        (
            entry.symbol,
            entry.generated_at.isoformat(),
            entry.target_date.isoformat(),
            entry.model,
            entry.point,
            entry.lower,
            entry.upper,
            entry.origin_price,
            None,
            None,
        )
        for entry in entries
    ]
    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO forecasts
            (symbol, generated_at, target_date, model, point, lower, upper,
             origin_price, actual, settled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return cursor.rowcount


def _read(conn: sqlite3.Connection, settled: bool) -> pd.DataFrame:
    clause = "IS NOT NULL" if settled else "IS NULL"
    frame = pd.read_sql_query(
        f"SELECT * FROM forecasts WHERE actual {clause} ORDER BY target_date",
        conn,
    )
    if not frame.empty:
        frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    return frame


def pending(conn: sqlite3.Connection) -> pd.DataFrame:
    """Forecasts whose target date has not been graded yet."""
    return _read(conn, settled=False)


def settled(conn: sqlite3.Connection) -> pd.DataFrame:
    """Forecasts that have been graded against what actually happened."""
    return _read(conn, settled=True)


def settle(conn: sqlite3.Connection, prices: dict[str, pd.Series]) -> int:
    """Grade every pending forecast against realised prices.

    `prices` maps a symbol to its close series. The realised price is the first
    close on or after the target date, so a target landing on a holiday settles
    against the next trading day rather than silently failing.
    """
    outstanding = pending(conn)
    if outstanding.empty:
        return 0

    updated = 0
    now = datetime.datetime.now().isoformat()
    for _, row in outstanding.iterrows():
        series = prices.get(row["symbol"])
        if series is None or series.empty:
            continue

        target = pd.Timestamp(row["target_date"])
        onward = series[series.index >= target]
        if onward.empty:
            continue

        conn.execute(
            "UPDATE forecasts SET actual = ?, settled_at = ? WHERE id = ?",
            (float(onward.iloc[0]), now, int(row["id"])),
        )
        updated += 1

    conn.commit()
    return updated


def record_stats(conn: sqlite3.Connection) -> dict[str, float]:
    """Skill and calibration over the settled entries.

    Because the price at forecast time is stored, the naive guess can be
    rebuilt after the fact and the settled record scored against it — a
    forward-only scorecard, computed without re-running any model.
    """
    frame = settled(conn)
    stats: dict[str, float] = {
        "recorded": float(len(pd.read_sql_query("SELECT id FROM forecasts", conn))),
        "settled": float(len(frame)),
        "pending": float(len(pending(conn))),
    }
    if frame.empty:
        return stats

    frame = frame.dropna(subset=["actual", "point"])
    error = (frame["actual"] - frame["point"]).abs()
    stats["mae"] = float(error.mean())

    naive = frame.dropna(subset=["origin_price"])
    if not naive.empty:
        naive_error = (naive["actual"] - naive["origin_price"]).abs()
        stats["naive_mae"] = float(naive_error.mean())
        stats["skill"] = (
            1.0 - float(error[naive.index].mean() / naive_error.mean())
            if naive_error.mean()
            else 0.0
        )
        stats["beat_naive"] = float((error[naive.index] < naive_error).mean())

    banded = frame.dropna(subset=["lower", "upper"])
    if not banded.empty:
        inside = (banded["actual"] >= banded["lower"]) & (
            banded["actual"] <= banded["upper"]
        )
        stats["coverage"] = float(inside.mean())

    return stats
