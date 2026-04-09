"""
SQLite persistence layer for em-core.

This is the shared on-disk state for the TripWire Telegram bot and the EM
Dashboard. Both apps open the *same* SQLite file so that:

- A ticker the user adds via ``/watch`` on Telegram shows up on the dashboard's
  "My Watchlist" page, and vice versa.
- Expected move levels computed by the dashboard during the trading day are
  cached and reused by the bot's scanner on the next tick (lock semantics
  live in :mod:`em_core.locking`).
- Earnings data fetched nightly is shared between the scanner's
  suppression logic and the dashboard's [EARNINGS] tag.
- The scanner can dedupe alerts across restarts so a user doesn't get pinged
  twice for the same 1\u03c3 break.

File location
-------------
By default we look at ``TRIPWIRE_DB_PATH`` env var, falling back to
``~/tripwire-bot/tripwire.db``. Callers can always override by passing
``db_path=`` explicitly.

Schema
------
- ``watchlists``         \u2014 per-chat ticker subscriptions (bot + dashboard)
- ``levels_cache``       \u2014 cached daily/weekly/monthly expected move levels
- ``alert_dedupe``       \u2014 ``(chat_id, ticker, level, lock_key)`` once-only alerts
- ``earnings_cache``     \u2014 next-earnings date per ticker with TTL

Thread/async safety
-------------------
All methods go through a single ``sqlite3.Connection`` opened with
``check_same_thread=False`` and guarded by a :class:`threading.Lock`. That
keeps sync Flask handlers and async bot coroutines from stepping on each
other. SQLite itself serializes writes internally; the lock only prevents
cursor-sharing races.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

# \u2500\u2500\u2500 DEFAULT PATH \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def _default_db_path() -> str:
    env = os.environ.get("TRIPWIRE_DB_PATH")
    if env:
        return env
    return str(Path.home() / "tripwire-bot" / "tripwire.db")


# \u2500\u2500\u2500 SCHEMA \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlists (
    chat_id       INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    direction     TEXT    NOT NULL DEFAULT 'neutral',
    entry_price   REAL,
    added_at      TEXT    NOT NULL,
    meta_json     TEXT,
    PRIMARY KEY (chat_id, ticker)
);

CREATE INDEX IF NOT EXISTS idx_watchlists_ticker ON watchlists(ticker);

CREATE TABLE IF NOT EXISTS levels_cache (
    ticker        TEXT    NOT NULL,
    timeframe     TEXT    NOT NULL,      -- 'daily' | 'weekly' | 'monthly'
    price         REAL    NOT NULL,
    iv            REAL    NOT NULL,
    one_sigma     REAL    NOT NULL,
    levels_json   TEXT    NOT NULL,      -- em_core.iv.expected_move_levels output
    source        TEXT    NOT NULL,      -- 'iv' | 'straddle' | 'options_chain'
    expiration    TEXT,                  -- contract expiration ISO date (for IV source)
    dte           INTEGER,
    fixed_at      TEXT    NOT NULL,      -- when this level was locked (ISO)
    expires_at    TEXT,                  -- lock expiry (ISO, nullable for daily)
    earnings_flag INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, timeframe)
);

CREATE TABLE IF NOT EXISTS alert_dedupe (
    chat_id       INTEGER NOT NULL,
    ticker        TEXT    NOT NULL,
    level_name    TEXT    NOT NULL,      -- e.g. '1sigma_upper'
    lock_key      TEXT    NOT NULL,      -- e.g. '2026-04-04' (daily) or '2026-04-10' (weekly)
    fired_at      TEXT    NOT NULL,
    PRIMARY KEY (chat_id, ticker, level_name, lock_key)
);

CREATE INDEX IF NOT EXISTS idx_alert_dedupe_ticker ON alert_dedupe(ticker);

CREATE TABLE IF NOT EXISTS earnings_cache (
    ticker          TEXT PRIMARY KEY,
    next_earnings   TEXT,                -- ISO date or NULL if unknown
    when_code       TEXT,                -- 'bmo' | 'amc' | 'unknown'
    refreshed_at    TEXT    NOT NULL,
    ttl_seconds     INTEGER NOT NULL DEFAULT 86400
);

CREATE TABLE IF NOT EXISTS kv_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


# \u2500\u2500\u2500 DATACLASSES \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
@dataclass(frozen=True)
class WatchRow:
    chat_id: int
    ticker: str
    direction: str
    entry_price: Optional[float]
    added_at: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class LevelRow:
    ticker: str
    timeframe: str
    price: float
    iv: float
    one_sigma: float
    levels: dict[str, float]
    source: str
    expiration: Optional[str]
    dte: Optional[int]
    fixed_at: str
    expires_at: Optional[str]
    earnings_flag: bool


@dataclass(frozen=True)
class EarningsRow:
    ticker: str
    next_earnings: Optional[str]
    when_code: str
    refreshed_at: str
    ttl_seconds: int


# \u2500\u2500\u2500 STORAGE CLASS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
class Storage:
    """Thin SQLite wrapper \u2014 one instance per process, shared across threads/tasks."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage txns explicitly
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # \u2500\u2500\u2500 WATCHLISTS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def add_watch(
        self,
        chat_id: int,
        ticker: str,
        direction: str = "neutral",
        entry_price: Optional[float] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        row = {
            "chat_id": chat_id,
            "ticker": ticker.upper(),
            "direction": direction,
            "entry_price": entry_price,
            "added_at": datetime.utcnow().isoformat(),
            "meta_json": json.dumps(meta or {}),
        }
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO watchlists(chat_id, ticker, direction, entry_price, added_at, meta_json)
                VALUES(:chat_id, :ticker, :direction, :entry_price, :added_at, :meta_json)
                ON CONFLICT(chat_id, ticker) DO UPDATE SET
                    direction=excluded.direction,
                    entry_price=excluded.entry_price,
                    meta_json=excluded.meta_json
                """,
                row,
            )

    def remove_watch(self, chat_id: int, ticker: str) -> bool:
        with self._tx() as c:
            cur = c.execute(
                "DELETE FROM watchlists WHERE chat_id=? AND ticker=?",
                (chat_id, ticker.upper()),
            )
            return cur.rowcount > 0

    def get_watches(self, chat_id: int) -> list[WatchRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM watchlists WHERE chat_id=? ORDER BY added_at",
                (chat_id,),
            ).fetchall()
        return [self._row_to_watch(r) for r in rows]

    def get_all_watches(self) -> list[WatchRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM watchlists ORDER BY chat_id, ticker"
            ).fetchall()
        return [self._row_to_watch(r) for r in rows]

    def get_unique_watched_tickers(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT ticker FROM watchlists ORDER BY ticker"
            ).fetchall()
        return [r["ticker"] for r in rows]

    @staticmethod
    def _row_to_watch(r: sqlite3.Row) -> WatchRow:
        return WatchRow(
            chat_id=r["chat_id"],
            ticker=r["ticker"],
            direction=r["direction"],
            entry_price=r["entry_price"],
            added_at=r["added_at"],
            meta=json.loads(r["meta_json"] or "{}"),
        )

    # \u2500\u2500\u2500 LEVELS CACHE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def put_level(self, row: LevelRow) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO levels_cache(
                    ticker, timeframe, price, iv, one_sigma, levels_json,
                    source, expiration, dte, fixed_at, expires_at, earnings_flag
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, timeframe) DO UPDATE SET
                    price=excluded.price,
                    iv=excluded.iv,
                    one_sigma=excluded.one_sigma,
                    levels_json=excluded.levels_json,
                    source=excluded.source,
                    expiration=excluded.expiration,
                    dte=excluded.dte,
                    fixed_at=excluded.fixed_at,
                    expires_at=excluded.expires_at,
                    earnings_flag=excluded.earnings_flag
                """,
                (
                    row.ticker.upper(),
                    row.timeframe,
                    row.price,
                    row.iv,
                    row.one_sigma,
                    json.dumps(row.levels),
                    row.source,
                    row.expiration,
                    row.dte,
                    row.fixed_at,
                    row.expires_at,
                    1 if row.earnings_flag else 0,
                ),
            )

    def get_level(self, ticker: str, timeframe: str) -> Optional[LevelRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM levels_cache WHERE ticker=? AND timeframe=?",
                (ticker.upper(), timeframe),
            ).fetchone()
        if r is None:
            return None
        return LevelRow(
            ticker=r["ticker"],
            timeframe=r["timeframe"],
            price=r["price"],
            iv=r["iv"],
            one_sigma=r["one_sigma"],
            levels=json.loads(r["levels_json"]),
            source=r["source"],
            expiration=r["expiration"],
            dte=r["dte"],
            fixed_at=r["fixed_at"],
            expires_at=r["expires_at"],
            earnings_flag=bool(r["earnings_flag"]),
        )

    def clear_level(self, ticker: str, timeframe: Optional[str] = None) -> int:
        with self._tx() as c:
            if timeframe:
                cur = c.execute(
                    "DELETE FROM levels_cache WHERE ticker=? AND timeframe=?",
                    (ticker.upper(), timeframe),
                )
            else:
                cur = c.execute(
                    "DELETE FROM levels_cache WHERE ticker=?", (ticker.upper(),)
                )
            return cur.rowcount

    # \u2500\u2500\u2500 ALERT DEDUPE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def try_claim_alert(
        self,
        chat_id: int,
        ticker: str,
        level_name: str,
        lock_key: str,
    ) -> bool:
        """Atomically record that an alert has fired. Returns ``True`` if this
        call is the one that actually inserted it (caller should send the
        alert); returns ``False`` if a previous call already claimed it (caller
        should suppress)."""
        try:
            with self._tx() as c:
                c.execute(
                    """
                    INSERT INTO alert_dedupe(chat_id, ticker, level_name, lock_key, fired_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        ticker.upper(),
                        level_name,
                        lock_key,
                        datetime.utcnow().isoformat(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def purge_alert_dedupe(self, older_than_days: int = 45) -> int:
        cutoff = datetime.utcnow().timestamp() - older_than_days * 86400
        cutoff_iso = datetime.utcfromtimestamp(cutoff).isoformat()
        with self._tx() as c:
            cur = c.execute(
                "DELETE FROM alert_dedupe WHERE fired_at < ?", (cutoff_iso,)
            )
            return cur.rowcount

    # \u2500\u2500\u2500 EARNINGS CACHE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def put_earnings(
        self,
        ticker: str,
        next_earnings: Optional[str],
        when_code: str = "unknown",
        ttl_seconds: int = 86400,
    ) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO earnings_cache(ticker, next_earnings, when_code, refreshed_at, ttl_seconds)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    next_earnings=excluded.next_earnings,
                    when_code=excluded.when_code,
                    refreshed_at=excluded.refreshed_at,
                    ttl_seconds=excluded.ttl_seconds
                """,
                (
                    ticker.upper(),
                    next_earnings,
                    when_code,
                    datetime.utcnow().isoformat(),
                    ttl_seconds,
                ),
            )

    def get_earnings(self, ticker: str) -> Optional[EarningsRow]:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM earnings_cache WHERE ticker=?", (ticker.upper(),)
            ).fetchone()
        if r is None:
            return None
        return EarningsRow(
            ticker=r["ticker"],
            next_earnings=r["next_earnings"],
            when_code=r["when_code"],
            refreshed_at=r["refreshed_at"],
            ttl_seconds=r["ttl_seconds"],
        )

    # \u2500\u2500\u2500 KV STATE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def kv_get(self, key: str) -> Optional[str]:
        with self._lock:
            r = self._conn.execute(
                "SELECT value FROM kv_state WHERE key=?", (key,)
            ).fetchone()
        return r["value"] if r else None

    def kv_set(self, key: str, value: str) -> None:
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO kv_state(key, value, updated_at) VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, datetime.utcnow().isoformat()),
            )


__all__ = [
    "Storage",
    "WatchRow",
    "LevelRow",
    "EarningsRow",
    "SCHEMA",
]
