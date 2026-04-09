"""
Earnings calendar + session-window helper for em-core.

This module is deliberately provider-agnostic. Polygon's free tier doesn't
ship a clean "upcoming earnings" endpoint, and different accounts/tiers have
different data sources available (Polygon ticker events, Finnhub, FMP,
yfinance, hand-curated CSV, etc.). Rather than bake in one vendor, we expose:

1. A thin :class:`EarningsProvider` protocol \u2014 any callable that maps a
   ticker to a ``(next_earnings_iso, when_code)`` tuple.
2. :class:`EarningsService` which:
       - Wraps a provider with SQLite-backed caching via :class:`em_core.storage.Storage`
       - Exposes :meth:`refresh_all` for the nightly cron
       - Exposes :meth:`is_in_earnings_window` for the scanner + dashboard

Earnings window semantics
-------------------------
If a ticker reports before market open ("bmo") on trading day D, the
suppression window is D (the regular session after the report). If a ticker
reports after market close ("amc") on day D, the window is the NEXT trading
day (so D+1's regular session). If "when" is unknown, we suppress on the
announce date itself and on the following trading day, to be safe.

During a ticker's earnings window:
- TripWire scanner skips 1\u03c3/2\u03c3/3\u03c3 break alerts
- EM Dashboard surfaces an ``[EARNINGS]`` tag on the card
- Daily expected move should be computed from the options chain straddle,
  not the ATM IV (the straddle prices in the "event premium"). That override
  lives in :mod:`em_core.iv` and consults this module via
  :meth:`is_in_earnings_window`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Optional, Protocol

from .locking import ET, is_trading_day, next_trading_day
from .storage import EarningsRow, Storage

log = logging.getLogger(__name__)

# \u2500\u2500\u2500 PROVIDER PROTOCOL \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
ProviderResult = tuple[Optional[str], str]
"""\u0060\u0060(next_earnings_iso_date, when_code)\u0060\u0060 where when_code is one of
\u0060\u0060'bmo'\u0060\u0060 (before market open), \u0060\u0060'amc'\u0060\u0060 (after market close), or
\u0060\u0060'unknown'\u0060\u0060. \u0060\u0060None\u0060\u0060 date means no upcoming earnings found."""


class EarningsProvider(Protocol):
    """Anything callable that resolves next earnings for a ticker."""

    def __call__(self, ticker: str) -> ProviderResult: ...


# \u2500\u2500\u2500 SERVICE \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
@dataclass
class EarningsWindow:
    ticker: str
    announce_date: str           # ISO date of the announce
    session_date: str            # ISO date of the suppressed trading session
    when_code: str
    source: str                  # 'cache' | 'fresh' | 'unknown'


class EarningsService:
    """SQLite-cached earnings lookups with window semantics."""

    def __init__(
        self,
        storage: Storage,
        provider: EarningsProvider,
        ttl_seconds: int = 86_400,
    ):
        self.storage = storage
        self.provider = provider
        self.ttl_seconds = ttl_seconds

    # \u2500\u2500\u2500 CACHE HELPERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def _is_fresh(self, row: EarningsRow) -> bool:
        try:
            refreshed = datetime.fromisoformat(row.refreshed_at)
        except ValueError:
            return False
        age = (datetime.utcnow() - refreshed).total_seconds()
        return age <= row.ttl_seconds

    def get(self, ticker: str, *, force_refresh: bool = False) -> Optional[EarningsRow]:
        """Return the cached earnings row, refreshing via the provider if stale."""
        ticker = ticker.upper()
        row = self.storage.get_earnings(ticker)
        if row and not force_refresh and self._is_fresh(row):
            return row
        try:
            next_date, when_code = self.provider(ticker)
        except Exception as exc:  # provider failures must not take the scanner down
            log.warning("earnings provider failed for %s: %s", ticker, exc)
            return row  # stale is better than nothing
        self.storage.put_earnings(
            ticker, next_date, when_code=when_code or "unknown", ttl_seconds=self.ttl_seconds
        )
        return self.storage.get_earnings(ticker)

    def refresh_all(self, tickers: Iterable[str]) -> dict[str, int]:
        """Force-refresh a batch of tickers. Returns a small counters dict.

        Intended to be called nightly (e.g. via APScheduler at 02:00 ET) by
        the TripWire bot or the dashboard cron. Both apps can call this \u2014
        the underlying write is idempotent per ticker."""
        stats = {"ok": 0, "err": 0, "noop": 0}
        for t in tickers:
            try:
                before = self.storage.get_earnings(t)
                self.get(t, force_refresh=True)
                after = self.storage.get_earnings(t)
                if before and after and before.refreshed_at == after.refreshed_at:
                    stats["noop"] += 1
                else:
                    stats["ok"] += 1
            except Exception as exc:
                log.warning("refresh_all error for %s: %s", t, exc)
                stats["err"] += 1
        return stats

    # \u2500\u2500\u2500 WINDOW LOGIC \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    def compute_window(self, row: EarningsRow) -> Optional[EarningsWindow]:
        """Convert a cached earnings row into a suppressed trading session."""
        if not row.next_earnings:
            return None
        try:
            announce = date.fromisoformat(row.next_earnings)
        except ValueError:
            return None

        when = (row.when_code or "unknown").lower()
        if when == "bmo":
            session = announce
        elif when == "amc":
            session = next_trading_day(announce)
        else:
            # Unknown \u2192 assume AMC (more conservative, covers overnight gap into D+1)
            session = next_trading_day(announce)
        # If the announce date itself is a non-trading day (weekend/holiday),
        # bump the session forward to the next trading day.
        while not is_trading_day(session):
            session = next_trading_day(session)
        return EarningsWindow(
            ticker=row.ticker,
            announce_date=announce.isoformat(),
            session_date=session.isoformat(),
            when_code=when,
            source="cache",
        )

    def is_in_earnings_window(
        self,
        ticker: str,
        now_et: Optional[datetime] = None,
    ) -> bool:
        """Return True iff ``ticker`` is inside its suppressed trading session.

        Reads from cache only \u2014 the scanner calls this once per ticker per
        tick and must not stall on network IO. Nightly ``refresh_all`` is
        responsible for keeping the cache warm."""
        if now_et is None:
            now_et = datetime.now(ET)
        today = now_et.astimezone(ET).date().isoformat()
        row = self.storage.get_earnings(ticker.upper())
        if row is None:
            return False
        window = self.compute_window(row)
        if window is None:
            return False
        return window.session_date == today

    def days_until(self, ticker: str, now_et: Optional[datetime] = None) -> Optional[int]:
        """Calendar days from today to the announce date, or ``None`` if unknown."""
        row = self.storage.get_earnings(ticker.upper())
        if row is None or not row.next_earnings:
            return None
        try:
            announce = date.fromisoformat(row.next_earnings)
        except ValueError:
            return None
        today = (now_et or datetime.now(ET)).astimezone(ET).date()
        return (announce - today).days


# \u2500\u2500\u2500 BUILT-IN PROVIDERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def static_provider(table: dict[str, ProviderResult]) -> EarningsProvider:
    """Build a provider backed by an in-memory dict. Useful for tests and
    for manually seeding the cache from a hand-curated CSV."""
    def _lookup(ticker: str) -> ProviderResult:
        return table.get(ticker.upper(), (None, "unknown"))
    return _lookup


def null_provider() -> EarningsProvider:
    """Provider that always returns 'no upcoming earnings'. Use when the app
    has no earnings data source configured \u2014 scanner will never suppress on
    earnings, dashboard will never render the [EARNINGS] tag."""
    def _lookup(ticker: str) -> ProviderResult:  # noqa: ARG001
        return (None, "unknown")
    return _lookup


def polygon_ticker_events_provider(
    client,  # SyncPolygonClient or AsyncPolygonClient \u2014 sync-only callsite here
) -> EarningsProvider:
    """Provider that hits Polygon's ``/vX/reference/tickers/{ticker}/events``
    endpoint and picks the nearest future ``ticker_change`` / earnings
    announcement. NOTE: Polygon's public earnings coverage is thin; this
    provider returns ``(None, 'unknown')`` if nothing parseable is found.

    This is a *sync* wrapper \u2014 intended for the nightly refresh cron in
    either the dashboard or the bot's startup hook, not the per-tick path.
    """
    def _lookup(ticker: str) -> ProviderResult:
        try:
            path = f"/vX/reference/tickers/{ticker.upper()}/events"
            data = client._get(path, {"types": "ticker_change"}, ticker=ticker)  # noqa: SLF001
        except Exception as exc:
            log.debug("polygon events fetch failed for %s: %s", ticker, exc)
            return (None, "unknown")
        events = (data or {}).get("results", {}).get("events", []) or []
        today = date.today().isoformat()
        upcoming = sorted(
            (e for e in events if (e.get("date") or "") >= today),
            key=lambda e: e.get("date", ""),
        )
        if not upcoming:
            return (None, "unknown")
        return (upcoming[0].get("date"), "unknown")
    return _lookup


__all__ = [
    "EarningsProvider",
    "ProviderResult",
    "EarningsService",
    "EarningsWindow",
    "static_provider",
    "null_provider",
    "polygon_ticker_events_provider",
]
