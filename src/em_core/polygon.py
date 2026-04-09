"""
Polygon.io REST client for em-core.

Exposes both a synchronous and an asynchronous client sharing the same URL
builders, response parsers, and exception hierarchy. The EM Dashboard (Flask,
sync, ThreadPoolExecutor) uses :class:`SyncPolygonClient`; the TripWire bot
(python-telegram-bot, asyncio) uses :class:`AsyncPolygonClient`.

Both clients expose the same method surface so dashboard and bot code can share
helpers in :mod:`em_core.iv`, :mod:`em_core.earnings`, etc.

Environment
-----------
Reads ``POLYGON_API_KEY`` from the environment by default. Override by passing
``api_key=`` to the client constructors.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
DEFAULT_TIMEOUT = 10.0

# Index tickers need special handling — Polygon aggs use I:SPX style,
# snapshot endpoints use I:SPX, but the dashboard's display layer uses "SPX".
INDEX_TICKERS: dict[str, str] = {
    "SPX": "I:SPX",
    "NDX": "I:NDX",
    "RUT": "I:RUT",
    "VIX": "I:VIX",
    "DJI": "I:DJI",
}


# ─── EXCEPTIONS ─────────────────────────────────────────────────────────
class PolygonError(Exception):
    """Base class for all Polygon client errors."""


class TickerNotFound(PolygonError):
    """Ticker does not exist or has no data for the requested endpoint."""


class PolygonRateLimited(PolygonError):
    """Polygon returned HTTP 429."""


class PolygonAuthError(PolygonError):
    """Polygon returned HTTP 401 or 403 — check POLYGON_API_KEY."""


# ─── URL BUILDERS (pure, shared by sync + async) ─────────────────────────
def _resolve_ticker(ticker: str) -> str:
    """Map display ticker (SPX) to Polygon API ticker (I:SPX). Non-index
    tickers pass through unchanged."""
    t = ticker.upper()
    return INDEX_TICKERS.get(t, t)


def _url_daily_bars(ticker: str, start: str, end: str) -> str:
    api_ticker = _resolve_ticker(ticker)
    return f"/v2/aggs/ticker/{api_ticker}/range/1/day/{start}/{end}"


def _url_options_snapshot(underlying: str) -> str:
    return f"/v3/snapshot/options/{underlying.upper()}"


def _url_stock_snapshot(ticker: str) -> str:
    return f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}"


def _url_universal_snapshot() -> str:
    return "/v3/snapshot"


# ─── RESPONSE HANDLING ───────────────────────────────────────────────────
def _handle_status(status_code: int, body_preview: str, ticker: str | None = None) -> None:
    """Raise the appropriate PolygonError subclass for a non-2xx response."""
    if status_code == 404:
        raise TickerNotFound(
            f"Polygon 404 for ticker {ticker!r}" if ticker else "Polygon 404"
        )
    if status_code == 429:
        raise PolygonRateLimited("Polygon rate limit hit (HTTP 429)")
    if status_code in (401, 403):
        raise PolygonAuthError(
            f"Polygon auth error HTTP {status_code}: {body_preview[:200]}"
        )
    if status_code >= 400:
        raise PolygonError(f"Polygon HTTP {status_code}: {body_preview[:200]}")


def _parse_daily_bars(data: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not data or not data.get("results"):
        return None
    return data["results"]


# ─── BASE CLIENT (shared config) ─────────────────────────────────────────
class _BasePolygonClient:
    """Config holder shared by sync and async variants."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self.api_key:
            log.warning("PolygonClient initialized with no API key")
        self.base_url = base_url
        self.timeout = timeout

    def _params_with_auth(self, params: dict[str, Any] | None) -> dict[str, Any]:
        out = dict(params) if params else {}
        out["apiKey"] = self.api_key
        return out


# ─── SYNC CLIENT (for EM Dashboard) ──────────────────────────────────────
class SyncPolygonClient(_BasePolygonClient):
    """Synchronous Polygon client built on ``httpx.Client``.

    Safe to call from Flask request handlers and from threads in a
    ``ThreadPoolExecutor``. Each client instance holds one underlying
    connection pool; create one per process and share it across threads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=3.0),
        )

    # context manager support
    def __enter__(self) -> "SyncPolygonClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, Any] | None = None, ticker: str | None = None) -> dict[str, Any]:
        full_params = self._params_with_auth(params)
        try:
            resp = self._client.get(path, params=full_params)
        except httpx.ConnectError as e:
            raise PolygonError(f"Connection error: {e}") from e
        except httpx.TimeoutException as e:
            raise PolygonError(f"Timeout: {e}") from e
        _handle_status(resp.status_code, resp.text, ticker=ticker)
        return resp.json()

    # ─── public methods ──────────────────────────────────────────────
    def get_daily_bars(self, ticker: str, days: int = 55) -> list[dict[str, Any]] | None:
        """Return the last ``days`` daily bars for ``ticker``, newest first."""
        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 40)).strftime("%Y-%m-%d")
        data = self._get(
            _url_daily_bars(ticker, start, today),
            {"adjusted": "true", "sort": "desc", "limit": str(days + 5)},
            ticker=ticker,
        )
        return _parse_daily_bars(data)

    def get_options_snapshot(
        self,
        underlying: str,
        expiration_date: str | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        """Fetch option contracts for ``underlying``. Returns the raw
        ``results`` list (may be empty)."""
        params: dict[str, Any] = {"limit": str(limit)}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date.gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date.lte"] = expiration_date_lte
        data = self._get(_url_options_snapshot(underlying), params, ticker=underlying)
        return data.get("results", []) or []

    def get_stock_snapshot(self, ticker: str) -> dict[str, Any] | None:
        """Fetch the v2 single-ticker snapshot."""
        try:
            data = self._get(_url_stock_snapshot(ticker), ticker=ticker)
        except TickerNotFound:
            return None
        return data.get("ticker")

    def get_universal_snapshot(self, tickers: list[str] | str) -> list[dict[str, Any]]:
        """Fetch v3 universal snapshot for one or many tickers (index-aware)."""
        if isinstance(tickers, str):
            tickers = [tickers]
        resolved = [_resolve_ticker(t) for t in tickers]
        data = self._get(
            _url_universal_snapshot(),
            {"ticker.any_of": ",".join(resolved)},
        )
        return data.get("results", []) or []


# ─── ASYNC CLIENT (for TripWire bot) ─────────────────────────────────────
class AsyncPolygonClient(_BasePolygonClient):
    """Asynchronous Polygon client built on ``httpx.AsyncClient``.

    Designed for use inside the TripWire bot's asyncio event loop. Create one
    client per application lifecycle (e.g. in ``_run()`` in ``bot/main.py``)
    and close it during shutdown. The client is safe to share across all
    coroutines in the same event loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "AsyncPolygonClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=3.0),
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise PolygonError("AsyncPolygonClient not started — call start() first")
        return self._client

    async def _get(self, path: str, params: dict[str, Any] | None = None, ticker: str | None = None) -> dict[str, Any]:
        client = self._require_client()
        full_params = self._params_with_auth(params)
        try:
            resp = await client.get(path, params=full_params)
        except httpx.ConnectError as e:
            raise PolygonError(f"Connection error: {e}") from e
        except httpx.TimeoutException as e:
            raise PolygonError(f"Timeout: {e}") from e
        _handle_status(resp.status_code, resp.text, ticker=ticker)
        return resp.json()

    # ─── public methods (same surface as SyncPolygonClient) ──────────
    async def get_daily_bars(self, ticker: str, days: int = 55) -> list[dict[str, Any]] | None:
        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days + 40)).strftime("%Y-%m-%d")
        data = await self._get(
            _url_daily_bars(ticker, start, today),
            {"adjusted": "true", "sort": "desc", "limit": str(days + 5)},
            ticker=ticker,
        )
        return _parse_daily_bars(data)

    async def get_options_snapshot(
        self,
        underlying: str,
        expiration_date: str | None = None,
        expiration_date_gte: str | None = None,
        expiration_date_lte: str | None = None,
        limit: int = 250,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": str(limit)}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date.gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date.lte"] = expiration_date_lte
        data = await self._get(_url_options_snapshot(underlying), params, ticker=underlying)
        return data.get("results", []) or []

    async def get_stock_snapshot(self, ticker: str) -> dict[str, Any] | None:
        try:
            data = await self._get(_url_stock_snapshot(ticker), ticker=ticker)
        except TickerNotFound:
            return None
        return data.get("ticker")

    async def get_universal_snapshot(self, tickers: list[str] | str) -> list[dict[str, Any]]:
        if isinstance(tickers, str):
            tickers = [tickers]
        resolved = [_resolve_ticker(t) for t in tickers]
        data = await self._get(
            _url_universal_snapshot(),
            {"ticker.any_of": ",".join(resolved)},
        )
        return data.get("results", []) or []


__all__ = [
    "BASE_URL",
    "INDEX_TICKERS",
    "PolygonError",
    "TickerNotFound",
    "PolygonRateLimited",
    "PolygonAuthError",
    "SyncPolygonClient",
    "AsyncPolygonClient",
]
