"""
Implied volatility and expected-move math for em-core.

This module is the single source of truth for IV, realized vol, expected-move,
ATM straddle extraction, and confidence scoring used by both the EM Dashboard
and the TripWire Telegram bot.

Convention decisions
--------------------
- **252 trading days per year** is the time base for all sqrt(T) math. This
  matches the dashboard's production behavior and is the canonical convention
  used in equity volatility modeling.
- **IV is passed as a percentage** (e.g. ``22.5`` for 22.5% annualized vol),
  matching the dashboard's internal representation. Internal math divides by
  100 automatically.
- **Expected move dicts always include 1σ, 2σ, and 3σ** levels so callers
  (dashboard cards, TripWire 1σ/2σ/3σ alerts) share a single output shape.
- **Returned floats are rounded to 4 decimal places**, also matching the
  dashboard. Callers render with 2dp as needed.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

TRADING_DAYS_PER_YEAR = 252
DEFAULT_WEEKLY_TRADING_DAYS = 5


# ─── EXPECTED MOVE (generic + convenience wrappers) ──────────────────────
def expected_move_levels(
    price: float,
    one_sigma: float,
    center: float | None = None,
) -> dict[str, float]:
    """Build a dashboard-style EM level dict from a 1σ dollar move.

    Returns sigma1/2/3 dollar amounts, low/high bands at each sigma, and the
    center (mid) used to anchor the bands.
    """
    if center is None:
        center = price
    return {
        "sigma1": round(one_sigma, 4),
        "sigma2": round(one_sigma * 2, 4),
        "sigma3": round(one_sigma * 3, 4),
        "low1": round(center - one_sigma, 4),
        "high1": round(center + one_sigma, 4),
        "low2": round(center - one_sigma * 2, 4),
        "high2": round(center + one_sigma * 2, 4),
        "low3": round(center - one_sigma * 3, 4),
        "high3": round(center + one_sigma * 3, 4),
        "mid": round(center, 4),
    }


def calc_expected_move(
    price: float,
    iv_pct: float,
    periods: float,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
    center: float | None = None,
) -> dict[str, float]:
    """Generic sqrt(T) expected move.

    ``iv_pct`` is annualized IV in percent (e.g. ``22.5``). ``periods`` is the
    number of time periods to project forward, expressed in the same units as
    ``periods_per_year``. Default is trading days (252/year).
    """
    if center is None:
        center = price
    one_sigma = center * (iv_pct / 100.0) * math.sqrt(periods / periods_per_year)
    return expected_move_levels(price, one_sigma, center=center)


def calc_daily_em_iv(
    price: float,
    iv_pct: float,
    center: float | None = None,
) -> dict[str, float]:
    """Daily expected move from annualized IV. Price × IV / sqrt(252)."""
    return calc_expected_move(price, iv_pct, periods=1, center=center)


def calc_weekly_em_iv(
    price: float,
    iv_pct: float,
    trading_days: int | None = None,
    center: float | None = None,
) -> dict[str, float]:
    """Weekly expected move from annualized IV.

    ``trading_days`` defaults to 5 (full M-F week). Pass a smaller number for
    shortened weeks (e.g. 4 for a Good Friday week).
    """
    if trading_days is None:
        trading_days = DEFAULT_WEEKLY_TRADING_DAYS
    return calc_expected_move(price, iv_pct, periods=trading_days, center=center)


def calc_daily_em_straddle(
    price: float,
    straddle_price: float,
    center: float | None = None,
) -> dict[str, float]:
    """Daily expected move from a 0–1 DTE ATM straddle premium.

    For very short-dated straddles, the straddle premium is approximately one
    standard-deviation move (slightly overstates σ but is within a few
    percent at 0–1 DTE).
    """
    return expected_move_levels(price, straddle_price, center=center)


def calc_weekly_em_straddle(
    price: float,
    straddle_price: float,
    center: float | None = None,
) -> dict[str, float]:
    """Weekly expected move from ATM straddle: (straddle / price) × 0.85 × price = 1σ.

    The 0.85 factor converts a full straddle premium to an approximate 1σ
    move for weekly expirations. Matches dashboard production behavior.
    """
    if center is None:
        center = price
    one_sigma = price * (straddle_price / price) * 0.85
    return expected_move_levels(price, one_sigma, center=center)


# ─── REALIZED VOLATILITY ─────────────────────────────────────────────────
def calc_historical_vol(
    bars: list[dict[str, Any]] | None,
    lookback: int = 20,
) -> float | None:
    """Close-to-close log-return realized volatility, annualized.

    ``bars`` is a list of Polygon daily bars ordered **newest first** (as
    returned by ``AsyncPolygonClient.get_daily_bars`` / ``SyncPolygonClient.get_daily_bars``
    with ``sort=desc``). Each bar must have a ``c`` (close) field.

    Returns the annualized vol **as a percent** (e.g. 18.3 for 18.3%), or
    ``None`` if there are not enough bars or they are invalid.
    """
    if not bars or len(bars) < lookback + 1:
        return None
    closes = [b.get("c", 0) for b in bars[:lookback + 1]]
    returns: list[float] = []
    for i in range(len(closes) - 1):
        if closes[i] > 0 and closes[i + 1] > 0:
            returns.append(math.log(closes[i] / closes[i + 1]))
    if len(returns) < 5:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance * TRADING_DAYS_PER_YEAR) * 100


# ─── IV RATE OF CHANGE ───────────────────────────────────────────────────
def calc_iv_roc(history: list[dict[str, Any]] | None) -> dict[str, float | None]:
    """Compute IV rate-of-change from an IV history series.

    ``history`` is a list of dicts ordered **newest first**, where
    ``history[0]`` is today, ``history[1]`` is 1 day ago, etc. Each entry
    must have an ``iv`` field.

    Returns a dict with keys: ``iv_roc_1d``, ``iv_roc_5d``, ``iv_roc_20d``,
    ``iv_roc_accel`` (acceleration = today's 5d ROC minus yesterday's 5d ROC).
    Values are percent changes rounded to 1dp, or ``None`` if not enough history.
    """
    roc: dict[str, float | None] = {
        "iv_roc_1d": None,
        "iv_roc_5d": None,
        "iv_roc_20d": None,
        "iv_roc_accel": None,
    }
    if not history or len(history) < 2:
        return roc
    current_iv = history[0].get("iv", 0)
    if current_iv <= 0:
        return roc
    # 1-day ROC
    if len(history) >= 2 and history[1].get("iv", 0) > 0:
        roc["iv_roc_1d"] = round((current_iv - history[1]["iv"]) / history[1]["iv"] * 100, 1)
    # 5-day ROC
    if len(history) >= 6 and history[5].get("iv", 0) > 0:
        roc["iv_roc_5d"] = round((current_iv - history[5]["iv"]) / history[5]["iv"] * 100, 1)
    # 20-day ROC
    if len(history) >= 21 and history[20].get("iv", 0) > 0:
        roc["iv_roc_20d"] = round((current_iv - history[20]["iv"]) / history[20]["iv"] * 100, 1)
    # Acceleration: today's 5d ROC minus yesterday's 5d ROC
    if len(history) >= 7 and roc["iv_roc_5d"] is not None:
        if history[1].get("iv", 0) > 0 and history[6].get("iv", 0) > 0:
            yesterday_5d = (history[1]["iv"] - history[6]["iv"]) / history[6]["iv"] * 100
            roc["iv_roc_accel"] = round(roc["iv_roc_5d"] - yesterday_5d, 1)
    return roc


# ─── ATM IV EXTRACTION ───────────────────────────────────────────────────
def extract_atm_iv_average(
    contracts: Iterable[dict[str, Any]],
    current_price: float,
    n_nearest: int = 6,
) -> float | None:
    """Extract ATM IV by averaging the ``n_nearest`` strikes to spot.

    Dashboard production convention. Returns IV **as a percent** (e.g. 22.5),
    or ``None`` if no contracts have usable IV.
    """
    with_iv = [
        c for c in contracts
        if c.get("implied_volatility") and c["implied_volatility"] > 0
    ]
    if not with_iv:
        return None
    with_iv.sort(
        key=lambda o: abs(o.get("details", {}).get("strike_price", 0) - current_price)
    )
    atm = with_iv[:n_nearest]
    if not atm:
        return None
    avg_iv = sum(o["implied_volatility"] for o in atm) / len(atm)
    return avg_iv * 100


def extract_atm_iv_interpolated(
    contracts: Iterable[dict[str, Any]],
    expiration: str,
    current_price: float,
) -> dict[str, Any] | None:
    """Extract ATM IV by interpolating between the two strikes bracketing spot.

    TripWire-native convention — more precise at a single strike point but
    noisier than the averaged version when strikes are sparse. Restricts to
    contracts matching the given ``expiration``.

    Returns a dict with keys ``atm_iv`` (fraction, e.g. 0.225 for 22.5%),
    ``expiration``, ``dte``, ``atm_strike`` — or ``None`` if interpolation
    is not possible.
    """
    from datetime import date

    matching = [
        c for c in contracts
        if c.get("details", {}).get("expiration_date") == expiration
        and c.get("implied_volatility") is not None
    ]
    if not matching:
        return None

    # Group by strike, averaging call + put IV at each strike for a stable estimate
    strikes: dict[float, dict[str, dict]] = {}
    for c in matching:
        strike = c["details"]["strike_price"]
        ctype = c["details"]["contract_type"]
        strikes.setdefault(strike, {})[ctype] = c

    strike_ivs: list[tuple[float, float]] = []
    for strike, types in strikes.items():
        ivs = [
            types[t]["implied_volatility"]
            for t in ("call", "put")
            if t in types and types[t].get("implied_volatility")
        ]
        if ivs:
            strike_ivs.append((strike, sum(ivs) / len(ivs)))

    if not strike_ivs:
        return None

    strike_ivs.sort(key=lambda x: abs(x[0] - current_price))
    if len(strike_ivs) == 1:
        atm_iv = strike_ivs[0][1]
        atm_strike = strike_ivs[0][0]
    else:
        s1, iv1 = strike_ivs[0]
        s2, iv2 = strike_ivs[1]
        gap = abs(s2 - s1)
        if gap == 0:
            atm_iv = (iv1 + iv2) / 2
        else:
            w1 = 1 - abs(current_price - s1) / gap
            w1 = max(0.0, min(1.0, w1))
            w2 = 1 - w1
            atm_iv = iv1 * w1 + iv2 * w2
        atm_strike = s1

    exp_date = date.fromisoformat(expiration)
    dte = max((exp_date - date.today()).days, 1)
    return {
        "atm_iv": atm_iv / 100.0,
        "expiration": expiration,
        "dte": dte,
        "atm_strike": atm_strike,
    }


# ─── ATM STRADDLE EXTRACTION ─────────────────────────────────────────────
def extract_atm_straddle(
    contracts: Iterable[dict[str, Any]],
    price: float,
) -> dict[str, float] | None:
    """Extract an ATM straddle (call mid + put mid at the strike nearest spot).

    Uses bid-ask mid when available, falling back to last trade price.
    Returns ``{"straddle": <total premium>, "strike": <atm strike>}`` or
    ``None`` if the chain lacks both call and put quotes at a common strike.
    """
    calls: list[dict[str, float]] = []
    puts: list[dict[str, float]] = []
    for o in contracts:
        details = o.get("details", {})
        ctype = details.get("contract_type", "")
        strike = details.get("strike_price", 0) or 0
        last_quote = o.get("last_quote", {}) or {}
        bid = last_quote.get("bid", 0) or 0
        ask = last_quote.get("ask", 0) or 0
        mid_price: float = 0.0
        if bid > 0 and ask > 0:
            mid_price = (bid + ask) / 2
        elif o.get("last_trade", {}).get("price"):
            mid_price = o["last_trade"]["price"]
        if strike > 0 and mid_price > 0:
            if ctype == "call":
                calls.append({"strike": strike, "mid": mid_price})
            elif ctype == "put":
                puts.append({"strike": strike, "mid": mid_price})

    if not calls or not puts:
        return None

    common_strikes = sorted(
        set(c["strike"] for c in calls) & set(p["strike"] for p in puts),
        key=lambda s: abs(s - price),
    )
    if not common_strikes:
        return None

    atm_strike = common_strikes[0]
    atm_call = next((c for c in calls if c["strike"] == atm_strike), None)
    atm_put = next((p for p in puts if p["strike"] == atm_strike), None)
    if not atm_call or not atm_put:
        return None

    return {"straddle": atm_call["mid"] + atm_put["mid"], "strike": atm_strike}


# ─── EXPIRATION HELPERS ──────────────────────────────────────────────────
def find_nearest_expiration(
    contracts: Iterable[dict[str, Any]],
    today_iso: str,
    prefer_within_days: int = 7,
) -> str | None:
    """Return the nearest usable expiration date in a contracts list.

    Prefers the closest expiration within ``prefer_within_days`` calendar days
    of ``today_iso`` (YYYY-MM-DD); falls back to the next available expiration
    beyond that window.
    """
    from datetime import date

    expirations: set[str] = set()
    for c in contracts:
        exp = c.get("details", {}).get("expiration_date")
        if exp:
            expirations.add(exp)
    if not expirations:
        return None

    future = sorted(e for e in expirations if e >= today_iso)
    if not future:
        return None

    today_d = date.fromisoformat(today_iso)
    for exp in future:
        exp_d = date.fromisoformat(exp)
        if (exp_d - today_d).days <= prefer_within_days:
            return exp
    return future[0]


# ─── CONFIDENCE SCORE ────────────────────────────────────────────────────
def compute_confidence_score(
    chain_subset: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Composite confidence score (0–100) for an options chain subset.

    Weights:
      - 30%: open interest (capped at 5000)
      - 25%: volume (capped at 1000)
      - 25%: IV completeness (% of contracts with usable IV)
      - 20%: bid-ask spread (lower = better)

    Returns ``{"score": composite, "oi": ..., "volume": ..., "iv_completeness": ..., "spread": ...}``
    with each subscore already scaled 0–100.
    """
    chain_list = list(chain_subset)
    if not chain_list:
        return {"score": 0, "oi": 0, "volume": 0, "iv_completeness": 0, "spread": 0}

    total_oi = sum(c.get("open_interest", 0) or 0 for c in chain_list)
    total_vol = sum((c.get("day", {}) or {}).get("volume", 0) or 0 for c in chain_list)

    iv_count = sum(1 for c in chain_list if c.get("implied_volatility") is not None)
    iv_completeness = iv_count / len(chain_list) if chain_list else 0.0

    spreads: list[float] = []
    for c in chain_list:
        q = c.get("last_quote", {}) or {}
        bid = q.get("bid", 0) or 0
        ask = q.get("ask", 0) or 0
        mid = (bid + ask) / 2
        if mid > 0:
            spreads.append((ask - bid) / mid)
    avg_spread_pct = sum(spreads) / len(spreads) if spreads else 1.0

    oi_score = min(total_oi / 5000, 1.0) * 100
    vol_score = min(total_vol / 1000, 1.0) * 100
    iv_score = iv_completeness * 100
    spread_score = max(0, 100 - avg_spread_pct * 500)

    composite = int(
        oi_score * 0.30
        + vol_score * 0.25
        + iv_score * 0.25
        + spread_score * 0.20
    )
    composite = max(0, min(100, composite))

    return {
        "score": composite,
        "oi": int(oi_score),
        "volume": int(vol_score),
        "iv_completeness": int(iv_score),
        "spread": int(spread_score),
    }


# ─── EARNINGS-DAY EM OVERRIDE ───────────────────────────────────────────
def effective_daily_em(
    price: float,
    iv_pct: float,
    contracts: list[dict] | None = None,
    is_earnings_day: bool = False,
    center: float | None = None,
) -> dict[str, Any]:
    """Compute the daily expected move, automatically switching to a
    straddle-based calculation on earnings days.

    On earnings days, ATM IV understates the actual expected move because the
    realized gap includes \"event premium\" that is priced into options but not
    reflected in the annualized IV surface. The short-dated ATM straddle *does*
    price in that premium, so we use it as the 1σ anchor instead.

    Parameters
    ----------
    price : float
        Current underlying price.
    iv_pct : float
        Annualized ATM IV in percent (e.g. ``22.5``). Always used as the
        fallback, and as the ``iv`` metadata key even on earnings days.
    contracts : list[dict] | None
        Raw Polygon options snapshot contracts list. Only needed on earnings
        days — if ``None`` while ``is_earnings_day=True``, falls back to IV.
    is_earnings_day : bool
        Whether the ticker is inside its earnings suppression window
        (i.e. ``EarningsService.is_in_earnings_window`` returned ``True``).
    center : float | None
        Optional override for the band center.

    Returns
    -------
    dict with keys:
        ``levels`` — the standard sigma1/2/3 + low/high dict
        ``source`` — ``\"iv\"`` | ``\"straddle\"`` | ``\"straddle_fallback_iv\"
        ``iv``     — the annualized IV used (percent)
        ``one_sigma`` — the 1σ dollar move
        ``earnings`` — bool
    """
    result: dict[str, Any] = {
        "iv": iv_pct,
        "earnings": is_earnings_day,
    }

    if is_earnings_day and contracts is not None:
        straddle_data = extract_atm_straddle(contracts, price)
        if straddle_data and straddle_data["straddle"] > 0:
            levels = calc_daily_em_straddle(
                price, straddle_data["straddle"], center=center,
            )
            result["levels"] = levels
            result["one_sigma"] = straddle_data["straddle"]
            result["source"] = "straddle"
            result["straddle_strike"] = straddle_data["strike"]
            result["straddle_premium"] = straddle_data["straddle"]
            return result
        # Straddle extraction failed — fall through to IV with a warning source
        result["source"] = "straddle_fallback_iv"
    else:
        result["source"] = "iv"

    levels = calc_daily_em_iv(price, iv_pct, center=center)
    result["levels"] = levels
    result["one_sigma"] = levels["sigma1"]
    return result


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_WEEKLY_TRADING_DAYS",
    "expected_move_levels",
    "calc_expected_move",
    "calc_daily_em_iv",
    "calc_weekly_em_iv",
    "calc_daily_em_straddle",
    "calc_weekly_em_straddle",
    "calc_historical_vol",
    "calc_iv_roc",
    "extract_atm_iv_average",
    "extract_atm_iv_interpolated",
    "extract_atm_straddle",
    "find_nearest_expiration",
    "compute_confidence_score",
    "effective_daily_em",
]
