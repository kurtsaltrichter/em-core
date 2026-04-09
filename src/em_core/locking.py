"""
Level locking for em-core.

Single source of truth for the daily / weekly / monthly "lock" boundaries used
by both the EM Dashboard and the TripWire bot to decide when a cached expected
move level is still valid vs needs to be recalculated.

Lock boundaries (locked by user decision, 2026-04-04)
-----------------------------------------------------
- **Daily**:  midnight (00:00:00) America/New_York. A daily level computed at
  any time during trading day D is valid until the next midnight ET.
- **Weekly**: 4:30 PM America/New_York on the last trading day of the week
  (usually Friday, Thursday in a Good Friday week, etc.).
- **Monthly**: 4:30 PM America/New_York on the last trading day of the month.

Holidays
--------
The NYSE holiday calendar is hard-coded for 2025-2027. Update
:data:`US_MARKET_HOLIDAYS` at the start of each new year, or replace with a
dynamic source (pandas_market_calendars) when it becomes worth the extra
dependency.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional

import pytz

ET = pytz.timezone("US/Eastern")

# NYSE/NASDAQ closed dates. Keep this sorted by year.
US_MARKET_HOLIDAYS: set[str] = {
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed, Jul 4 = Sat)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",  # New Year's Day
    "2027-01-18",  # MLK Day
    "2027-02-15",  # Presidents' Day
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial Day
    "2027-06-18",  # Juneteenth (observed, Jun 19 = Sat)
    "2027-07-05",  # Independence Day (observed, Jul 4 = Sun)
    "2027-09-06",  # Labor Day
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas (observed, Dec 25 = Sat)
}

MARKET_CLOSE_HOUR = 16
WEEKLY_LOCK_MINUTE = 30  # 4:30 PM ET grace window after 4:00 close


# ─── TRADING DAY HELPERS ─────────────────────────────────────────────────
def _as_date(d: date | datetime | str) -> date:
    if isinstance(d, str):
        return date.fromisoformat(d)
    if isinstance(d, datetime):
        return d.date()
    return d


def is_trading_day(d: date | datetime) -> bool:
    """Return True if ``d`` is a weekday and not a US market holiday."""
    the_date = _as_date(d)
    if the_date.weekday() >= 5:  # Sat/Sun
        return False
    return the_date.strftime("%Y-%m-%d") not in US_MARKET_HOLIDAYS


def previous_trading_day(d: date | datetime) -> date:
    """Return the nearest trading day strictly before ``d``."""
    current = _as_date(d) - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def next_trading_day(d: date | datetime) -> date:
    """Return the nearest trading day strictly after ``d``."""
    current = _as_date(d) + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)
    return current


# ─── WEEK / MONTH BOUNDARIES ─────────────────────────────────────────────
def get_week_end_trading_day(now_et: Optional[datetime] = None) -> tuple[str, int]:
    """Return ``(iso_date, trading_days_this_week)`` for the last trading day
    of the current calendar week (Mon-Fri) in ET.

    If ``now_et`` falls on a weekend, the \"current week\" rolls to the
    following Mon-Fri block.
    """
    if now_et is None:
        now_et = datetime.now(ET)

    if now_et.weekday() >= 5:
        # Sat/Sun → jump to next Monday
        days_to_monday = 7 - now_et.weekday()
        monday = (now_et + timedelta(days=days_to_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        monday = (now_et - timedelta(days=now_et.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    trading_days: list[datetime] = []
    for i in range(5):
        day = monday + timedelta(days=i)
        if is_trading_day(day):
            trading_days.append(day)

    if not trading_days:
        # Degenerate case (shouldn't happen in practice): fall back to Friday
        days_ahead = (4 - now_et.weekday()) % 7
        friday = now_et + timedelta(days=days_ahead)
        return friday.strftime("%Y-%m-%d"), 5

    last_day = trading_days[-1]
    return last_day.strftime("%Y-%m-%d"), len(trading_days)


def get_month_end_trading_day(now_et: Optional[datetime] = None) -> str:
    """Return the ISO date of the last trading day of the current month in ET."""
    if now_et is None:
        now_et = datetime.now(ET)

    # Jump to the first of next month, then walk back to the last trading day
    year = now_et.year
    month = now_et.month
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    candidate = first_next - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate.strftime("%Y-%m-%d")


# ─── LOCK BOUNDARIES ───────────────────────────────────────────────────
def daily_lock_boundary(now_et: Optional[datetime] = None) -> datetime:
    """Return the most recent midnight ET (inclusive lower bound) for the
    daily level lock.

    A cached daily level whose ``fixed_at`` is >= this boundary is still
    current; anything older needs to be recalculated.
    """
    if now_et is None:
        now_et = datetime.now(ET)
    elif now_et.tzinfo is None:
        now_et = ET.localize(now_et)
    else:
        now_et = now_et.astimezone(ET)

    return now_et.replace(hour=0, minute=0, second=0, microsecond=0)


def weekly_lock_expiry(now_et: Optional[datetime] = None) -> datetime:
    """Return the expiry timestamp for the current weekly level lock:
    4:30 PM ET on the last trading day of the current week."""
    if now_et is None:
        now_et = datetime.now(ET)

    week_end_str, _ = get_week_end_trading_day(now_et)
    week_end_date = datetime.fromisoformat(week_end_str)
    return ET.localize(
        week_end_date.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=WEEKLY_LOCK_MINUTE,
            second=0,
            microsecond=0,
        )
    )


def monthly_lock_expiry(now_et: Optional[datetime] = None) -> datetime:
    """Return the expiry timestamp for the current monthly level lock:
    4:30 PM ET on the last trading day of the current month."""
    if now_et is None:
        now_et = datetime.now(ET)

    month_end_str = get_month_end_trading_day(now_et)
    month_end_date = datetime.fromisoformat(month_end_str)
    return ET.localize(
        month_end_date.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=WEEKLY_LOCK_MINUTE,
            second=0,
            microsecond=0,
        )
    )


# ─── LOCK VALIDITY CHECKS ────────────────────────────────────────────────
def _normalize_fixed_at(fixed_at: str | datetime) -> datetime:
    """Parse a stored ``fixed_at`` / ``expires_at`` string or datetime into an
    ET-aware datetime."""
    if isinstance(fixed_at, str):
        dt = datetime.fromisoformat(fixed_at)
    else:
        dt = fixed_at
    if dt.tzinfo is None:
        dt = ET.localize(dt)
    return dt.astimezone(ET)


def is_daily_lock_current(
    fixed_at: str | datetime,
    now_et: Optional[datetime] = None,
) -> bool:
    """True iff the daily-level timestamp is still inside the current lock
    window (i.e. on or after the most recent midnight ET)."""
    try:
        fixed = _normalize_fixed_at(fixed_at)
    except (ValueError, TypeError):
        return False
    return fixed >= daily_lock_boundary(now_et)


def is_weekly_lock_current(
    expires_at: str | datetime,
    now_et: Optional[datetime] = None,
) -> bool:
    """True iff the stored weekly expiry is still in the future."""
    try:
        expires = _normalize_fixed_at(expires_at)
    except (ValueError, TypeError):
        return False
    if now_et is None:
        now_et = datetime.now(ET)
    elif now_et.tzinfo is None:
        now_et = ET.localize(now_et)
    else:
        now_et = now_et.astimezone(ET)
    return now_et < expires


def is_monthly_lock_current(
    expires_at: str | datetime,
    now_et: Optional[datetime] = None,
) -> bool:
    """True iff the stored monthly expiry is still in the future."""
    return is_weekly_lock_current(expires_at, now_et)  # same logic, different expiry source


__all__ = [
    "ET",
    "US_MARKET_HOLIDAYS",
    "is_trading_day",
    "previous_trading_day",
    "next_trading_day",
    "get_week_end_trading_day",
    "get_month_end_trading_day",
    "daily_lock_boundary",
    "weekly_lock_expiry",
    "monthly_lock_expiry",
    "is_daily_lock_current",
    "is_weekly_lock_current",
    "is_monthly_lock_current",
]
