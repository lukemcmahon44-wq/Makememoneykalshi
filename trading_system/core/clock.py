"""
Market clock utilities.

Provides session detection, time-to-open/close calculations, and a curated
US equity holiday list covering 2025-2027.  All time handling is done in the
America/New_York timezone via pytz.

Usage
-----
    from trading_system.core.clock import MarketClock

    clock = MarketClock()
    if clock.is_market_open():
        print("Regular session active, closes in", clock.seconds_to_close(), "s")
"""

from __future__ import annotations

import datetime
from typing import Set

import pytz

from trading_system.core.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ET = pytz.timezone("America/New_York")

# Regular session window (Eastern time)
_OPEN_TIME = datetime.time(9, 30, 0)
_CLOSE_TIME = datetime.time(16, 0, 0)

# Pre-market / after-hours boundaries (informational; used for session_name)
_PRE_MARKET_START = datetime.time(4, 0, 0)
_AFTER_HOURS_END = datetime.time(20, 0, 0)

# ---------------------------------------------------------------------------
# Holiday calendar — NYSE observed holidays 2025-2027
# ---------------------------------------------------------------------------

_NYSE_HOLIDAYS: Set[datetime.date] = {
    # 2025
    datetime.date(2025, 1, 1),    # New Year's Day
    datetime.date(2025, 1, 20),   # Martin Luther King Jr. Day
    datetime.date(2025, 2, 17),   # Presidents' Day
    datetime.date(2025, 4, 18),   # Good Friday
    datetime.date(2025, 5, 26),   # Memorial Day
    datetime.date(2025, 6, 19),   # Juneteenth
    datetime.date(2025, 7, 4),    # Independence Day
    datetime.date(2025, 9, 1),    # Labor Day
    datetime.date(2025, 11, 27),  # Thanksgiving Day
    datetime.date(2025, 12, 25),  # Christmas Day
    # 2026
    datetime.date(2026, 1, 1),    # New Year's Day
    datetime.date(2026, 1, 19),   # Martin Luther King Jr. Day
    datetime.date(2026, 2, 16),   # Presidents' Day
    datetime.date(2026, 4, 3),    # Good Friday
    datetime.date(2026, 5, 25),   # Memorial Day
    datetime.date(2026, 6, 19),   # Juneteenth
    datetime.date(2026, 7, 3),    # Independence Day (observed, Fri)
    datetime.date(2026, 9, 7),    # Labor Day
    datetime.date(2026, 11, 26),  # Thanksgiving Day
    datetime.date(2026, 12, 25),  # Christmas Day
    # 2027
    datetime.date(2027, 1, 1),    # New Year's Day
    datetime.date(2027, 1, 18),   # Martin Luther King Jr. Day
    datetime.date(2027, 2, 15),   # Presidents' Day
    datetime.date(2027, 3, 26),   # Good Friday
    datetime.date(2027, 5, 31),   # Memorial Day
    datetime.date(2027, 6, 18),   # Juneteenth (observed, Fri)
    datetime.date(2027, 7, 5),    # Independence Day (observed, Mon)
    datetime.date(2027, 9, 6),    # Labor Day
    datetime.date(2027, 11, 25),  # Thanksgiving Day
    datetime.date(2027, 12, 24),  # Christmas Day (observed, Fri)
}


# ---------------------------------------------------------------------------
# MarketClock
# ---------------------------------------------------------------------------

class MarketClock:
    """
    Stateless market clock.  All methods derive the current time from
    ``datetime.datetime.now`` in the Eastern timezone so no external state
    is required.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def now_et(self) -> datetime.datetime:
        """Return the current datetime in Eastern time."""
        return datetime.datetime.now(_ET)

    def is_market_open(self) -> bool:
        """
        Return True if US equities are currently in their regular trading
        session (09:30–16:00 ET, Mon–Fri, excluding NYSE holidays).
        """
        now = self.now_et()
        if now.weekday() >= 5:          # Saturday=5, Sunday=6
            return False
        if now.date() in _NYSE_HOLIDAYS:
            return False
        current_time = now.time().replace(tzinfo=None)
        return _OPEN_TIME <= current_time < _CLOSE_TIME

    def is_crypto_trading(self) -> bool:
        """
        Crypto markets trade 24/7.  Always returns True.
        """
        return True

    def seconds_to_open(self) -> float:
        """
        Return the number of seconds until the next regular session open.

        Returns 0.0 if the market is currently open.
        """
        if self.is_market_open():
            return 0.0

        now = self.now_et()

        # Walk forward day-by-day until we find the next trading day
        candidate = now.date()
        for _ in range(14):  # look at most 2 weeks ahead
            candidate_open = _ET.localize(
                datetime.datetime.combine(candidate, _OPEN_TIME)
            )
            # Is this day a valid trading day and in the future?
            if (
                candidate.weekday() < 5
                and candidate not in _NYSE_HOLIDAYS
                and candidate_open > now
            ):
                delta = (candidate_open - now).total_seconds()
                return max(delta, 0.0)
            candidate += datetime.timedelta(days=1)

        log.warning("Could not find a trading day within 14 days")
        return float("inf")

    def seconds_to_close(self) -> float:
        """
        Return the number of seconds until the current session closes.

        Returns 0.0 if the market is not open (already closed).
        """
        if not self.is_market_open():
            return 0.0

        now = self.now_et()
        close_today = _ET.localize(
            datetime.datetime.combine(now.date(), _CLOSE_TIME)
        )
        return max((close_today - now).total_seconds(), 0.0)

    def session_name(self) -> str:
        """
        Return a string describing the current market session:

        * ``'regular'``      — 09:30–16:00 ET on a trading day
        * ``'pre_market'``   — 04:00–09:30 ET on a trading day
        * ``'after_hours'``  — 16:00–20:00 ET on a trading day
        * ``'closed'``       — all other times (weekend, holiday, overnight)
        """
        now = self.now_et()
        t = now.time().replace(tzinfo=None)

        is_weekday = now.weekday() < 5
        is_holiday = now.date() in _NYSE_HOLIDAYS

        if not is_weekday or is_holiday:
            return "closed"

        if _OPEN_TIME <= t < _CLOSE_TIME:
            return "regular"
        if _PRE_MARKET_START <= t < _OPEN_TIME:
            return "pre_market"
        if _CLOSE_TIME <= t < _AFTER_HOURS_END:
            return "after_hours"
        return "closed"

    def time_since_open(self) -> float:
        """
        Return the number of seconds elapsed since the regular session opened
        today.

        Returns 0.0 if the market has not yet opened today or is not in a
        regular session.
        """
        if not self.is_market_open():
            return 0.0

        now = self.now_et()
        open_today = _ET.localize(
            datetime.datetime.combine(now.date(), _OPEN_TIME)
        )
        return max((now - open_today).total_seconds(), 0.0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_holiday(self, date: datetime.date | None = None) -> bool:
        """Return True if *date* (default: today ET) is a NYSE holiday."""
        if date is None:
            date = self.now_et().date()
        return date in _NYSE_HOLIDAYS

    def next_open(self) -> datetime.datetime:
        """Return the datetime of the next regular session open (Eastern time)."""
        now = self.now_et()
        candidate = now.date()
        for _ in range(14):
            candidate_open = _ET.localize(
                datetime.datetime.combine(candidate, _OPEN_TIME)
            )
            if (
                candidate.weekday() < 5
                and candidate not in _NYSE_HOLIDAYS
                and candidate_open > now
            ):
                return candidate_open
            candidate += datetime.timedelta(days=1)
        raise RuntimeError("No trading day found within 14 days")

    def __repr__(self) -> str:  # pragma: no cover
        now = self.now_et()
        return (
            f"MarketClock(session={self.session_name()!r}, "
            f"now_et={now.strftime('%Y-%m-%d %H:%M:%S %Z')})"
        )
