"""
Market hours and session management. Handles US equity hours and 24/7 crypto.
"""

from __future__ import annotations

import datetime as dt
from typing import Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # python < 3.9
    from backports.zoneinfo import ZoneInfo  # type: ignore


NY_TZ = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# Static list of full US market holidays (extend yearly)
US_MARKET_HOLIDAYS_2024_2026 = {
    dt.date(2024, 1, 1),  dt.date(2024, 1, 15), dt.date(2024, 2, 19),
    dt.date(2024, 3, 29), dt.date(2024, 5, 27), dt.date(2024, 6, 19),
    dt.date(2024, 7, 4),  dt.date(2024, 9, 2),  dt.date(2024, 11, 28),
    dt.date(2024, 12, 25),
    dt.date(2025, 1, 1),  dt.date(2025, 1, 20), dt.date(2025, 2, 17),
    dt.date(2025, 4, 18), dt.date(2025, 5, 26), dt.date(2025, 6, 19),
    dt.date(2025, 7, 4),  dt.date(2025, 9, 1),  dt.date(2025, 11, 27),
    dt.date(2025, 12, 25),
    dt.date(2026, 1, 1),  dt.date(2026, 1, 19), dt.date(2026, 2, 16),
    dt.date(2026, 4, 3),  dt.date(2026, 5, 25), dt.date(2026, 6, 19),
    dt.date(2026, 7, 3),  dt.date(2026, 9, 7),  dt.date(2026, 11, 26),
    dt.date(2026, 12, 25),
}


class MarketClock:
    """Trading session helpers. Use UTC internally."""

    def __init__(self, config=None):
        self.config = config

    @staticmethod
    def now_utc() -> dt.datetime:
        return dt.datetime.now(UTC)

    @staticmethod
    def now_ny() -> dt.datetime:
        return dt.datetime.now(NY_TZ)

    @staticmethod
    def is_us_equity_open(now_utc: dt.datetime | None = None) -> bool:
        now = (now_utc or MarketClock.now_utc()).astimezone(NY_TZ)
        if now.weekday() >= 5:
            return False
        if now.date() in US_MARKET_HOLIDAYS_2024_2026:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now < market_close

    @staticmethod
    def minutes_to_us_close(now_utc: dt.datetime | None = None) -> int:
        now = (now_utc or MarketClock.now_utc()).astimezone(NY_TZ)
        close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = (close - now).total_seconds() / 60
        return max(0, int(delta))

    @staticmethod
    def session_start(asset_type: str, now_utc: dt.datetime | None = None) -> dt.datetime:
        """Return the most recent session start in UTC for VWAP / session windows."""
        now = now_utc or MarketClock.now_utc()
        if asset_type == "crypto":
            # Crypto resets at UTC midnight
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        ny = now.astimezone(NY_TZ)
        market_open_ny = ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if ny < market_open_ny:
            market_open_ny -= dt.timedelta(days=1)
        return market_open_ny.astimezone(UTC)

    @staticmethod
    def is_open(asset: str, asset_type: str) -> bool:
        if asset_type == "crypto":
            return True
        return MarketClock.is_us_equity_open()

    @staticmethod
    def time_to_session_open(asset_type: str = "equity") -> int:
        """Seconds until next open. Crypto is always open."""
        if asset_type == "crypto":
            return 0
        if MarketClock.is_us_equity_open():
            return 0
        now = MarketClock.now_ny()
        target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now >= target:
            target += dt.timedelta(days=1)
        # Skip weekends/holidays
        while target.weekday() >= 5 or target.date() in US_MARKET_HOLIDAYS_2024_2026:
            target += dt.timedelta(days=1)
        return int((target - now).total_seconds())

    @staticmethod
    def session_bounds_for_date(date: dt.date) -> Tuple[dt.datetime, dt.datetime]:
        """Return (open, close) UTC for US equity session on given date."""
        open_ny = dt.datetime.combine(date, dt.time(9, 30, tzinfo=NY_TZ))
        close_ny = dt.datetime.combine(date, dt.time(16, 0, tzinfo=NY_TZ))
        return open_ny.astimezone(UTC), close_ny.astimezone(UTC)
