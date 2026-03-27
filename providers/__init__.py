"""
providers/ — Modular edge-data providers.

Each provider module exposes a single function:

    estimate(market: dict) -> Optional[float]

Where `market` is the raw Kalshi market dict and the return value is a
probability in the range 0–100, or None if the provider cannot handle
this market.
"""

from providers.weather import WeatherProvider
from providers.sports import SportsProvider
from providers.news import NewsProvider
from providers.polling import PollingProvider

ALL_PROVIDERS = [
    WeatherProvider(),
    SportsProvider(),
    NewsProvider(),
    PollingProvider(),
]
