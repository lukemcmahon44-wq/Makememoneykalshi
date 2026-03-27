"""
providers/weather.py — Probability estimates for weather-related Kalshi markets.

Uses OpenWeatherMap 5-day / 3-hour forecast API (free tier).
Matches markets containing temperature, precipitation, or weather-event keywords
plus a recognisable city name.

Returned probability is 0–100 (percentage).
"""

import logging
import os
import re
import traceback
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "")
_OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
_OWM_CURRENT_URL  = "https://api.openweathermap.org/data/2.5/weather"

# Keywords that indicate this is a weather market
_WEATHER_KEYWORDS = [
    "temperature", "degrees", "fahrenheit", "celsius",
    "rain", "rainfall", "precipitation", "snow", "snowfall",
    "hurricane", "tornado", "storm", "flood", "drought",
    "high temp", "low temp", "heat", "freeze", "frost",
    "sunny", "cloudy", "overcast",
]

# Common US/world cities Kalshi tends to list
_CITY_MAP = {
    "new york":     "New York,US",
    "nyc":          "New York,US",
    "los angeles":  "Los Angeles,US",
    "la":           "Los Angeles,US",
    "chicago":      "Chicago,US",
    "houston":      "Houston,US",
    "miami":        "Miami,US",
    "dallas":       "Dallas,US",
    "phoenix":      "Phoenix,US",
    "seattle":      "Seattle,US",
    "san francisco":"San Francisco,US",
    "denver":       "Denver,US",
    "boston":       "Boston,US",
    "washington":   "Washington,US",
    "dc":           "Washington,US",
    "atlanta":      "Atlanta,US",
    "london":       "London,GB",
    "paris":        "Paris,FR",
    "tokyo":        "Tokyo,JP",
}


def _is_weather_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _WEATHER_KEYWORDS)


def _extract_city(title: str) -> Optional[str]:
    t = title.lower()
    for key, city_q in _CITY_MAP.items():
        if key in t:
            return city_q
    return None


def _extract_threshold(title: str) -> Optional[float]:
    """Pull a numeric temperature / inch value from the title."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?|inches?|in\.?)", title, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _fetch_forecast(city_query: str) -> Optional[list]:
    if not _OWM_KEY:
        logger.warning("OPENWEATHER_API_KEY not set")
        return None
    try:
        resp = requests.get(
            _OWM_FORECAST_URL,
            params={"q": city_query, "appid": _OWM_KEY, "units": "imperial", "cnt": 40},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("list", [])
    except Exception:
        logger.error("WeatherProvider fetch error:\n%s", traceback.format_exc())
        return None


class WeatherProvider:
    name = "weather"

    def estimate(self, market: dict) -> Optional[float]:
        title = market.get("title", "") or market.get("question", "")
        if not _is_weather_market(title):
            return None

        city = _extract_city(title)
        if not city:
            return None

        forecasts = _fetch_forecast(city)
        if not forecasts:
            return None

        title_lower = title.lower()

        # ── Temperature above/below threshold ────────────────────────────────
        threshold = _extract_threshold(title)
        if threshold and ("above" in title_lower or "exceed" in title_lower or
                          "over" in title_lower or "high" in title_lower):
            highs = [f["main"]["temp_max"] for f in forecasts[:8]]  # next 24 h
            hits = sum(1 for h in highs if h >= threshold)
            prob = (hits / len(highs)) * 100
            logger.debug("WeatherProvider: %s | threshold=%.1f | prob=%.1f", title, threshold, prob)
            return round(prob, 1)

        if threshold and ("below" in title_lower or "under" in title_lower or
                          "low" in title_lower or "drop" in title_lower):
            lows = [f["main"]["temp_min"] for f in forecasts[:8]]
            hits = sum(1 for l in lows if l <= threshold)
            prob = (hits / len(lows)) * 100
            logger.debug("WeatherProvider: %s | threshold=%.1f | prob=%.1f", title, threshold, prob)
            return round(prob, 1)

        # ── Precipitation ────────────────────────────────────────────────────
        if any(kw in title_lower for kw in ["rain", "precipitation", "snow", "snowfall"]):
            precip_slots = [
                f for f in forecasts[:8]
                if f.get("rain", {}).get("3h", 0) > 0
                or f.get("snow", {}).get("3h", 0) > 0
                or any(w["main"].lower() in ("rain", "snow", "drizzle", "thunderstorm")
                       for w in f.get("weather", []))
            ]
            prob = (len(precip_slots) / min(8, len(forecasts))) * 100
            logger.debug("WeatherProvider precip: %s | prob=%.1f", title, prob)
            return round(prob, 1)

        logger.debug("WeatherProvider: matched weather market but no specific rule: %s", title)
        return None
