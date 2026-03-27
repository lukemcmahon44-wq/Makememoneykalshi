"""
providers/sports.py — Probability estimates for sports-outcome Kalshi markets.

Uses The Odds API (https://the-odds-api.com) free tier.
Matches markets that contain team names or sport-outcome language.
Returns implied probability from the best available moneyline.
"""

import logging
import os
import re
import traceback
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_ODDS_KEY = os.getenv("ODDS_API_KEY", "")
_ODDS_BASE = "https://api.the-odds-api.com/v4"

# Supported sport keys on The Odds API
_SPORT_KEYS = [
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_nba",
    "basketball_ncaab",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_usa_mls",
    "soccer_epl",
    "tennis_atp_french_open",
    "tennis_wta_french_open",
]

_SPORT_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "mls", "ncaa",
    "super bowl", "world series", "stanley cup", "nba finals",
    "championship", "playoffs", "win", "beat", "defeat",
    "football", "basketball", "baseball", "hockey", "soccer",
    "match", "game", "series",
]


def _is_sports_market(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _SPORT_KEYWORDS)


def _american_to_prob(american: int) -> float:
    """Convert American moneyline odds to implied probability (0–100)."""
    if american > 0:
        return 100 / (american + 100) * 100
    else:
        return abs(american) / (abs(american) + 100) * 100


def _fetch_odds(sport_key: str) -> Optional[list]:
    if not _ODDS_KEY:
        logger.warning("ODDS_API_KEY not set")
        return None
    try:
        resp = requests.get(
            f"{_ODDS_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": _ODDS_KEY,
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            },
            timeout=10,
        )
        if resp.status_code == 422:
            # Sport not currently active
            return []
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.error("SportsProvider fetch error [%s]:\n%s", sport_key, traceback.format_exc())
        return None


def _best_implied_prob(game: dict, team_name: str) -> Optional[float]:
    """
    Find the best (lowest-vig) implied probability for `team_name` across
    all bookmakers in a game dict.
    """
    team_name_lower = team_name.lower()
    probs = []
    for bookmaker in game.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if team_name_lower in outcome.get("name", "").lower():
                    probs.append(_american_to_prob(outcome["price"]))
    if not probs:
        return None
    # Return average of all books (reduces single-book bias)
    return sum(probs) / len(probs)


def _extract_teams(title: str) -> list:
    """
    Very simple heuristic: split on 'vs', 'vs.', 'v ', '@', ' over ', ' beat '
    and return the two fragments.
    """
    for sep in [" vs. ", " vs ", " v ", " @ ", " over ", " beat ", " defeat "]:
        if sep in title.lower():
            idx = title.lower().index(sep)
            team_a = title[:idx].strip()
            team_b = title[idx + len(sep):].strip()
            # Strip trailing fluff like " to win" or "?"
            team_b = re.sub(r"\s*(to win|win|wins|\?|in game \d).*$", "", team_b, flags=re.IGNORECASE).strip()
            team_a = re.sub(r"^(will\s+|who\s+wins[:\s]+)", "", team_a, flags=re.IGNORECASE).strip()
            return [team_a, team_b]
    return []


class SportsProvider:
    name = "sports"

    def __init__(self):
        self._games_cache: dict = {}  # sport_key -> list of games

    def _load_all_games(self) -> None:
        """Populate cache with live odds for all supported sports."""
        self._games_cache = {}
        for sport_key in _SPORT_KEYS:
            games = _fetch_odds(sport_key)
            if games:
                self._games_cache[sport_key] = games

    def estimate(self, market: dict) -> Optional[float]:
        title = market.get("title", "") or market.get("question", "")
        if not _is_sports_market(title):
            return None

        teams = _extract_teams(title)
        if not teams:
            logger.debug("SportsProvider: could not extract teams from: %s", title)
            return None

        if not self._games_cache:
            self._load_all_games()

        team_a = teams[0]

        for sport_key, games in self._games_cache.items():
            for game in games:
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                if (team_a.lower() in home.lower() or team_a.lower() in away.lower()):
                    prob = _best_implied_prob(game, team_a)
                    if prob is not None:
                        logger.debug("SportsProvider: %s → %.1f%% (sport=%s)",
                                     title, prob, sport_key)
                        return round(prob, 1)

        logger.debug("SportsProvider: no matching game found for: %s", title)
        return None

    def refresh(self) -> None:
        """Force a cache refresh (call once per scan cycle)."""
        self._load_all_games()
