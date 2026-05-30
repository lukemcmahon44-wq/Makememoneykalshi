"""
Ranking — order surviving candidates so we buy the cheapest first, with
deterministic tie-breaks.

Sort key (ascending):
  1. Yes ask price in cents (95 before 96 before … before 99)
  2. Negative size at the best ask (more liquidity wins)
  3. Soonest sensible close_time (earlier close_time wins; missing -> last)
  4. Ticker, alphabetical (deterministic last resort)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from strategy.filter import Candidate


_FAR_FUTURE = datetime(9999, 12, 31, tzinfo=timezone.utc)


def _close_time(raw: dict) -> datetime:
    val = raw.get("close_time") or raw.get("expiration_time")
    if not val:
        return _FAR_FUTURE
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except ValueError:
        return _FAR_FUTURE


def rank(candidates: Iterable[Candidate]) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            c.yes_ask_cents,
            -c.yes_ask_size,
            _close_time(c.raw),
            c.ticker,
        ),
    )
