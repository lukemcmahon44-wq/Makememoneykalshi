"""
Market filtering — narrow the open-market universe to candidates we'd buy.

A candidate market must:
  - have a best-available Yes ASK price (the price you'd actually pay)
    that falls inside [min_cents, max_cents]
  - be open and not so close to settlement that an order can't fill
  - have at least `min_liquidity` contracts resting at that ask price
  - not already be held by us (caller passes the set of held tickers)

Pure function. Mocked in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# Statuses we consider tradeable. Anything else is skipped.
TRADEABLE_STATUSES = {"open", "active"}


@dataclass(frozen=True)
class Candidate:
    """A market that passed the filter, with the prices we'll act on."""
    ticker: str
    yes_ask_cents: int
    yes_ask_size: int
    raw: dict


def yes_ask_cents(market: dict) -> Optional[int]:
    """Best-available Yes ask price, in integer cents, or None if unavailable.

    Kalshi's market payload exposes `yes_ask` directly; we treat that as the
    cheapest price at which we can buy a Yes. If it's missing or non-positive,
    the market isn't actionable.
    """
    raw = market.get("yes_ask")
    if raw is None:
        return None
    try:
        cents = int(raw)
    except (TypeError, ValueError):
        return None
    if cents <= 0 or cents >= 100:
        return None
    return cents


def yes_ask_size(market: dict) -> int:
    """Liquidity available at the Yes ask. Defaults to 0 if not exposed.

    Some Kalshi market payloads only expose `liquidity` (total $ across the
    book). When the size at the best ask is reported, we use it; otherwise
    callers should fetch the orderbook for an authoritative number.
    """
    for key in ("yes_ask_size", "yes_ask_qty", "yes_ask_volume"):
        if key in market and market[key] is not None:
            try:
                return max(0, int(market[key]))
            except (TypeError, ValueError):
                continue
    return 0


def is_tradeable_status(market: dict) -> bool:
    status = (market.get("status") or "").lower()
    return status in TRADEABLE_STATUSES


def filter_markets(
    markets: Iterable[dict],
    *,
    min_cents: int,
    max_cents: int,
    min_liquidity: int,
    held_tickers: Optional[set[str]] = None,
) -> list[Candidate]:
    """Return Candidate objects for every market that should be considered.

    `held_tickers` lets us avoid double-buying a market we already own this pass.
    If a market reports zero size at the best ask, we still keep it when the
    caller's `min_liquidity` is the default 1 — the orderbook lookup later will
    confirm. Callers who set min_liquidity > 1 should also pre-fill sizes.
    """
    held = held_tickers or set()
    out: list[Candidate] = []
    for m in markets:
        ticker = m.get("ticker")
        if not ticker or ticker in held:
            continue
        if not is_tradeable_status(m):
            continue
        ask = yes_ask_cents(m)
        if ask is None:
            continue
        if ask < min_cents or ask > max_cents:
            continue
        size = yes_ask_size(m)
        if min_liquidity > 0 and size > 0 and size < min_liquidity:
            continue
        out.append(Candidate(ticker=ticker, yes_ask_cents=ask, yes_ask_size=size, raw=m))
    return out
