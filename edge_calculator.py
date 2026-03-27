"""
edge_calculator.py — Run all providers against a market and return the best edge.

Edge = my_probability(%) - kalshi_yes_price(%)

Only the first provider that returns a non-None estimate is used; providers
are tried in priority order: weather → sports → news → polling.

If no provider matches, returns None.
"""

import logging
import traceback
from dataclasses import dataclass
from typing import Optional

from providers import ALL_PROVIDERS

logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    ticker: str
    question: str
    provider_name: str
    my_probability: float    # 0–100
    kalshi_price: float      # 0–100 (yes price in cents = probability %)
    edge: float              # my_probability - kalshi_price


def _extract_yes_price(market: dict) -> Optional[float]:
    """
    Extract the current YES ask price as a percentage (0–100).

    Kalshi market dicts include `yes_ask` or `last_price` in cents.
    We use `yes_ask` for limit-order entry pricing.
    """
    # Prefer the best ask (what we'd pay to buy YES)
    yes_ask = market.get("yes_ask")
    if yes_ask is not None:
        return float(yes_ask)

    # Fall back to last traded price
    last_price = market.get("last_price")
    if last_price is not None:
        return float(last_price)

    return None


def calculate_edge(market: dict) -> Optional[EdgeResult]:
    """
    Try every provider in order.  Return an EdgeResult for the first
    provider that produces an estimate, or None if none match.

    Provider failures are caught and logged individually.
    """
    ticker   = market.get("ticker", "")
    question = market.get("title", "") or market.get("question", "")

    kalshi_price = _extract_yes_price(market)
    if kalshi_price is None:
        logger.debug("No YES price available for %s — skipping", ticker)
        return None

    for provider in ALL_PROVIDERS:
        try:
            my_prob = provider.estimate(market)
        except Exception:
            logger.warning(
                "Provider '%s' raised an exception on market '%s':\n%s",
                provider.name, ticker, traceback.format_exc()
            )
            continue

        if my_prob is None:
            continue  # provider doesn't handle this market

        edge = round(my_prob - kalshi_price, 2)
        logger.info(
            "EVAL %s | provider=%s | my_prob=%.1f%% | kalshi=%.1f¢ | edge=%+.1f%%",
            ticker, provider.name, my_prob, kalshi_price, edge
        )
        return EdgeResult(
            ticker=ticker,
            question=question,
            provider_name=provider.name,
            my_probability=my_prob,
            kalshi_price=kalshi_price,
            edge=edge,
        )

    return None
