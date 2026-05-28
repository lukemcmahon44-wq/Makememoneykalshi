"""
strategy.py — PURE candidate-selection and position-sizing logic.

This module performs NO network I/O and has NO side effects. It takes plain
data (market dicts, balance, the set of tickers we already touch) and returns a
list of `IntendedOrder`s. That purity is what makes the trading logic unit
testable and trustworthy — see test_strategy.py.

The global deployed-capital cap is intentionally NOT enforced here. It depends
on order-by-order accumulation and is the caller's responsibility (trader.py),
which is why candidates are returned sorted by ask price descending: if the cap
cuts the caller off, the safest (highest-probability) orders are placed first.

Every number is a `decimal.Decimal`. There is no `float` in this module's
money path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any, Iterable, Optional

import config


# ──────────────────────────────────────────────────────────────────────────────
# Output type
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class IntendedOrder:
    """A fully-sized, ready-to-place YES buy. Decimals are the source of truth;
    `count_fp` and `yes_price_dollars` are the exact strings the API wants."""

    ticker: str
    count: Decimal          # whole contracts unless the market allows fractions
    price: Decimal          # YES ask, aligned to the market tick (dollars)
    dollar_amount: Decimal  # count * price
    side: str = "yes"

    @property
    def count_fp(self) -> str:
        """Fixed-point contract count, e.g. '1.00'."""
        return f"{self.count:.2f}"

    @property
    def yes_price_dollars(self) -> str:
        """4-decimal dollar price string, e.g. '0.9700'."""
        return f"{self.price:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Small Decimal helpers (local copy keeps this module free of network imports)
# ──────────────────────────────────────────────────────────────────────────────
def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def tick_size(price_level_structure: Any) -> Decimal:
    """Return the price increment for a market's tick structure.

    `linear_cent` -> $0.01, `deci_cent`/`tapered_deci_cent` -> $0.001. In the
    95–99¢ band a tapered structure uses deci-cent ticks, so treating any
    deci-cent variant as $0.001 is correct there. Unknown/None -> penny.
    """
    name = ""
    if isinstance(price_level_structure, str):
        name = price_level_structure.lower()
    elif isinstance(price_level_structure, dict):
        name = str(price_level_structure.get("name") or price_level_structure.get("type") or "").lower()
    if "deci_cent" in name:
        return Decimal("0.001")
    return Decimal("0.01")


def _round_down_to_increment(value: Decimal, increment: Decimal) -> Decimal:
    """Floor `value` to a multiple of `increment` (e.g. whole contracts)."""
    steps = (value / increment).to_integral_value(rounding=ROUND_DOWN)
    result = steps * increment
    return result.quantize(increment)


def parse_yes_ask(market: dict) -> Optional[Decimal]:
    """The YES ask in dollars. Prefers `yes_ask_dollars`; falls back to a legacy
    integer-cent `yes_ask` if that is all the market carries."""
    ask = _to_decimal(market.get("yes_ask_dollars"))
    if ask is not None:
        return ask
    cents = _to_decimal(market.get("yes_ask"))
    if cents is None:
        return None
    return cents / Decimal(100) if cents > 1 else cents


def parse_ask_liquidity(market: dict) -> Optional[Decimal]:
    """Contracts resting at the YES ask. Prefers `yes_ask_size_fp`."""
    size = _to_decimal(market.get("yes_ask_size_fp"))
    if size is not None:
        return size
    return _to_decimal(market.get("yes_ask_size"))


def minutes_to_close(market: dict, now: datetime) -> Optional[Decimal]:
    """Minutes until the market closes, or None if no timestamp is available.
    May be negative if the market has already closed."""
    raw = market.get("close_time") or market.get("expiration_time")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return Decimal(str((dt - now).total_seconds())) / Decimal(60)


# ──────────────────────────────────────────────────────────────────────────────
# The strategy
# ──────────────────────────────────────────────────────────────────────────────
def select_and_size(
    markets: Iterable[dict],
    balance: Decimal,
    held_tickers: Optional[set[str]] = None,
    *,
    now: Optional[datetime] = None,
) -> list[IntendedOrder]:
    """Select high-probability YES markets and size each position.

    Args:
        markets: market dicts (as returned by GET /markets).
        balance: total account balance in dollars (Decimal).
        held_tickers: tickers we already hold or have a resting order on; these
            are skipped so we never double up within a market.
        now: injectable UTC clock for testing; defaults to datetime.now(utc).

    Returns:
        IntendedOrders sorted by ask price DESCENDING (safest first). The caller
        enforces the global deployed-capital cap as it places them.
    """
    now = now or datetime.now(timezone.utc)
    held = held_tickers or set()
    balance = balance if isinstance(balance, Decimal) else Decimal(str(balance))
    candidates: list[IntendedOrder] = []

    for market in markets:
        ticker = market.get("ticker")
        if not ticker or ticker in held:
            continue
        if (market.get("status") or "").lower() != "open":
            continue

        ask = parse_yes_ask(market)
        if ask is None:
            continue
        # The 95–99¢ probability filter. 1.00 is explicitly excluded: no edge.
        if ask < config.MIN_PROBABILITY_PRICE or ask > config.MAX_PROBABILITY_PRICE:
            continue

        mins = minutes_to_close(market, now)
        if mins is not None and mins < config.MIN_MINUTES_TO_CLOSE:
            continue

        # Align the limit price to the market tick, rounding DOWN so we never bid
        # above the displayed ask (a missed fill is fine; an overpay is not).
        tick = tick_size(market.get("price_level_structure"))
        price = ask.quantize(tick, rounding=ROUND_DOWN)
        if price <= 0:
            continue

        # Size as a fixed fraction of balance, capped per market.
        dollar_target = min(balance * config.POSITION_PCT, config.MAX_POSITION_PER_MARKET)
        increment = Decimal("0.01") if market.get("fractional_trading_enabled") else Decimal("1")
        contracts = _round_down_to_increment(dollar_target / price, increment)
        if contracts <= 0:
            continue

        # Require enough resting liquidity at the ask to fill us.
        liquidity = parse_ask_liquidity(market)
        if liquidity is None or liquidity < config.MIN_LIQUIDITY_CONTRACTS or liquidity < contracts:
            continue

        dollar_amount = contracts * price
        if dollar_amount < config.MIN_ORDER_DOLLARS:
            continue

        candidates.append(
            IntendedOrder(
                ticker=ticker,
                count=contracts,
                price=price,
                dollar_amount=dollar_amount,
            )
        )

    # Safest (highest implied-probability) first, so the deployed cap clips the
    # marginal candidates rather than the best ones.
    candidates.sort(key=lambda o: o.price, reverse=True)
    return candidates
