"""
Position sizer — decides contract count given live cash, ask price, and
sizing mode. All math is in integer cents.

The contract count we return is what we'll actually place, given:
  - Kalshi's 1-contract minimum
  - Available size at the ask (we never try to sweep more than what's resting)
  - Fee-inclusive budget
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy.fees import fee_cents, total_outlay_cents


@dataclass(frozen=True)
class SizeDecision:
    contracts: int                 # 0 => can't afford / skip this market
    price_cents: int               # ask we'll bid at
    cost_cents: int                # price * contracts
    fee_cents: int                 # estimated fee
    total_outlay_cents: int        # cost + fee  (this many cents leave the account)
    reason: str                    # human-readable explanation for the log


def _fits_budget(
    contracts: int, price_cents: int, working_budget_cents: int
) -> bool:
    return total_outlay_cents(contracts, price_cents) <= working_budget_cents


def size_fixed_dollar(
    *,
    working_budget_cents: int,
    fixed_trade_size_cents: int,
    price_cents: int,
    available_size: int,
) -> SizeDecision:
    """One small slice per market, capped at FIXED_TRADE_SIZE_USD."""
    if price_cents <= 0 or price_cents >= 100:
        return _skip(price_cents, "invalid price")
    if working_budget_cents <= 0:
        return _skip(price_cents, "no working budget left")

    slice_budget = min(fixed_trade_size_cents, working_budget_cents)
    if slice_budget <= 0:
        return _skip(price_cents, "slice budget non-positive")

    # Naive upper bound, then trim down until fees fit.
    n = slice_budget // price_cents
    while n > 0 and not _fits_budget(n, price_cents, slice_budget):
        n -= 1

    if n <= 0:
        return _skip(
            price_cents,
            f"cannot afford 1 contract at {price_cents}¢ within slice {slice_budget}¢",
        )

    if available_size > 0 and n > available_size:
        n = available_size

    if n <= 0:
        return _skip(price_cents, "no resting liquidity at the ask")

    return _accept(n, price_cents, reason=f"fixed slice {slice_budget}¢ -> {n} contracts")


def size_all_in(
    *,
    working_budget_cents: int,
    price_cents: int,
    available_size: int,
) -> SizeDecision:
    """One concentrated bet — buy as many contracts as the full balance can afford."""
    if price_cents <= 0 or price_cents >= 100:
        return _skip(price_cents, "invalid price")
    if working_budget_cents <= 0:
        return _skip(price_cents, "no working budget left")

    n = working_budget_cents // price_cents
    while n > 0 and not _fits_budget(n, price_cents, working_budget_cents):
        n -= 1

    if n <= 0:
        return _skip(
            price_cents,
            f"cannot afford 1 contract at {price_cents}¢ with budget {working_budget_cents}¢",
        )

    if available_size > 0 and n > available_size:
        n = available_size

    if n <= 0:
        return _skip(price_cents, "no resting liquidity at the ask")

    return _accept(n, price_cents, reason=f"all-in budget {working_budget_cents}¢ -> {n} contracts")


def _accept(contracts: int, price_cents: int, *, reason: str) -> SizeDecision:
    return SizeDecision(
        contracts=contracts,
        price_cents=price_cents,
        cost_cents=contracts * price_cents,
        fee_cents=fee_cents(contracts, price_cents),
        total_outlay_cents=total_outlay_cents(contracts, price_cents),
        reason=reason,
    )


def _skip(price_cents: int, reason: str) -> SizeDecision:
    return SizeDecision(
        contracts=0,
        price_cents=price_cents,
        cost_cents=0,
        fee_cents=0,
        total_outlay_cents=0,
        reason=reason,
    )


def can_afford_cheapest(
    *,
    working_budget_cents: int,
    cheapest_price_cents: int,
) -> bool:
    """True iff the working budget can still afford one contract at the
    cheapest qualifying price including fees. Used to terminate the pass."""
    return _fits_budget(1, cheapest_price_cents, working_budget_cents)
