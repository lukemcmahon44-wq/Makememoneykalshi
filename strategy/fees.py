"""
Kalshi trading fees — integer-cent math.

Per the published formula (verified May 2026 fee schedule):

    fee_dollars_per_contract = 0.07 * price * (1 - price)

The total fee for a fill is ceil-rounded to the nearest cent.

  fee_cents = ceil(0.07 * contracts * price_dollars * (1 - price_dollars) * 100)

We compute it in integer cents to avoid floating-point drift:

  numerator   = 7 * contracts * price_cents * (100 - price_cents)
  denominator = 10_000
  fee_cents   = ceil(numerator / denominator)

All inputs and outputs are integers.
"""

from __future__ import annotations


def fee_cents(contracts: int, price_cents: int) -> int:
    """Estimated taker fee in integer cents.

    The fee formula peaks near 50¢ and is small at 95–99¢:
    at 95¢ for 1 contract -> ceil(7*1*95*5 / 10000) = ceil(3325/10000) = 1
    at 99¢ for 1 contract -> ceil(7*1*99*1 / 10000) = ceil(693 /10000) = 1
    """
    if contracts <= 0 or price_cents <= 0 or price_cents >= 100:
        return 0
    numer = 7 * contracts * price_cents * (100 - price_cents)
    # ceiling division for positive ints
    return (numer + 9_999) // 10_000


def cost_cents(contracts: int, price_cents: int) -> int:
    """Cash cost of buying `contracts` at `price_cents` each, excluding fee."""
    if contracts <= 0:
        return 0
    return contracts * price_cents


def total_outlay_cents(contracts: int, price_cents: int) -> int:
    """Cash that will leave the account: price * count + fee."""
    return cost_cents(contracts, price_cents) + fee_cents(contracts, price_cents)
