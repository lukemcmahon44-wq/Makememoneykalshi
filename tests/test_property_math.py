"""
Property-style invariants on the money math — covered with explicit cases
rather than hypothesis to keep the dependency footprint small.
"""

import math
import random

from strategy.fees import fee_cents, total_outlay_cents
from strategy.sizer import can_afford_cheapest, size_all_in, size_fixed_dollar


def test_fee_is_nonnegative_for_every_valid_input():
    for contracts in (1, 2, 5, 10, 100, 1000, 10000):
        for price in range(1, 100):
            assert fee_cents(contracts, price) >= 0


def test_total_outlay_strictly_greater_than_cost_when_fee_positive():
    for contracts in (1, 5, 50, 500):
        for price in range(1, 100):
            cost = contracts * price
            outlay = total_outlay_cents(contracts, price)
            assert outlay >= cost
            if price > 0 and price < 100:
                assert outlay > cost   # fee > 0 for every interior price


def test_size_fixed_dollar_never_exceeds_budget():
    rng = random.Random(42)
    for _ in range(500):
        budget = rng.randint(0, 10_000)
        slice_ = rng.randint(0, 5_000)
        price = rng.randint(1, 99)
        size = rng.randint(0, 200)
        d = size_fixed_dollar(
            working_budget_cents=budget,
            fixed_trade_size_cents=slice_,
            price_cents=price,
            available_size=size,
        )
        if d.contracts > 0:
            assert d.total_outlay_cents <= min(budget, slice_), (budget, slice_, price, d)


def test_size_all_in_never_exceeds_budget():
    rng = random.Random(43)
    for _ in range(500):
        budget = rng.randint(0, 10_000)
        price = rng.randint(1, 99)
        size = rng.randint(0, 200)
        d = size_all_in(
            working_budget_cents=budget,
            price_cents=price,
            available_size=size,
        )
        if d.contracts > 0:
            assert d.total_outlay_cents <= budget


def test_size_all_in_is_maximal_within_budget():
    """If we return n contracts, n+1 must not fit."""
    rng = random.Random(44)
    for _ in range(200):
        budget = rng.randint(96, 10_000)
        price = rng.randint(80, 99)
        d = size_all_in(working_budget_cents=budget, price_cents=price, available_size=10**9)
        if d.contracts > 0:
            next_outlay = total_outlay_cents(d.contracts + 1, price)
            assert next_outlay > budget, (budget, price, d.contracts, next_outlay)


def test_can_afford_cheapest_agrees_with_size_fixed_dollar_at_minimum():
    for price in range(1, 100):
        for budget in range(0, 200):
            affordable = can_afford_cheapest(
                working_budget_cents=budget, cheapest_price_cents=price,
            )
            d = size_fixed_dollar(
                working_budget_cents=budget,
                fixed_trade_size_cents=budget,
                price_cents=price,
                available_size=10**9,
            )
            # The two predicates must agree.
            assert affordable == (d.contracts >= 1), (price, budget, affordable, d.contracts)
