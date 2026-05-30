"""Fee math — integer cents, no float drift."""

import math

from strategy.fees import cost_cents, fee_cents, total_outlay_cents


def _reference_fee_cents(contracts: int, price_cents: int) -> int:
    """The published formula, computed with floats, then ceiled to a cent.
    Kept around so the tests double-check the integer implementation."""
    if contracts <= 0 or price_cents <= 0 or price_cents >= 100:
        return 0
    p = price_cents / 100.0
    raw_dollars = 0.07 * contracts * p * (1 - p)
    return math.ceil(raw_dollars * 100)


def test_fee_zero_for_nonpositive_inputs():
    assert fee_cents(0, 95) == 0
    assert fee_cents(5, 0) == 0
    assert fee_cents(5, 100) == 0
    assert fee_cents(-1, 95) == 0


def test_fee_matches_published_formula_across_band():
    # At 95–99¢ we expect 1¢ per contract for small counts (formula is tiny).
    for price in range(95, 100):
        for count in (1, 2, 5, 10, 50):
            assert fee_cents(count, price) == _reference_fee_cents(count, price), (
                count, price, fee_cents(count, price), _reference_fee_cents(count, price)
            )


def test_fee_is_one_cent_at_95_cents_one_contract():
    # Spec example: about $0.01 per contract at $0.95.
    assert fee_cents(1, 95) == 1


def test_fee_is_one_cent_at_99_cents_one_contract():
    assert fee_cents(1, 99) == 1


def test_fee_ceils_not_floors():
    # 1 contract at 50¢ has raw fee 0.07 * 1 * 0.5 * 0.5 = $0.0175 = 1.75¢ → 2¢
    assert fee_cents(1, 50) == 2


def test_cost_and_outlay():
    assert cost_cents(3, 95) == 285
    # Fee at 95¢ for 3 contracts: ceil(7*3*95*5 / 10000) = ceil(9975/10000) = 1
    assert fee_cents(3, 95) == 1
    assert total_outlay_cents(3, 95) == 286


def test_outlay_for_one_dollar_budget_at_95():
    # Spec example: at $1.00 budget, ~1 contract at 95¢ + 1¢ fee = 96¢ outlay.
    assert total_outlay_cents(1, 95) == 96
