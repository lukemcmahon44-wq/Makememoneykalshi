"""Sizer math — fee-aware, integer-cent, no-double-spend at the edges."""

from strategy.sizer import (
    can_afford_cheapest,
    size_all_in,
    size_fixed_dollar,
)


def test_fixed_dollar_one_contract_at_95_with_one_dollar_budget():
    # $1.00 budget, fixed slice $1.00, price 95¢ → 1 contract, fee 1¢, outlay 96¢.
    d = size_fixed_dollar(
        working_budget_cents=100,
        fixed_trade_size_cents=100,
        price_cents=95,
        available_size=10,
    )
    assert d.contracts == 1
    assert d.cost_cents == 95
    assert d.fee_cents == 1
    assert d.total_outlay_cents == 96


def test_fixed_dollar_cannot_afford_one_contract():
    # 50¢ working budget, price 95¢: no way.
    d = size_fixed_dollar(
        working_budget_cents=50,
        fixed_trade_size_cents=100,
        price_cents=95,
        available_size=10,
    )
    assert d.contracts == 0
    assert "cannot afford" in d.reason


def test_fixed_dollar_caps_at_slice_not_total_budget():
    # Plenty of total budget but slice is $1 — only 1 contract.
    d = size_fixed_dollar(
        working_budget_cents=10_000,
        fixed_trade_size_cents=100,
        price_cents=95,
        available_size=99,
    )
    assert d.contracts == 1


def test_fixed_dollar_caps_at_resting_liquidity():
    d = size_fixed_dollar(
        working_budget_cents=10_000,
        fixed_trade_size_cents=1_000,   # would fit ~10 contracts at 95¢
        price_cents=95,
        available_size=3,
    )
    assert d.contracts == 3


def test_fixed_dollar_trims_to_fit_fee():
    # 96¢ budget, 1 contract at 95¢ + 1¢ fee = 96¢ outlay -> fits exactly.
    fits = size_fixed_dollar(
        working_budget_cents=96,
        fixed_trade_size_cents=96,
        price_cents=95,
        available_size=10,
    )
    assert fits.contracts == 1
    # 95¢ budget — price alone fits but fee doesn't, so we trim to 0.
    trims = size_fixed_dollar(
        working_budget_cents=95,
        fixed_trade_size_cents=95,
        price_cents=95,
        available_size=10,
    )
    assert trims.contracts == 0


def test_all_in_uses_full_balance():
    # $10 balance, price 95¢ + 1¢ fee per ~ batch. The integer-cent loop
    # finds the largest n such that 95n + fee(n,95) <= 1000.
    # Try 10: cost 950, fee ceil(7*10*95*5/10000) = ceil(33250/10000) = 4 → 954 ≤ 1000 ✓
    d = size_all_in(
        working_budget_cents=1000,
        price_cents=95,
        available_size=999,
    )
    assert d.contracts == 10
    assert d.cost_cents == 950
    assert d.fee_cents == 4
    assert d.total_outlay_cents == 954


def test_all_in_zero_budget():
    d = size_all_in(working_budget_cents=0, price_cents=95, available_size=10)
    assert d.contracts == 0
    assert "no working budget" in d.reason


def test_all_in_capped_by_resting_size():
    # Plenty of cash but only 4 contracts resting at the ask.
    d = size_all_in(
        working_budget_cents=10_000,
        price_cents=95,
        available_size=4,
    )
    assert d.contracts == 4


def test_can_afford_cheapest_boundary():
    # 95¢ price needs 95 + ceil(7*1*95*5/10000)=1 = 96¢ outlay.
    assert can_afford_cheapest(working_budget_cents=96, cheapest_price_cents=95)
    assert not can_afford_cheapest(working_budget_cents=95, cheapest_price_cents=95)


def test_invalid_price_yields_skip():
    d = size_fixed_dollar(
        working_budget_cents=100, fixed_trade_size_cents=100, price_cents=0, available_size=1,
    )
    assert d.contracts == 0
    assert "invalid price" in d.reason
    d = size_all_in(working_budget_cents=100, price_cents=100, available_size=1)
    assert d.contracts == 0
    assert "invalid price" in d.reason
