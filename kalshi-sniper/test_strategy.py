"""
test_strategy.py — Unit tests for the pure selection + sizing logic.

Run from this directory:
    python -m pytest -v
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import config
import strategy
from strategy import IntendedOrder, select_and_size

# A fixed clock so close-time filters are deterministic.
NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)

COUNT_FP_RE = re.compile(r"^\d+\.\d{2}$")
PRICE_RE = re.compile(r"^\d+\.\d{4}$")


def make_market(
    ticker: str = "TEST-MKT",
    ask: str | None = "0.97",
    size: str = "1000",
    status: str = "open",
    close_in_minutes: int = 600,
    fractional: bool = False,
    structure: str = "linear_cent",
) -> dict:
    close = NOW + timedelta(minutes=close_in_minutes)
    return {
        "ticker": ticker,
        "status": status,
        "yes_ask_dollars": ask,
        "yes_ask_size_fp": size,
        "close_time": close.isoformat().replace("+00:00", "Z"),
        "fractional_trading_enabled": fractional,
        "price_level_structure": structure,
    }


def run(markets, balance="10", held=None):
    return select_and_size(markets, Decimal(balance), held or set(), now=NOW)


# ── Price-band filter (the 95–99¢ rule) ────────────────────────────────────────
def test_lower_bound_inclusive_095():
    assert len(run([make_market(ask="0.95")])) == 1


def test_just_below_band_094_excluded():
    assert run([make_market(ask="0.94")]) == []


def test_upper_bound_inclusive_099():
    assert len(run([make_market(ask="0.99")])) == 1


def test_one_dollar_excluded():
    # 1.00 implies certainty: no edge, no profit. Must be excluded.
    assert run([make_market(ask="1.00")]) == []


# ── Sizing ──────────────────────────────────────────────────────────────────--
def test_ten_dollars_ten_percent_one_contract_at_097():
    # $10 * 10% = $1 target; $1 / $0.97 = 1.03 -> rounds DOWN to 1 contract.
    res = run([make_market(ask="0.97")], balance="10")
    assert len(res) == 1
    order = res[0]
    assert order.count == Decimal("1")
    assert order.price == Decimal("0.97")
    assert order.dollar_amount == Decimal("0.97")
    assert order.count_fp == "1.00"
    assert order.yes_price_dollars == "0.9700"


def test_per_market_cap_applied():
    # Big balance: 10% would be huge, but MAX_POSITION_PER_MARKET caps the target.
    res = run([make_market(ask="0.95", size="100000")], balance="100000")
    assert len(res) == 1
    # contracts * price must not exceed the per-market cap.
    assert res[0].dollar_amount <= config.MAX_POSITION_PER_MARKET


# ── Dedup ─────────────────────────────────────────────────────────────────────
def test_held_ticker_excluded():
    assert run([make_market(ticker="HELD")], held={"HELD"}) == []


# ── Time-to-close filter ────────────────────────────────────────────────────--
def test_closing_too_soon_excluded():
    assert run([make_market(close_in_minutes=3)]) == []


def test_closing_after_floor_included():
    assert len(run([make_market(close_in_minutes=config.MIN_MINUTES_TO_CLOSE + 1)])) == 1


# ── Liquidity filter ────────────────────────────────────────────────────────--
def test_insufficient_liquidity_excluded():
    too_thin = str(config.MIN_LIQUIDITY_CONTRACTS - 1)
    assert run([make_market(size=too_thin)]) == []


def test_liquidity_at_floor_included():
    enough = str(config.MIN_LIQUIDITY_CONTRACTS)
    assert len(run([make_market(size=enough)])) == 1


# ── Status filter ─────────────────────────────────────────────────────────────
def test_non_open_market_excluded():
    assert run([make_market(status="closed")]) == []


def test_missing_ask_excluded():
    assert run([make_market(ask=None)]) == []


# ── Fractional handling ────────────────────────────────────────────────────────
def test_no_fractional_contracts_when_disabled():
    # Balance chosen so the raw size is fractional (4.7 / 0.97 = 4.84...).
    res = run([make_market(ask="0.97", fractional=False)], balance="47")
    assert len(res) == 1
    count = res[0].count
    assert count == count.to_integral_value()  # whole contracts only
    assert count == Decimal("4")


def test_fractional_contracts_when_enabled():
    res = run([make_market(ask="0.97", fractional=True)], balance="47")
    assert len(res) == 1
    # 4.7 / 0.97 = 4.8453... -> floored to the 0.01 increment.
    assert res[0].count == Decimal("4.84")


# ── Ordering ───────────────────────────────────────────────────────────────────
def test_candidates_sorted_by_price_descending():
    markets = [
        make_market(ticker="A", ask="0.95"),
        make_market(ticker="C", ask="0.99"),
        make_market(ticker="B", ask="0.97"),
    ]
    res = run(markets)
    prices = [o.price for o in res]
    assert prices == sorted(prices, reverse=True)
    assert [o.ticker for o in res] == ["C", "B", "A"]


# ── Decimal precision: no float ever in the money path ─────────────────────────
def test_no_float_in_output():
    res = run([make_market(ask="0.96")])
    assert len(res) == 1
    order = res[0]
    for value in (order.count, order.price, order.dollar_amount):
        assert isinstance(value, Decimal)
        assert not isinstance(value, float)
    # The exact strings handed to the API must be well-formed fixed-point.
    assert COUNT_FP_RE.match(order.count_fp)
    assert PRICE_RE.match(order.yes_price_dollars)


def test_deci_cent_tick_alignment():
    # A deci-cent market with a sub-penny ask keeps three decimals.
    res = run([make_market(ask="0.955", structure="deci_cent")])
    assert len(res) == 1
    assert res[0].price == Decimal("0.955")
    assert res[0].yes_price_dollars == "0.9550"
