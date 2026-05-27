"""
Unit tests for the broker simulator + smart router.
"""

from __future__ import annotations

import asyncio

import pytest

from trading_system.data.store import LOBSnapshot
from trading_system.db.repository import DatabaseRepository
from trading_system.execution.broker import AlpacaBroker
from trading_system.execution.market_maker import AvellanedaStoikovExecutor
from trading_system.execution.order_manager import OrderManager
from trading_system.execution.slippage import SlippageTracker
from trading_system.execution.smart_router import SmartOrderRouter


@pytest.fixture
def broker(test_config):
    return AlpacaBroker(test_config)


@pytest.fixture
def order_manager(test_config, broker):
    db = DatabaseRepository(test_config)
    return OrderManager(broker, db, SlippageTracker(db), cooldown_secs=1)


class TestSimulatedBroker:
    def test_buy_then_sell(self, broker):
        broker.update_sim_last_price("BTCUSDT", 100.0)
        res = asyncio.run(broker.submit_order("BTCUSDT", "buy", 0.5,
                                                "market", reference_price=100))
        assert res.success
        assert res.fill_qty == 0.5
        portfolio = asyncio.run(broker.get_portfolio_state())
        assert "BTCUSDT" in portfolio.positions
        # Sell
        res2 = asyncio.run(broker.submit_order("BTCUSDT", "sell", 0.5,
                                                 "market", reference_price=110))
        assert res2.success

    def test_rejects_oversized_buy(self, broker):
        broker.update_sim_last_price("BTCUSDT", 100.0)
        res = asyncio.run(broker.submit_order(
            "BTCUSDT", "buy", 10000, "market", reference_price=100))
        assert not res.success

    def test_close_position(self, broker):
        broker.update_sim_last_price("BTCUSDT", 100.0)
        asyncio.run(broker.submit_order("BTCUSDT", "buy", 0.5,
                                          reference_price=100))
        res = asyncio.run(broker.close_position("BTCUSDT"))
        assert res.success


class TestSmartRouter:
    def test_market_for_small_orders(self):
        router = SmartOrderRouter()
        plan = router.route("BTCUSDT", 100, "buy", 10_000, None, 1_000_000)
        assert plan["type"] == "market"

    def test_twap_for_large_orders(self):
        router = SmartOrderRouter()
        lob = LOBSnapshot(bids=[(100, 5)], asks=[(100.1, 5)])
        plan = router.route("BTCUSDT", 50_000, "buy", 100_000, lob, 1_000_000)
        assert plan["type"] == "twap"
        assert len(plan["schedule"]) >= 3

    def test_adaptive_limit_for_medium(self):
        router = SmartOrderRouter()
        lob = LOBSnapshot(bids=[(100, 5)], asks=[(100.1, 5)])
        plan = router.route("BTCUSDT", 2500, "buy", 100_000, lob, 1_000_000)
        assert plan["type"] == "adaptive_limit"


class TestAvellanedaStoikov:
    def test_optimal_quotes_bracket_mid(self):
        as_exec = AvellanedaStoikovExecutor(risk_aversion=0.1, k=1.5, T=1.0)
        quotes = as_exec.optimal_quotes(
            mid=100, inventory_qty=0, total_capital=10_000,
            asset_vol=0.02, time_remaining=0.5,
        )
        assert quotes["bid"] < quotes["reservation_price"] < quotes["ask"]

    def test_inventory_skews_reservation(self):
        as_exec = AvellanedaStoikovExecutor()
        q_long = as_exec.optimal_quotes(100, 5, 1000, 0.02, 1.0)
        q_short = as_exec.optimal_quotes(100, -5, 1000, 0.02, 1.0)
        assert q_long["reservation_price"] < q_short["reservation_price"]


class TestOrderManager:
    def test_cooldown(self, order_manager):
        assert order_manager.can_enter("BTCUSDT") is True
        order_manager.last_entry_time["BTCUSDT"] = __import__("time").time()
        assert order_manager.can_enter("BTCUSDT") is False
