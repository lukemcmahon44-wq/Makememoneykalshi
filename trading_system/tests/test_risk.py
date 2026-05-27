"""
Unit tests for circuit breakers + position sizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_system.db.repository import DatabaseRepository
from trading_system.risk.circuit_breaker import CircuitBreaker
from trading_system.risk.correlation import is_overcrowded
from trading_system.risk.sizer import PositionSizer
from trading_system.risk.var_engine import historical_var, parametric_var


@pytest.fixture
def db(test_config):
    return DatabaseRepository(test_config)


@pytest.fixture
def breaker(test_config, db):
    return CircuitBreaker(test_config, db)


class TestCircuitBreaker:
    def test_flash_crash_triggers(self, breaker):
        prices = [100, 99, 98, 95, 92]  # -8% in 5 bars
        assert breaker.check_flash_crash("BTCUSDT", prices) is True
        assert breaker.asset_is_suspended("BTCUSDT")

    def test_flash_crash_no_trigger(self, breaker):
        prices = [100, 100.1, 99.5, 99.8, 99.9]
        assert breaker.check_flash_crash("BTCUSDT", prices) is False

    def test_daily_loss_halts(self, breaker, test_config):
        assert breaker.check_daily_loss(-0.05) is True
        assert "DAILY_LOSS" in breaker.active_halts

    def test_daily_loss_recovers(self, breaker):
        breaker.check_daily_loss(-0.05)
        breaker.check_daily_loss(-0.005)
        assert "DAILY_LOSS" not in breaker.active_halts

    def test_drawdown_levels(self, breaker):
        # 10% dd → MODERATE (>= 8% but < 15%)
        lvl, dd = breaker.check_portfolio_drawdown(90.0, 100.0)
        assert lvl == "MODERATE"

        # 18% dd → SEVERE (>= 15% but < 25%)
        lvl, dd = breaker.check_portfolio_drawdown(82.0, 100.0)
        assert lvl == "SEVERE"

        # 30% dd → CRITICAL (>= 25%)
        lvl, dd = breaker.check_portfolio_drawdown(70.0, 100.0)
        assert lvl == "CRITICAL"

        # No drawdown
        lvl, dd = breaker.check_portfolio_drawdown(105.0, 100.0)
        assert lvl is None

    def test_position_stop_loss(self, breaker):
        position = {"direction": "long", "entry_price": 100.0,
                     "entry_atr": 1.0}
        hit, stop = breaker.check_position_loss("AAPL", position, 96.0)
        assert hit is True

        hit, stop = breaker.check_position_loss("AAPL", position, 98.0)
        assert hit is False

    def test_manual_reset(self, breaker):
        breaker.check_portfolio_drawdown(70.0, 100.0)
        assert breaker.requires_manual_reset
        breaker.manual_reset()
        assert not breaker.requires_manual_reset

    def test_asset_suspension_expires(self, breaker):
        breaker.suspend_asset("BTC", duration_hours=0.0001, reason="test")
        import time
        time.sleep(0.5)
        assert not breaker.asset_is_suspended("BTC")


class TestPositionSizer:
    def test_kelly_size_positive_edge(self, test_config):
        s = PositionSizer(test_config)
        sz = s.kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
        assert 0.01 <= sz <= 0.20

    def test_kelly_clamps_extreme(self, test_config):
        s = PositionSizer(test_config)
        # Floor: avg_loss=0 returns safe default
        sz = s.kelly_size(0.5, 0.01, 0.0)
        assert sz == 0.02

    def test_vol_scalar_bounds(self, test_config):
        s = PositionSizer(test_config)
        assert 0.3 <= s.volatility_scalar(0.1) <= 3.0
        assert 0.3 <= s.volatility_scalar(2.0) <= 3.0

    def test_correlation_penalty(self, test_config):
        s = PositionSizer(test_config)
        # No open positions: no penalty
        assert s.correlation_penalty("AAPL", {}, {}) == 1.0
        # High correlation: large penalty
        open_pos = {"MSFT": {}}
        corr = {("AAPL", "MSFT"): 0.9}
        assert s.correlation_penalty("AAPL", open_pos, corr) == 0.40

    def test_final_size_respects_caps(self, test_config):
        s = PositionSizer(test_config)
        size_usd, size_pct, _ = s.compute_final_size(
            "AAPL", signal=0.9, portfolio_value=100_000,
            trade_history=[], open_positions={}, corr_matrix={},
            asset_annual_vol=0.20, macro_multiplier=1.0, garch_scalar=1.0,
        )
        assert size_pct <= test_config.MAX_POSITION_PCT + 1e-6
        assert size_usd > 0

    def test_total_long_cap(self, test_config):
        s = PositionSizer(test_config)
        # 70% already long; cap at 80% means new can be at most 10%
        open_pos = {f"X{i}": {"direction": "long", "size_pct": 0.10}
                     for i in range(7)}
        size_usd, size_pct, _ = s.compute_final_size(
            "AAPL", 0.9, 100_000, [], open_pos, {}, 0.20, 1.0, 1.0)
        assert size_pct <= 0.11


class TestVaR:
    def test_historical_var_returns_negative(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 1000)
        var, cvar = historical_var(returns, 0.95)
        assert var < 0
        assert cvar <= var

    def test_parametric_var_close_to_historical(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 5000)
        hvar, hcvar = historical_var(returns, 0.95)
        pvar, pcvar = parametric_var(returns, 0.95)
        # Should be in same ballpark
        assert abs(hvar - pvar) < 0.01


class TestCorrelation:
    def test_overcrowded_detects(self):
        corr = {("BTC", "ETH"): 0.85}
        crowded, conflicts = is_overcrowded(
            "BTC", {"ETH": {}}, corr, threshold=0.7)
        assert crowded
        assert "ETH" in conflicts
