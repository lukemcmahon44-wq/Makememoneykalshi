"""
End-to-end integration tests. Verifies the core orchestration paths work.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest.engine import BacktestEngine
from trading_system.data.store import Candle, MarketDataStore
from trading_system.db.repository import DatabaseRepository
from trading_system.execution.broker import AlpacaBroker
from trading_system.portfolio.manager import PortfolioManager
from trading_system.risk.sizer import PositionSizer
from trading_system.signals.ensemble import SignalEnsemble


def test_market_data_store_basic():
    store = MarketDataStore()
    cd = Candle(ts=int(time.time()), open=100, high=101, low=99, close=100.5,
                 volume=1000)
    store.append_candle("BTCUSDT", "1m", cd)
    assert store.latest_close("BTCUSDT", "1m") == 100.5
    assert len(store.get_candles("BTCUSDT", "1m", allow_stale=True)) == 1


def test_data_store_staleness():
    store = MarketDataStore(max_staleness_secs=1)
    cd = Candle(ts=int(time.time()) - 100, open=1, high=1, low=1, close=1,
                 volume=1)
    store.append_candle("BTCUSDT", "1m", cd)
    # force last_update_time backward
    store.last_update_time["BTCUSDT"] = time.time() - 100
    from trading_system.data.store import DataStaleError
    with pytest.raises(DataStaleError):
        store.get_candles("BTCUSDT", "1m", allow_stale=False)


def test_db_writes_and_reads(test_config):
    db = DatabaseRepository(test_config)
    db.write_candle("BTCUSDT", "1m", int(time.time()), 1, 2, 0.5, 1.5, 100)
    rows = db.get_candles("BTCUSDT", "1m", limit=5)
    assert len(rows) == 1
    db.write_nav(int(time.time()), 10_000, 5_000, 5_000, 2)
    assert db.get_portfolio_value() == 10_000


def test_signal_ensemble_threshold():
    e = SignalEnsemble()
    decision, conviction = e.threshold_check(0.85)
    assert decision == "ENTER" and conviction == "HIGH"
    decision, _ = e.threshold_check(0.20)
    assert decision == "NO_SIGNAL"


def test_signal_ensemble_composite():
    e = SignalEnsemble()
    e.disable_ml()
    signals = {
        "technical_trend": 0.6, "technical_reversion": -0.1,
        "microstructure": 0.7, "stat_arb": 0.0, "sentiment": 0.3,
        "onchain": 0.4, "ml_prediction": 0.0,
    }
    composite, meta = e.compute_composite(signals, {"adx_regime": "trending"},
                                            asset_type="crypto")
    assert -1.0 <= composite <= 1.0


def test_portfolio_manager_lifecycle(test_config):
    db = DatabaseRepository(test_config)
    pm = PortfolioManager(db, starting_cash=10_000)
    pm.open_position("BTCUSDT", "long", 0.1, 100.0, 1.0, 0.7, 0.10, 1000)
    nav = pm.update_nav({"BTCUSDT": 110.0})
    assert nav["total"] > 10_000   # profit
    outcome = pm.close_position("BTCUSDT", 110.0)
    assert outcome["pnl"] > 0


def test_backtest_engine_runs(test_config):
    """Mini backtest with sine-wave price + simple signal."""
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    base = 100 + 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0, 0.05, n)
    close = base + noise
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": rng.uniform(100, 1000, n),
    }, index=idx)
    data = {"BTCUSDT": df}

    def signal_fn(asset, hist):
        if len(hist) < 30:
            return 0.0
        c = hist["close"].values
        z = (c[-1] - np.mean(c[-30:])) / (np.std(c[-30:]) + 1e-8)
        return float(np.clip(-z / 2, -1.0, 1.0))   # mean reversion

    engine = BacktestEngine(test_config)
    sizer = PositionSizer(test_config)
    result = engine.run(data, signal_fn, sizer, start_capital=10_000)
    assert "metrics" in result
    assert "trades" in result
    assert isinstance(result["metrics"].get("sharpe"), float)


def test_simulated_broker_buy_sell(test_config):
    broker = AlpacaBroker(test_config)
    broker.update_sim_last_price("BTCUSDT", 50.0)
    res = asyncio.run(broker.submit_order("BTCUSDT", "buy", 1.0, "market",
                                            reference_price=50.0))
    assert res.success
    sell = asyncio.run(broker.submit_order("BTCUSDT", "sell", 1.0, "market",
                                              reference_price=60.0))
    assert sell.success
