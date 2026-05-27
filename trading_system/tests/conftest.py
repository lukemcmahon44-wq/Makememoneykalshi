"""
Shared pytest fixtures.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root on sys.path when running tests directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def test_config(tmp_path):
    from trading_system.core.config import TradingConfig
    cfg = TradingConfig()
    cfg.DB_PATH = str(tmp_path / "test.db")
    cfg.MODEL_DIR = str(tmp_path / "models")
    cfg.DATA_CACHE_DIR = str(tmp_path / "cache")
    cfg.LOG_DIR = str(tmp_path / "logs")
    cfg.PAPER_MODE = True
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def synthetic_candles():
    """Generate 500 synthetic 1m candles with a mild uptrend + noise."""
    import time

    import numpy as np

    from trading_system.data.store import Candle
    rng = np.random.default_rng(42)
    n = 500
    drift = 0.00005
    sigma = 0.0008
    log_prices = np.cumsum(rng.normal(drift, sigma, n)) + np.log(100)
    prices = np.exp(log_prices)
    ts0 = int(time.time()) - n * 60
    candles = []
    for i in range(n):
        c_close = float(prices[i])
        c_open = float(prices[i - 1]) if i > 0 else c_close
        c_high = max(c_open, c_close) * (1 + abs(rng.normal(0, 0.0005)))
        c_low = min(c_open, c_close) * (1 - abs(rng.normal(0, 0.0005)))
        candles.append(Candle(
            ts=ts0 + i * 60, open=c_open, high=c_high, low=c_low,
            close=c_close, volume=float(rng.uniform(100, 1000)),
        ))
    return candles
