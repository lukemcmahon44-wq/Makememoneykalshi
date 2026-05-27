"""
Central configuration for the trading system.

All tunable parameters live here. Environment variables override defaults.
Never put numeric constants in trading logic files - they go here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "y", "on")


def _env_list(key: str, default: List[str]) -> List[str]:
    raw = os.getenv(key, "")
    if not raw:
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]


@dataclass
class TradingConfig:
    # ── Broker ────────────────────────────────────────────────────────────
    ALPACA_API_KEY: str = field(default_factory=lambda: _env("ALPACA_API_KEY"))
    ALPACA_SECRET_KEY: str = field(default_factory=lambda: _env("ALPACA_SECRET_KEY"))
    ALPACA_BASE_URL: str = field(default_factory=lambda: _env(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    PAPER_MODE: bool = field(default_factory=lambda: _env_bool("PAPER_MODE", True))

    # ── Data sources ──────────────────────────────────────────────────────
    FRED_API_KEY: str = field(default_factory=lambda: _env("FRED_API_KEY"))
    ALPHA_VANTAGE_KEY: str = field(default_factory=lambda: _env("ALPHA_VANTAGE_KEY"))
    REDDIT_CLIENT_ID: str = field(default_factory=lambda: _env("REDDIT_CLIENT_ID"))
    REDDIT_CLIENT_SECRET: str = field(default_factory=lambda: _env("REDDIT_CLIENT_SECRET"))
    REDDIT_USER_AGENT: str = field(default_factory=lambda: _env(
        "REDDIT_USER_AGENT", "AlgoTrader/1.0"))

    # ── Alerts ────────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    TELEGRAM_CHAT_ID: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))

    # ── Asset universe ────────────────────────────────────────────────────
    EQUITY_UNIVERSE: List[str] = field(default_factory=lambda: _env_list(
        "EQUITY_UNIVERSE",
        ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD"]))
    CRYPTO_UNIVERSE: List[str] = field(default_factory=lambda: _env_list(
        "CRYPTO_UNIVERSE",
        ["BTCUSD", "ETHUSD", "SOLUSD"]))
    BINANCE_UNIVERSE: List[str] = field(default_factory=lambda: _env_list(
        "BINANCE_UNIVERSE",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]))
    PAIRS_UNIVERSE: List[str] = field(default_factory=lambda: _env_list(
        "PAIRS_UNIVERSE",
        ["SPY/QQQ", "BTCUSD/ETHUSD"]))
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])

    # ── Risk parameters ───────────────────────────────────────────────────
    MAX_DRAWDOWN_PCT: float = field(default_factory=lambda: _env_float("MAX_DRAWDOWN_PCT", 0.08))
    DAILY_LOSS_LIMIT: float = field(default_factory=lambda: _env_float("DAILY_LOSS_LIMIT_PCT", 0.03))
    MAX_POSITION_PCT: float = field(default_factory=lambda: _env_float("MAX_POSITION_PCT", 0.15))
    MAX_TOTAL_LONG: float = field(default_factory=lambda: _env_float("MAX_TOTAL_LONG_PCT", 0.80))
    KELLY_FRACTION: float = field(default_factory=lambda: _env_float("KELLY_FRACTION", 0.25))
    TARGET_ANNUAL_VOL: float = field(default_factory=lambda: _env_float("TARGET_ANNUAL_VOL", 0.20))
    CVAR_LIMIT: float = field(default_factory=lambda: _env_float("CVAR_LIMIT", 0.02))

    # ── Drawdown levels ───────────────────────────────────────────────────
    DRAWDOWN_L1: float = 0.08
    DRAWDOWN_L2: float = 0.15
    DRAWDOWN_L3: float = 0.25

    # ── Signal thresholds ─────────────────────────────────────────────────
    SIGNAL_ENTRY_THRESHOLD: float = field(default_factory=lambda: _env_float(
        "SIGNAL_ENTRY_THRESHOLD", 0.55))
    HIGH_CONVICTION_THRESHOLD: float = field(default_factory=lambda: _env_float(
        "HIGH_CONVICTION_THRESHOLD", 0.75))
    COOLDOWN_MINUTES: int = field(default_factory=lambda: _env_int("COOLDOWN_MINUTES", 15))
    MIN_AGREEING_SIGNALS: int = 3
    MAX_DATA_STALENESS_SECS: int = 45
    MIN_ORDER_USD: float = 50.0

    # ── ML / training ─────────────────────────────────────────────────────
    ML_OOS_AUC_GATE: float = 0.54
    ML_OOS_ACC_GATE: float = 0.52
    TFT_ENCODER_LEN: int = 96
    TFT_PREDICTION_LEN: int = 12

    # ── System paths ──────────────────────────────────────────────────────
    LOG_LEVEL: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    DB_PATH: str = field(default_factory=lambda: _env("DB_PATH", "./trading_system/data/trading.db"))
    MODEL_DIR: str = field(default_factory=lambda: _env("MODEL_DIR", "./trading_system/models/"))
    DATA_CACHE_DIR: str = field(default_factory=lambda: _env(
        "DATA_CACHE_DIR", "./trading_system/data/cache/"))
    LOG_DIR: str = field(default_factory=lambda: _env("LOG_DIR", "./trading_system/logs/"))
    CLOSE_ON_SHUTDOWN: bool = field(default_factory=lambda: _env_bool("CLOSE_ON_SHUTDOWN", False))

    # ── Execution ─────────────────────────────────────────────────────────
    COMMISSION_EQUITY: float = 0.0
    COMMISSION_CRYPTO: float = 0.0015
    SLIPPAGE_EQUITY: float = 0.0002
    SLIPPAGE_CRYPTO: float = 0.0003

    def validate(self) -> List[str]:
        """Return list of issues; empty list = valid."""
        issues: List[str] = []
        if not self.PAPER_MODE and not (self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY):
            issues.append("Live mode requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
        if self.MAX_DRAWDOWN_PCT <= 0 or self.MAX_DRAWDOWN_PCT > 0.5:
            issues.append("MAX_DRAWDOWN_PCT must be in (0, 0.5]")
        if self.KELLY_FRACTION <= 0 or self.KELLY_FRACTION > 1.0:
            issues.append("KELLY_FRACTION must be in (0, 1]")
        if self.TARGET_ANNUAL_VOL <= 0:
            issues.append("TARGET_ANNUAL_VOL must be > 0")
        return issues

    def ensure_dirs(self) -> None:
        for path in (self.MODEL_DIR, self.DATA_CACHE_DIR, self.LOG_DIR,
                     str(Path(self.DB_PATH).parent)):
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def all_assets(self) -> List[str]:
        return list(self.EQUITY_UNIVERSE) + list(self.CRYPTO_UNIVERSE)

    def asset_type(self, asset: str) -> str:
        if asset in self.CRYPTO_UNIVERSE or asset.upper().endswith("USD") or asset.upper().endswith("USDT"):
            return "crypto"
        return "equity"

    def pairs_parsed(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for p in self.PAIRS_UNIVERSE:
            if "/" in p:
                a, b = p.split("/", 1)
                out.append((a.strip(), b.strip()))
        return out


_singleton: TradingConfig | None = None


def load_config() -> TradingConfig:
    """Singleton config loader. Validates and ensures directories exist."""
    global _singleton
    if _singleton is None:
        cfg = TradingConfig()
        cfg.ensure_dirs()
        issues = cfg.validate()
        if issues:
            from loguru import logger
            for issue in issues:
                logger.warning(f"CONFIG ISSUE: {issue}")
        _singleton = cfg
    return _singleton
