"""
Centralized configuration management for the trading system.
Loads environment variables via python-dotenv, provides typed attributes,
validates required keys, and exposes a singleton accessor.
"""

from __future__ import annotations

import os
from typing import List, Optional
from dotenv import load_dotenv


class TradingConfig:
    """
    Singleton configuration class.  Reads from the environment (and an
    optional .env file) at construction time and exposes typed attributes
    for every system parameter.
    """

    # ------------------------------------------------------------------
    # Alpaca / Broker
    # ------------------------------------------------------------------
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_BASE_URL: str
    PAPER_MODE: bool

    # ------------------------------------------------------------------
    # External data providers
    # ------------------------------------------------------------------
    FRED_API_KEY: str
    ALPHA_VANTAGE_KEY: str

    # ------------------------------------------------------------------
    # Reddit / social sentiment
    # ------------------------------------------------------------------
    REDDIT_CLIENT_ID: str
    REDDIT_CLIENT_SECRET: str
    REDDIT_USER_AGENT: str

    # ------------------------------------------------------------------
    # Telegram notifications
    # ------------------------------------------------------------------
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # ------------------------------------------------------------------
    # Asset universes
    # ------------------------------------------------------------------
    EQUITY_UNIVERSE: List[str]
    CRYPTO_UNIVERSE: List[str]
    PAIRS_UNIVERSE: List[str]

    # ------------------------------------------------------------------
    # Risk parameters
    # ------------------------------------------------------------------
    MAX_DRAWDOWN_PCT: float
    DAILY_LOSS_LIMIT: float
    MAX_POSITION_PCT: float
    MAX_TOTAL_LONG: float
    KELLY_FRACTION: float
    TARGET_ANNUAL_VOL: float
    CVAR_LIMIT: float

    # ------------------------------------------------------------------
    # Signal thresholds
    # ------------------------------------------------------------------
    SIGNAL_ENTRY_THRESHOLD: float
    HIGH_CONVICTION_THRESHOLD: float
    COOLDOWN_MINUTES: int

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------
    LOG_LEVEL: str
    DB_PATH: str
    MODEL_DIR: str
    DATA_CACHE_DIR: str
    CLOSE_ON_SHUTDOWN: bool

    def __init__(self, env_file: Optional[str] = None) -> None:
        # Load .env file if it exists; environment variables already set
        # take precedence (override=False).
        load_dotenv(dotenv_path=env_file, override=False)

        # ---- Alpaca / Broker ----------------------------------------
        self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
        self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
        self.ALPACA_BASE_URL = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self.PAPER_MODE = self._parse_bool(os.getenv("PAPER_MODE", "true"))

        # ---- External data providers --------------------------------
        self.FRED_API_KEY = os.getenv("FRED_API_KEY", "")
        self.ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

        # ---- Reddit --------------------------------------------------
        self.REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
        self.REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.REDDIT_USER_AGENT = os.getenv(
            "REDDIT_USER_AGENT", "trading_system/1.0"
        )

        # ---- Telegram ------------------------------------------------
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

        # ---- Asset universes -----------------------------------------
        self.EQUITY_UNIVERSE = self._parse_list(
            os.getenv(
                "EQUITY_UNIVERSE",
                "SPY,QQQ,AAPL,NVDA,TSLA,MSFT,AMZN,META,GOOGL,AMD",
            )
        )
        self.CRYPTO_UNIVERSE = self._parse_list(
            os.getenv("CRYPTO_UNIVERSE", "BTCUSD,ETHUSD,SOLUSD")
        )
        self.PAIRS_UNIVERSE = self._parse_list(
            os.getenv("PAIRS_UNIVERSE", "SPY/QQQ,BTCUSD/ETHUSD")
        )

        # ---- Risk parameters -----------------------------------------
        self.MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.08"))
        self.DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))
        self.MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.15"))
        self.MAX_TOTAL_LONG = float(os.getenv("MAX_TOTAL_LONG", "0.80"))
        self.KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
        self.TARGET_ANNUAL_VOL = float(os.getenv("TARGET_ANNUAL_VOL", "0.20"))
        self.CVAR_LIMIT = float(os.getenv("CVAR_LIMIT", "0.02"))

        # ---- Signal thresholds ---------------------------------------
        self.SIGNAL_ENTRY_THRESHOLD = float(
            os.getenv("SIGNAL_ENTRY_THRESHOLD", "0.55")
        )
        self.HIGH_CONVICTION_THRESHOLD = float(
            os.getenv("HIGH_CONVICTION_THRESHOLD", "0.75")
        )
        self.COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "15"))

        # ---- System --------------------------------------------------
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        self.DB_PATH = os.getenv("DB_PATH", "./data/trading.db")
        self.MODEL_DIR = os.getenv("MODEL_DIR", "./models/")
        self.DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "./data/cache/")
        self.CLOSE_ON_SHUTDOWN = self._parse_bool(
            os.getenv("CLOSE_ON_SHUTDOWN", "false")
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate configuration.  Raises ValueError if truly required keys
        (ALPACA_API_KEY, ALPACA_SECRET_KEY) are missing.  Logs warnings for
        optional keys that are empty.
        """
        # Import here to avoid circular dependency at module load time.
        from trading_system.core.logger import get_logger

        log = get_logger(__name__)

        # Truly required
        missing_required: list[str] = []
        if not self.ALPACA_API_KEY:
            missing_required.append("ALPACA_API_KEY")
        if not self.ALPACA_SECRET_KEY:
            missing_required.append("ALPACA_SECRET_KEY")

        if missing_required:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_required)}"
            )

        # Optional but warn if absent
        optional_recommended = {
            "FRED_API_KEY": self.FRED_API_KEY,
            "ALPHA_VANTAGE_KEY": self.ALPHA_VANTAGE_KEY,
            "REDDIT_CLIENT_ID": self.REDDIT_CLIENT_ID,
            "REDDIT_CLIENT_SECRET": self.REDDIT_CLIENT_SECRET,
            "TELEGRAM_BOT_TOKEN": self.TELEGRAM_BOT_TOKEN,
            "TELEGRAM_CHAT_ID": self.TELEGRAM_CHAT_ID,
        }
        for key, value in optional_recommended.items():
            if not value:
                log.warning(
                    "Optional config key is not set — some features will be disabled",
                    key=key,
                )

        log.info(
            "Configuration validated successfully",
            paper_mode=self.PAPER_MODE,
            equity_universe_size=len(self.EQUITY_UNIVERSE),
            crypto_universe_size=len(self.CRYPTO_UNIVERSE),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_bool(value: str) -> bool:
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _parse_list(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TradingConfig(paper_mode={self.PAPER_MODE}, "
            f"alpaca_url={self.ALPACA_BASE_URL}, "
            f"db={self.DB_PATH})"
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[TradingConfig] = None


def get_config(env_file: Optional[str] = None) -> TradingConfig:
    """
    Return the global TradingConfig singleton.  On the first call the
    instance is created (optionally loading *env_file*).  Subsequent calls
    ignore *env_file* and return the cached instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = TradingConfig(env_file=env_file)
    return _config_instance


def reset_config() -> None:
    """
    Destroy the singleton so the next call to get_config() creates a fresh
    instance.  Useful in tests.
    """
    global _config_instance
    _config_instance = None
