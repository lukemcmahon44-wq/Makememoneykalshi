"""
Historical data bootstrapper.

Downloads 2-year OHLCV history for all assets in the universe and seeds the
MarketDataStore.  Caches results as Parquet files so subsequent starts are
instantaneous.

Methods
-------
download_equity_history(ticker, period, interval) -> pd.DataFrame
    Fetch equity history via yfinance.

download_crypto_history(symbol, days) -> pd.DataFrame
    Fetch crypto klines from the Binance public REST API.

bootstrap_all(config, store) -> None
    Bootstraps every asset in the configured universe.

save_to_cache(df, asset, timeframe, cache_dir) -> Path
    Write a DataFrame to a Parquet file.

load_from_cache(asset, timeframe, cache_dir) -> pd.DataFrame | None
    Read a Parquet file back, or return None if it doesn't exist/is stale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_KLINE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}
BINANCE_KLINE_LIMIT = 1000       # max rows per REST request
BINANCE_REQUEST_DELAY = 0.2      # seconds between paginated requests

# Binance symbol aliases (store key -> Binance REST symbol)
BINANCE_SYMBOL_MAP: Dict[str, str] = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "BNBUSD": "BNBUSDT",
}

# Cache staleness: don't re-download if file is younger than this
CACHE_MAX_AGE_SECONDS = 3600 * 6   # 6 hours

# OHLCV column names (used for both equity and crypto DataFrames)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "timestamp"]


# ---------------------------------------------------------------------------
# Parquet cache helpers
# ---------------------------------------------------------------------------

def _cache_path(asset: str, timeframe: str, cache_dir: str) -> Path:
    """Return the Parquet file path for (asset, timeframe)."""
    safe_asset = asset.replace("/", "_").upper()
    return Path(cache_dir) / f"{safe_asset}_{timeframe}.parquet"


def save_to_cache(df: pd.DataFrame, asset: str, timeframe: str, cache_dir: str) -> Path:
    """
    Persist *df* as a compressed Parquet file.

    Parameters
    ----------
    df :
        OHLCV DataFrame with DatetimeIndex.
    asset :
        Asset identifier (e.g. ``"BTCUSD"``).
    timeframe :
        Timeframe string (e.g. ``"1d"``).
    cache_dir :
        Directory where the Parquet file will be stored.

    Returns
    -------
    Path
        The path to the written file.
    """
    path = _cache_path(asset, timeframe, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=True)
    logger.info("[downloader] Saved %d rows to cache: %s", len(df), path)
    return path


def load_from_cache(
    asset: str,
    timeframe: str,
    cache_dir: str,
    max_age_seconds: int = CACHE_MAX_AGE_SECONDS,
) -> Optional[pd.DataFrame]:
    """
    Load cached OHLCV data if the file exists and is fresh enough.

    Parameters
    ----------
    asset, timeframe :
        Identify the asset/timeframe pair.
    cache_dir :
        Directory to search.
    max_age_seconds :
        If the file is older than this, return ``None`` so the caller
        re-downloads.

    Returns
    -------
    pd.DataFrame or None
    """
    path = _cache_path(asset, timeframe, cache_dir)
    if not path.exists():
        return None
    file_age = time.time() - path.stat().st_mtime
    if file_age > max_age_seconds:
        logger.debug(
            "[downloader] Cache file for %s/%s is %.0fh old; will re-download.",
            asset, timeframe, file_age / 3600,
        )
        return None
    try:
        df = pd.read_parquet(path)
        logger.info("[downloader] Loaded %d rows from cache: %s", len(df), path)
        return df
    except Exception as exc:
        logger.warning("[downloader] Failed to read cache %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Equity history (yfinance)
# ---------------------------------------------------------------------------

def download_equity_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch equity OHLCV history using yfinance.

    Parameters
    ----------
    ticker :
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``).
    period :
        Data look-back period understood by yfinance (e.g. ``"2y"``, ``"1y"``).
    interval :
        Bar interval (e.g. ``"1d"``, ``"1h"``, ``"1m"``).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, vwap, timestamp
        Index: UTC DatetimeIndex
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from exc

    logger.info("[downloader] Downloading equity %s (period=%s, interval=%s)…", ticker, period, interval)
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed for {ticker}: {exc}") from exc

    if raw.empty:
        logger.warning("[downloader] yfinance returned empty DataFrame for %s.", ticker)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    # Flatten MultiIndex columns if yfinance returned them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    df = pd.DataFrame(index=raw.index)
    df["open"] = raw.get("open", pd.Series(dtype=float))
    df["high"] = raw.get("high", pd.Series(dtype=float))
    df["low"] = raw.get("low", pd.Series(dtype=float))
    df["close"] = raw.get("close", pd.Series(dtype=float))
    df["volume"] = raw.get("volume", pd.Series(dtype=float)).fillna(0.0)

    # Compute VWAP as (high+low+close)/3 weighted proxy (typical price)
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

    # Timestamp as unix float
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df["timestamp"] = df.index.astype("int64") / 1e9

    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index.name = "datetime"
    return df[OHLCV_COLUMNS]


# ---------------------------------------------------------------------------
# Crypto history (Binance REST)
# ---------------------------------------------------------------------------

def download_crypto_history(
    symbol: str,
    days: int = 730,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch crypto OHLCV history from the Binance public REST API.

    Parameters
    ----------
    symbol :
        Binance trading pair (e.g. ``"BTCUSDT"``).  If a store key like
        ``"BTCUSD"`` is supplied it is mapped automatically.
    days :
        Number of calendar days of history to fetch (default 730 = ~2 years).
    interval :
        Kline interval string understood by Binance
        (e.g. ``"1d"``, ``"1h"``, ``"1m"``).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, vwap, timestamp
        Index: UTC DatetimeIndex
    """
    import requests  # standard library or already in requirements

    # Map store key to Binance symbol
    binance_sym = BINANCE_SYMBOL_MAP.get(symbol.upper(), symbol.upper())
    logger.info(
        "[downloader] Downloading crypto %s (Binance: %s, days=%d, interval=%s)…",
        symbol, binance_sym, days, interval,
    )

    end_ms = int(time.time() * 1000)
    start_ms = int((time.time() - days * 86400) * 1000)

    all_rows: List[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": binance_sym,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": BINANCE_KLINE_LIMIT,
        }
        try:
            resp = requests.get(
                BINANCE_KLINES_URL,
                params=params,
                timeout=20,
            )
            resp.raise_for_status()
            rows = resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Binance klines request failed for {binance_sym}: {exc}") from exc

        if not rows:
            break

        all_rows.extend(rows)

        # The last row's close time tells us where we got to
        last_close_ms = int(rows[-1][6])
        current_start = last_close_ms + 1

        if len(rows) < BINANCE_KLINE_LIMIT:
            break   # reached the end

        time.sleep(BINANCE_REQUEST_DELAY)

    if not all_rows:
        logger.warning("[downloader] Binance returned no klines for %s.", binance_sym)
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    return _parse_binance_klines(all_rows)


def _parse_binance_klines(rows: List[list]) -> pd.DataFrame:
    """
    Convert raw Binance kline rows to a clean OHLCV DataFrame.

    Binance kline format (indices):
    0  open time (ms)
    1  open
    2  high
    3  low
    4  close
    5  volume
    6  close time (ms)
    7  quote asset volume
    8  number of trades
    9  taker buy base asset volume
    10 taker buy quote asset volume
    11 ignore
    """
    records = []
    for r in rows:
        try:
            open_ms = int(r[0])
            dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
            volume = float(r[5])
            quote_vol = float(r[7])
            # VWAP proxy: quote_volume / base_volume (i.e. average price)
            vwap = quote_vol / volume if volume > 0 else float(r[4])
            records.append({
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": volume,
                "vwap": vwap,
                "timestamp": open_ms / 1000.0,
                "datetime": dt,
            })
        except (IndexError, ValueError, ZeroDivisionError):
            continue

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.set_index("datetime").sort_index()
    df.index.name = "datetime"
    return df[OHLCV_COLUMNS]


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class HistoricalDownloader:
    """
    Downloads and caches historical OHLCV data for all assets in the universe
    and optionally seeds a :class:`~trading_system.data.store.MarketDataStore`.

    Parameters
    ----------
    cache_dir :
        Directory where Parquet cache files are stored.  Defaults to
        ``DATA_CACHE_DIR`` env var or ``./data/cache/``.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = (
            cache_dir
            or os.environ.get("DATA_CACHE_DIR", "./data/cache/")
        )

    # ------------------------------------------------------------------
    # Per-asset methods
    # ------------------------------------------------------------------

    def download_equity_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download equity history, with cache read/write."""
        cached = load_from_cache(ticker, interval, self.cache_dir)
        if cached is not None:
            return cached

        df = download_equity_history(ticker, period=period, interval=interval)
        if not df.empty:
            save_to_cache(df, ticker, interval, self.cache_dir)
        return df

    def download_crypto_history(
        self,
        symbol: str,
        days: int = 730,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download crypto history from Binance, with cache read/write."""
        cached = load_from_cache(symbol, interval, self.cache_dir)
        if cached is not None:
            return cached

        df = download_crypto_history(symbol, days=days, interval=interval)
        if not df.empty:
            save_to_cache(df, symbol, interval, self.cache_dir)
        return df

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap_all(
        self,
        config,
        store,
        equity_period: str = "2y",
        equity_interval: str = "1d",
        crypto_days: int = 730,
        crypto_interval: str = "1d",
    ) -> None:
        """
        Download history for every asset in the configured universe and seed
        the in-memory :class:`~trading_system.data.store.MarketDataStore`.

        Parameters
        ----------
        config :
            A :class:`~trading_system.core.config.TradingConfig` instance.
        store :
            A :class:`~trading_system.data.store.MarketDataStore` instance.
        """
        loop = asyncio.get_event_loop()

        equity_universe: List[str] = getattr(config, "EQUITY_UNIVERSE", [])
        crypto_universe: List[str] = getattr(config, "CRYPTO_UNIVERSE", [])

        logger.info(
            "[downloader] Bootstrapping %d equity + %d crypto assets…",
            len(equity_universe),
            len(crypto_universe),
        )

        # ------------------------------------------------------------------
        # Equities
        # ------------------------------------------------------------------
        for ticker in equity_universe:
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda t=ticker: self.download_equity_history(t, equity_period, equity_interval),
                )
                await self._seed_store(store, ticker, equity_interval, df)
            except Exception as exc:
                logger.error("[downloader] Equity bootstrap failed for %s: %s", ticker, exc)

        # ------------------------------------------------------------------
        # Crypto
        # ------------------------------------------------------------------
        for symbol in crypto_universe:
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda s=symbol: self.download_crypto_history(s, crypto_days, crypto_interval),
                )
                await self._seed_store(store, symbol, crypto_interval, df)
            except Exception as exc:
                logger.error("[downloader] Crypto bootstrap failed for %s: %s", symbol, exc)

        logger.info("[downloader] Bootstrap complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _seed_store(
        store,
        asset: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> None:
        """Write DataFrame rows into the MarketDataStore candle buffer."""
        if df is None or df.empty:
            logger.warning("[downloader] No data to seed for %s/%s.", asset, timeframe)
            return

        rows_written = 0
        for row in df.itertuples():
            ohlcv = {
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "vwap": float(row.vwap),
                "timestamp": float(row.timestamp),
            }
            await store.update_candle(asset, timeframe, ohlcv)
            rows_written += 1

        logger.info(
            "[downloader] Seeded %d candles into store for %s/%s.",
            rows_written, asset, timeframe,
        )
