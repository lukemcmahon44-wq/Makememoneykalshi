"""
trading_system.data — market data ingestion and storage layer.

Public API
----------
MarketDataStore, DataStaleError        — in-memory store  (store.py)
AlpacaFeed                             — Alpaca WS feed    (feed_alpaca.py)
BinanceFeed                            — Binance WS feed   (feed_binance.py)
MacroFeed, MacroDataStore              — FRED macro feed   (feed_macro.py)
OnChainFeed, OnChainDataStore          — CoinMetrics feed  (feed_onchain.py)
SentimentFeed, SentimentDataStore      — sentiment feed    (feed_sentiment.py)
HistoricalDownloader                   — history bootstrap (downloader.py)
"""

from trading_system.data.store import DataStaleError, MarketDataStore
from trading_system.data.feed_alpaca import AlpacaFeed
from trading_system.data.feed_binance import BinanceFeed
from trading_system.data.feed_macro import MacroFeed, MacroDataStore
from trading_system.data.feed_onchain import OnChainFeed, OnChainDataStore
from trading_system.data.feed_sentiment import SentimentFeed, SentimentDataStore
from trading_system.data.downloader import HistoricalDownloader

__all__ = [
    "DataStaleError",
    "MarketDataStore",
    "AlpacaFeed",
    "BinanceFeed",
    "MacroFeed",
    "MacroDataStore",
    "OnChainFeed",
    "OnChainDataStore",
    "SentimentFeed",
    "SentimentDataStore",
    "HistoricalDownloader",
]
