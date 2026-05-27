"""
Master async orchestrator. Wires every subsystem and runs forever.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .alerts import init_alerter, send_telegram_alert
from .core.clock import MarketClock
from .core.config import TradingConfig, load_config
from .core.kill_switch import KillSwitch
from .core.logger import configure_logger, get_logger
from .data.downloader import bootstrap_history
from .data.feed_alpaca import AlpacaFeed
from .data.feed_binance import BinanceFeed
from .data.feed_macro import FREDFeed
from .data.feed_onchain import OnChainFeed
from .data.feed_sentiment import SentimentFeed
from .data.store import MarketDataStore
from .db.repository import DatabaseRepository
from .execution.broker import AlpacaBroker
from .execution.order_manager import OrderManager
from .execution.slippage import SlippageTracker
from .execution.smart_router import SmartOrderRouter
from .ml.predictor import MLPredictor
from .portfolio.manager import PortfolioManager
from .risk.circuit_breaker import CircuitBreaker
from .risk.sizer import PositionSizer
from .signals._common import atr, closes
from .signals.ensemble import SignalEnsemble
from .signals.microstructure.flow import composite_flow
from .signals.microstructure.hawkes import HawkesCache
from .signals.microstructure.lob_signals import lob_composite
from .signals.onchain.btc_signals import compute_btc_onchain_signal
from .signals.onchain.eth_signals import compute_eth_onchain_signal
from .signals.regime.garch import GARCHRegime
from .signals.regime.macro_overlay import (compute_macro_multiplier,
                                              equity_signal_gate)
from .signals.sentiment.news_nlp import FinBERTScorer, compute_news_sentiment
from .signals.sentiment.reddit_nlp import reddit_signal_for_asset
from .signals.technical.reversion import composite_reversion
from .signals.technical.trend import adx_regime, composite_trend


log = get_logger("main")


class TradingSystem:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.store = MarketDataStore(max_staleness_secs=config.MAX_DATA_STALENESS_SECS)
        self.db = DatabaseRepository(config)
        self.clock = MarketClock(config)
        self.kill_switch = KillSwitch(config)
        self.broker = AlpacaBroker(config)
        self.slippage = SlippageTracker(self.db)
        self.order_mgr = OrderManager(self.broker, self.db, self.slippage,
                                        cooldown_secs=config.COOLDOWN_MINUTES * 60)
        self.router = SmartOrderRouter()
        self.sizer = PositionSizer(config)
        self.circuits = CircuitBreaker(config, self.db)
        self.portfolio = PortfolioManager(self.db, starting_cash=10_000.0)
        self.ensemble = SignalEnsemble()
        self.predictor = MLPredictor(config.MODEL_DIR)
        self.hawkes_cache = HawkesCache()
        self.finbert: FinBERTScorer | None = None
        self.fred_feed = FREDFeed(config.FRED_API_KEY)
        self.onchain_feed = OnChainFeed()
        self.sentiment_feed = SentimentFeed(
            config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET,
            config.REDDIT_USER_AGENT, config.ALPHA_VANTAGE_KEY,
            config.all_assets,
        )
        self.alpaca_equity_feed = AlpacaFeed(
            self.store, config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            config.EQUITY_UNIVERSE, stream="equity")
        self.alpaca_crypto_feed = AlpacaFeed(
            self.store, config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            config.CRYPTO_UNIVERSE, stream="crypto")
        self.binance_feed = BinanceFeed(self.store, config.BINANCE_UNIVERSE)
        self._shutdown = False
        self.trade_history: List[Dict[str, Any]] = []
        self.garch_models: Dict[str, GARCHRegime] = {}

    # ── startup ──────────────────────────────────────────────────────────
    async def startup(self) -> None:
        log.info("=== TRADING SYSTEM STARTUP ===")
        log.info(f"PAPER_MODE={self.config.PAPER_MODE} "
                  f"EQUITIES={self.config.EQUITY_UNIVERSE} "
                  f"CRYPTO={self.config.CRYPTO_UNIVERSE}")
        init_alerter(self.config)
        # Broker check
        ok = await self.broker.test_connection()
        log.info(f"Broker connection: {'OK' if ok else 'FAILED'}")
        portfolio = await self.broker.get_portfolio_state()
        self.portfolio.update_from_broker(portfolio)
        log.info(f"Initial portfolio value ~ ${portfolio.value:,.2f}")
        # Bootstrap history
        try:
            await bootstrap_history(self.store, self.config.all_assets,
                                      timeframes=("1m", "5m", "15m", "1h", "1d"),
                                      bars_required=250)
        except Exception as e:
            log.error(f"Bootstrap history failed: {e}")
        # Load ML model if accepted
        self.predictor.load()
        if self.predictor.is_ready():
            self.ensemble.enable_ml()
        else:
            self.ensemble.disable_ml()
        # FinBERT (best-effort)
        try:
            self.finbert = FinBERTScorer()
            self.finbert.load()
        except Exception as e:
            log.warning(f"FinBERT load failed: {e}")
        log.info("=== SYSTEM ONLINE ===")
        await send_telegram_alert(
            f"Trading system online (paper={self.config.PAPER_MODE})")

    # ── signal generation loop ───────────────────────────────────────────
    async def signal_loop(self) -> None:
        log.info("Signal loop active")
        while not self._shutdown:
            try:
                await self._generate_and_act()
            except Exception as e:
                log.error(f"Signal loop error: {e}", exc_info=True)
            await asyncio.sleep(15)

    async def _generate_and_act(self) -> None:
        if self.kill_switch.is_engaged():
            return
        if not self.circuits.trading_allowed():
            return
        fred = self.fred_feed.snapshot()
        macro_mult, macro_flags = compute_macro_multiplier(fred)
        equity_allowed = equity_signal_gate(fred)

        # Mark NAV
        mark_prices: Dict[str, float] = {}
        for asset in self.config.all_assets:
            c = self.store.latest_close(asset, "1m")
            if c:
                mark_prices[asset] = c
                self.broker.update_sim_last_price(asset, c)
        nav = self.portfolio.update_nav(mark_prices)

        # Drawdown circuit
        dd_level, dd_val = self.circuits.check_portfolio_drawdown(
            nav["total"], nav["peak_value"])
        if dd_level is not None:
            log.warning(f"Drawdown breach {dd_level} {dd_val:.1%}")
        self.circuits.check_daily_loss(nav["daily_pnl_pct"])

        # Per-asset signal computation
        for asset in self.config.all_assets:
            if self.circuits.asset_is_suspended(asset):
                continue
            asset_type = self.config.asset_type(asset)
            if asset_type == "equity" and not equity_allowed:
                continue
            try:
                await self._evaluate_asset(asset, asset_type, macro_mult,
                                             mark_prices, fred)
            except Exception as e:
                log.error(f"{asset} eval error: {e}", exc_info=True)

    async def _evaluate_asset(self, asset: str, asset_type: str,
                                macro_mult: float, mark_prices: Dict[str, float],
                                fred: Dict[str, Any]) -> None:
        candles_1m = self.store.get_candles(asset, "1m", n=200, allow_stale=True)
        if len(candles_1m) < 60:
            return
        c1m = closes(candles_1m)
        atr_val = atr(candles_1m, 14)

        # Volatility / GARCH-based size scalar
        garch = self.garch_models.setdefault(asset, GARCHRegime())
        import numpy as np
        rets_pct = np.diff(c1m) / c1m[:-1]
        size_scalar = garch.position_size_scalar(
            self.config.TARGET_ANNUAL_VOL, rets_pct)

        # Technical signals
        s_trend = composite_trend(candles_1m)
        s_rev = composite_reversion(candles_1m, session_vwap=self.store.get_vwap(asset))
        adx_info = adx_regime(candles_1m)

        # Microstructure
        try:
            lob = self.store.get_lob(asset, allow_stale=True)
            s_micro = lob_composite(lob)
        except Exception:
            s_micro = 0.0
        try:
            trades = self.store.get_recent_trades(asset, n=500)
            s_flow = composite_flow(trades)
            if trades:
                ts_list = [t.ts for t in trades]
                s_hawkes = self.hawkes_cache.update_and_signal(asset, ts_list, time.time())
            else:
                s_hawkes = 0.0
        except Exception:
            s_flow = 0.0; s_hawkes = 0.0
        s_micro_full = float(np.clip(0.4 * s_micro + 0.3 * s_flow + 0.3 * s_hawkes,
                                       -1.0, 1.0))

        # Sentiment
        sentiment_snapshot = self.sentiment_feed.snapshot()
        posts = sentiment_snapshot["reddit_posts"].get(asset.replace("USD", "")
                                                         .replace("USDT", ""), [])
        s_reddit = reddit_signal_for_asset(posts)
        news_items = sentiment_snapshot["news_items"].get(asset, [])
        try:
            s_news = compute_news_sentiment(news_items, asset, self.finbert)
        except Exception:
            s_news = 0.0
        s_sentiment = float(np.clip(0.5 * s_reddit + 0.5 * s_news, -1.0, 1.0))

        # On-chain (crypto only)
        s_onchain = 0.0
        if asset_type == "crypto" and "BTC" in asset:
            df = self.onchain_feed.snapshot().get("BTC")
            if df is not None:
                s_onchain, _ = compute_btc_onchain_signal(df)
        elif asset_type == "crypto" and "ETH" in asset:
            df = self.onchain_feed.snapshot().get("ETH")
            if df is not None:
                s_onchain, _ = compute_eth_onchain_signal(df)

        # ML signal (gated off until OOS gating passes)
        s_ml = 0.0
        if self.predictor.is_ready() and self.ensemble.ml_enabled:
            try:
                import pandas as pd
                features_row = pd.DataFrame()   # populated from features.py in live build-out
                if not features_row.empty:
                    res = self.predictor.signal(features_row)
                    s_ml = float(res.get("signal", 0))
            except Exception:
                s_ml = 0.0

        signals = {
            "technical_trend": s_trend,
            "technical_reversion": s_rev,
            "microstructure": s_micro_full,
            "stat_arb": 0.0,   # populated by pair runners (not implemented here)
            "sentiment": s_sentiment,
            "onchain": s_onchain,
            "ml_prediction": s_ml,
        }
        regime_data = {
            "adx_regime": adx_info["regime"],
            "garch_size_scalar": size_scalar,
            "hmm_multiplier": 1.0,
            "macro_multiplier": macro_mult,
        }
        composite, meta = self.ensemble.compute_composite(
            signals, regime_data, asset_type=asset_type)
        decision, conviction = self.ensemble.threshold_check(
            composite, entry=self.config.SIGNAL_ENTRY_THRESHOLD,
            high_conviction=self.config.HIGH_CONVICTION_THRESHOLD)

        # Persist signals for IC tracking
        for cat, val in signals.items():
            try:
                self.db.write_signal(asset, cat, float(val), composite, decision)
            except Exception:
                pass

        # Required: 3 agreeing signals minimum
        agreeing = sum(1 for v in signals.values() if np.sign(v) == np.sign(composite) and abs(v) > 0.2)
        if decision != "ENTER" or agreeing < self.config.MIN_AGREEING_SIGNALS:
            await self._check_exit(asset, composite, mark_prices.get(asset))
            return

        if not self.order_mgr.can_enter(asset):
            return
        if asset in self.portfolio.positions:
            return

        current_price = mark_prices.get(asset, candles_1m[-1].close)
        if current_price <= 0:
            return
        ann_vol = float(np.std(rets_pct, ddof=0) * np.sqrt(60 * 24 * 252)) if len(rets_pct) > 30 else 0.5
        dollar_size, pct, breakdown = self.sizer.compute_final_size(
            asset, composite, self.portfolio.total_value(mark_prices),
            self.trade_history, self.portfolio.get_positions_snapshot(),
            {}, ann_vol, macro_mult, size_scalar,
        )
        if dollar_size < self.config.MIN_ORDER_USD:
            return

        adv_usd = max(sum(c.close * c.volume for c in candles_1m[-390:]), 1.0)
        plan = self.router.route(asset, dollar_size,
                                   "buy" if composite > 0 else "sell",
                                   self.portfolio.total_value(mark_prices),
                                   self.store.get_lob(asset, allow_stale=True),
                                   adv_usd)
        log.info(f"ENTER {asset} composite={composite:.2f} "
                  f"conviction={conviction} size=${dollar_size:.0f} plan={plan['type']}")
        res = await self.order_mgr.execute_plan(plan, current_price)
        if res.success and res.fill_qty > 0:
            direction = "long" if composite > 0 else "short"
            trade_id = self.db.open_trade(
                asset, "buy" if direction == "long" else "sell", direction,
                res.fill_qty, res.fill_price,
                signal_strength=abs(composite), composite_signal=composite,
                notes=f"{conviction} {breakdown}",
            )
            self.portfolio.open_position(
                asset, direction, res.fill_qty, res.fill_price, atr_val,
                composite, pct, dollar_size, trade_id,
            )
            await send_telegram_alert(
                f"ENTER {asset} {direction} qty={res.fill_qty:.4f} "
                f"price={res.fill_price:.4f} signal={composite:.2f}")

    async def _check_exit(self, asset: str, composite: float,
                           current_price: float | None) -> None:
        pos = self.portfolio.positions.get(asset)
        if pos is None or current_price is None or current_price <= 0:
            return
        import numpy as np
        # Stop-loss check
        hit_stop, _ = self.circuits.check_position_loss(
            asset, {"direction": pos.direction, "entry_price": pos.entry_price,
                     "entry_atr": pos.entry_atr}, current_price)
        flipped = np.sign(composite) != (1 if pos.direction == "long" else -1)
        weak = abs(composite) < 0.30
        if hit_stop or flipped or weak:
            res = await self.broker.close_position(asset)
            if res.success:
                outcome = self.portfolio.close_position(asset, res.fill_price)
                if outcome:
                    self.trade_history.append(outcome)
                    self.ensemble.record_observation(
                        {"technical_trend": composite,
                          "technical_reversion": 0,
                          "microstructure": 0,
                          "stat_arb": 0,
                          "sentiment": 0,
                          "onchain": 0,
                          "ml_prediction": 0},
                        outcome["pnl_pct"])
                log.info(f"EXIT {asset} fill={res.fill_price:.4f} reason="
                          f"{'STOP' if hit_stop else 'FLIP' if flipped else 'WEAK'}")

    # ── auxiliary loops ──────────────────────────────────────────────────
    async def rebalance_loop(self) -> None:
        while not self._shutdown:
            try:
                self.ensemble.update_weights()
            except Exception as e:
                log.error(f"weight update error: {e}")
            await asyncio.sleep(3600)

    async def analytics_loop(self) -> None:
        while not self._shutdown:
            try:
                mark_prices = {a: self.store.latest_close(a, "1m") or 0
                                for a in self.config.all_assets}
                mark_prices = {a: p for a, p in mark_prices.items() if p}
                nav = self.portfolio.update_nav(mark_prices)
                log.info(f"NAV ${nav['total']:.2f} daily {nav['daily_pnl_pct']:.2%} "
                          f"dd {nav['drawdown']:.2%} positions={len(self.portfolio.positions)}")
            except Exception as e:
                log.error(f"analytics loop error: {e}")
            await asyncio.sleep(300)

    async def health_check_loop(self) -> None:
        while not self._shutdown:
            try:
                stale = self.circuits.check_data_staleness(
                    self.store, max_age_seconds=self.config.MAX_DATA_STALENESS_SECS)
                if stale:
                    await send_telegram_alert(f"Stale data: {stale}")
                # Broker ping
                if not await self.broker.ping():
                    await send_telegram_alert("BROKER PING FAILED")
            except Exception as e:
                log.error(f"health loop error: {e}")
            await asyncio.sleep(30)

    async def graceful_shutdown(self) -> None:
        log.info("Beginning graceful shutdown")
        self._shutdown = True
        try:
            await self.broker.cancel_all_orders()
            if self.config.CLOSE_ON_SHUTDOWN:
                for asset in list(self.portfolio.positions):
                    await self.broker.close_position(asset)
            self.db.write_system_state({
                "event": "shutdown",
                "shutdown_time": datetime.now(timezone.utc).isoformat(),
                "n_positions": len(self.portfolio.positions),
            })
        finally:
            self.db.close()
        log.info("Shutdown complete")

    async def run(self) -> None:
        await self.startup()
        tasks = [
            asyncio.create_task(self.alpaca_equity_feed.run(), name="feed_alpaca_eq"),
            asyncio.create_task(self.alpaca_crypto_feed.run(), name="feed_alpaca_cr"),
            asyncio.create_task(self.binance_feed.run(), name="feed_binance"),
            asyncio.create_task(self.fred_feed.run(), name="feed_fred"),
            asyncio.create_task(self.onchain_feed.run(), name="feed_onchain"),
            asyncio.create_task(self.sentiment_feed.run(), name="feed_sentiment"),
            asyncio.create_task(self.signal_loop(), name="signal_loop"),
            asyncio.create_task(self.rebalance_loop(), name="rebalance"),
            asyncio.create_task(self.analytics_loop(), name="analytics"),
            asyncio.create_task(self.health_check_loop(), name="health"),
            asyncio.create_task(self.kill_switch.monitor(), name="kill_switch"),
        ]
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutdown requested")
        except Exception as e:
            log.critical(f"Main loop fatal: {e}", exc_info=True)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await self.graceful_shutdown()


def main_entry() -> None:
    configure_logger()
    config = load_config()
    system = TradingSystem(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Signal handlers
    try:
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(
            system.graceful_shutdown()))
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(
            system.graceful_shutdown()))
    except NotImplementedError:
        pass
    try:
        loop.run_until_complete(system.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main_entry()
