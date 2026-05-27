"""
Circuit breakers - per-asset, daily, drawdown levels, system health.

Knight Capital lost $440M because they had no kill switches. We have many.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.logger import get_logger

log = get_logger(__name__)


class CircuitBreaker:
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.active_halts: set[str] = set()
        self.suspended_assets: Dict[str, float] = {}    # asset -> resume_ts
        self.requires_manual_reset: bool = False
        self.last_event_log: List[Dict[str, Any]] = []

    # ── LEVEL 1: ASSET-LEVEL ────────────────────────────────────────────
    def check_flash_crash(self, asset: str,
                          price_history_5m: List[float]) -> bool:
        if len(price_history_5m) < 5:
            return False
        change = (price_history_5m[-1] / price_history_5m[0]) - 1
        if change < -0.07:
            self.suspend_asset(asset, duration_hours=2,
                                reason=f"FLASH_CRASH_{change:.1%}")
            return True
        return False

    def check_position_loss(self, asset: str, position: Dict[str, Any],
                             current_price: float) -> Tuple[bool, float]:
        entry_price = float(position["entry_price"])
        atr_val = float(position.get("entry_atr", 0.01 * entry_price))
        if position.get("direction") == "long":
            stop_price = entry_price - 2.5 * atr_val
            hit = current_price < stop_price
        else:
            stop_price = entry_price + 2.5 * atr_val
            hit = current_price > stop_price
        if hit:
            self._log("STOP_LOSS", asset,
                       f"hit stop {stop_price:.4f} curr {current_price:.4f}",
                       severity="WARNING")
        return hit, stop_price

    # ── LEVEL 2: DAILY ───────────────────────────────────────────────────
    def check_daily_loss(self, daily_pnl_pct: float) -> bool:
        if daily_pnl_pct < -self.config.DAILY_LOSS_LIMIT:
            if "DAILY_LOSS" not in self.active_halts:
                self.active_halts.add("DAILY_LOSS")
                self._log("CIRCUIT_L2", "PORTFOLIO",
                           f"daily loss {daily_pnl_pct:.1%} > limit",
                           severity="HIGH")
            return True
        if daily_pnl_pct > -0.01:
            self.active_halts.discard("DAILY_LOSS")
        return False

    # ── LEVEL 3: DRAWDOWN ────────────────────────────────────────────────
    def check_portfolio_drawdown(self, portfolio_value: float, peak_value: float
                                  ) -> Tuple[Optional[str], float]:
        if peak_value <= 0:
            return None, 0.0
        dd = (peak_value - portfolio_value) / peak_value
        if dd >= self.config.DRAWDOWN_L3:
            self._trigger_halt("DRAWDOWN_L3", "CRITICAL", close_all=True,
                                halt_days=999, manual_reset=True)
            return "CRITICAL", dd
        if dd >= self.config.DRAWDOWN_L2:
            self._trigger_halt("DRAWDOWN_L2", "SEVERE", close_all=True,
                                halt_days=7, manual_reset=True)
            return "SEVERE", dd
        if dd >= self.config.DRAWDOWN_L1:
            self._trigger_halt("DRAWDOWN_L1", "MODERATE", close_all=True,
                                halt_hours=24, manual_reset=False)
            return "MODERATE", dd
        return None, dd

    # ── LEVEL 4: SYSTEM HEALTH ───────────────────────────────────────────
    def check_data_staleness(self, data_store, max_age_seconds: int = 45
                              ) -> List[str]:
        stale: List[str] = []
        for asset in list(data_store.assets):
            if data_store.asset_age_secs(asset) > max_age_seconds:
                stale.append(asset)
                self.suspend_asset(asset, duration_hours=0.5,
                                     reason="STALE_DATA")
        return stale

    def check_order_anomaly(self, recent_orders: List[Dict[str, Any]],
                             normal_daily_volume: float) -> bool:
        now = time.time()
        orders_last_60s = sum(1 for o in recent_orders
                                 if now - o.get("timestamp", 0) < 60)
        normal_per_60s = max(normal_daily_volume / 390, 1.0)
        if orders_last_60s > 10 * normal_per_60s:
            self._trigger_halt("ORDER_STORM", "CRITICAL", close_all=False,
                                halt_hours=0.5, manual_reset=True)
            self._log("ORDER_STORM", "SYSTEM",
                       f"{orders_last_60s} orders/60s (normal {normal_per_60s:.1f})",
                       severity="CRITICAL")
            return True
        return False

    # ── HELPERS ──────────────────────────────────────────────────────────
    def suspend_asset(self, asset: str, duration_hours: float, reason: str
                      ) -> None:
        resume_at = time.time() + duration_hours * 3600
        self.suspended_assets[asset] = resume_at
        self._log("ASSET_SUSPENDED", asset,
                   f"{reason} until {datetime.fromtimestamp(resume_at, tz=timezone.utc).isoformat()}",
                   severity="WARNING")

    def asset_is_suspended(self, asset: str) -> bool:
        resume = self.suspended_assets.get(asset)
        if resume is None:
            return False
        if time.time() >= resume:
            del self.suspended_assets[asset]
            return False
        return True

    def trading_allowed(self) -> bool:
        return not self.requires_manual_reset and not self.active_halts

    def _trigger_halt(self, code: str, severity: str, close_all: bool,
                       halt_hours: Optional[float] = None,
                       halt_days: Optional[float] = None,
                       manual_reset: bool = False) -> None:
        self.active_halts.add(code)
        if manual_reset:
            self.requires_manual_reset = True
        self._log("CIRCUIT_HALT", "PORTFOLIO",
                   f"{code} severity={severity} close_all={close_all} "
                   f"halt_hours={halt_hours} halt_days={halt_days} "
                   f"manual_reset={manual_reset}", severity=severity)

    def manual_reset(self, reason: str = "operator override") -> None:
        self.active_halts.clear()
        self.suspended_assets.clear()
        self.requires_manual_reset = False
        self._log("MANUAL_RESET", "SYSTEM", reason, severity="INFO")

    def _log(self, event_type: str, asset: str, message: str,
              severity: str = "INFO") -> None:
        payload = {
            "timestamp": int(time.time()),
            "event_type": event_type,
            "asset": asset,
            "severity": severity,
            "message": message,
            "portfolio_value": self.db.get_portfolio_value(),
        }
        self.last_event_log.append(payload)
        self.last_event_log = self.last_event_log[-50:]
        try:
            self.db.write_risk_event(payload)
        except Exception as e:
            log.error(f"Failed to persist risk event: {e}")
        log_method = log.error if severity in ("CRITICAL", "SEVERE", "HIGH") else log.info
        log_method(f"[{event_type}] {asset}: {message}")
