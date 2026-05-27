"""
Portfolio manager. Tracks state, computes daily PnL, peak value, drawdown.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from ..core.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrackedPosition:
    asset: str
    direction: str         # 'long' or 'short'
    qty: float
    entry_price: float
    entry_atr: float
    entry_signal: float
    entry_time: float
    size_pct: float
    size_usd: float
    trade_id: Optional[int] = None

    def mark_to_market(self, current_price: float) -> float:
        if self.direction == "long":
            return self.qty * (current_price - self.entry_price)
        return self.qty * (self.entry_price - current_price)


class PortfolioManager:
    def __init__(self, db, starting_cash: float = 10_000.0):
        self.db = db
        self.cash = starting_cash
        self.positions: Dict[str, TrackedPosition] = {}
        self.peak_value = starting_cash
        self.start_of_day_value = starting_cash
        self.nav_history: Deque[float] = deque(maxlen=10_000)
        self._last_day_marker = int(time.time() // 86400)

    def update_from_broker(self, broker_portfolio) -> None:
        self.cash = float(broker_portfolio.cash)
        # We don't blindly overwrite local TrackedPositions; broker positions
        # are reconciled but local entry metadata (signal, atr) is preserved.
        for sym, pos in broker_portfolio.positions.items():
            if sym not in self.positions:
                self.positions[sym] = TrackedPosition(
                    asset=sym, direction=pos.side, qty=float(pos.qty),
                    entry_price=float(pos.avg_entry_price), entry_atr=0.0,
                    entry_signal=0.0, entry_time=time.time(),
                    size_pct=0.0, size_usd=float(pos.qty * pos.avg_entry_price),
                )

    def total_value(self, mark_prices: Dict[str, float]) -> float:
        value = self.cash
        for asset, pos in self.positions.items():
            price = mark_prices.get(asset, pos.entry_price)
            if pos.direction == "long":
                value += pos.qty * price
            else:
                value += pos.qty * (2 * pos.entry_price - price)
        return float(value)

    def update_nav(self, mark_prices: Dict[str, float]) -> Dict[str, float]:
        total = self.total_value(mark_prices)
        self.peak_value = max(self.peak_value, total)

        # Detect day rollover
        cur_day = int(time.time() // 86400)
        if cur_day != self._last_day_marker:
            self.start_of_day_value = total
            self._last_day_marker = cur_day

        daily_pnl = (total - self.start_of_day_value)
        daily_pnl_pct = daily_pnl / max(self.start_of_day_value, 1e-8)
        drawdown = (self.peak_value - total) / max(self.peak_value, 1e-8)
        self.nav_history.append(total)

        try:
            self.db.write_nav(
                int(time.time()), total, self.cash,
                total - self.cash, len(self.positions),
                daily_pnl, drawdown, self.peak_value,
            )
        except Exception as e:
            log.error(f"NAV write error: {e}")
        return {"total": total, "daily_pnl": daily_pnl,
                "daily_pnl_pct": daily_pnl_pct, "drawdown": drawdown,
                "peak_value": self.peak_value}

    def open_position(self, asset: str, direction: str, qty: float,
                       entry_price: float, entry_atr: float,
                       entry_signal: float, size_pct: float, size_usd: float,
                       trade_id: Optional[int] = None) -> None:
        self.positions[asset] = TrackedPosition(
            asset=asset, direction=direction, qty=qty,
            entry_price=entry_price, entry_atr=entry_atr,
            entry_signal=entry_signal, entry_time=time.time(),
            size_pct=size_pct, size_usd=size_usd, trade_id=trade_id,
        )

    def close_position(self, asset: str, exit_price: float
                        ) -> Optional[Dict[str, float]]:
        pos = self.positions.pop(asset, None)
        if pos is None:
            return None
        pnl = pos.mark_to_market(exit_price)
        pnl_pct = pnl / max(pos.size_usd, 1e-8)
        if pos.trade_id is not None:
            try:
                self.db.close_trade(pos.trade_id, exit_price, pnl, pnl_pct)
            except Exception:
                pass
        return {"pnl": pnl, "pnl_pct": pnl_pct, "holding_secs": time.time() - pos.entry_time}

    def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            asset: {
                "direction": p.direction,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "size_pct": p.size_pct,
                "size_usd": p.size_usd,
                "entry_signal": p.entry_signal,
                "entry_time": p.entry_time,
            }
            for asset, p in self.positions.items()
        }
