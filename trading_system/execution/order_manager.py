"""
Order lifecycle manager. Submits, tracks, logs to DB.

Implements adaptive limit chase: place limit, wait, re-price if unfilled,
fall back to market on max chases.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger
from .broker import AlpacaBroker, OrderResult
from .slippage import SlippageTracker

log = get_logger(__name__)


class OrderManager:
    def __init__(self, broker: AlpacaBroker, db, slippage: SlippageTracker,
                  cooldown_secs: int = 900):
        self.broker = broker
        self.db = db
        self.slippage = slippage
        self.cooldown_secs = cooldown_secs
        self.last_entry_time: Dict[str, float] = {}
        self.recent_orders: List[Dict[str, Any]] = []

    def can_enter(self, asset: str) -> bool:
        last = self.last_entry_time.get(asset, 0)
        return time.time() - last >= self.cooldown_secs

    def _track(self, asset: str, side: str, qty: float, order_type: str,
                limit_price: Optional[float]) -> int:
        try:
            order_id = self.db.log_order({
                "asset": asset, "side": side, "qty": qty,
                "order_type": order_type, "limit_price": limit_price,
                "submitted_at": int(time.time()), "status": "SUBMITTED",
            })
        except Exception as e:
            log.error(f"DB order log error: {e}")
            order_id = -1
        self.recent_orders.append({"asset": asset, "side": side,
                                    "timestamp": time.time()})
        self.recent_orders = self.recent_orders[-500:]
        return order_id

    async def execute_plan(self, plan: Dict[str, Any],
                            reference_price: float) -> OrderResult:
        asset = plan["asset"]; side = plan["side"]
        qty_usd = float(plan["qty_usd"])
        ptype = plan["type"]
        if reference_price <= 0:
            return OrderResult(False, "", 0, 0, "REJECTED",
                                 "no reference price")
        if ptype == "market":
            qty = qty_usd / reference_price
            order_id = self._track(asset, side, qty, "market", None)
            res = await self.broker.submit_order(
                asset, side, qty, "market", reference_price=reference_price)
            self._record_fill(order_id, res, reference_price, asset, side, "market", qty)
            return res
        if ptype == "adaptive_limit":
            return await self._adaptive_limit(plan, reference_price)
        if ptype == "twap":
            return await self._twap(plan, reference_price)
        return OrderResult(False, "", 0, 0, "REJECTED",
                             f"unknown plan {ptype}")

    async def _adaptive_limit(self, plan: Dict[str, Any],
                               reference_price: float) -> OrderResult:
        asset = plan["asset"]; side = plan["side"]
        qty_usd = float(plan["qty_usd"])
        limit_price = float(plan["limit_price"])
        chase_increment = float(plan.get("chase_increment", limit_price * 0.001))
        max_chases = int(plan.get("max_chases", 3))
        chase_interval = int(plan.get("chase_interval", 30))

        qty = qty_usd / max(limit_price, 1e-8)
        order_id = self._track(asset, side, qty, "limit", limit_price)
        for attempt in range(max_chases):
            res = await self.broker.submit_order(
                asset, side, qty, "limit", limit_price=limit_price,
                reference_price=reference_price)
            if res.success and res.status == "FILLED":
                self._record_fill(order_id, res, limit_price, asset, side,
                                    "limit", qty)
                return res
            await asyncio.sleep(chase_interval)
            limit_price += chase_increment if side == "buy" else -chase_increment

        # Fallback to market
        log.info(f"{asset}: adaptive limit gave up, falling back to market")
        res = await self.broker.submit_order(
            asset, side, qty, "market", reference_price=reference_price)
        self._record_fill(order_id, res, reference_price, asset, side, "market", qty)
        return res

    async def _twap(self, plan: Dict[str, Any], reference_price: float
                     ) -> OrderResult:
        asset = plan["asset"]; side = plan["side"]
        schedule = plan.get("schedule", [])
        total_filled = 0.0; total_qty = 0.0; last_status = "PENDING"
        last_order_id = ""
        for s in schedule:
            await asyncio.sleep(max(0, s["delay_secs"]))
            sub_plan = {"type": "adaptive_limit", "asset": asset, "side": side,
                         "qty_usd": s["slice_usd"],
                         "limit_price": reference_price}
            res = await self._adaptive_limit(sub_plan, reference_price)
            if res.success:
                total_filled += res.fill_price * res.fill_qty
                total_qty += res.fill_qty
                last_status = res.status
                last_order_id = res.broker_order_id
        avg = total_filled / total_qty if total_qty > 0 else 0
        return OrderResult(success=total_qty > 0,
                             broker_order_id=last_order_id,
                             fill_price=avg, fill_qty=total_qty,
                             status=last_status)

    def _record_fill(self, order_id: int, res: OrderResult,
                      expected: float, asset: str, side: str, otype: str,
                      qty: float) -> None:
        if not res.success or res.fill_price <= 0:
            return
        try:
            self.db.update_order_fill(order_id, res.fill_price, res.fill_qty,
                                        res.status)
        except Exception:
            pass
        self.slippage.log_fill(asset, expected, res.fill_price, side, qty, otype)
        if side == "buy":
            self.last_entry_time[asset] = time.time()
