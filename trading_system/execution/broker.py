"""
Alpaca broker abstraction. Handles equities + crypto with one interface.

Includes paper-mode local simulator that fills at supplied prices so the
rest of the system can run without API keys for testing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.logger import get_logger

log = get_logger(__name__)


@dataclass
class Position:
    asset: str
    qty: float
    avg_entry_price: float
    side: str          # 'long' or 'short'

    @property
    def is_long(self) -> bool:
        return self.side == "long"


@dataclass
class Portfolio:
    cash: float = 10_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    peak_value: float = 10_000.0

    @property
    def value(self) -> float:
        return self.cash + sum(p.qty * p.avg_entry_price for p in self.positions.values())


@dataclass
class OrderResult:
    success: bool
    broker_order_id: str
    fill_price: float
    fill_qty: float
    status: str
    message: str = ""
    order_id_internal: Optional[int] = None


class AlpacaBroker:
    """
    Wrapper that uses alpaca-py if available; otherwise falls back to local
    simulated broker (paper sim). Same async interface either way.
    """

    def __init__(self, config):
        self.config = config
        self.paper_mode = config.PAPER_MODE
        self._client = None
        self._crypto_client = None
        self._sim_portfolio = Portfolio(cash=10_000.0)
        self._sim_last_price: Dict[str, float] = {}
        self._init_clients()

    def _init_clients(self) -> None:
        if not (self.config.ALPACA_API_KEY and self.config.ALPACA_SECRET_KEY):
            log.warning("Alpaca creds missing - using local sim broker")
            return
        try:
            from alpaca.trading.client import TradingClient  # type: ignore
            self._client = TradingClient(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY,
                paper=self.paper_mode,
            )
            log.info(f"Alpaca client initialised (paper={self.paper_mode})")
        except ImportError:
            log.warning("alpaca-py not installed - using local sim broker")
        except Exception as e:
            log.error(f"Alpaca client init failed: {e} - using sim")

    async def test_connection(self) -> bool:
        if self._client is None:
            return True   # sim mode
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._client.get_account)
            return True
        except Exception as e:
            log.error(f"Alpaca connection test failed: {e}")
            return False

    async def ping(self) -> bool:
        return await self.test_connection()

    async def reconnect(self) -> None:
        self._init_clients()

    async def get_portfolio_state(self) -> Portfolio:
        if self._client is None:
            return self._sim_portfolio
        loop = asyncio.get_event_loop()
        try:
            account = await loop.run_in_executor(None, self._client.get_account)
            positions = await loop.run_in_executor(None, self._client.get_all_positions)
            cash = float(account.cash)
            portfolio = Portfolio(cash=cash)
            for p in positions:
                qty = float(p.qty)
                side = "long" if qty > 0 else "short"
                portfolio.positions[p.symbol] = Position(
                    asset=p.symbol,
                    qty=abs(qty),
                    avg_entry_price=float(p.avg_entry_price),
                    side=side,
                )
            portfolio.peak_value = max(self._sim_portfolio.peak_value, portfolio.value)
            self._sim_portfolio.peak_value = portfolio.peak_value
            return portfolio
        except Exception as e:
            log.error(f"get_portfolio_state error: {e}")
            return self._sim_portfolio

    async def submit_order(self, asset: str, side: str, qty: float,
                            order_type: str = "market",
                            limit_price: Optional[float] = None,
                            reference_price: Optional[float] = None
                            ) -> OrderResult:
        if self._client is None:
            return self._sim_submit(asset, side, qty, order_type,
                                     limit_price, reference_price)
        loop = asyncio.get_event_loop()
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
            from alpaca.trading.requests import (LimitOrderRequest,  # type: ignore
                                                   MarketOrderRequest)
            asset_type = self.config.asset_type(asset)
            tif = TimeInForce.GTC if asset_type == "crypto" else TimeInForce.DAY
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            symbol = asset
            if order_type == "limit" and limit_price is not None:
                req = LimitOrderRequest(symbol=symbol, qty=qty, side=order_side,
                                          time_in_force=tif, limit_price=limit_price)
            else:
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=order_side,
                                           time_in_force=tif)
            resp = await loop.run_in_executor(None, self._client.submit_order, req)
            fill_price = float(getattr(resp, "filled_avg_price", limit_price or 0) or 0)
            fill_qty = float(getattr(resp, "filled_qty", 0) or 0)
            return OrderResult(
                success=True,
                broker_order_id=str(resp.id),
                fill_price=fill_price,
                fill_qty=fill_qty,
                status=str(resp.status),
            )
        except Exception as e:
            log.error(f"Alpaca submit_order error: {e}")
            return OrderResult(success=False, broker_order_id="",
                                fill_price=0, fill_qty=0,
                                status="REJECTED", message=str(e))

    def _sim_submit(self, asset: str, side: str, qty: float,
                     order_type: str, limit_price: Optional[float],
                     reference_price: Optional[float]) -> OrderResult:
        ref = reference_price or limit_price or self._sim_last_price.get(asset, 0)
        if not ref:
            return OrderResult(False, "", 0, 0, "REJECTED",
                                "sim: no reference price")
        slippage = self.config.SLIPPAGE_CRYPTO if self.config.asset_type(asset) == "crypto" \
            else self.config.SLIPPAGE_EQUITY
        slip_sign = 1 if side == "buy" else -1
        fill_price = ref * (1 + slip_sign * slippage)
        commission = qty * fill_price * (
            self.config.COMMISSION_CRYPTO if self.config.asset_type(asset) == "crypto"
            else self.config.COMMISSION_EQUITY
        )

        if side == "buy":
            cost = qty * fill_price + commission
            if cost > self._sim_portfolio.cash + 1e-3:
                return OrderResult(False, "", 0, 0, "REJECTED",
                                     "sim: insufficient cash")
            self._sim_portfolio.cash -= cost
            existing = self._sim_portfolio.positions.get(asset)
            if existing:
                total_qty = existing.qty + qty
                existing.avg_entry_price = (
                    (existing.qty * existing.avg_entry_price + qty * fill_price) / total_qty
                )
                existing.qty = total_qty
            else:
                self._sim_portfolio.positions[asset] = Position(
                    asset=asset, qty=qty, avg_entry_price=fill_price, side="long"
                )
        else:    # sell
            existing = self._sim_portfolio.positions.get(asset)
            if not existing or existing.qty < qty - 1e-9:
                return OrderResult(False, "", 0, 0, "REJECTED",
                                     "sim: no position to sell")
            proceeds = qty * fill_price - commission
            self._sim_portfolio.cash += proceeds
            existing.qty -= qty
            if existing.qty <= 1e-9:
                del self._sim_portfolio.positions[asset]

        self._sim_last_price[asset] = fill_price
        return OrderResult(True, f"sim-{uuid.uuid4().hex[:12]}",
                             fill_price, qty, "FILLED")

    async def cancel_all_orders(self) -> None:
        if self._client is None:
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._client.cancel_orders)
        except Exception as e:
            log.error(f"cancel_all_orders error: {e}")

    async def close_position(self, asset: str) -> OrderResult:
        portfolio = await self.get_portfolio_state()
        pos = portfolio.positions.get(asset)
        if not pos:
            return OrderResult(True, "", 0, 0, "NOT_HELD",
                                 "no position to close")
        side = "sell" if pos.is_long else "buy"
        return await self.submit_order(asset, side, pos.qty, "market",
                                         reference_price=pos.avg_entry_price)

    def update_sim_last_price(self, asset: str, price: float) -> None:
        if price > 0:
            self._sim_last_price[asset] = price
