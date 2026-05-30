"""
Executor — turns a SizeDecision into a real (or dry-run) order at Kalshi.

Responsibilities:
  - Generate a unique, idempotent client_order_id per attempt.
  - In dry-run (LIVE_TRADING=False): log the intended order, place nothing.
  - In live mode: submit a limit YES buy at the current ask, then re-read
    the order + fills to confirm what really happened (no order is assumed
    to have filled). Handle MarketClosedError by reporting it cleanly.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from kalshi import KalshiClient, KalshiAPIError, MarketClosedError
from strategy.sizer import SizeDecision

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    ticker: str
    intended_contracts: int
    intended_price_cents: int
    submitted: bool                      # did we actually call POST /portfolio/orders?
    order_id: Optional[str] = None       # Kalshi's order id, if submitted
    filled_contracts: int = 0
    filled_cost_cents: int = 0           # actual cash spent on fills
    filled_fee_cents: int = 0            # actual fees paid on fills
    status: str = ""                     # "dry_run" | "filled" | "partial" | "resting"
                                         # | "canceled" | "rejected" | "market_closed"
                                         # | "error"
    message: str = ""


class Executor:
    def __init__(self, client: KalshiClient, *, live: bool, fill_settle_seconds: float = 1.5):
        self.client = client
        self.live = live
        self.fill_settle_seconds = fill_settle_seconds

    def execute(self, ticker: str, decision: SizeDecision) -> ExecutionResult:
        if decision.contracts <= 0:
            return ExecutionResult(
                ticker=ticker,
                intended_contracts=0,
                intended_price_cents=decision.price_cents,
                submitted=False,
                status="skipped",
                message=decision.reason,
            )

        if not self.live:
            logger.info(
                "DRY-RUN ORDER | %s | side=yes count=%d price=%d¢ cost=%d¢ fee≈%d¢ outlay≈%d¢ | %s",
                ticker, decision.contracts, decision.price_cents,
                decision.cost_cents, decision.fee_cents, decision.total_outlay_cents,
                decision.reason,
            )
            return ExecutionResult(
                ticker=ticker,
                intended_contracts=decision.contracts,
                intended_price_cents=decision.price_cents,
                submitted=False,
                status="dry_run",
                message=decision.reason,
            )

        client_order_id = f"hp-{uuid.uuid4().hex[:24]}"
        logger.info(
            "SUBMIT ORDER  | %s | side=yes count=%d price=%d¢ outlay≤%d¢ | id=%s",
            ticker, decision.contracts, decision.price_cents,
            decision.total_outlay_cents, client_order_id,
        )

        try:
            order = self.client.place_order(
                ticker=ticker,
                side="yes",
                count=decision.contracts,
                yes_price_cents=decision.price_cents,
                client_order_id=client_order_id,
            )
        except MarketClosedError as exc:
            logger.warning("Market %s closed between scan and submit: %s", ticker, exc)
            return ExecutionResult(
                ticker=ticker,
                intended_contracts=decision.contracts,
                intended_price_cents=decision.price_cents,
                submitted=False,
                status="market_closed",
                message=str(exc),
            )
        except KalshiAPIError as exc:
            logger.error("Order rejected for %s: %s", ticker, exc)
            return ExecutionResult(
                ticker=ticker,
                intended_contracts=decision.contracts,
                intended_price_cents=decision.price_cents,
                submitted=False,
                status="rejected",
                message=str(exc),
            )

        order_id = str(order.get("order_id") or order.get("id") or "")
        result = ExecutionResult(
            ticker=ticker,
            intended_contracts=decision.contracts,
            intended_price_cents=decision.price_cents,
            submitted=True,
            order_id=order_id or None,
            status="submitted",
            message=f"client_order_id={client_order_id}",
        )

        if self.fill_settle_seconds > 0:
            time.sleep(self.fill_settle_seconds)

        if order_id:
            self._reconcile(result, order_id)
        else:
            logger.warning(
                "Order for %s came back without an order_id; cannot reconcile fills",
                ticker,
            )
        return result

    def _reconcile(self, result: ExecutionResult, order_id: str) -> None:
        try:
            order = self.client.get_order(order_id)
        except KalshiAPIError as exc:
            logger.warning("Could not re-read order %s: %s", order_id, exc)
            order = {}
        try:
            fills = self.client.get_fills(order_id=order_id)
        except KalshiAPIError as exc:
            logger.warning("Could not read fills for order %s: %s", order_id, exc)
            fills = []

        filled = 0
        cost = 0
        fee = 0
        for f in fills:
            try:
                count = int(f.get("count") or 0)
                price = int(f.get("yes_price") or f.get("price") or 0)
            except (TypeError, ValueError):
                continue
            filled += count
            cost += count * price
            try:
                fee += int(f.get("fees") or f.get("fee") or 0)
            except (TypeError, ValueError):
                pass

        result.filled_contracts = filled
        result.filled_cost_cents = cost
        result.filled_fee_cents = fee

        status = (order.get("status") or "").lower() if order else ""
        remaining = order.get("remaining_count") if order else None

        if filled == 0:
            result.status = "resting" if status in ("resting", "open") else (status or "submitted")
        elif filled < result.intended_contracts:
            result.status = "partial"
        else:
            result.status = "filled"

        logger.info(
            "RECONCILE     | %s | order_id=%s status=%s filled=%d/%d cost=%d¢ fees=%d¢ remaining=%s",
            result.ticker, order_id, result.status,
            filled, result.intended_contracts, cost, fee, remaining,
        )
