"""
test_trader.py — Unit tests for the loop's guard rails.

These exercise Trader.run_cycle() against a FakeClient (no network), focusing on
the safety logic that the pure strategy tests can't reach: the order-by-order
deployed-capital cap, in-cycle dedup, the balance floor, the daily-loss halt,
and — importantly — correct deployed-capital accounting when a fill-or-kill order
is *killed* (deploys nothing) versus *filled*.

Run from this directory:  python -m pytest -v
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import config
import trader
from kalshi_client import KalshiAPIError
from state import Store

FAR = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat().replace("+00:00", "Z")
TODAY_TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mk(ticker: str, ask: str = "0.95", size: str = "1000") -> dict:
    return {
        "ticker": ticker,
        "status": "open",
        "yes_ask_dollars": ask,
        "yes_ask_size_fp": size,
        "close_time": FAR,
        "fractional_trading_enabled": False,
        "price_level_structure": "linear_cent",
    }


class FakeClient:
    """A scripted stand-in for KalshiClient. Records placed orders."""

    def __init__(
        self,
        balance="100",
        markets=None,
        positions=None,
        resting=None,
        settlements=None,
        order_result=None,
        order_error=False,
    ):
        self._balance = Decimal(balance)
        self._markets = markets or []
        self._positions = positions or []
        self._resting = resting or []
        self._settlements = settlements or []
        self._order_result = order_result
        self._order_error = order_error
        self.placed_orders: list[dict] = []

    def get_settlements(self, min_ts=None, limit=200):
        return self._settlements

    def get_balance(self):
        return self._balance

    def get_positions(self):
        return self._positions

    def get_resting_orders(self):
        return self._resting

    def get_markets(self, status="open"):
        return self._markets

    def place_order(self, **kw):
        self.placed_orders.append(kw)
        if self._order_error:
            raise KalshiAPIError("simulated failure")
        if self._order_result is not None:
            return self._order_result
        # Default: a fully-filled order echoing the requested count.
        return {"order_id": "oid", "status": "executed", "fill_count_fp": f"{kw['count']:.2f}"}


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = Store(path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def restore_flags():
    """Save/restore the module-level mode flags so tests can flip them freely."""
    dry, demo = config.DRY_RUN, config.USE_DEMO
    yield
    config.DRY_RUN, config.USE_DEMO = dry, demo


def fills(store: Store) -> list[dict]:
    return [
        dict(r)
        for r in store._conn.execute(
            "SELECT ticker, status, dry_run, count, dollar_amount FROM fills ORDER BY id"
        ).fetchall()
    ]


# ── Dry-run ─────────────────────────────────────────────────────────────────--
def test_dry_run_logs_but_places_nothing(store):
    config.DRY_RUN = True
    client = FakeClient(balance="10", markets=[mk("A", "0.97")])
    trader.Trader(client, store).run_cycle()
    assert client.placed_orders == []          # no real order submitted
    rows = fills(store)
    assert len(rows) == 1 and rows[0]["ticker"] == "A" and rows[0]["dry_run"] == 1


# ── Dedup ─────────────────────────────────────────────────────────────────────
def test_held_position_is_not_rebought(store):
    config.DRY_RUN = True
    client = FakeClient(
        balance="100",
        markets=[mk("HELD"), mk("NEW")],
        positions=[{"ticker": "HELD", "position_fp": "1", "market_exposure_dollars": "1"}],
    )
    trader.Trader(client, store).run_cycle()
    rows = fills(store)
    assert [r["ticker"] for r in rows] == ["NEW"]


# ── Deployed-capital cap (dry-run simulates deployment) ─────────────────────────
def test_deployed_cap_stops_placement(store):
    config.DRY_RUN = True
    # balance 100 -> cap 80. A pre-existing position already uses $74, so only one
    # more $4.75 order (5 contracts @ 0.95) fits before the cap is hit.
    client = FakeClient(
        balance="100",
        markets=[mk("A"), mk("B"), mk("C")],
        positions=[{"ticker": "POS", "position_fp": "1", "market_exposure_dollars": "74"}],
    )
    trader.Trader(client, store).run_cycle()
    rows = fills(store)
    assert len(rows) == 1 and rows[0]["ticker"] == "A"


# ── Balance floor ───────────────────────────────────────────────────────────--
def test_balance_floor_halts_buying(store):
    config.DRY_RUN = True
    client = FakeClient(balance="3", markets=[mk("A")])  # below MIN_BALANCE
    trader.Trader(client, store).run_cycle()
    assert fills(store) == [] and client.placed_orders == []


# ── Daily-loss halt ─────────────────────────────────────────────────────────--
def test_daily_loss_limit_halts_buying(store):
    config.DRY_RUN = True
    loss = {
        "ticker": "LOSS",
        "settled_time": TODAY_TS,
        "revenue": 0,                       # got nothing back
        "yes_total_cost_dollars": "6.00",   # paid $6  -> pnl = -6, beyond -$5 limit
        "no_total_cost_dollars": "0",
        "fee_cost": 0,
    }
    client = FakeClient(balance="100", markets=[mk("A")], settlements=[loss])
    t = trader.Trader(client, store)
    t.run_cycle()
    assert t._loss_halted is True
    assert fills(store) == [] and client.placed_orders == []


# ── Fill-or-kill accounting (live mode) ────────────────────────────────────────
def test_killed_fok_order_deploys_nothing(store):
    """A killed FOK order must NOT consume the deployed-capital budget, so the
    next candidate is still attempted."""
    config.DRY_RUN = False
    killed = {"order_id": "o", "status": "canceled", "fill_count_fp": "0.00"}
    # deployed $74 of an $80 cap leaves room for exactly one $4.75 order at a
    # time. With kills deploying nothing, BOTH markets should be attempted.
    client = FakeClient(
        balance="100",
        markets=[mk("A"), mk("B")],
        positions=[{"ticker": "POS", "position_fp": "1", "market_exposure_dollars": "74"}],
        order_result=killed,
    )
    trader.Trader(client, store).run_cycle()
    assert len(client.placed_orders) == 2          # both attempted (none deployed)
    rows = fills(store)
    assert len(rows) == 2 and all(r["status"] == "canceled" and r["dry_run"] == 0 for r in rows)


def test_filled_fok_order_consumes_cap(store):
    """A filled order DOES consume the budget, so the cap stops the next one."""
    config.DRY_RUN = False
    client = FakeClient(
        balance="100",
        markets=[mk("A"), mk("B")],
        positions=[{"ticker": "POS", "position_fp": "1", "market_exposure_dollars": "74"}],
        # order_result=None -> FakeClient returns a fully-filled order.
    )
    trader.Trader(client, store).run_cycle()
    assert len(client.placed_orders) == 1          # first fills, cap stops the second
    rows = fills(store)
    assert len(rows) == 1 and rows[0]["status"] == "executed"


def test_order_api_error_is_isolated(store):
    """A failing order is logged and does not crash the cycle or deploy capital."""
    config.DRY_RUN = False
    client = FakeClient(balance="100", markets=[mk("A"), mk("B")], order_error=True)
    trader.Trader(client, store).run_cycle()        # must not raise
    assert len(client.placed_orders) == 2           # both attempted; neither deployed
    rows = fills(store)
    assert len(rows) == 2 and all(r["status"] is None for r in rows)
