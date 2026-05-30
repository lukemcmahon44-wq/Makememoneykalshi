"""Dry-run executor places zero orders and logs intent."""

import logging
from unittest.mock import MagicMock

from executor import Executor
from strategy.sizer import SizeDecision


def _decision(contracts=1, price=95):
    return SizeDecision(
        contracts=contracts,
        price_cents=price,
        cost_cents=contracts * price,
        fee_cents=1 if contracts > 0 else 0,
        total_outlay_cents=contracts * price + (1 if contracts > 0 else 0),
        reason="test",
    )


def test_dry_run_never_calls_place_order(caplog):
    fake_client = MagicMock()
    ex = Executor(fake_client, live=False)
    with caplog.at_level(logging.INFO):
        result = ex.execute("KXTEST-1", _decision(2, 97))
    fake_client.place_order.assert_not_called()
    fake_client.get_order.assert_not_called()
    fake_client.get_fills.assert_not_called()
    assert result.submitted is False
    assert result.status == "dry_run"
    assert result.intended_contracts == 2
    assert result.intended_price_cents == 97
    assert any("DRY-RUN ORDER" in r.message for r in caplog.records)
    assert any("KXTEST-1" in r.message for r in caplog.records)


def test_dry_run_zero_contracts_skipped():
    fake_client = MagicMock()
    ex = Executor(fake_client, live=False)
    result = ex.execute("KXTEST-2", _decision(0, 95))
    fake_client.place_order.assert_not_called()
    assert result.status == "skipped"
    assert result.submitted is False


def test_live_path_submits_then_reconciles(monkeypatch):
    fake_client = MagicMock()
    fake_client.place_order.return_value = {"order_id": "ord_abc", "status": "resting"}
    fake_client.get_order.return_value = {
        "order_id": "ord_abc", "status": "filled", "remaining_count": 0,
    }
    fake_client.get_fills.return_value = [
        {"count": 2, "yes_price": 97, "fees": 1},
    ]
    ex = Executor(fake_client, live=True, fill_settle_seconds=0)
    result = ex.execute("KXLIVE", _decision(2, 97))
    fake_client.place_order.assert_called_once()
    fake_client.get_order.assert_called_once_with("ord_abc")
    fake_client.get_fills.assert_called_once_with(order_id="ord_abc")
    assert result.submitted is True
    assert result.order_id == "ord_abc"
    assert result.filled_contracts == 2
    assert result.filled_cost_cents == 194
    assert result.filled_fee_cents == 1
    assert result.status == "filled"


def test_live_path_market_closed_returns_clean_status():
    from kalshi import MarketClosedError
    fake_client = MagicMock()
    fake_client.place_order.side_effect = MarketClosedError(
        400, "market closed", "POST", "/portfolio/orders",
    )
    ex = Executor(fake_client, live=True, fill_settle_seconds=0)
    result = ex.execute("KXCLOSED", _decision(1, 96))
    assert result.submitted is False
    assert result.status == "market_closed"
