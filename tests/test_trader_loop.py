"""Capital deployment loop halts at zero and respects already-held positions."""

from unittest.mock import MagicMock

from trader import run_pass


def _market(ticker, *, ask, size=10):
    return {"ticker": ticker, "yes_ask": ask, "yes_ask_size": size, "status": "open"}


def _client_with(markets, *, balance_cents, positions=None):
    client = MagicMock()
    client.get_balance_cents.return_value = balance_cents
    client.iter_markets.return_value = iter(markets)
    client.get_positions.return_value = positions or []
    return client


def test_pass_halts_when_balance_below_cheapest():
    # 50¢ in the account, cheapest qualifying price is 95¢ — must halt up-front.
    client = _client_with([_market("X", ask=95)], balance_cents=50)
    summary = run_pass(client, live=False, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.halted
    assert summary.halt_reason == "insufficient_balance_at_start"
    assert summary.orders_submitted == 0
    client.iter_markets.assert_not_called()


def test_pass_walks_cheapest_first_in_fixed_dollar_dry_run():
    markets = [
        _market("PRICEY", ask=99, size=99),
        _market("MID",    ask=97, size=99),
        _market("CHEAP",  ask=95, size=99),
    ]
    client = _client_with(markets, balance_cents=500)   # $5
    summary = run_pass(client, live=False, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.markets_scanned == 3
    assert summary.markets_considered == 3
    # $5 balance with $1 slices buys 1 contract per market until the balance
    # is below the cheapest price+fee.  At 95+1, 97+1, 99+1 the dry-run loop
    # commits ~96 + ~98 + ~100 = 294¢, leaving 206¢.  We should have hit at
    # least three submissions (one per market available) and stopped because
    # we ran out of candidates, not balance.
    assert summary.orders_submitted == 3
    # Submitted = filled+resting+dry, but in dry run nothing actually fills.
    assert summary.orders_filled == 0


def test_pass_stops_when_capital_exhausted_during_walk():
    markets = [_market(f"M{i}", ask=95, size=99) for i in range(20)]
    client = _client_with(markets, balance_cents=300)   # $3 -> 3 contracts at 96¢ outlay
    summary = run_pass(client, live=False, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.orders_submitted == 3
    assert summary.halted
    assert summary.halt_reason == "exhausted_capital"
    assert summary.ending_balance_cents < 96


def test_all_in_places_one_order_then_stops():
    markets = [
        _market("ONE",   ask=95, size=999),
        _market("TWO",   ask=95, size=999),
        _market("THREE", ask=97, size=999),
    ]
    client = _client_with(markets, balance_cents=1000)
    summary = run_pass(client, live=False, sizing_mode="ALL_IN_PER_MARKET",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.orders_submitted == 1
    assert summary.halted
    assert summary.halt_reason == "all_in_complete"


def test_held_tickers_are_skipped():
    markets = [
        _market("HELD",   ask=95, size=99),
        _market("FRESH",  ask=97, size=99),
    ]
    client = _client_with(
        markets,
        balance_cents=200,
        positions=[{"ticker": "HELD", "position": 5}],
    )
    summary = run_pass(client, live=False, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.markets_considered == 1   # only FRESH survived the held filter


def test_balance_read_failure_halts_pass_cleanly():
    from kalshi import KalshiAPIError
    client = MagicMock()
    client.get_balance_cents.side_effect = KalshiAPIError(
        500, "boom", "GET", "/portfolio/balance",
    )
    summary = run_pass(client, live=False, sizing_mode="FIXED_DOLLAR",
                       fixed_trade_size_usd=1.0, price_band=(95, 99),
                       min_liquidity=1)
    assert summary.halted
    assert summary.halt_reason == "balance_read_failed"
    assert summary.errors == 1
    client.iter_markets.assert_not_called()
