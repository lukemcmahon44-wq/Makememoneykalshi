"""Filter keeps only 95-99¢ Yes asks that are open, not held, and liquid enough."""

from strategy.filter import filter_markets, yes_ask_cents


def _market(ticker, *, ask=97, size=10, status="open", **extra):
    m = {"ticker": ticker, "yes_ask": ask, "yes_ask_size": size, "status": status}
    m.update(extra)
    return m


def test_keeps_only_95_to_99():
    markets = [
        _market("LOW", ask=10),
        _market("MID", ask=50),
        _market("EDGE_LOW", ask=95),
        _market("INSIDE", ask=97),
        _market("EDGE_HIGH", ask=99),
        _market("TOO_HIGH", ask=100),
    ]
    out = filter_markets(markets, min_cents=95, max_cents=99, min_liquidity=1)
    tickers = {c.ticker for c in out}
    assert tickers == {"EDGE_LOW", "INSIDE", "EDGE_HIGH"}


def test_drops_non_open():
    markets = [
        _market("OPEN_OK", ask=96, status="open"),
        _market("ACTIVE_OK", ask=96, status="active"),
        _market("SETTLED", ask=96, status="settled"),
        _market("CLOSED", ask=96, status="closed"),
        _market("FINALIZED", ask=96, status="finalized"),
    ]
    out = filter_markets(markets, min_cents=95, max_cents=99, min_liquidity=1)
    assert {c.ticker for c in out} == {"OPEN_OK", "ACTIVE_OK"}


def test_drops_held_tickers():
    markets = [_market("HELD", ask=97), _market("NEW", ask=97)]
    out = filter_markets(
        markets, min_cents=95, max_cents=99, min_liquidity=1,
        held_tickers={"HELD"},
    )
    assert [c.ticker for c in out] == ["NEW"]


def test_respects_liquidity_floor_when_known():
    markets = [
        _market("THIN", ask=97, size=2),
        _market("THICK", ask=97, size=50),
    ]
    out = filter_markets(markets, min_cents=95, max_cents=99, min_liquidity=10)
    assert [c.ticker for c in out] == ["THICK"]


def test_unknown_size_passes_when_min_liquidity_is_one():
    # Some Kalshi payloads don't expose yes_ask_size; we let those through
    # so the orderbook check downstream can be authoritative.
    market = {"ticker": "Q", "yes_ask": 97, "status": "open"}
    out = filter_markets([market], min_cents=95, max_cents=99, min_liquidity=1)
    assert [c.ticker for c in out] == ["Q"]


def test_drops_markets_without_yes_ask():
    markets = [
        {"ticker": "NO_PRICE", "status": "open"},
        _market("HAS_PRICE", ask=97),
    ]
    out = filter_markets(markets, min_cents=95, max_cents=99, min_liquidity=1)
    assert [c.ticker for c in out] == ["HAS_PRICE"]


def test_yes_ask_cents_rejects_garbage():
    assert yes_ask_cents({"yes_ask": None}) is None
    assert yes_ask_cents({"yes_ask": "abc"}) is None
    assert yes_ask_cents({"yes_ask": 0}) is None
    assert yes_ask_cents({"yes_ask": 100}) is None
    assert yes_ask_cents({"yes_ask": 97}) == 97
