"""Ranker orders 95→99 with deterministic tie-breaks."""

from strategy.filter import Candidate
from strategy.ranker import rank


def _c(ticker, *, ask, size=10, close_time=None):
    raw = {"ticker": ticker, "yes_ask": ask}
    if close_time is not None:
        raw["close_time"] = close_time
    return Candidate(ticker=ticker, yes_ask_cents=ask, yes_ask_size=size, raw=raw)


def test_orders_by_price_ascending():
    cands = [
        _c("D", ask=99),
        _c("A", ask=95),
        _c("C", ask=98),
        _c("B", ask=97),
    ]
    ordered = [c.ticker for c in rank(cands)]
    assert ordered == ["A", "B", "C", "D"]


def test_tiebreak_prefers_more_liquidity():
    cands = [
        _c("THIN", ask=95, size=1),
        _c("THICK", ask=95, size=999),
        _c("MED", ask=95, size=50),
    ]
    ordered = [c.ticker for c in rank(cands)]
    assert ordered == ["THICK", "MED", "THIN"]


def test_tiebreak_prefers_sooner_close_time():
    cands = [
        _c("LATE", ask=95, size=10, close_time="2030-01-01T00:00:00Z"),
        _c("SOON", ask=95, size=10, close_time="2026-06-01T00:00:00Z"),
    ]
    ordered = [c.ticker for c in rank(cands)]
    assert ordered == ["SOON", "LATE"]


def test_tiebreak_alphabetical_last_resort():
    cands = [
        _c("BBB", ask=95, size=10),
        _c("AAA", ask=95, size=10),
    ]
    ordered = [c.ticker for c in rank(cands)]
    assert ordered == ["AAA", "BBB"]


def test_full_priority_chain_is_stable():
    cands = [
        _c("CHEAP_THIN_LATE_Z", ask=95, size=1, close_time="2030-01-01T00:00:00Z"),
        _c("CHEAP_THICK_SOON_A", ask=95, size=99, close_time="2026-06-01T00:00:00Z"),
        _c("PRICEY_THICK_SOON_A", ask=99, size=99, close_time="2026-06-01T00:00:00Z"),
    ]
    ordered = [c.ticker for c in rank(cands)]
    assert ordered == [
        "CHEAP_THICK_SOON_A",
        "CHEAP_THIN_LATE_Z",
        "PRICEY_THICK_SOON_A",
    ]
