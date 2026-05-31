"""
trader.py — One full deployment pass + the 4-hour re-evaluation loop.

A "pass" is: read live balance, scan markets, filter+rank, then walk the
ranked list buying sized orders until we can't afford even one contract at
the cheapest qualifying price.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import config
from executor import Executor, ExecutionResult
from kalshi import KalshiAPIError, KalshiClient
from strategy.filter import Candidate, filter_markets
from strategy.ranker import rank
from strategy.sizer import (
    SizeDecision,
    can_afford_cheapest,
    size_all_in,
    size_fixed_dollar,
)

logger = logging.getLogger(__name__)


@dataclass
class PassSummary:
    started_at: float
    finished_at: float = 0.0
    starting_balance_cents: int = 0
    ending_balance_cents: int = 0
    markets_scanned: int = 0
    markets_considered: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_skipped: int = 0
    errors: int = 0
    halted: bool = False
    halt_reason: str = ""


def _held_tickers(client: KalshiClient) -> set[str]:
    """Tickers in which we currently have a non-zero net position.

    Kalshi's market_positions entries expose `position` as a SIGNED integer:
    positive means long Yes, negative means long No, zero means flat (even if
    `total_traded` is large from a round-trip). We must not use total_traded
    as a fallback or we'd permanently lock ourselves out of any market we've
    ever traded.
    """
    try:
        positions = client.get_positions()
    except KalshiAPIError as exc:
        logger.warning("Could not read positions; treating as none held. (%s)", exc)
        return set()
    held: set[str] = set()
    for p in positions:
        ticker = p.get("ticker") or p.get("market_ticker")
        if not ticker:
            continue
        raw = p.get("position")
        if raw is None:
            continue
        try:
            if int(raw) != 0:
                held.add(ticker)
        except (TypeError, ValueError):
            # If the field is present but unparseable, be conservative and
            # treat the market as held — better to skip a trade than to
            # double-buy.
            held.add(ticker)
    return held


def _scan(client: KalshiClient) -> list[dict]:
    markets: list[dict] = []
    try:
        for m in client.iter_markets(status="open"):
            markets.append(m)
    except KalshiAPIError as exc:
        logger.error("Market scan failed: %s", exc)
        raise
    return markets


def _decide(
    *,
    candidate: Candidate,
    working_budget_cents: int,
    fixed_trade_size_cents: int,
    mode: str,
) -> SizeDecision:
    if mode == "ALL_IN_PER_MARKET":
        return size_all_in(
            working_budget_cents=working_budget_cents,
            price_cents=candidate.yes_ask_cents,
            available_size=candidate.yes_ask_size,
        )
    return size_fixed_dollar(
        working_budget_cents=working_budget_cents,
        fixed_trade_size_cents=fixed_trade_size_cents,
        price_cents=candidate.yes_ask_cents,
        available_size=candidate.yes_ask_size,
    )


def run_pass(
    client: KalshiClient,
    *,
    live: bool = config.LIVE_TRADING,
    sizing_mode: str = config.SIZING_MODE,
    fixed_trade_size_usd: float = config.FIXED_TRADE_SIZE_USD,
    price_band: tuple[int, int] = (config.PRICE_BAND_MIN_CENTS, config.PRICE_BAND_MAX_CENTS),
    min_liquidity: int = config.MIN_LIQUIDITY_CONTRACTS,
) -> PassSummary:
    """Execute one full scan → filter → rank → deploy capital pass."""
    summary = PassSummary(started_at=time.time())
    executor = Executor(client, live=live)

    # 1. Balance — fail safe if unreadable.
    try:
        balance_cents = client.get_balance_cents()
    except KalshiAPIError as exc:
        logger.error("Balance read failed — skipping this pass. (%s)", exc)
        summary.halted = True
        summary.halt_reason = "balance_read_failed"
        summary.errors += 1
        summary.finished_at = time.time()
        return summary

    summary.starting_balance_cents = balance_cents
    working_budget = balance_cents
    logger.info("BALANCE       | $%.2f (%d¢)", balance_cents / 100, balance_cents)

    min_cents, max_cents = price_band
    fixed_trade_size_cents = int(round(fixed_trade_size_usd * 100))

    if not can_afford_cheapest(working_budget_cents=working_budget, cheapest_price_cents=min_cents):
        logger.info(
            "NO CAPITAL    | %d¢ can't afford 1 contract at %d¢ + fee. Skipping pass.",
            working_budget, min_cents,
        )
        summary.halted = True
        summary.halt_reason = "insufficient_balance_at_start"
        summary.finished_at = time.time()
        summary.ending_balance_cents = working_budget
        return summary

    # 2. Scan.
    try:
        markets = _scan(client)
    except KalshiAPIError:
        summary.errors += 1
        summary.halted = True
        summary.halt_reason = "scan_failed"
        summary.finished_at = time.time()
        summary.ending_balance_cents = working_budget
        return summary
    summary.markets_scanned = len(markets)

    held = _held_tickers(client)
    if held:
        logger.info("HOLDINGS      | already in %d markets; skipping those", len(held))

    # 3. Filter + rank.
    candidates = filter_markets(
        markets,
        min_cents=min_cents,
        max_cents=max_cents,
        min_liquidity=min_liquidity,
        held_tickers=held,
    )
    ranked = rank(candidates)
    summary.markets_considered = len(ranked)
    logger.info(
        "FILTERED      | %d/%d markets in [%d..%d]¢ band, not held, tradeable",
        len(ranked), len(markets), min_cents, max_cents,
    )

    if not ranked:
        logger.info("NO CANDIDATES | nothing in the price band this pass")
        summary.halted = True
        summary.halt_reason = "no_candidates"
        summary.finished_at = time.time()
        summary.ending_balance_cents = working_budget
        return summary

    # 4. Deploy.
    bought_this_pass: set[str] = set()
    for cand in ranked:
        if cand.ticker in bought_this_pass:
            continue
        if not can_afford_cheapest(working_budget_cents=working_budget, cheapest_price_cents=min_cents):
            logger.info(
                "NO CAPITAL    | %d¢ remaining; halting deployment for this pass.",
                working_budget,
            )
            summary.halted = True
            summary.halt_reason = "exhausted_capital"
            break

        decision = _decide(
            candidate=cand,
            working_budget_cents=working_budget,
            fixed_trade_size_cents=fixed_trade_size_cents,
            mode=sizing_mode,
        )

        logger.info(
            "CONSIDER      | %s ask=%d¢ size=%s -> %s",
            cand.ticker, cand.yes_ask_cents, cand.yes_ask_size or "?", decision.reason,
        )

        if decision.contracts <= 0:
            summary.orders_skipped += 1
            continue

        result = executor.execute(cand.ticker, decision)
        if result.status in ("market_closed", "rejected", "error"):
            summary.errors += 1
            continue

        if result.status == "dry_run":
            # In dry run we still pretend the budget would have been spent
            # so we exit the loop at a realistic point.
            summary.orders_submitted += 1
            bought_this_pass.add(cand.ticker)
            working_budget -= decision.total_outlay_cents
            if sizing_mode == "ALL_IN_PER_MARKET":
                logger.info("ALL-IN done   | stopping after one market.")
                summary.halted = True
                summary.halt_reason = "all_in_complete"
                break
            continue

        # Live path
        summary.orders_submitted += 1
        bought_this_pass.add(cand.ticker)
        if result.status == "filled":
            summary.orders_filled += 1
        # Always debit the full intended outlay — even on a partial fill,
        # Kalshi reserves the resting portion's cash from `balance`. Using
        # the filled-only number here would let us over-commit the local
        # working_budget within a single pass.
        working_budget -= decision.total_outlay_cents

        if sizing_mode == "ALL_IN_PER_MARKET":
            logger.info("ALL-IN done   | stopping after one market.")
            summary.halted = True
            summary.halt_reason = "all_in_complete"
            break

    summary.ending_balance_cents = max(0, working_budget)
    summary.finished_at = time.time()
    logger.info(
        "PASS DONE     | scanned=%d considered=%d submitted=%d filled=%d "
        "skipped=%d errors=%d duration=%.1fs",
        summary.markets_scanned, summary.markets_considered,
        summary.orders_submitted, summary.orders_filled,
        summary.orders_skipped, summary.errors,
        summary.finished_at - summary.started_at,
    )
    return summary


def run_forever(client: KalshiClient, *, recheck_interval_hours: float = config.RECHECK_INTERVAL_HOURS) -> None:
    """Run passes forever, sleeping recheck_interval_hours between them.

    Ctrl-C exits cleanly. Transient API errors don't crash the loop —
    they're logged and the next pass will retry.
    """
    sleep_seconds = max(60.0, recheck_interval_hours * 3600)
    while True:
        cycle_started = time.time()
        try:
            run_pass(client)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Unhandled exception in pass; will retry after sleep.")
        elapsed = time.time() - cycle_started
        nap = max(60.0, sleep_seconds - elapsed)
        logger.info("SLEEP         | %.1f hours until next pass (%.0fs)", nap / 3600, nap)
        try:
            time.sleep(nap)
        except KeyboardInterrupt:
            raise
