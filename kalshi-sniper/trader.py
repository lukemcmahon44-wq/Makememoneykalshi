"""
trader.py — Main scan -> size -> execute -> log loop for the Kalshi sniper.

Run it:
    python trader.py

On first launch it runs in DEMO + DRY-RUN (see config.py / README.md): it logs
the trades it WOULD place and submits nothing. Flip the env flags only after
you have validated behaviour against the demo environment.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import config
import strategy
from kalshi_client import KalshiAPIError, KalshiClient, to_decimal
from state import Store

logger = logging.getLogger("sniper")


# ──────────────────────────────────────────────────────────────────────────────
# Logging + banner
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    os.makedirs(config.LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = time.gmtime  # log in UTC

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def print_banner() -> None:
    mode = "DRY-RUN (no orders placed)" if config.DRY_RUN else "*** LIVE TRADING ***"
    is_demo_host = "demo" in config.HOST.lower()
    env = "DEMO" if is_demo_host else "*** PRODUCTION ***"
    bar = "=" * 64
    lines = [
        bar,
        "  KALSHI HIGH-PROBABILITY AUTO-SNIPER",
        bar,
        f"  Mode            : {mode}",
        f"  Environment     : {env}",
        f"  Base URL        : {config.BASE_URL}",
        f"  Probability band: {config.MIN_PROBABILITY_PRICE}–{config.MAX_PROBABILITY_PRICE}",
        f"  Position size   : {config.POSITION_PCT * 100:.0f}% of balance, "
        f"cap ${config.MAX_POSITION_PER_MARKET}/market",
        f"  Deployed cap    : {config.MAX_DEPLOYED_PCT * 100:.0f}% of balance",
        f"  Balance floor   : ${config.MIN_BALANCE}",
        f"  Daily loss limit: ${config.DAILY_LOSS_LIMIT}",
        f"  Scan interval   : {config.SCAN_INTERVAL_SECONDS}s",
        bar,
    ]
    log = logger.warning if (not config.DRY_RUN or not is_demo_host) else logger.info
    for line in lines:
        log(line)


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio math helpers
# ──────────────────────────────────────────────────────────────────────────────
def _position_exposure(position: dict) -> Decimal:
    exp = to_decimal(position.get("market_exposure_dollars"))
    if exp is None:
        cents = to_decimal(position.get("market_exposure"))
        exp = (cents / Decimal(100)) if cents is not None else Decimal(0)
    return abs(exp)


def _resting_buy_notional(order: dict) -> Decimal:
    if (order.get("action") or "").lower() != "buy":
        return Decimal(0)
    remaining = to_decimal(order.get("remaining_count_fp"))
    if remaining is None:
        remaining = to_decimal(order.get("remaining_count")) or Decimal(0)
    price = to_decimal(order.get("yes_price_dollars"))
    if price is None:
        cents = to_decimal(order.get("yes_price"))
        price = (cents / Decimal(100)) if cents is not None else Decimal(0)
    return remaining * price


def deployed_capital(positions: list[dict], resting: list[dict]) -> Decimal:
    total = Decimal(0)
    for p in positions:
        total += _position_exposure(p)
    for o in resting:
        total += _resting_buy_notional(o)
    return total


def touched_tickers(positions: list[dict], resting: list[dict]) -> set[str]:
    """Tickers we already hold or have a resting order on — never double up."""
    tickers: set[str] = set()
    for p in positions:
        pos = to_decimal(p.get("position_fp"))
        if pos is None:
            pos = to_decimal(p.get("position")) or Decimal(0)
        if p.get("ticker") and pos != 0:
            tickers.add(p["ticker"])
    for o in resting:
        if o.get("ticker"):
            tickers.add(o["ticker"])
    return tickers


# ──────────────────────────────────────────────────────────────────────────────
# Trader
# ──────────────────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self, client: KalshiClient, store: Store) -> None:
        self.client = client
        self.store = store
        self._stop = False
        self._loss_halted = False

    # Signal handlers just flip a flag; the loop shuts down cleanly.
    def request_stop(self, signum: int, _frame: Any) -> None:
        logger.warning("Received signal %s — shutting down after current step.", signum)
        self._stop = True

    def _refresh_settlements_and_pnl(self) -> Decimal:
        """Pull TODAY's settlements, persist them, and return today's realized P&L.

        Bounded with min_ts = start of today (UTC) so we don't refetch the whole
        settlement history every cycle."""
        try:
            start_of_today = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            settlements = self.client.get_settlements(min_ts=int(start_of_today.timestamp()))
            self.store.record_settlements(settlements)
        except KalshiAPIError as exc:
            logger.error("Could not refresh settlements: %s", exc)
        return self.store.realized_pnl_today()

    def _trading_active(self) -> bool:
        """True if the exchange is currently permitting trading. Fail-open: if the
        status can't be fetched or the field is absent, assume active — order
        placement itself rejects anyway if the exchange is genuinely closed."""
        try:
            status = self.client.get_exchange_status()
        except KalshiAPIError as exc:
            logger.warning("Could not fetch exchange status (%s) — proceeding.", exc)
            return True
        active = status.get("trading_active")
        return True if active is None else bool(active)

    def run_cycle(self) -> None:
        # 0. Realized-P&L / daily-loss guard (settlements refreshed first).
        pnl_today = self._refresh_settlements_and_pnl()
        if pnl_today <= -config.DAILY_LOSS_LIMIT:
            self._loss_halted = True

        # 1. Fresh balance — never sized from a cached value.
        balance = self.client.get_balance()

        # Once the daily loss limit is hit, stop buying for the rest of the run.
        if self._loss_halted:
            logger.warning(
                "DAILY LOSS LIMIT HIT (realized P&L today=$%s, limit=$%s). "
                "NEW BUYING HALTED until manual restart.",
                f"{pnl_today:.2f}",
                config.DAILY_LOSS_LIMIT,
            )
            self._status_line(balance, Decimal(0), 0, 0, 0)
            return

        if balance < config.MIN_BALANCE:
            logger.warning(
                "Balance $%s below floor $%s — skipping new trades this cycle.",
                f"{balance:.2f}",
                config.MIN_BALANCE,
            )
            self._status_line(balance, Decimal(0), 0, 0, 0)
            return

        # 1.5 Skip when the exchange isn't accepting trades (avoids firing orders
        #     outside trading hours or during maintenance). Fail-open on errors.
        if not self._trading_active():
            logger.info("Exchange not trading right now — skipping this cycle.")
            return

        # 2. Existing positions + resting orders -> dedup set + deployed capital.
        positions = self.client.get_positions()
        resting = self.client.get_resting_orders()
        held = touched_tickers(positions, resting)
        deployed = deployed_capital(positions, resting)
        cap = config.MAX_DEPLOYED_PCT * balance

        # 3. Deployed-capital gate.
        if deployed >= cap:
            logger.info(
                "Fully deployed ($%s >= cap $%s) — skipping buys this cycle.",
                f"{deployed:.2f}",
                f"{cap:.2f}",
            )
            self._status_line(balance, deployed, len(positions), 0, 0)
            return

        # 4. All open markets.
        markets = self.client.get_markets(status="open")
        if config.DERIVE_ASK_FROM_ORDERBOOK:
            self._backfill_missing_asks(markets, held)

        # 5. Pure selection + sizing.
        candidates = strategy.select_and_size(markets, balance, held)

        # 6. Re-verify balance immediately before placing (the cycle may have
        #    taken a while), then place order-by-order respecting the running cap.
        balance = self.client.get_balance()
        if balance < config.MIN_BALANCE:
            logger.warning(
                "Balance $%s dropped below floor $%s before placement — skipping.",
                f"{balance:.2f}",
                config.MIN_BALANCE,
            )
            self._status_line(balance, deployed, len(positions), len(candidates), 0)
            return
        cap = config.MAX_DEPLOYED_PCT * balance
        placed = 0
        for order in candidates:
            if order.ticker in held:  # belt-and-suspenders against double-fire
                continue
            # Pre-check against the INTENDED notional so we never place an order
            # that would breach the cap.
            if deployed + order.dollar_amount > cap:
                logger.info(
                    "Deployed cap reached ($%s + $%s > $%s) — stopping placement this cycle.",
                    f"{deployed:.2f}",
                    f"{order.dollar_amount:.2f}",
                    f"{cap:.2f}",
                )
                break
            # Advance the running total by what ACTUALLY filled: a killed
            # fill-or-kill order deploys nothing and leaves the ticker to retry.
            filled = self._place(order)
            if filled > 0:
                placed += 1
                deployed += filled
                held.add(order.ticker)

        self._status_line(balance, deployed, len(positions), len(candidates), placed)

    def _place(self, order: strategy.IntendedOrder) -> Decimal:
        """Place one order (or simulate it in dry-run).

        Returns the dollar notional actually deployed: the full amount in
        dry-run; the *filled* amount in live mode; or Decimal(0) if nothing
        filled or the call failed. The caller advances the running deployed-
        capital total by this, so a killed fill-or-kill order deploys nothing.
        """
        client_order_id = str(uuid.uuid4())

        if config.DRY_RUN:
            logger.info(
                "[DRY RUN] WOULD BUY %s  %s @ %s  = $%s",
                order.ticker,
                order.count_fp,
                order.yes_price_dollars,
                f"{order.dollar_amount:.2f}",
            )
            self.store.log_order(
                ticker=order.ticker,
                side=order.side,
                action="buy",
                price=order.price,
                count=order.count,
                dollar_amount=order.dollar_amount,
                client_order_id=client_order_id,
                result=None,
                dry_run=True,
            )
            return order.dollar_amount

        try:
            result = self.client.place_order(
                ticker=order.ticker,
                side=order.side,
                action="buy",
                count=order.count,
                yes_price=order.price,
                client_order_id=client_order_id,
            )
        except KalshiAPIError as exc:
            logger.error("Order FAILED for %s: %s", order.ticker, exc)
            self.store.log_order(
                ticker=order.ticker,
                side=order.side,
                action="buy",
                price=order.price,
                count=order.count,
                dollar_amount=order.dollar_amount,
                client_order_id=client_order_id,
                result={"error": str(exc)},
                dry_run=False,
            )
            return Decimal(0)

        # How much actually filled? Fill-or-kill is all-or-nothing, but we read
        # the reported fill count to be exact. If it's absent we conservatively
        # assume the order deployed — overstating the running total is safe for
        # the cap, whereas understating it could let us over-deploy.
        fill = to_decimal(result.get("fill_count_fp"))
        if fill is None:
            fill = to_decimal(result.get("fill_count"))
        if fill is None:
            filled_notional = order.dollar_amount
            fill_str = "unknown"
        else:
            filled_notional = fill * order.price
            fill_str = str(fill)
        logger.info(
            "ORDER %s %s @ %s | client_id=%s | order_id=%s | status=%s | filled=%s | deployed=$%s",
            order.ticker,
            order.count_fp,
            order.yes_price_dollars,
            client_order_id,
            result.get("order_id", "?"),
            result.get("status", "?"),
            fill_str,
            f"{filled_notional:.2f}",
        )
        self.store.log_order(
            ticker=order.ticker,
            side=order.side,
            action="buy",
            price=order.price,
            count=order.count,
            dollar_amount=order.dollar_amount,
            client_order_id=client_order_id,
            result=result,
            dry_run=False,
        )
        return filled_notional

    def _backfill_missing_asks(self, markets: list[dict], held: set[str]) -> None:
        """Optionally derive yes_ask_dollars from the order book for markets that
        lack it. Bounded by MAX_ORDERBOOK_LOOKUPS_PER_CYCLE to respect rate limits."""
        looked = 0
        for m in markets:
            if looked >= config.MAX_ORDERBOOK_LOOKUPS_PER_CYCLE:
                break
            if m.get("yes_ask_dollars") is not None:
                continue
            if (m.get("status") or "").lower() != "open" or m.get("ticker") in held:
                continue
            ticker = m.get("ticker")
            if not ticker:
                continue
            try:
                ask = self.client.derive_yes_ask(ticker)
            except KalshiAPIError:
                continue
            looked += 1
            if ask is not None:
                m["yes_ask_dollars"] = f"{ask:.4f}"

    def _status_line(
        self,
        balance: Decimal,
        deployed: Decimal,
        positions: int,
        candidates: int,
        placed: int,
    ) -> None:
        pct = (deployed / balance * 100) if balance > 0 else Decimal(0)
        logger.info(
            "STATUS | balance=$%s | deployed=$%s (%s%%) | positions=%d | "
            "candidates=%d | placed=%d | mode=%s",
            f"{balance:.2f}",
            f"{deployed:.2f}",
            f"{pct:.0f}",
            positions,
            candidates,
            placed,
            "DRY-RUN" if config.DRY_RUN else "LIVE",
        )

    # ── Loop ──────────────────────────────────────────────────────────────────
    def run_forever(self) -> None:
        logger.info("Sniper loop starting (scan every %ds).", config.SCAN_INTERVAL_SECONDS)
        while not self._stop:
            start = time.monotonic()
            try:
                self.run_cycle()
            except KalshiAPIError as exc:
                logger.error("API error this cycle (continuing): %s", exc)
            except Exception:  # never let one cycle crash the bot
                logger.exception("Unhandled exception in cycle (continuing).")
            elapsed = time.monotonic() - start
            self._sleep_responsive(max(0.0, config.SCAN_INTERVAL_SECONDS - elapsed))
        self._shutdown()

    def _sleep_responsive(self, seconds: float) -> None:
        """Sleep in small slices so a stop signal is honoured promptly."""
        deadline = time.monotonic() + seconds
        while not self._stop and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))

    def _shutdown(self) -> None:
        logger.warning("Shutting down.")
        try:
            resting = self.client.get_resting_orders()
        except KalshiAPIError as exc:
            logger.error("Could not list resting orders on shutdown: %s", exc)
            resting = []
        if resting:
            logger.warning("There are %d resting order(s) at shutdown:", len(resting))
            for o in resting:
                logger.warning("  resting: %s order_id=%s", o.get("ticker"), o.get("order_id"))
            if config.CANCEL_ON_EXIT and not config.DRY_RUN:
                for o in resting:
                    oid = o.get("order_id")
                    if oid and self.client.cancel_order(oid):
                        logger.warning("  cancelled %s", oid)
        else:
            logger.info("No resting orders at shutdown.")
        self.store.close()
        logger.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def build_client() -> KalshiClient:
    return KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_path=config.KALSHI_PRIVATE_KEY_PATH,
        host=config.HOST,
        api_prefix=config.API_PREFIX,
        private_key_password=config.KALSHI_PRIVATE_KEY_PASSWORD,
        read_rate_limit=config.READ_RATE_LIMIT,
        write_rate_limit=config.WRITE_RATE_LIMIT,
        max_retries=config.MAX_RETRIES,
        backoff_base=config.RETRY_BACKOFF_BASE,
        timeout=config.REQUEST_TIMEOUT,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Kalshi high-probability auto-sniper.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan/size/place cycle and exit (good for a first dry-run).",
    )
    args = parser.parse_args(argv)

    setup_logging()
    config.require_credentials()  # crashes loudly with a clear message if missing
    client = build_client()
    store = Store(config.DB_PATH)
    bot = Trader(client, store)

    signal.signal(signal.SIGINT, bot.request_stop)
    signal.signal(signal.SIGTERM, bot.request_stop)

    print_banner()
    if args.once:
        logger.info("Running a single cycle (--once).")
        try:
            bot.run_cycle()
        finally:
            bot._shutdown()
    else:
        bot.run_forever()


if __name__ == "__main__":
    main()
