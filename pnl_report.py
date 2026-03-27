"""
pnl_report.py — Standalone script to print a full trade history table.

Usage:
    python pnl_report.py
    python pnl_report.py --provider news
    python pnl_report.py --since 2024-01-01
"""

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import db


def _fmt_pnl(cents: float) -> str:
    if cents is None:
        return "  —"
    sign = "+" if cents >= 0 else ""
    return f"{sign}{cents:.1f}¢"


def _fmt_price(price) -> str:
    if price is None:
        return "—"
    return f"{float(price):.1f}¢"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Kalshi bot P&L report")
    parser.add_argument("--provider", help="Filter by provider name")
    parser.add_argument("--since", help="Only show trades opened on/after this date (YYYY-MM-DD)")
    parser.add_argument("--open-only", action="store_true", help="Show only open trades")
    parser.add_argument("--closed-only", action="store_true", help="Show only closed trades")
    args = parser.parse_args()

    db.init_db()
    trades = db.get_all_trades()

    # ── Filters ───────────────────────────────────────────────────────────────
    if args.provider:
        trades = [t for t in trades if t["provider"] == args.provider]
    if args.since:
        cutoff = args.since
        trades = [t for t in trades if t["opened_at"] >= cutoff]
    if args.open_only:
        trades = [t for t in trades if t["status"] == "open"]
    if args.closed_only:
        trades = [t for t in trades if t["status"] == "closed"]

    if not trades:
        print("No trades found matching the given filters.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    try:
        from tabulate import tabulate
    except ImportError:
        print("Install tabulate: pip install tabulate")
        sys.exit(1)

    headers = [
        "ID", "Ticker", "Question", "Provider",
        "My%", "Kalshi¢", "Edge%",
        "Entry", "Exit", "P&L",
        "Status", "Reason", "Opened At",
    ]

    rows = []
    for t in trades:
        question = (t["question"] or "")[:40] + ("…" if len(t["question"] or "") > 40 else "")
        rows.append([
            t["id"],
            t["ticker"],
            question,
            t["provider"],
            f"{t['my_probability']:.1f}",
            f"{t['kalshi_price']:.1f}",
            f"{t['edge_score']:.1f}",
            _fmt_price(t["entry_price"]),
            _fmt_price(t["exit_price"]),
            _fmt_pnl(t["pnl_cents"]),
            t["status"],
            t["exit_reason"] or "—",
            (t["opened_at"] or "")[:16],
        ])

    print()
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print()

    # ── Summary footer ────────────────────────────────────────────────────────
    closed = [t for t in trades if t["status"] == "closed"]
    if closed:
        total_pnl = sum(t["pnl_cents"] or 0 for t in closed)
        wins      = sum(1 for t in closed if (t["pnl_cents"] or 0) > 0)
        win_rate  = wins / len(closed) * 100
        sign      = "+" if total_pnl >= 0 else ""
        print(f"  Closed trades : {len(closed)}")
        print(f"  Win rate      : {win_rate:.1f}%  ({wins}W / {len(closed)-wins}L)")
        print(f"  Total P&L     : {sign}{total_pnl:.1f}¢  (${total_pnl/100:.2f})")

    open_trades = [t for t in trades if t["status"] == "open"]
    if open_trades:
        print(f"  Open positions: {len(open_trades)}")

    print()


if __name__ == "__main__":
    main()
