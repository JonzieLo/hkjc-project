"""
audit_missing_dates.py
======================

Compares HKJC's authoritative race-fixture list (scraped from the
`<select id='selectId'>` dropdown on the localresults page) against the
local `races` table and reports exactly which dates are missing.

Uses HKJCAsyncScraper.get_valid_race_dates(), which is already in the
codebase. This is the right source of truth -- the dropdown contains
every actual HKJC fixture, so any date in the dropdown but not in our
DB is a genuine miss (no need to guess about holidays / off-season).

Usage:
    # Audit 2024 onward (default)
    python -m hkjc_engine.data.audit_missing_dates

    # Audit a specific year range
    python -m hkjc_engine.data.audit_missing_dates --start_year 2023 --end_year 2026

    # Output as a list ready to feed to backfill_dates
    python -m hkjc_engine.data.audit_missing_dates --format args
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL
from hkjc_engine.data.scraper import HKJCAsyncScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger(__name__)


async def get_authoritative_dates(start_year: int, end_year: int) -> list[str]:
    """Returns sorted list of YYYY-MM-DD strings (note the dash format).
    `get_valid_race_dates` returns YYYY/MM/DD; we normalize."""
    scraper = HKJCAsyncScraper(max_concurrent_requests=2)
    try:
        slash_dates = await scraper.get_valid_race_dates(start_year, end_year)
    finally:
        await scraper.close_session()
    # Convert YYYY/MM/DD -> YYYY-MM-DD for SQL comparison
    return [datetime.strptime(d, "%Y/%m/%d").strftime("%Y-%m-%d")
            for d in slash_dates]


def get_local_dates(start_year: int, end_year: int) -> set[str]:
    """Returns set of dates we already have in `races`, in YYYY-MM-DD format."""
    engine = create_engine(DB_URL)
    q = text("""
        SELECT DISTINCT race_date::text
        FROM races
        WHERE EXTRACT(YEAR FROM race_date) BETWEEN :s AND :e
    """)
    with engine.connect() as conn:
        rows = conn.execute(q, {"s": start_year, "e": end_year}).fetchall()
    return {r[0] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2024)
    ap.add_argument("--end_year",   type=int, default=datetime.now().year)
    ap.add_argument("--format", choices=["table", "args"], default="table",
                    help="'table' for human-readable output; 'args' for "
                         "space-separated dates ready to pipe to backfill")
    args = ap.parse_args()

    log.info("Scraping authoritative date list from HKJC dropdown "
             "(%d-%d)...", args.start_year, args.end_year)
    authoritative = asyncio.run(
        get_authoritative_dates(args.start_year, args.end_year))
    log.info("HKJC reports %d race meetings in the requested year range.",
             len(authoritative))

    log.info("Reading local `races` table...")
    local = get_local_dates(args.start_year, args.end_year)
    log.info("Local DB has %d distinct race dates in that range.", len(local))

    missing = sorted([d for d in authoritative if d not in local])
    extra   = sorted(local - set(authoritative))

    if args.format == "args":
        # Just print dates, space-separated, ready to feed to backfill_dates
        if missing:
            print(" ".join(missing))
        return

    print("\n" + "=" * 60)
    print(f"  HKJC FIXTURE AUDIT  ({args.start_year}-{args.end_year})")
    print("=" * 60)
    print(f"  Authoritative meetings (per HKJC dropdown):  {len(authoritative)}")
    print(f"  Local DB has:                                {len(local)}")
    print(f"  MISSING from local:                          {len(missing)}")
    print(f"  EXTRA in local (shouldn't happen):           {len(extra)}")
    print("=" * 60)

    if missing:
        print("\nMISSING dates (need backfill):")
        for d in missing:
            dow = datetime.strptime(d, "%Y-%m-%d").strftime("%a")
            print(f"  {d}  ({dow})")

        print("\nTo backfill all, run:")
        print(f"  python -m hkjc_engine.data.backfill_dates {' '.join(missing)}")

    if extra:
        print("\nEXTRA dates (in local DB but NOT in HKJC dropdown -- investigate):")
        for d in extra:
            print(f"  {d}")
        print("\n  These are usually harmless (e.g. test data) but worth a")
        print("  sanity check -- if they're real meetings the dropdown might")
        print("  be paginated and the scraper might be missing pages.")

    if not missing and not extra:
        print("\n  All HKJC fixtures present in local DB. No backfill needed.")


if __name__ == "__main__":
    main()