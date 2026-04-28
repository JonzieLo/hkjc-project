"""
backfill_dates.py
=================

Backfills missing race dates by re-running the full daily_updater pipeline
on each. Designed to be safe to re-run -- daily_updater's underlying
operations are idempotent (UPSERTs and recomputes), so running on a date
that's already partially populated will fill gaps without breaking
existing rows.

Two operating modes:

    Explicit list:
        python -m hkjc_engine.data.backfill_dates 2026-04-19 2026-04-20

    Range:
        python -m hkjc_engine.data.backfill_dates --start 2026-04-15 --end 2026-04-22

WARNINGS
--------
1. Each date triggers a full re-scrape. Don't run this against months of
   data unless you genuinely need to -- HKJC's WAF will start rejecting
   you. A reasonable rate is one date per 30-60 seconds, which the
   underlying scraper already enforces internally.

2. `update_run_styles`, `compute_pace`, `compute_trueskill`, and
   `add_features` in Phase 2/3 of daily_updater are GLOBAL recomputes,
   not per-date. They run after every date is scraped, which is correct
   but slow. For a 1-2 day backfill this is fine; for >5 dates it gets
   tedious. Consider running scraping for all dates first, then a single
   feature recompute at the end -- there's a --skip_features flag for that.

3. Always run the diagnostic query in the runbook BEFORE backfilling.
   "Missing dates" can mean three different things and the right action
   depends on which.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta

from hkjc_engine.config import DB_URL
from hkjc_engine.data import (
    scraper,
    scraper_dividends,
    scraper_horse_numbers,
    patch_finish_position,
    patch_race_class,
    patch_race_win_odds,
    update_run_styles,
    compute_pace,
    compute_class_and_track,
    compute_trueskill,
    add_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger(__name__)


def _scrape_one_date(target_date: str) -> None:
    """Phase 1 only: scrape + dividends + horse numbers for one date.
    Idempotent at the DB level via the scrapers' UPSERT semantics."""
    log.info("Scraping %s ...", target_date)
    asyncio.run(scraper.main(target_date))
    asyncio.run(scraper_dividends.main(target_date))
    asyncio.run(scraper_horse_numbers.main(target_date))


def _patch_one_date(target_date: str) -> None:
    """Phase 2: patches that take a target date."""
    log.info("Patching %s ...", target_date)
    asyncio.run(patch_race_win_odds.main(target_date))
    asyncio.run(patch_race_class.main(target_date))
    asyncio.run(patch_finish_position.main(target_date))


def _recompute_global_features() -> None:
    """Phase 2/3: global recomputes (run once at end of backfill)."""
    log.info("Recomputing run-styles + pace + class/track + trueskill + features...")
    update_run_styles.update_historical_run_styles(DB_URL)
    compute_pace.compute_bucketed_pace_ema(DB_URL, alpha=0.33)
    compute_class_and_track.compute_advanced_features(DB_URL)
    compute_trueskill.compute_historical_trueskill(DB_URL)
    add_features.compute_rolling_features(DB_URL)


def parse_dates(args) -> list[str]:
    if args.dates:
        # Validate each
        for d in args.dates:
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except ValueError:
                raise SystemExit(f"Invalid date: {d}. Use YYYY-MM-DD.")
        return args.dates

    if args.start and args.end:
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d")
            end = datetime.strptime(args.end, "%Y-%m-%d")
        except ValueError as e:
            raise SystemExit(f"Invalid date in range: {e}")
        if end < start:
            raise SystemExit("--end must be on or after --start")
        # Generate dates -- we do NOT filter by Sun/Wed here because
        # midweek public holidays / Saturday meetings exist. The scraper's
        # `process_race_day` returns False for non-race days harmlessly.
        out = []
        d = start
        while d <= end:
            # Skip August (off-season) -- matches scraper.main behavior
            if d.month != 8:
                out.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        return out

    raise SystemExit("Provide either positional dates or --start/--end.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="*",
                    help="Explicit dates to backfill (YYYY-MM-DD).")
    ap.add_argument("--start", help="Range start (YYYY-MM-DD).")
    ap.add_argument("--end",   help="Range end (YYYY-MM-DD, inclusive).")
    ap.add_argument("--skip_features", action="store_true",
                    help="Skip the global feature recompute. Useful when "
                         "backfilling many dates -- run once at the end.")
    ap.add_argument("--features_only", action="store_true",
                    help="Skip scraping, just run the global feature "
                         "recompute. For when scraping is already done.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the dates that would be processed, but "
                         "don't actually scrape.")
    args = ap.parse_args()

    if args.features_only:
        log.info("--features_only: running global recomputes only.")
        _recompute_global_features()
        return

    dates = parse_dates(args)
    if not dates:
        log.warning("No dates to process.")
        return

    log.info("=" * 60)
    log.info("BACKFILL PLAN")
    log.info("=" * 60)
    log.info("Dates to process : %d", len(dates))
    log.info("First / last     : %s / %s", dates[0], dates[-1])
    log.info("Skip features    : %s", args.skip_features)
    log.info("=" * 60)

    if args.dry_run:
        for d in dates:
            log.info("  would process: %s", d)
        return

    if len(dates) > 10:
        log.warning("Backfilling %d dates. This will hit HKJC for an "
                    "extended time. Continue? (Ctrl+C to abort, "
                    "5 second pause...)", len(dates))
        try:
            import time
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Aborted.")
            return

    failures = []
    for i, d in enumerate(dates, start=1):
        log.info("[%d/%d] %s", i, len(dates), d)
        try:
            _scrape_one_date(d)
            _patch_one_date(d)
        except Exception:
            log.exception("FAILED on %s -- continuing with next date.", d)
            failures.append(d)

    if not args.skip_features:
        log.info("All scraping complete. Running global feature recomputes...")
        try:
            _recompute_global_features()
        except Exception:
            log.exception("Feature recompute FAILED. Run with --features_only "
                          "to retry.")
            sys.exit(2)

    log.info("=" * 60)
    log.info("BACKFILL COMPLETE: %d/%d dates processed",
             len(dates) - len(failures), len(dates))
    if failures:
        log.warning("Failed dates: %s", ", ".join(failures))
        log.warning("Re-run with these dates as positional args to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()