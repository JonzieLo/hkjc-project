"""
Daily data-ingestion pipeline.

Runs all patch/compute stages for a given race date against the shared Postgres DB. 
Replaces the old subprocess-launching orchestrator — every stage is now an importable function, so the pipeline runs in-process without repeated interpreter startup.

Usage:
    python -m hkjc_engine.data.daily_updater 2026-04-22
or:
    python -m hkjc_engine.data.daily_updater # prompts for date
"""
import asyncio
import logging
import sys

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def run_daily_update(target_date: str) -> None:
    """Execute the full ingestion pipeline for a single race date."""
    logging.info("=" * 60)
    logging.info(f" STARTING DAILY PIPELINE FOR: {target_date}")
    logging.info("=" * 60)

    logging.info("\n--- PHASE 1: SCRAPING NEW DATA ---")
    asyncio.run(scraper.main(target_date))
    asyncio.run(scraper_dividends.main(target_date))
    asyncio.run(scraper_horse_numbers.main(target_date))

    logging.info("\n--- PHASE 2: PATCHING MISSING FIELDS ---")
    asyncio.run(patch_race_win_odds.main(target_date))
    asyncio.run(patch_race_class.main(target_date))
    asyncio.run(patch_finish_position.main(target_date))
    update_run_styles.update_historical_run_styles(DB_URL)

    logging.info("\n--- PHASE 3: COMPUTING PREDICTIVE FEATURES ---")
    compute_pace.compute_bucketed_pace_ema(DB_URL, alpha=0.33)
    compute_class_and_track.compute_advanced_features(DB_URL)
    compute_trueskill.compute_historical_trueskill(DB_URL)
    add_features.compute_rolling_features(DB_URL)

    logging.info("=" * 60)
    logging.info(f" UPDATE COMPLETE. DATABASE READY FOR {target_date} BACKTESTING.")
    logging.info("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = input("Enter the race date to ingest (YYYY-MM-DD): ")
    run_daily_update(target_date)
