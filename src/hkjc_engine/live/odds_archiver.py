import redis
from hkjc_engine.config import redis_client
import json
import datetime
import time
import logging
import sys
from sqlalchemy import create_engine, text
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class HKJCOddsArchiver:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.r_cache = redis_client()
        self.target_pools = ['WIN', 'PLA', 'QIN', 'QPL', 'TRI']

    def get_time_to_race(self, venue, race_no):
        """Calculates seconds until jump. Negative numbers mean the race is delayed/past scheduled time."""
        data = self.r_cache.get(f"live_race_metadata:{venue}:{race_no}")
        if data:
            meta = json.loads(data)
            if meta and 'time' in meta and ":" in meta['time']:
                try:
                    now = datetime.datetime.now()
                    time_str = meta['time']
                    start_time = datetime.datetime.strptime(time_str, "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    return (start_time - now).total_seconds()
                except ValueError:
                    pass
        return 9999

    def parse_and_archive(self, venue, race_no, race_date_str, phase='UNKNOWN'):
        """
        Persist a snapshot of all pool odds.

        Parameters
        ----------
        phase : str
            One of:
              'PRE_STOP_SELL'  — sell window is open, odds still mutating
              'POST_STOP_SELL' — sell has stopped, late-money tote settlement in progress
              'FINAL'          — written once when race_status=CLOSED. The canonical dividend.
              'UNKNOWN'        — STOP_SELL time marker missing (older data; pre-Plan-C)
        """
        race_id = f"{race_date_str.replace('-', '')}_{venue}_{race_no:02d}"
        scrape_time = datetime.datetime.now()
        stop_sell_iso = self.r_cache.get(f"stop_sell_time:{venue}:{race_no}")
        stop_sell_dt = None
        if stop_sell_iso:
            try:
                stop_sell_dt = datetime.datetime.fromisoformat(stop_sell_iso)
            except ValueError:
                pass

        # Compute offset in seconds; negative means before STOP_SELL,positive means during the late-money window, NULL means no anchor
        seconds_vs_stop_sell = None
        if stop_sell_dt is not None:
            seconds_vs_stop_sell = (scrape_time - stop_sell_dt).total_seconds()

        records_to_insert = []

        for p_type in self.target_pools:
            raw_data = self.r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:{p_type}")
            if not raw_data:
                continue

            try:
                data = json.loads(raw_data)
                meetings = data.get('data', {}).get('raceMeetings', [])
                if not meetings: continue

                pools = meetings[0].get('pmPools', [])
                for pool in pools:
                    odds_type = pool.get('oddsType')
                    races = pool.get('leg', {}).get('races', [])
                    if not races:
                        races = pool.get('races', [])

                    if int(race_no) in races and odds_type == p_type:
                        for node in pool.get('oddsNodes', []):
                            val = node.get('value') or node.get('oddsValue')
                            combo_str = node.get('combination') or node.get('combString')

                            if val is not None and combo_str:
                                parts = str(combo_str).replace(',', '-').split('-')
                                clean_final = "-".join(str(int(p)) for p in parts if p.strip())

                                try:
                                    clean_odds = float(val)
                                    records_to_insert.append({
                                        'race_id': race_id,
                                        'timestamp': scrape_time,
                                        'pool_type': odds_type,
                                        'combination': clean_final,
                                        'odds': clean_odds,
                                        'phase': phase,
                                        'seconds_vs_stop_sell': seconds_vs_stop_sell,
                                    })
                                except ValueError:
                                    pass
            except Exception as e:
                logging.error(f"Failed to parse {p_type} API odds for R{race_no}: {e}")

        if records_to_insert:
            try:
                self._save_to_postgres(records_to_insert)
                logging.info(f"Archived {len(records_to_insert)} odds combos for {venue} R{race_no} "
                             f"[phase={phase}].")
            except Exception as e:
                logging.error(f"Failed to save R{race_no} to Postgres: {e}")
        else:
            logging.warning(f"R{race_no} parsed, but NO VALID ODDS FOUND.")

    def _save_to_postgres(self, records):
        insert_query = text("""
            INSERT INTO live_odds_history
                (race_id, timestamp, pool_type, combination, odds, phase, seconds_vs_stop_sell)
            VALUES
                (:race_id, :timestamp, :pool_type, :combination, :odds, :phase, :seconds_vs_stop_sell)
            ON CONFLICT (race_id, timestamp, pool_type, combination) DO NOTHING;
        """)

        with self.engine.begin() as conn:
            conn.execute(insert_query, records)


if __name__ == "__main__":
    archiver = HKJCOddsArchiver(DB_URL)

    VENUE = sys.argv[1].upper() if len(sys.argv) > 1 else "ST"

    logging.info(f"Starting High-Frequency Background Odds Archiver for {VENUE}...")
    logging.info("State machine: STOP_SELL → keep polling pre-close intraday snapshots. "
                 "CLOSED → capture ONE final post-late-money snapshot, then retire race.")
    closed_races = set()

    try:
        while True:
            TODAY_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
            current_race_str = archiver.r_cache.get(f"current_scraping_race:{VENUE}")

            if not current_race_str:
                logging.info("Waiting for live_scraper to broadcast an active race...")
                time.sleep(10)
                continue

            current_race = int(current_race_str)

            if current_race in closed_races:
                time.sleep(5)
                continue

            race_status = archiver.r_cache.get(f"race_status:{VENUE}:{current_race}")

            # --- STATE MACHINE ---
            # CLOSED  → scraper has finished its 180s late-money window. This is the ONLY moment we write the final canonical snapshot to Postgres, since dividends are now fully settled.
            # STOP_SELL → the live bot owns this signal. We keep scraping intraday snapshots (so odds history is dense) but do NOT treat it as terminal.
            # anything else → normal sell-window polling, continue writing intraday history at usual cadence.
            if race_status == "CLOSED":
                logging.info(f"🔒 R{current_race} CLOSED (post-late-money). "
                             f"Capturing final settled tote snapshot...")
                archiver.parse_and_archive(VENUE, current_race, TODAY_DATE, phase='FINAL')
                closed_races.add(current_race)
                logging.info(f"Final snapshot saved for R{current_race}. "
                             f"Waiting for live_scraper to advance to R{current_race + 1}...")
                time.sleep(2)
                continue

            # Not CLOSED yet — keep writing intraday history. (STOP_SELL lands here too: we continue polling during the 3-minute late-money capture window so we have dense coverage of how the final dividend converges.)
            if race_status in ('STOP_SELL', 'STOPSELL'):
                archiver.parse_and_archive(VENUE, current_race, TODAY_DATE, phase='POST_STOP_SELL')
            else:
                archiver.parse_and_archive(VENUE, current_race, TODAY_DATE, phase='PRE_STOP_SELL')

            seconds_to_jump = archiver.get_time_to_race(VENUE, current_race)

            # Tighter cadence inside the post-STOP_SELL window to catch every tote settlement tick before CLOSED arrives.
            if race_status in ('STOP_SELL', 'STOPSELL'):
                time.sleep(5)
            elif seconds_to_jump <= 120 and seconds_to_jump > -600:
                time.sleep(5)
            elif seconds_to_jump <= 300:
                time.sleep(12)
            else:
                time.sleep(30)

    except KeyboardInterrupt:
        logging.info("Archiver shut down cleanly.")
