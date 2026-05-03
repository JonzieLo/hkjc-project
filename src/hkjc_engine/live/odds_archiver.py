import redis
from hkjc_engine.config import redis_client
import json
import datetime
import time
import logging
import sys
import re
from sqlalchemy import create_engine, text
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HKJCOddsArchiver:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.r_cache = redis_client()
        self.target_pools = ['WIN', 'PLA', 'QIN', 'QPL', 'TRI', 'F4', 'QTT', 'DBL']
        
        # State tracker to detect stale ticks (volume shifts without odds changes)
        self.previous_tick_states = {}

    def get_time_to_race(self, venue, race_no):
        """Calculates seconds until jump. Negative numbers mean the race is delayed/past scheduled time."""
        data = self.r_cache.get(f"live_race_metadata:{venue}:{race_no}")
        if data:
            meta = json.loads(data)
            if meta and 'time' in meta and ":" in meta['time']:
                try:
                    now = datetime.datetime.now()
                    start_time = datetime.datetime.strptime(meta['time'], "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    return (start_time - now).total_seconds()
                except ValueError:
                    pass
        return 9999

    def _parse_exchange_time(self, time_str, fallback_dt):
        """Safely parse the HKJC server timestamp, fallback to local time if empty or invalid."""
        if not time_str:
            return fallback_dt
        try:
            # Python's fromisoformat parses '2026-05-02T19:50:53.608+08:00' natively
            return datetime.datetime.fromisoformat(time_str)
        except ValueError:
            return fallback_dt

    def parse_and_archive(self, venue, race_no, race_date_str, phase='UNKNOWN'):
        """
        Persist a snapshot of all pool odds and total investments.
        """
        race_id = f"{race_date_str.replace('-', '')}_{venue}_{race_no:02d}"
        local_scrape_time = datetime.datetime.now()
        
        stop_sell_iso = self.r_cache.get(f"stop_sell_time:{venue}:{race_no}")
        stop_sell_dt = None
        if stop_sell_iso:
            try: 
                stop_sell_dt = datetime.datetime.fromisoformat(stop_sell_iso)
            except ValueError: 
                pass

        records_to_insert = []
        liquidity_to_insert = []

        if race_id not in self.previous_tick_states:
            self.previous_tick_states[race_id] = {}

        # --- 1. PRE-FETCH AND PARSE POOL LIQUIDITY ---
        pool_liquidity_map = {}
        raw_inv_data = self.r_cache.get(f"live_investment_raw:{venue}:{race_no}")
        if raw_inv_data:
            try:
                inv_json = json.loads(raw_inv_data)
                meetings = inv_json.get('data', {}).get('raceMeetings', [])
                if meetings and 'poolInvs' in meetings[0]:
                    for p in meetings[0]['poolInvs']:
                        o_type = p.get('oddsType')
                        inv_val = p.get('investment')
                        last_update = p.get('lastUpdateTime')
                        
                        if o_type and inv_val is not None:
                            # Strip formatting (e.g. "$12,345" -> 12345)
                            stripped = re.sub(r'[^\d]', '', str(inv_val))
                            clean_total = int(stripped) if stripped else 0
                            
                            # Use exchange time for the liquidity record
                            exchange_time = self._parse_exchange_time(last_update, local_scrape_time)
                            
                            # Calculate offset based on exchange time, stripping tzinfo to avoid aware/naive math errors
                            sec_vs_stop_sell = None
                            if stop_sell_dt:
                                sec_vs_stop_sell = (exchange_time.replace(tzinfo=None) - stop_sell_dt.replace(tzinfo=None)).total_seconds()

                            pool_liquidity_map[o_type] = {
                                'total': clean_total,
                                'timestamp': exchange_time,
                                'sec_vs_stop': sec_vs_stop_sell
                            }
            except Exception as e:
                logging.error(f"Failed to parse investment payload for R{race_no}: {e}")

        # --- 2. PROCESS ODDS MATRIX & MERGE LIQUIDITY ---
        for p_type in self.target_pools:
            
            # Append to Liquidity Write Queue
            liq_data = pool_liquidity_map.get(p_type)
            if liq_data is not None:
                liquidity_to_insert.append({
                    'race_id': race_id,
                    'timestamp': liq_data['timestamp'],
                    'pool_type': p_type,
                    'total_investment': liq_data['total'],
                    'phase': phase,
                    'seconds_vs_stop_sell': liq_data['sec_vs_stop'],
                })

            raw_odds_data = self.r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:{p_type}")
            if not raw_odds_data:
                continue

            try:
                data = json.loads(raw_odds_data)
                meetings = data.get('data', {}).get('raceMeetings', [])
                if not meetings: continue

                pools = meetings[0].get('pmPools', [])
                for pool in pools:
                    odds_type = pool.get('oddsType')
                    races = pool.get('leg', {}).get('races', [])
                    if not races: races = pool.get('races', [])

                    if int(race_no) in races and odds_type == p_type:
                        
                        # Extract the exact server timestamp for this specific odds matrix
                        odds_last_update = pool.get('lastUpdateTime')
                        odds_exchange_time = self._parse_exchange_time(odds_last_update, local_scrape_time)
                        
                        odds_sec_vs_stop = None
                        if stop_sell_dt:
                            odds_sec_vs_stop = (odds_exchange_time.replace(tzinfo=None) - stop_sell_dt.replace(tzinfo=None)).total_seconds()

                        current_odds_dict = {}
                        for node in pool.get('oddsNodes', []):
                            val = node.get('value') or node.get('oddsValue')
                            combo_str = node.get('combination') or node.get('combString')

                            if val is not None and combo_str:
                                parts = str(combo_str).replace(',', '-').split('-')
                                clean_final = "-".join(str(int(p)) for p in parts if p.strip())

                                try:
                                    clean_odds = float(val)
                                    current_odds_dict[clean_final] = clean_odds
                                    records_to_insert.append({
                                        'race_id': race_id,
                                        'timestamp': odds_exchange_time, # Insert exact exchange tick time
                                        'pool_type': odds_type,
                                        'combination': clean_final,
                                        'odds': clean_odds,
                                        'phase': phase,
                                        'seconds_vs_stop_sell': odds_sec_vs_stop,
                                    })
                                except ValueError:
                                    pass
                        
                        # --- 3. EDGE CASE: STALE TICK DETECTION ---
                        current_odds_hash = hash(frozenset(current_odds_dict.items()))
                        prev_state = self.previous_tick_states[race_id].get(p_type, {})
                        
                        prev_pool_size = prev_state.get('pool_size')
                        prev_odds_hash = prev_state.get('odds_hash')
                        current_total = liq_data['total'] if liq_data else None

                        if prev_pool_size is not None and current_total is not None:
                            # If capital entered the pool but the tote did not update the odds matrix
                            if current_total != prev_pool_size and current_odds_hash == prev_odds_hash:
                                logging.warning(f"⚠️ STALE TICK: {p_type} volume mutated "
                                                f"(${prev_pool_size} -> ${current_total}) but odds matrix froze at {odds_exchange_time}.")
                        
                        # Update local state cache
                        self.previous_tick_states[race_id][p_type] = {
                            'pool_size': current_total,
                            'odds_hash': current_odds_hash
                        }

            except Exception as e:
                logging.error(f"Failed to parse {p_type} API odds for R{race_no}: {e}")

        # Execute Bulk Writes
        if records_to_insert:
            try:
                self._save_odds_to_postgres(records_to_insert)
            except Exception as e:
                logging.error(f"Failed to save odds to Postgres: {e}")
                
        if liquidity_to_insert:
            try:
                self._save_liquidity_to_postgres(liquidity_to_insert)
                logging.info(f"Archived {len(records_to_insert)} combos & {len(liquidity_to_insert)} pool sizes for {venue} R{race_no} [phase={phase}].")
            except Exception as e:
                logging.error(f"Failed to save liquidity to Postgres: {e}")

    def _save_odds_to_postgres(self, records):
        insert_query = text("""
            INSERT INTO live_odds_history
                (race_id, timestamp, pool_type, combination, odds, phase, seconds_vs_stop_sell)
            VALUES
                (:race_id, :timestamp, :pool_type, :combination, :odds, :phase, :seconds_vs_stop_sell)
            ON CONFLICT (race_id, timestamp, pool_type, combination) DO NOTHING;
        """)
        with self.engine.begin() as conn:
            conn.execute(insert_query, records)

    def _save_liquidity_to_postgres(self, records):
        insert_query = text("""
            INSERT INTO pool_liquidity_history
                (race_id, timestamp, pool_type, total_investment, phase, seconds_vs_stop_sell)
            VALUES
                (:race_id, :timestamp, :pool_type, :total_investment, :phase, :seconds_vs_stop_sell)
            ON CONFLICT (race_id, timestamp, pool_type) DO NOTHING;
        """)
        with self.engine.begin() as conn:
            conn.execute(insert_query, records)


if __name__ == "__main__":
    archiver = HKJCOddsArchiver(DB_URL)

    VENUE = sys.argv[1].upper() if len(sys.argv) > 1 else "ST"

    logging.info(f"Starting High-Frequency Background Odds & Liquidity Archiver for {VENUE}...")
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
            if race_status == "CLOSED":
                logging.info(f"🔒 R{current_race} CLOSED (post-late-money). "
                             f"Capturing final settled tote snapshot...")
                archiver.parse_and_archive(VENUE, current_race, TODAY_DATE, phase='FINAL')
                closed_races.add(current_race)
                
                # Purge stale tick cache for memory management
                archiver.previous_tick_states.pop(f"{TODAY_DATE.replace('-', '')}_{VENUE}_{current_race:02d}", None)
                
                logging.info(f"Final snapshot saved for R{current_race}. "
                             f"Waiting for live_scraper to advance to R{current_race + 1}...")
                time.sleep(2)
                continue

            # Not CLOSED yet — keep writing intraday history.
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