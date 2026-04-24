import asyncio
import aiohttp
import random
import re
import logging
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import date, datetime
from sqlalchemy import create_engine, text
from hkjc_engine.config import DB_URL

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("scrape_horse_no.log"), logging.StreamHandler()]
)

class AsyncHorseNoScraper:
    def __init__(self, db_url, max_concurrent_requests=5):
        self.engine = create_engine(db_url)
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://racing.hkjc.com/'
        }

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_html(self, url):
        await self.init_session()
        async with self.semaphore:
            await asyncio.sleep(random.uniform(0.5, 1.5))
            try:
                async with self.session.get(url, timeout=15) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logging.error(f"Fetch error for {url}: {str(e)[:100]}")
                return None

    def get_missing_races(self, target_date=None):
        logging.info(f"Querying database for missing saddle cloths (Date Filter: {target_date or 'ALL'})...")
        
        query_str = """
            SELECT DISTINCT e.race_id, r.race_date, r.venue, r.race_no
            FROM race_entries e
            JOIN races r ON e.race_id = r.race_id
            WHERE e.horse_no IS NULL
        """

        if target_date:
            query_str += " AND r.race_date = :target_date"
            
        query_str += " ORDER BY r.race_date DESC"
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query_str), {"target_date": target_date}).fetchall()
            
        logging.info(f"Found {len(results)} races requiring saddle cloth updates.")
        
        races = []
        for row in results:
            races.append({
                'race_id': row[0],
                'race_date': row[1],
                'venue': row[2],
                'race_no': row[3]
            })
        return races

    async def process_race(self, race_data):
        race_id = race_data['race_id']
        venue = race_data['venue']
        race_no = race_data['race_no']
        
        # Format date for the new URL structure
        raw_date = race_data['race_date']
        if isinstance(raw_date, (date, datetime)):
            formatted_date = raw_date.strftime("%Y/%m/%d")
        else:
            formatted_date = str(raw_date).replace('-', '/')
            
        try:
            # Using the modern HKJC endpoint
            url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={formatted_date}&Racecourse={venue}&RaceNo={race_no}"
            
            html = await self.fetch_html(url)
            if not html: return False

            # Parse with BeautifulSoup to isolate the performance table
            soup = BeautifulSoup(html, 'html.parser')
            perf_div = soup.find('div', class_='performance')
            
            if not perf_div:
                logging.warning(f"No performance div found for {race_id}")
                return False

            # Read the isolated table
            tables = pd.read_html(StringIO(str(perf_div)), flavor='html5lib')
            if not tables:
                return False
                
            df_results = tables[0]
            
            # Normalize columns to match your other scraper logic
            df_results.columns = [str(col).strip().lower().replace(' ', '_') for col in df_results.columns]
            
            if 'horse_no.' not in df_results.columns or 'horse' not in df_results.columns:
                logging.warning(f"Missing expected columns in {race_id}")
                return False
            
            db_updates = []
            
            for _, row in df_results.iterrows():
                horse_no_raw = row.get('horse_no.')
                horse_name_raw = row.get('horse')
                
                if pd.isna(horse_no_raw) or pd.isna(horse_name_raw):
                    continue
                    
                match = re.search(r'\(([A-Z0-9]{4})\)', str(horse_name_raw))
                if match:
                    horse_code = match.group(1)
                    try:
                        horse_no = int(float(horse_no_raw))
                        db_updates.append({
                            'horse_no': horse_no,
                            'horse_code': horse_code,
                            'race_id': race_id
                        })
                    except ValueError:
                        pass 
                        
            if db_updates:
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE race_entries 
                        SET horse_no = :horse_no
                        WHERE race_id = :race_id AND horse_code = :horse_code
                    """), db_updates)
                    
            logging.info(f"Updated Saddle Cloths for {race_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed processing {race_id}: {str(e)[:100]}")
            return False

async def main(target_date=None):
    # DB_URL loaded from hkjc_engine.config
    scraper = AsyncHorseNoScraper(DB_URL, max_concurrent_requests=5)

    races_to_scrape = scraper.get_missing_races(target_date)
    
    if not races_to_scrape:
        logging.info("All races already have horse numbers. Nothing to do!")
        return

    logging.info(f"--- Starting Async Scraping Pipeline ({len(races_to_scrape)} races) ---")
    
    tasks = [scraper.process_race(race_data) for race_data in races_to_scrape]
    await asyncio.gather(*tasks)

    await scraper.close_session()
    logging.info("--- Saddle Cloth Pipeline Completed ---")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = None
        
    asyncio.run(main(target_date))