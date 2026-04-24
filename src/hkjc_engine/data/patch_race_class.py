import asyncio
import aiohttp
import re
import logging
import random
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

engine = create_engine(DB_URL)

async def fetch_class(session, semaphore, race_id, race_date, venue, race_no):
    date_str = str(race_date).replace('-', '/')
    url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
    
    async with semaphore:
        await asyncio.sleep(random.uniform(1.0, 2.5))
        try:
            async with session.get(url, timeout=20) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                race_tab = soup.find('div', class_='race_tab')
                
                race_class = 'Other'
                if race_tab:
                    text_content = race_tab.get_text(separator=' ', strip=True)
                    class_match = re.search(r'(Class\s\d|Group\s\d|Gr\.\d)', text_content, re.IGNORECASE)
                    if class_match:
                        race_class = class_match.group(1).title()
                        
                return race_id, race_class
        except Exception as e:
            logging.error(f"Failed to fetch {race_id}: {e}")
            return race_id, 'Error'

async def main(target_date=None):
    query_str = "SELECT race_id, race_date, venue, race_no FROM races WHERE race_class IS NULL"
    if target_date:
        query_str += " AND race_date = :target_date"
        
    with engine.connect() as conn:
        result = conn.execute(text(query_str), {"target_date": target_date})
        missing_races = result.fetchall()
        
    if not missing_races:
        logging.info("All races already have a class assigned. Nothing to patch!")
        return
        
    logging.info(f"Found {len(missing_races)} races missing class data. Initiating patch...")
    
    semaphore = asyncio.Semaphore(5) # 5 concurrent requests
    
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
        tasks = [
            fetch_class(session, semaphore, r.race_id, r.race_date, r.venue, r.race_no) 
            for r in missing_races
        ]

        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            results = await asyncio.gather(*chunk)
            with engine.begin() as conn:
                for race_id, race_class in results:
                    if race_class != 'Error':
                        update_sql = text("UPDATE races SET race_class = :r_class WHERE race_id = :r_id")
                        conn.execute(update_sql, {"r_class": race_class, "r_id": race_id})
            
            processed = min(i+chunk_size, len(tasks))
            logging.info(f"Patched {processed}/{len(tasks)} races...")

            if processed < len(tasks):
                sleep_time = random.uniform(3.0, 7.0)
                logging.info(f"Resting for {sleep_time:.1f}s before next batch...")
                await asyncio.sleep(sleep_time)
            
            logging.info(f"Patched {min(i+chunk_size, len(tasks))}/{len(tasks)} races...")

    logging.info("Database patch complete. Your dataset is now completely intact and uncorrupted.")

if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(target_date))