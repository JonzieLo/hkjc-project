import asyncio
import aiohttp
import random
import sys
import logging
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import Column, String, Integer, Numeric, UniqueConstraint, create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from hkjc_engine.config import DB_URL

# DB_URL loaded from hkjc_engine.config
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class RaceDividend(Base):
    __tablename__ = 'race_dividends'
    id = Column(Integer, primary_key=True)
    race_id = Column(String(50), index=True)
    pool = Column(String(100)) 
    combination = Column(String(500)) 
    dividend = Column(Numeric(12, 2)) 

    __table_args__ = (UniqueConstraint('race_id', 'pool', 'combination', name='_race_pool_combo_uc'),)

Base.metadata.create_all(bind=engine)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class HKJCDividendScraper:
    def __init__(self, max_concurrent_requests=5):
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
            await asyncio.sleep(random.uniform(1.0, 2.5))
            try:
                async with self.session.get(url, timeout=15) as response:
                    if response.status == 404: return None
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                logging.error(f"Fetch error: {url} -> {e}")
                return None

    async def scrape_race_dividends(self, date_str, venue, race_no):
        url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
        html = await self.fetch_html(url)
        if not html: return []

        dividends = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            race_id = f"{date_str.replace('/','')}_{venue}_{race_no:02d}"
            
            div_tab = soup.find('div', class_='dividend_tab')
            if not div_tab: return []
            
            rows = div_tab.find_all('tr')
            current_pool = ""
            
            for row in rows:
                tds = row.find_all('td')
                if not tds or len(tds) < 2 or 'Pool' in tds[0].text or 'Dividend' in tds[0].text:
                    continue
                
                if tds[0].has_attr('rowspan'):
                    current_pool = tds[0].get_text(strip=True).upper()
                    combo = tds[1].get_text(strip=True)
                    div_val = tds[2].get_text(strip=True).replace(',', '')
                else:
                    combo = tds[0].get_text(strip=True)
                    div_val = tds[1].get_text(strip=True).replace(',', '')
                
                try:
                    clean_combo = ",".join([c.strip() for c in combo.split(',')])
                    dividends.append({
                        'race_id': race_id,
                        'pool': current_pool,
                        'combination': clean_combo,
                        'dividend': float(div_val)
                    })
                except ValueError:
                    continue
                    
            return dividends
        except Exception as e:
            logging.error(f"Error parsing {date_str} R{race_no}: {e}")
            return []

def save_to_db(dividends):
    if not dividends: return
    session = SessionLocal()
    try:
        for div in dividends:
            from sqlalchemy.dialects.postgresql import insert
            stmt = insert(RaceDividend).values(div).on_conflict_do_nothing()
            session.execute(stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"DB Error: {e}")
    finally:
        session.close()

async def main():
    scraper = HKJCDividendScraper(max_concurrent_requests=8)

    target_date = sys.argv[1] if len(sys.argv) > 1 else None

    logging.info("Querying database for valid race meetings...")
    with engine.connect() as conn:
        query_str = """
            SELECT r1.race_date, r1.venue, MAX(r1.race_no) as max_r 
            FROM races r1
            WHERE EXISTS (
                SELECT 1 FROM races r2 
                LEFT JOIN race_dividends d ON r2.race_id = d.race_id
                WHERE r2.race_date = r1.race_date AND d.race_id IS NULL
            )
        """
        params = {}

        if target_date:
            query_str += " WHERE r1.race_date = :target_date "
            params['target_date'] = target_date
            logging.info(f"Filtering dividends for specific date: {target_date}")
            
        query_str += " GROUP BY race_date, venue ORDER BY race_date DESC"
        
        meetings = conn.execute(text(query_str), params).fetchall()

    if not meetings:
        logging.warning("No races found in database for the specified date/criteria.")
        return

    logging.info(f"Found {len(meetings)} valid meetings to scrape dividends.")

    for i, (m_date, venue, max_r) in enumerate(meetings):
        date_str = m_date.strftime("%Y/%m/%d")
        logging.info(f"Targeting: {date_str} at {venue} ({max_r} races)...")
        
        tasks = [scraper.scrape_race_dividends(date_str, venue, r) for r in range(1, max_r + 1)]
        results = await asyncio.gather(*tasks)
        
        flat_list = [item for sublist in results for item in sublist]
        if flat_list:
            save_to_db(flat_list)
            logging.info(f"   -> Saved {len(flat_list)} dividend records.")
            
        if i < len(meetings) - 1:
            sleep_time = random.uniform(4.0, 8.0)
            await asyncio.sleep(sleep_time)

    await scraper.close_session()
    logging.info("Dividend Scrape Complete.")

if __name__ == "__main__":
    asyncio.run(main())