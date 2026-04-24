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

async def fetch_odds(session, semaphore, race_id, race_date, venue, race_no):
    date_str = str(race_date).replace('-', '/')
    url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
    
    async with semaphore:
        await asyncio.sleep(random.uniform(1.0, 2.5))
        try:
            async with session.get(url, timeout=20) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                performance_div = soup.find('div', class_='performance')
                if not performance_div:
                    return race_id, None
                    
                target_table = performance_div.find('table', class_='table_bd')
                if not target_table:
                    return race_id, None

                race_results_odds = []
                
                thead = target_table.find('thead')
                if not thead:
                    return race_id, None
                    
                header_row = thead.find('tr')
                header_cells = header_row.find_all('td')
                
                odds_col_idx = -1
                horse_col_idx = -1
                
                for idx, cell in enumerate(header_cells):
                    text_val = cell.get_text().strip()
                    
                    # Look for either 'Win Odds' or 'Odds'
                    if "Win Odds" in text_val or text_val == "Odds":
                        odds_col_idx = idx
                    elif horse_col_idx == -1 and "Horse" in text_val and "No" not in text_val:
                        horse_col_idx = idx
                        
                if odds_col_idx == -1 or horse_col_idx == -1:
                    logging.warning(f"[{race_id}] Could not find column headers. Found cols: {[c.get_text().strip() for c in header_cells]}")
                    return race_id, None

                tbody = target_table.find('tbody')
                if not tbody:
                    rows = target_table.find_all('tr')[1:]
                else:
                    rows = tbody.find_all('tr')
                
                valid_horses = 0
                for row in rows:
                    cells = row.find_all('td')
                    
                    if len(cells) > max(odds_col_idx, horse_col_idx):
                        try:
                            horse_text = cells[horse_col_idx].get_text(separator=" ", strip=True)
                            match = re.search(r'\(([A-Z][0-9]{3})\)', horse_text)
                            
                            if match:
                                horse_code = match.group(1)
                            else:
                                a_tag = cells[horse_col_idx].find('a')
                                if a_tag and 'horseid=' in a_tag.get('href', ''):
                                    fallback_match = re.search(r'_([A-Z][0-9]{3})', a_tag['href'])
                                    if fallback_match:
                                        horse_code = fallback_match.group(1)
                                    else:
                                        continue
                                else:
                                    continue
                                
                            raw_odds = cells[odds_col_idx].get_text(strip=True)
                            
                            # Scrub non-numeric characters from odds
                            odds_match = re.search(r'([0-9.]+)', raw_odds)
                            if not odds_match:
                                continue
                                
                            win_odds = float(odds_match.group(1))
                            
                            race_results_odds.append({
                                "horse_code": horse_code,
                                "win_odds": win_odds
                            })
                            valid_horses += 1
                        except Exception:
                            continue 
                
                if valid_horses == 0:
                    logging.warning(f"[{race_id}] Table parsed, but 0 valid odds extracted.")
                    return race_id, None
                    
                sample = race_results_odds[0]
                logging.info(f"[{race_id}] Success! Found {valid_horses} odds. (Sample -> Horse {sample['horse_code']}: {sample['win_odds']})")
                return race_id, race_results_odds
                
        except Exception as e:
            logging.error(f"Failed to fetch odds for {race_id}: {e}")
            return race_id, 'Error'

async def main(target_date=None):
    query_str = """
        SELECT DISTINCT r.race_id, r.race_date, r.venue, r.race_no 
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.win_odds IS NULL AND e.finish_position IS NOT NULL
    """
    
    if target_date:
        query_str += " AND r.race_date = :target_date"
        
    with engine.connect() as conn:
        result = conn.execute(text(query_str), {"target_date": target_date})
        missing = result.fetchall()
        
    if not missing:
        logging.info("All finished horses have Odds assigned. Nothing to patch!")
        return
        
    logging.info(f"Found {len(missing)} races missing Odds data. Initiating patch...")
    
    semaphore = asyncio.Semaphore(5)
    chunk_size = 50
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i:i+chunk_size]
            tasks = [fetch_odds(session, semaphore, r.race_id, r.race_date, r.venue, r.race_no) for r in chunk]
            results = await asyncio.gather(*tasks)
            
            batch_updates = 0
            with engine.begin() as conn:
                for race_id, odds_list in results:
                    if odds_list and odds_list != 'Error':
                        for entry in odds_list:
                            conn.execute(text("""
                                UPDATE race_entries 
                                SET win_odds = :odds 
                                WHERE race_id = :r_id AND horse_code = :h_code
                            """), {
                                "odds": entry['win_odds'], 
                                "r_id": race_id, 
                                "h_code": entry['horse_code']
                            })
                            batch_updates += 1
            
            processed = min(i + chunk_size, len(missing))
            logging.info(f"Patched batch {(i // chunk_size) + 1}... ({processed}/{len(missing)} races) | Successfully wrote {batch_updates} odds to DB!")

            if processed < len(missing):
                sleep_time = random.uniform(3.0, 7.0)
                logging.info(f"Resting for {sleep_time:.1f}s before next batch...")
                await asyncio.sleep(sleep_time)

    logging.info("Database odds patch complete.")

if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(target_date))