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

async def fetch_finish_position(session, semaphore, race_id, race_date, venue, race_no):
    date_str = str(race_date).replace('-', '/')
    url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
    
    async with semaphore:
        await asyncio.sleep(random.uniform(1.0, 2.5)) # jitter
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

                race_results_pos = []
                
                thead = target_table.find('thead')
                if not thead:
                    return race_id, None
                    
                header_row = thead.find('tr')
                header_cells = header_row.find_all('td')
                
                pla_col_idx = -1
                horse_col_idx = -1
                
                for idx, cell in enumerate(header_cells):
                    text_val = cell.get_text().strip()
                    if "Pla." in text_val:
                        pla_col_idx = idx
                    elif horse_col_idx == -1 and "Horse" in text_val and "No" not in text_val:
                        horse_col_idx = idx
                        
                if pla_col_idx == -1 or horse_col_idx == -1:
                    return race_id, None

                tbody = target_table.find('tbody')
                if not tbody:
                    rows = target_table.find_all('tr')[1:]
                else:
                    rows = tbody.find_all('tr')
                
                valid_horses = 0
                for row in rows:
                    cells = row.find_all('td')
                    
                    if len(cells) > max(pla_col_idx, horse_col_idx):
                        try:
                            horse_text = cells[horse_col_idx].get_text(separator=" ", strip=True)
                            
                            # 4-character horse code (e.g. "J123")
                            match = re.search(r'\(([A-Z0-9]{4})\)', horse_text)
                            if match:
                                horse_code = match.group(1)
                            else:
                                # FALLBACK: Extract directly from the URL link
                                a_tag = cells[horse_col_idx].find('a')
                                if a_tag and 'horseid=' in a_tag.get('href', ''):
                                    fallback_match = re.search(r'_([A-Z0-9]{4})', a_tag['href'])
                                    if fallback_match:
                                        horse_code = fallback_match.group(1)
                                    else:
                                        continue
                                else:
                                    continue
                                
                            raw_pla = cells[pla_col_idx].get_text(strip=True)
                            
                            # Only update if the horse actually finished (ignore 'WX', 'DNF', 'PU')
                            if raw_pla.isdigit():
                                finish_position = int(raw_pla)
                                
                                race_results_pos.append({
                                    "horse_code": horse_code,
                                    "finish_position": finish_position
                                })
                                valid_horses += 1
                        except Exception:
                            continue 
                
                if valid_horses == 0:
                    logging.warning(f"[{race_id}] Table parsed, but 0 valid finish positions extracted.")
                    return race_id, None
                    
                return race_id, race_results_pos
                
        except Exception as e:
            logging.error(f"Failed to fetch finish positions for {race_id}: {e}")
            return race_id, 'Error'

async def main(target_date=None):
    query_str = """
        SELECT DISTINCT r.race_id, r.race_date, r.venue, r.race_no 
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.finish_position IS NULL
    """
    
    if target_date:
        query_str += " AND r.race_date = :target_date"
        
    with engine.connect() as conn:
        result = conn.execute(text(query_str), {"target_date": target_date})
        missing = result.fetchall()
        
    if not missing:
        logging.info("All races already have a Finish Position assigned. Nothing to patch!")
        return
        
    logging.info(f"Found {len(missing)} races missing Finish Position data. Initiating patch...")
    
    semaphore = asyncio.Semaphore(5)
    chunk_size = 50
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'}) as session:
        for i in range(0, len(missing), chunk_size):
            chunk = missing[i:i+chunk_size]
            tasks = [fetch_finish_position(session, semaphore, r.race_id, r.race_date, r.venue, r.race_no) for r in chunk]
            results = await asyncio.gather(*tasks)
            
            with engine.begin() as conn:
                for race_id, pos_list in results:
                    if pos_list and pos_list != 'Error':
                        for entry in pos_list:
                            conn.execute(text("""
                                UPDATE race_entries 
                                SET finish_position = :pos 
                                WHERE race_id = :r_id AND horse_code = :h_code
                            """), {
                                "pos": entry['finish_position'], 
                                "r_id": race_id, 
                                "h_code": entry['horse_code']
                            })
            
            processed = min(i + chunk_size, len(missing))
            logging.info(f"Patched batch {(i // chunk_size) + 1}... ({processed}/{len(missing)} races)")

            if processed < len(missing):
                sleep_time = random.uniform(3.0, 7.0)
                logging.info(f"Resting for {sleep_time:.1f}s before next batch...")
                await asyncio.sleep(sleep_time)

    logging.info("Database patch complete. Finish Positions are fully updated.")

if __name__ == "__main__":
    import sys
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(target_date))