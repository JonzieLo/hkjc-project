import asyncio
import aiohttp
import random
import re
import sys
import logging
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime
from hkjc_engine.data.schema import engine, SessionLocal, Base, Race, Horse, RaceEntry
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("scrape_log.log"), logging.StreamHandler()]
)

class HKJCAsyncScraper:
    def __init__(self, max_concurrent_requests=2):
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://racing.hkjc.com/'
        }
        self.horse_cache = {}

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
                logging.info(f"  [Network] Requesting URL: {url}")
                async with self.session.get(url, timeout=15) as response:
                    logging.info(f"  [Network] Status {response.status} received for {url}")
                    if response.status == 200:
                        return await response.text()
                    else:
                        logging.warning(f"Non-200 status: {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logging.error(f"Timeout reaching {url}")
                return None
            except Exception as e:
                logging.error(f"Network Error for {url}: {e}")
                return None
    
    async def get_valid_race_dates(self, start_year=2010, end_year=2026):
        valid_dates = set()
        logging.info("--- Mapping race dates ---")
        for year in range(start_year, end_year + 1):
            url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={year}/01/01"
            html = await self.fetch_html(url)
            
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                select = soup.find('select', id='selectId')
                if select:
                    for option in select.find_all('option'):
                        raw_date = option['value']

                        d_obj = datetime.strptime(raw_date, "%d/%m/%Y")
                        formatted_date = d_obj.strftime("%Y/%m/%d")

                        if start_year <= d_obj.year <= end_year:
                            valid_dates.add(formatted_date)

        sorted_dates = sorted(list(valid_dates))
        logging.info(f"Discovery complete. Found exactly {len(sorted_dates)} valid race days.")
        return sorted_dates
            
    async def get_horse_profile(self, horse_code, full_horse_id):
        if horse_code in self.horse_cache:
            return self.horse_cache[horse_code]

        url = f"https://racing.hkjc.com/en-us/local/information/horse?horseid={full_horse_id}"
        html = await self.fetch_html(url)
        
        profile_data = {
            'sire': 'Unknown', 'dam': 'Unknown', 
            'origin': 'UNK', 'sex': 'Unknown', 'import_type': 'UNK'
        }
        
        if not html:
            self.horse_cache[horse_code] = profile_data
            return profile_data
            
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for tr in soup.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) >= 3:
                    label = tds[0].get_text(strip=True)
                    val = tds[2].get_text(strip=True)
                    
                    if label == 'Sire':
                        profile_data['sire'] = val
                    elif label == 'Dam':
                        profile_data['dam'] = val
                    elif label == 'Country of Origin / Age':
                        profile_data['origin'] = val.split('/')[0].strip() if '/' in val else val
                    elif label == 'Colour / Sex':
                        profile_data['sex'] = val.split('/')[1].strip() if '/' in val else val
                    elif label == 'Import Type':
                        profile_data['import_type'] = val
            
            self.horse_cache[horse_code] = profile_data
            return profile_data
            
        except Exception as e:
            logging.error(f"Profile error for {horse_code}: {e}")
            return profile_data
        
    async def get_sectional_times(self, date_str, race_no):
        d_obj = datetime.strptime(date_str, "%Y/%m/%d")
        sec_date = d_obj.strftime("%d/%m/%Y")
        
        url = f"https://racing.hkjc.com/en-us/local/information/displaysectionaltime?racedate={sec_date}&RaceNo={race_no}"
        html = await self.fetch_html(url)
        if not html: return None
            
        try:
            tables = pd.read_html(StringIO(html), flavor='html5lib')
            if not tables: return None
                
            df = None
            for table in tables:
                if isinstance(table.columns, pd.MultiIndex):
                    new_cols = []
                    for col in table.columns:
                        col_name = col[-1]
                        new_cols.append(str(col_name).strip().lower().replace(' ', '_'))
                    table.columns = new_cols
                else:
                    table.columns = [str(col).strip().lower().replace(' ', '_') for col in table.columns]
                
                if 'horse_no.' in table.columns:
                    df = table
                    break
            
            if df is None: return None

            df = df.dropna(axis=1, how='all')
            sec_cols = [c for c in df.columns if 'sec' in c]
            
            result_df = pd.DataFrame()
            result_df['horse_no.'] = df['horse_no.']
            
            def extract_sec_time(val):
                if pd.isna(val): return None
                match = re.search(r'(\d{1,2}:\d{2}\.\d{2}|\d{2}\.\d{2})', str(val))
                return match.group(1) if match else None

            for i in range(6):
                col_name = f'sec{i+1}'
                if i < len(sec_cols):
                    sec_data = df[sec_cols[i]]
                    if isinstance(sec_data, pd.DataFrame):
                        sec_data = sec_data.iloc[:, 0]
                    result_df[col_name] = sec_data.apply(extract_sec_time)
                else:
                    result_df[col_name] = None
                    
            return result_df
            
        except Exception as e:
            logging.error(f"Sectional parsing error: {str(e)[:100]}")
            return None

    async def get_historical_results(self, date_str, venue, race_no):
        url = f"https://racing.hkjc.com/en-us/local/information/localresults?racedate={date_str}&Racecourse={venue}&RaceNo={race_no}"
        html = await self.fetch_html(url)
        if not html: return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')

            metadata = {'distance': 0, 'track_condition': 'UNKNOWN', 'rail_placement': 'UNK', 'race_class': 'Other'}
            race_tab = soup.find('div', class_='race_tab')
            if race_tab:
                text_content = race_tab.get_text(separator=' ', strip=True)

                class_match = re.search(r'(Class\s\d|Group\s\d|Gr\.\d)', text_content, re.IGNORECASE)
                if class_match:
                    metadata['race_class'] = class_match.group(1).title()

                dist_match = re.search(r'(\d+)M', text_content)
                if dist_match: metadata['distance'] = int(dist_match.group(1))
                
                going_match = re.search(r'Going\s*:\s*(.*?)(?:Course|Time|$)', text_content)
                if going_match: metadata['track_condition'] = going_match.group(1).strip()[:50]
                
                course_match = re.search(r'Course\s*:\s*TURF - "(.*?)" Course', text_content)
                if course_match: metadata['rail_placement'] = course_match.group(1).strip()
                elif 'AWT' in text_content or 'ALL WEATHER' in text_content.upper(): metadata['rail_placement'] = 'AWT'

            perf_div = soup.find('div', class_='performance')
            if not perf_div: return None

            tables = pd.read_html(StringIO(str(perf_div)), flavor='html5lib')
            if not tables: return None
            df = tables[0]
            df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
            if 'running_position' in df.columns: df = df.drop(columns=['running_position'])
            df = df.map(lambda x: " ".join(str(x).split()) if isinstance(x, str) else x)

            horse_links = {}
            for a_tag in perf_div.find_all('a', href=True):
                if 'horseid=' in a_tag['href']:
                    full_id = a_tag['href'].split('horseid=')[1]
                    short_code = full_id.split('_')[-1]
                    horse_links[short_code] = full_id

            df['race_id'] = f"{date_str.replace('/','')}_{venue}_{race_no:02d}"
            df['race_date'] = date_str
            df['distance'] = metadata['distance']
            df['track_condition'] = metadata['track_condition']
            df['rail_placement'] = metadata['rail_placement']
            df['race_class'] = metadata['race_class']
            
            return df, horse_links
            
        except Exception as e:
            logging.error(f"Parsing error Race {race_no}: {str(e)[:100]}")
            return None

def parse_time(time_str):
    try:
        if ':' in time_str:
            m, s = time_str.split(':')
            return float(m) * 60 + float(s)
        return float(time_str)
    except: return None

def extract_horse_code(horse_str):
    match = re.search(r'\((.*?)\)', horse_str)
    return match.group(1) if match else horse_str

def save_to_db(df, horse_profiles):
    if df.empty:
        return

    unique_horses = df.drop_duplicates(subset=['horse'])
    horse_records = []
    for _, row in unique_horses.iterrows():
        h_code = extract_horse_code(row['horse'])
        profile = horse_profiles.get(h_code, {
            'sire': 'Unknown', 'dam': 'Unknown', 
            'origin': 'UNK', 'sex': 'Unknown', 'import_type': 'UNK'
        })
        horse_records.append({
            'horse_code': h_code,
            'horse_name': row['horse'].split('(')[0].strip(),
            'sire': profile.get('sire'),
            'dam': profile.get('dam'),
            'origin': profile.get('origin'),
            'sex': profile.get('sex'),
            'import_type': profile.get('import_type'),
            'historical_run_style': 'Unknown'
        })

    unique_races = df.drop_duplicates(subset=['race_id'])
    race_records = []
    for _, row in unique_races.iterrows():
        r_id = row['race_id']
        race_records.append({
            'race_id': r_id,
            'race_date': datetime.strptime(row['race_date'], "%Y/%m/%d").date(),
            'race_class': row['race_class'],
            'venue': r_id.split('_')[1],
            'race_no': int(r_id.split('_')[2]),
            'distance': int(row['distance']),
            'track_condition': row['track_condition'],
            'rail_placement': row['rail_placement']
        })

    entries_df = pd.DataFrame()
    entries_df['race_id'] = df['race_id']
    entries_df['horse_code'] = df['horse'].apply(extract_horse_code)
    entries_df['jockey'] = df['jockey']
    entries_df['draw'] = pd.to_numeric(df.get('dr.', ''), errors='coerce')
    entries_df['actual_weight'] = pd.to_numeric(df.get('act._wt.', ''), errors='coerce')
    entries_df['final_time'] = df['finish_time'].apply(parse_time)

    for i in range(1, 7):
        col_name = f'sec{i}'
        entries_df[f'sec{i}_time'] = df.get(col_name, pd.Series(dtype=object)).apply(parse_time)
        
    entries_df['finish_position'] = pd.to_numeric(df.get('pla.', ''), errors='coerce')

    try:
        with engine.begin() as conn:

            if horse_records:
                stmt_horses = insert(Horse).values(horse_records)
                stmt_horses = stmt_horses.on_conflict_do_nothing(index_elements=['horse_code'])
                conn.execute(stmt_horses)
            
            if race_records:
                stmt_races = insert(Race).values(race_records)
                stmt_races = stmt_races.on_conflict_do_nothing(index_elements=['race_id'])
                conn.execute(stmt_races)

            race_ids_list = [r['race_id'] for r in race_records]
            conn.execute(
                text("DELETE FROM race_entries WHERE race_id = ANY(:rids)"), 
                {"rids": race_ids_list}
            )

            # D. Bulk Insert Entries via Pandas to_sql
            entries_df.to_sql(
                'race_entries', 
                conn, 
                if_exists='append', 
                index=False, 
                method='multi',
                chunksize=1000
            )
            
    except Exception as e:
        logging.error(f"Bulk DB Load Error: {e}")

def reset_database():
    session = SessionLocal()
    try:
        logging.info("Starting database reset...")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logging.info("Database successfully wiped clean.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error during reset: {e}")
    finally:
        session.close()

async def process_race_day(scraper, racedate, venue):
    logging.info(f"Probing {racedate} at {venue} (Race 1)...")

    probe_result = await scraper.get_historical_results(racedate, venue, 1)
    if probe_result is None:
        return False
        
    race_tasks = [scraper.get_historical_results(racedate, venue, r_no) for r_no in range(2, 13)]
    race_results = await asyncio.gather(*race_tasks)
    
    valid_races = [probe_result] + [res for res in race_results if res is not None]
    
    all_dfs = []
    all_horse_links = {}
    for df, links in valid_races:
        all_horse_links.update(links)

        race_no = int(df['race_id'].iloc[0].split('_')[2])
        sec_df = await scraper.get_sectional_times(racedate, race_no)
        
        if sec_df is not None:
            df['horse_no.'] = df['horse_no.'].astype(str)
            sec_df['horse_no.'] = sec_df['horse_no.'].astype(str)
            df = pd.merge(df, sec_df, on='horse_no.', how='left')
        
        all_dfs.append(df)

    from sqlalchemy import text
    with engine.connect() as conn:
        existing_result = conn.execute(text("SELECT horse_code FROM horses"))
        known_horses = {row[0] for row in existing_result}

    new_horses_to_scrape = {}
    for code, full_id in all_horse_links.items():
        if code not in known_horses:
            new_horses_to_scrape[code] = full_id

    if new_horses_to_scrape:
        logging.info(f"Found {len(new_horses_to_scrape)} new horses making their debut. Scraping profiles...")
        profile_tasks = [scraper.get_horse_profile(code, f_id) for code, f_id in new_horses_to_scrape.items()]
        profiles = await asyncio.gather(*profile_tasks)
        horse_profiles_dict = dict(zip(new_horses_to_scrape.keys(), profiles))
    else:
        logging.info("All horses running today are already in the DB. Skipping profile scrape!")
        horse_profiles_dict = {}

    master_df = pd.concat(all_dfs, ignore_index=True)
    save_to_db(master_df, horse_profiles_dict)
    logging.info(f"--- Loaded {racedate} {venue} into Database ---")
    
    return True

async def main(target_date_str=None):
    # reset_database() 
    scraper = HKJCAsyncScraper(max_concurrent_requests=5)
    
    if target_date_str is None and len(sys.argv) > 1:
        target_date_str = sys.argv[1]
        
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            all_dates = [target_date]
            logging.info(f"Targeting specific date: {target_date_str}")
        except ValueError:
            logging.error(f"Invalid date format: {target_date_str}. Use YYYY-MM-DD.")
            return
    else:
        start_date = datetime(2010, 1, 1)
        end_date = datetime.now()
        all_dates = pd.date_range(start_date, end_date, freq='D')
        logging.info("No date argument found. Running full historical range.")
    valid_dates = [d for d in all_dates if d.month != 8] 
    venues = ["ST", "HV"]
    
    total_dates = len(valid_dates)
    for i, racedate in enumerate(valid_dates):
        date_str = racedate.strftime("%Y/%m/%d")
        for venue in venues:
            await process_race_day(scraper, date_str, venue)

        if i < total_dates - 1:
            sleep_time = random.uniform(5.0, 10.0)
            logging.info(f"Day complete. Resting for {sleep_time:.1f}s before next date...")
            await asyncio.sleep(sleep_time)

    await scraper.close_session()
    logging.info("--- Pipeline Completed ---")

if __name__ == "__main__":
    asyncio.run(main())