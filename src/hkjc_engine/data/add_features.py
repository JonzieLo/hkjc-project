import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import timedelta
import logging
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def compute_rolling_features(db_url):
    engine = create_engine(db_url)
    
    logging.info("Fetching complete racing history for feature engineering...")
    query = """
        SELECT e.entry_id, e.race_id, r.race_date, e.horse_code, e.jockey, e.finish_position 
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.finish_position IS NOT NULL
        ORDER BY r.race_date ASC, r.race_no ASC
    """
    df = pd.read_sql(query, engine)
    df['race_date'] = pd.to_datetime(df['race_date'])
    
    horse_last_seen = {}
    jockey_history = {}
    
    db_updates = []
    
    grouped_races = df.groupby('race_id', sort=False)
    total_races = len(grouped_races)
    
    logging.info(f"Processing rolling features across {total_races} historical races...")
    
    count = 0
    for race_id, race_df in grouped_races:
        count += 1
        current_date = race_df['race_date'].iloc[0]
        ninety_days_ago = current_date - timedelta(days=90)
        
        race_updates = []
        
        for _, row in race_df.iterrows():
            e_id = row['entry_id']
            horse = row['horse_code']
            jockey = row['jockey']
            
            # --- Days Since Last Race ---
            if horse in horse_last_seen:
                days_rest = (current_date - horse_last_seen[horse]).days
            else:
                days_rest = 21.0  # Default
                
            # --- Rolling 90-Day Jockey Win % ---
            if jockey in jockey_history:
                valid_rides = [w for d, w in jockey_history[jockey] if d >= ninety_days_ago]
                jockey_pct = sum(valid_rides) / len(valid_rides) if valid_rides else 0.08
            else:
                jockey_pct = 0.08
                
            race_updates.append({
                "e_id": e_id,
                "days_rest": float(days_rest),
                "jockey_pct": float(jockey_pct)
            })
            
        db_updates.extend(race_updates)
        
        for _, row in race_df.iterrows():
            horse = row['horse_code']
            jockey = row['jockey']
            is_win = 1 if row['finish_position'] == 1 else 0
            
            horse_last_seen[horse] = current_date
            
            if jockey not in jockey_history:
                jockey_history[jockey] = []
            jockey_history[jockey].append((current_date, is_win))
                        
        if count % 1000 == 0:
            logging.info(f"Engineered {count}/{total_races} races...")

    logging.info("Writing new features to database...")
    with engine.begin() as conn:
        chunk_size = 5000
        for i in range(0, len(db_updates), chunk_size):
            chunk = db_updates[i:i+chunk_size]
            conn.execute(text("""
                UPDATE race_entries 
                SET days_since_last_race = :days_rest, 
                    jockey_win_pct = :jockey_pct
                WHERE entry_id = :e_id
            """), chunk)
            
    logging.info("Successfully mapped rolling Time-Series features.")

if __name__ == "__main__":
    compute_rolling_features(DB_URL)