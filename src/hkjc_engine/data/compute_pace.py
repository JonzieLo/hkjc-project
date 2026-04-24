import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_distance_bucket(distance):
    if distance <= 1200: return 'sprint'
    elif distance <= 1650: return 'middle'
    else: return 'route'

def compute_bucketed_pace_ema(db_url, alpha=0.33):
    engine = create_engine(db_url)
    
    logging.info("Fetching complete racing history for Bucketed Pace calculation...")
    
    query = """
        SELECT e.entry_id, e.race_id, r.race_date, r.distance, e.horse_code, 
               e.sec1_time, e.sec2_time, e.sec3_time, e.sec4_time, e.sec5_time, e.sec6_time
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.finish_position IS NOT NULL
        ORDER BY r.race_date ASC, r.race_no ASC
    """
    df = pd.read_sql(query, engine)

    for i in range(1, 7):
        df[f'sec{i}_time'] = pd.to_numeric(df[f'sec{i}_time'], errors='coerce')

    conditions = [
        (df['distance'] <= 1200),
        (df['distance'].between(1400, 1650)),
        (df['distance'].between(1800, 2000)),
        (df['distance'] >= 2200)
    ]
    
    df['raw_early'] = df['sec1_time']
    df['raw_finish'] = np.select(conditions, [df['sec3_time'], df['sec4_time'], df['sec5_time'], df['sec6_time']], default=np.nan)
    
    choices_mid = [
        df['sec2_time'], 
        df[['sec2_time', 'sec3_time']].mean(axis=1), 
        df[['sec2_time', 'sec3_time', 'sec4_time']].mean(axis=1), 
        df[['sec2_time', 'sec3_time', 'sec4_time', 'sec5_time']].mean(axis=1)
    ]
    df['raw_mid'] = np.select(conditions, choices_mid, default=np.nan)

    phases = ['early', 'mid', 'finish']
    for phase in phases:
        col = f'raw_{phase}'
        z_col = f'race_{phase}_z'
        race_mean = df.groupby('race_id')[col].transform('mean')
        race_std = df.groupby('race_id')[col].transform('std').replace(0, 1.0).fillna(1.0)
        df[z_col] = -1.0 * ((df[col] - race_mean) / race_std)

    horse_ema = {}
    db_updates = []
    
    grouped_races = df.groupby('race_id', sort=False)
    total_races = len(grouped_races)
    
    logging.info(f"Processing Distance-Bucketed EMAs across {total_races} races...")
    
    count = 0
    for race_id, race_df in grouped_races:
        count += 1
        race_dist = race_df['distance'].iloc[0]
        bucket = get_distance_bucket(race_dist)
        
        race_updates = []

        for _, row in race_df.iterrows():
            horse = row['horse_code']
            
            if horse not in horse_ema:
                horse_ema[horse] = {
                    'sprint': {'early': 0.0, 'mid': 0.0, 'finish': 0.0},
                    'middle': {'early': 0.0, 'mid': 0.0, 'finish': 0.0},
                    'route':  {'early': 0.0, 'mid': 0.0, 'finish': 0.0}
                }
                
            current_profile = horse_ema[horse][bucket]
            
            race_updates.append({
                "e_id": row['entry_id'],
                "ema_early": float(current_profile['early']),
                "ema_mid": float(current_profile['mid']),
                "ema_finish": float(current_profile['finish'])
            })
            
        db_updates.extend(race_updates)

        for _, row in race_df.iterrows():
            horse = row['horse_code']
            
            for phase in phases:
                actual_z = row[f'race_{phase}_z']
                
                if pd.notna(actual_z):
                    old_ema = horse_ema[horse][bucket][phase]
                    new_ema = (actual_z * alpha) + (old_ema * (1 - alpha))
                    horse_ema[horse][bucket][phase] = new_ema
                    
        if count % 1000 == 0:
            logging.info(f"Processed {count}/{total_races} races...")

    logging.info("Writing Bucketed Pace Vectors to database...")
    with engine.begin() as conn:
        chunk_size = 5000
        for i in range(0, len(db_updates), chunk_size):
            chunk = db_updates[i:i+chunk_size]
            conn.execute(text("""
                UPDATE race_entries 
                SET ema_early_z = :ema_early, ema_mid_z = :ema_mid, ema_finish_z = :ema_finish
                WHERE entry_id = :e_id
            """), chunk)
            
    logging.info("Successfully engineered all Bucketed Pace Vectors!")

if __name__ == "__main__":
    # DB_URL loaded from hkjc_engine.config
    compute_bucketed_pace_ema(DB_URL, alpha=0.33)