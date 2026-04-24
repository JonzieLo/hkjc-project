import pandas as pd
from sqlalchemy import create_engine, text
import logging
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_class_level(class_str):
    """Converts HKJC string classes into a numeric hierarchy."""
    if not isinstance(class_str, str): return 99
    s = class_str.upper()
    if 'GROUP 1' in s or 'G1' in s: return 1
    if 'GROUP 2' in s or 'G2' in s: return 2
    if 'GROUP 3' in s or 'G3' in s: return 3
    if 'CLASS 1' in s: return 4
    if 'CLASS 2' in s: return 5
    if 'CLASS 3' in s: return 6
    if 'CLASS 4' in s: return 7
    if 'CLASS 5' in s: return 8
    if 'GRIFFIN' in s: return 9  # Restricted young horses
    return 99

def is_wet_track(cond_str):
    """Flags anomalous/wet tracks using your parsed HKJC dictionary."""
    if not isinstance(cond_str, str): return False
    s = cond_str.upper()
    # Exclude "GOOD TO YIELDING"
    wet_keywords = ['YIELDING', 'SOFT', 'HEAVY', 'WET', 'SLOW', 'RAIN']
    if 'GOOD TO YIELDING' in s: return False
    return any(k in s for k in wet_keywords)

def compute_advanced_features(db_url):
    engine = create_engine(db_url)
    
    logging.info("Fetching complete racing history for Class & Track processing...")
    query = """
        SELECT e.entry_id, e.race_id, r.race_date, e.horse_code, 
               e.finish_position, r.race_class, r.track_condition
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.finish_position IS NOT NULL
        ORDER BY r.race_date ASC, r.race_no ASC
    """
    df = pd.read_sql(query, engine)
    
    # State Trackers
    horse_last_class = {} # horse_code -> integer class level
    horse_wet_form = {}   # horse_code -> {'wet_runs': 0, 'wet_wins': 0}
    
    db_updates = []
    grouped_races = df.groupby('race_id', sort=False)
    
    logging.info(f"Processing {len(grouped_races)} chronological races...")
    
    count = 0
    for race_id, race_df in grouped_races:
        count += 1
        
        # Determine Race Meta
        race_class_str = race_df['race_class'].iloc[0]
        current_class_lvl = get_class_level(race_class_str)
        is_wet_today = is_wet_track(race_df['track_condition'].iloc[0])
        
        race_updates = []
        for _, row in race_df.iterrows():
            horse = row['horse_code']
            
            # --- Class Movement ---
            last_class_lvl = horse_last_class.get(horse, current_class_lvl)
            is_drop = 1 if current_class_lvl > last_class_lvl else 0
            is_rise = 1 if current_class_lvl < last_class_lvl else 0
            
            # --- Wet Track Affinity ---
            wet_pct = 0.0
            if horse in horse_wet_form and horse_wet_form[horse]['wet_runs'] > 0:
                wet_pct = horse_wet_form[horse]['wet_wins'] / horse_wet_form[horse]['wet_runs']
                
            race_updates.append({
                "e_id": row['entry_id'],
                "drop": is_drop,
                "rise": is_rise,
                "wet_pct": float(wet_pct)
            })
            
        db_updates.extend(race_updates)

        for _, row in race_df.iterrows():
            horse = row['horse_code']
            is_win = 1 if row['finish_position'] == 1 else 0
            
            # Update Class
            if current_class_lvl != 99:
                horse_last_class[horse] = current_class_lvl
                
            # Update Wet Form (Only if today is wet)
            if is_wet_today:
                if horse not in horse_wet_form:
                    horse_wet_form[horse] = {'wet_runs': 0, 'wet_wins': 0}
                horse_wet_form[horse]['wet_runs'] += 1
                horse_wet_form[horse]['wet_wins'] += is_win
                
        if count % 1000 == 0:
            logging.info(f"Processed {count} races...")

    logging.info("Writing advanced features to database...")
    with engine.begin() as conn:
        chunk_size = 5000
        for i in range(0, len(db_updates), chunk_size):
            chunk = db_updates[i:i+chunk_size]
            conn.execute(text("""
                UPDATE race_entries 
                SET is_class_drop = :drop, 
                    is_class_rise = :rise, 
                    wet_win_pct = :wet_pct
                WHERE entry_id = :e_id
            """), chunk)
            
    logging.info("Class Drops and Wet Track Affinity successfully mapped!")

if __name__ == "__main__":
    compute_advanced_features(DB_URL)