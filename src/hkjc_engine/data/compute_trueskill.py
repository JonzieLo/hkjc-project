import pandas as pd
import trueskill
from sqlalchemy import create_engine, text
import logging
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def compute_historical_trueskill(db_url):
    engine = create_engine(db_url)
    env = trueskill.TrueSkill(mu=25.0, sigma=8.33, beta=4.16, tau=0.08, draw_probability=0.0)

    horse_ratings = {}
    
    logging.info("Fetching races in strict chronological order...")
    with engine.connect() as conn:
        races_df = pd.read_sql("""
            SELECT race_id, race_date 
            FROM races 
            ORDER BY race_date ASC, race_no ASC
        """, conn)
    
    race_ids = races_df['race_id'].tolist()
    total_races = len(race_ids)
    
    db_updates = []
    
    logging.info(f"Processing {total_races} races...")
    
    with engine.connect() as conn:
        for idx, r_id in enumerate(race_ids):
            entries_df = pd.read_sql(f"""
                SELECT entry_id, horse_code, finish_position 
                FROM race_entries 
                WHERE race_id = '{r_id}' AND finish_position IS NOT NULL
                ORDER BY finish_position ASC
            """, conn)
            
            if entries_df.empty:
                continue
            
            match_ratings = []
            ranks = []
            entry_ids = []
            
            for _, row in entries_df.iterrows():
                h_code = row['horse_code']
                pos = int(row['finish_position'])
                e_id = row['entry_id']
                if h_code not in horse_ratings:
                    horse_ratings[h_code] = env.create_rating()
                    
                current_rating = horse_ratings[h_code]
                match_ratings.append((current_rating,)) 
                ranks.append(pos)
                entry_ids.append(e_id)
                
                db_updates.append({
                    "e_id": e_id,
                    "mu": float(current_rating.mu),
                    "sigma": float(current_rating.sigma)
                })
            
            try:
                new_ratings = env.rate(match_ratings, ranks=ranks)
            except ValueError as e:
                logging.error(f"Error rating race {r_id}: {e}")
                continue
            
            for i, h_code in enumerate(entries_df['horse_code']):
                horse_ratings[h_code] = new_ratings[i][0]
                
            if (idx + 1) % 500 == 0:
                logging.info(f"Processed {idx + 1}/{total_races} races...")

    logging.info("Writing Pre-Race Latent States to database...")
    with engine.begin() as conn:
        chunk_size = 5000
        for i in range(0, len(db_updates), chunk_size):
            chunk = db_updates[i:i+chunk_size]
            conn.execute(text("""
                UPDATE race_entries 
                SET pre_race_mu = :mu, pre_race_sigma = :sigma 
                WHERE entry_id = :e_id
            """), chunk)
            
    logging.info("Successfully generated and saved Latent States (TrueSkill).")

if __name__ == "__main__":
    compute_historical_trueskill(DB_URL)