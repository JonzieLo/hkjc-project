import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def classify_run_style(early_pct, late_pct):
    if pd.isna(early_pct) or pd.isna(late_pct): return 'Unknown'
    
    if early_pct <= 0.20:
        return 'Front-Runner'
    elif early_pct <= 0.60:
        return 'Stalker'
    elif early_pct > 0.60 and late_pct <= 0.30:
        return 'Closer'
    else:
        return 'Backmarker' # Slow early, slow late.

def update_historical_run_styles(db_url):
    engine = create_engine(db_url)
    logging.info("Fetching comprehensive sectional data...")
    
    query = """
        SELECT e.race_id, e.horse_code, 
               e.sec1_time, e.sec2_time, e.sec3_time, e.sec4_time, e.sec5_time, e.sec6_time
        FROM race_entries e
        WHERE e.sec1_time IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    
    df['early_rank'] = df.groupby('race_id')['sec1_time'].rank(method='min')
    df['field_size'] = df.groupby('race_id')['sec1_time'].transform('count')
    df['early_pct'] = (df['early_rank'] - 1) / (df['field_size'] - 1).replace(0, 1)
    
    df['final_sec_time'] = df[['sec6_time', 'sec5_time', 'sec4_time', 'sec3_time', 'sec2_time', 'sec1_time']].bfill(axis=1).iloc[:, 0]
    
    df['late_rank'] = df.groupby('race_id')['final_sec_time'].rank(method='min')
    df['late_pct'] = (df['late_rank'] - 1) / (df['field_size'] - 1).replace(0, 1)
    
    career_stats = df.groupby('horse_code')[['early_pct', 'late_pct']].mean().reset_index()
    
    career_stats['run_style'] = career_stats.apply(lambda row: classify_run_style(row['early_pct'], row['late_pct']), axis=1)
    
    logging.info("Pushing run styles to PostgreSQL...")
    with engine.begin() as conn:
        for _, row in career_stats.iterrows():
            update_sql = text("UPDATE horses SET historical_run_style = :style WHERE horse_code = :code")
            conn.execute(update_sql, {"style": row['run_style'], "code": row['horse_code']})
            
    logging.info("Database updated with Run Styles.")

if __name__ == "__main__":
    update_historical_run_styles(DB_URL)