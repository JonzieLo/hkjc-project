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
    

def _get_final_section_time(row: pd.Series) -> float:
    for col in ('sec6_time', 'sec5_time', 'sec4_time','sec3_time', 'sec2_time', 'sec1_time'):
        val = row[col]
        if pd.notna(val):
            return float(val)
    return np.nan

def update_historical_run_styles(db_url: str) -> None:
    engine = create_engine(db_url)
    logging.info("Fetching sectional data in chronological order...")
 
    query = """
        SELECT
            e.entry_id,
            e.race_id,
            r.race_date,
            r.race_no,
            e.horse_code,
            e.sec1_time, e.sec2_time, e.sec3_time, e.sec4_time, e.sec5_time, e.sec6_time
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.sec1_time IS NOT NULL
        ORDER BY r.race_date ASC, r.race_no ASC
    """
    df = pd.read_sql(query, engine)
 
    for i in range(1, 7):
        df[f'sec{i}_time'] = pd.to_numeric(df[f'sec{i}_time'], errors='coerce')
 
    df['final_sec_time'] = df.apply(_get_final_section_time, axis=1)
    df['early_rank'] = df.groupby('race_id')['sec1_time'].rank(method='min')
    df['field_size'] = df.groupby('race_id')['sec1_time'].transform('count')
    df['early_pct']  = (df['early_rank'] - 1) / (df['field_size'] - 1).replace(0, 1)
    df['late_rank'] = df.groupby('race_id')['final_sec_time'].rank(method='min')
    df['late_pct']  = (df['late_rank'] - 1) / (df['field_size'] - 1).replace(0, 1)

 
    horse_early_sum: dict[str, float] = {}
    horse_late_sum:  dict[str, float] = {}
    horse_count:     dict[str, int]   = {}
    per_entry_style: list[dict] = []
    grouped = df.groupby('race_id', sort=False)
    logging.info(f"Computing expanding-window run styles over {len(grouped)} races...")
 
    for race_id, race_df in grouped:
        for _, row in race_df.iterrows():
            horse = row['horse_code']
            e_id  = row['entry_id']

            n = horse_count.get(horse, 0)
            if n >= 3:   # require at least 3 prior sectional observations
                ep = horse_early_sum[horse] / n
                lp = horse_late_sum[horse]  / n
                style = classify_run_style(ep, lp)
            else:
                style = 'Unknown'    # not enough history yet
 
            per_entry_style.append({'e_id': e_id, 'style': style})

        for _, row in race_df.iterrows():
            horse = row['horse_code']
            ep    = row['early_pct']
            lp    = row['late_pct']
            if pd.isna(ep) or pd.isna(lp):
                continue
            if horse not in horse_count:
                horse_early_sum[horse] = 0.0
                horse_late_sum[horse]  = 0.0
                horse_count[horse]     = 0
            horse_early_sum[horse] += ep
            horse_late_sum[horse]  += lp
            horse_count[horse]     += 1

    logging.info(f"Writing {len(per_entry_style):,} per-entry run styles to race_entries...")
    with engine.begin() as conn:
        chunk_size = 5000
        for i in range(0, len(per_entry_style), chunk_size):
            chunk = per_entry_style[i:i + chunk_size]
            conn.execute(text("""
                UPDATE race_entries
                SET historical_run_style = :style
                WHERE entry_id = :e_id
            """), chunk)

    final_styles = []
    for horse, n in horse_count.items():
        if n >= 3:
            ep = horse_early_sum[horse] / n
            lp = horse_late_sum[horse]  / n
            style = classify_run_style(ep, lp)
        else:
            style = 'Unknown'
        final_styles.append({'code': horse, 'style': style})
 
    logging.info(f"Writing {len(final_styles):,} career run styles to horses table...")
    with engine.begin() as conn:
        for row in final_styles:
            conn.execute(
                text("UPDATE horses SET historical_run_style = :style WHERE horse_code = :code"),
                {'style': row['style'], 'code': row['code']}
            )
 
    logging.info("Run styles written (temporal leakage eliminated).")
 
 
if __name__ == "__main__":
    update_historical_run_styles()