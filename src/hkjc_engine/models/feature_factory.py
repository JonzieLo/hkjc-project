import pandas as pd
import numpy as np
import json
import os
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
def calculate_base_margin(win_odds_series):
    try:
        odds = win_odds_series.astype(float)
        if len(odds.dropna()) <= 1:
            return np.zeros(len(odds))
        pi = 1.0 / odds
        overround = pi.sum()
        if overround <= 0 or np.isnan(overround):
            return np.zeros(len(odds))
        z = np.log(overround) / np.log(len(pi.dropna())) + 1.0
        p_fair = pi ** z
        p_fair = p_fair / p_fair.sum()
        p_hurdle = p_fair * 0.825
        if p_hurdle.isnull().any():
            p_hurdle = p_hurdle.fillna(p_hurdle.mean())
        p_clipped = np.clip(p_hurdle, 1e-5, 1 - 1e-5)
        return np.log(p_clipped / (1 - p_clipped))
    except Exception as e:
        return np.zeros(len(win_odds_series))


class HKJCFeatureFactory:
    def __init__(self, db_url, betas_path='mcmc_implementation/model_betas.json'):
        self.engine = create_engine(db_url)
        if os.path.exists(betas_path):
            with open(betas_path, 'r') as f:
                self.betas = json.load(f)
        else:
            self.betas = {}

        self.track_geometry = {
            'ST': {
                'A': {'straight': 430, 'width': 30.5},
                'A+2': {'straight': 430, 'width': 28.5},
                'A+3': {'straight': 430, 'width': 27.5},
                'B': {'straight': 430, 'width': 26.0},
                'B+2': {'straight': 430, 'width': 24.0},
                'C': {'straight': 430, 'width': 21.3},
                'C+3': {'straight': 430, 'width': 18.3},
                'AWT': {'straight': 365, 'width': 22.8},
            },
            'HV': {
                'A': {'straight': 312, 'width': 30.5},
                'A+2': {'straight': 310, 'width': 28.5},
                'B': {'straight': 338, 'width': 26.5},
                'B+2': {'straight': 338, 'width': 24.5},
                'B+3': {'straight': 338, 'width': 23.5},
                'C': {'straight': 334, 'width': 22.5},
                'C+3': {'straight': 335, 'width': 19.5},
            }
        }

    def fetch_raw_data(self, start_date='2018-01-01', end_date='2024-01-01'):
        query = text("""
            WITH CareerCounts AS (
                SELECT 
                    e.race_id, 
                    e.horse_code,
                    ROW_NUMBER() OVER (PARTITION BY e.horse_code ORDER BY r.race_date ASC, r.race_id ASC) as career_run_number
                FROM race_entries e
                JOIN races r ON e.race_id = r.race_id
            )
            SELECT 
                r.race_id, r.race_date, r.venue, r.distance, r.track_condition, r.rail_placement, r.race_class,
                e.horse_code, e.draw, e.actual_weight, e.win_odds, e.finish_position,
                e.ema_early_z, e.ema_mid_z, e.ema_finish_z,
                e.pre_race_mu, e.pre_race_sigma, e.jockey, e.days_since_last_race, 
                e.is_class_drop, e.is_class_rise,
                CASE WHEN cc.career_run_number = 1 THEN 1 ELSE 0 END as is_maiden
            FROM races r
            JOIN race_entries e ON r.race_id = e.race_id
            JOIN CareerCounts cc ON e.race_id = cc.race_id AND e.horse_code = cc.horse_code
            WHERE r.race_date >= :start_date AND r.race_date < :end_date
            AND e.win_odds IS NOT NULL
            AND e.finish_position IS NOT NULL
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})
        return df

    def calculate_residual_targets(self, df):
        df['implied_prob'] = 1.0 / df['win_odds'].astype(float)

        T = 1.15
        df['powered_prob'] = df['implied_prob'] ** T

        race_totals = df.groupby('race_id')['implied_prob'].transform('sum')
        df['P_public'] = df['implied_prob'] / race_totals
        if 'finish_position' in df.columns:
            df['is_winner'] = (df['finish_position'] == 1).astype(float)
        df['residual_target'] = df['is_winner'] - df['P_public']
        return df
    
    def engineer_features(self, df):
        df = self.calculate_residual_targets(df)
        
        def get_jockey_alpha(row):
            venue_dist = f"{row['venue']}_{row['distance']}"
            jockey = row['jockey']
            return self.betas.get(venue_dist, {}).get('jockey_alphas', {}).get(jockey, 0.0)

        df['jockey_alpha'] = df.apply(get_jockey_alpha, axis=1)

        def get_geo(row, key):
            venue = str(row['venue']).upper()
            rail = str(row['rail_placement']).upper().replace(" ", "")
            if 'AWT' in venue or 'ALL WEATHER' in rail:
                return self.track_geometry['ST']['AWT'][key]
            return self.track_geometry.get(venue, {}).get(rail, {}).get(key, 25.0)

        df['track_width'] = df.apply(lambda x: get_geo(x, 'width'), axis=1)
        df['straight_length'] = df.apply(lambda x: get_geo(x, 'straight'), axis=1)
        df['draw_per_meter'] = df['draw'].astype(float) / df['track_width']
    
        group = df.groupby('race_id')
        df['is_winner'] = (df['finish_position'] == 1).astype(int)

        df['weight_delta'] = df['actual_weight'].astype(float) - 120.0
        df['draw'] = df['draw'].astype(float)
        
        df['pre_race_mu'] = df['pre_race_mu'].astype(float).fillna(25.0)
        df['pre_race_sigma'] = df['pre_race_sigma'].astype(float).fillna(8.33)
        
        mu_mean = group['pre_race_mu'].transform('mean')
        sigma_sq_mean = group['pre_race_sigma'].transform(lambda x: (x**2).mean())
        
        # Statistical advantage over the field mean
        df['ts_advantage'] = (df['pre_race_mu'] - mu_mean) / np.sqrt(df['pre_race_sigma']**2 + sigma_sq_mean)

        pace_cols = ['ema_early_z', 'ema_mid_z', 'ema_finish_z']
        for col in pace_cols:
            df[col] = df[col].astype(float).fillna(0.0)
            field_mean = group[col].transform('mean')
            field_std = group[col].transform('std').replace(0, 1.0).fillna(1.0)
            relative_name = f"relative_{col.split('_')[1]}_pace"
            df[relative_name] = (df[col] - field_mean) / field_std

        df['days_since_last_race'] = df['days_since_last_race'].astype(float).fillna(14.0)
        df['is_class_drop'] = df['is_class_drop'].astype(float).fillna(0.0)
        df['is_class_rise'] = df['is_class_rise'].astype(float).fillna(0.0)
         
        df['draw_x_early_pace'] = df['draw'] * df['relative_early_pace']
        df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
        
        df['class_drop_x_ts'] = df['is_class_drop'] * df['ts_advantage']
        df['class_rise_x_ts'] = df['is_class_rise'] * df['ts_advantage']

        return df.dropna(subset=['residual_target'])