import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sqlalchemy import text

from hkjc_engine.config import artifact
import hkjc_engine.models.ensemble  # noqa: F401
from hkjc_engine.models.feature_factory import HKJCFeatureFactory, calculate_base_margin

logging.basicConfig(level=logging.INFO, format='%(message)s')

FEATURES_A = [
    'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
    'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
    'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length',
]

def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

class LiveRacePredictor:
    def __init__(self, db_url,
                 model_a_win_path=None,
                 model_a_pla_path=None,
                 model_b_path=None,
                 calib_a_win_path=None,
                 calib_a_pla_path=None,
                 calib_b_path=None,
                 stacker_win_path=None,
                 stacker_pla_path=None,
                 stacker_exo_path=None):
        self.factory      = HKJCFeatureFactory(db_url)
        
        # Safely wrap paths in artifact() if not explicitly provided
        self.model_a_win  = joblib.load(model_a_win_path or artifact('wf_model_a.pkl'))
        self.model_a_pla  = joblib.load(model_a_pla_path or artifact('wf_model_a_pla.pkl'))
        self.model_b      = joblib.load(model_b_path or artifact('wf_model_b.pkl'))
        
        self.calibrator_a_win = joblib.load(calib_a_win_path or artifact('wf_calib_a.pkl'))
        self.calibrator_a_pla = joblib.load(calib_a_pla_path or artifact('wf_calib_a_pla.pkl'))
        self.calibrator_b     = joblib.load(calib_b_path or artifact('wf_calib_b.pkl'))
        
        self.stacker_win = joblib.load(stacker_win_path or artifact('wf_stacker_win.pkl'))
        self.stacker_pla = joblib.load(stacker_pla_path or artifact('wf_stacker_pla.pkl'))
        self.stacker_exo = joblib.load(stacker_exo_path or artifact('wf_stacker_exo.pkl'))
        
        self.features_a   = FEATURES_A

    @staticmethod
    def _class_level(class_str):
        if not isinstance(class_str, str): return 99
        s = str(class_str).upper()
        if 'GROUP 1' in s or 'G1' in s: return 1
        if 'GROUP 2' in s or 'G2' in s: return 2
        if 'GROUP 3' in s or 'G3' in s: return 3
        if 'CLASS 1' in s: return 4
        if 'CLASS 2' in s: return 5
        if 'CLASS 3' in s: return 6
        if 'CLASS 4' in s: return 7
        if 'CLASS 5' in s: return 8
        if 'GRIFFIN' in s: return 9
        return 99

    def _fetch_historical_states(self, horse_codes):
        if not horse_codes: return {}
        query = text("""
            WITH RankedRaces AS (
                SELECT e.horse_code, e.ema_early_z, e.ema_mid_z, e.ema_finish_z, e.pre_race_mu, e.pre_race_sigma,
                       r.race_class AS last_race_class, r.race_date  AS last_race_date,
                       ROW_NUMBER() OVER(PARTITION BY e.horse_code ORDER BY r.race_date DESC, r.race_no DESC) AS rn
                FROM race_entries e JOIN races r ON e.race_id = r.race_id
                WHERE e.horse_code IN :h_codes AND e.finish_position IS NOT NULL
            ) SELECT * FROM RankedRaces WHERE rn = 1
        """)
        with self.factory.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"h_codes": tuple(horse_codes)})
        return df.set_index('horse_code').to_dict('index')

    def predict_live_race(self, today_class, venue, distance, rail_placement, entries_list):
        df = pd.DataFrame(entries_list)
        df['live_odds']     = pd.to_numeric(df['live_odds'], errors='coerce')
        df['live_pla_odds'] = pd.to_numeric(df.get('live_pla_odds', 0.0), errors='coerce')
        df = df.dropna(subset=['live_odds'])
        df = df[df['live_odds'] > 1.0].reset_index(drop=True)
        if df.empty or len(df) < 2: return pd.DataFrame()

        hist = self._fetch_historical_states(df['horse_code'].tolist())
        df['is_maiden'] = df['horse_code'].map(lambda x: 1 if x not in hist else 0)
        for col in ('ema_early_z', 'ema_mid_z', 'ema_finish_z'):
            df[col] = df['horse_code'].map(lambda x, c=col: hist.get(x, {}).get(c, 0.0))
        for col, sentinel in (('pre_race_mu', None), ('pre_race_sigma', None)):
            df[col] = df['horse_code'].map(lambda x, c=col: hist.get(x, {}).get(c))

        df['last_class']     = df['horse_code'].map(lambda x: hist.get(x, {}).get('last_race_class', ''))
        df['last_class_lvl'] = df['last_class'].apply(self._class_level)
        df['is_class_drop'] = ((df['last_class_lvl'] != 99) & (df['last_class_lvl'] < self._class_level(today_class))).astype(float)
        df['is_class_rise'] = ((df['last_class_lvl'] != 99) & (df['last_class_lvl'] > self._class_level(today_class))).astype(float)
        df['last_race_date'] = pd.to_datetime(df['horse_code'].map(lambda x: hist.get(x, {}).get('last_race_date')))
        df['days_since_last_race'] = (pd.Timestamp.now() - df['last_race_date']).dt.days.fillna(14.0)

        df['race_id']        = f"LIVE_{venue}_{distance}_{pd.Timestamp.now().strftime('%H%M')}"
        df['venue']          = venue
        df['distance']       = int(distance)
        df['race_class']     = today_class
        df['rail_placement'] = rail_placement
        df['track_condition']= 'GOOD'
        df['win_odds']       = df['live_odds']
        df['finish_position'] = 99

        df = self.factory.engineer_features(df)
        if df.empty or len(df) < 2: return pd.DataFrame()

        df['is_class_drop']  = df['is_class_drop'].astype(float).fillna(0.0)
        df['is_class_rise']  = df['is_class_rise'].astype(float).fillna(0.0)
        df['is_maiden']      = df['is_maiden'].astype(float)
        df['draw_x_early_pace']      = df['draw'] * df['relative_early_pace']
        df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
        df['class_drop_x_ts']        = df['is_class_drop'] * df['ts_advantage']
        df['class_rise_x_ts']        = df['is_class_rise'] * df['ts_advantage']

        # ---- Model A (WIN) ----
        dmat_a_win = xgb.DMatrix(df[self.features_a])
        dmat_a_win.set_base_margin(calculate_base_margin(df['win_odds']))
        p_a_win_cal = self.calibrator_a_win.predict_proba(_softmax(self.model_a_win.predict(dmat_a_win)))[:, 1]
        p_a_win_cal /= p_a_win_cal.sum()

        # ---- Model A (PLA) ----
        dmat_a_pla = xgb.DMatrix(df[self.features_a])
        pla_base_odds = df['live_pla_odds'].where(df['live_pla_odds'] > 1.0, df['win_odds'] / 3.0)
        places_paid = 3.0 if len(df) >= 7 else 2.0
        pi_pla = 1.0 / pla_base_odds
        p_fair_pla = (places_paid / pi_pla.sum()) * pi_pla
        p_clipped_pla = np.clip(p_fair_pla, 1e-5, 1 - 1e-5)
        
        dmat_a_pla.set_base_margin(np.log(p_clipped_pla / (1.0 - p_clipped_pla)))
        p_a_pla_cal = self.calibrator_a_pla.predict_proba(_softmax(self.model_a_pla.predict(dmat_a_pla)))[:, 1]
        p_a_pla_cal /= p_a_pla_cal.sum()

        # ---- Model B (CoxPH) ----
        cox_features = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden', 'track_width', 'straight_length'
        ]
        raw_b = self.model_b.predict_partial_hazard(df[cox_features])
        p_b_softmax = raw_b / raw_b.sum()
        
        bins = [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf]
        df['pace_archetype'] = pd.cut(df['relative_early_pace'], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
        p_b_cal = self.calibrator_b.predict_proba(p_b_softmax.values, strata=df['pace_archetype'].values)[:, 1]
        p_b_cal /= p_b_cal.sum()

        # ---- Public market ----
        p_mkt_win = 1.0 / df['win_odds'].values
        p_mkt_win /= p_mkt_win.sum()

        p_mkt_pla = 1.0 / pla_base_odds.values
        p_mkt_pla /= p_mkt_pla.sum()

        race_ids = np.array([df['race_id'].iloc[0]] * len(df))
        
        # Route to respective stackers
        df['P_model_win'] = self.stacker_win.predict(np.column_stack([p_a_win_cal, p_b_cal, p_mkt_win]), race_ids)
        df['P_model_pla'] = self.stacker_pla.predict(np.column_stack([p_a_pla_cal, p_b_cal, p_mkt_pla]), race_ids)
        df['P_model_exo'] = self.stacker_exo.predict(np.column_stack([p_a_win_cal, p_b_cal, p_mkt_win]), race_ids)
        
        # Backward compatibility for single-pool EV debug output
        df['P_model'] = df['P_model_win']
        df['EV'] = df['P_model'] * df['live_odds'] - 1.0

        keep_cols = ['horse_no', 'horse_code', 'jockey', 'draw', 'actual_weight', 'live_odds', 'live_pla_odds', 'relative_early_pace', 'P_model_win', 'P_model_pla', 'P_model_exo', 'EV']
        return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)