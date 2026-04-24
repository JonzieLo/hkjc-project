import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import sys
from sqlalchemy import text

import hkjc_engine.models.ensemble  # noqa: F401  (required so joblib can unpickle BetaCalibrator / Stacker)
from hkjc_engine.models.feature_factory import HKJCFeatureFactory, calculate_base_margin

logging.basicConfig(level=logging.INFO, format='%(message)s')


FEATURES_A = [
    'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
    'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
    'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length',
]
FEATURES_B = FEATURES_A + [
    'draw_x_early_pace', 'straight_x_finish_pace',
    'class_drop_x_ts', 'class_rise_x_ts', 'days_since_last_race',
]


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


class LiveRacePredictor:
    def __init__(self, db_url,
                 model_a_path='wf_model_a.pkl',
                 model_b_path='wf_model_b.pkl',
                 calib_a_path='wf_calib_a.pkl',
                 calib_b_path='wf_calib_b.pkl',
                 stacker_path='wf_stacker.pkl'):
        self.factory      = HKJCFeatureFactory(db_url)
        self.model_a      = joblib.load(model_a_path)
        self.model_b      = joblib.load(model_b_path)
        self.calibrator_a = joblib.load(calib_a_path)
        self.calibrator_b = joblib.load(calib_b_path)
        self.stacker      = joblib.load(stacker_path)
        self.features_a   = FEATURES_A
        self.features_b   = FEATURES_B

    # ---- Class-level helper (same conventions as backtester) ------------
    @staticmethod
    def _class_level(class_str):
        if not isinstance(class_str, str):
            class_str = f"CLASS {class_str}"
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
        """Most recent race per horse for EMAs, TrueSkill, class flags."""
        if not horse_codes:
            return {}
        query = text("""
            WITH RankedRaces AS (
                SELECT
                    e.horse_code,
                    e.ema_early_z, e.ema_mid_z, e.ema_finish_z,
                    e.pre_race_mu, e.pre_race_sigma,
                    r.race_class AS last_race_class,
                    r.race_date  AS last_race_date,
                    ROW_NUMBER() OVER(
                        PARTITION BY e.horse_code
                        ORDER BY r.race_date DESC
                    ) AS rn
                FROM race_entries e
                JOIN races r ON e.race_id = r.race_id
                WHERE e.horse_code IN :h_codes
                  AND e.finish_position IS NOT NULL
            )
            SELECT * FROM RankedRaces WHERE rn = 1
        """)
        with self.factory.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"h_codes": tuple(horse_codes)})
        return df.set_index('horse_code').to_dict('index')


    def predict_live_race(self,
                          today_class, venue, distance, rail_placement,
                          entries_list):
        """
        entries_list rows must contain:
            horse_no, horse_code, jockey, draw, actual_weight,
            live_odds, live_pla_odds
        """
        df = pd.DataFrame(entries_list)
        df['live_odds']     = pd.to_numeric(df['live_odds'], errors='coerce')
        df['live_pla_odds'] = pd.to_numeric(df.get('live_pla_odds', 0.0), errors='coerce')
        df = df.dropna(subset=['live_odds'])
        df = df[df['live_odds'] > 1.0].reset_index(drop=True)
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        # --- Historical Enrichment ---
        hist = self._fetch_historical_states(df['horse_code'].tolist())
        df['is_maiden'] = df['horse_code'].map(lambda x: 1 if x not in hist else 0)
        for col in ('ema_early_z', 'ema_mid_z', 'ema_finish_z',
                    'pre_race_mu', 'pre_race_sigma'):
            df[col] = df['horse_code'].map(
                lambda x, c=col: hist.get(x, {}).get(c, 0.0)
            )

        df['last_class']     = df['horse_code'].map(
            lambda x: hist.get(x, {}).get('last_race_class', ''))
        df['last_class_lvl'] = df['last_class'].apply(self._class_level)
        today_lvl            = self._class_level(today_class)
        df['is_class_drop']  = (df['last_class_lvl'] < today_lvl).astype(float)
        df['is_class_rise']  = (df['last_class_lvl'] > today_lvl).astype(float)

        df['last_race_date'] = df['horse_code'].map(
            lambda x: hist.get(x, {}).get('last_race_date'))
        df['last_race_date'] = pd.to_datetime(df['last_race_date'])
        df['days_since_last_race'] = (
            pd.Timestamp.now() - df['last_race_date']
        ).dt.days.fillna(14.0)

        # --- Feature Factory Fields ---
        df['race_id']        = f"LIVE_{venue}_{distance}_{pd.Timestamp.now().strftime('%H%M')}"
        df['venue']          = venue
        df['distance']       = int(distance)
        df['race_class']     = today_class
        df['rail_placement'] = rail_placement
        df['track_condition']= 'GOOD'
        df['win_odds']       = df['live_odds']
        # engineer_features reads finish_position → dummy stub
        # avoids KeyError while producing all-zero residual_target
        df['finish_position'] = 99

        df = self.factory.engineer_features(df)
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        # --- Interaction Cols (recompute defensively vs feature_factory) ---
        df['is_class_drop']  = df['is_class_drop'].astype(float).fillna(0.0)
        df['is_class_rise']  = df['is_class_rise'].astype(float).fillna(0.0)
        df['is_maiden']      = df['is_maiden'].astype(float)
        df['draw_x_early_pace']      = df['draw'] * df['relative_early_pace']
        df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
        df['class_drop_x_ts']        = df['is_class_drop'] * df['ts_advantage']
        df['class_rise_x_ts']        = df['is_class_rise'] * df['ts_advantage']
        df['days_since_last_race']   = df['days_since_last_race'].fillna(14.0).astype(float)

        # ---- Model A (rank:pairwise, market-residual) ----
        dmat_a = xgb.DMatrix(df[self.features_a])
        dmat_a.set_base_margin(calculate_base_margin(df['win_odds']))
        raw_a       = self.model_a.predict(dmat_a)
        p_a_race    = _softmax(raw_a)
        p_a_cal     = self.calibrator_a.predict_proba(p_a_race)[:, 1]
        p_a_cal     = p_a_cal / p_a_cal.sum()

        # ---- Model B (grouped softmax, raw logits) ----
        dmat_b = xgb.DMatrix(df[self.features_b])
        raw_b       = self.model_b.predict(dmat_b)
        p_b_race    = _softmax(raw_b)
        p_b_cal     = self.calibrator_b.predict_proba(p_b_race)[:, 1]
        p_b_cal     = p_b_cal / p_b_cal.sum()

        # ---- Public market ----
        p_mkt = 1.0 / df['win_odds'].values
        p_mkt = p_mkt / p_mkt.sum()

        # ---- Benter log-linear stacker (single race → dummy race_id) ----
        P = np.column_stack([p_a_cal, p_b_cal, p_mkt])
        race_ids = np.array([df['race_id'].iloc[0]] * len(df))
        df['P_model']   = self.stacker.predict(P, race_ids)
        df['P_A']       = p_a_cal
        df['P_B']       = p_b_cal
        df['P_public']  = p_mkt
        df['fair_odds'] = 1.0 / df['P_model']
        df['EV']        = df['P_model'] * df['live_odds'] - 1.0

        keep_cols = [
            'horse_no', 'horse_code', 'jockey', 'draw', 'actual_weight',
            'live_odds', 'live_pla_odds',
            'P_A', 'P_B', 'P_public', 'P_model',
            'fair_odds', 'EV',
        ]
        return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)
