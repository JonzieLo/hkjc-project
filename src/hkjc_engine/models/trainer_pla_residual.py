"""
Model A (PLA) — XGBoost market-residual learner for the PLACE pool.
"""
from __future__ import annotations

import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL, artifact
from hkjc_engine.data.stop_sell_loader import attach_win_anchor
from hkjc_engine.models.ensemble import SmoothedIsotonicCalibrator
from hkjc_engine.models.feature_factory import HKJCFeatureFactory

logging.basicConfig(level=logging.INFO, format='%(message)s')

def _grouped_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    out = np.empty_like(scores, dtype=float)
    s = pd.Series(scores)
    for rid, idx in pd.Series(race_ids).groupby(race_ids).groups.items():
        x = s.iloc[idx].values
        x = x - x.max()
        e = np.exp(x); e /= e.sum()
        out[idx] = e
    return out

class XGBResidualTrainerPLA:
    FEATURES: list[str] = [
        'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
        'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
        'ts_advantage', 'is_maiden', 'jockey_alpha',
        'track_width', 'straight_length',
    ]

    def __init__(self, db_url: str = DB_URL):
        self.engine = create_engine(db_url)
        self.factory = HKJCFeatureFactory(db_url)

    def _attach_anchors(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Attach WIN anchor first
        df = attach_win_anchor(
            df, self.engine, odds_col='win_odds', out_col='stop_sell_odds',
            fallback='final_with_drift_adj', drift_ratio_default=1.005
        )
        # 2. Attach PLA anchor
        q = text("""
            WITH ranked AS (
                SELECT
                    race_id, combination AS horse_no, odds AS stop_sell_pla_odds,
                    ROW_NUMBER() OVER (
                        PARTITION BY race_id, combination
                        ORDER BY
                            CASE phase
                                WHEN 'PRE_STOP_SELL'  THEN 1
                                WHEN 'UNKNOWN'        THEN 2
                                WHEN 'POST_STOP_SELL' THEN 3
                                WHEN 'FINAL'          THEN 4
                                ELSE 5
                            END ASC,
                            timestamp DESC
                    ) AS rn
                FROM live_odds_history
                WHERE pool_type = 'PLA'
                  AND odds > 1.0 AND odds < 999
                  AND race_id = ANY(:rids)
            )
            SELECT race_id, horse_no, stop_sell_pla_odds
            FROM ranked WHERE rn = 1
        """)
        rids = df['race_id'].astype(str).unique().tolist()
        with self.engine.connect() as conn:
            pla_df = pd.read_sql(q, conn, params={'rids': rids})
        
        df['horse_no'] = df['horse_no'].astype(str)
        if not pla_df.empty:
            pla_df['horse_no'] = pla_df['horse_no'].astype(str)
            df = df.merge(pla_df, on=['race_id', 'horse_no'], how='left')
        else:
            df['stop_sell_pla_odds'] = np.nan

        # SYNTHETIC PLACE ODDS FALLBACK: Solves the pre-2026 missing data crash
        df['stop_sell_pla_odds'] = df['stop_sell_pla_odds'].fillna(
            np.maximum(1.05, df['stop_sell_odds'].astype(float) / 3.0)
        )
        return df

    def train(self, start_date: str = '2018-01-01', end_date: str | None = None,
              save_path: str = artifact('wf_model_a_pla.pkl'),
              calibrator_path: str = artifact('wf_calib_a_pla.pkl'),
              oof_csv_path: str = artifact('model_a_pla_oof_predictions.csv')) -> dict:
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        logging.info("Fetching and engineering PLA training data...")
        raw_df = self.factory.fetch_raw_data(start_date, end_date)
        df = self.factory.engineer_features(raw_df)
        df = self._attach_anchors(df).sort_values('race_id').reset_index(drop=True)

        field_sizes = df.groupby('race_id')['horse_no'].transform('count')
        df['places_paid'] = np.where(field_sizes >= 7, 3.0, 2.0)
        df['is_placed'] = (df['finish_position'] <= df['places_paid']).astype(int)

        X = df[self.FEATURES]
        y = df['is_placed']
        groups = df['race_id']
        logging.info("Training PLA on %d entries across %d races.", len(df), groups.nunique())

        # MULTI-WINNER BASE MARGIN LOGIC
        df['pi'] = 1.0 / df['stop_sell_pla_odds'].astype(float)
        df['sum_pi'] = df.groupby('race_id')['pi'].transform('sum')
        df['p_fair'] = (df['places_paid'] / df['sum_pi']) * df['pi']
        df['p_clipped'] = np.clip(df['p_fair'], 1e-5, 1 - 1e-5)
        base_margin = np.log(df['p_clipped'] / (1.0 - df['p_clipped']))

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df))

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 0.1,
            'lambda': 1.0,
        }

        for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
            dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dtrain.set_base_margin(base_margin.iloc[tr])
            dval = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
            dval.set_base_margin(base_margin.iloc[va])

            model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=1000,
                              evals=[(dtrain, 'train'), (dval, 'eval')],
                              early_stopping_rounds=50, verbose_eval=False)
            oof_preds[va] = model.predict(dval)

            fold_loss = log_loss(y.iloc[va], oof_preds[va])
            logging.info("Fold %d | PLA LogLoss: %.5f", fold, fold_loss)

        df['P_model'] = oof_preds
        calibrator = SmoothedIsotonicCalibrator().fit(df['P_model'].values, y.values)
        joblib.dump(calibrator, calibrator_path)
        
        df['P_calibrated'] = calibrator.predict(df['P_model'].values)

        logging.info("PLA OOF LogLoss — pre-cal: %.5f | post-cal: %.5f",
                     log_loss(y, df['P_model']), log_loss(y, df['P_calibrated']))

        dall = xgb.DMatrix(X, label=y)
        dall.set_base_margin(base_margin)
        final_model = xgb.train(params=xgb_params, dtrain=dall, num_boost_round=150)
        joblib.dump(final_model, save_path)

        export_df = df[['race_id', 'horse_code', 'horse_no', 'finish_position', 'is_placed', 'stop_sell_odds', 'stop_sell_pla_odds', 'P_calibrated']].copy()
        export_df.to_csv(oof_csv_path, index=False)
        logging.info("Exported Model A PLA OOF to %s", oof_csv_path)

        return {'final_model_path': save_path, 'calibrator_path': calibrator_path, 'oof_path': oof_csv_path}

if __name__ == "__main__":
    XGBResidualTrainerPLA().train()