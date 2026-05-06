"""
Model A — XGBoost market-residual learner.

Drift-aware refactor (§1a)
--------------------------
The base_margin used to be derived from `race_entries.win_odds`, which is the FINAL settled dividend including post-STOP_SELL late money. 
That introduced a look-ahead bias of roughly 0.6%-10% per pool (see drift_diagnostic). 
This trainer now anchors base_margin on the STOP_SELL implied probability instead:
    base_margin = calculate_base_margin(stop_sell_odds)

Pre-Plan-C history is missing from live_odds_history. For those rows we fall back to FINAL odds with a uniform multiplicative drift correction (R = mean drift ratio per pool from drift_diagnostic). 
This is biased zero in expectation, which is the property the residual learner needs.

The OOF export keeps `win_odds` for downstream EV inspection but adds `stop_sell_odds` so the stacker / backtester can consistently use the point-in-time anchor.
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
from sqlalchemy import create_engine

from hkjc_engine.config import DB_URL, artifact
from hkjc_engine.data.stop_sell_loader import (
    attach_win_anchor,
    coverage_report,
)
from hkjc_engine.models.ensemble import SmoothedIsotonicCalibrator
from hkjc_engine.models.feature_factory import (
    HKJCFeatureFactory,
    calculate_base_margin,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

def _grouped_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Softmax within each race group, broadcasting back to the full vector."""
    out = np.empty_like(scores, dtype=float)
    s = pd.Series(scores)
    for rid, idx in pd.Series(race_ids).groupby(race_ids).groups.items():
        x = s.iloc[idx].values
        x = x - x.max()
        e = np.exp(x); e /= e.sum()
        out[idx] = e
    return out


class XGBResidualTrainer:
    """Market-residual XGBoost — anchored on STOP_SELL odds for point-in-time
    correctness."""

    FEATURES: list[str] = [
        'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
        'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
        'ts_advantage', 'is_maiden', 'jockey_alpha',
        'track_width', 'straight_length',
    ]

    def __init__(self,
                 db_url: str = DB_URL,
                 anchor_fallback: str = 'final_with_drift_adj',
                 fallback_drift_ratio: float = 1.005):
        """
        Parameters
        ----------
        anchor_fallback : str
            How to handle race_entries rows with no live_odds_history snapshot:
              'final_with_drift_adj' (recommended) — divide FINAL odds by
                the empirical mean drift ratio (1.005 for WIN). Unbiased.
              'final' — use FINAL verbatim. Reintroduces look-ahead bias on
                pre-Plan-C history; only safe when live coverage is ~100%.
              'drop' — exclude rows entirely.
        fallback_drift_ratio : float
            Multiplicative drift used by 'final_with_drift_adj' fallback.
            For WIN pool: 1.005. Override after the drift forecaster is
            trained, using its conditional median predictions instead.
        """
        self.engine = create_engine(db_url)
        self.factory = HKJCFeatureFactory(db_url)
        self.anchor_fallback = anchor_fallback
        self.fallback_drift_ratio = fallback_drift_ratio

    # ---- engineering ----
    def _attach_anchor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach `stop_sell_odds` per (race_id, horse_no)."""
        if 'horse_no' not in df.columns:
            raise ValueError("trainer_residual requires `horse_no` in raw data; "
                             "make sure HKJCFeatureFactory.fetch_raw_data "
                             "selects e.horse_no.")
        df = attach_win_anchor(
            df, self.engine,
            odds_col='win_odds',
            out_col='stop_sell_odds',
            fallback=self.anchor_fallback,
            drift_ratio_default=self.fallback_drift_ratio,
        )
        return df

    # ---- main entry ----
    def train(self,
              start_date: str = '2018-01-01',
              end_date: str | None = None,
              save_path: str = artifact('xgb_prod_model.pkl'),
              calibrator_path: str = artifact('xgb_prod_calibrator.pkl'),
              oof_csv_path: str = artifact('model_a_oof_predictions.csv'),
              ) -> dict:
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        logging.info("Fetching and engineering training data...")
        raw_df = self.factory.fetch_raw_data(start_date, end_date)
        if 'horse_no' not in raw_df.columns:
            # Older fetch_raw_data may not select horse_no — patch in if missing
            raise ValueError("fetch_raw_data must select horse_no for the "
                             "STOP_SELL anchor merge to work.")
        df = self.factory.engineer_features(raw_df)
        df = self._attach_anchor(df).sort_values('race_id').reset_index(drop=True)

        # ---- coverage diagnostic ----
        rep = coverage_report(self.engine,
                              df['race_id'].astype(str).unique().tolist())
        logging.info("STOP_SELL coverage: %d / %d races have any live snapshot, "
                     "%d have POST_STOP_SELL specifically.",
                     rep.get('n_races_with_win', 0),
                     rep.get('n_races_total', 0),
                     rep.get('n_races_with_post', 0))

        X = df[self.FEATURES]
        y = df['is_winner']
        groups = df['race_id']
        logging.info("Training on %d entries across %d races.",
                     len(df), groups.nunique())

        # base_margin from STOP_SELL anchor, not FINAL
        df['base_margin'] = (df.groupby('race_id')['stop_sell_odds']
                             .transform(calculate_base_margin))
        base_margin = df['base_margin']

        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df))

        xgb_params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 0.1,
            'lambda': 1.0,
        }

        for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
            tr_groups = (groups.iloc[tr].value_counts(sort=False)
                         [groups.iloc[tr].unique()].values)
            va_groups = (groups.iloc[va].value_counts(sort=False)
                         [groups.iloc[va].unique()].values)

            dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
            dtrain.set_base_margin(base_margin.iloc[tr])
            dtrain.set_group(tr_groups)

            dval = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
            dval.set_base_margin(base_margin.iloc[va])
            dval.set_group(va_groups)

            model = xgb.train(
                params=xgb_params, dtrain=dtrain, num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                early_stopping_rounds=50, verbose_eval=False,
            )
            oof_preds[va] = model.predict(dval)

            fold_prob = _grouped_softmax(oof_preds[va], groups.iloc[va].values)
            fold_loss = log_loss(y.iloc[va], fold_prob)
            logging.info("Fold %d | LogLoss: %.5f", fold, fold_loss)

        df['raw_score'] = oof_preds
        df['P_model'] = _grouped_softmax(oof_preds, groups.values)

        # Sanity: STOP_SELL public consensus should logloss-beat FINAL public consensus only marginally. If much worse, something is wrong with the anchor (probably mis-mapped horse_no).
        df['P_pub_final_raw'] = 1.0 / df['win_odds']
        df['P_pub_final'] = (df['P_pub_final_raw']
                             / df.groupby('race_id')['P_pub_final_raw'].transform('sum'))
        df['P_pub_stop_raw'] = 1.0 / df['stop_sell_odds']
        df['P_pub_stop'] = (df['P_pub_stop_raw']
                            / df.groupby('race_id')['P_pub_stop_raw'].transform('sum'))
        logging.info("Public LogLoss — FINAL: %.5f | STOP_SELL: %.5f",
                     log_loss(y, df['P_pub_final']),
                     log_loss(y, df['P_pub_stop']))

        # Calibrator
        calibrator = SmoothedIsotonicCalibrator().fit(df['P_model'].values, y.values)
        joblib.dump(calibrator, calibrator_path)
        
        df['P_cal_raw'] = calibrator.predict(df['P_model'].values)
        df['P_calibrated'] = (df['P_cal_raw'] / df.groupby('race_id')['P_cal_raw'].transform('sum'))
        logging.info("OOF LogLoss — pre-cal: %.5f | post-cal: %.5f",
                     log_loss(y, df['P_model']),
                     log_loss(y, df['P_calibrated']))

        # Final (full-data) refit
        dall = xgb.DMatrix(X, label=y)
        dall.set_base_margin(base_margin)
        all_groups = (groups.value_counts(sort=False)[groups.unique()].values)
        dall.set_group(all_groups)
        final_model = xgb.train(params=xgb_params, dtrain=dall,
                                num_boost_round=150)
        joblib.dump(final_model, save_path)
        logging.info("Final model saved to %s", save_path)

        # Export OOF — INCLUDE stop_sell_odds so the stacker uses it for P_mkt
        export_df = df[['race_id', 'horse_code', 'horse_no', 'finish_position', 'win_odds', 'stop_sell_odds', 'P_calibrated']].copy()
        export_df.to_csv(oof_csv_path, index=False)
        logging.info("Exported Model A OOF predictions to %s", oof_csv_path)

        return {
            'final_model_path': save_path,
            'calibrator_path': calibrator_path,
            'oof_path': oof_csv_path,
            'oof_logloss_post_cal': float(log_loss(y, df['P_calibrated'])),
        }


if __name__ == "__main__":
    XGBResidualTrainer().train()
