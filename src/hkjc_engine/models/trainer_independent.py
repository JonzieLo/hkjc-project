import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import joblib
import logging
from datetime import datetime

from hkjc_engine.config import DB_URL
from hkjc_engine.models.feature_factory import HKJCFeatureFactory
from hkjc_engine.models.ensemble import StratifiedSmoothedIsotonicCalibrator

logging.basicConfig(level=logging.INFO, format='%(message)s')

class CoxIndependentTrainer:
    def __init__(self, db_url):
        self.factory = HKJCFeatureFactory(db_url)
        self.features = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden',  'track_width', 'straight_length'
            # 'jockey_alpha' --> no convergence
        ]
    
    def train(self, start_date='2018-01-01', end_date='2022-07-01', save_path='cox_ind_model.pkl', calibrator_path='cox_ind_calibrator.pkl'):
        logging.info("Fetching and engineering training data...")
        raw_df = self.factory.fetch_raw_data(start_date, end_date)
        df = self.factory.engineer_features(raw_df)
        df = df.sort_values(by='race_id').reset_index(drop=True)
        
        # Survival setup: event=1 for all, duration=finish_position (lower is higher strength)
        df['event'] = 1
        df['duration'] = df['finish_position']
        
        bins = [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf]
        df['pace_archetype'] = pd.cut(df['relative_early_pace'], bins=bins, labels=[0, 1, 2, 3, 4])
        df['pace_archetype'] = df['pace_archetype'].astype(int)
        
        train_cols = self.features + ['race_id', 'duration', 'event', 'pace_archetype']
        
        logging.info(f"Training on {len(df)} entries across {df['race_id'].nunique()} races.")
        
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df))
        
        cph_params = {'penalizer': 0.05, 'l1_ratio': 0.1}
        
        fold = 1
        for train_idx, val_idx in gkf.split(df, df['duration'], df['race_id']):
            X_train = df.loc[train_idx, train_cols].copy()
            X_val = df.loc[val_idx, train_cols].copy()

            cph = CoxPHFitter(**cph_params)
            
            # Exact Plackett-Luce likelihood via race_id strata.
            # Gamma frailty integration via pace_archetype clustering.
            cph.fit(
                X_train,
                duration_col='duration',
                event_col='event',
                strata=['race_id'],
                cluster_col='pace_archetype',
                robust=True,
                fit_options={'step_size': 0.5} 
            )
            
            # Extract partial hazard \exp(\beta^T X). Higher hazard = higher win probability.
            oof_preds[val_idx] = cph.predict_partial_hazard(X_val)

            fold_df = pd.DataFrame({
                'race_id': df.loc[val_idx, 'race_id'].values,
                'hazard': oof_preds[val_idx],
                'is_winner': (df.loc[val_idx, 'finish_position'] == 1).astype(int).values
            })
            
            fold_df['fold_prob'] = fold_df['hazard'] / fold_df.groupby('race_id')['hazard'].transform('sum')
            fold_loss = log_loss(fold_df['is_winner'], fold_df['fold_prob'])
            logging.info(f"Fold {fold} | Logloss: {fold_loss:.5f}")
            fold += 1

        df['hazard_score'] = oof_preds
        df['P_model'] = df['hazard_score'] / df.groupby('race_id')['hazard_score'].transform('sum')

        calibrator = StratifiedSmoothedIsotonicCalibrator().fit(
            df['P_model'].values, 
            df['pace_archetype'].values, 
            (df['finish_position'] == 1).astype(int).values
        )
        joblib.dump(calibrator, calibrator_path)

        df['P_calibrated_raw'] = calibrator.predict(df['P_model'].values, strata=df['pace_archetype'].values)
        df['P_calibrated'] = df['P_calibrated_raw'] / df.groupby('race_id')['P_calibrated_raw'].transform('sum')
        
        # Final Full-Data Fit
        final_cph = CoxPHFitter(**cph_params)
        final_cph.fit(
            df[train_cols],
            duration_col='duration',
            event_col='event',
            strata=['race_id'],
            cluster_col='pace_archetype',
            robust=True
        )
        joblib.dump(final_cph, save_path)
        logging.info(f"Final model saved to {save_path}")

        # Export OOF predictions for Stacker
        export_df = df[['race_id', 'horse_code', 'finish_position', 'win_odds', 'P_calibrated']].copy()
        export_df.to_csv('model_b_oof_predictions.csv', index=False)
        logging.info("Exported Model B predictions to model_b_oof_predictions.csv")

if __name__ == "__main__":
    trainer = CoxIndependentTrainer(DB_URL)
    today_str = datetime.now().strftime('%Y-%m-%d')
    trainer.train(start_date='2018-01-01', end_date=today_str)