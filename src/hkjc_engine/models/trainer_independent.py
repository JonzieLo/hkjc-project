import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import joblib
import logging
from sklearn.calibration import IsotonicRegression

from hkjc_engine.config import DB_URL
from hkjc_engine.models.feature_factory import HKJCFeatureFactory
from hkjc_engine.models.ensemble import BetaCalibrator

logging.basicConfig(level=logging.INFO, format='%(message)s')

class XGBIndependentTrainer:
    def __init__(self, db_url):
        self.factory = HKJCFeatureFactory(db_url)
        self.features = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length',
            'draw_x_early_pace', 'straight_x_finish_pace', 
            'class_drop_x_ts', 'class_rise_x_ts', 
            'days_since_last_race'
        ]
    
    def train(self, start_date='2018-01-01', end_date='2022-07-01', save_path='xgb_ind_model.pkl', calibrator_path='xgb_ind_calibrator.pkl'):
        logging.info("Fetching and engineering training data...")
        raw_df = self.factory.fetch_raw_data(start_date, end_date)
        df = self.factory.engineer_features(raw_df)
        df = df.sort_values(by='race_id').reset_index(drop=True)
        
        X = df[self.features]
        y = df['is_winner'] 
        groups = df['race_id']
        
        logging.info(f"Training on {len(df)} entries across {df['race_id'].nunique()} races.")
        
        gkf = GroupKFold(n_splits=5)
        oof_preds = np.zeros(len(df))
        
        #  Hyperparameters from grid search
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1.5, 
            'lambda': 1.0 
        }
        
        fold = 1
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            evals = [(dtrain, 'train'), (dval, 'eval')]
            
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                # num_boost_round=340,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            oof_preds[val_idx] = model.predict(dval)

            fold_df = pd.DataFrame({
                'race_id': groups.iloc[val_idx].values,
                'raw_score': oof_preds[val_idx],
                'is_winner': y_val.values
            })
            
            fold_df['fold_prob'] = fold_df['raw_score'] / fold_df.groupby('race_id')['raw_score'].transform('sum')
            fold_loss = log_loss(fold_df['is_winner'], fold_df['fold_prob'])
            top_picks = fold_df.loc[fold_df.groupby('race_id')['fold_prob'].idxmax()]
            top_1_acc = top_picks['is_winner'].mean()
            logging.info(f"Fold {fold}| Logloss: {fold_loss:.5f}| Top-1 Accuracy: {top_1_acc:.1%}")
            fold += 1

        df['raw_score'] = oof_preds
        df['P_model'] = df['raw_score'] / df.groupby('race_id')['raw_score'].transform('sum')

        df['P_public_raw'] = 1.0 / df['win_odds']
        df['P_public'] = df['P_public_raw'] / df.groupby('race_id')['P_public_raw'].transform('sum')
        public_loss = log_loss(df['finish_position'] == 1, df['P_public'])
        print(f"Public Consensus LogLoss: {public_loss:.5f}")
        
        # calibrator = IsotonicRegression(y_min=1e-4, y_max=0.99, out_of_bounds='clip')
        # calibrator.fit(df['P_model'], y)
        # joblib.dump(calibrator, calibrator_path)
        # df['P_calibrated_raw'] = calibrator.predict(df['P_model'])

        calibrator = BetaCalibrator().fit(df['P_model'].values, y.values)
        joblib.dump(calibrator, calibrator_path)

        df['P_calibrated_raw'] = calibrator.predict(df['P_model'].values)
        df['P_calibrated'] = df['P_calibrated_raw'] / df.groupby('race_id')['P_calibrated_raw'].transform('sum')
        
        pre_cal_loss = log_loss(y, df['P_model'])
        post_cal_loss = log_loss(y, df['P_calibrated'])
        
        logging.info(f"Overall Pre-Calibration OOF LogLoss:  {pre_cal_loss:.5f}")
        logging.info(f"Overall Post-Calibration OOF LogLoss: {post_cal_loss:.5f}")

        # Final Training
        dall = xgb.DMatrix(X, label=y)
        final_model = xgb.train(params=xgb_params, dtrain=dall, num_boost_round=340)
        
        joblib.dump(final_model, save_path)
        logging.info(f"Final model saved to {save_path}")

        importances = final_model.get_score(importance_type='gain')
        logging.info("\n--- Feature Importances (Gain) ---")
        for feat, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{feat:<22}: {score:.2f}")

        # Overlay Analysis
        # df['edge'] = df['P_calibrated'] / df['P_public']
        # massive_overlays = df[(df['edge'] > 1.5) & (df['finish_position'] == 1)].copy()
        # logging.info("\n--- Top 15 Value Winners (Model B vs Public) ---")
        # display_cols = ['race_date', 'venue', 'race_id', 'horse_code', 'win_odds', 'P_public', 'P_calibrated', 'edge']
        # massive_overlays = massive_overlays.sort_values('win_odds', ascending=False)
        # logging.info(massive_overlays[display_cols].head(15).to_string(index=False))
        # total_value_bets = len(df[df['edge'] > 1.5])
        # if total_value_bets > 0:
        #     roi_hypothetical = (massive_overlays['win_odds'].sum() - total_value_bets) / total_value_bets
        #     logging.info(f"\nBlindly betting all >1.5 Edge horses would yield a {roi_hypothetical:.1%} ROI.")

        # --- NEW: Export OOF predictions for the Ensemble Optimizer ---
        export_df = df[['race_id', 'horse_code', 'finish_position', 'win_odds', 'P_calibrated']].copy()
        export_df.to_csv('model_b_oof_predictions.csv', index=False)
        logging.info("Exported Model B predictions to model_b_oof_predictions.csv")

if __name__ == "__main__":
    # DB_URL loaded from hkjc_engine.config
    trainer = XGBIndependentTrainer(DB_URL)
    today_str = datetime.now().strftime('%Y-%m-%d')
    trainer.train(
        start_date='2018-01-01', 
        end_date=today_str, 
        save_path='xgb_ind_model.pkl', 
        calibrator_path='xgb_ind_calibrator.pkl'
    )