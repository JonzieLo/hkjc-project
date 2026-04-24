import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import joblib
import logging
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from hkjc_engine.models.ensemble import BetaCalibrator
from hkjc_engine.models.feature_factory import HKJCFeatureFactory, calculate_base_margin
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')
def true_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class XGBResidualTrainer:
    def __init__(self, db_url):
        self.factory = HKJCFeatureFactory(db_url)
        self.features = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length',
            # 'draw_x_early_pace', 'straight_x_finish_pace', 
            # 'class_drop_x_ts', 'class_rise_x_ts', 
            # 'days_since_last_race'
        ]
    
    def train(self, start_date='2018-01-01', end_date='2022-07-01', save_path='xgb_logodds_model_v2.pkl', calibrator_path='model_calibrator_v2.pkl'):
        logging.info("Fetching and engineering training data...")
        raw_df = self.factory.fetch_raw_data(start_date, end_date)
        df = self.factory.engineer_features(raw_df)
        df = df.sort_values(by='race_id').reset_index(drop=True)
        X = df[self.features]
        y = df['is_winner'] 
        
        df['base_margin'] = df.groupby('race_id')['win_odds'].transform(calculate_base_margin)
        base_margin = df['base_margin']

        groups = df['race_id']
        logging.info(f"Training on {len(df)} entries across {df['race_id'].nunique()} races.")
        
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
            'lambda': 1.0 
        }
        
        fold = 1
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, y_train, margin_train = X.iloc[train_idx], y.iloc[train_idx], base_margin.iloc[train_idx]
            X_val, y_val, margin_val = X.iloc[val_idx], y.iloc[val_idx], base_margin.iloc[val_idx]

            train_groups = groups.iloc[train_idx].value_counts(sort=False)[groups.iloc[train_idx].unique()].values
            val_groups = groups.iloc[val_idx].value_counts(sort=False)[groups.iloc[val_idx].unique()].values
            # Feed the public odds as the starting point!
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtrain.set_base_margin(margin_train)
            dtrain.set_group(train_groups)

            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_base_margin(margin_val)
            dval.set_group(val_groups)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            
            model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
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
            fold_df['fold_prob'] = fold_df.groupby('race_id')['raw_score'].transform(lambda x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
            fold_loss = log_loss(fold_df['is_winner'], fold_df['fold_prob'])
            top_1_acc = fold_df.loc[fold_df.groupby('race_id')['fold_prob'].idxmax()]['is_winner'].mean()
            logging.info(f"Fold {fold}| Logloss: {fold_loss:.5f}| Top-1 Accuracy: {top_1_acc:.1%}")
            fold += 1

        df['raw_score'] = oof_preds
        df['P_model'] = df.groupby('race_id')['raw_score'].transform(lambda x: np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
        df['P_public_raw'] = 1.0 / df['win_odds']
        df['P_public'] = df['P_public_raw'] / df.groupby('race_id')['P_public_raw'].transform('sum')
        public_loss = log_loss(y, df['P_public'])
        logging.info(f"Public Consensus LogLoss: {public_loss:.5f}")
        
        # calibrator = IsotonicRegression(y_min=1e-4, y_max=0.99, out_of_bounds='clip')
        # calibrator.fit(df['P_model'], y)
        # joblib.dump(calibrator, calibrator_path)

        # calibrator = LogisticRegression(C=1e5, solver='lbfgs')
        # calibrator.fit(oof_preds.reshape(-1, 1), y)
        # joblib.dump(calibrator, 'model_calibrator.pkl')

        calibrator = BetaCalibrator().fit(df['P_model'].values, y.values)
        joblib.dump(calibrator, calibrator_path)
        df['P_calibrated_raw'] = calibrator.predict(df['P_model'].values)
        df['P_calibrated'] = df['P_calibrated_raw'] / df.groupby('race_id')['P_calibrated_raw'].transform('sum')
        
        pre_cal_loss = log_loss(y, df['P_model'])
        post_cal_loss = log_loss(y, df['P_calibrated'])
        
        logging.info(f"Overall Pre-Calibration OOF LogLoss:  {pre_cal_loss:.5f}")
        logging.info(f"Overall Post-Calibration OOF LogLoss: {post_cal_loss:.5f}")

        dall = xgb.DMatrix(X, label=y)
        dall.set_base_margin(base_margin)
        all_groups = groups.value_counts(sort=False)[groups.unique()].values
        dall.set_group(all_groups)
        final_model = xgb.train(params=xgb_params, dtrain=dall, num_boost_round=150)
        
        joblib.dump(final_model, save_path)
        logging.info(f"Final model saved to {save_path}")

        importances = final_model.get_score(importance_type='gain')
        logging.info("\n--- Feature Importances (Gain) ---")
        for feat, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{feat:<22}: {score:.2f}")
        export_df = df[['race_id', 'horse_code', 'finish_position', 'win_odds', 'P_calibrated']].copy()
        export_df.to_csv('model_a_oof_predictions.csv', index=False)
        logging.info("Exported Model A predictions to model_a_oof_predictions.csv")

if __name__ == "__main__":
    # DB_URL loaded from hkjc_engine.config
    trainer = XGBResidualTrainer(DB_URL)
    today_str = datetime.now().strftime('%Y-%m-%d')
    trainer.train(
        start_date='2018-01-01', 
        end_date=today_str, 
        save_path='xgb_prod_model.pkl', 
        calibrator_path='xgb_prod_calibrator.pkl'
    )