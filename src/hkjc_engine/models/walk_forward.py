import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import joblib
from hkjc_engine.models.trainer_residual import XGBResidualTrainer
from hkjc_engine.models.trainer_independent import XGBIndependentTrainer
from hkjc_engine.models.ensemble import EnsembleOptimizer
from hkjc_engine.models.backtester import XGBEnsembleBacktester
from hkjc_engine.models.theta_optimizer import calibrate_global
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_walk_forward_validation():
    TRAIN_START_DATE = '2018-01-01'
    TEST_START_DATE = '2024-01-01'  
    TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    test_months = pd.date_range(start=TEST_START_DATE, end=TEST_END_DATE, freq='MS')
    
    trainer_A = XGBResidualTrainer(DB_URL)
    trainer_B = XGBIndependentTrainer(DB_URL)
    
    master_ledger = []
    current_bankroll = 100000.0 
    
    print("=" * 50)    
    print(" STARTING LOG-LINEAR WALK-FORWARD TEST")
    print("=" * 50)

    for i in range(len(test_months) - 1):
        window_train_end = test_months[i].strftime('%Y-%m-%d')
        window_test_start = test_months[i].strftime('%Y-%m-%d')
        window_test_end = test_months[i+1].strftime('%Y-%m-%d')
        
        print(f"\n--- Processing Window {i+1}/{len(test_months)-1} ---")
        print(f"TRAIN: {TRAIN_START_DATE} to {window_train_end}")
        print(f"TEST : {window_test_start} to {window_test_end}")
        
        model_a_path = 'wf_model_a.pkl'
        model_b_path = 'wf_model_b.pkl'

        logging.info("Training Model A (Residuals)...")
        trainer_A.train(start_date=TRAIN_START_DATE, end_date=window_train_end, save_path=model_a_path, calibrator_path='wf_calib_a.pkl')
        
        logging.info("Training Model B (Grouped Softmax)...")
        trainer_B.train(start_date=TRAIN_START_DATE, end_date=window_train_end, save_path=model_b_path, calibrator_path='wf_calib_b.pkl')

        logging.info("Fitting Log-Linear Stacker for this window...")
        optimizer = EnsembleOptimizer('model_a_oof_predictions.csv', 'model_b_oof_predictions.csv')
        stacker = optimizer.fit_stacker()
        joblib.dump(stacker, 'wf_stacker.pkl')
        logging.info("Calculating Dynamic Shrinkage on OOF Tail...")
        try:
            df_a = pd.read_csv('model_a_oof_predictions.csv')
            df_b = pd.read_csv('model_b_oof_predictions.csv')
            df_oof = pd.merge(
                df_a[['race_id', 'horse_code', 'finish_position', 'win_odds', 'P_calibrated']],
                df_b[['race_id', 'horse_code', 'P_calibrated']],
                on=['race_id', 'horse_code'], suffixes=('_A', '_B')
            )
            df_oof['P_public'] = 1.0 / df_oof['win_odds']
            df_oof['P_public'] /= df_oof.groupby('race_id')['P_public'].transform('sum')
            
            P_mat = np.column_stack([df_oof['P_calibrated_A'].values, df_oof['P_calibrated_B'].values, df_oof['P_public'].values])
            df_oof['P_ens'] = stacker.predict(P_mat, df_oof['race_id'].values)
            tail = df_oof[(df_oof['P_ens'] * df_oof['win_odds'] - 1.0 > 0) & (df_oof['win_odds'] <= 50.0)]
            
            if len(tail) > 10:
                actual_wins = (tail['finish_position'] == 1).sum()
                expected_wins = tail['P_ens'].sum()
                
                # Laplace smoothing to prevent extreme variance from small samples
                dyn_shrinkage = (actual_wins + 1.0) / (expected_wins + 1.0)
                dyn_shrinkage = min(max(dyn_shrinkage, 0.60), 0.95)
            else:
                dyn_shrinkage = 0.85
            logging.info(f"Dynamic Shrinkage set to: {dyn_shrinkage:.4f}")
            live_config = {"shrinkage": float(dyn_shrinkage)}
            with open('live_config.json', 'w') as f:
                json.dump(live_config, f)
        except Exception as e:
            logging.warning(f"Failed to calculate dynamic shrinkage, defaulting to 0.85. Error: {e}")
            dyn_shrinkage = 0.85

        try:
            theta_input = df_oof[['race_id', 'horse_code', 'finish_position']].copy()
            theta_input['P_model'] = df_oof['P_ens']
            theta_csv_path = 'wf_theta_input.csv'
            theta_input.to_csv(theta_csv_path, index=False)

            theta_fit = calibrate_global(csv_path=theta_csv_path, n_bootstrap=0)
            THETA_2 = theta_fit['theta_2']
            THETA_3 = theta_fit['theta_3']
            logging.info(f"Dynamic theta this window: t2={THETA_2:.4f} t3={THETA_3:.4f}")

            # Persist alongside live_config.json so run_bot.py can pick them up
            with open('live_config.json', 'w') as f:
                json.dump({
                    'shrinkage': float(dyn_shrinkage),
                    'theta_2':   float(THETA_2),
                    'theta_3':   float(THETA_3),
                }, f)
        except Exception as e:
            logging.warning(f"Failed to refit theta this window, using global defaults. Error: {e}")
        logging.info(f"Backtesting on unseen data: {window_test_start} to {window_test_end}...")
        
        backtester = XGBEnsembleBacktester(db_url=DB_URL, stacker_path='wf_stacker.pkl', shrinkage=dyn_shrinkage, starting_bankroll=current_bankroll)
        backtester.model_a = joblib.load(model_a_path)
        backtester.model_b = joblib.load(model_b_path)
        backtester.calibrator_a = joblib.load('wf_calib_a.pkl')
        backtester.calibrator_b = joblib.load('wf_calib_b.pkl')
        backtester.bankroll = current_bankroll
        backtester.initial_bankroll = current_bankroll
        
        monthly_ledger, current_bankroll = backtester.run_backtest(
            start_date=window_test_start, 
            end_date=window_test_end
        )
        
        master_ledger.extend(monthly_ledger)
        print(f"Bankroll after {window_test_start}: ${current_bankroll:,.2f}")

    print("\n" + "=" * 50)
    print("       FINAL OUT-OF-SAMPLE STITCHED RESULTS")
    print("\n" + "=" * 50)
    
    if not master_ledger:
        print("No bets placed during walk-forward validation.")
        return
        
    df_ledger = pd.DataFrame(master_ledger)
    export_path = 'walk_forward_master_ledger.csv'
    df_ledger.to_csv(export_path, index=False)
    
    total_bets = len(df_ledger)
    total_staked = df_ledger['stake'].sum()
    total_profit = df_ledger['profit'].sum()
    win_rate = (df_ledger['profit'] > 0).mean() * 100
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
    
    print(f"Total Bets Placed : {total_bets}")
    print(f"Total Staked      : ${total_staked:,.2f}")
    print(f"Total Net Profit  : ${total_profit:,.2f}")
    print(f"Ending Bankroll   : ${current_bankroll:,.2f}")
    print(f"Win Rate          : {win_rate:.2f}%")
    print(f"Return on Invest. : {roi:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    run_walk_forward_validation()