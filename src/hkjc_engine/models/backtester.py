import logging
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sqlalchemy import create_engine, text
from hkjc_engine.models.feature_factory import HKJCFeatureFactory, calculate_base_margin
import hkjc_engine.models.ensemble
from hkjc_engine.models.betting_policy import qualify_and_size
from hkjc_engine.config import DB_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')

def true_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class XGBEnsembleBacktester:
    def __init__(self, db_url, stacker_path='wf_stacker.pkl', shrinkage=0.85, starting_bankroll=100000.0):
        self.engine = create_engine(db_url)
        self.factory = HKJCFeatureFactory(db_url)
        self.model_a = joblib.load('wf_model_a.pkl') # Rank:Pairwise Residual
        self.model_b = joblib.load('wf_model_b.pkl')  # Grouped Softmax Independent
        self.calibrator_a = joblib.load('wf_calib_a.pkl')
        self.calibrator_b = joblib.load('wf_calib_b.pkl')
        self.stacker = joblib.load(stacker_path)
        
        self.bankroll = starting_bankroll
        self.initial_bankroll = starting_bankroll
        self.shrinkage = shrinkage
        
        # Model A 12-feature
        self.features_a = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length'
        ]
        
        # Model B 17-feature
        self.features_b = [
            'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
            'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
            'ts_advantage', 'is_maiden', 'jockey_alpha', 'track_width', 'straight_length',
            'draw_x_early_pace', 'straight_x_finish_pace', 
            'class_drop_x_ts', 'class_rise_x_ts', 
            'days_since_last_race'
        ]

    def run_backtest(self, start_date='2024-01-01', end_date='2026-01-01'):
        logging.info("Starting Full Ensemble Financial Backtest...")
        
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
                e.horse_code, e.horse_no, e.draw, e.actual_weight, e.win_odds, e.finish_position,
                e.ema_early_z, e.ema_mid_z, e.ema_finish_z,
                e.pre_race_mu, e.pre_race_sigma, e.jockey, e.days_since_last_race, 
                e.is_class_drop, e.is_class_rise,
                CASE WHEN cc.career_run_number = 1 THEN 1 ELSE 0 END as is_maiden,
                d.combination AS win_combination, d.dividend AS win_dividend
            FROM races r
            JOIN race_entries e ON r.race_id = e.race_id
            JOIN CareerCounts cc ON e.race_id = cc.race_id AND e.horse_code = cc.horse_code
            LEFT JOIN race_dividends d ON r.race_id = d.race_id AND d.pool = 'WIN'
            WHERE r.race_date >= :start_date AND r.race_date < :end_date
            AND e.win_odds IS NOT NULL
            AND e.finish_position IS NOT NULL
            ORDER BY r.race_date ASC, r.race_no ASC
        """)
        
        with self.engine.connect() as conn:
            raw_data = pd.read_sql(query, conn, params={"start_date": start_date, "end_date": end_date})

        grouped_races = raw_data.groupby('race_id')
        
        bets_placed = 0
        winning_bets = 0
        total_staked = 0.0
        bet_ledger = []
        oof_export_list = []

        for race_id, race_df in grouped_races:
            df = self.factory.engineer_features(race_df.copy())
            
            df['is_class_drop'] = df['is_class_drop'].astype(float).fillna(0.0)
            df['is_class_rise'] = df['is_class_rise'].astype(float).fillna(0.0)
            df['is_maiden'] = df['is_maiden'].astype(float)
            df['draw_x_early_pace'] = df['draw'] * df['relative_early_pace']
            df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
            df['class_drop_x_ts'] = df['is_class_drop'] * df['ts_advantage']
            df['class_rise_x_ts'] = df['is_class_rise'] * df['ts_advantage']

            # dmatrix = xgb.DMatrix(df[self.features])
            
            # --- MODEL A (Market Residuals) ---
            dmatrix_a = xgb.DMatrix(df[self.features_a])
            df['base_margin'] = calculate_base_margin(df['win_odds'])
            dmatrix_a.set_base_margin(df['base_margin'])
            
            df['raw_score_A'] = self.model_a.predict(dmatrix_a)
            df['P_softmax_A'] = true_softmax(df['raw_score_A'])
            df['P_cal_A'] = self.calibrator_a.predict_proba(df['P_softmax_A'].values)[:, 1]

            # --- MODEL B (Grouped Softmax Physics) ---
            dmatrix_b = xgb.DMatrix(df[self.features_b])
            df['raw_score_B'] = self.model_b.predict(dmatrix_b)
            df['P_softmax_B'] = true_softmax(df['raw_score_B'])
            df['P_cal_B'] = self.calibrator_b.predict_proba(df['P_softmax_B'].values)[:, 1]

            # --- PUBLIC MARKET ---
            df['P_public'] = 1.0 / df['win_odds']
            df['P_public'] = df['P_public'] / df['P_public'].sum()

            # --- BENTER LOG-LINEAR ENSEMBLE ---
            P_matrix = np.column_stack([
                df['P_cal_A'].values, 
                df['P_cal_B'].values, 
                df['P_public'].values
            ])
            df['P_model'] = self.stacker.predict(P_matrix, df['race_id'].values)
            
            # Log all +EV opportunities BEFORE filtering (for offline analysis)
            df['EV_naive'] = df['P_model'] * df['win_odds'] - 1.0
            pos = df[(df['EV_naive'] > 0.0) & (df['win_odds'] <= 50.0)]
            if not pos.empty:
                with open('all_positive_ev_log.csv', 'a') as f:
                    for _, row in pos.iterrows():
                        f.write(f"{race_id},{row['horse_code']},{row['win_odds']},{row['P_model']},{row['EV_naive']},{0.02}\n")

            keep_idx, stakes = qualify_and_size(
                p_raw=df['P_model'].values,
                odds=df['win_odds'].values,
                bankroll=self.bankroll,
                kelly_fraction=0.35,
                shrinkage=self.shrinkage,
                base_hurdle=0.005,# Baker-McHale base EV requirement
                longshot_buffer=0.005,
                longshot_threshold=25.0,
                per_bet_cap=0.05,
                race_cap=0.10,
                top_k=10
            )
            
            oof_export_list.append(df[['race_id', 'horse_code', 'finish_position', 'P_model']].copy())

            if keep_idx.size == 0:
                continue

            for local_i, stake in zip(keep_idx, stakes):
                target = df.iloc[local_i]
                
                if stake < 10: 
                    continue 
                    
                bets_placed += 1
                total_staked += stake
                self.bankroll -= stake
                
                is_win = (str(target['horse_no']).replace('.0', '') == str(target['win_combination']).replace('.0', ''))
                
                if is_win:
                    payout = (stake / 10.0) * float(target['win_dividend'])
                    profit = payout - stake
                    self.bankroll += payout
                    winning_bets += 1
                else:
                    profit = -stake

                p_adj = min(self.shrinkage * target['P_model'], 1 - 1e-9)
                ev_adj = p_adj * target['win_odds'] - 1.0
                
                bet_ledger.append({
                    'race_id': race_id,
                    'horse_code': target['horse_code'],
                    'win_odds': target['win_odds'],
                    'P_public': target['P_public'],
                    'P_model': target['P_model'],
                    'P_shrunk': p_adj,
                    'EV_naive': target['EV_naive'],
                    'EV_shrunk': ev_adj,
                    'stake': stake,
                    'profit': profit
                })

        monthly_profit = self.bankroll - self.initial_bankroll
        monthly_roi = (monthly_profit / total_staked) * 100 if total_staked > 0 else 0.0
        win_rate = (winning_bets / bets_placed) * 100 if bets_placed > 0 else 0.0
        lifetime_profit = self.bankroll - 100000.0

        logging.info("\n" + "="*50)
        logging.info("   BENTER LOG-LINEAR ENSEMBLE BACKTEST")
        logging.info("="*50)
        logging.info(f"Starting Bankroll : ${self.initial_bankroll:,.2f}")
        logging.info(f"Ending Bankroll   : ${self.bankroll:,.2f}")
        logging.info("-" * 50)
        logging.info(f"Monthly Net Profit: ${monthly_profit:,.2f}")
        logging.info(f"Total Bets Placed : {bets_placed}")
        logging.info(f"Total Staked      : ${total_staked:,.2f}")
        logging.info(f"Win Rate          : {win_rate:.2f}%")
        logging.info(f"Monthly ROI       : {monthly_roi:.2f}%")
        logging.info("-" * 50)
        logging.info(f"LIFETIME PROFIT   : ${lifetime_profit:,.2f}")
        logging.info("="*50)
        if oof_export_list:
            export_df = pd.concat(oof_export_list, ignore_index=True)
            with open('ensemble_oof_results.csv', 'a', newline='') as f:
                export_df.to_csv(f, header=f.tell()==0, index=False)
        return bet_ledger, self.bankroll

if __name__ == "__main__":
    backtester = XGBEnsembleBacktester(DB_URL)
    backtester.run_backtest()