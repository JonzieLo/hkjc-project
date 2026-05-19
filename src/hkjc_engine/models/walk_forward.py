"""
Walk-forward validation orchestrator (Multi-Agent refactor).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from hkjc_engine.config import DB_URL, artifact
from hkjc_engine.models.backtester import XGBEnsembleBacktester
from hkjc_engine.models.drift_forecaster import DriftForecaster
from hkjc_engine.models.ensemble import EnsembleOptimizer
from hkjc_engine.models.theta_optimizer import calibrate_global
from hkjc_engine.models.trainer_independent import CoxIndependentTrainer
from hkjc_engine.models.trainer_residual import XGBResidualTrainer
from hkjc_engine.models.trainer_pla_residual import XGBResidualTrainerPLA

logging.basicConfig(level=logging.INFO, format='%(message)s')

def _refit_dynamic_shrinkage(df_oof: pd.DataFrame, stacker, tail_odds_col: str = 'stop_sell_odds', max_odds: float = 50.0, default: float = 0.85) -> float:
    df = df_oof.copy()
    if tail_odds_col not in df.columns:
        tail_odds_col = 'win_odds_x'
    df['P_pub_raw'] = 1.0 / df[tail_odds_col]
    df['P_pub'] = (df['P_pub_raw'] / df.groupby('race_id')['P_pub_raw'].transform('sum'))
    df['I_valid'] = (df['stop_sell_odds'].notna() & (df['stop_sell_odds'] != df['win_odds_x'])).astype(int)
    P_mat = np.column_stack([df['P_calibrated_x'].values, df['P_calibrated_y'].values, df['P_pub'].values, df['I_valid'].values])
    df['P_ens'] = stacker.predict(P_mat, df['race_id'].values)
    tail = df[(df['P_ens'] * df[tail_odds_col] - 1.0 > 0) & (df[tail_odds_col] <= max_odds)]
    if len(tail) <= 10: return default
    actual = (tail['finish_position_x'] == 1).sum()
    expected = tail['P_ens'].sum()
    shr = (actual + 1.0) / (expected + 1.0)
    return float(min(max(shr, 0.60), 0.95))

def _refit_stratified_shrinkage(df_oof: pd.DataFrame, stacker, tail_odds_col: str = 'stop_sell_odds', max_odds: float = 50.0, global_default: float = 0.85, min_band_n: int = 100) -> dict[str, float]:
    from hkjc_engine.models.betting_policy import STRATIFIED_SHRINKAGE_BANDS, STRATIFIED_SHRINKAGE_NAMES
    df = df_oof.copy()
    if tail_odds_col not in df.columns:
        tail_odds_col = 'win_odds_x'
    df['P_pub_raw'] = 1.0 / df[tail_odds_col]
    df['P_pub'] = (df['P_pub_raw'] / df.groupby('race_id')['P_pub_raw'].transform('sum'))
    df['I_valid'] = (df['stop_sell_odds'].notna() & (df['stop_sell_odds'] != df['win_odds_x'])).astype(int)
    P_mat = np.column_stack([df['P_calibrated_x'].values, df['P_calibrated_y'].values, df['P_pub'].values, df['I_valid'].values])
    df['P_ens'] = stacker.predict(P_mat, df['race_id'].values)
    edges = STRATIFIED_SHRINKAGE_BANDS
    names = STRATIFIED_SHRINKAGE_NAMES
    odds_mask = df[tail_odds_col] <= max_odds
    out: dict[str, float] = {}
    
    for i, name in enumerate(names):
        lo = edges[i]
        hi = edges[i + 1]
        band_mask = (df['P_pub'] >= lo) & (df['P_pub'] < hi) & odds_mask
        sub = df.loc[band_mask]
        n = len(sub)
        if n < min_band_n:
            continue
        actual = int((sub['finish_position_x'] == 1).sum())
        expected = float(sub['P_ens'].sum())
        shr = (actual + 1.0) / (expected + 1.0)
        shr = float(min(max(shr, 0.50), 1.00))
        out[name] = shr
    return out

def _persist_live_config(path: str, shrinkage, theta_2: float | None = None, theta_3: float | None = None, theta_place: float | None = None, theta_model_place: float | None = None, drift_stats: dict | None = None) -> None:
    cfg: dict = {}
    if isinstance(shrinkage, dict):
        cfg['shrinkage_stratified'] = {k: float(v) for k, v in shrinkage.items()}
    else:
        cfg['shrinkage'] = float(shrinkage)
    if theta_2 is not None: cfg['theta_2'] = float(theta_2)
    if theta_3 is not None: cfg['theta_3'] = float(theta_3)
    if theta_place is not None: cfg['theta_place'] = float(theta_place)
    if theta_model_place is not None: cfg['theta_model_place'] = float(theta_model_place)
    if drift_stats: cfg['drift_stats'] = drift_stats
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)

def _refit_drift_forecaster(start: str, end: str) -> DriftForecaster | None:
    try:
        f = DriftForecaster().train(start_date=start, end_date=end)
        return f
    except RuntimeError as e:
        return None

def run_walk_forward_validation(train_start: str = '2018-01-01',
                                test_start: str = '2024-01-01',
                                test_end: str | None = None,
                                starting_bankroll: float = 100_000.0,
                                pools: tuple[str, ...] | None = None,
                                exotic_odds_source: str = 'real',
                                theta_place: float | None = None,
                                theta_model_place: float | None = None,
                                stratified_shrinkage: bool = False) -> pd.DataFrame:
    
    test_end = test_end or datetime.now().strftime('%Y-%m-%d')
    test_months = pd.date_range(start=test_start, end=test_end, freq='6MS')

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts   = pd.Timestamp(test_end)
    if len(test_months) == 0:
        test_months = pd.DatetimeIndex([test_start_ts])
    if test_months[-1] < test_end_ts:
        test_months = test_months.append(pd.DatetimeIndex([test_end_ts]))

    trainer_a_win = XGBResidualTrainer(DB_URL)
    trainer_a_pla = XGBResidualTrainerPLA(DB_URL)
    trainer_b = CoxIndependentTrainer(DB_URL)

    master_ledger: list[dict] = []
    bankroll = starting_bankroll

    print("=" * 50)
    print(" DRIFT-AWARE MULTI-AGENT WALK-FORWARD")
    print("=" * 50)

    for i in range(len(test_months) - 1):
        train_end = test_months[i].strftime('%Y-%m-%d')
        test_window_start = train_end
        test_window_end = test_months[i + 1].strftime('%Y-%m-%d')

        print(f"\n--- Window {i+1}/{len(test_months)-1} ---")
        print(f"TRAIN: {train_start} -> {train_end}")
        print(f"TEST : {test_window_start} -> {test_window_end}")

        model_b_path  = artifact('wf_model_b.pkl')
        # --- 1. Train Models ---
        model_a_win_path  = artifact('wf_model_a.pkl')
        calib_a_win_path  = artifact('wf_calib_a.pkl')
        model_a_pla_path  = artifact('wf_model_a_pla.pkl')
        calib_a_pla_path  = artifact('wf_calib_a_pla.pkl')
        calib_b_path  = artifact('wf_calib_b.pkl')

        logging.info("Training Model A (WIN Residual)...")
        trainer_a_win.train(start_date=train_start, end_date=train_end,
                            save_path=model_a_win_path, calibrator_path=calib_a_win_path,
                            oof_csv_path='model_a_oof_predictions.csv')
                            
        logging.info("Training Model A (PLA Residual)...")
        trainer_a_pla.train(start_date=train_start, end_date=train_end,
                            save_path=model_a_pla_path, calibrator_path=calib_a_pla_path,
                            oof_csv_path='model_a_pla_oof_predictions.csv')

        logging.info("Training Model B (Grouped Softmax)...")
        trainer_b.train(start_date=train_start, end_date=train_end,
                        save_path=model_b_path, calibrator_path=calib_b_path)

        # --- 2. Fit Multi-Agent Stackers ---
        logging.info("Fitting Multi-Agent Stackers...")
        
        stacker_win = EnsembleOptimizer('model_a_oof_predictions.csv', 'model_b_oof_predictions.csv', mode='win').fit_stacker()
        joblib.dump(stacker_win, artifact('wf_stacker_win.pkl'))
        
        stacker_pla = EnsembleOptimizer('model_a_pla_oof_predictions.csv', 'model_b_oof_predictions.csv', mode='pla').fit_stacker()
        joblib.dump(stacker_pla, artifact('wf_stacker_pla.pkl'))
        
        stacker_exo = EnsembleOptimizer('model_a_oof_predictions.csv', 'model_b_oof_predictions.csv', mode='exotics').fit_stacker()
        joblib.dump(stacker_exo, artifact('wf_stacker_exo.pkl'))

        # --- 3. Dynamic shrinkage ---
        try:
            df_a = pd.read_csv('model_a_oof_predictions.csv')
            df_b = pd.read_csv('model_b_oof_predictions.csv')
            df_oof = pd.merge(df_a, df_b[['race_id', 'horse_code', 'P_calibrated']], on=['race_id', 'horse_code'])
            if stratified_shrinkage:
                shrinkage = _refit_stratified_shrinkage(df_oof, stacker_exo, tail_odds_col='stop_sell_odds')
            else:
                shrinkage = _refit_dynamic_shrinkage(df_oof, stacker_exo, tail_odds_col='stop_sell_odds')
        except Exception as e:
            shrinkage = 0.85
            df_oof = pd.DataFrame()

        if isinstance(shrinkage, dict):
            logging.info("Stratified shrinkage fitted: %s", {k: f"{v:.4f}" for k, v in shrinkage.items()})
        else:
            logging.info("Dynamic shrinkage = %.4f", shrinkage)

        # --- 4. Refit Henery thetas ---
        theta_2 = theta_3 = None
        if not df_oof.empty:
            try:
                # REPLACE THE P_mat DEFINITION HERE:
                df_oof['I_valid'] = (df_oof['stop_sell_odds'].notna() & (df_oof['stop_sell_odds'] != df_oof['win_odds_x'])).astype(int)
                P_mat = np.column_stack([
                    df_oof['P_calibrated_x'].values, 
                    df_oof['P_calibrated_y'].values, 
                    (1.0 / df_oof['stop_sell_odds']).values,
                    df_oof['I_valid'].values
                ])
                
                df_oof['P_ens'] = stacker_exo.predict(P_mat, df_oof['race_id'].values)
            except Exception as e:
                pass

        # --- 5. Refit drift forecaster ---
        forecaster = _refit_drift_forecaster(train_start, train_end)
        _persist_live_config('artifacts/live_config.json', shrinkage=shrinkage, theta_2=theta_2, theta_3=theta_3, theta_place=theta_place, theta_model_place=theta_model_place)

        # --- 6. Backtest ---
        logging.info("Backtesting %s -> %s...", test_window_start, test_window_end)
        from hkjc_engine.models.multi_pool_backtester import MultiPoolBacktester
        backtester = MultiPoolBacktester(
            db_url=DB_URL, 
            shrinkage=shrinkage, starting_bankroll=bankroll,
            theta_2=theta_2 or 0.8824, theta_3=theta_3 or 0.7760,
            drift_forecaster=forecaster, pools=pools, exotic_odds_source=exotic_odds_source,
            theta_place=theta_place, theta_model_place=theta_model_place,
        )
        
        backtester.model_a_win = joblib.load(model_a_win_path)
        backtester.model_a_pla = joblib.load(model_a_pla_path)
        backtester.model_b = joblib.load(model_b_path)
        backtester.calibrator_a_win = joblib.load(calib_a_win_path)
        backtester.calibrator_a_pla = joblib.load(calib_a_pla_path)
        backtester.calibrator_b = joblib.load(calib_b_path)
        backtester.stacker_win = joblib.load(artifact('wf_stacker_win.pkl'))
        backtester.stacker_pla = joblib.load(artifact('wf_stacker_pla.pkl'))
        backtester.stacker_exo = joblib.load(artifact('wf_stacker_exo.pkl'))

        backtester.bankroll = bankroll
        backtester.initial_bankroll = bankroll

        monthly_ledger, bankroll = backtester.run_backtest(start_date=test_window_start, end_date=test_window_end)
        master_ledger.extend(monthly_ledger)
        print(f"Bankroll after {test_window_start}: ${bankroll:,.2f}")

    print("\n" + "=" * 50)
    print("       FINAL OUT-OF-SAMPLE STITCHED RESULTS")
    print("=" * 50)
    if not master_ledger:
        return pd.DataFrame()

    df_ledger = pd.DataFrame(master_ledger)
    df_ledger.to_csv(artifact('walk_forward_master_ledger.csv'), index=False)

    total_bets   = len(df_ledger)
    total_staked = df_ledger['stake'].sum()
    total_profit = df_ledger['profit'].sum()
    win_rate     = (df_ledger['profit'] > 0).mean() * 100
    roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

    print(f"Total Bets Placed : {total_bets}")
    print(f"Total Staked      : ${total_staked:,.2f}")
    print(f"Total Net Profit  : ${total_profit:,.2f}")
    print(f"Ending Bankroll   : ${bankroll:,.2f}")
    print(f"Win Rate          : {win_rate:.2f}%")
    print(f"Return on Invest. : {roi:.2f}%")
    print("=" * 50)

    if 'pool' in df_ledger.columns:
        print("\n" + "=" * 50)
        print("       PER-POOL STITCHED BREAKDOWN")
        print("=" * 50)
        print(f"{'Pool':<6} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Staked':>12} {'Profit':>12} {'ROI':>8}")
        for pool, g in df_ledger.groupby('pool'):
            n = len(g); w = int(g['is_win'].sum()); s = float(g['stake'].sum()); pf = float(g['profit'].sum())
            print(f"{pool:<6} {n:>6} {w:>6} {(w/n*100) if n else 0:>7.2f}% ${s:>11,.0f} ${pf:>+11,.0f} {(pf/s*100) if s else 0:>+7.2f}%")
    return df_ledger

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_start', default='2018-01-01')
    ap.add_argument('--test_start',  default='2024-01-01')
    ap.add_argument('--test_end',    default=None)
    ap.add_argument('--bankroll', type=float, default=100_000.0)
    ap.add_argument('--pools', default=None)
    ap.add_argument('--exotic_odds_source', default='real', choices=['real', 'synthetic', 'hybrid'])
    ap.add_argument('--theta_place', type=float, default=None)
    ap.add_argument('--theta_model_place', type=float, default=None)
    ap.add_argument('--stratified_shrinkage', action='store_true')
    args = ap.parse_args()

    pools_tuple = tuple(p.strip().upper() for p in args.pools.split(',')) if args.pools else None
    run_walk_forward_validation(
        train_start=args.train_start, test_start=args.test_start, test_end=args.test_end,
        starting_bankroll=args.bankroll, pools=pools_tuple, exotic_odds_source=args.exotic_odds_source,
        theta_place=args.theta_place, theta_model_place=args.theta_model_place,
        stratified_shrinkage=args.stratified_shrinkage,
    )