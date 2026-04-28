"""
Walk-forward validation orchestrator (drift-aware refactor).

What changed
------------
1. Trainer A (residual) now anchors `base_margin` on STOP_SELL odds; OOF
   CSV exports include `stop_sell_odds`.
2. Stacker fits with `market_anchor='stop_sell'` so `P_mkt` is
   point-in-time correct.
3. Optional drift forecaster fit per window: if the forecaster artifacts
   exist, the backtester picks them up and uses pool-conditional
   (mu_R, sigma_R, median_R) for sizing. Falls back to the static
   `DEFAULT_DRIFT_STATS` in `betting_policy` when not available.
4. Dynamic-shrinkage and theta refit logic is preserved verbatim, but the
   OOF tail definition now uses `stop_sell_odds` for the EV check (the
   FINAL-odds tail was systematically over-counting +EV horses).
5. live_config.json now also persists pool-conditional drift stats so the
   live bot can pick them up without re-querying the database.
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
from hkjc_engine.models.trainer_independent import XGBIndependentTrainer
from hkjc_engine.models.trainer_residual import XGBResidualTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _refit_dynamic_shrinkage(df_oof: pd.DataFrame,
                             stacker,
                             tail_odds_col: str = 'stop_sell_odds',
                             max_odds: float = 50.0,
                             default: float = 0.85) -> float:
    """Calibration-shrinkage from OOF tail.

    CHANGED: tail definition now uses STOP_SELL odds (the anchor at
    decision time) rather than FINAL. Otherwise the +EV tail is biased by
    the late-money drop and the implied shrinkage drifts too high.
    """
    df = df_oof.copy()
    if tail_odds_col not in df.columns:
        logging.warning("%s missing — using FINAL odds for shrinkage tail.",
                        tail_odds_col)
        tail_odds_col = 'win_odds'

    df['P_pub_raw'] = 1.0 / df[tail_odds_col]
    df['P_pub'] = (df['P_pub_raw']
                   / df.groupby('race_id')['P_pub_raw'].transform('sum'))

    P_mat = np.column_stack([
        df['P_calibrated_A'].values,
        df['P_calibrated_B'].values,
        df['P_pub'].values,
    ])
    df['P_ens'] = stacker.predict(P_mat, df['race_id'].values)
    tail = df[(df['P_ens'] * df[tail_odds_col] - 1.0 > 0)
              & (df[tail_odds_col] <= max_odds)]
    if len(tail) <= 10:
        return default

    actual = (tail['finish_position'] == 1).sum()
    expected = tail['P_ens'].sum()
    shr = (actual + 1.0) / (expected + 1.0)        # Laplace smoothing
    return float(min(max(shr, 0.60), 0.95))


def _persist_live_config(path: str,
                         shrinkage: float,
                         theta_2: float | None = None,
                         theta_3: float | None = None,
                         drift_stats: dict | None = None) -> None:
    cfg = {'shrinkage': float(shrinkage)}
    if theta_2 is not None: cfg['theta_2'] = float(theta_2)
    if theta_3 is not None: cfg['theta_3'] = float(theta_3)
    if drift_stats:         cfg['drift_stats'] = drift_stats
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)


def _refit_drift_forecaster(start: str, end: str) -> DriftForecaster | None:
    """Try to refit the drift forecaster on the cumulative training window.

    Returns the trained forecaster on success, or None if there isn't
    enough live_odds_history coverage yet.
    """
    try:
        f = DriftForecaster().train(start_date=start, end_date=end)
        return f
    except RuntimeError as e:
        logging.warning("Drift forecaster refit skipped: %s", e)
        return None


# ---------------------------------------------------------------------------
# Walk-forward main
# ---------------------------------------------------------------------------

def run_walk_forward_validation(train_start: str = '2018-01-01',
                                test_start: str = '2024-01-01',
                                test_end: str | None = None,
                                starting_bankroll: float = 100_000.0,
                                ) -> pd.DataFrame:
    test_end = test_end or datetime.now().strftime('%Y-%m-%d')
    test_months = pd.date_range(start=test_start, end=test_end, freq='MS')

    trainer_a = XGBResidualTrainer(DB_URL)
    trainer_b = XGBIndependentTrainer(DB_URL)

    master_ledger: list[dict] = []
    bankroll = starting_bankroll

    print("=" * 50)
    print(" DRIFT-AWARE LOG-LINEAR WALK-FORWARD")
    print("=" * 50)

    for i in range(len(test_months) - 1):
        train_end = test_months[i].strftime('%Y-%m-%d')
        test_window_start = train_end
        test_window_end = test_months[i + 1].strftime('%Y-%m-%d')

        print(f"\n--- Window {i+1}/{len(test_months)-1} ---")
        print(f"TRAIN: {train_start} -> {train_end}")
        print(f"TEST : {test_window_start} -> {test_window_end}")

        # --- 1. Train Models A, B on STOP_SELL anchor ---
        model_a_path  = artifact('wf_model_a.pkl')
        model_b_path  = artifact('wf_model_b.pkl')
        calib_a_path  = artifact('wf_calib_a.pkl')
        calib_b_path  = artifact('wf_calib_b.pkl')
        stacker_path  = artifact('wf_stacker.pkl')

        logging.info("Training Model A (residual, STOP_SELL anchor)...")
        trainer_a.train(start_date=train_start, end_date=train_end,
                        save_path=model_a_path,
                        calibrator_path=calib_a_path,
                        oof_csv_path='model_a_oof_predictions.csv')

        logging.info("Training Model B (grouped softmax)...")
        trainer_b.train(start_date=train_start, end_date=train_end,
                        save_path=model_b_path,
                        calibrator_path=calib_b_path,
                        oof_csv_path='model_b_oof_predictions.csv')

        # --- 2. Fit stacker on STOP_SELL P_mkt ---
        logging.info("Fitting log-linear stacker (STOP_SELL P_mkt)...")
        stacker = EnsembleOptimizer(
            'model_a_oof_predictions.csv',
            'model_b_oof_predictions.csv',
            market_anchor='stop_sell',
        ).fit_stacker()
        joblib.dump(stacker, stacker_path)

        # --- 3. Dynamic shrinkage on STOP_SELL tail ---
        try:
            df_a = pd.read_csv('model_a_oof_predictions.csv')
            df_b = pd.read_csv('model_b_oof_predictions.csv')
            df_oof = pd.merge(
                df_a[['race_id', 'horse_code', 'finish_position',
                      'win_odds', 'stop_sell_odds', 'P_calibrated']]
                  if 'stop_sell_odds' in df_a.columns
                  else df_a[['race_id', 'horse_code', 'finish_position',
                             'win_odds', 'P_calibrated']],
                df_b[['race_id', 'horse_code', 'P_calibrated']],
                on=['race_id', 'horse_code'], suffixes=('_A', '_B'))
            shrinkage = _refit_dynamic_shrinkage(df_oof, stacker,
                                                 tail_odds_col='stop_sell_odds')
        except Exception as e:
            logging.warning("Dynamic shrinkage failed (%s); using 0.85", e)
            shrinkage = 0.85
            df_oof = pd.DataFrame()
        logging.info("Dynamic shrinkage = %.4f", shrinkage)

        # --- 4. Refit Henery thetas on the SAME OOF ---
        theta_2 = theta_3 = None
        if not df_oof.empty:
            try:
                P_mat = np.column_stack([
                    df_oof['P_calibrated_A'].values,
                    df_oof['P_calibrated_B'].values,
                    (1.0 / df_oof['stop_sell_odds']).values
                        if 'stop_sell_odds' in df_oof.columns
                        else (1.0 / df_oof['win_odds']).values,
                ])
                df_oof['P_ens'] = stacker.predict(P_mat, df_oof['race_id'].values)
                theta_input = df_oof[['race_id', 'horse_code', 'finish_position']].copy()
                theta_input['P_model'] = df_oof['P_ens']
                theta_csv = artifact('wf_theta_input.csv')
                theta_input.to_csv(theta_csv, index=False)
                theta_fit = calibrate_global(csv_path=theta_csv, n_bootstrap=0)
                theta_2 = theta_fit['theta_2']
                theta_3 = theta_fit['theta_3']
                logging.info("theta_2=%.4f  theta_3=%.4f", theta_2, theta_3)
            except Exception as e:
                logging.warning("Theta refit failed: %s", e)

        # --- 5. (Optional) refit drift forecaster on growing window ---
        forecaster = _refit_drift_forecaster(train_start, train_end)

        # --- 6. Persist live config ---
        _persist_live_config('live_config.json',
                             shrinkage=shrinkage,
                             theta_2=theta_2, theta_3=theta_3)

        # --- 7. Backtest on the test window ---
        logging.info("Backtesting %s -> %s...", test_window_start, test_window_end)
        backtester = XGBEnsembleBacktester(
            db_url=DB_URL, stacker_path=stacker_path,
            shrinkage=shrinkage,
            starting_bankroll=bankroll,
            theta_2=theta_2 or 0.8824,
            theta_3=theta_3 or 0.7760,
            drift_forecaster=forecaster,
        )
        backtester.model_a = joblib.load(model_a_path)
        backtester.model_b = joblib.load(model_b_path)
        backtester.calibrator_a = joblib.load(calib_a_path)
        backtester.calibrator_b = joblib.load(calib_b_path)
        backtester.bankroll = bankroll
        backtester.initial_bankroll = bankroll

        monthly_ledger, bankroll = backtester.run_backtest(
            start_date=test_window_start, end_date=test_window_end)
        master_ledger.extend(monthly_ledger)
        print(f"Bankroll after {test_window_start}: ${bankroll:,.2f}")

    # --- Final summary ---
    print("\n" + "=" * 50)
    print("       FINAL OUT-OF-SAMPLE STITCHED RESULTS")
    print("=" * 50)
    if not master_ledger:
        print("No bets placed during walk-forward.")
        return pd.DataFrame()

    df_ledger = pd.DataFrame(master_ledger)
    export_path = artifact('walk_forward_master_ledger.csv')
    df_ledger.to_csv(export_path, index=False)

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
    return df_ledger


if __name__ == "__main__":
    run_walk_forward_validation()
