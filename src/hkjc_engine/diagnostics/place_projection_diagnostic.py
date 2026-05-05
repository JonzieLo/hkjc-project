"""
pla_model_diagnostic.py (formerly place_projection_diagnostic.py)
=================================================================

Evaluates the native PLA residual model (Model A PLA) calibration under the 
Multi-Agent Architecture.

Since we no longer project PLA odds from WIN odds using the Henery formula, 
this diagnostic directly bins the native `P_calibrated` output from the PLA 
trainer and compares it to the empirical `is_placed` hit rate.

Usage
-----
    python -m hkjc_engine.diagnostics.place_projection_diagnostic \\
        --oof_csv model_a_pla_oof_predictions.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
log = logging.getLogger(__name__)

def _binned_calibration(df: pd.DataFrame, p_col: str, edges: np.ndarray) -> pd.DataFrame:
    """Bin horses by `p_col`, compute empirical place rate vs predicted."""
    bins = pd.cut(df[p_col], bins=edges, include_lowest=True)
    grouped = df.groupby(bins, observed=True)
    
    rows = []
    for bin_label, g in grouped:
        if len(g) == 0:
            continue
            
        n = len(g)
        empirical = float(g['is_placed'].mean())
        predicted = float(g[p_col].mean())
        
        rows.append({
            'bin': str(bin_label),
            'n_horses': n,
            'empirical_rate': empirical,
            'predicted_rate': predicted,
            'residual': empirical - predicted  # Positive = Model Underestimates
        })
    return pd.DataFrame(rows)

def run_diagnostic(oof_csv: str) -> None:
    p = Path(oof_csv)
    if not p.exists():
        log.error(f"OOF CSV {oof_csv} not found. Please check the path.")
        return

    df = pd.read_csv(p)
    required_cols = {'race_id', 'is_placed', 'P_calibrated'}
    if not required_cols.issubset(df.columns):
        log.error(f"CSV missing required columns. Found: {df.columns.tolist()}")
        return

    df = df.dropna(subset=['P_calibrated', 'is_placed'])
    
    log.info(f"Loaded {len(df)} entries across {df['race_id'].nunique()} races.")
    overall_emp = df['is_placed'].mean()
    overall_pred = df['P_calibrated'].mean()
    log.info(f"Overall Empirical Place Rate: {overall_emp:.4f}")
    log.info(f"Overall Predicted Place Rate: {overall_pred:.4f}\n")

    # Define bins focusing heavily on the longshot and mid-ranges
    edges = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.01])
    
    cal = _binned_calibration(df, 'P_calibrated', edges)

    log.info("=" * 75)
    log.info("   NATIVE PLA MODEL CALIBRATION (Binned by P_calibrated)")
    log.info("=" * 75)
    log.info("Negative residual = Model OVERESTIMATES probability (Overconfident)")
    log.info("Positive residual = Model UNDERESTIMATES probability (Underconfident)\n")
    
    log.info(f"{'P_model bin':<16} {'N_Horses':>8}  {'Empirical':>12} {'Predicted':>12}  {'Residual':>12}")
    log.info("-" * 75)
    
    for _, r in cal.iterrows():
        log.info(f"{r['bin']:<16} {r['n_horses']:>8}  "
                 f"{r['empirical_rate']:>12.4f} "
                 f"{r['predicted_rate']:>12.4f}  "
                 f"{r['residual']:>12.4f}")
    log.info("=" * 75)

    # Calculate Mean Absolute Error across bins (weighted by N)
    mae = np.average(np.abs(cal['residual']), weights=cal['n_horses'])
    log.info(f"\nWeighted Mean Absolute Error (MAE): {mae:.5f}")
    
    if mae < 0.015:
        log.info(">>> VERDICT: Excellent Calibration. The native PLA model is highly accurate.")
    elif mae < 0.030:
        log.info(">>> VERDICT: Acceptable Calibration. Minor deviations, but structurally sound.")
    else:
        log.info(">>> VERDICT: Poor Calibration. The model is systematically biased.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof_csv', default='model_a_pla_oof_predictions.csv',
                    help="Path to the Model A PLA OOF predictions CSV.")
    args = ap.parse_args()

    run_diagnostic(args.oof_csv)