"""
Benter discounted Harville theta calibration via MLE.

Ingests a CSV with (race_id, horse_code, finish_position, P_model) where
P_model is the final calibrated ensemble win probability. Fits theta_2 and
theta_3 by maximum likelihood on actual 1st/2nd/3rd finishes, following
Henery (1981) and Lo & Bacon-Shone (2008).

Usage:
    python theta_optimizer.py --input ensemble_oof_results.csv
"""
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _race_blocks(df):
    """
    Return per-race arrays needed for vectorized likelihood:
        probs_list : list of 1D np.ndarray (all P_model per race, renormalized)
        p1, p2, p3 : 1D arrays indexed by race (winner / 2nd / 3rd probability)
    Only races with valid 1st/2nd/3rd are retained.
    """
    df = df.sort_values(['race_id', 'finish_position']).reset_index(drop=True)

    # Defensive per-race renormalization
    df['P_model'] = df['P_model'] / df.groupby('race_id')['P_model'].transform('sum')

    probs_list, p1_list, p2_list, p3_list = [], [], [], []
    dropped = 0
    for _, race in df.groupby('race_id', sort=False):
        fp = race['finish_position'].values
        pm = race['P_model'].values.astype(float)
        # Must have at least one row at each of 1, 2, 3 (no ties, no missing)
        if (fp == 1).sum() != 1 or (fp == 2).sum() != 1 or (fp == 3).sum() != 1:
            dropped += 1
            continue
        if len(pm) < 3 or np.any(pm <= 0):
            dropped += 1
            continue
        probs_list.append(pm)
        p1_list.append(pm[fp == 1][0])
        p2_list.append(pm[fp == 2][0])
        p3_list.append(pm[fp == 3][0])

    if dropped:
        logging.info(f"Dropped {dropped} races (dead heats / missing placings / invalid probs).")
    logging.info(f"Retained {len(probs_list)} races for MLE.")
    return probs_list, np.array(p1_list), np.array(p2_list), np.array(p3_list)

# Likelihoods
def _exacta_nll(theta, probs_list, p1, p2):
    t2 = float(theta[0])
    nll = 0.0
    for pm, p_1, p_2 in zip(probs_list, p1, p2):
        sum_t2 = np.sum(pm ** t2)
        den2 = sum_t2 - p_1 ** t2
        if den2 <= 0:
            return 1e12
        prob = p_1 * (p_2 ** t2 / den2)
        nll -= np.log(max(prob, 1e-15))
    return nll


def _trifecta_nll_joint(theta, probs_list, p1, p2, p3):
    """Joint 2D NLL over (theta_2, theta_3) on trifecta chain."""
    t2, t3 = float(theta[0]), float(theta[1])
    nll = 0.0
    for pm, p_1, p_2, p_3 in zip(probs_list, p1, p2, p3):
        sum_t2 = np.sum(pm ** t2)
        sum_t3 = np.sum(pm ** t3)
        den2 = sum_t2 - p_1 ** t2
        den3 = sum_t3 - p_1 ** t3 - p_2 ** t3
        if den2 <= 0 or den3 <= 0:
            return 1e12
        prob = p_1 * (p_2 ** t2 / den2) * (p_3 ** t3 / den3)
        nll -= np.log(max(prob, 1e-15))
    return nll


def calibrate_thetas(csv_path='ensemble_oof_results.csv',
                     start_date=None, end_date=None):
    logging.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    required = {'race_id', 'horse_code', 'finish_position', 'P_model'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if 'race_date' in df.columns and (start_date or end_date):
        df['race_date'] = pd.to_datetime(df['race_date'])
        if start_date:
            df = df[df['race_date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['race_date'] < pd.Timestamp(end_date)]
        logging.info(f"Date-filtered to {len(df)} rows.")

    df = df.dropna(subset=['race_id', 'finish_position', 'P_model'])
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
    df = df.dropna(subset=['finish_position'])
    df['finish_position'] = df['finish_position'].astype(int)

    probs_list, p1, p2, p3 = _race_blocks(df)
    if len(probs_list) < 50:
        raise RuntimeError(f"Too few valid races for stable MLE: {len(probs_list)}")

    logging.info("\nStage 1: Fitting theta_2 from exacta likelihood (sanity check)...")
    res2 = minimize(
        _exacta_nll, x0=[0.85], args=(probs_list, p1, p2),
        bounds=[(0.4, 1.0)], method='L-BFGS-B',
        options={'ftol': 1e-10, 'maxiter': 200},
    )
    t2_exacta = float(res2.x[0])
    logging.info(f"  theta_2 (exacta-only MLE) = {t2_exacta:.4f}   NLL = {res2.fun:.2f}")

    logging.info("\nStage 2: Jointly fitting (theta_2, theta_3) on trifecta likelihood...")
    res3 = minimize(
        _trifecta_nll_joint,
        x0=[t2_exacta, 0.75],
        args=(probs_list, p1, p2, p3),
        bounds=[(0.4, 1.0), (0.3, 1.0)],
        method='L-BFGS-B',
        options={'ftol': 1e-10, 'maxiter': 300},
    )
    t2_joint, t3_joint = float(res3.x[0]), float(res3.x[1])

    logging.info("\n" + "=" * 50)
    logging.info("       THETA CALIBRATION RESULTS")
    logging.info("=" * 50)
    logging.info(f"Races used                : {len(probs_list)}")
    logging.info(f"theta_2 (exacta-only)     : {t2_exacta:.4f}")
    logging.info(f"theta_2 (joint trifecta)  : {t2_joint:.4f}")
    logging.info(f"theta_3 (joint trifecta)  : {t3_joint:.4f}")
    logging.info(f"Joint trifecta NLL        : {res3.fun:.2f}")
    logging.info("=" * 50)
    logging.info("\nDrop these into run_live_bot.py:")
    logging.info(f"    THETA_2 = {t2_joint:.4f}")
    logging.info(f"    THETA_3 = {t3_joint:.4f}")

    return {'theta_2_exacta': t2_exacta,
            'theta_2_joint':  t2_joint,
            'theta_3_joint':  t3_joint,
            'trifecta_nll':   float(res3.fun),
            'n_races':        len(probs_list)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='ensemble_oof_results.csv', help='CSV with race_id, horse_code, finish_position, P_model')
    p.add_argument('--start-date', default=None, help='Optional: ISO date, only used if race_date column present')
    p.add_argument('--end-date', default=None)
    args = p.parse_args()
    calibrate_thetas(args.input, args.start_date, args.end_date)
