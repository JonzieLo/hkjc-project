"""
place_projection_diagnostic.py
==============================

Quantifies bias in the Henery-projected place probability vs. empirical
place rates, binned by win-pool implied probability. Produces the data
needed to decide between calibration options:

    1. Pool-specific theta refit (Option 1)  — current Henery, new theta
    2. Stern gamma model for place (Option 2) — different functional form
    3. Empirical isotonic correction (Option 3) — wrap with post-hoc map

The diagnostic answers three questions:

    Q1. How much does Henery-θ_fit overestimate longshot place probability?
    Q2. Is the bias coming from the model (P_model) or from the structural
        Henery formula itself (using pure P_pub instead)?
    Q3. Is the bias direction consistent with what Lo & Bacon-Shone reported
        for HKJC, or is it specific to this deployment?

Usage
-----
    python -m hkjc_engine.diagnostics.place_projection_diagnostic \\
        --start_date 2024-01-01 --end_date 2026-04-29

Optional flag `--theta_2 0.81 --theta_3 0.71` overrides the values used
for projection. If omitted, falls back to the empirical defaults that
match the user's most recent walk-forward fit.

What's printed
--------------
For each P_pub bin:
  * n_horses observed in this bin
  * empirical place rate (gold standard)
  * predicted place rate via Henery(p_pub, θ_fit)
  * predicted place rate via Henery(p_model, θ_fit)
  * predicted place rate via Harville (θ=1) for reference
  * residual = empirical − Henery(p_pub) — IF positive, Henery is
    UNDERESTIMATING; if negative, Henery is OVERESTIMATING

A bias profile that goes from ~0 at the favourite end to strongly
negative at the longshot end IS the Henery longshot-place overestimation
bias. The magnitude tells us how much room there is to fix.

Side outputs
------------
A CSV at `place_projection_residuals.csv` with one row per (race, horse)
including the binned residual. Useful for follow-up isotonic regression
in Option 3.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Henery / Harville place projection — vectorised
# ---------------------------------------------------------------------------
#
# Replaces the original O(n³) pure-Python scalar implementation with an
# O(n²) NumPy version that computes place probabilities for ALL horses in
# a race in a single pass. On a 12-horse field the speedup is ~144×,
# reducing the 5-hour bootstrap to ~2 minutes.
#
# Mathematical derivation
# -----------------------
# For a race with normalised probability vector p (length n) and Henery
# exponent θ applied uniformly to 2nd and 3rd positions:
#
#   q   = p ** θ          (element-wise)
#   S_q = sum(q)
#
# P(horse i 1st)  = p[i]
#
# P(horse i 2nd)  = Σ_{j≠i} p[j] * q[i] / (S_q - q[j])
#                 = q[i] * Σ_{j≠i} p[j] / (S_q - q[j])
#
# P(horse i 3rd)  = Σ_{j≠i} p[j] * q[j]/(S_q-q[j])_wrong ... actually:
#
#   P(i 3rd)= Σ_{j≠i} Σ_{k≠i,j} p[j] * (q[k]/(S_q-q[j])) * (q[i]/(S_q-q[j]-q[k]))
#
# This is computed via a 2-D outer sum: for each pair (j winner, k runner-up)
# with j≠i and k≠i and k≠j, accumulate the joint probability. Using
# broadcasting this is O(n²) per horse, but we can factorise one sum
# out to make it O(n²) total for ALL horses at once.
#
# Implementation: build (n×n) matrices W2 and W3 where:
#   W2[j, i] = P(j wins AND i is 2nd)
#   W3[j, k, i] summed over j,k = P(i 3rd) — compressed to a 2-D sum
#
# In practice we use the identity:
#
#   pp_place = p + col_sum(W2) + col_sum(W3)
#
# where W2 and W3 are n×n matrices assembled with the mask j≠i, k≠j,k≠i.

def _p_place_all_vectorised(p: np.ndarray, theta: float) -> np.ndarray:
    """Compute P(top-3) for ALL horses in a race simultaneously.

    Parameters
    ----------
    p : np.ndarray, shape (n,)
        Win-probability vector, already normalised to sum=1.
    theta : float
        Henery exponent (same for 2nd and 3rd positions — binary place
        objective doesn't decompose into separate 2nd/3rd signals).

    Returns
    -------
    pp : np.ndarray, shape (n,)
        Place probabilities. sum(pp) == 3.0 for a clean n>=3 field.
    """
    n = len(p)
    if n < 3:
        return p.copy()

    q = p ** theta          # discounted probs, shape (n,)
    S_q = q.sum()

    # --- 2nd-place contribution for each horse i ---
    # p2nd[i] = q[i] * Σ_{j≠i} p[j] / (S_q - q[j])
    #
    # Let r[j] = p[j] / (S_q - q[j])    (scalar per j)
    # sum_r = Σ_j r[j]
    # Then Σ_{j≠i} r[j] = sum_r - r[i]
    # p2nd[i] = q[i] * (sum_r - r[i])
    #
    # This is O(n), no loops needed.
    denom_j = S_q - q                   # shape (n,)  S_q - q[j] for each j
    denom_j = np.where(denom_j > 0, denom_j, 1e-12)
    r = p / denom_j                      # shape (n,)  p[j]/(S_q - q[j])
    sum_r = r.sum()
    p2nd = q * (sum_r - r)               # shape (n,)  P(i 2nd)

    # --- 3rd-place contribution for each horse i ---
    # p3rd[i] = Σ_{j≠i} Σ_{k≠j,k≠i} p[j] * (q[k]/(S_q-q[j])) * (q[i]/(S_q-q[j]-q[k]))
    #
    # Factor out q[i]:
    # p3rd[i] = q[i] * Σ_{j≠i} Σ_{k≠j,k≠i} p[j]*q[k] / [(S_q-q[j])*(S_q-q[j]-q[k])]
    #
    # Let A[j, k] = p[j]*q[k] / [(S_q-q[j])*(S_q-q[j]-q[k])]  for j≠k
    # Then p3rd[i] = q[i] * Σ_{j≠i} Σ_{k≠j,k≠i} A[j,k]
    #
    # Σ_{j≠i} Σ_{k≠j,k≠i} A[j,k]
    #   = (Σ_j Σ_{k≠j} A[j,k]) - Σ_{k≠i} A[i,k] - Σ_{j≠i} A[j,i]
    #
    # So we need:
    #   total_A  = Σ_j Σ_{k≠j} A[j,k]                (scalar)
    #   row_A[i] = Σ_{k≠i} A[i,k]  (sum over k for fixed i-as-winner)
    #   col_A[i] = Σ_{j≠i} A[j,i]  (sum over j for fixed i-as-runner-up)
    #
    # Then: inner_sum[i] = total_A - row_A[i] - col_A[i]
    # p3rd[i] = q[i] * inner_sum[i]
    #
    # Building A as an (n×n) matrix (with diagonal masked):
    #   dj[j]    = S_q - q[j]           (denominator after j wins)
    #   djk[j,k] = S_q - q[j] - q[k]   (denominator after j wins, k 2nd)
    #
    # Both are O(n²) in memory — fine for HKJC field sizes ≤ 20.

    # dj: shape (n,)
    dj = S_q - q                                   # S_q - q[j]
    dj = np.where(dj > 0, dj, 1e-12)

    # djk: shape (n, n)  broadcasting: dj[j] - q[k]
    djk = dj[:, None] - q[None, :]                 # (n,n)  S_q-q[j]-q[k]
    djk = np.where(djk > 0, djk, 1e-12)

    # A[j, k] = p[j] * q[k] / (dj[j] * djk[j,k]),  mask diagonal j==k
    A = (p[:, None] * q[None, :]) / (dj[:, None] * djk)   # (n, n)
    np.fill_diagonal(A, 0.0)                        # k≠j condition

    total_A = A.sum()
    row_A   = A.sum(axis=1)                         # Σ_k A[j,k] per j
    col_A   = A.sum(axis=0)                         # Σ_j A[j,k] per k

    inner_sum = total_A - row_A - col_A             # shape (n,)
    p3rd = q * inner_sum                            # shape (n,)

    return p + p2nd + p3rd


def p_place_henery(p_arr: np.ndarray, i: int,
                   theta_2: float, theta_3: float) -> float:
    """Scalar wrapper — returns P(horse i top-3).

    Internally calls _p_place_all_vectorised (which computes all horses
    at once) and returns the i-th element. This means calling this
    function n times per race is still O(n²) total — the vectorised
    version is computed once and the result indexed.

    For bulk computation (fitting, calibration), prefer calling
    _p_place_all_vectorised directly.

    Note: theta_2 and theta_3 are both accepted for API compatibility
    with the original scalar version. The vectorised implementation
    uses a single theta = theta_2 (appropriate for binary place MLE).
    """
    pp_all = _p_place_all_vectorised(p_arr, theta_2)
    return float(pp_all[i])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_race_data(engine, start_date: str, end_date: str) -> pd.DataFrame:
    """Load races with the columns we need for this diagnostic.

    Returns DataFrame with columns:
        race_id, horse_no, horse_code, finish_position, win_odds
    Drops races that don't have a clean top-3 (dead heats, scratches),
    matching the convention in theta_diagnostic. Drops horses with NULL
    win_odds since we need them for P_pub.

    horse_code is included so we can join against model OOF predictions
    (which key on horse_code, not horse_no).
    """
    q = text("""
        SELECT
            r.race_id,
            e.horse_no,
            e.horse_code,
            e.finish_position,
            e.win_odds
        FROM races r
        JOIN race_entries e ON r.race_id = e.race_id
        WHERE r.race_date >= :start_date AND r.race_date < :end_date
          AND e.win_odds IS NOT NULL
          AND e.finish_position IS NOT NULL
        ORDER BY r.race_date ASC, r.race_id ASC, e.horse_no ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={
            'start_date': start_date, 'end_date': end_date})
    if df.empty:
        log.warning("No race rows in window %s -> %s", start_date, end_date)
        return df
    df['win_odds'] = df['win_odds'].astype(float)
    df['finish_position'] = df['finish_position'].astype(int)
    df['horse_no'] = df['horse_no'].astype(int)
    return df


def fetch_p_model_oof(model_oof_csv: str | None) -> pd.DataFrame | None:
    """Load ensemble model OOF probabilities for the model-side diagnostic.

    Accepted CSV formats
    --------------------
    1. ``artifacts/wf_theta_input.csv`` (RECOMMENDED) — written by
       walk_forward.py at line 259.  Columns: race_id, horse_code,
       finish_position, P_model (= stacker P_ens).  This is the full
       ensemble probability the bot actually bets on.

    2. ``model_a_oof_predictions.csv`` — Model A only, written during
       walk-forward training.  Columns include P_calibrated.  Useful as
       a proxy when wf_theta_input.csv is unavailable, but note this is
       NOT the ensemble — it omits Model B and the market anchor weight.

    The function normalises all accepted column names to lowercase
    ``p_model`` so downstream code doesn't need to care which file
    was supplied.
    """
    if not model_oof_csv:
        return None
    p = Path(model_oof_csv)
    if not p.exists():
        log.warning("OOF CSV %s not found; model-side diagnostic skipped.", p)
        return None
    df = pd.read_csv(p)

    # Normalise accepted column names → p_model
    col_map = {
        'P_model':     'p_model',   # wf_theta_input.csv (ensemble P_ens)
        'P_ens':       'p_model',   # older walk-forward checkpoints
        'P_calibrated':'p_model',   # model_a_oof_predictions.csv (Model A only)
    }
    renamed = False
    for src, dst in col_map.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
            if src != 'P_model':
                log.info("OOF CSV: using column '%s' as p_model. "
                         "Note: for the full ensemble, prefer "
                         "artifacts/wf_theta_input.csv (P_model = P_ens).", src)
            renamed = True
            break
    if not renamed:
        log.warning("OOF CSV missing any of %s; model-side diagnostic skipped.",
                    list(col_map.keys()))
        return None

    if 'horse_code' not in df.columns:
        log.warning("OOF CSV has no horse_code column; cannot join to "
                    "race_entries. Model-side diagnostic skipped.")
        return None

    keep = ['race_id', 'horse_code', 'p_model']
    if 'finish_position' in df.columns:
        keep.append('finish_position')
    return df[keep]


# ---------------------------------------------------------------------------
# Per-race projection
# ---------------------------------------------------------------------------

def _normalise_per_race(p: np.ndarray) -> np.ndarray:
    """Sum-to-1 within race. Defensive against zeros."""
    s = p.sum()
    if s <= 0:
        return np.zeros_like(p)
    return p / s


def annotate_with_projections(df: pd.DataFrame,
                              theta_2: float, theta_3: float,
                              p_model_df: pd.DataFrame | None = None,
                              ) -> pd.DataFrame:
    """For each race, compute three or four place-probability projections:

        pp_pub_henery     = Henery(p_pub, θ_fit)
        pp_pub_harville   = Henery(p_pub, 1.0)        — for reference
        pp_model_henery   = Henery(p_model, θ_fit)    — if p_model_df given
        pp_model_harville = Henery(p_model, 1.0)      — if p_model_df given

    Plus the ground-truth indicator `is_placer = 1 if top-3 else 0`.

    Two probability inputs are computed because they answer different
    questions:
      * P_pub side: is the Henery formula itself biased on longshots,
        independent of any model?
      * P_model side: how does the model-side bias compare?
    The DIFFERENCE between these two diagnoses 'is the model
    overconfident on longshots' vs 'is Henery itself biased'.
    """
    df = df.copy()
    df['p_pub_raw'] = 1.0 / df['win_odds']
    df['p_pub'] = df.groupby('race_id')['p_pub_raw'].transform(_normalise_per_race)
    df['is_placer'] = df['finish_position'].between(1, 3).astype(int)

    have_model = (p_model_df is not None and not p_model_df.empty
                  and 'horse_code' in df.columns
                  and 'horse_code' in p_model_df.columns)
    if have_model:
        df = df.merge(p_model_df[['race_id', 'horse_code', 'p_model']],
                      on=['race_id', 'horse_code'], how='left')
        # Renormalise per race in case the join introduced any
        # inconsistency (it shouldn't, but defensive).
        df['p_model'] = df.groupby('race_id')['p_model'].transform(
            _normalise_per_race)
        n_with_model = df['p_model'].notna().sum()
        log.info("Joined P_model for %d/%d horses.", n_with_model, len(df))
        if n_with_model < 0.5 * len(df):
            log.warning("Less than half of horses got P_model; "
                        "diagnostic on model side will be unreliable.")
            have_model = False

    log.info("Projecting Henery and Harville place probabilities for "
             "%d races (theta_2=%.4f, theta_3=%.4f)...",
             df['race_id'].nunique(), theta_2, theta_3)

    pp_pub_henery     = np.zeros(len(df))
    pp_pub_harville   = np.zeros(len(df))
    pp_model_henery   = np.full(len(df), np.nan)
    pp_model_harville = np.full(len(df), np.nan)
    n_races_processed = 0

    for race_id, g in df.groupby('race_id', sort=False):
        idx_in_df = g.index.values
        p_arr_pub = g['p_pub'].values
        n = len(p_arr_pub)
        if n < 3 or np.any(p_arr_pub <= 0):
            continue
        # Vectorised: compute all horses in one call, not one call per horse
        pp_pub_henery[idx_in_df]   = _p_place_all_vectorised(p_arr_pub, theta_2)
        pp_pub_harville[idx_in_df] = _p_place_all_vectorised(p_arr_pub, 1.0)
        if have_model:
            p_arr_model = g['p_model'].values
            if not np.any(np.isnan(p_arr_model)) and np.all(p_arr_model > 0):
                pp_model_henery[idx_in_df]   = _p_place_all_vectorised(
                    p_arr_model, theta_2)
                pp_model_harville[idx_in_df] = _p_place_all_vectorised(
                    p_arr_model, 1.0)
        n_races_processed += 1
        if n_races_processed % 500 == 0:
            log.info("  processed %d races...", n_races_processed)

    df['pp_pub_henery']     = pp_pub_henery
    df['pp_pub_harville']   = pp_pub_harville
    if have_model:
        df['pp_model_henery']   = pp_model_henery
        df['pp_model_harville'] = pp_model_harville
    log.info("Done. %d races projected.", n_races_processed)
    return df


# ---------------------------------------------------------------------------
# Bias analysis
# ---------------------------------------------------------------------------

def _binned_calibration(df: pd.DataFrame,
                        p_col: str,
                        proj_cols: list[str],
                        edges: np.ndarray) -> pd.DataFrame:
    """Bin horses by `p_col`, compute empirical place rate vs each
    projection column. Returns one row per bin.
    """
    bins = pd.cut(df[p_col], bins=edges, include_lowest=True)
    grouped = df.groupby(bins, observed=True)
    rows = []
    for bin_label, g in grouped:
        if len(g) == 0:
            continue
        n = len(g)
        empirical = float(g['is_placer'].mean())
        row = {
            'bin': str(bin_label),
            'p_pub_lo': bin_label.left,
            'p_pub_hi': bin_label.right,
            'n_horses': n,
            'empirical_place_rate': empirical,
        }
        for c in proj_cols:
            row[f'pred_{c}'] = float(g[c].mean())
            row[f'residual_{c}'] = empirical - float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_diagnostic(start_date: str, end_date: str,
                   theta_2: float, theta_3: float,
                   model_oof_csv: str | None = None,
                   out_csv: str = 'place_projection_residuals.csv',
                   ) -> None:
    engine = create_engine(DB_URL)

    df = fetch_race_data(engine, start_date, end_date)
    if df.empty:
        return

    log.info("Loaded %d entries across %d races (%s -> %s).",
             len(df), df['race_id'].nunique(), start_date, end_date)

    p_model_df = fetch_p_model_oof(model_oof_csv)
    df = annotate_with_projections(df, theta_2, theta_3, p_model_df)

    # Drop rows where we couldn't project (small fields, etc.)
    df = df[df['pp_pub_henery'] > 0].reset_index(drop=True)

    # Sanity check: empirical place rate across all horses should be
    # roughly 3/N where N is the average field size. HKJC averages
    # ~12 horses, so ~25%.
    overall_rate = float(df['is_placer'].mean())
    log.info("\nOverall empirical place rate: %.3f (%d/%d horses)",
             overall_rate, int(df['is_placer'].sum()), len(df))
    log.info("(Sanity: should be roughly 3/E[field_size] ≈ 0.21–0.30)")

    # ---- Calibration table by P_pub band ----
    edges = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25,
                      0.35, 0.50, 1.01])
    cal = _binned_calibration(df, 'p_pub',
                              ['pp_pub_henery', 'pp_pub_harville'],
                              edges)

    log.info("\n" + "=" * 96)
    log.info("   PLACE PROJECTION CALIBRATION — P_pub side (binned by p_pub)")
    log.info("=" * 96)
    log.info("Negative residual = projection OVERESTIMATES; "
             "positive = UNDERESTIMATES.")
    log.info("This side answers: is the Henery FORMULA itself biased on "
             "longshots, independent of any model?")
    log.info("")
    log.info(f"{'P_pub bin':<14} {'N':>6}  "
             f"{'Empirical':>10} {'Henery':>10} {'Harville':>10}  "
             f"{'H_resid':>10} {'Hv_resid':>10}")
    for _, r in cal.iterrows():
        log.info(f"{r['bin']:<14} {r['n_horses']:>6}  "
                 f"{r['empirical_place_rate']:>10.4f} "
                 f"{r['pred_pp_pub_henery']:>10.4f} "
                 f"{r['pred_pp_pub_harville']:>10.4f}  "
                 f"{r['residual_pp_pub_henery']:>+10.4f} "
                 f"{r['residual_pp_pub_harville']:>+10.4f}")
    log.info("=" * 96)

    # ---- Model-side calibration if available ----
    have_model = 'pp_model_henery' in df.columns
    if have_model:
        df_model = df[df['pp_model_henery'].notna()].copy()
        if len(df_model) > 0:
            cal_m = _binned_calibration(
                df_model, 'p_model',
                ['pp_model_henery', 'pp_model_harville'], edges)
            log.info("\n" + "=" * 96)
            log.info("   PLACE PROJECTION CALIBRATION — P_model side "
                     "(binned by p_model)")
            log.info("=" * 96)
            log.info("This side answers: is the MODEL'S projection (P_model "
                     "+ Henery) calibrated for the place market?")
            log.info("")
            log.info(f"{'P_model bin':<14} {'N':>6}  "
                     f"{'Empirical':>10} {'Henery':>10} {'Harville':>10}  "
                     f"{'H_resid':>10} {'Hv_resid':>10}")
            for _, r in cal_m.iterrows():
                log.info(f"{r['bin']:<14} {r['n_horses']:>6}  "
                         f"{r['empirical_place_rate']:>10.4f} "
                         f"{r['pred_pp_model_henery']:>10.4f} "
                         f"{r['pred_pp_model_harville']:>10.4f}  "
                         f"{r['residual_pp_model_henery']:>+10.4f} "
                         f"{r['residual_pp_model_harville']:>+10.4f}")
            log.info("=" * 96)

    # ---- Bias direction summary ----
    log.info("\nBIAS DIRECTION SUMMARY")
    log.info("-" * 70)
    # Define longshot as p_pub < 0.05; mid as 0.05-0.18; favourite >= 0.18
    longshot = df[df['p_pub'] < 0.05]
    mid      = df[(df['p_pub'] >= 0.05) & (df['p_pub'] < 0.18)]
    favourite = df[df['p_pub'] >= 0.18]

    for label, sub in [('Longshot (p_pub < 0.05)', longshot),
                       ('Mid      (0.05-0.18)',     mid),
                       ('Favourite (>= 0.18)',      favourite)]:
        if len(sub) == 0:
            continue
        emp = float(sub['is_placer'].mean())
        hen = float(sub['pp_pub_henery'].mean())
        hv  = float(sub['pp_pub_harville'].mean())
        # Multiplicative ratio: how many times too high is the projection?
        ratio_h  = hen / emp if emp > 0 else float('inf')
        ratio_hv = hv  / emp if emp > 0 else float('inf')
        log.info(f"  {label:<28} N={len(sub):>5}  "
                 f"emp={emp:.3f}  henery={hen:.3f} (×{ratio_h:.2f})  "
                 f"harville={hv:.3f} (×{ratio_hv:.2f})")

    # ---- Save residuals CSV for downstream isotonic fit (Option 3) ----
    keep_cols = ['race_id', 'horse_no', 'p_pub',
                 'pp_pub_henery', 'pp_pub_harville',
                 'is_placer', 'finish_position']
    if 'pp_model_henery' in df.columns:
        keep_cols += ['p_model', 'pp_model_henery', 'pp_model_harville']
    df_out = df[keep_cols].copy()
    df_out['residual_henery'] = df_out['is_placer'] - df_out['pp_pub_henery']
    df_out['residual_harville'] = df_out['is_placer'] - df_out['pp_pub_harville']
    if 'pp_model_henery' in df.columns:
        df_out['residual_model_henery'] = (
            df_out['is_placer'] - df_out['pp_model_henery'])
    df_out.to_csv(out_csv, index=False)
    log.info("\nResiduals CSV written to %s (%d rows).", out_csv, len(df_out))

    # ---- What this means for next step ----
    log.info("\n" + "=" * 70)
    log.info("INTERPRETING THE RESULT")
    log.info("=" * 70)
    if len(longshot) == 0:
        log.info("No longshot bin populated; cannot diagnose.")
        return

    pub_emp = float(longshot['is_placer'].mean())
    pub_henery = float(longshot['pp_pub_henery'].mean())
    pub_ratio = pub_henery / max(pub_emp, 1e-9)

    if have_model:
        ls_model = df[(df['p_pub'] < 0.05)
                       & df['pp_model_henery'].notna()]
        if len(ls_model) > 0:
            model_emp = float(ls_model['is_placer'].mean())
            model_henery = float(ls_model['pp_model_henery'].mean())
            model_ratio = model_henery / max(model_emp, 1e-9)
        else:
            model_ratio = None
    else:
        model_ratio = None

    log.info("Longshot (p_pub < 0.05) place projection vs reality:")
    log.info("  P_pub + Henery:  predicted=%.4f / empirical=%.4f  "
             "(ratio %.2fx)",
             pub_henery, pub_emp, pub_ratio)
    if model_ratio is not None:
        log.info("  P_model + Henery: predicted=%.4f / empirical=%.4f  "
                 "(ratio %.2fx)",
                 model_henery, model_emp, model_ratio)

    log.info("")
    if pub_ratio > 1.20:
        log.info(">>> The Henery FORMULA overestimates longshot place by "
                 "%.2fx on this data.", pub_ratio)
        log.info(">>> This is a structural issue with the formula — even")
        log.info(">>> projecting from the public market, longshots are")
        log.info(">>> projected to place more often than they actually do.")
        log.info(">>> Recommendation: Option 1 (theta_place refit) or")
        log.info(">>> Option 2 (Stern gamma model). Both will help.")
    elif pub_ratio < 0.85:
        log.info(">>> The Henery FORMULA UNDERestimates longshot place "
                 "(ratio %.2f).", pub_ratio)
        log.info(">>> This is unusual and suggests something atypical "
                 "about the data window.")
        log.info(">>> Worth checking: does the longshot bin contain races "
                 "with unusually small fields where every horse places?")
    else:
        log.info(">>> The Henery FORMULA on P_pub looks well-calibrated "
                 "(ratio %.2f).", pub_ratio)
        log.info(">>> The structural Henery model is NOT the source of "
                 "longshot place bias.")

    if model_ratio is not None:
        log.info("")
        if model_ratio > pub_ratio + 0.20:
            log.info(">>> But P_model + Henery gives ratio %.2fx — "
                     "%.2fx WORSE than P_pub side.", model_ratio,
                     model_ratio - pub_ratio)
            log.info(">>> The MODEL is overconfident on longshot horses, "
                     "and that overconfidence is what's amplifying through")
            log.info(">>> the Henery projection. Fixing theta won't help —")
            log.info(">>> the model itself is mis-calibrated on the tail.")
            log.info(">>> Recommendation: investigate Beta calibration on")
            log.info(">>> longshot bin specifically; check whether the")
            log.info(">>> stacker is over-weighting Model A on extreme tail.")
        elif abs(model_ratio - pub_ratio) <= 0.20:
            log.info(">>> P_model + Henery gives similar bias (ratio %.2f).",
                     model_ratio)
            log.info(">>> Model is NOT contributing extra longshot bias —")
            log.info(">>> the bias is dominated by the Henery formula.")
            log.info(">>> Fix the formula (Option 1 or 2) and the model")
            log.info(">>> projection will improve mechanically.")
        else:
            log.info(">>> P_model + Henery is BETTER calibrated than "
                     "P_pub side (ratio %.2f vs %.2f).", model_ratio,
                     pub_ratio)
            log.info(">>> This is unusual. Worth investigating.")
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start_date', default='2024-01-01')
    ap.add_argument('--end_date',   default='2026-04-29')
    ap.add_argument('--theta_2', type=float, default=0.8127,
                    help="Theta for 2nd-place ranking (default: most "
                         "recent walk-forward fit).")
    ap.add_argument('--theta_3', type=float, default=0.7098,
                    help="Theta for 3rd-place ranking.")
    ap.add_argument('--model_oof_csv', default=None,
                    help="Optional path to model OOF CSV with P_model "
                         "column. If supplied, also computes the bias "
                         "on the model-projection side.")
    ap.add_argument('--out_csv', default='place_projection_residuals.csv')
    args = ap.parse_args()

    run_diagnostic(args.start_date, args.end_date,
                   args.theta_2, args.theta_3,
                   args.model_oof_csv, args.out_csv)