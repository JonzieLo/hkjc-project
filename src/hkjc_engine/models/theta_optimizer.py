"""
Benter discounted Harville theta calibration via MLE.

Drop-in replacement for src/hkjc_engine/models/theta_optimizer.py.

Two operating modes:

    GLOBAL (default — backward compatible):
        Fits a single (theta_2, theta_3) on the full input. Identical interface
        to the previous version: prints results, returns a dict.

    STRATIFIED (--stratify):
        Fits independent (theta_2, theta_3) per (distance_bucket, field_bucket,
        class_bucket). Used to answer the question "does theta vary with race
        context?" before committing to any architectural change.

        If the input CSV is missing the columns needed for stratification
        (distance, field_size, race_class), this mode will join to PostgreSQL
        on `race_id` to enrich them. Set --no-db-enrich to disable.

In both modes, bootstrap standard errors on theta are reported, and bins with
fewer than --min-races-per-bin races fall back to the parent stratum (drop
class -> drop field -> use global). This prevents unstable point estimates
from being acted on.

Input CSV: race_id, horse_code, finish_position, P_model
        (race_date column is optional but enables --start-date / --end-date)

Output:
    --output PATH.json : structured fit table with per-bin theta + SE + n
    stdout             : human-readable summary

Usage:
    # Global fit only (existing behavior)
    python theta_optimizer.py --input ensemble_oof_results.csv

    # Stratified fit, write to JSON for run_bot.py to consume at inference
    python theta_optimizer.py --input ensemble_oof_results.csv \\
        --stratify --output contextual_thetas.json --bootstrap 200

    # Stratified, but skip DB enrichment (CSV must already have race covariates)
    python theta_optimizer.py --input ensemble_oof_results.csv \\
        --stratify --no-db-enrich --output contextual_thetas.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stratification taxonomy
# ---------------------------------------------------------------------------

# Distance-only stratification. Two bins, calibrated to the empirical sample-
# size distribution: with ~2k OOF races, splitting at 1400m gives ~900 per bin,
# which lands bootstrap SE on theta_2 in the 0.025-0.030 range -- enough power
# to detect a 0.05+ spread as a real difference rather than noise. Earlier
# 12-bin (distance x field x class) scheme was underpowered: 79% of races fell
# into two bins, the other ten were below n=200 and fell back to global anyway.
# Field-size and class can be reintroduced once OOF sample exceeds ~6k races.
DISTANCE_BUCKETS = [(0, 1400, "sprint"), (1400, 9999, "route")]

# A bin needs at least this many fully-resolved (1st/2nd/3rd present) races
# before we trust its independent fit.
DEFAULT_MIN_RACES_PER_BIN = 400


def _bucket_label(value, buckets):
    for lo, hi, lab in buckets:
        if lo <= value < hi:
            return lab
    return None


def _bin_key(distance, field_size, race_class):
    """field_size and race_class accepted for signature compatibility but
    deliberately ignored under the distance-only scheme."""
    d = _bucket_label(distance, DISTANCE_BUCKETS)
    return d


# ---------------------------------------------------------------------------
# Likelihood functions (vectorized over races)
# ---------------------------------------------------------------------------

def _race_blocks(df: pd.DataFrame):
    """
    Returns lists of per-race arrays needed for likelihood evaluation.
    Drops races with dead heats, missing placings, or non-positive probs.
    """
    df = df.sort_values(['race_id', 'finish_position']).reset_index(drop=True)
    df['P_model'] = df['P_model'] / df.groupby('race_id')['P_model'].transform('sum')

    probs_list, p1_list, p2_list, p3_list, race_ids_kept = [], [], [], [], []
    dropped = 0
    for race_id, race in df.groupby('race_id', sort=False):
        fp = race['finish_position'].values
        pm = race['P_model'].values.astype(float)
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
        race_ids_kept.append(race_id)
    if dropped:
        log.info(f"Dropped {dropped} races (dead heats / missing placings / invalid probs).")
    return (probs_list,
            np.array(p1_list), np.array(p2_list), np.array(p3_list),
            race_ids_kept)


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


# ---------------------------------------------------------------------------
# Fit a single bin
# ---------------------------------------------------------------------------

def _fit_one(probs_list, p1, p2, p3) -> dict[str, Any]:
    """Two-stage MLE: theta_2 from exacta, then joint (theta_2, theta_3)."""
    res2 = minimize(
        _exacta_nll, x0=[0.85], args=(probs_list, p1, p2),
        bounds=[(0.4, 1.0)], method='L-BFGS-B',
        options={'ftol': 1e-10, 'maxiter': 200},
    )
    t2_init = float(res2.x[0])
    res3 = minimize(
        _trifecta_nll_joint, x0=[t2_init, 0.75],
        args=(probs_list, p1, p2, p3),
        bounds=[(0.4, 1.0), (0.3, 1.0)], method='L-BFGS-B',
        options={'ftol': 1e-10, 'maxiter': 300},
    )
    return {
        'theta_2_exacta':  t2_init,
        'theta_2':         float(res3.x[0]),
        'theta_3':         float(res3.x[1]),
        'trifecta_nll':    float(res3.fun),
        'trifecta_nll_per_race': float(res3.fun / max(len(probs_list), 1)),
        'n_races':         len(probs_list),
    }


def _bootstrap_se(probs_list, p1, p2, p3, n_boot, seed=42) -> dict[str, float]:
    """Bootstrap standard errors on (theta_2, theta_3). Race-level resampling."""
    if n_boot <= 0 or len(probs_list) < 50:
        return {'theta_2_se': float('nan'), 'theta_3_se': float('nan')}
    rng = np.random.default_rng(seed)
    n = len(probs_list)
    t2_samples, t3_samples = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_probs = [probs_list[i] for i in idx]
        boot_p1 = p1[idx]; boot_p2 = p2[idx]; boot_p3 = p3[idx]
        try:
            fit = _fit_one(boot_probs, boot_p1, boot_p2, boot_p3)
            t2_samples.append(fit['theta_2'])
            t3_samples.append(fit['theta_3'])
        except Exception:
            continue
    if len(t2_samples) < n_boot // 2:
        return {'theta_2_se': float('nan'), 'theta_3_se': float('nan')}
    return {
        'theta_2_se': float(np.std(t2_samples, ddof=1)),
        'theta_3_se': float(np.std(t3_samples, ddof=1)),
        'theta_2_q025': float(np.quantile(t2_samples, 0.025)),
        'theta_2_q975': float(np.quantile(t2_samples, 0.975)),
        'theta_3_q025': float(np.quantile(t3_samples, 0.025)),
        'theta_3_q975': float(np.quantile(t3_samples, 0.975)),
    }


# ---------------------------------------------------------------------------
# DB enrichment for stratified mode
# ---------------------------------------------------------------------------

def _enrich_from_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins distance / race_class from the `races` table, computes field_size
    from `race_entries`. No-op if all three columns already exist on df.
    """
    needed = ['distance', 'race_class', 'field_size']
    have = [c for c in needed if c in df.columns]
    if len(have) == 3:
        return df

    log.info("Enriching from DB: missing %s", set(needed) - set(have))
    try:
        from sqlalchemy import create_engine, text
        from hkjc_engine.config import DB_URL
    except ImportError as e:
        raise RuntimeError(
            "DB enrichment required but hkjc_engine.config not importable. "
            "Either provide distance/field_size/race_class columns in the input "
            "CSV, or run with --no-db-enrich."
        ) from e

    race_ids = df['race_id'].unique().tolist()
    if not race_ids:
        return df
    engine = create_engine(DB_URL)
    races_q = text("""
        SELECT race_id, distance, race_class
        FROM races
        WHERE race_id IN :rids
    """)
    field_q = text("""
        SELECT race_id, COUNT(*) AS field_size
        FROM race_entries
        WHERE race_id IN :rids AND finish_position IS NOT NULL
        GROUP BY race_id
    """)
    with engine.connect() as conn:
        rmeta = pd.read_sql(races_q, conn, params={'rids': tuple(race_ids)})
        fmeta = pd.read_sql(field_q, conn, params={'rids': tuple(race_ids)})
    enriched = df.merge(rmeta, on='race_id', how='left', suffixes=('', '_db'))
    enriched = enriched.merge(fmeta, on='race_id', how='left', suffixes=('', '_db'))
    n_missing = enriched['distance'].isna().sum()
    if n_missing:
        log.warning("%d rows could not be enriched from DB (race_id not in races table)",
                    n_missing)
    return enriched.dropna(subset=['distance', 'race_class', 'field_size'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_global(csv_path: str = 'ensemble_oof_results.csv',
                     start_date: str | None = None,
                     end_date:   str | None = None,
                     n_bootstrap: int = 0) -> dict[str, Any]:
    """Backward-compatible global fit."""
    log.info(f"Loading {csv_path} ...")
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
        log.info(f"Date-filtered to {len(df)} rows.")

    df = df.dropna(subset=['race_id', 'finish_position', 'P_model'])
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
    df = df.dropna(subset=['finish_position'])
    df['finish_position'] = df['finish_position'].astype(int)

    probs_list, p1, p2, p3, _ = _race_blocks(df)
    if len(probs_list) < 50:
        raise RuntimeError(f"Too few valid races for stable MLE: {len(probs_list)}")

    log.info(f"Retained {len(probs_list)} races for MLE.")
    log.info("Fitting global (theta_2, theta_3) ...")
    fit = _fit_one(probs_list, p1, p2, p3)
    if n_bootstrap > 0:
        log.info(f"Bootstrapping SE with {n_bootstrap} resamples ...")
        fit.update(_bootstrap_se(probs_list, p1, p2, p3, n_bootstrap))

    log.info("\n" + "=" * 50)
    log.info("       THETA CALIBRATION (GLOBAL)")
    log.info("=" * 50)
    log.info(f"Races used                : {fit['n_races']}")
    log.info(f"theta_2 (exacta-only)     : {fit['theta_2_exacta']:.4f}")
    log.info(f"theta_2 (joint trifecta)  : {fit['theta_2']:.4f}"
             + (f"  (SE {fit['theta_2_se']:.4f})" if 'theta_2_se' in fit else ''))
    log.info(f"theta_3 (joint trifecta)  : {fit['theta_3']:.4f}"
             + (f"  (SE {fit['theta_3_se']:.4f})" if 'theta_3_se' in fit else ''))
    log.info(f"Trifecta NLL (per race)   : {fit['trifecta_nll_per_race']:.4f}")
    log.info("=" * 50)
    log.info("\nDrop these into run_bot.py:")
    log.info(f"    THETA_2 = {fit['theta_2']:.4f}")
    log.info(f"    THETA_3 = {fit['theta_3']:.4f}")
    return fit


def calibrate_stratified(csv_path: str,
                         output_path: str | None = None,
                         min_races_per_bin: int = DEFAULT_MIN_RACES_PER_BIN,
                         enrich_from_db: bool = True,
                         n_bootstrap: int = 0,
                         start_date: str | None = None,
                         end_date:   str | None = None) -> dict[str, Any]:
    """Per-bin fit with global fallback. Writes structured JSON for inference."""
    log.info(f"Loading {csv_path} ...")
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

    df = df.dropna(subset=['race_id', 'finish_position', 'P_model'])
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce')
    df = df.dropna(subset=['finish_position'])
    df['finish_position'] = df['finish_position'].astype(int)

    if enrich_from_db:
        df = _enrich_from_db(df)
    else:
        for col in ('distance', 'field_size', 'race_class'):
            if col not in df.columns:
                raise ValueError(
                    f"--no-db-enrich set but column '{col}' missing from CSV. "
                    "Either include the column or drop --no-db-enrich."
                )

    df['_bin'] = df.apply(
        lambda r: _bin_key(r['distance'], r['field_size'], r['race_class']),
        axis=1,
    )
    df = df.dropna(subset=['_bin'])

    # Global fallback fit (always computed)
    probs_g, p1_g, p2_g, p3_g, _ = _race_blocks(df.drop(columns=['_bin']))
    if len(probs_g) < 50:
        raise RuntimeError(f"Too few valid races for global fallback fit: {len(probs_g)}")
    log.info(f"\nGlobal fallback: fitting on {len(probs_g)} races ...")
    global_fit = _fit_one(probs_g, p1_g, p2_g, p3_g)
    if n_bootstrap > 0:
        global_fit.update(_bootstrap_se(probs_g, p1_g, p2_g, p3_g, n_bootstrap))
    log.info(f"  GLOBAL  t2={global_fit['theta_2']:.4f}  "
             f"t3={global_fit['theta_3']:.4f}  n={global_fit['n_races']}")

    # Per-bin fits
    per_bin: dict[str, dict] = {}
    summary_rows = []
    for bin_key, sub in df.groupby('_bin', sort=True):
        probs, p1, p2, p3, _ = _race_blocks(sub.drop(columns=['_bin']))
        n = len(probs)
        if n < min_races_per_bin:
            log.warning(f"  {bin_key:<22} n={n:<5d}  -> too few, will fall back to global")
            summary_rows.append({
                'bin': bin_key, 'n_races': n,
                'theta_2': None, 'theta_3': None, 'fallback': True,
            })
            continue
        fit = _fit_one(probs, p1, p2, p3)
        if n_bootstrap > 0:
            fit.update(_bootstrap_se(probs, p1, p2, p3, n_bootstrap))
        per_bin[bin_key] = fit
        msg = (f"  {bin_key:<22} n={n:<5d}  "
               f"t2={fit['theta_2']:.4f}  t3={fit['theta_3']:.4f}  "
               f"NLL/race={fit['trifecta_nll_per_race']:.4f}")
        if 'theta_2_se' in fit:
            msg += f"  (SE2={fit['theta_2_se']:.3f}, SE3={fit['theta_3_se']:.3f})"
        log.info(msg)
        summary_rows.append({
            'bin': bin_key, 'n_races': n,
            'theta_2': fit['theta_2'], 'theta_3': fit['theta_3'],
            'theta_2_se': fit.get('theta_2_se'),
            'theta_3_se': fit.get('theta_3_se'),
            'nll_per_race': fit['trifecta_nll_per_race'],
            'fallback': False,
        })

    # ----- Variance check: is theta really context-dependent? -----
    if per_bin:
        t2s = np.array([b['theta_2'] for b in per_bin.values()])
        t3s = np.array([b['theta_3'] for b in per_bin.values()])
        spread_t2 = float(t2s.max() - t2s.min())
        spread_t3 = float(t3s.max() - t3s.min())
        # Compare spread to typical bootstrap SE of the bins
        med_se2 = (np.median([b.get('theta_2_se', np.nan) for b in per_bin.values()])
                   if n_bootstrap > 0 else float('nan'))
        med_se3 = (np.median([b.get('theta_3_se', np.nan) for b in per_bin.values()])
                   if n_bootstrap > 0 else float('nan'))
        log.info("\n" + "=" * 70)
        log.info("       STRATIFICATION DIAGNOSTIC")
        log.info("=" * 70)
        log.info(f"theta_2 across bins: min={t2s.min():.4f}  max={t2s.max():.4f}  "
                 f"spread={spread_t2:.4f}")
        log.info(f"theta_3 across bins: min={t3s.min():.4f}  max={t3s.max():.4f}  "
                 f"spread={spread_t3:.4f}")
        if n_bootstrap > 0:
            log.info(f"Median bootstrap SE: t2={med_se2:.4f}  t3={med_se3:.4f}")
            ratio2 = spread_t2 / med_se2 if med_se2 > 0 else float('nan')
            ratio3 = spread_t3 / med_se3 if med_se3 > 0 else float('nan')
            log.info(f"Spread / median-SE:  t2={ratio2:.2f}x  t3={ratio3:.2f}x")
            if ratio2 < 2.5 and ratio3 < 2.5:
                log.info(">>> Spread is comparable to noise. Stratification probably "
                         "NOT justified. Stick with global theta.")
            elif ratio2 > 4.0 or ratio3 > 4.0:
                log.info(">>> Spread substantially exceeds noise. Stratification likely "
                         "improves calibration; deploy contextual table.")
            else:
                log.info(">>> Marginal evidence for context-dependence. Validate via "
                         "held-out NLL before deploying.")
        log.info("=" * 70)

    # ----- Output -----
    out = {
        '_global':  global_fit,
        '_bins':    per_bin,
        '_meta': {
            'min_races_per_bin': min_races_per_bin,
            'n_bootstrap':       n_bootstrap,
            'distance_buckets':  DISTANCE_BUCKETS,
            'scheme':            'distance_only_v2',
            'csv_path':          csv_path,
        },
    }
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        log.info(f"\nWrote contextual theta table to {output_path}")

    return out


def get_thetas_for_race(distance: float, field_size: int, race_class: str,
                        table: dict) -> tuple[float, float]:
    """
    Inference helper for run_bot.py. Returns (theta_2, theta_3) for the given
    race covariates, falling back to global if the bin has insufficient data.
    """
    bin_key = _bin_key(distance, field_size, race_class)
    if bin_key and bin_key in table.get('_bins', {}):
        b = table['_bins'][bin_key]
        return float(b['theta_2']), float(b['theta_3'])
    g = table['_global']
    return float(g['theta_2']), float(g['theta_3'])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='ensemble_oof_results.csv',
                   help='CSV with race_id, horse_code, finish_position, P_model')
    p.add_argument('--start-date', default=None,
                   help='Optional ISO date filter (requires race_date column)')
    p.add_argument('--end-date', default=None)
    p.add_argument('--stratify', action='store_true',
                   help='Fit per-(distance,field,class) bin instead of global only')
    p.add_argument('--output', default=None,
                   help='[--stratify only] Write JSON theta table to this path')
    p.add_argument('--min-races-per-bin', type=int, default=DEFAULT_MIN_RACES_PER_BIN)
    p.add_argument('--no-db-enrich', action='store_true',
                   help='[--stratify only] Require covariate columns on input CSV')
    p.add_argument('--bootstrap', type=int, default=0,
                   help='Bootstrap resamples for theta SE (0 = skip)')
    args = p.parse_args()

    if args.stratify:
        if not args.output:
            log.warning("--stratify without --output: results printed but not saved")
        calibrate_stratified(
            csv_path=args.input,
            output_path=args.output,
            min_races_per_bin=args.min_races_per_bin,
            enrich_from_db=not args.no_db_enrich,
            n_bootstrap=args.bootstrap,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        calibrate_global(
            csv_path=args.input,
            start_date=args.start_date,
            end_date=args.end_date,
            n_bootstrap=args.bootstrap,
        )


if __name__ == "__main__":
    main()


# Backward-compatibility alias for any importer of the old function name.
calibrate_thetas = calibrate_global