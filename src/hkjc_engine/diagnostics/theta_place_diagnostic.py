"""
theta_place_diagnostic.py
=========================

Fits θ_place by maximum likelihood on observed binary place outcomes.

Background
----------
Your existing `theta_diagnostic.py` fits (θ_2, θ_3) jointly against
exacta and trifecta outcomes — i.e. against the FULL ordering of top-3.
That likelihood weights races by how well the model predicts the exact
1st-2nd-3rd permutation, which is dominated by short-priced winners
because trifectas with longshots in any position are individually rare.

The PLACE market doesn't care about ordering. A horse places if it
finishes top-3, period. The optimal θ for projecting marginal place
probabilities is not the same θ that minimises trifecta NLL — they're
different objectives, and using the wrong one produces the systematic
bias you see in `place_projection_diagnostic.py`:

    HKJC data, theta_2=0.81, theta_3=0.71:
      p_pub < 0.02:  predicted 0.0647, actual 0.0442 — 47% over
      p_pub > 0.50:  predicted 0.8992, actual 0.8640 —  4% over

This module fits a single θ_place such that the binary-place log-
likelihood is minimised:

    L(θ) = Σ horses [ y * log(p_place_henery(p_pub, θ, θ))
                    + (1-y) * log(1 - p_place_henery(...)) ]

where y ∈ {0,1} is observed top-3 indicator.

Both 2nd and 3rd-place exponents in the projection use the same θ,
because the binary place objective doesn't have a separate signal for
'how does third place differ from second.' That separation requires
joint trifecta data, which the existing theta_diagnostic already
handles.

Once fitted, drop θ_place into `multi_pool_backtester.MultiPoolBacktester`
via the new `theta_place` constructor parameter. `p_place` will use
this for marginal place projection while `p_trio` and exacta logic
keep using the joint-fit θ_2 / θ_3 from the existing diagnostic.

Usage
-----
    python -m hkjc_engine.diagnostics.theta_place_diagnostic \\
        --start_date 2018-01-01 --end_date 2026-04-29

For a model-side fit (using P_model instead of P_pub), supply
`--model_oof_csv model_a_oof_predictions.csv`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL
from hkjc_engine.diagnostics.place_projection_diagnostic import (
    fetch_race_data, fetch_p_model_oof, _normalise_per_race, p_place_henery, _p_place_all_vectorised
)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,        # stdout, not stderr — avoids NativeCommandError
)
# Force UTF-8 on Windows so θ renders correctly instead of \u03b8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def _race_place_logloss(theta: float,
                        races_p: list[np.ndarray],
                        races_y: list[np.ndarray]) -> float:
    """Negative log-likelihood of binary place outcomes under
    Henery-θ projection.

    Uses _p_place_all_vectorised to compute all horses in a race
    simultaneously — O(n²) per race instead of O(n³). On HKJC data
    (n≈12, ~5,400 races) this reduces one MLE evaluation from ~40s
    to ~0.3s, making the 50-iteration bootstrap feasible in ~2 minutes
    instead of ~5 hours.
    """
    from hkjc_engine.diagnostics.place_projection_diagnostic import (
        _p_place_all_vectorised,
    )
    eps = 1e-12
    total_nll = 0.0
    for p_arr, y in zip(races_p, races_y):
        if len(p_arr) < 3 or np.any(p_arr <= 0):
            continue
        pp = _p_place_all_vectorised(p_arr, theta)   # shape (n,) all at once
        pp = np.clip(pp, eps, 1.0 - eps)
        y_arr = np.asarray(y, dtype=float)
        total_nll -= float(
            (y_arr * np.log(pp) + (1.0 - y_arr) * np.log(1.0 - pp)).sum()
        )
    return total_nll


def _prepare_race_arrays(df: pd.DataFrame,
                         prob_col: str = 'p_pub',
                         ) -> tuple[list, list, int]:
    """Group the long-format DataFrame by race and produce
    (per-race probability arrays, per-race place-indicator arrays).
    """
    races_p, races_y = [], []
    dropped = 0
    for race_id, g in df.groupby('race_id', sort=False):
        p_arr = g[prob_col].to_numpy(dtype=float)
        y = g['is_placer'].to_numpy(dtype=int)
        if len(p_arr) < 3 or np.any(p_arr <= 0) or np.any(np.isnan(p_arr)):
            dropped += 1
            continue
        races_p.append(p_arr)
        races_y.append(y)
    return races_p, races_y, dropped


# ---------------------------------------------------------------------------
# Fit and bootstrap
# ---------------------------------------------------------------------------

def fit_theta_place(races_p: list[np.ndarray],
                    races_y: list[np.ndarray],
                    bracket: tuple[float, float] = (0.4, 1.5),
                    ) -> dict:
    """1-D MLE for θ_place. Bracket extends past 1.0 because the
    diagnostic suggests the optimum could be near 1 (Harville-like) on
    HKJC data — we don't want to constrain to θ < 1 as the fit goal."""
    res = minimize_scalar(
        _race_place_logloss,
        args=(races_p, races_y),
        bounds=bracket, method='bounded',
        options={'xatol': 1e-5, 'maxiter': 200},
    )
    return {
        'theta_place': float(res.x),
        'nll': float(res.fun),
        'nll_per_race': float(res.fun / max(len(races_p), 1)),
        'n_races': len(races_p),
        'converged': bool(res.success),
    }


def bootstrap_theta_place(races_p: list[np.ndarray],
                          races_y: list[np.ndarray],
                          n_boot: int = 50,
                          seed: int = 42,
                          ) -> dict:
    """Race-level bootstrap SE on θ_place. Sample races with
    replacement (NOT individual horses — races are the natural
    independent unit)."""
    rng = np.random.default_rng(seed)
    n = len(races_p)
    estimates = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_p = [races_p[i] for i in idx]
        sub_y = [races_y[i] for i in idx]
        res = fit_theta_place(sub_p, sub_y)
        estimates.append(res['theta_place'])
        if (b + 1) % 10 == 0:
            log.info("  bootstrap %d/%d done...", b + 1, n_boot)
    arr = np.array(estimates)
    return {
        'mean':  float(arr.mean()),
        'std':   float(arr.std()),
        'p2.5':  float(np.percentile(arr, 2.5)),
        'p97.5': float(np.percentile(arr, 97.5)),
        'n_boot': n_boot,
    }


# ---------------------------------------------------------------------------
# Calibration check (post-fit)
# ---------------------------------------------------------------------------

def post_fit_calibration(df: pd.DataFrame,
                         theta_place: float,
                         theta_2_old: float,
                         theta_3_old: float,
                         prob_col: str = 'p_pub') -> pd.DataFrame:
    """For every horse, compute place-prob projections under three
    parameter settings:
      * old: Henery(p, θ_2_exacta, θ_3_trifecta)  — what the bot uses today
      * new: Henery(p, θ_place, θ_place)           — proposed
      * harville: Henery(p, 1.0, 1.0)              — for context

    Returns a binned calibration table comparing all three to empirical.
    """
    pp_old      = np.zeros(len(df))
    pp_new      = np.zeros(len(df))
    pp_harville = np.zeros(len(df))
    for race_id, g in df.groupby('race_id', sort=False):
        p_arr = g[prob_col].values
        idx = g.index.values
        if len(p_arr) < 3 or np.any(p_arr <= 0):
            continue
        pp_old[idx]      = _p_place_all_vectorised(p_arr, theta_2_old)
        pp_new[idx]      = _p_place_all_vectorised(p_arr, theta_place)
        pp_harville[idx] = _p_place_all_vectorised(p_arr, 1.0)
    df = df.copy()
    df['pp_old'], df['pp_new'], df['pp_harville'] = pp_old, pp_new, pp_harville

    edges = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.50, 1.01])
    df['bin'] = pd.cut(df[prob_col], bins=edges, include_lowest=True)
    rows = []
    for bin_label, g in df.groupby('bin', observed=True):
        if len(g) == 0:
            continue
        rows.append({
            'bin':      str(bin_label),
            'n':        len(g),
            'emp':      float(g['is_placer'].mean()),
            'old':      float(g['pp_old'].mean()),
            'new':      float(g['pp_new'].mean()),
            'harville': float(g['pp_harville'].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(start_date: str, end_date: str,
        theta_2_old: float, theta_3_old: float,
        n_boot: int = 0,
        model_oof_csv: str | None = None) -> None:
    engine = create_engine(DB_URL)
    df = fetch_race_data(engine, start_date, end_date)
    if df.empty:
        return
    log.info("Loaded %d horses across %d races (%s -> %s).",
             len(df), df['race_id'].nunique(), start_date, end_date)

    df['p_pub_raw'] = 1.0 / df['win_odds']
    df['p_pub'] = df.groupby('race_id')['p_pub_raw'].transform(_normalise_per_race)
    df['is_placer'] = df['finish_position'].between(1, 3).astype(int)

    races_p, races_y, dropped = _prepare_race_arrays(df, prob_col='p_pub')
    if dropped:
        log.info("Dropped %d races (small fields or invalid probs).", dropped)
    log.info("Fitting θ_place against %d races (P_pub side)...",
             len(races_p))

    res = fit_theta_place(races_p, races_y)
    log.info("\n" + "=" * 64)
    log.info("   θ_place FIT RESULTS — P_pub side")
    log.info("=" * 64)
    log.info("  θ_place fitted  : %.4f", res['theta_place'])
    log.info("  NLL             : %.2f", res['nll'])
    log.info("  NLL per race    : %.4f", res['nll_per_race'])
    log.info("  Races used      : %d", res['n_races'])
    log.info("  Converged       : %s", res['converged'])

    if n_boot > 0:
        log.info("\nBootstrapping θ_place SE (%d iterations)...", n_boot)
        boot = bootstrap_theta_place(races_p, races_y, n_boot=n_boot)
        log.info("  Bootstrap mean  : %.4f", boot['mean'])
        log.info("  Bootstrap SE    : %.4f", boot['std'])
        log.info("  95%% CI          : [%.4f, %.4f]",
                 boot['p2.5'], boot['p97.5'])

    log.info("\n" + "=" * 64)
    log.info("   POST-FIT CALIBRATION COMPARISON (P_pub side)")
    log.info("=" * 64)
    cal = post_fit_calibration(df, res['theta_place'],
                                theta_2_old, theta_3_old, prob_col='p_pub')
    log.info(f"{'P_pub bin':<14} {'N':>5}  {'emp':>7}  "
             f"{'OLD(θ_2,θ_3)':>13} {'NEW(θ_place)':>13} {'Harville':>9}")
    for _, r in cal.iterrows():
        log.info(f"{r['bin']:<14} {r['n']:>5}  {r['emp']:>7.4f}  "
                 f"{r['old']:>13.4f} {r['new']:>13.4f} "
                 f"{r['harville']:>9.4f}")

    # Rough goodness summary: sum of |residual| across bins, weighted by N
    def total_abs_resid(col):
        return float((cal['n'] * (cal['emp'] - cal[col]).abs()).sum()
                     / cal['n'].sum())
    log.info("")
    log.info("Mean abs residual (lower is better):")
    log.info("  OLD (θ_2, θ_3) : %.5f", total_abs_resid('old'))
    log.info("  NEW (θ_place)  : %.5f", total_abs_resid('new'))
    log.info("  Harville (θ=1) : %.5f", total_abs_resid('harville'))
    log.info("=" * 64)

    # Model-side fit if OOF supplied
    if model_oof_csv:
        log.info("\nModel-side fit requested...")
        p_model_df = fetch_p_model_oof(model_oof_csv)
        if p_model_df is None or 'horse_code' not in df.columns:
            log.warning("Cannot do model-side fit (missing data).")
        else:
            df_m = df.merge(p_model_df[['race_id', 'horse_code', 'p_model']],
                            on=['race_id', 'horse_code'], how='inner')
            df_m['p_model'] = df_m.groupby('race_id')['p_model'].transform(
                _normalise_per_race)
            races_p_m, races_y_m, _ = _prepare_race_arrays(df_m,
                                                            prob_col='p_model')
            res_m = fit_theta_place(races_p_m, races_y_m)
            log.info("  θ_place (model side): %.4f", res_m['theta_place'])
            log.info("  Compare to P_pub side: %.4f", res['theta_place'])
            if abs(res['theta_place'] - res_m['theta_place']) > 0.05:
                log.info("  -> Sides DIVERGE; the model is contributing")
                log.info("     extra distortion. Investigate model calibration.")
            else:
                log.info("  -> Sides AGREE; θ_place is robust across "
                         "probability source.")

    # Recommendation
    log.info("\n" + "=" * 64)
    log.info("RECOMMENDATION")
    log.info("=" * 64)
    log.info("Drop the fitted θ_place into multi_pool_backtester.py:")
    log.info("    theta_place = %.4f", res['theta_place'])
    log.info("If walk_forward.py refits per window (recommended),")
    log.info("persist this value in live_config.json alongside θ_2 / θ_3.")
    log.info("=" * 64)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start_date', default='2018-01-01')
    ap.add_argument('--end_date',   default='2026-04-29')
    ap.add_argument('--theta_2',    type=float, default=0.8127,
                    help="Existing exacta-fit θ_2 (for OLD-vs-NEW comparison).")
    ap.add_argument('--theta_3',    type=float, default=0.7098,
                    help="Existing trifecta-fit θ_3 (for comparison).")
    ap.add_argument('--n_boot',     type=int, default=0,
                    help="Bootstrap iterations for SE on θ_place (0 = skip).")
    ap.add_argument('--model_oof_csv', default=None,
                    help="Path to OOF CSV for model-side theta fit. "
                         "Use artifacts/wf_theta_input.csv (produced by "
                         "walk_forward.py) — it contains the full stacker "
                         "ensemble P_ens as the P_model column, which is "
                         "what the bot actually bets on. Passing "
                         "model_a_oof_predictions.csv also works (uses "
                         "P_calibrated = Model A only) but is less "
                         "representative of the live policy.")
    args = ap.parse_args()
    run(args.start_date, args.end_date,
        args.theta_2, args.theta_3,
        args.n_boot, args.model_oof_csv)