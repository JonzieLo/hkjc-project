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


def _refit_stratified_shrinkage(df_oof: pd.DataFrame,
                                stacker,
                                tail_odds_col: str = 'stop_sell_odds',
                                max_odds: float = 50.0,
                                global_default: float = 0.85,
                                min_band_n: int = 100,
                                ) -> dict[str, float]:
    """Per-band calibration shrinkage from OOF data.

    Same Laplace-smoothed actual/expected estimator as
    _refit_dynamic_shrinkage, but computed independently within each
    P_pub band. The deepest-longshot bin gets a more aggressive
    correction without compressing mid-range probabilities where the
    model is already calibrated.

    IMPORTANT: unlike _refit_dynamic_shrinkage, we do NOT restrict to
    the "+EV-as-judged-by-model" tail. Doing so is sample-selection
    bias for stratified fitting — the model's overconfidence is
    DEFINED by the gap between predicted and actual win rates ACROSS
    THE FULL BAND, including horses the model didn't pick as bets.
    Restricting to "+EV horses" measures conditional-on-being-picked
    calibration, which is necessarily close to 1.0 when the model is
    overconfident (because it picks horses it thinks will win).

    The previous (buggy) tail-filter version produced shrinkages near
    1.0 in the mid-range and 0.85 (default fallback) for longshots —
    LESS aggressive than the global 0.93, the opposite of what we want.

    Methodology
    -----------
    For each band B in STRATIFIED_SHRINKAGE_BANDS:
      1. Filter OOF rows to horses whose normalised P_pub falls in B.
      2. Cap odds at max_odds (drops 100:1+ horses where model has
         essentially no signal anyway).
      3. shr_B = (actual_wins + 1) / (sum(P_ens) + 1)   (Laplace)
      4. Clamp to [0.50, 1.00] for safety.
      5. Bands with < min_band_n keep the global_default.

    Returns
    -------
    dict mapping STRATIFIED_SHRINKAGE_NAMES → fitted shrinkage scalar.
    """
    from hkjc_engine.models.betting_policy import (
        STRATIFIED_SHRINKAGE_BANDS,
        STRATIFIED_SHRINKAGE_NAMES,
    )

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

    edges = STRATIFIED_SHRINKAGE_BANDS
    names = STRATIFIED_SHRINKAGE_NAMES

    # Cap extreme odds (100:1+) where model signal is too noisy to
    # contribute meaningfully. Does NOT filter on +EV — that's the
    # critical fix vs the v1 implementation.
    odds_mask = df[tail_odds_col] <= max_odds

    out: dict[str, float] = {}
    logging.info("Stratified shrinkage by P_pub band (Laplace-smoothed):")
    logging.info("  %-12s  %6s  %6s  %9s  %9s  %s",
                 "band", "N", "wins", "exp", "shr", "kept?")

    for i, name in enumerate(names):
        lo = edges[i]
        hi = edges[i + 1]
        band_mask = (df['P_pub'] >= lo) & (df['P_pub'] < hi) & odds_mask
        sub = df.loc[band_mask]
        n = len(sub)
        if n < min_band_n:
            logging.info("  %-12s  %6d  %6s  %9s  %9s  default (n<%d)",
                         name, n, '-', '-', f'{global_default:.4f}',
                         min_band_n)
            continue
        actual = int((sub['finish_position'] == 1).sum())
        expected = float(sub['P_ens'].sum())
        shr = (actual + 1.0) / (expected + 1.0)
        shr = float(min(max(shr, 0.50), 1.00))
        out[name] = shr
        logging.info("  %-12s  %6d  %6d  %9.2f  %9.4f  fit",
                     name, n, actual, expected, shr)

    return out


def _persist_live_config(path: str,
                         shrinkage,                              # float | dict
                         theta_2: float | None = None,
                         theta_3: float | None = None,
                         theta_place: float | None = None,
                         theta_model_place: float | None = None,
                         drift_stats: dict | None = None) -> None:
    """Persist live config. shrinkage can be a float (legacy uniform)
    or a dict {band_name: shrinkage}. The bot reads the same key
    regardless and dispatches via lookup_shrinkage.
    """
    cfg: dict = {}
    if isinstance(shrinkage, dict):
        cfg['shrinkage_stratified'] = {k: float(v) for k, v in shrinkage.items()}
    else:
        cfg['shrinkage'] = float(shrinkage)
    if theta_2 is not None:           cfg['theta_2'] = float(theta_2)
    if theta_3 is not None:           cfg['theta_3'] = float(theta_3)
    if theta_place is not None:       cfg['theta_place'] = float(theta_place)
    if theta_model_place is not None: cfg['theta_model_place'] = float(theta_model_place)
    if drift_stats:                   cfg['drift_stats'] = drift_stats
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
                                pools: tuple[str, ...] | None = None,
                                exotic_odds_source: str = 'real',
                                theta_place: float | None = None,
                                theta_model_place: float | None = None,
                                stratified_shrinkage: bool = False,
                                ) -> pd.DataFrame:
    """
    Parameters
    ----------
    pools : tuple of str, optional
        If None (default), runs WIN-only via XGBEnsembleBacktester
        — this is the legacy path, identical to previous behaviour.
        If a tuple like ('WIN', 'PLA', 'QIN', 'QPL', 'TRI'), runs the
        MultiPoolBacktester instead. The multi-pool path settles all
        five pools against `race_dividends`. WIN always uses real
        STOP_SELL odds; exotic pricing is governed by
        `exotic_odds_source`. See `MultiPoolBacktester` docstring.
    exotic_odds_source : {'real', 'synthetic', 'hybrid'}
        How exotic pools are priced:
        - 'real' (default): only fire exotic bets when live_odds_history
          has a snapshot for the (race, pool, combo). WIN bets fire
          on every race regardless. Best accuracy, smaller exotic
          sample. Use this once you have live coverage.
        - 'synthetic': always use θ-symmetric Harville projections of
          public WIN-pool implieds. Larger exotic sample, but ROI is
          an approximation (real exotic markets don't price as exact
          Harville-from-WIN). Use only for stress-testing bet selection
          on the full historical window.
        - 'hybrid': real when present, synthetic otherwise.
    theta_place : float, optional
        Henery exponent for the PLA PUBLIC side (synthetic pub_odds).
        Fit against P_pub via theta_place_diagnostic.py. Default: θ_2.
    theta_model_place : float, optional
        Henery exponent for the PLA MODEL side (pp_model). Fit against
        P_ens via theta_place_diagnostic --model_oof_csv. Lower than
        theta_place corrects model longshot overconfidence.
        Default: same as theta_place.
    stratified_shrinkage : bool, default False
        If True, fit shrinkage separately within each P_pub band
        (deepest-longshot, longshot, mid, fav, heavy-fav) instead of one
        global scalar. Addresses model overconfidence on the longshot
        tail at the source rather than per-pool. The fitted dict is
        passed to all backtester sizing paths and persisted in
        live_config.json. See _refit_stratified_shrinkage.
    """
    test_end = test_end or datetime.now().strftime('%Y-%m-%d')
    test_months = pd.date_range(start=test_start, end=test_end, freq='MS')

    # Window-construction edge cases handled below:
    #
    # 1. `test_end` is mid-month (e.g. today is April 29, test_end='2026-04-29'):
    #    `freq='MS'` only emits month-start anchors. Without intervention the
    #    last anchor would be April 1 and we'd close one window short of
    #    test_end. Append test_end so a final April 1 → April 29 window forms.
    #
    # 2. `test_end` already lands on a month-start: don't append a duplicate.
    #
    # 3. test_start to test_end is entirely within a single partial month
    #    (e.g. test_start='2026-04-06', test_end='2026-04-29'): there are no
    #    month-start anchors in range, so test_months would be empty. Insert
    #    test_start as the leading anchor so we still get one window.
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts   = pd.Timestamp(test_end)
    if len(test_months) == 0:
        test_months = pd.DatetimeIndex([test_start_ts])
    if test_months[-1] < test_end_ts:
        test_months = test_months.append(pd.DatetimeIndex([test_end_ts]))

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
        # NOTE: XGBIndependentTrainer.train() does not accept oof_csv_path —
        # it hardcodes the output filename to 'model_b_oof_predictions.csv'
        # in the current working directory, which is what we want here.
        trainer_b.train(start_date=train_start, end_date=train_end,
                        save_path=model_b_path,
                        calibrator_path=calib_b_path)

        # --- 2. Fit stacker on STOP_SELL P_mkt ---
        logging.info("Fitting log-linear stacker (STOP_SELL P_mkt)...")
        stacker = EnsembleOptimizer(
            'model_a_oof_predictions.csv',
            'model_b_oof_predictions.csv',
            market_anchor='stop_sell',
        ).fit_stacker()
        joblib.dump(stacker, stacker_path)

        # --- 3. Dynamic shrinkage on STOP_SELL tail ---
        # Choice of fit: global scalar (legacy) vs per-band stratified
        # (recommended once theta_place_diagnostic confirms model
        # overconfidence on longshots). Stratified addresses the bias at
        # the input level — every projection that consumes P_ens (PLA,
        # QIN, QPL, TRI) gets corrected automatically.
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
            if stratified_shrinkage:
                shrinkage = _refit_stratified_shrinkage(
                    df_oof, stacker, tail_odds_col='stop_sell_odds')
            else:
                shrinkage = _refit_dynamic_shrinkage(
                    df_oof, stacker, tail_odds_col='stop_sell_odds')
        except Exception as e:
            logging.warning("Dynamic shrinkage failed (%s); using 0.85", e)
            shrinkage = 0.85
            df_oof = pd.DataFrame()

        if isinstance(shrinkage, dict):
            logging.info("Stratified shrinkage fitted: %s",
                         {k: f"{v:.4f}" for k, v in shrinkage.items()})
        else:
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
                             theta_2=theta_2, theta_3=theta_3,
                             theta_place=theta_place,
                             theta_model_place=theta_model_place)

        # --- 7. Backtest on the test window ---
        logging.info("Backtesting %s -> %s%s...",
                     test_window_start, test_window_end,
                     f" [pools={','.join(pools)}]" if pools else "")
        if pools:
            from hkjc_engine.models.multi_pool_backtester import MultiPoolBacktester
            backtester = MultiPoolBacktester(
                db_url=DB_URL, stacker_path=stacker_path,
                shrinkage=shrinkage,           # float OR dict — backtester handles both
                starting_bankroll=bankroll,
                theta_2=theta_2 or 0.8824,
                theta_3=theta_3 or 0.7760,
                drift_forecaster=forecaster,
                pools=pools,
                exotic_odds_source=exotic_odds_source,
                theta_place=theta_place,
                theta_model_place=theta_model_place,
            )
        else:
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

    # Per-pool stitched breakdown — answers 'which pool drove the result?'
    # across the whole walk-forward, not just the final window.
    if 'pool' in df_ledger.columns:
        print("\n" + "=" * 50)
        print("       PER-POOL STITCHED BREAKDOWN")
        print("=" * 50)
        print(f"{'Pool':<6} {'Bets':>6} {'Wins':>6} {'WinRate':>8} "
              f"{'Staked':>12} {'Profit':>12} {'ROI':>8}")
        for pool, g in df_ledger.groupby('pool'):
            n = len(g)
            w = int(g['is_win'].sum())
            s = float(g['stake'].sum())
            pf = float(g['profit'].sum())
            wr = (w / n * 100) if n else 0.0
            r  = (pf / s * 100) if s else 0.0
            print(f"{pool:<6} {n:>6} {w:>6} {wr:>7.2f}% "
                  f"${s:>11,.0f} ${pf:>+11,.0f} {r:>+7.2f}%")
        print("=" * 50)

        # Per-(pool, odds_source) breakdown if the run mixed sources
        if ('odds_source' in df_ledger.columns
                and df_ledger['odds_source'].nunique() > 1):
            print("\n" + "-" * 50)
            print("    BY ODDS SOURCE")
            print("-" * 50)
            print(f"{'Pool':<6} {'Source':<10} {'Bets':>6} "
                  f"{'WinRate':>8} {'ROI':>8}")
            for (pool, src), g in df_ledger.groupby(['pool', 'odds_source']):
                n = len(g); w = int(g['is_win'].sum())
                s = float(g['stake'].sum()); pf = float(g['profit'].sum())
                wr = (w / n * 100) if n else 0.0
                r  = (pf / s * 100) if s else 0.0
                print(f"{pool:<6} {src:<10} {n:>6} {wr:>7.2f}% {r:>+7.2f}%")
            print("-" * 50)

    return df_ledger


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Walk-forward backtest with optional multi-pool sizing.")
    ap.add_argument('--train_start', default='2018-01-01')
    ap.add_argument('--test_start',  default='2024-01-01')
    ap.add_argument('--test_end',    default=None)
    ap.add_argument('--bankroll', type=float, default=100_000.0)
    ap.add_argument('--pools', default=None,
                    help="Comma-separated subset of WIN,PLA,QIN,QPL,TRI. "
                         "If unset, runs WIN-only (legacy behaviour).")
    ap.add_argument('--exotic_odds_source', default='real',
                    choices=['real', 'synthetic', 'hybrid'],
                    help="How exotic pools (PLA/QIN/QPL/TRI) are priced. "
                         "Ignored when --pools omits all exotics. See "
                         "MultiPoolBacktester docstring for details. "
                         "Default 'real'.")
    ap.add_argument('--theta_place', type=float, default=None,
                    help="Henery exponent for PLA public-side place "
                         "projection (synthetic pub_odds, debug header). "
                         "Fit via theta_place_diagnostic.py against P_pub. "
                         "Default: θ_2.")
    ap.add_argument('--theta_model_place', type=float, default=None,
                    help="Henery exponent for PLA model-side place "
                         "projection (pp_model). Fit against P_ens via "
                         "theta_place_diagnostic --model_oof_csv "
                         "artifacts/wf_theta_input.csv. "
                         "Default: same as theta_place.")
    ap.add_argument('--stratified_shrinkage', action='store_true',
                    help="Fit shrinkage per P_pub band instead of one "
                         "global scalar. Addresses model longshot "
                         "overconfidence at the input level — propagates "
                         "to all pools (PLA/QIN/QPL/TRI) automatically.")
    args = ap.parse_args()

    pools_tuple = None
    if args.pools:
        pools_tuple = tuple(p.strip().upper() for p in args.pools.split(','))

    run_walk_forward_validation(
        train_start=args.train_start,
        test_start=args.test_start,
        test_end=args.test_end,
        starting_bankroll=args.bankroll,
        pools=pools_tuple,
        exotic_odds_source=args.exotic_odds_source,
        theta_place=args.theta_place,
        theta_model_place=args.theta_model_place,
        stratified_shrinkage=args.stratified_shrinkage,
    )