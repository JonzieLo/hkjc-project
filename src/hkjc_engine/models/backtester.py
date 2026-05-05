"""
Walk-forward backtester (drift-aware refactor).

Key changes
-----------
1. Public market and Model A `base_margin` use STOP_SELL odds (when
   available in `live_odds_history`) rather than FINAL settled dividends.
2. `qualify_and_size` is called per pool with pool-conditional drift stats.
   When a `DriftForecaster` is supplied, drift stats come from the
   forecaster's predictions; otherwise the static `DEFAULT_DRIFT_STATS`
   from `betting_policy` are used.
3. Bet PnL still settles on FINAL `win_dividend` from `race_dividends`
   (that's what actually pays out), but the EV used to gate the bet uses
   STOP_SELL × median_R, matching live-execution conditions.
4. (Structural Update) Pace Archetypes and I_valid flags are computed at 
   inference time to correctly route PyMC stacker regimes and Stratified Isotonic Calibrators.
"""
from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL, artifact
from hkjc_engine.data.stop_sell_loader import attach_win_anchor
from hkjc_engine.models.feature_factory import (
    HKJCFeatureFactory,
    calculate_base_margin,
)
from hkjc_engine.models.betting_policy import (
    DEFAULT_DRIFT_STATS,
    DriftStats,
    qualify_and_size,
)
import hkjc_engine.models.ensemble  # noqa: F401  for joblib unpickling

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _drift_stats_for_race(forecaster, win_features_df: pd.DataFrame
                          ) -> dict[int, DriftStats]:
    """Return per-horse-index DriftStats by querying the forecaster.

    `win_features_df` is the feature dataframe for ONE race only. Returns
    a dict mapping local row index -> DriftStats. If forecaster is None
    or features are missing, returns an empty dict; caller falls back to
    DEFAULT_DRIFT_STATS['WIN'].
    """
    if forecaster is None or win_features_df is None or win_features_df.empty:
        return {}
    try:
        preds = forecaster.predict_win(win_features_df)
        if preds.empty:
            return {}
        out = {}
        for i, row in preds.reset_index(drop=True).iterrows():
            out[i] = DriftStats(
                mu_R=float(row['mu_R']),
                sigma_R=float(row['sigma_R']),
                median_R=float(row['median_R']),
            )
        return out
    except Exception as e:
        logging.warning("Drift forecaster predict failed (%s); using "
                        "static defaults.", e)
        return {}


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class XGBEnsembleBacktester:

    FEATURES_A = [
        'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
        'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
        'ts_advantage', 'is_maiden', 'jockey_alpha',
        'track_width', 'straight_length',
    ]
    FEATURES_B = FEATURES_A + [
        'draw_x_early_pace', 'straight_x_finish_pace',
        'class_drop_x_ts', 'class_rise_x_ts', 'days_since_last_race',
    ]

    def __init__(self,
                 db_url: str = DB_URL,
                 stacker_path: str = artifact('wf_stacker.pkl'),
                 shrinkage: float = 0.85,
                 starting_bankroll: float = 100_000.0,
                 theta_2: float = 0.8824,
                 theta_3: float = 0.7760,
                 drift_forecaster=None,
                 anchor_fallback: str = 'final_with_drift_adj'):
        self.engine = create_engine(db_url)
        self.factory = HKJCFeatureFactory(db_url)
        self.model_a = joblib.load(artifact('wf_model_a.pkl'))
        self.model_b = joblib.load(artifact('wf_model_b.pkl'))
        self.calibrator_a = joblib.load(artifact('wf_calib_a.pkl'))
        self.calibrator_b = joblib.load(artifact('wf_calib_b.pkl'))
        self.stacker = joblib.load(stacker_path)

        self.bankroll = starting_bankroll
        self.initial_bankroll = starting_bankroll
        self.shrinkage = shrinkage
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.drift_forecaster = drift_forecaster
        self.anchor_fallback = anchor_fallback

    def _attach_stop_sell(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'horse_no' not in df.columns:
            raise ValueError("Backtester requires horse_no in input data.")
        return attach_win_anchor(df, self.engine,
                                 odds_col='win_odds',
                                 out_col='stop_sell_odds',
                                 fallback=self.anchor_fallback)

    def _drift_features_for_race(self, race_id: str) -> pd.DataFrame:
        """Lazy-load drift features for ONE race (only when forecaster set)."""
        if self.drift_forecaster is None:
            return pd.DataFrame()
        try:
            from hkjc_engine.models.drift_forecaster import build_win_features
            return build_win_features(self.engine, [race_id])
        except Exception:
            return pd.DataFrame()

    def run_backtest(self,
                     start_date: str = '2024-01-01',
                     end_date: str = '2026-01-01'
                     ) -> tuple[list[dict], float]:
        logging.info("Drift-aware ensemble backtest...")

        query = text("""
            WITH CareerCounts AS (
                SELECT
                    e.race_id, e.horse_code,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.horse_code
                        ORDER BY r.race_date ASC, r.race_id ASC
                    ) AS career_run_number
                FROM race_entries e
                JOIN races r ON e.race_id = r.race_id
            )
            SELECT
                r.race_id, r.race_date, r.venue, r.distance, r.track_condition,
                r.rail_placement, r.race_class,
                e.horse_code, e.horse_no, e.draw, e.actual_weight,
                e.win_odds, e.finish_position,
                e.ema_early_z, e.ema_mid_z, e.ema_finish_z,
                e.pre_race_mu, e.pre_race_sigma, e.jockey, e.days_since_last_race,
                e.is_class_drop, e.is_class_rise,
                CASE WHEN cc.career_run_number = 1 THEN 1 ELSE 0 END AS is_maiden,
                d.combination AS win_combination, d.dividend AS win_dividend
            FROM races r
            JOIN race_entries e ON r.race_id = e.race_id
            JOIN CareerCounts cc ON e.race_id = cc.race_id
                                AND e.horse_code = cc.horse_code
            LEFT JOIN race_dividends d ON r.race_id = d.race_id AND d.pool = 'WIN'
            WHERE r.race_date >= :start_date AND r.race_date < :end_date
              AND e.win_odds IS NOT NULL
              AND e.finish_position IS NOT NULL
            ORDER BY r.race_date ASC, r.race_no ASC
        """)
        with self.engine.connect() as conn:
            raw = pd.read_sql(query, conn,
                              params={"start_date": start_date,
                                      "end_date": end_date})

        bet_ledger: list[dict] = []
        oof_export: list[pd.DataFrame] = []
        bets_placed = winning_bets = 0
        total_staked = 0.0

        for race_id, race_df in raw.groupby('race_id'):
            df = self.factory.engineer_features(race_df.copy())
            df = self._attach_stop_sell(df)
            if df.empty or len(df) < 2:
                continue

            # Data Quality Indicator for the Stacker
            df['I_valid'] = (df['stop_sell_odds'].notna() & (df['stop_sell_odds'] != df['win_odds'])).astype(int)

            # ---- defensive interaction recompute (parity with original) ----
            df['is_class_drop'] = df['is_class_drop'].astype(float).fillna(0.0)
            df['is_class_rise'] = df['is_class_rise'].astype(float).fillna(0.0)
            df['is_maiden']     = df['is_maiden'].astype(float)
            df['draw_x_early_pace']      = df['draw'] * df['relative_early_pace']
            df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
            df['class_drop_x_ts']        = df['is_class_drop'] * df['ts_advantage']
            df['class_rise_x_ts']        = df['is_class_rise'] * df['ts_advantage']

            # ---- Model A on STOP_SELL anchor ----
            dmat_a = xgb.DMatrix(df[self.FEATURES_A])
            df['base_margin'] = calculate_base_margin(df['stop_sell_odds'])
            dmat_a.set_base_margin(df['base_margin'])
            df['raw_a'] = self.model_a.predict(dmat_a)
            df['P_a_softmax'] = _softmax(df['raw_a'].values)
            # Model A utilizes global SmoothedIsotonicCalibrator
            df['P_a_cal'] = self.calibrator_a.predict_proba(df['P_a_softmax'].values)[:, 1]

            # ---- Model B (CoxPH) ----
            cox_features = [
                'relative_early_pace', 'relative_mid_pace', 'relative_finish_pace',
                'weight_delta', 'draw', 'is_class_drop', 'is_class_rise',
                'ts_advantage', 'is_maiden', 'track_width', 'straight_length'
            ]
            df['raw_b'] = self.model_b.predict_partial_hazard(df[cox_features])
            df['P_b_softmax'] = df['raw_b'] / df['raw_b'].sum()
            
            # Pace Archetype Bucketing for Model B Stratified Isotonic Calibrator
            bins = [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf]
            df['pace_archetype'] = pd.cut(df['relative_early_pace'], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
            df['P_b_cal'] = self.calibrator_b.predict_proba(df['P_b_softmax'].values, strata=df['pace_archetype'].values)[:, 1]

            # ---- Public market on STOP_SELL ----
            df['P_pub_raw'] = 1.0 / df['stop_sell_odds']
            df['P_pub'] = df['P_pub_raw'] / df['P_pub_raw'].sum()

            # ---- Stacker (with Data Quality Context for FLB debiasing) ----
            P = np.column_stack([df['P_a_cal'].values,
                                 df['P_b_cal'].values,
                                 df['P_pub'].values])
            df['P_model'] = self.stacker.predict(P, df['race_id'].values, I_valid=df['I_valid'].values)
            df['EV_naive'] = df['P_model'] * df['stop_sell_odds'] - 1.0

            # ---- Drift stats: per-horse from forecaster, fallback to static ----
            drift_overrides_per_idx = {}
            if self.drift_forecaster is not None:
                feats = self._drift_features_for_race(race_id)
                if not feats.empty:
                    # Align forecaster output to df row order via horse_no
                    preds = self.drift_forecaster.predict_win(feats)
                    if not preds.empty:
                        preds = preds.set_index('horse_no')
                        # Build per-row override dict: row index -> DriftStats
                        for ridx, row in df.reset_index(drop=True).iterrows():
                            hn = str(row['horse_no'])
                            if hn in preds.index:
                                p = preds.loc[hn]
                                drift_overrides_per_idx[ridx] = DriftStats(
                                    mu_R=float(p['mu_R']),
                                    sigma_R=float(p['sigma_R']),
                                    median_R=float(p['median_R']),
                                )

            # ---- Sizing: per-row mu_R/median_R when forecaster is on ----
            # We use qualify_and_size with the WIN pool default if no
            # forecaster, or a single-row build with the override otherwise.
            keep_idx, stakes = self._size_win_pool(df, drift_overrides_per_idx)

            oof_export.append(df[['race_id', 'horse_code',
                                  'finish_position', 'P_model']].copy())

            for local_i, stake in zip(keep_idx, stakes):
                target = df.iloc[local_i]
                if stake < 10:
                    continue
                bets_placed += 1
                total_staked += stake
                self.bankroll -= stake

                is_win = (str(target['horse_no']).replace('.0', '')
                          == str(target['win_combination']).replace('.0', ''))
                if is_win:
                    payout = (stake / 10.0) * float(target['win_dividend'])
                    profit = payout - stake
                    self.bankroll += payout
                    winning_bets += 1
                else:
                    profit = -stake

                # Diagnostics: report both EV-at-STOP_SELL (gating EV) and
                # EV-at-FINAL (settlement EV). The gap between them is the
                # closing-line drift cost per bet.
                p_adj = min(self.shrinkage * target['P_model'], 1 - 1e-9)
                ev_at_stop  = p_adj * target['stop_sell_odds'] - 1.0
                ev_at_final = p_adj * target['win_odds']      - 1.0

                bet_ledger.append({
                    'race_id':       race_id,
                    'horse_code':    target['horse_code'],
                    'horse_no':      target['horse_no'],
                    'win_odds':      target['win_odds'],
                    'stop_sell_odds': target['stop_sell_odds'],
                    'P_pub':         target['P_pub'],
                    'P_model':       target['P_model'],
                    'P_shrunk':      p_adj,
                    'EV_at_stop':    ev_at_stop,
                    'EV_at_final':   ev_at_final,
                    'EV_naive':      target['EV_naive'],
                    'stake':         stake,
                    'profit':        profit,
                })

        # ---- summary ----
        monthly_profit = self.bankroll - self.initial_bankroll
        roi = (monthly_profit / total_staked) * 100 if total_staked > 0 else 0
        win_rate = (winning_bets / bets_placed) * 100 if bets_placed > 0 else 0

        logging.info("\n" + "=" * 50)
        logging.info("   DRIFT-AWARE BACKTEST RESULTS")
        logging.info("=" * 50)
        logging.info("Starting Bankroll : $%,.2f", self.initial_bankroll)
        logging.info("Ending Bankroll   : $%,.2f", self.bankroll)
        logging.info("Net Profit        : $%,.2f", monthly_profit)
        logging.info("Total Bets        : %d", bets_placed)
        logging.info("Total Staked      : $%,.2f", total_staked)
        logging.info("Win Rate          : %.2f%%", win_rate)
        logging.info("ROI               : %.2f%%", roi)
        logging.info("=" * 50)

        if oof_export:
            export_df = pd.concat(oof_export, ignore_index=True)
            with open('ensemble_oof_results.csv', 'a', newline='') as f:
                export_df.to_csv(f, header=f.tell() == 0, index=False)
        return bet_ledger, self.bankroll

    # ------------------------------------------------------------------
    # Internal: per-pool sizing with drift overrides
    # ------------------------------------------------------------------
    def _size_win_pool(self, df: pd.DataFrame,
                       overrides: dict[int, DriftStats]
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Apply drift-aware sizing on the WIN pool with optional per-row
        DriftStats overrides from the forecaster.

        If no overrides, uses the static WIN-pool defaults via
        `qualify_and_size`. Otherwise computes per-row EV/Kelly with the
        forecaster's predictions and runs the race-level cap manually.
        """
        if not overrides:
            return qualify_and_size(
                p_raw=df['P_model'].values,
                odds=df['stop_sell_odds'].values,
                bankroll=self.bankroll,
                kelly_fraction=0.35,
                shrinkage=self.shrinkage,
                base_hurdle=0.005,
                longshot_buffer=0.005,
                longshot_threshold=25.0,
                per_bet_cap=0.05,
                race_cap=0.10,
                top_k=10,
                pool='WIN',
            )

        # Per-row sizing with custom drift stats
        from hkjc_engine.models.betting_policy import (
            fractional_kelly_stake, get_ev_hurdle, race_cap_for_pool,
        )

        n = len(df)
        stakes = np.zeros(n)
        evs = np.zeros(n)
        for i in range(n):
            ds = overrides.get(i, DEFAULT_DRIFT_STATS['WIN'])
            override_dict = {'WIN': ds}
            odds = float(df['stop_sell_odds'].iloc[i])
            p = float(df['P_model'].iloc[i])

            ev = (min(self.shrinkage * p, 1 - 1e-9)
                  * odds * ds.median_R - 1.0)
            evs[i] = ev
            hurdle = get_ev_hurdle(odds, base=0.005, longshot_buffer=0.005,
                                   longshot_threshold=25.0,
                                   pool='WIN', drift_override=override_dict)
            if ev < hurdle or odds > 25.0:
                continue
            stakes[i] = fractional_kelly_stake(
                p_raw=p, odds=odds, bankroll=self.bankroll,
                kelly_fraction=0.35,
                shrinkage=self.shrinkage,
                per_bet_cap=0.05, min_stake_abs=10.0,
                pool='WIN', drift_override=override_dict,
            )

        # Race-level cap with WIN-pool defaults (overrides per-horse, but
        # cap is race-level so we keep global WIN sigma).
        cap_used = race_cap_for_pool(0.10, 'WIN')
        cap_dollars = cap_used * self.bankroll
        total = stakes.sum()
        if total > cap_dollars > 0:
            stakes *= cap_dollars / total

        keep = np.where(stakes >= 10.0)[0]
        return keep, stakes[keep]


if __name__ == "__main__":
    XGBEnsembleBacktester().run_backtest()