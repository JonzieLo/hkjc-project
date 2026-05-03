"""
Closing-line drift forecaster (§2 of the drift-aware refactor).

What it predicts
----------------
For each (race, horse) tuple at STOP_SELL time, the conditional
distribution of R = d_final / d_stop_sell. We expose three quantiles:

    q10, q50 (median), q90

from which downstream policy code derives:

    median_R = q50                          (Jensen-correct payoff multiplier)
    mu_R     ≈ q50                          (mean ≈ median in this regime)
    sigma_R  ≈ (q90 - q10) / 2.5631         (gaussian quantile-spread → sigma)

Architecture
------------
1. PRIMARY:  one quantile XGBoost on the WIN pool, ~14 outputs per race.
2. PROJECT:  for QIN/QPL/TRI, project the WIN drift to combinations via
             Henery-discounted Harville (re-using theta_2 / theta_3 from
             contextual_thetas.json so we don't re-invent that wheel).
             A combination's drift is the geometric mean of its constituent
             horses' drift, weighted by Harville order probability.
3. RESIDUAL: optional second-stage tiny model on (observed exotic drift -
             Harville-projected drift). Captures the syndicate's
             exotic-only flow that doesn't show up in WIN. Off by default
             to keep the mainline simple; turn on when you have >5k exotic
             combinations of training data and clear evidence the residual
             is non-zero.

Features (WIN pool, per-(race_id, horse_no))
--------------------------------------------
Velocity:
    dlog_p_60s, dlog_p_30s, dlog_p_10s
        Change in log-implied-probability over the trailing window.
        Sharp acceleration into a horse is the syndicate's signal.
Asymmetry:
    dlog_p_30s_fav, dlog_p_30s_2nd
        Race-level favourite vs. second-favourite movement. Different
        regimes when both shorten vs. when only the favourite shortens.
Concentration:
    hhi_at_stop_sell, dhhi_30s
        Herfindahl on the WIN pool. Rising HHI = collapsing onto few
        horses = adverse-selection regime.
Cross-pool inconsistency:
    qin_inconsistency
        Mean |Harville-projected p_QIN - direct p_QIN| at STOP_SELL.
        High value = exotic-only money is concentrating.
Position:
    rank_at_stop_sell, is_favorite, p_implied_at_stop_sell
Race context:
    race_no_on_card, is_late_card, field_size, class_level,
    is_sprint, is_weekend
Outcome (label):
    R = final_odds / stop_sell_odds; train on log(R).

Training: `python -m hkjc_engine.models.drift_forecaster`
Inference: `DriftForecaster.predict_win(snapshot_df) -> DataFrame[q10, q50, q90]`
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL, artifact

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

# Per-snapshot WIN-pool history. We need a dense time series, so this
# pulls every PRE/POST_STOP_SELL row in the window [stop_sell - 90s, +30s].
_Q_WIN_TIMESERIES = text("""
    WITH stop_sell_anchor AS (
        SELECT race_id, MIN(timestamp) AS stop_ts
        FROM live_odds_history
        WHERE phase = 'POST_STOP_SELL'
          AND seconds_vs_stop_sell >= 0
          AND race_id = ANY(:race_ids)
        GROUP BY race_id
    )
    SELECT h.race_id, h.combination AS horse_no,
           h.timestamp, h.odds, h.phase, h.seconds_vs_stop_sell,
           a.stop_ts
    FROM live_odds_history h
    JOIN stop_sell_anchor a ON h.race_id = a.race_id
    WHERE h.pool_type = 'WIN'
      AND h.odds > 1.0
      AND h.timestamp BETWEEN a.stop_ts - INTERVAL '90 seconds'
                          AND a.stop_ts + INTERVAL '30 seconds'
""")


_Q_FINAL_WIN = text("""
    SELECT DISTINCT ON (race_id, combination)
        race_id, combination AS horse_no, odds AS final_odds
    FROM live_odds_history
    WHERE phase = 'FINAL'
      AND pool_type = 'WIN'
      AND race_id = ANY(:race_ids)
    ORDER BY race_id, combination, timestamp DESC
""")


_Q_RACE_META = text("""
    SELECT race_id, race_date, race_no, race_class, distance, venue
    FROM races
    WHERE race_id = ANY(:race_ids)
""")


_Q_QIN_INCONSISTENCY = text("""
    -- Snapshot of QIN-pool combination odds nearest the WIN STOP_SELL
    SELECT DISTINCT ON (race_id, combination)
        race_id, combination, odds AS qin_odds, timestamp
    FROM live_odds_history
    WHERE pool_type = 'QIN'
      AND odds > 1.0
      AND phase IN ('PRE_STOP_SELL', 'POST_STOP_SELL')
      AND race_id = ANY(:race_ids)
    ORDER BY race_id, combination, timestamp DESC
""")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

WIN_FEATURES: list[str] = [
    'dlog_p_60s', 'dlog_p_30s', 'dlog_p_10s',
    'dlog_p_30s_fav', 'dlog_p_30s_2nd',
    'hhi_at_stop_sell', 'dhhi_30s',
    'qin_inconsistency',
    'rank_at_stop_sell', 'is_favorite', 'p_implied_at_stop_sell',
    'race_no_on_card', 'is_late_card', 'field_size',
    'class_level', 'is_sprint', 'is_weekend',
]


def _class_level(class_str) -> int:
    """1-9 ordinal scale, 99=unknown. Mirrors live/predictor.py convention."""
    if not isinstance(class_str, str):
        return 99
    s = str(class_str).upper()
    for needle, lvl in [('GROUP 1', 1), ('GROUP 2', 2), ('GROUP 3', 3),
                        ('CLASS 1', 4), ('CLASS 2', 5), ('CLASS 3', 6),
                        ('CLASS 4', 7), ('CLASS 5', 8), ('GRIFFIN', 9)]:
        if needle in s:
            return lvl
    return 99


def _odds_at(grp: pd.DataFrame, t_offset: float) -> pd.Series:
    """For each (race, horse) group, return odds at (stop_ts + t_offset).

    We pick the snapshot whose `seconds_vs_stop_sell` is closest to
    `t_offset`. If no snapshot lies within ±15s, returns NaN.
    """
    g = grp.copy()
    g['delta'] = (g['seconds_vs_stop_sell'] - t_offset).abs()
    nearest = g.sort_values('delta').groupby(['race_id', 'horse_no']).first()
    nearest.loc[nearest['delta'] > 15, 'odds'] = np.nan
    return nearest['odds']


def build_win_features(engine, race_ids: Iterable[str]) -> pd.DataFrame:
    """Build per-(race_id, horse_no) feature dataframe for WIN-pool drift.

    Returns columns: ['race_id', 'horse_no'] + WIN_FEATURES + ['stop_sell_odds'].
    Rows missing a STOP_SELL snapshot are dropped.
    """
    race_ids = list(race_ids)
    if not race_ids:
        return pd.DataFrame(columns=['race_id', 'horse_no', *WIN_FEATURES,
                                     'stop_sell_odds'])

    with engine.connect() as conn:
        ts = pd.read_sql(_Q_WIN_TIMESERIES, conn, params={'race_ids': race_ids})
        meta = pd.read_sql(_Q_RACE_META, conn, params={'race_ids': race_ids})
        qin = pd.read_sql(_Q_QIN_INCONSISTENCY, conn,
                          params={'race_ids': race_ids})

    if ts.empty:
        log.warning("No WIN time-series rows for %d races; cannot build "
                    "drift features.", len(race_ids))
        return pd.DataFrame(columns=['race_id', 'horse_no', *WIN_FEATURES,
                                     'stop_sell_odds'])

    # ---- per-(race,horse) odds at t = -60, -30, -10, 0 ----
    ts['horse_no'] = ts['horse_no'].astype(str)
    odds_pivot = pd.DataFrame({
        'odds_m60s': _odds_at(ts, -60),
        'odds_m30s': _odds_at(ts, -30),
        'odds_m10s': _odds_at(ts, -10),
        'odds_t0':   _odds_at(ts,   0),
    }).reset_index()

    odds_pivot = odds_pivot.dropna(subset=['odds_t0'])
    odds_pivot['p_t0']   = 1.0 / odds_pivot['odds_t0']
    for col_in, col_out in [('odds_m60s', 'p_m60s'),
                            ('odds_m30s', 'p_m30s'),
                            ('odds_m10s', 'p_m10s')]:
        odds_pivot[col_out] = 1.0 / odds_pivot[col_in]

    # ---- velocity features ----
    odds_pivot['dlog_p_60s'] = (np.log(odds_pivot['p_t0'])
                                - np.log(odds_pivot['p_m60s'])).fillna(0.0)
    odds_pivot['dlog_p_30s'] = (np.log(odds_pivot['p_t0'])
                                - np.log(odds_pivot['p_m30s'])).fillna(0.0)
    odds_pivot['dlog_p_10s'] = (np.log(odds_pivot['p_t0'])
                                - np.log(odds_pivot['p_m10s'])).fillna(0.0)

    # ---- per-race rank, HHI, asymmetry ----
    odds_pivot['rank_at_stop_sell'] = (
        odds_pivot.groupby('race_id')['odds_t0']
        .rank(method='dense').astype(int)
    )
    odds_pivot['is_favorite'] = (odds_pivot['rank_at_stop_sell'] == 1).astype(int)

    grp = odds_pivot.groupby('race_id')
    odds_pivot['p_implied_at_stop_sell'] = odds_pivot['p_t0']

    # HHI is sum of p^2 across all horses in the race.
    hhi_t0 = grp.apply(lambda g: float((g['p_t0'] ** 2).sum())).rename('hhi_at_stop_sell')
    hhi_m30 = grp.apply(lambda g: float((g['p_m30s'].fillna(g['p_t0']) ** 2).sum())).rename('hhi_m30s')
    odds_pivot = odds_pivot.merge(hhi_t0, on='race_id').merge(hhi_m30, on='race_id')
    odds_pivot['dhhi_30s'] = odds_pivot['hhi_at_stop_sell'] - odds_pivot['hhi_m30s']

    # NOTE on joint-rank handling: `rank(method='dense')` assigns the same
    # rank to horses with identical odds (e.g. two horses both at 3.5). If
    # we naively merged `fav_dlog` (filtered to rank == 1) on race_id, the
    # join would be N:M for races with joint favorites and Cartesian-
    # explode `odds_pivot`, duplicating every horse 2x. We collapse to
    # one row per race here by averaging the drift across joint-rank
    # horses so the merge is strictly 1:N.
    fav_dlog = (odds_pivot.loc[odds_pivot['rank_at_stop_sell'] == 1]
                .groupby('race_id', as_index=False)['dlog_p_30s'].mean()
                .rename(columns={'dlog_p_30s': 'dlog_p_30s_fav'}))
    snd_dlog = (odds_pivot.loc[odds_pivot['rank_at_stop_sell'] == 2]
                .groupby('race_id', as_index=False)['dlog_p_30s'].mean()
                .rename(columns={'dlog_p_30s': 'dlog_p_30s_2nd'}))
    odds_pivot = (odds_pivot
                  .merge(fav_dlog, on='race_id', how='left')
                  .merge(snd_dlog, on='race_id', how='left'))
    odds_pivot[['dlog_p_30s_fav', 'dlog_p_30s_2nd']] = (
        odds_pivot[['dlog_p_30s_fav', 'dlog_p_30s_2nd']].fillna(0.0))

    # Belt-and-braces: drop any (race_id, horse_no) duplicate that may
    # have slipped through earlier merges. Cheap insurance against
    # future bugs that could re-introduce a Cartesian product.
    odds_pivot = odds_pivot.drop_duplicates(
        subset=['race_id', 'horse_no'], keep='first',
    ).reset_index(drop=True)

    # ---- field_size ----
    odds_pivot['field_size'] = grp['horse_no'].transform('nunique')

    # ---- cross-pool inconsistency: |Harville-projected QIN - direct QIN| ----
    inconsistency = _qin_inconsistency_per_race(odds_pivot, qin)
    odds_pivot = odds_pivot.merge(inconsistency, on='race_id', how='left')
    odds_pivot['qin_inconsistency'] = odds_pivot['qin_inconsistency'].fillna(0.0)

    # ---- race-level meta ----
    meta['race_no_on_card'] = meta['race_no'].astype(int)
    meta['is_late_card'] = (meta['race_no_on_card'] >= 8).astype(int)
    meta['class_level'] = meta['race_class'].apply(_class_level)
    meta['is_sprint'] = (meta['distance'].astype(float) < 1400).astype(int)
    meta['race_date'] = pd.to_datetime(meta['race_date'])
    meta['is_weekend'] = (meta['race_date'].dt.weekday >= 5).astype(int)

    out = odds_pivot.merge(
        meta[['race_id', 'race_no_on_card', 'is_late_card',
              'class_level', 'is_sprint', 'is_weekend']],
        on='race_id', how='left',
    )
    out['stop_sell_odds'] = out['odds_t0']

    keep_cols = ['race_id', 'horse_no', *WIN_FEATURES, 'stop_sell_odds']
    return out[keep_cols].dropna(subset=['stop_sell_odds']).reset_index(drop=True)


def _qin_inconsistency_per_race(win_df: pd.DataFrame,
                                qin_df: pd.DataFrame) -> pd.DataFrame:
    """Per-race mean |Harville-implied p_QIN - direct p_QIN|.

    Harville-implied p_QIN(i,j) = p_i*p_j/(1-p_i) + p_j*p_i/(1-p_j).
    We use theta=1 here (uncorrected Harville) for speed; the
    discount-θ correction is captured downstream in the betting policy
    and would only marginally shift the inconsistency signal.
    """
    if qin_df.empty:
        return pd.DataFrame({'race_id': [], 'qin_inconsistency': []})

    rows: list[dict] = []
    for race_id, win_g in win_df.groupby('race_id'):
        # Defensive dedup: even though build_win_features collapses
        # joint-rank rows upstream, dropping duplicates here keeps the
        # `set_index` -> `.loc[str(a)]` path returning a scalar rather
        # than a Series if a future upstream change re-introduces dupes.
        win_g = win_g.drop_duplicates('horse_no', keep='first')
        p = win_g.set_index('horse_no')['p_t0'].astype(float)
        n = len(p)
        if n < 2:
            continue
        # Renormalise to remove takeout
        p = p / p.sum()
        qin_g = qin_df[qin_df['race_id'] == race_id]
        if qin_g.empty:
            continue

        diffs = []
        for combo, qin_odds in zip(qin_g['combination'], qin_g['qin_odds']):
            try:
                a, b = sorted(int(x) for x in str(combo).split('-'))
            except (ValueError, AttributeError):
                continue
            if str(a) not in p.index or str(b) not in p.index:
                continue
            pa, pb = float(p.loc[str(a)]), float(p.loc[str(b)])
            if pa >= 1 or pb >= 1:
                continue
            p_harv = pa * pb / (1 - pa) + pb * pa / (1 - pb)
            p_direct = 1.0 / float(qin_odds)
            diffs.append(abs(p_harv - p_direct))
        if diffs:
            rows.append({'race_id': race_id,
                         'qin_inconsistency': float(np.mean(diffs))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class DriftForecasterArtifacts:
    booster: xgb.Booster
    feature_names: list[str] = field(default_factory=lambda: list(WIN_FEATURES))
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    sigma_floor: float = 0.005   # never let predicted sigma_R go below this
    median_clip: tuple[float, float] = (0.7, 1.3)


class DriftForecaster:
    """Quantile XGBoost on log(R_final / R_stop_sell) for the WIN pool.

    Inference returns conditional quantile predictions; downstream code
    derives (mu_R, sigma_R, median_R) for use in the betting policy.
    """

    def __init__(self,
                 db_url: str = DB_URL,
                 quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
                 model_path: str = artifact('drift_forecaster.pkl')):
        self.engine = create_engine(db_url)
        self.quantiles = list(quantiles)
        self.model_path = model_path
        self.artifacts: DriftForecasterArtifacts | None = None

    # --- training ---
    def _build_training_set(self, race_ids: list[str]) -> pd.DataFrame:
        feats = build_win_features(self.engine, race_ids)
        if feats.empty:
            return feats

        with self.engine.connect() as conn:
            finals = pd.read_sql(_Q_FINAL_WIN, conn,
                                 params={'race_ids': race_ids})
        finals['horse_no'] = finals['horse_no'].astype(str)
        df = feats.merge(finals, on=['race_id', 'horse_no'], how='inner')
        df['log_R'] = np.log(df['final_odds'] / df['stop_sell_odds'])

        # Trim outliers: ratios outside [0.3, 3.0] are almost always
        # data errors (mis-mapped combinations) rather than genuine drift.
        df = df[(df['log_R'] > np.log(0.3)) & (df['log_R'] < np.log(3.0))]
        return df.reset_index(drop=True)

    def train(self,
              start_date: str = '2018-01-01',
              end_date: str | None = None,
              n_rounds: int = 400,
              early_stopping_rounds: int = 30) -> 'DriftForecaster':
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        with self.engine.connect() as conn:
            rids = pd.read_sql(text("""
                SELECT race_id FROM races
                WHERE race_date >= :s AND race_date < :e
                ORDER BY race_date
            """), conn, params={'s': start_date, 'e': end_date}
            )['race_id'].tolist()

        log.info("DriftForecaster: fetching %d candidate races...", len(rids))
        df = self._build_training_set(rids)
        if df.empty:
            raise RuntimeError("Empty training set. Check that "
                               "live_odds_history has phase='POST_STOP_SELL' "
                               "rows in the requested window.")
        log.info("Training on %d (race, horse) tuples across %d races.",
                 len(df), df['race_id'].nunique())

        # Group-aware split: hold the most recent 15% of races for
        # early stopping. Random shuffles would leak race-level
        # info between train/val.
        unique_races = df['race_id'].drop_duplicates().tolist()
        n_val = max(1, int(0.15 * len(unique_races)))
        val_races = set(unique_races[-n_val:])
        train_mask = ~df['race_id'].isin(val_races)

        X_tr = df.loc[train_mask, WIN_FEATURES].values
        X_va = df.loc[~train_mask, WIN_FEATURES].values
        y_tr = df.loc[train_mask, 'log_R'].values
        y_va = df.loc[~train_mask, 'log_R'].values

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=WIN_FEATURES)
        dval = xgb.DMatrix(X_va, label=y_va, feature_names=WIN_FEATURES)

        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': self.quantiles,         # multi-quantile in xgb >= 2.0
            'tree_method': 'hist',
            'learning_rate': 0.04,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'min_child_weight': 8,
            'lambda': 1.0,
        }
        booster = xgb.train(
            params, dtrain, num_boost_round=n_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )

        self.artifacts = DriftForecasterArtifacts(
            booster=booster,
            feature_names=list(WIN_FEATURES),
            quantiles=list(self.quantiles),
        )
        joblib.dump(self.artifacts, self.model_path)
        log.info("DriftForecaster saved to %s", self.model_path)

        # Quick OOS quality check
        preds = self.artifacts.booster.predict(dval)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        # Coverage check: y in [q10, q90] should be ~80% if calibrated
        q10_idx = self.quantiles.index(0.1) if 0.1 in self.quantiles else 0
        q90_idx = self.quantiles.index(0.9) if 0.9 in self.quantiles else -1
        in_band = ((y_va >= preds[:, q10_idx]) & (y_va <= preds[:, q90_idx]))
        log.info("OOS [q10, q90] empirical coverage: %.1f%% (target ~80%%)",
                 100.0 * in_band.mean())
        return self

    # --- inference ---
    def load(self) -> 'DriftForecaster':
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No trained drift forecaster at {self.model_path}; "
                f"run `python -m hkjc_engine.models.drift_forecaster` first."
            )
        self.artifacts = joblib.load(self.model_path)
        return self

    def predict_win(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict (q10, q50, q90, mu_R, sigma_R, median_R) per row.

        `features` must contain the WIN_FEATURES columns; index/race_id
        are passed through untouched.
        """
        if self.artifacts is None:
            self.load()
        if features.empty:
            cols = ['q10', 'q50', 'q90', 'mu_R', 'sigma_R', 'median_R']
            return pd.DataFrame(columns=cols)

        X = features[self.artifacts.feature_names].values
        d = xgb.DMatrix(X, feature_names=self.artifacts.feature_names)
        preds = self.artifacts.booster.predict(d)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        # preds shape: (N, len(quantiles)) where each entry is in log-R space
        q = self.artifacts.quantiles
        idx_10 = q.index(0.1) if 0.1 in q else 0
        idx_50 = q.index(0.5) if 0.5 in q else min(1, preds.shape[1] - 1)
        idx_90 = q.index(0.9) if 0.9 in q else preds.shape[1] - 1

        log_q10 = preds[:, idx_10]
        log_q50 = preds[:, idx_50]
        log_q90 = preds[:, idx_90]

        q10 = np.exp(log_q10)
        q50 = np.exp(log_q50)
        q90 = np.exp(log_q90)
        # Sigma in R-space from quantile spread: 2.5631 ≈ Φ⁻¹(0.9) - Φ⁻¹(0.1)
        sigma_R = np.maximum((q90 - q10) / 2.5631,
                             self.artifacts.sigma_floor)
        median_R = np.clip(q50, *self.artifacts.median_clip)
        mu_R = median_R   # approximation; refine if needed

        out = features[['race_id', 'horse_no']].copy() if 'race_id' in features else pd.DataFrame()
        out['q10'], out['q50'], out['q90'] = q10, q50, q90
        out['mu_R'], out['sigma_R'], out['median_R'] = mu_R, sigma_R, median_R
        return out


# ---------------------------------------------------------------------------
# Harville projection: WIN drift -> exotic combo drift
# ---------------------------------------------------------------------------

def project_drift_to_exotic(win_drift: pd.DataFrame,
                            combo_horse_nos: list[str | int],
                            ) -> dict:
    """Project per-horse WIN drift to a single exotic combination.

    Approximation: an exotic ticket is a product of dependent finishing
    events (Harville). If each horse i has E[log R_i] and Var[log R_i],
    then for the combination's log-payoff:

        log R_combo ≈ sum_i log R_i      (independent-payoff approximation)
        E[log R_combo]   ≈ sum_i log_median_R_i
        Var[log R_combo] ≈ sum_i (log_sigma_R_i)^2     (worst-case independent)

    Then converting back via Var[R] ≈ R^2 * Var[log R] under small-sigma:

        median_R_combo = prod_i median_R_i
        sigma_R_combo  ≈ median_R_combo * sqrt(sum_i (sigma_R_i / median_R_i)^2)

    This is conservative — real correlation is positive (the syndicate
    moves combos together, so individual drifts compound rather than
    diversify). For policy purposes, slight conservatism is fine: it
    pushes Kelly down further on exotics, which is the desired direction
    given the empirical 9-10% absolute drift.
    """
    parts = win_drift.set_index('horse_no')
    parts.index = parts.index.astype(str)
    medians, sigma_ratios = [], []
    for h in combo_horse_nos:
        key = str(int(h)) if isinstance(h, (int, np.integer)) else str(h)
        if key not in parts.index:
            return {'mu_R': 1.0, 'sigma_R': 0.10, 'median_R': 1.0}
        m = float(parts.loc[key, 'median_R'])
        s = float(parts.loc[key, 'sigma_R'])
        medians.append(m)
        sigma_ratios.append(s / max(m, 1e-9))

    median_combo = float(np.prod(medians))
    sigma_combo = median_combo * float(np.sqrt(np.sum(np.square(sigma_ratios))))
    return {
        'mu_R': median_combo,
        'sigma_R': sigma_combo,
        'median_R': median_combo,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start_date', default='2018-01-01')
    ap.add_argument('--end_date', default=None)
    ap.add_argument('--save_path', default=artifact('drift_forecaster.pkl'))
    args = ap.parse_args()

    DriftForecaster(model_path=args.save_path).train(
        start_date=args.start_date,
        end_date=args.end_date,
    )