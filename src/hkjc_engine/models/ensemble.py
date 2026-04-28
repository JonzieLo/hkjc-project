"""
Beta calibrator + Benter log-linear stacker.

Drift-aware refactor (§1b)
--------------------------
The optimiser for the third (market) column previously used
`P_mkt = 1 / win_odds`, normalised. That FINAL-odds anchor included
late-money information the live engine cannot see at execution. The
stacker would fit `P_mkt` an inflated weight that does not generalise to
live conditions.

Now `EnsembleOptimizer` reads `stop_sell_odds` from the OOF CSV (exported
by trainer_residual.py and trainer_independent.py) and uses
`1 / stop_sell_odds` as the public-consensus probability. If the column
is absent — e.g. running this on legacy OOF outputs — we fall back to
`win_odds` with a warning so the user is aware the fit is biased.

Other changes
-------------
* `BetaCalibrator` and `BenterLogLinearStacker` are unchanged.
* The grouped-softmax helper (`GroupedSoftmaxObjective`) is unchanged.
* The optimiser reports BOTH the FINAL-anchored and STOP_SELL-anchored
  baseline log-losses so the user can see how much "edge" was being
  attributed to look-ahead bias.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# BetaCalibrator (unchanged)
# ---------------------------------------------------------------------------

class BetaCalibrator:
    def __init__(self, enforce_monotone=True, C=1e10):
        self.enforce_monotone = enforce_monotone
        self.C = C
        self.lr_ = None
        self.mode_ = 'full'

    def _clip(self, s):
        return np.clip(np.asarray(s, dtype=float).ravel(), 1e-12, 1 - 1e-12)

    def _features_full(self, s):
        s = self._clip(s)
        return np.column_stack([np.log(s), -np.log(1.0 - s)])

    def _features_a_only(self, s):
        s = self._clip(s)
        return np.log(s).reshape(-1, 1)

    def _features_b_only(self, s):
        s = self._clip(s)
        return -np.log(1.0 - s).reshape(-1, 1)

    def fit(self, scores, y):
        X = self._features_full(scores)
        self.lr_ = LogisticRegression(C=self.C, solver='lbfgs', fit_intercept=True)
        self.lr_.fit(X, np.asarray(y).ravel())
        if self.enforce_monotone:
            a, b = self.lr_.coef_[0]
            if a < 0 and b < 0:
                self.lr_ = LogisticRegression(C=self.C, solver='lbfgs', fit_intercept=True)
                self.lr_.fit(np.zeros((len(scores), 1)), y)
                self.mode_ = 'intercept_only'
            elif a < 0:
                self.lr_ = LogisticRegression(C=self.C, solver='lbfgs', fit_intercept=True)
                self.lr_.fit(self._features_b_only(scores), y)
                self.mode_ = 'b_only'
            elif b < 0:
                self.lr_ = LogisticRegression(C=self.C, solver='lbfgs', fit_intercept=True)
                self.lr_.fit(self._features_a_only(scores), y)
                self.mode_ = 'a_only'
        return self

    def _design(self, scores):
        if self.mode_ == 'full':   return self._features_full(scores)
        if self.mode_ == 'a_only': return self._features_a_only(scores)
        if self.mode_ == 'b_only': return self._features_b_only(scores)
        return np.zeros((len(self._clip(scores)), 1))

    def predict(self, scores):
        return self.lr_.predict_proba(self._design(scores))[:, 1]

    def predict_proba(self, scores):
        p = self.predict(scores)
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# Grouped softmax loss (unchanged)
# ---------------------------------------------------------------------------

class GroupedSoftmaxObjective:
    def __init__(self, group_sizes):
        self.group_sizes = np.asarray(group_sizes, dtype=np.int64)
        self.boundaries  = np.concatenate(([0], np.cumsum(self.group_sizes)))

    def __call__(self, predt, dtrain):
        y = dtrain.get_label()
        grad = np.empty_like(predt, dtype=np.float64)
        hess = np.empty_like(predt, dtype=np.float64)
        for k in range(len(self.group_sizes)):
            lo, hi = self.boundaries[k], self.boundaries[k + 1]
            s = predt[lo:hi].astype(np.float64); s -= s.max()
            e = np.exp(s); p = e / e.sum()
            grad[lo:hi] = p - y[lo:hi]
            hess[lo:hi] = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess


def grouped_logloss_eval(group_sizes):
    boundaries = np.concatenate(([0], np.cumsum(group_sizes)))
    def _feval(predt, dtrain):
        y = dtrain.get_label()
        total, n = 0.0, len(group_sizes)
        for k in range(n):
            lo, hi = boundaries[k], boundaries[k + 1]
            s = predt[lo:hi] - predt[lo:hi].max()
            p = np.exp(s); p /= p.sum()
            w = int(np.argmax(y[lo:hi]))
            total += -np.log(max(p[w], 1e-15))
        return 'race_logloss', total / max(n, 1)
    return _feval


# ---------------------------------------------------------------------------
# Benter log-linear stacker (unchanged interface)
# ---------------------------------------------------------------------------

class BenterLogLinearStacker:
    def __init__(self, bounds=(0.0, 3.0), model_names=None):
        self.bounds = bounds
        self.model_names = model_names
        self.weights = None
        self.loss_ = None

    def _normalize_per_race(self, logp, race_ids):
        s = pd.Series(logp, index=race_ids)
        s = s - s.groupby(level=0).transform('max')
        e = np.exp(s.values)
        df = pd.DataFrame({'e': e, 'rid': race_ids})
        df['z'] = df.groupby('rid')['e'].transform('sum')
        return (df['e'] / df['z']).values

    def _prepare(self, P):
        return np.clip(np.asarray(P, dtype=float), 1e-12, 1 - 1e-12)

    def fit(self, P, race_ids, y, x0=None, verbose=False):
        P = self._prepare(P)
        logP = np.log(P)
        race_ids = np.asarray(race_ids)
        K = P.shape[1]
        self.baseline_losses_ = [log_loss(y, P[:, k]) for k in range(K)]

        def nll(w):
            ens_log = logP @ w
            p_ens = self._normalize_per_race(ens_log, race_ids)
            p_ens = np.clip(p_ens, 1e-15, 1 - 1e-15)
            return log_loss(y, p_ens)

        x0 = np.ones(K) if x0 is None else np.asarray(x0, dtype=float)
        res = minimize(nll, x0, method='L-BFGS-B',
                       bounds=[self.bounds] * K,
                       options={'maxiter': 500, 'ftol': 1e-10})
        self.weights = res.x
        self.loss_ = float(res.fun)

        if verbose:
            names = self.model_names or [f"M{k}" for k in range(K)]
            logging.info("\n--- Benter Log-Linear Stacker ---")
            for name, base, w in zip(names, self.baseline_losses_, self.weights):
                logging.info("%-10s | standalone LL = %.5f | weight = %+.4f",
                             name, base, w)
            logging.info("Ensemble LogLoss       = %.5f", self.loss_)
        return self

    def predict(self, P, race_ids):
        P = self._prepare(P)
        return self._normalize_per_race(np.log(P) @ self.weights,
                                        np.asarray(race_ids))


# ---------------------------------------------------------------------------
# EnsembleOptimizer — STOP_SELL-aware
# ---------------------------------------------------------------------------

class EnsembleOptimizer:
    """Fits the Benter log-linear stacker on STOP_SELL-anchored P_mkt.

    Parameters
    ----------
    market_anchor : 'stop_sell' | 'final'
        Which odds column to use for the public-consensus column. Defaults
        to 'stop_sell' (point-in-time correct). 'final' is provided only
        for backwards-compatible diagnostics.
    """

    def __init__(self,
                 model_a_csv: str = 'model_a_oof_predictions.csv',
                 model_b_csv: str = 'model_b_oof_predictions.csv',
                 include_market: bool = True,
                 weight_bounds: tuple = (0.0, 3.0),
                 market_anchor: str = 'stop_sell'):
        self.df_a = pd.read_csv(model_a_csv)
        self.df_b = pd.read_csv(model_b_csv)
        self.include_market = include_market
        self.weight_bounds = weight_bounds
        self.market_anchor = market_anchor

    def _resolve_anchor_column(self, df: pd.DataFrame) -> str:
        if self.market_anchor == 'stop_sell':
            if 'stop_sell_odds' not in df.columns:
                logging.warning(
                    "stop_sell_odds missing from OOF CSVs — falling back to "
                    "win_odds (FINAL). Re-train with the refactored "
                    "trainer_residual / trainer_independent to remove "
                    "look-ahead bias from P_mkt.")
                return 'win_odds'
            return 'stop_sell_odds'
        return 'win_odds'

    def _merge_oof(self) -> pd.DataFrame:
        a = self.df_a.rename(columns={'P_calibrated': 'P_A'})
        b = self.df_b.rename(columns={'P_calibrated': 'P_B'})

        # Prefer stop_sell_odds from EITHER side; if both present, A wins.
        keep_a = ['race_id', 'horse_code', 'finish_position', 'P_A', 'win_odds']
        if 'stop_sell_odds' in a.columns:
            keep_a.append('stop_sell_odds')
        keep_b = ['race_id', 'horse_code', 'P_B']
        if 'stop_sell_odds' in b.columns and 'stop_sell_odds' not in keep_a:
            keep_b.append('stop_sell_odds')

        df = pd.merge(a[keep_a], b[keep_b],
                      on=['race_id', 'horse_code'], how='inner')
        df = df.dropna(subset=['P_A', 'P_B', 'finish_position', 'win_odds'])
        df['is_winner'] = (df['finish_position'] == 1).astype(int)

        anchor_col = self._resolve_anchor_column(df)
        logging.info("Stacker market anchor: %s", anchor_col)

        # Build P_mkt from chosen anchor; also compute the FINAL anchor
        # for diagnostic comparison.
        df['P_mkt_raw'] = 1.0 / df[anchor_col]
        df['P_mkt'] = (df['P_mkt_raw']
                       / df.groupby('race_id')['P_mkt_raw'].transform('sum'))

        df['P_mkt_final_raw'] = 1.0 / df['win_odds']
        df['P_mkt_final'] = (df['P_mkt_final_raw']
                             / df.groupby('race_id')['P_mkt_final_raw'].transform('sum'))

        # Drop races without exactly one winner (leakage guard).
        winners_per_race = df.groupby('race_id')['is_winner'].sum()
        good_races = winners_per_race[winners_per_race == 1].index
        dropped = df['race_id'].nunique() - len(good_races)
        if dropped > 0:
            logging.info("Dropping %d races with !=1 winner.", dropped)
        df = df[df['race_id'].isin(good_races)].reset_index(drop=True)

        for col in ('P_A', 'P_B', 'P_mkt', 'P_mkt_final'):
            df[col] = df[col] / df.groupby('race_id')[col].transform('sum')
        return df

    def fit_stacker(self) -> BenterLogLinearStacker:
        logging.info("Merging Model A and Model B OOF predictions...")
        df = self._merge_oof()
        logging.info("Aligned %d runs across %d races.",
                     len(df), df['race_id'].nunique())

        loss_a   = log_loss(df['is_winner'], df['P_A'])
        loss_b   = log_loss(df['is_winner'], df['P_B'])
        loss_mkt = log_loss(df['is_winner'], df['P_mkt'])
        loss_mkt_final = log_loss(df['is_winner'], df['P_mkt_final'])
        gap_diag = loss_mkt - loss_mkt_final
        logging.info("\n--- Baseline LogLoss ---")
        logging.info("Public (FINAL)       : %.5f   <-- look-ahead biased", loss_mkt_final)
        logging.info("Public (STOP_SELL)   : %.5f   <-- point-in-time correct", loss_mkt)
        logging.info("FINAL - STOP_SELL gap: %+.5f   <-- size of late-money signal", -gap_diag)
        logging.info("Model A (Residual)   : %.5f", loss_a)
        logging.info("Model B (Physics)    : %.5f", loss_b)

        if self.include_market:
            cols  = ['P_A', 'P_B', 'P_mkt']
            names = ['P_A', 'P_B', 'P_mkt_stop_sell']
        else:
            cols  = ['P_A', 'P_B']
            names = ['P_A', 'P_B']

        P = df[cols].values
        stacker = BenterLogLinearStacker(
            bounds=self.weight_bounds, model_names=names,
        ).fit(P=P, race_ids=df['race_id'].values,
              y=df['is_winner'].values, verbose=True)

        best_single = (min(loss_a, loss_b, loss_mkt) if self.include_market
                       else min(loss_a, loss_b))
        if stacker.loss_ < best_single:
            logging.info("SUCCESS: ensemble beats best standalone by %.5f.",
                         best_single - stacker.loss_)
        else:
            logging.warning("Ensemble did NOT beat best standalone. "
                            "Check anchor column / merge keys.")
        return stacker


if __name__ == "__main__":
    try:
        opt = EnsembleOptimizer(
            'model_a_oof_predictions.csv',
            'model_b_oof_predictions.csv',
            include_market=True,
            market_anchor='stop_sell',
        )
        stacker = opt.fit_stacker()
        joblib.dump(stacker, 'ensemble_stacker.pkl')
        logging.info("Stacker saved to ensemble_stacker.pkl")
    except FileNotFoundError:
        logging.error("OOF CSVs not found. Run both trainers first.")
