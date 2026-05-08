import logging
import joblib
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator

logging.basicConfig(level=logging.INFO, format='%(message)s')

class SmoothedIsotonicCalibrator:
    def __init__(self):
        self.ir = IsotonicRegression(y_min=1e-6, y_max=1-1e-6, out_of_bounds='clip')
        self.spline = None
        self.min_val = 1e-6
        self.max_val = 1 - 1e-6

    def fit(self, scores, y):
        scores = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self.ir.fit(scores, y)
        x_unique = np.unique(scores)
        y_iso = self.ir.predict(x_unique)
        if len(x_unique) > 3:
            self.spline = PchipInterpolator(x_unique, y_iso)
        else:
            self.spline = None
        return self

    def predict(self, scores):
        scores = np.asarray(scores, dtype=float).ravel()
        if self.spline is not None:
            p = self.spline(scores)
        else:
            p = self.ir.predict(scores)
        return np.clip(p, self.min_val, self.max_val)

    def predict_proba(self, scores):
        p = self.predict(scores)
        return np.column_stack([1 - p, p])


class StratifiedSmoothedIsotonicCalibrator:
    def __init__(self):
        self.calibrators = {}
        self.global_calibrator = SmoothedIsotonicCalibrator()

    def fit(self, scores, strata, y):
        scores = np.asarray(scores, dtype=float).ravel()
        strata = np.asarray(strata).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self.global_calibrator.fit(scores, y)
        for s in np.unique(strata):
            mask = (strata == s)
            if mask.sum() > 50:
                calib = SmoothedIsotonicCalibrator()
                calib.fit(scores[mask], y[mask])
                self.calibrators[s] = calib
        return self

    def predict(self, scores, strata=None):
        scores = np.asarray(scores, dtype=float).ravel()
        if strata is None:
            return self.global_calibrator.predict(scores)
        strata = np.asarray(strata).ravel()
        p = np.zeros_like(scores, dtype=float)
        for s in np.unique(strata):
            mask = (strata == s)
            if s in self.calibrators:
                p[mask] = self.calibrators[s].predict(scores[mask])
            else:
                p[mask] = self.global_calibrator.predict(scores[mask])
        return p

    def predict_proba(self, scores, strata=None):
        p = self.predict(scores, strata)
        return np.column_stack([1 - p, p])


class BayesianHierarchicalStacker:
    def __init__(self, mode='exotics'):
        self.weights = None
        self.loss_ = None
        self.mode = mode
        
    def fit(self, P, race_ids, y, I_valid, verbose=True):
        logP = np.log(np.clip(P, 1e-12, 1 - 1e-12))
        race_idx, unique_races = pd.factorize(race_ids)
        
        with pm.Model() as model:
            w_A = pm.HalfNormal("w_A", sigma=1.0)
            w_B = pm.HalfNormal("w_B", sigma=1.0)
            
            # Use two global scalar parameters instead of 3,400+ race-specific parameters
            if self.mode == 'win':
                w_mkt_live = pm.TruncatedNormal("w_mkt_live", mu=0.05, sigma=0.05, lower=0.0, upper=0.15)
                w_mkt_fallback = pm.TruncatedNormal("w_mkt_fallback", mu=0.05, sigma=0.05, lower=0.0, upper=0.15)
            else:
                w_mkt_live = pm.TruncatedNormal("w_mkt_live", mu=0.85, sigma=0.5, lower=0.0, upper=2.0)
                w_mkt_fallback = pm.TruncatedNormal("w_mkt_fallback", mu=0.05, sigma=0.05, lower=0.0, upper=0.5)
            
            w_mkt_expanded = pt.where(I_valid == 1, w_mkt_live, w_mkt_fallback)
            
            logits = w_A * logP[:, 0] + w_B * logP[:, 1] + w_mkt_expanded * logP[:, 2]
            
            logits_clipped = pt.clip(logits, -50, 20)
            exp_logits = pt.exp(logits_clipped)
            
            # Add small epsilon to prevent NaN division on heavy underflows
            sum_exp = pt.bincount(race_idx, weights=exp_logits) + 1e-12
            
            P_ens = exp_logits / sum_exp[race_idx]
            y_obs = pm.Bernoulli("y_obs", p=P_ens, observed=y)
            
            logging.info(f"Fitting Bayesian Stacker ({self.mode.upper()}) via ADVI...")
            # progressbar=False hides the repetitive noisy ADVI progress bars
            mean_field = pm.fit(n=30000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01), progressbar=False)
            trace = mean_field.sample(1000)
            
        self.weights = np.array([
            trace.posterior['w_A'].mean().item(),
            trace.posterior['w_B'].mean().item(),
            trace.posterior['w_mkt_live'].mean().item(),
            trace.posterior['w_mkt_fallback'].mean().item()
        ])
        
        P_4col = np.column_stack([P, I_valid])
        P_ens_final = self.predict(P_4col, race_ids)
        
        self.loss_ = log_loss(y, P_ens_final)
        
        if verbose:
            logging.info(f"\n--- Stacker Weights ({self.mode.upper()}) ---")
            logging.info(f"w_A (Residual)       : {self.weights[0]:.4f}")
            logging.info(f"w_B (Physics/Cox)    : {self.weights[1]:.4f}")
            logging.info(f"w_mkt (Live-Tick)    : {self.weights[2]:.4f}")
            logging.info(f"w_mkt (Fallback)     : {self.weights[3]:.4f}")
            logging.info(f"Ensemble LogLoss     : {self.loss_:.5f}")
            
        return self

    def predict(self, P, race_ids):
        # 1. Separate probabilities from the validity indicator
        if P.shape[1] == 4:
            I_valid = P[:, 3]
            P_probs = P[:, :3]
        else:
            # Fallback for live inference (run_bot.py) which only passes 3 columns
            I_valid = np.ones(len(P))
            P_probs = P
            
        logP = np.log(np.clip(P_probs, 1e-12, 1 - 1e-12))
        
        w_A = self.weights[0]
        w_B = self.weights[1]
        
        # 2. Dynamically assign market weight based on I_valid
        if len(self.weights) == 4:
            w_mkt_live = self.weights[2]
            w_mkt_fallback = self.weights[3]
            w_mkt = np.where(I_valid == 1, w_mkt_live, w_mkt_fallback)
        else:
            w_mkt = self.weights[2]
            
        # Vectorized Log-Linear Combination
        logits = logP[:, 0] * w_A + logP[:, 1] * w_B + logP[:, 2] * w_mkt
        
        s = pd.Series(logits, index=race_ids)
        s = s - s.groupby(level=0).transform('max')
        e = np.exp(s.values)
        
        df = pd.DataFrame({'e': e, 'rid': race_ids})
        df['z'] = df.groupby('rid')['e'].transform('sum')
        return (df['e'] / df['z']).values


class EnsembleOptimizer:
    def __init__(self, model_a_csv, model_b_csv, include_market=True, market_anchor='stop_sell', mode='exotics'):
        self.df_a = pd.read_csv(model_a_csv)
        self.df_b = pd.read_csv(model_b_csv)
        self.include_market = include_market
        self.market_anchor = market_anchor
        self.mode = mode

    def fit_stacker(self):
        logging.info(f"Merging Model A and Model B OOF predictions for {self.mode.upper()}...")
        df = pd.merge(self.df_a, self.df_b, on=['race_id', 'horse_code'])
        
        # Safely resolve column names handling Pandas _x/_y suffix collision logic
        win_odds_col = 'win_odds_x' if 'win_odds_x' in df.columns else 'win_odds'
        fp_col = 'finish_position_x' if 'finish_position_x' in df.columns else 'finish_position'
        
        if 'stop_sell_odds' in df.columns:
            df['I_valid'] = (df['stop_sell_odds'].notna() & (df['stop_sell_odds'] != df[win_odds_col])).astype(int)
        elif 'stop_sell_pla_odds' in df.columns:
            df['I_valid'] = 1 # We matched rows with synthetic/live PLA odds
        else:
            df['I_valid'] = 0
            
        if self.mode == 'pla':
            target_odds = 'stop_sell_pla_odds' if 'stop_sell_pla_odds' in df.columns else win_odds_col
        else:
            target_odds = 'stop_sell_odds' if 'stop_sell_odds' in df.columns else win_odds_col
            
        # Fallback handling for target_odds missing/zeroes
        df['P_mkt_raw'] = 1.0 / np.maximum(df[target_odds].fillna(1.0).astype(float), 1.0)
        df['P_mkt'] = df['P_mkt_raw'] / df.groupby('race_id')['P_mkt_raw'].transform('sum')
        
        # Clean dataframe from missing calibration scores or targets
        df = df.dropna(subset=['P_calibrated_x', 'P_calibrated_y', 'P_mkt', fp_col])
        
        P = df[['P_calibrated_x', 'P_calibrated_y', 'P_mkt']].values
        
        if self.mode == 'pla' and 'is_placed' in df.columns:
            y = df['is_placed'].astype(int).values
        else:
            y = (df[fp_col] == 1).astype(int).values
        
        stacker = BayesianHierarchicalStacker(mode=self.mode).fit(
            P=P, race_ids=df['race_id'].values, y=y, I_valid=df['I_valid'].values
        )
        return stacker