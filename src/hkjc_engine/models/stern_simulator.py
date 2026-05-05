import os
import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.stats import gamma, multivariate_t, t
from scipy.optimize import least_squares
import logging

log = logging.getLogger(__name__)

class CopulaGammaSimulator:
    def __init__(self, r_shape: float = 2.5, n_quad_nodes: int = 64, copula_path: str = 'pace_copula.npy', dof: float = 4.0):
        # Keep r_shape for backwards compatibility with existing backtester calls
        self.default_r = float(r_shape) 
        self.n_nodes = n_quad_nodes
        self.copula_path = copula_path
        
        # dof (degrees of freedom) controls joint tail dependence for the t-Copula
        # Lower dof = more extreme joint pace collapses.
        self.dof = dof
        
        # Standard Laguerre quadrature weights and roots (for e^-x weight)
        # We no longer calculate `self.measure` here globally because `r` is now heteroskedastic
        self.x_k, self.w_k = np.polynomial.laguerre.laggauss(self.n_nodes)
        
        # Load the global 5x5 pace copula correlation matrix
        if os.path.exists(self.copula_path):
            self.global_sigma = np.load(self.copula_path)
            log.debug(f"Loaded global pace copula from {self.copula_path}")
        else:
            self.global_sigma = np.eye(5)
            log.warning(f"Copula file {self.copula_path} missing. Falling back to independent Stern model.")

    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find the nearest positive-definite matrix (Higham, 1988)."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            spacing = np.spacing(np.linalg.norm(A))
            I = np.eye(A.shape[0])
            k = 1
            while True:
                try:
                    eig_min = np.min(np.real(np.linalg.eigvals(A3)))
                    A_test = A3 + I * (-eig_min * (k**2) + spacing)
                    np.linalg.cholesky(A_test)
                    return A_test
                except np.linalg.LinAlgError:
                    k += 1

    def build_race_sigma(self, pace_z_scores: np.ndarray) -> np.ndarray:
        """Projects the global 5x5 pace copula onto the NxN race field."""
        # Standard normal boundaries for quintiles
        bins = [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf]
        archetypes = np.asarray(pd.cut(pace_z_scores, bins=bins, labels=[0, 1, 2, 3, 4]), dtype=int)
        
        n_horses = len(archetypes)
        race_sigma = np.zeros((n_horses, n_horses))
        
        for i in range(n_horses):
            for j in range(n_horses):
                race_sigma[i, j] = self.global_sigma[archetypes[i], archetypes[j]]
                
        np.fill_diagonal(race_sigma, 1.0) # Self-correlation is strictly 1.0
        return race_sigma

    def _get_heteroskedastic_shapes(self, pace_z_scores: np.ndarray, field_size: int) -> np.ndarray:
        """
        Dynamic shape parameters (r_i). 
        Lower r = Higher Variance (Fatter right tails / Traffic Frailty)
        """
        if pace_z_scores is None:
            return np.full(field_size, self.default_r)
            
        r_i = np.zeros(field_size)
        for i, z in enumerate(pace_z_scores):
            if z > 0.5:    # Front-Runner (Clean air, strictly physical variance)
                r_i[i] = 3.0
            elif z < -0.5: # Closer (Traffic dependent, heavy right-tail variance)
                r_i[i] = 1.8 
            else:          # Stalker
                r_i[i] = 2.5
        return r_i

    def _win_probs_from_lambdas(self, lambdas: np.ndarray, r_i: np.ndarray) -> np.ndarray:
        """Calculates exact analytical win probabilities for heterogeneous Gamma distributions."""
        n = len(lambdas)
        p_win = np.zeros(n)
        for i in range(n):
            ratios = lambdas / lambdas[i]
            scaled_x = np.outer(self.x_k, ratios)
            
            # Scipy broadcasting evaluates gammaincc(r_j, scaled_x_j) perfectly
            beat_probs = sp.gammaincc(r_i, scaled_x)
            beat_probs[:, i] = 1.0  # Horse i doesn't have to beat itself
            
            joint_beat_prob = np.prod(beat_probs, axis=1)
            
            # Compute dynamic Laguerre measure because r_i is now heterogeneous
            pdf_part = (self.x_k ** (r_i[i] - 1.0)) / sp.gamma(r_i[i])
            
            p_win[i] = np.sum(self.w_k * pdf_part * joint_beat_prob)
            
        return p_win / np.sum(p_win)

    def fit_lambdas(self, p_target: np.ndarray, r_i: np.ndarray) -> np.ndarray:
        p_target = np.asarray(p_target, dtype=float)
        
        # Prevent np.log(0.0) float64 underflow
        p_target = np.clip(p_target, 1e-6, 1.0)
        p_target /= np.sum(p_target)
        
        n = len(p_target)
        if n < 2: return np.ones(n)

        # Initial guess must respect the varying r_i scaling
        alpha_guess = np.log(p_target ** (1.0 / r_i))
        alpha_guess -= np.mean(alpha_guess)

        def objective(alpha):
            lambdas = np.exp(alpha)
            p_calc = self._win_probs_from_lambdas(lambdas, r_i)
            return (p_calc - p_target) / np.sqrt(p_target)

        res = least_squares(objective, x0=alpha_guess, method='lm', xtol=1e-8, ftol=1e-8)
        lambdas_fit = np.exp(res.x)
        lambdas_fit /= np.mean(lambdas_fit) 
        return lambdas_fit

    def simulate_exotics(self, p_target: np.ndarray, horse_nos: list[str], pace_z_scores: np.ndarray = None, n_paths: int = 16384) -> dict:
        """
        pace_z_scores: Optional 1D array of `relative_early_pace` for the N horses. 
                       If None, falls back to standard independent Stern assumption.
        """
        n_horses = len(p_target)
        if n_horses < 3: return {'WIN': {}, 'PLA': {}, 'QIN': {}, 'QPL': {}, 'TRI': {}}

        # Extract heteroskedastic shape distributions based on pace
        r_i = self._get_heteroskedastic_shapes(pace_z_scores, n_horses)
        
        # Fit scale (lambda) parameters accommodating the heterogeneous shapes
        lambdas = self.fit_lambdas(p_target, r_i)

        # 1. Copula Covariance Matrix Generation
        if pace_z_scores is not None and len(pace_z_scores) == n_horses:
            empirical_sigma = self.build_race_sigma(pace_z_scores)
            cov_matrix = self._nearest_positive_definite(empirical_sigma)
        else:
            cov_matrix = np.eye(n_horses)
        
        # 2. Student-t Correlated draws (Generates extreme joint pace collapses)
        mvt = multivariate_t(loc=np.zeros(n_horses), shape=cov_matrix, df=self.dof, allow_singular=True)
        Z_sim = mvt.rvs(size=n_paths)
        
        # 3. Transform to Correlated Uniforms via the t-CDF
        U_corr = t.cdf(Z_sim, df=self.dof)
        
        # 4. Map to Marginals applying heterogeneous traffic-frailty parameters
        X_standard = gamma.ppf(U_corr, a=r_i)
        T_sim = X_standard / lambdas
        
        # 5. Extract Rank Configurations
        ranks = np.argsort(T_sim, axis=1)
        top_1 = ranks[:, 0]
        top_2 = ranks[:, :2]
        top_3 = ranks[:, :3]

        horse_nos = np.array(horse_nos)
        def _format_counts(arr, k):
            sorted_combos = np.sort(arr, axis=1)
            unique_combos, counts = np.unique(sorted_combos, axis=0, return_counts=True)
            probs = {}
            for combo_idx, count in zip(unique_combos, counts):
                combo_str = "-".join(str(x) for x in sorted(horse_nos[combo_idx], key=int))
                probs[combo_str] = count / n_paths
            return probs

        results = {}
        win_counts = np.bincount(top_1, minlength=n_horses)
        results['WIN'] = {horse_nos[i]: win_counts[i] / n_paths for i in range(n_horses)}
        
        pla_counts = np.bincount(top_3.flatten(), minlength=n_horses)
        results['PLA'] = {horse_nos[i]: pla_counts[i] / n_paths for i in range(n_horses)}
        
        results['QIN'] = _format_counts(top_2, 2)
        
        qpl_pairs = np.vstack([top_3[:, [0, 1]], top_3[:, [0, 2]], top_3[:, [1, 2]]])
        results['QPL'] = _format_counts(qpl_pairs, 2)
        results['TRI'] = _format_counts(top_3, 3)

        return results