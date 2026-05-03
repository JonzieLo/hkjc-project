import numpy as np
from scipy.stats import gamma, multivariate_normal, norm

class CopulaGammaSimulator(SternGammaSimulator):
    
    def __init__(self, r_shape: float = 2.5, n_quad_nodes: int = 64):
        super().__init__(r_shape, n_quad_nodes)
        
    def _nearest_positive_definite(self, A):
        """Find the nearest positive-definite matrix to input A."""
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if self._is_pd(A3): return A3
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_pd(A3):
            eig_min = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-eig_min * k**2 + spacing)
            k += 1
        return A3

    def _is_pd(self, B):
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def simulate_exotics(self, p_target: np.ndarray, horse_nos: list[str], Sigma: np.ndarray, n_paths: int = 16384) -> dict:
        """
        Sigma: Empirical rank-correlation matrix (N x N) of the field's pace archetypes.
        """
        n_horses = len(p_target)
        if n_horses < 3: return {}

        lambdas = self.fit_lambdas(p_target)
        
        # 1. Regularize Sigma to ensure positive definiteness
        cov_matrix = self._nearest_positive_definite(Sigma)
        
        # 2. Draw Multivariate Normal paths
        mvn = multivariate_normal(mean=np.zeros(n_horses), cov=cov_matrix, allow_singular=True)
        Z_sim = mvn.rvs(size=n_paths)
        
        # 3. Copula transform: map to correlated Uniforms
        U_corr = norm.cdf(Z_sim)
        
        # 4. Map to marginal Gammas
        X_standard = gamma.ppf(U_corr, a=self.r)
        T_sim = X_standard / lambdas
        
        # 5. Extract finishes (Identical to original implementation)
        ranks = np.argsort(T_sim, axis=1)
        # ... proceed with combinatorial extraction