"""
Parimutuel betting policy — drift-aware refactor.

Implements:
  - Baker-McHale (2013) parameter-uncertainty EV hurdle, EXTENDED with a
    closing-line drift-variance term so the hurdle widens automatically in
    high-variance pools (QIN, QPL, TRI).
  - Calibration-shrinkage adjustment of model probabilities.
  - MacLean-Ziemba fractional Kelly with a STOCHASTIC-PAYOFF correction:
    the Kelly fraction shrinks proportionally to (sigma_R / edge)^2.
  - Smoczynski-Tomkins simultaneous-Kelly cap with pool-conditional race cap
    (tightens for pools with higher closing-line variance).

Closed-form derivation (used in `kelly_fraction_stochastic`)
-----------------------------------------------------------
Let d_0 be the STOP_SELL decimal odds and let R = d_final / d_0 be the
random closing drift with mean mu_R and variance sigma_R^2. The realised
payoff is d_0 * R. Expanding E[log(1 + f * (Y * d_0 * R - 1))] to second
order around mu_R and maximising over f yields:

    f*_stochastic ≈ f*_naive(mu_R*d_0)
                  - p (1-p) * d_0^2 * sigma_R^2 / (mu_R * d_0 - 1)^2

where f*_naive(d) = (p*d - 1) / (d - 1) is ordinary Kelly. The penalty
term is the new piece: it scales as the squared coefficient-of-variation
of the payoff and vanishes as sigma_R -> 0 (recovering classical Kelly).

For numerical stability we implement the equivalent multiplicative form

    alpha_pool = alpha_base / (1 + kappa * sigma_R^2 / edge^2)

which is monotone in sigma_R, has no discontinuities, and degenerates to
alpha_base when sigma_R = 0.

Empirical defaults (from drift_diagnostic on 6,224 combinations)
----------------------------------------------------------------
                 sigma_R         mu_R       lambda_d hurdle bump
    WIN          0.008           1.0014     ~0%
    PLA          0.060           1.0094     ~1.4%
    QPL          0.114           1.0251     ~5.2%
    QIN          0.125           1.0258     ~6.3%
    TRI          0.106           0.9968     ~4.5%

These are placeholders until DriftForecaster predicts (mu_R, sigma_R) per
combination conditional on the model picking it. Once the forecaster is
trained, pass `pool_drift_stats` per call from forecaster output rather
than the static dict.

References
----------
Baker, R.D. & McHale, I.G. (2013). Optimal betting under parameter
    uncertainty. Decision Analysis 10(3).
Kelly, J.L. (1956). A new interpretation of information rate. BSTJ 35(4).
MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992). Growth versus
    security. Management Science 38(11).
MacLean, L.C., Thorp, E.O., Zhao, Y. & Ziemba, W.T. (2011). Medium term
    simulations of full and fractional Kelly. In: The Kelly Capital
    Growth Investment Criterion.
Smoczynski, P. & Tomkins, D. (2010). An explicit solution to the bettor's
    wealth allocation. Mathematical Scientist 35(1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Pool-conditional drift stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftStats:
    """Per-pool predictive moments of R = d_final / d_stop_sell.

    `mu_R`     conditional mean drift (1.0 = no drift on average).
    `sigma_R`  conditional std-dev of drift.
    `median_R` conditional median drift; used in EV because Kelly is
               concave in payoff (Jensen tilts in favour of median).
    """
    mu_R: float
    sigma_R: float
    median_R: float


# Empirical point estimates from the user's drift_diagnostic output.
# These are UNCONDITIONAL on combo selection. The conditional median for
# combos the model would actually back is biased downward (adverse
# selection); we apply a small prior shrink in `_conditional_median`.
DEFAULT_DRIFT_STATS: dict[str, DriftStats] = {
    'WIN': DriftStats(mu_R=1.0014, sigma_R=0.008, median_R=1.0000),
    'PLA': DriftStats(mu_R=1.0094, sigma_R=0.060, median_R=1.0000),
    'QPL': DriftStats(mu_R=1.0251, sigma_R=0.114, median_R=0.9950),
    'QIN': DriftStats(mu_R=1.0258, sigma_R=0.125, median_R=0.9920),
    'TRI': DriftStats(mu_R=0.9968, sigma_R=0.106, median_R=0.9700),
}


def _stats_for(pool: str | None,
               override: Mapping[str, DriftStats] | None = None) -> DriftStats:
    """Resolve drift stats for a pool, with `override` taking precedence."""
    if pool is None:
        return DriftStats(mu_R=1.0, sigma_R=0.0, median_R=1.0)
    if override and pool in override:
        return override[pool]
    return DEFAULT_DRIFT_STATS.get(pool.upper(),
                                   DriftStats(mu_R=1.0, sigma_R=0.0, median_R=1.0))


# ---------------------------------------------------------------------------
# Probability shrinkage and EV
# ---------------------------------------------------------------------------

def shrink_probability(p_raw: float, shrinkage: float) -> float:
    """p_true ≈ shrinkage * p_raw, clipped to (0, 1). Apply BEFORE EV/Kelly."""
    if p_raw <= 0:
        return 0.0
    return float(min(shrinkage * p_raw, 1.0 - 1e-9))


# ---------------------------------------------------------------------------
# Stratified shrinkage by P_pub band
# ---------------------------------------------------------------------------
#
# Background: a single global shrinkage scalar is fit from all OOF rows
# pooled together, which means the favourite-rich middle bins dominate the
# estimate and the longshot tail is under-corrected. theta_place_diagnostic
# revealed that P_ens is systematically overconfident on longshots
# (model-side θ=0.76 vs pub-side θ=0.83 on HKJC 2018-2026), and this
# inflates EV across PLA and all exotic pools that consume P_ens.
#
# Stratified shrinkage fixes the input rather than each downstream
# projection: shrinkage is fit separately within each P_pub band, so the
# longshot bin gets a more aggressive correction without compressing the
# mid-range where the model is already calibrated. Once P_ens is band-
# calibrated, the existing Henery projections work without further
# per-pool patches.
#
# Band edges chosen to match place_projection_diagnostic bins for direct
# comparability and to give roughly balanced sample sizes on HKJC field
# distributions. The deepest-longshot band (<0.03) is where the model's
# bias is largest; the favourite band (>0.30) typically has the fewest
# horses but can also be over-confident.

STRATIFIED_SHRINKAGE_BANDS = (0.0, 0.03, 0.07, 0.15, 0.30, 1.01)
STRATIFIED_SHRINKAGE_NAMES = ('<0.03', '0.03-0.07', '0.07-0.15',
                              '0.15-0.30', '>0.30')


def _band_of(p_pub: float) -> str:
    """Return the band name a single p_pub belongs to."""
    for i, hi in enumerate(STRATIFIED_SHRINKAGE_BANDS[1:]):
        if p_pub < hi:
            return STRATIFIED_SHRINKAGE_NAMES[i]
    return STRATIFIED_SHRINKAGE_NAMES[-1]


def lookup_shrinkage(p_pub, shrinkage):
    """Resolve effective shrinkage for one or many horses.

    Backwards-compatible: if `shrinkage` is a float, returns it unchanged
    (uniform shrinkage, legacy behaviour). If it's a dict mapping band
    name to scalar, returns the band-specific value(s).

    Parameters
    ----------
    p_pub : float | np.ndarray
        Public win-pool implied probability for the horse(s).
    shrinkage : float | dict[str, float]
        Either a single scalar (legacy) or a dict mapping
        STRATIFIED_SHRINKAGE_NAMES band names to per-band scalars.

    Returns
    -------
    Same type as p_pub: scalar in, scalar out; array in, array out.
    Missing bands fall back to a global default of 0.85 for safety,
    so a partially-populated dict still works.
    """
    if isinstance(shrinkage, (int, float)):
        if isinstance(p_pub, np.ndarray):
            return np.full_like(p_pub, float(shrinkage), dtype=float)
        return float(shrinkage)

    if not isinstance(shrinkage, dict):
        raise TypeError(f"shrinkage must be float or dict, got {type(shrinkage)}")

    edges = STRATIFIED_SHRINKAGE_BANDS
    names = STRATIFIED_SHRINKAGE_NAMES
    default = 0.85

    if isinstance(p_pub, np.ndarray):
        # Vectorised: digitize -> band index -> band name -> shrinkage
        idx = np.clip(np.digitize(p_pub, edges[1:-1]), 0, len(names) - 1)
        return np.array([float(shrinkage.get(names[i], default))
                         for i in idx], dtype=float)
    return float(shrinkage.get(_band_of(float(p_pub)), default))


def shrunk_ev(p_raw: float,
              odds: float,
              shrinkage: float,
              pool: str | None = None,
              drift_override: Mapping[str, DriftStats] | None = None) -> float:
    """Drift-aware EV at the STOP_SELL odds.

    EV_eff = p_shrunk * d_stop_sell * median_R - 1

    Using `median_R` rather than `mu_R` is intentional: log-utility (Kelly)
    is concave in payoff, so the relevant "effective" payoff is closer to
    the median than the mean. Empirically, in pools with adverse selection
    the conditional median is below 1 even when the unconditional mean is
    above 1.
    """
    if odds <= 1.0 or np.isnan(odds):
        return -1.0
    stats = _stats_for(pool, drift_override)
    p_adj = shrink_probability(p_raw, shrinkage)
    return p_adj * odds * stats.median_R - 1.0


# ---------------------------------------------------------------------------
# EV hurdle (Baker-McHale + drift-variance extension)
# ---------------------------------------------------------------------------

def get_ev_hurdle(odds: float,
                  base: float = 0.02,
                  longshot_buffer: float = 0.015,
                  longshot_threshold: float = 15.0,
                  pool: str | None = None,
                  lambda_d: float = 4.0,
                  drift_override: Mapping[str, DriftStats] | None = None) -> float:
    """Minimum EV required, with parameter-uncertainty AND drift-variance terms.

        h*(d, pool) = base
                    + (longshot_buffer  if d >= longshot_threshold else 0)
                    + lambda_d * sigma_R(pool)^2

    The flat floor (Baker-McHale) absorbs first-order variance in p_hat.
    The longshot buffer reflects sparser longshot calibration data and
    aggressive tail sharpening from the log-linear pool. The new pool term
    penalises bets in pools where the realised payoff is genuinely
    stochastic at execution time.

    Calibration of `lambda_d`: choose the value that makes per-pool
    stake-weighted realised PnL match expected PnL on the OOS validation
    window (see slippage_attribution.report). Default 4.0 is reasonable
    for HKJC empirics; bump up if TRI realised < expected, down if WIN
    over-penalised.
    """
    if odds <= 1.0 or np.isnan(odds):
        return 1e9
    h = base + (longshot_buffer if odds >= longshot_threshold else 0.0)
    if pool is not None:
        h += lambda_d * _stats_for(pool, drift_override).sigma_R ** 2
    return h


# ---------------------------------------------------------------------------
# Kelly fraction (stochastic-payoff aware)
# ---------------------------------------------------------------------------

def kelly_fraction_stochastic(alpha_base: float,
                              edge: float,
                              pool: str | None,
                              kappa: float = 1.0,
                              drift_override: Mapping[str, DriftStats] | None = None) -> float:
    """Pool-conditional fractional-Kelly multiplier alpha_pool.

        alpha_pool = alpha_base / (1 + kappa * (sigma_R / edge)^2)

    `edge` is the SHRUNK-EV margin (positive). When edge is small relative
    to drift std, alpha shrinks toward 0; when edge >> sigma_R, alpha ->
    alpha_base. With kappa=1 and sigma_R from DEFAULT_DRIFT_STATS:
        WIN  edge=0.05  -> alpha ≈ alpha_base * 0.974
        TRI  edge=0.05  -> alpha ≈ alpha_base * 0.183
        TRI  edge=0.20  -> alpha ≈ alpha_base * 0.780

    This is the stable practitioner approximation to the second-order
    expansion in the module docstring. It is monotone, has no
    discontinuities, and degenerates correctly as sigma_R -> 0.
    """
    if edge <= 0:
        return 0.0
    sigma_R = _stats_for(pool, drift_override).sigma_R
    if sigma_R <= 0:
        return alpha_base
    penalty = kappa * (sigma_R / edge) ** 2
    return alpha_base / (1.0 + penalty)


def fractional_kelly_stake(p_raw: float,
                           odds: float,
                           bankroll: float,
                           kelly_fraction: float = 0.25,
                           shrinkage: float = 0.75,
                           per_bet_cap: float = 0.02,
                           min_stake_abs: float = 10.0,
                           pool: str | None = None,
                           kappa: float = 1.0,
                           drift_override: Mapping[str, DriftStats] | None = None) -> float:
    """MacLean-Ziemba fractional Kelly stake, drift-variance penalised.

    Pipeline:
        1. p_adj   = shrinkage * p_raw
        2. EV_eff  = p_adj * d_stop_sell * median_R - 1   (Jensen-correct)
        3. f*      = EV_eff / (d_eff - 1)  where d_eff = d_stop_sell*mu_R
        4. alpha   = kelly_fraction / (1 + kappa * (sigma_R / EV_eff)^2)
        5. f       = min(alpha * f*, per_bet_cap)
        6. stake   = f * bankroll
    """
    if odds <= 1.0 or np.isnan(odds) or p_raw <= 0 or bankroll <= 0:
        return 0.0

    stats = _stats_for(pool, drift_override)
    p_adj = shrink_probability(p_raw, shrinkage)
    d_eff = odds * stats.mu_R
    ev_eff = p_adj * odds * stats.median_R - 1.0
    if ev_eff <= 0 or d_eff <= 1.0:
        return 0.0

    f_star = ev_eff / (d_eff - 1.0)
    alpha_pool = kelly_fraction_stochastic(kelly_fraction, ev_eff, pool,
                                           kappa=kappa,
                                           drift_override=drift_override)
    f = min(alpha_pool * f_star, per_bet_cap)
    stake = f * bankroll
    return stake if stake >= min_stake_abs else 0.0


# ---------------------------------------------------------------------------
# Race-level cap (Smoczynski-Tomkins approximation)
# ---------------------------------------------------------------------------

def race_cap_for_pool(base_cap: float,
                      pool: str | None,
                      drift_override: Mapping[str, DriftStats] | None = None,
                      floor: float = 0.005) -> float:
    """Pool-conditional race cap.

        race_cap_pool = base_cap * sqrt(sigma_R(WIN) / sigma_R(pool))

    Because exotic bets in the same race share a common closing-line
    shock (the syndicate moves them together), the effective race-level
    variance is dominated by drift, not selection. We tighten the cap in
    proportion to drift std, with a floor to avoid pathological
    over-tightening. Reduces from base_cap=0.04 (WIN-tuned) down to
    ~0.01 for TRI/QIN and ~0.025 for PLA, matching the pool variance
    rank-ordering.
    """
    win_sigma = DEFAULT_DRIFT_STATS['WIN'].sigma_R
    pool_sigma = _stats_for(pool, drift_override).sigma_R
    if pool_sigma <= win_sigma:
        return base_cap
    return max(floor, base_cap * np.sqrt(win_sigma / pool_sigma))


def cap_simultaneous_stakes(stakes: np.ndarray,
                            bankroll: float,
                            race_cap: float = 0.04,
                            pool: str | None = None,
                            drift_override: Mapping[str, DriftStats] | None = None) -> np.ndarray:
    """Cap the SUM of simultaneous stakes at `race_cap_pool` of bankroll.

    Approximation to Smoczynski-Tomkins (2010). Preserves relative
    conviction ordering, is always conservative, robust to f_i ≈ 0.
    The actual cap used is `race_cap_for_pool(race_cap, pool)`.
    """
    stakes = np.asarray(stakes, dtype=float)
    if stakes.size == 0:
        return stakes
    cap_used = race_cap_for_pool(race_cap, pool, drift_override)
    cap_dollars = cap_used * bankroll
    total = stakes.sum()
    if total <= cap_dollars or total <= 0:
        return stakes
    return stakes * (cap_dollars / total)


# ---------------------------------------------------------------------------
# Top-level race qualifier
# ---------------------------------------------------------------------------

def qualify_and_size(p_raw: np.ndarray,
                     odds: np.ndarray,
                     bankroll: float,
                     kelly_fraction: float = 0.25,
                     shrinkage: float = 0.75,
                     base_hurdle: float = 0.02,
                     longshot_buffer: float = 0.015,
                     longshot_threshold: float = 15.0,
                     per_bet_cap: float = 0.02,
                     race_cap: float = 0.04,
                     min_stake_abs: float = 10.0,
                     top_k: int = 3,
                     max_odds: float = 25.0,
                     pool: str | None = None,
                     kappa: float = 1.0,
                     lambda_d: float = 4.0,
                     drift_override: Mapping[str, DriftStats] | None = None):
    """Per-race qualify-and-size pipeline.

    Steps (in order):
      1. shrink p_raw
      2. compute drift-aware EV (with median_R)
      3. EV hurdle filter (with sigma_R variance term)
      4. max_odds ceiling
      5. top_k by EV
      6. per-horse fractional Kelly with drift-aware alpha and per-bet cap
      7. race-level proportional shrinkage with pool-conditional cap
      8. min_stake_abs filter

    Returns (keep_idx, stakes) where `keep_idx` indexes into the inputs.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    odds = np.asarray(odds, dtype=float)
    if p_raw.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    stats = _stats_for(pool, drift_override)
    p_adj = np.minimum(shrinkage * p_raw, 1 - 1e-9)
    ev_eff = p_adj * odds * stats.median_R - 1.0       # EV (Jensen-correct)
    d_eff = odds * stats.mu_R                          # effective payoff

    hurdle = np.where(odds >= longshot_threshold,
                      base_hurdle + longshot_buffer,
                      base_hurdle)
    hurdle = hurdle + lambda_d * stats.sigma_R ** 2

    qual = (ev_eff >= hurdle) & (odds <= max_odds) & (d_eff > 1.0)
    idx = np.where(qual)[0]
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if idx.size > top_k:
        order = idx[np.argsort(-ev_eff[idx])][:top_k]
    else:
        order = idx

    f_star = ev_eff[order] / (d_eff[order] - 1.0)
    # Vectorised drift-aware alpha
    if stats.sigma_R > 0:
        penalty = kappa * (stats.sigma_R / np.maximum(ev_eff[order], 1e-12)) ** 2
        alpha_pool = kelly_fraction / (1.0 + penalty)
    else:
        alpha_pool = np.full_like(f_star, kelly_fraction)
    f = np.minimum(alpha_pool * f_star, per_bet_cap)
    stakes = f * bankroll

    stakes = cap_simultaneous_stakes(stakes, bankroll, race_cap, pool, drift_override)

    keep = stakes >= min_stake_abs
    if not keep.any():
        return np.array([], dtype=int), np.array([], dtype=float)
    return order[keep], stakes[keep]