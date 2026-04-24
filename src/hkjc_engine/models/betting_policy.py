"""
Parimutuel betting policy.

Implements:
  - Baker-McHale (2013) parameter-uncertainty EV hurdle
  - Calibration-shrinkage adjustment of model probabilities
  - MacLean-Ziemba fractional Kelly (MacLean, Ziemba & Blazenko 1992; MacLean, Thorp, Zhao & Ziemba 2011)
  - Smoczynski-Tomkins (2010) simultaneous-Kelly cap via proportional shrinkage to a race-level bankroll fraction

References
----------
Baker, R.D. & McHale, I.G. (2013). Optimal betting under parameter uncertainty: improving the Kelly criterion. Decision Analysis 10(3).
Kelly, J.L. (1956). A new interpretation of information rate. BSTJ 35(4).
MacLean, L.C., Ziemba, W.T. & Blazenko, G. (1992). Growth versus security in dynamic investment analysis. Management Science 38(11): 1562-1585.
MacLean, L.C., Thorp, E.O., Zhao, Y. & Ziemba, W.T. (2011). Medium term simulations of the full Kelly and fractional Kelly investment strategies. In: The Kelly Capital Growth Investment Criterion.
Smoczynski, P. & Tomkins, D. (2010). An explicit solution to the problem of optimizing the allocations of a bettor's wealth when wagering on horse races. Mathematical Scientist 35(1): 10-17.
"""
import numpy as np

def shrink_probability(p_raw: float, shrinkage: float) -> float:
    """
    Apply the empirical calibration ratio c = Sum(y) / Sum(p_hat) to a raw
    model probability. Under a scale-miscalibration assumption,

        p_true ~= shrinkage * p_raw

    Call this BEFORE computing EV or Kelly fraction.
    """
    if p_raw <= 0:
        return 0.0
    return min(shrinkage * p_raw, 1.0 - 1e-9)


def shrunk_ev(p_raw: float, odds: float, shrinkage: float) -> float:
    """EV = shrinkage*p_hat*d - 1. Returns negative values; caller filters."""
    if odds <= 1.0 or np.isnan(odds):
        return -1.0
    return shrink_probability(p_raw, shrinkage) * odds - 1.0


def get_ev_hurdle(odds: float,
                  base: float = 0.02,
                  longshot_buffer: float = 0.015,
                  longshot_threshold: float = 15.0) -> float:
    """
    Minimum EV required to place a bet, derived from the Baker-McHale (2013) parameter-uncertainty framework.

    Structure:
        h*(d) = base                          if d < longshot_threshold
              = base + longshot_buffer        if d >= longshot_threshold

    The flat floor absorbs first-order variance in p_hat induced by finite training data. 
    The longshot buffer reflects that our OOF sample has fewer longshot observations (so sigma_p is genuinely larger) AND that log-linear pool sharpening is most aggressive at the tails.
    Replaces the ad-hoc 0.10 + log(1+d)*0.12 formula, which required 43% EV at odds 15 and 57% at odds 50 -- effectively a longshot ban with no decision-theoretic justification.
    """
    if odds <= 1.0 or np.isnan(odds):
        return 1e9
    if odds >= longshot_threshold:
        return base + longshot_buffer
    return base

def fractional_kelly_stake(p_raw: float,
                           odds: float,
                           bankroll: float,
                           kelly_fraction: float = 0.25,
                           shrinkage: float = 0.75,
                           per_bet_cap: float = 0.02,
                           min_stake_abs: float = 10.0) -> float:
    """
    MacLean-Ziemba fractional Kelly stake on a single horse.

    Steps:
        1. Shrink probability:           p_adj = shrinkage * p_raw
        2. Full Kelly on shrunk prob:    f* = (p_adj*d - 1) / (d - 1)
        3. Fractional Kelly:             f = kelly_fraction * f*
        4. Per-bet cap:                  f = min(f, per_bet_cap)
        5. Stake in dollars:             stake = f * bankroll
        6. Skip if stake < min_stake_abs

    Parameters
    ----------
    p_raw : model's raw win probability (BEFORE shrinkage)
    odds : decimal odds (HKJC quote, e.g. 5.2)
    bankroll : current bankroll in dollars
    kelly_fraction : MacLean-Ziemba alpha in (0, 1]. Recommended 0.25-0.50.
        Growth retained ~= alpha*(2-alpha); drawdown risk retained ~= alpha^2.
        At alpha=0.25, you retain 44% of full-Kelly growth with 6% of the drawdown.
    shrinkage : empirical calibration ratio c = sum(y) / sum(p_hat).
        Set from the most recent walk-forward window; default 0.75 matches your recent empirical slip of ~0.73.
    per_bet_cap : hard upper bound as a fraction of bankroll. 2% is a volatility guard against pathological single-bet exposure.
    min_stake_abs : skip bets below this dollar amount.

    Returns
    -------
    Stake in dollars (0.0 if the bet should be skipped).
    """
    if odds <= 1.0 or np.isnan(odds) or p_raw <= 0 or bankroll <= 0:
        return 0.0

    p_adj = shrink_probability(p_raw, shrinkage)
    ev_adj = p_adj * odds - 1.0
    if ev_adj <= 0:
        return 0.0

    b = odds - 1.0
    f_star = ev_adj / b
    f = min(kelly_fraction * f_star, per_bet_cap)

    stake = f * bankroll
    return stake if stake >= min_stake_abs else 0.0

def cap_simultaneous_stakes(stakes: np.ndarray,
                            bankroll: float,
                            race_cap: float = 0.04) -> np.ndarray:
    """
    When betting multiple mutually-exclusive horses in the same race, cap the SUM of stakes at `race_cap` of bankroll (default 4%). 
    If the naive sum of per-horse Kelly stakes exceeds the cap, shrink all stakes proportionally.

    This is an approximation to the Smoczynski-Tomkins (2010) exact iterative solution. It:
      - preserves the relative ordering of per-horse conviction,
      - is always conservative (under-stakes vs. exact ST),
      - avoids numerical fragility when an individual f_i is near zero.

    The 4% cap corresponds to roughly alpha=0.25 Kelly on a typical 2-3 horse bet set with 1.3 edge ratio.

    Parameters
    ----------
    stakes : per-horse dollar stakes from fractional_kelly_stake
    bankroll : current bankroll
    race_cap : max fraction of bankroll risked on a single race

    Returns
    -------
    np.ndarray of (possibly shrunk) stakes.
    """
    stakes = np.asarray(stakes, dtype=float)
    if stakes.size == 0:
        return stakes
    total = stakes.sum()
    cap_dollars = race_cap * bankroll
    if total <= cap_dollars or total <= 0:
        return stakes
    return stakes * (cap_dollars / total)


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
                     max_odds: float = 25.0):
    """
    Given aligned per-horse arrays of p_raw (raw model probabilities) and odds for a SINGLE race, return (keep_idx, stakes) where keep_idx are indices into the input arrays and stakes are the dollar amounts.

    Pipeline (in order):
      1. shrinkage applied to p_raw
      2. EV hurdle filter
      3. max_odds ceiling
      4. top_k by shrunk EV
      5. per-horse fractional Kelly with per-bet cap
      6. race-level proportional shrinkage (Smoczynski-Tomkins cap)
      7. min_stake_abs filter
    """
    p_raw = np.asarray(p_raw, dtype=float)
    odds  = np.asarray(odds,  dtype=float)
    if p_raw.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    p_adj  = np.minimum(shrinkage * p_raw, 1 - 1e-9)
    ev_adj = p_adj * odds - 1.0
    hurdle = np.where(odds >= longshot_threshold,
                      base_hurdle + longshot_buffer, base_hurdle)

    qual = (ev_adj >= hurdle) & (odds <= max_odds) & (odds > 1.0)
    idx = np.where(qual)[0]
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if idx.size > top_k:
        order = idx[np.argsort(-ev_adj[idx])][:top_k]
    else:
        order = idx

    b = odds[order] - 1.0
    f_star = ev_adj[order] / b
    f = np.minimum(kelly_fraction * f_star, per_bet_cap)
    stakes = f * bankroll

    stakes = cap_simultaneous_stakes(stakes, bankroll, race_cap)

    keep = stakes >= min_stake_abs
    if not keep.any():
        return np.array([], dtype=int), np.array([], dtype=float)
    return order[keep], stakes[keep]