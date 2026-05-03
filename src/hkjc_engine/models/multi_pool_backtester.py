"""
Multi-pool drift-aware backtester (Step 1 of the multi-pool roadmap).

Extends `XGBEnsembleBacktester` (which only settles WIN) to also settle
PLA / QIN / QPL / TRI. Sizing per pool reuses the live bot's existing
exotic-odds combinatoric logic (`p_quinella`, `p_quinella_place`, `p_trio`)
combined with the drift-aware `qualify_and_size` policy.

Coverage modes
--------------
This is the REAL-ODDS path: a race only contributes exotic bets if its
exotic odds are observable in `race_dividends`. Pre-Plan-C history that
has only WIN/PLA dividends contributes WIN+PLA bets but skips exotics.
The synthetic-odds path (project exotic odds from WIN-pool implied
probs) is Step 2 and is intentionally not implemented here — the two
paths give answers to different questions and conflating them breaks
the calibration story.

How exotic odds are derived for backtesting
-------------------------------------------
At backtest time we don't have the LIVE odds tape for every of the 91
QIN combinations or 364 TRI combinations per race. What we DO have is
the `race_dividends` row for the *winning* combination. We therefore
size only against bets that actually settled (i.e. combos that won,
plus a sample of unwon combos with FINAL-implied odds estimated from
the empirical takeout-adjusted public market).

This is the principled trade-off:

  * For each combination the bot's policy would have flagged at
    STOP_SELL, we estimate its FINAL-payoff odds. If the combo is the
    winner, we use the actual `race_dividends.dividend / 10`. If not,
    the bet pays $0 — we don't need exotic odds to settle a losing
    bet, only to gate the pre-bet EV.
  * The pre-bet EV gate uses STOP_SELL public odds projected from the
    WIN pool via Harville θ=1 (Step 2's synthetic-odds path), which
    gives an unbiased EV gate. The settlement then uses real
    `race_dividends`. This is a deliberate hybrid: gate on synthetic
    pre-bet odds (we have to — no live tape), settle on real post-bet
    dividends (we should — that's the actual payoff).

Important caveat (read this if the ROI looks too good)
------------------------------------------------------
Hybrid odds bias. Synthetic Harville-θ=1 odds at STOP_SELL are an
unbiased estimate of the *public consensus* but not of the *actual
parimutuel offer* the bot would have faced. Real exotic odds tend to
include extra inefficiency (longshots overround, syndicate movements
that don't show in WIN), and we ignore both directions of that. The
Step 1 backtest therefore prices exotic *policy* fidelity (does the
qualifier pick winning combos?) more accurately than exotic *PnL*
fidelity (does it make money against a real book?). Use Step 3
(real-odds path on POST_STOP_SELL coverage window) for true PnL
validation once you have ≥50 races of exotic coverage.
"""
from __future__ import annotations

import itertools
import logging
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from hkjc_engine.config import DB_URL, artifact
from hkjc_engine.data.stop_sell_loader import attach_win_anchor
from hkjc_engine.models.backtester import XGBEnsembleBacktester, _softmax
from hkjc_engine.models.betting_policy import (
    DEFAULT_DRIFT_STATS,
    DriftStats,
    cap_simultaneous_stakes,
    fractional_kelly_stake,
    get_ev_hurdle,
    qualify_and_size,
    race_cap_for_pool,
    shrunk_ev,
)
from hkjc_engine.models.drift_forecaster import (
    project_drift_to_exotic,
)
from hkjc_engine.models.feature_factory import calculate_base_margin

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# Pool naming reconciliation
# ---------------------------------------------------------------------------
#
# `live_odds_history` and the live bot use 3-letter HKJC API codes
# (WIN, PLA, QIN, QPL, TRI). `race_dividends`, populated by the
# results-page HTML scraper (`scraper_dividends.py`), uses the full
# English names HKJC prints on results pages: WIN, PLACE, QUINELLA,
# QUINELLA PLACE, TRIO. Two parallel naming conventions in the same
# database, never reconciled.
#
# We normalise at READ TIME rather than WRITE TIME — the dividend
# table is the system of record for what HKJC publishes, and a
# read-time map keeps the fix surgical without needing a one-shot
# UPDATE.
#
# CAREFUL: HKJC has two trifecta-family bets:
#   - TRIO   (unordered top-3, the default 'TRI' bet in the live bot)
#   - TIERCE (ordered top-3, an entirely different bet type)
# Our internal 'TRI' label maps to TRIO. We do NOT include TIERCE in
# the mapping — that would conflate two different markets.

POOL_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    'WIN':  ('WIN', 'WINNER'),
    'PLA':  ('PLA', 'PLACE'),
    'QIN':  ('QIN', 'QUINELLA'),
    'QPL':  ('QPL', 'QUINELLA PLACE'),
    'TRI':  ('TRI', 'TRIO'),
}


def _normalise_pool(stored_pool: str) -> str | None:
    """Map a stored race_dividends.pool string to our canonical 3-letter code.

    Returns None for pools we don't track (FORECAST, DOUBLE, TREBLE, etc.).
    """
    s = str(stored_pool).upper().strip()
    for canonical, variants in POOL_NAME_VARIANTS.items():
        if s in variants:
            return canonical
    return None


def _all_storage_names(pools: Iterable[str]) -> list[str]:
    """For each canonical pool code, expand to all storage variants."""
    out: list[str] = []
    for p in pools:
        out.extend(POOL_NAME_VARIANTS.get(p.upper(), (p.upper(),)))
    return out


# ---------------------------------------------------------------------------
# Combination helpers (shared with live bot conventions)
# ---------------------------------------------------------------------------

def _canonical_combo(combination) -> str:
    """Sort-and-dash-join. Mirrors live/snapshot_logger._canonical_combo
    so that bet-side and dividend-side combination strings are
    interchangeable. Accepts both ',' and '-' separators."""
    s = str(combination).strip()
    if "," in s or "-" in s:
        parts = sorted(int(x) for x in s.replace(",", "-").split("-") if x.strip())
        return "-".join(str(x) for x in parts)
    return s


# ---------------------------------------------------------------------------
# Harville-θ exotic probability projections
# ---------------------------------------------------------------------------
#
# These are the same math as live/run_bot.py but parameterised on theta_2
# / theta_3 instead of using globals, so the backtester can use the
# walk-forward-refit thetas from live_config.json.

def _p_order_by_idx(p_arr: np.ndarray, i1: int, i2: int,
                    i3: int | None, theta_2: float, theta_3: float) -> float:
    p1 = p_arr[i1]
    sum_t2 = np.sum(p_arr ** theta_2) - (p1 ** theta_2)
    if sum_t2 <= 0:
        return 0.0
    p2 = p_arr[i2]
    p_exact_2 = p1 * ((p2 ** theta_2) / sum_t2)
    if i3 is None:
        return float(p_exact_2)
    p3 = p_arr[i3]
    sum_t3 = np.sum(p_arr ** theta_3) - (p1 ** theta_3) - (p2 ** theta_3)
    if sum_t3 <= 0:
        return 0.0
    return float(p_exact_2 * ((p3 ** theta_3) / sum_t3))


def p_quinella(p_arr, i, j, theta_2, theta_3):
    return (_p_order_by_idx(p_arr, i, j, None, theta_2, theta_3)
            + _p_order_by_idx(p_arr, j, i, None, theta_2, theta_3))


def p_quinella_place(p_arr, i, j, theta_2, theta_3):
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        total += _p_order_by_idx(p_arr, i, j, k, theta_2, theta_3)
        total += _p_order_by_idx(p_arr, j, i, k, theta_2, theta_3)
        total += _p_order_by_idx(p_arr, i, k, j, theta_2, theta_3)
        total += _p_order_by_idx(p_arr, j, k, i, theta_2, theta_3)
        total += _p_order_by_idx(p_arr, k, i, j, theta_2, theta_3)
        total += _p_order_by_idx(p_arr, k, j, i, theta_2, theta_3)
    return total


def p_trio(p_arr, i, j, k, theta_2, theta_3):
    return sum(_p_order_by_idx(p_arr, *perm, theta_2, theta_3)
               for perm in itertools.permutations([i, j, k]))


def p_place(p_arr: np.ndarray, i: int,
            theta_2: float, theta_3: float) -> float:
    n = len(p_arr); p_1st = float(p_arr[i])
    p_2nd = sum(_p_order_by_idx(p_arr, j, i, None, theta_2, theta_3)
                for j in range(n) if j != i)
    p_3rd = 0.0
    for j in range(n):
        if j == i:
            continue
        for k in range(n):
            if k == i or k == j:
                continue
            p_3rd += _p_order_by_idx(p_arr, j, k, i, theta_2, theta_3)
    return p_1st + p_2nd + p_3rd


# ---------------------------------------------------------------------------
# Public-consensus exotic odds projection
# ---------------------------------------------------------------------------
#
# At backtest time we use the WIN-pool implied probabilities, project
# them through Harville with θ=1 (i.e. NO Henery discount — that's the
# "what the public would price this combo at" transformation), and then
# divide (1 - takeout_pool) by that probability to get the synthetic
# public odds. This is the backtest-time substitute for the live bot
# scraping QIN/QPL/TRI prices off Redis.

POOL_TAKEOUT = {
    'PLA': 0.175,
    'QIN': 0.175,
    'QPL': 0.175,
    'TRI': 0.250,
}


def _public_pla_odds(p_public: np.ndarray, i: int,
                     theta_2: float, theta_3: float) -> float:
    """Synthetic public PLA odds for horse i.

    `p_public` MUST be the public's WIN-pool probability (1/d_stop_sell
    normalised), NOT the model's P_model.

    CRITICAL: theta_2 and theta_3 must match the values used on the
    MODEL side (e.g. p_place(p_arr, theta_2_fit, theta_3_fit)). Using
    different thetas on the two sides creates a fictitious EV that has
    nothing to do with model edge — the asymmetry between Henery
    discount choices generates 5-15% phantom edge on every horse,
    independent of any real market mispricing.
    """
    p_pub_place = p_place(p_public, i, theta_2, theta_3)
    if p_pub_place <= 0:
        return 0.0
    return (1.0 - POOL_TAKEOUT['PLA']) / p_pub_place


def _public_combo_odds(p_public: np.ndarray, combo_idx: tuple,
                       calc_func, pool: str,
                       theta_2: float, theta_3: float) -> float:
    """Synthetic public exotic odds for a combo.

    Same caveat as `_public_pla_odds` — `p_public` must be the public's
    WIN-pool probability, AND theta_2/theta_3 must match the model side.
    Asymmetric theta would generate phantom EV.
    """
    p_pub = calc_func(p_public, *combo_idx, theta_2, theta_3)
    if p_pub <= 0:
        return 0.0
    return (1.0 - POOL_TAKEOUT[pool]) / p_pub


# ---------------------------------------------------------------------------
# Settlement helpers
# ---------------------------------------------------------------------------

def _build_dividend_lookup(dividends_df: pd.DataFrame) -> dict:
    """Map (race_id, pool_code, canonical_combo) -> dividend.

    `dividends_df` is the result of querying race_dividends. Combo
    strings are canonicalised to '-' separator + ascending order so
    QIN '3,1' and bet '1-3' match.
    """
    lookup: dict[tuple, float] = {}
    if dividends_df.empty:
        return lookup
    for _, row in dividends_df.iterrows():
        if pd.isna(row['pool_code']):
            continue
        key = (
            str(row['race_id']),
            str(row['pool_code']).upper(),
            _canonical_combo(row['combination']),
        )
        try:
            lookup[key] = float(row['dividend'])
        except (TypeError, ValueError):
            continue
    return lookup


def _winning_combos(finish_df: pd.DataFrame,
                    pla_combos_per_race: dict[str, set] | None = None,
                    ) -> dict:
    """Per-race winning combinations for each pool.

    Inputs
    ------
    finish_df : DataFrame[race_id, horse_no, finish_position]
    pla_combos_per_race : dict[race_id] -> set of horse_no strings that
        actually have a PLA dividend row for this race. If provided
        (recommended), this is the authoritative source for which
        horses paid place — it captures HKJC's field-size rules
        (no PLA in 3-runner fields, top-2 only in 4-runner fields,
        top-3 otherwise) and any race-specific cancellations directly
        from the dividend table. If omitted, the function falls back
        to a default top-3 assumption.

    Returns
    -------
    dict[race_id] -> {
        'WIN': '5',                       (winning horse_no)
        'PLA': {'1', '3', '5'},           (placer set; from dividends)
        'QIN': '3-5',                     (top-2 finishers, canonical)
        'QPL': {'1-3', '1-5', '3-5'},     (all pairs in top-3)
        'TRI': '1-3-5',                   (top-3 finishers, canonical)
    }

    Pool semantics
    --------------
    Per HKJC rules, PLA pays the top-3 horses (not field-size dependent
    in normal racing). This used to be coded as `top-3 if field>=7 else
    top-2`, which was incorrect. The authoritative source is the
    dividend table — if `pla_combos_per_race` is provided, we use that.

    QIN, QPL, TRI are determined by actual finishing order regardless
    of horse-number sort. The combinations are canonicalised to
    ascending horse-number form so a bet keyed '1-3' matches a race
    where horse 3 finished 1st and horse 1 finished 2nd.
    """
    out: dict[str, dict] = {}
    pla_combos_per_race = pla_combos_per_race or {}

    for race_id, g in finish_df.groupby('race_id'):
        g = g.sort_values('finish_position')
        if len(g) < 2:
            continue
        # Drop scratched / DNF (finish_position null or sentinel)
        g = g[g['finish_position'].notna() & (g['finish_position'] < 90)]
        if g.empty:
            continue

        top1 = int(g.iloc[0]['horse_no']) if len(g) >= 1 else None
        top2 = int(g.iloc[1]['horse_no']) if len(g) >= 2 else None
        top3 = int(g.iloc[2]['horse_no']) if len(g) >= 3 else None

        winner_no = str(top1)

        # PLA placers — prefer the dividend table when supplied,
        # otherwise default to top-3 finishers (HKJC standard rule).
        if race_id in pla_combos_per_race:
            placers_finish = pla_combos_per_race[race_id]
        else:
            placers_finish = {str(h) for h in (top1, top2, top3)
                              if h is not None}

        out[race_id] = {
            'WIN': winner_no,
            'PLA': placers_finish,
        }

        if top1 is not None and top2 is not None:
            out[race_id]['QIN'] = '-'.join(
                str(x) for x in sorted([top1, top2]))

            # QPL pairs draw from the same placer set used for PLA so
            # the rules stay coherent. If PLA paid only top-2 (4-runner
            # field), QPL has only one pair; if PLA paid all top-3,
            # QPL has three pairs.
            placer_ints = sorted(int(h) for h in placers_finish)
            qpl_pairs = set()
            for combo in itertools.combinations(placer_ints, 2):
                qpl_pairs.add('-'.join(str(x) for x in sorted(combo)))
            out[race_id]['QPL'] = qpl_pairs

        if top1 is not None and top2 is not None and top3 is not None:
            out[race_id]['TRI'] = '-'.join(
                str(x) for x in sorted([top1, top2, top3]))

    return out


# ---------------------------------------------------------------------------
# Multi-pool backtester
# ---------------------------------------------------------------------------

class MultiPoolBacktester(XGBEnsembleBacktester):
    """Drift-aware backtester for WIN + PLA + QIN + QPL + TRI.

    Inherits `XGBEnsembleBacktester` for model loading, STOP_SELL anchor
    handling, and stacker prediction. Overrides `run_backtest` to add
    pool-by-pool sizing and settlement. The original WIN-only
    `run_backtest` is preserved on the parent class.

    Exotic odds source modes
    ------------------------
    The exotic pools (PLA / QIN / QPL / TRI) need a price to gate EV
    against. Three modes:

    * 'real' (default) — Use real odds from `live_odds_history` when
      available. If a race has no live snapshot for a given pool, that
      pool is SKIPPED for that race (no exotic bets fired). This is the
      most accurate mode but only useful once enough live data exists.
      WIN bets always fire regardless of exotic coverage.

    * 'synthetic' — Use the θ-symmetric synthetic odds path
      (Harville projection of public WIN-pool implieds with takeout
      added back). Approximates the public exotic market but ignores
      pool-specific overround and longshot bias. Useful for
      stress-testing bet selection logic on the full historical
      window where real odds don't exist yet, but DON'T treat the
      ROI as a PnL projection.

    * 'hybrid' — Real odds when present, synthetic otherwise. Single
      ROI number across the full window, but mixes two odds sources
      so downstream calibration is harder. Use only if the noise from
      a small real-odds sample is unacceptable.

    Bet ledger rows are tagged with `odds_source = 'real' | 'synthetic'`
    so post-hoc analysis can split metrics by source.
    """

    POOLS_AVAILABLE: tuple[str, ...] = ('WIN', 'PLA', 'QIN', 'QPL', 'TRI')
    EXOTIC_ODDS_SOURCES: tuple[str, ...] = ('real', 'synthetic', 'hybrid')

    def __init__(self,
                 *args,
                 pools: Iterable[str] = ('WIN', 'PLA', 'QIN', 'QPL', 'TRI'),
                 top_n_horses_exotic: int = 8,
                 exotic_odds_source: str = 'real',
                 ledger_csv_path: str | None = None,
                 theta_place: float | None = None,
                 theta_model_place: float | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = tuple(p.upper() for p in pools)
        self.top_n_horses_exotic = top_n_horses_exotic
        for p in self.pools:
            if p not in self.POOLS_AVAILABLE:
                raise ValueError(f"Unknown pool {p}; must be one of "
                                 f"{self.POOLS_AVAILABLE}")
        self.exotic_odds_source = exotic_odds_source.lower()
        if self.exotic_odds_source not in self.EXOTIC_ODDS_SOURCES:
            raise ValueError(
                f"Unknown exotic_odds_source {exotic_odds_source!r}; "
                f"must be one of {self.EXOTIC_ODDS_SOURCES}")
        self.ledger_csv_path = ledger_csv_path

        # Two separate Henery exponents for place projection.
        #
        # Motivation: theta_place_diagnostic reveals that the stacker
        # ensemble (P_ens) needs a lower theta than P_pub to match
        # empirical place rates (0.76 vs 0.83 on HKJC 2018-2026). The
        # gap signals that P_ens is systematically overconfident on
        # longshot win probabilities, inflating pp_model and generating
        # phantom PLA EV.
        #
        # theta_pub_place  — PUBLIC side. Used for synthetic pub_odds
        #   (synthetic/hybrid mode) and the debug diagnostic header.
        #   Fit by theta_place_diagnostic against P_pub.
        #
        # theta_model_place — MODEL side. Used for pp_model in
        #   _size_pla. Fit against P_ens (--model_oof_csv=
        #   artifacts/wf_theta_input.csv). Lower than theta_pub_place
        #   directly corrects model longshot overconfidence.
        #
        # In REAL mode pub_odds come from the live market and don't
        # depend on theta, so only theta_model_place affects EV.
        # In SYNTHETIC mode the asymmetry is intentional — it represents
        # the model's known calibration gap vs the public market.
        #
        # Legacy: if only theta_place is supplied, both sides use it
        # (same behaviour as before, no phantom-EV risk).
        self.theta_pub_place   = (theta_place       if theta_place       is not None
                                  else self.theta_2)
        self.theta_model_place = (theta_model_place if theta_model_place is not None
                                  else self.theta_pub_place)
        self.theta_place = self.theta_pub_place   # backwards-compat alias

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------

    def _fetch_dividends(self, race_ids: list[str]) -> pd.DataFrame:
        """All dividends for the requested races, all pools.

        After migration 003 the table has a pool_code column with
        canonical 3-letter codes, so the join is a clean equality
        match. Rows whose pool isn't tracked have pool_code IS NULL
        and are filtered out by the WHERE clause.
        """
        if not race_ids:
            return pd.DataFrame(columns=['race_id', 'pool_code',
                                         'combination', 'dividend'])
        q = text("""
            SELECT race_id, pool_code, combination, dividend
            FROM race_dividends
            WHERE race_id = ANY(:race_ids)
              AND pool_code = ANY(:pools)
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={
                'race_ids': race_ids,
                'pools': list(self.pools),
            })
        if df.empty:
            return df
        df['dividend'] = df['dividend'].astype(float)
        return df

    def _fetch_real_exotic_odds(self,
                                race_ids: list[str],
                                ) -> dict[tuple[str, str], dict[str, float]]:
        """Pull real exotic odds from `live_odds_history` for the given races.

        Returns
        -------
        dict[(race_id, pool_canonical)] -> dict[combo_key -> odds]
            Where combo_key is the canonicalised combination ('1-3' for
            QIN, '1-3-5' for TRI, '5' for PLA single-horse). Pools we
            don't track and races without snapshots are absent from the
            outer dict; callers should treat absence as "no real odds
            available, fall back per the configured mode."

        Snapshot selection
        ------------------
        For each (race_id, pool_type, combination), we want the snapshot
        that best represents the price at STOP_SELL — that's what the
        live bot would have priced against. Three cases in your data:

        * Rows with phase='POST_STOP_SELL' (post-late-money settlement):
          ignore — those reflect dividend-time tote, not bet-time price.
        * Rows with phase='PRE_STOP_SELL': take the LATEST one (closest
          to STOP_SELL), since we want the gate to evaluate against the
          freshest pre-close price.
        * Rows with phase='UNKNOWN' (older Plan-C captures, no anchor):
          take the latest. Best available in the absence of phase tags.

        Combo-string canonicalisation: live_odds_history stores QIN
        combinations as '1-3' or '1,3', TRI as '1-3-5'. `_canonical_combo`
        normalises both to the dash form we use in the bet ledger and
        dividend lookup.

        Excluded odds: any row with odds <= 1.0 or odds >= 999 (the
        latter is HKJC's "no quote / sentinel" value; see your data
        sample for combos like '2-3' showing 999 odds — those are
        non-tradeable longshots that the public market hasn't priced).
        """
        if not race_ids:
            return {}
        # `live_odds_history.pool_type` already uses 3-letter codes
        # (WIN, PLA, QIN, QPL, TRI) — same convention as self.pools.
        # No mapping needed here, unlike race_dividends.
        q = text("""
            WITH ranked AS (
                SELECT
                    race_id, pool_type, combination, odds, phase,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY race_id, pool_type, combination
                        ORDER BY
                            CASE phase
                                WHEN 'PRE_STOP_SELL'  THEN 1
                                WHEN 'UNKNOWN'        THEN 2
                                WHEN 'POST_STOP_SELL' THEN 3
                                WHEN 'FINAL'          THEN 4
                                ELSE 5
                            END,
                            timestamp DESC
                    ) AS rn
                FROM live_odds_history
                WHERE race_id = ANY(:race_ids)
                  AND pool_type = ANY(:pools)
                  AND odds > 1.0
                  AND odds < 999
            )
            SELECT race_id, pool_type, combination, odds
            FROM ranked
            WHERE rn = 1
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={
                'race_ids': race_ids,
                'pools': list(self.pools),
            })
        if df.empty:
            return {}

        lookup: dict[tuple[str, str], dict[str, float]] = {}
        for _, row in df.iterrows():
            key = (str(row['race_id']), str(row['pool_type']))
            combo = _canonical_combo(row['combination'])
            try:
                odds = float(row['odds'])
            except (TypeError, ValueError):
                continue
            if odds <= 1.0:
                continue
            lookup.setdefault(key, {})[combo] = odds
        return lookup

    # ------------------------------------------------------------------
    # Per-race sizing for each pool
    # ------------------------------------------------------------------

    def _size_pla(self, p_arr: np.ndarray, p_public: np.ndarray,
                  horse_nos: list[str],
                  win_drift_df: pd.DataFrame,
                  live_pla_odds: dict[str, float] | None = None,
                  ) -> pd.DataFrame:
        """PLA sizing.

        Odds source per `self.exotic_odds_source`:
        * 'real':      use live_pla_odds[horse_no] if present, else SKIP.
        * 'synthetic': always use synthetic θ-symmetric public odds.
        * 'hybrid':    use live_pla_odds when present, synthetic otherwise.

        Critical correction (synthetic path): public PLA odds are derived
        from p_public with the SAME thetas used for pp_model. Asymmetric
        thetas would generate phantom EV — see `_public_pla_odds`.

        θ_place note: `pp_model` and the synthetic `pub_odds` both use
        `self.theta_place` (defaults to self.theta_2 for back-compat).
        Set theta_place via the constructor if a place-specific MLE
        produced a better-calibrated θ — see theta_place_diagnostic.
        """
        rows = []
        live_pla_odds = live_pla_odds or {}
        for i, hn in enumerate(horse_nos):
            # MODEL side: use theta_model_place, which was fit against
            # P_ens to compensate for model longshot overconfidence.
            # Lower than theta_pub_place → compresses pp_model downward
            # on longshots, correcting inflated EV signals.
            pp_model = p_place(p_arr, i,
                               self.theta_model_place, self.theta_model_place)
            if pp_model <= 0:
                continue

            # Resolve odds and source per mode
            if hn in live_pla_odds:
                pub_odds = float(live_pla_odds[hn])
                odds_source = 'real'
            elif self.exotic_odds_source == 'real':
                continue
            else:
                # SYNTHETIC side: use theta_pub_place (fit against P_pub),
                # symmetric with itself. Using theta_pub_place here (not
                # theta_model_place) is intentional — the public market
                # doesn't have the model's overconfidence, so we project
                # synthetic pub odds from the pub-calibrated theta.
                pub_odds = _public_pla_odds(p_public, i,
                                            self.theta_pub_place,
                                            self.theta_pub_place)
                odds_source = 'synthetic'

            if pub_odds <= 1.0:
                continue

            # Drift override per-horse if forecaster present
            override = None
            if win_drift_df is not None and not win_drift_df.empty:
                single = project_drift_to_exotic(win_drift_df, [int(hn)])
                override = {'PLA': DriftStats(**single)}

            # Per-horse shrinkage: when self.shrinkage is a dict
            # (stratified fit), pick the band by this horse's p_pub.
            # When it's a scalar (legacy), lookup_shrinkage returns it
            # unchanged — same behaviour as before.
            from hkjc_engine.models.betting_policy import lookup_shrinkage
            eff_shrinkage = lookup_shrinkage(float(p_public[i]), self.shrinkage)

            stake, ev_eff = self._size_one(pp_model, pub_odds,
                                           pool='PLA', drift=override,
                                           shrinkage=eff_shrinkage)
            rows.append({
                'pool': 'PLA',
                'combo': hn,
                'odds': pub_odds,
                'p_model': pp_model,
                'ev': ev_eff,
                'stake': stake,
                'odds_source': odds_source,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = self._apply_pool_cap(df, 'PLA')
        return df

    def _size_exotic(self, p_arr: np.ndarray, p_public: np.ndarray,
                     horse_nos: list[str],
                     pool: str, calc_func, comb_len: int,
                     win_drift_df: pd.DataFrame,
                     live_combo_odds: dict[str, float] | None = None,
                     ) -> pd.DataFrame:
        """Generic QIN/QPL/TRI sizing.

        Odds source per `self.exotic_odds_source`:
        * 'real':      use live_combo_odds[combo_key] if present, else SKIP.
        * 'synthetic': always use synthetic θ-symmetric public odds.
        * 'hybrid':    use live_combo_odds when present, synthetic otherwise.

        Restricts the search to the top-N horses by P_model to keep the
        combinatoric space tractable (matches live bot constant
        TOP_N_HORSES_EXO). Synthetic public odds (when used) are derived
        from p_public with the SAME thetas as p_model.

        Per-race top-K cap (TOP_K_PER_RACE):
        After EV-gating, only the top-K bets by EV are retained per race.
        This addresses the combinatoric amplification problem: when the
        model likes 3 horses strongly, all combos containing any two of
        them can pass the EV gate simultaneously, generating 15-27 TRI
        bets from a single race. The top-K cap forces the policy to be
        selective within each race, not just across races.

        Pool-specific limits (TOP_K_PER_RACE):
          QIN: 5  (C(8,2)=28 candidate combos)
          QPL: 5
          TRI: 4  (C(8,3)=56 candidate combos — largest combinatoric space)
        """
        # Per-pool top-K bets per race after EV gating.
        TOP_K_PER_RACE = {'QIN': 5, 'QPL': 5, 'TRI': 4}
        top_k = TOP_K_PER_RACE.get(pool.upper(), 5)

        n = len(p_arr)
        top_indices = list(np.argsort(-p_arr)[:min(self.top_n_horses_exotic, n)])
        rows = []
        live_combo_odds = live_combo_odds or {}

        for combo_idx in itertools.combinations(top_indices, comb_len):
            h_nums = sorted(int(horse_nos[i]) for i in combo_idx)
            combo_key = '-'.join(str(x) for x in h_nums)
            p_model = calc_func(p_arr, *combo_idx,
                                self.theta_2, self.theta_3)
            if p_model <= 0:
                continue

            # Resolve odds and source per mode
            if combo_key in live_combo_odds:
                pub_odds = float(live_combo_odds[combo_key])
                odds_source = 'real'
            elif self.exotic_odds_source == 'real':
                continue
            else:
                pub_odds = _public_combo_odds(p_public, combo_idx, calc_func,
                                              pool, self.theta_2, self.theta_3)
                odds_source = 'synthetic'

            if pub_odds <= 1.0:
                continue

            override = None
            if win_drift_df is not None and not win_drift_df.empty:
                combo_drift = project_drift_to_exotic(win_drift_df, h_nums)
                override = {pool: DriftStats(**combo_drift)}

            from hkjc_engine.models.betting_policy import lookup_shrinkage
            min_p_pub = float(min(p_public[idx] for idx in combo_idx))
            eff_shrinkage = lookup_shrinkage(min_p_pub, self.shrinkage)

            stake, ev_eff = self._size_one(p_model, pub_odds,
                                           pool=pool, drift=override,
                                           shrinkage=eff_shrinkage)
            rows.append({
                'pool': pool,
                'combo': combo_key,
                'odds': pub_odds,
                'p_model': p_model,
                'ev': ev_eff,
                'stake': stake,
                'odds_source': odds_source,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Apply per-race top-K cap: keep only the highest-EV qualifying bets.
        # Only bets with stake > 0 (i.e. passed EV gate) count toward the cap.
        # Non-qualifying bets (stake=0) are dropped entirely here — the
        # cap operates on qualified bets only.
        qualifying = df[df['stake'] > 0]
        if len(qualifying) > top_k:
            keep_combos = (qualifying.nlargest(top_k, 'ev')['combo'].values)
            df = df[df['combo'].isin(keep_combos)]

        df = self._apply_pool_cap(df, pool)
        return df

    def _size_one(self, p_raw: float, odds: float, pool: str,
                  drift, shrinkage: float | None = None
                  ) -> tuple[float, float]:
        """One-bet drift-aware sizing. Returns (stake, ev_eff).

        `shrinkage` parameter: if supplied, used directly (caller has
        already resolved the per-horse value via lookup_shrinkage). If
        None, falls back to self.shrinkage — which works for legacy
        scalar shrinkage but won't apply per-band stratification, so
        callers in the stratified path must pass a resolved scalar.
        """
        if pd.isna(odds) or odds <= 1.0 or p_raw <= 0:
            return 0.0, 0.0
        eff_shrinkage = shrinkage if shrinkage is not None else self.shrinkage
        # If self.shrinkage is a dict and caller didn't resolve, default
        # to the global fallback rather than crashing.
        if isinstance(eff_shrinkage, dict):
            from hkjc_engine.models.betting_policy import lookup_shrinkage
            eff_shrinkage = lookup_shrinkage(0.10, eff_shrinkage)  # mid-band default
        ev_eff = shrunk_ev(p_raw, odds, eff_shrinkage,
                           pool=pool, drift_override=drift)
        hurdle = get_ev_hurdle(odds, base=0.02, longshot_buffer=0.015,
                               longshot_threshold=15.0,
                               pool=pool, drift_override=drift)
        if ev_eff < hurdle:
            return 0.0, ev_eff
        stake = fractional_kelly_stake(
            p_raw, odds, self.bankroll,
            kelly_fraction=0.25, shrinkage=eff_shrinkage,
            per_bet_cap=0.02, min_stake_abs=10.0,
            pool=pool, drift_override=drift,
        )
        return stake, ev_eff

    def _size_win_pool(self, df: pd.DataFrame,
                       overrides: dict[int, DriftStats]
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Override of parent class _size_win_pool with stratified
        shrinkage support.

        Behaviour
        ---------
        * If self.shrinkage is a SCALAR: defers to the parent class
          implementation byte-for-byte. WIN ledger identical to legacy.
        * If self.shrinkage is a DICT: resolves per-row shrinkage from
          each horse's normalised P_pub via lookup_shrinkage, then
          calls qualify_and_size with a per-row shrinkage ARRAY.
          NumPy broadcasts the array shrinkage through the existing
          vectorised pipeline (line: np.minimum(shrinkage * p_raw,
          1 - 1e-9)) — no signature change to qualify_and_size needed.

        The drift-override path (forecaster active) is also handled,
        falling through to a per-row loop with the resolved scalar.
        """
        if not isinstance(self.shrinkage, dict):
            return super()._size_win_pool(df, overrides)

        # Stratified path. Compute per-row p_pub from stop_sell_odds
        # (race-normalised) so band lookup matches the OOF fit basis.
        from hkjc_engine.models.betting_policy import lookup_shrinkage
        p_pub_raw = 1.0 / df['stop_sell_odds'].values
        p_pub = p_pub_raw / p_pub_raw.sum()
        shr_arr = lookup_shrinkage(p_pub, self.shrinkage)

        if not overrides:
            # Vectorised path with per-row shrinkage array. NumPy
            # broadcasts shr_arr * p_raw element-wise inside
            # qualify_and_size. fractional_kelly_stake is called
            # per-row internally with the shrinkage scalar — actually,
            # qualify_and_size doesn't call fractional_kelly_stake, it
            # inlines the math. Verified above (betting_policy.py L344).
            return qualify_and_size(
                p_raw=df['P_model'].values,
                odds=df['stop_sell_odds'].values,
                bankroll=self.bankroll,
                kelly_fraction=0.35,
                shrinkage=shr_arr,
                base_hurdle=0.005,
                longshot_buffer=0.005,
                longshot_threshold=25.0,
                per_bet_cap=0.05,
                race_cap=0.10,
                top_k=10,
                pool='WIN',
            )

        # Per-row drift overrides path. Same loop as parent but using
        # row-resolved shrinkage scalar at each step.
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
            shr_i = float(shr_arr[i])

            ev = (min(shr_i * p, 1 - 1e-9) * odds * ds.median_R - 1.0)
            evs[i] = ev
            hurdle = get_ev_hurdle(odds, base=0.005, longshot_buffer=0.005,
                                   longshot_threshold=25.0,
                                   pool='WIN', drift_override=override_dict)
            if ev < hurdle or odds > 25.0:
                continue
            stakes[i] = fractional_kelly_stake(
                p_raw=p, odds=odds, bankroll=self.bankroll,
                kelly_fraction=0.35,
                shrinkage=shr_i,
                per_bet_cap=0.05, min_stake_abs=10.0,
                pool='WIN', drift_override=override_dict,
            )

        # Race-level cap, EV-ranked top-k filter — match parent class
        keep_mask = stakes > 0
        if not keep_mask.any():
            return np.array([], dtype=int), np.array([], dtype=float)
        keep_idx = np.where(keep_mask)[0]
        if len(keep_idx) > 10:
            keep_idx = keep_idx[np.argsort(-evs[keep_idx])[:10]]
        kept_stakes = stakes[keep_idx]
        kept_stakes = cap_simultaneous_stakes(kept_stakes, self.bankroll,
                                               race_cap=0.10, pool='WIN')
        final_keep = kept_stakes >= 10.0
        return keep_idx[final_keep], kept_stakes[final_keep]

    def _apply_pool_cap(self, df: pd.DataFrame, pool: str) -> pd.DataFrame:
        """Apply pool-conditional race cap. Filters out sub-min stakes."""
        qual = df['stake'] > 0
        if not qual.any():
            return df
        df = df.copy()
        df.loc[qual, 'stake'] = cap_simultaneous_stakes(
            df.loc[qual, 'stake'].values, self.bankroll,
            race_cap=0.04, pool=pool,
        )
        df.loc[df['stake'] < 10.0, 'stake'] = 0.0
        return df

    # ------------------------------------------------------------------
    # Per-race orchestrator (overrides parent run_backtest)
    # ------------------------------------------------------------------

    def run_backtest(self, start_date: str = '2024-01-01',
                     end_date: str = '2026-01-01',
                     ) -> tuple[list[dict], float]:
        log.info("Drift-aware MULTI-POOL backtest (%s)...",
                 ', '.join(self.pools))
        self._debug_printed_pla = False  # one diagnostic per invocation

        # Single SQL pulls everything we need. Note: NO LEFT JOIN to
        # race_dividends here — we fetch all dividends in a separate
        # query so we can index them once and look up across all pools.
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
                CASE WHEN cc.career_run_number = 1 THEN 1 ELSE 0 END AS is_maiden
            FROM races r
            JOIN race_entries e ON r.race_id = e.race_id
            JOIN CareerCounts cc ON e.race_id = cc.race_id
                                AND e.horse_code = cc.horse_code
            WHERE r.race_date >= :start_date AND r.race_date < :end_date
              AND e.win_odds IS NOT NULL
              AND e.finish_position IS NOT NULL
            ORDER BY r.race_date ASC, r.race_no ASC
        """)
        with self.engine.connect() as conn:
            raw = pd.read_sql(query, conn,
                              params={'start_date': start_date,
                                      'end_date': end_date})
        if raw.empty:
            log.warning("No race rows in window %s -> %s", start_date, end_date)
            return [], self.bankroll

        race_ids = raw['race_id'].astype(str).unique().tolist()
        dividends_df = self._fetch_dividends(race_ids)
        dividend_lookup = _build_dividend_lookup(dividends_df)

        # Real exotic odds — only fetched if any exotic pool is requested
        # AND we're not in pure-synthetic mode. The result is a dict
        # keyed on (race_id, pool) -> {combo_key: odds}.
        exotic_pools = [p for p in self.pools if p != 'WIN']
        if exotic_pools and self.exotic_odds_source != 'synthetic':
            live_exotic_odds = self._fetch_real_exotic_odds(race_ids)
        else:
            live_exotic_odds = {}

        # Log mode + per-pool live-odds coverage so the operator sees
        # immediately whether real odds will gate exotic bets, and on
        # what fraction of races.
        log.info("Exotic odds source: %s", self.exotic_odds_source)
        if exotic_pools and self.exotic_odds_source != 'synthetic':
            live_coverage_per_pool: dict[str, int] = {p: 0 for p in exotic_pools}
            for (rid, pool) in live_exotic_odds.keys():
                if pool in live_coverage_per_pool:
                    live_coverage_per_pool[pool] += 1
            log.info("Live exotic odds coverage:")
            for p in exotic_pools:
                n = live_coverage_per_pool.get(p, 0)
                pct = 100.0 * n / len(race_ids) if race_ids else 0.0
                if self.exotic_odds_source == 'real' and n == 0:
                    log.warning("  %s: %d / %d races (0.0%%) — REAL mode "
                                "and no live odds; ZERO %s bets will fire.",
                                p, n, len(race_ids), p)
                elif pct < 50.0 and self.exotic_odds_source == 'real':
                    log.warning("  %s: %d / %d races (%.1f%%) — sparse "
                                "real-odds coverage in REAL mode; expect "
                                "low bet count.",
                                p, n, len(race_ids), pct)
                else:
                    log.info("  %s: %d / %d races (%.1f%%)",
                             p, n, len(race_ids), pct)

        # Pool-by-pool dividend coverage summary — surfaces missing scraper
        # data before it silently corrupts the backtest. If a pool has zero
        # dividend rows, we WILL skip every bet on that pool below.
        if dividends_df.empty:
            log.warning("race_dividends returned ZERO rows for %d races in "
                        "[%s, %s]. Backtest will produce no settled bets. "
                        "Run `python -m hkjc_engine.diagnostics.dividend_coverage "
                        "%s %s` to diagnose.",
                        len(race_ids), start_date, end_date,
                        start_date, end_date)
        else:
            pool_coverage = dividends_df.groupby('pool_code')['race_id'].nunique()
            log.info("Dividend coverage by pool:")
            for pool in self.pools:
                n = int(pool_coverage.get(pool, 0))
                pct = 100.0 * n / len(race_ids) if race_ids else 0.0
                if pct < 50.0:
                    log.warning("  %s: %d / %d races (%.1f%%) — "
                                "LOW COVERAGE, settlement unreliable.",
                                pool, n, len(race_ids), pct)
                else:
                    log.info("  %s: %d / %d races (%.1f%%)",
                             pool, n, len(race_ids), pct)

        # Set of (race_id, canonical_pool) tuples that have at least one
        # dividend row. We require pool-level dividend presence before
        # settling a bet — otherwise a race with no PLA dividend data
        # would mark every PLA bet as a loser.
        settleable_pool_keys: set[tuple[str, str]] = set()
        if not dividends_df.empty:
            for (rid, pool), _ in dividends_df.groupby(['race_id', 'pool_code']):
                settleable_pool_keys.add((str(rid), str(pool)))

        # Build authoritative PLA-placer set per race directly from
        # race_dividends — this captures HKJC's field-size rules
        # (no PLA in 3-runner fields, top-2 in 4-runner fields, top-3
        # otherwise) without us having to reimplement them.
        pla_combos_per_race: dict[str, set] = {}
        if not dividends_df.empty:
            pla_rows = dividends_df[dividends_df['pool_code'] == 'PLA']
            for race_id, g in pla_rows.groupby('race_id'):
                pla_combos_per_race[str(race_id)] = {
                    _canonical_combo(c) for c in g['combination']
                }

        winning_combos = _winning_combos(
            raw[['race_id', 'horse_no', 'finish_position']].copy(),
            pla_combos_per_race=pla_combos_per_race,
        )

        log.info("Fetched %d races with dividends (%d rows across %d pools).",
                 dividends_df['race_id'].nunique() if not dividends_df.empty else 0,
                 len(dividends_df),
                 dividends_df['pool_code'].nunique() if not dividends_df.empty else 0)

        bet_ledger: list[dict] = []
        bets_per_pool: dict[str, int] = {p: 0 for p in self.pools}
        wins_per_pool: dict[str, int] = {p: 0 for p in self.pools}
        staked_per_pool: dict[str, float] = {p: 0.0 for p in self.pools}
        profit_per_pool: dict[str, float] = {p: 0.0 for p in self.pools}
        skipped_no_dividend: dict[str, int] = {p: 0 for p in self.pools}

        for race_id, race_df in raw.groupby('race_id'):
            df = self.factory.engineer_features(race_df.copy())
            df = self._attach_stop_sell(df)
            if df.empty or len(df) < 2:
                continue

            # Standard interactions (parity with parent backtester)
            df['is_class_drop'] = df['is_class_drop'].astype(float).fillna(0.0)
            df['is_class_rise'] = df['is_class_rise'].astype(float).fillna(0.0)
            df['is_maiden']     = df['is_maiden'].astype(float)
            df['draw_x_early_pace']      = df['draw'] * df['relative_early_pace']
            df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
            df['class_drop_x_ts']        = df['is_class_drop'] * df['ts_advantage']
            df['class_rise_x_ts']        = df['is_class_rise'] * df['ts_advantage']

            # Predict P_model (same path as parent class)
            dmat_a = xgb.DMatrix(df[self.FEATURES_A])
            df['base_margin'] = calculate_base_margin(df['stop_sell_odds'])
            dmat_a.set_base_margin(df['base_margin'])
            df['raw_a'] = self.model_a.predict(dmat_a)
            df['P_a_softmax'] = _softmax(df['raw_a'].values)
            df['P_a_cal'] = self.calibrator_a.predict_proba(df['P_a_softmax'].values)[:, 1]

            dmat_b = xgb.DMatrix(df[self.FEATURES_B])
            df['raw_b'] = self.model_b.predict(dmat_b)
            df['P_b_softmax'] = _softmax(df['raw_b'].values)
            df['P_b_cal'] = self.calibrator_b.predict_proba(df['P_b_softmax'].values)[:, 1]

            df['P_pub_raw'] = 1.0 / df['stop_sell_odds']
            df['P_pub'] = df['P_pub_raw'] / df['P_pub_raw'].sum()

            P = np.column_stack([df['P_a_cal'].values,
                                 df['P_b_cal'].values,
                                 df['P_pub'].values])
            df['P_model'] = self.stacker.predict(P, df['race_id'].values)
            p_arr = df['P_model'].to_numpy(dtype=float)
            p_public = df['P_pub'].to_numpy(dtype=float)
            horse_nos = df['horse_no'].astype(str).tolist()

            # ONE-SHOT DEBUG: print PLA EV breakdown for the first race only.
            # Confirms the EV-gate math is producing sensible numbers, and
            # pinpoints whether bet decisions are coming from real model
            # edge vs market or some other source.
            if not getattr(self, '_debug_printed_pla', False):
                self._debug_printed_pla = True
                from hkjc_engine.models.betting_policy import (
                    shrunk_ev, get_ev_hurdle, lookup_shrinkage,
                )
                # Header: report shrinkage compactly. For dict, show
                # the band->value mapping; for scalar, show the number.
                if isinstance(self.shrinkage, dict):
                    shr_str = "stratified[" + ", ".join(
                        f"{k}:{v:.3f}" for k, v in self.shrinkage.items()) + "]"
                else:
                    shr_str = f"{self.shrinkage:.4f}"
                log.info("DEBUG PLA EV breakdown for race %s "
                         "(theta_model_place=%.4f, theta_pub_place=%.4f, "
                         "shrinkage=%s):",
                         race_id, self.theta_model_place,
                         self.theta_pub_place, shr_str)
                log.info(f"{'horse':>6} {'P_pub':>7} {'P_model':>8} "
                         f"{'pp_pub':>8} {'pp_model':>9} "
                         f"{'pub_odds':>9} {'shr':>6} {'EV':>7} "
                         f"{'hurdle':>7} {'qual':>5}")
                for i, hn in enumerate(horse_nos[:8]):
                    pp_pub   = p_place(p_public, i,
                                       self.theta_pub_place,
                                       self.theta_pub_place)
                    pp_model = p_place(p_arr, i,
                                       self.theta_model_place,
                                       self.theta_model_place)
                    if pp_pub <= 0:
                        continue
                    pub_odds = (1 - 0.175) / pp_pub
                    # Resolve per-horse shrinkage so the debug EV matches
                    # what the policy actually uses for this horse.
                    eff_shr = lookup_shrinkage(float(p_public[i]),
                                                self.shrinkage)
                    ev = shrunk_ev(pp_model, pub_odds, eff_shr, pool='PLA')
                    hurdle = get_ev_hurdle(pub_odds, base=0.02,
                                           longshot_buffer=0.015,
                                           longshot_threshold=15.0,
                                           pool='PLA')
                    qual = 'YES' if ev >= hurdle else 'no'
                    log.info(f"  H{hn:<4} {p_public[i]:>7.3f} {p_arr[i]:>8.3f} "
                             f"{pp_pub:>8.3f} {pp_model:>9.3f} "
                             f"{pub_odds:>9.3f} {eff_shr:>6.3f} "
                             f"{ev:>+7.2%} {hurdle:>+7.2%} {qual:>5}")

            # Drift forecaster, if loaded
            win_drift_df = self._drift_features_for_race(race_id) \
                if self.drift_forecaster else pd.DataFrame()
            if not win_drift_df.empty and self.drift_forecaster:
                win_drift_df = self.drift_forecaster.predict_win(win_drift_df)

            # Per-pool sizing
            pool_dfs: dict[str, pd.DataFrame] = {}
            if 'WIN' in self.pools:
                # Re-use parent class WIN sizing path so the WIN ledger
                # is identical to what XGBEnsembleBacktester produces.
                drift_overrides_per_idx = {}
                if not win_drift_df.empty:
                    preds = win_drift_df.set_index('horse_no')
                    for ridx, hn in enumerate(horse_nos):
                        if hn in preds.index:
                            p = preds.loc[hn]
                            drift_overrides_per_idx[ridx] = DriftStats(
                                mu_R=float(p['mu_R']),
                                sigma_R=float(p['sigma_R']),
                                median_R=float(p['median_R']))
                keep_idx, stakes = self._size_win_pool(df, drift_overrides_per_idx)
                # Materialise to a pool df with the same schema as exotics.
                # WIN always uses real STOP_SELL odds, so source is 'real'.
                rows = []
                for ridx, st in zip(keep_idx, stakes):
                    rows.append({
                        'pool': 'WIN', 'combo': horse_nos[ridx],
                        'odds': float(df.iloc[ridx]['stop_sell_odds']),
                        'p_model': float(p_arr[ridx]),
                        'ev': float(p_arr[ridx] * df.iloc[ridx]['stop_sell_odds'] - 1.0),
                        'stake': float(st),
                        'odds_source': 'real',
                    })
                pool_dfs['WIN'] = pd.DataFrame(rows) if rows else \
                                  pd.DataFrame(columns=['pool','combo','odds',
                                                         'p_model','ev','stake',
                                                         'odds_source'])

            # Per-race live-odds slices for each exotic pool
            race_id_str = str(race_id)
            live_pla = live_exotic_odds.get((race_id_str, 'PLA'), {})
            live_qin = live_exotic_odds.get((race_id_str, 'QIN'), {})
            live_qpl = live_exotic_odds.get((race_id_str, 'QPL'), {})
            live_tri = live_exotic_odds.get((race_id_str, 'TRI'), {})

            if 'PLA' in self.pools:
                pool_dfs['PLA'] = self._size_pla(
                    p_arr, p_public, horse_nos, win_drift_df,
                    live_pla_odds=live_pla)
            if 'QIN' in self.pools:
                pool_dfs['QIN'] = self._size_exotic(
                    p_arr, p_public, horse_nos, 'QIN', p_quinella, 2,
                    win_drift_df, live_combo_odds=live_qin)
            if 'QPL' in self.pools:
                pool_dfs['QPL'] = self._size_exotic(
                    p_arr, p_public, horse_nos, 'QPL', p_quinella_place, 2,
                    win_drift_df, live_combo_odds=live_qpl)
            if 'TRI' in self.pools:
                pool_dfs['TRI'] = self._size_exotic(
                    p_arr, p_public, horse_nos, 'TRI', p_trio, 3,
                    win_drift_df, live_combo_odds=live_tri)

            # Master cross-pool cap (parity with live bot)
            total_stake = sum(d.loc[d['stake'] > 0, 'stake'].sum()
                              for d in pool_dfs.values() if not d.empty)
            cap_dollars = self.bankroll * 0.06
            if total_stake > cap_dollars > 0:
                shrink = cap_dollars / total_stake
                for d in pool_dfs.values():
                    if d.empty:
                        continue
                    d.loc[d['stake'] > 0, 'stake'] *= shrink
                    d.loc[d['stake'] < 10.0, 'stake'] = 0.0

            # Settle each bet against race_dividends + winning_combos
            wc = winning_combos.get(race_id, {})
            for pool, dfp in pool_dfs.items():
                if dfp.empty:
                    continue
                race_id_str = str(race_id)
                # Skip the entire pool for this race if no dividend data —
                # otherwise we'd mark every bet as a loser, badly biasing
                # the per-pool ROI down to -100%.
                if (race_id_str, pool) not in settleable_pool_keys:
                    n_skipped = int((dfp['stake'] >= 10.0).sum())
                    if n_skipped > 0:
                        skipped_no_dividend[pool] += n_skipped
                    continue

                for _, bet in dfp[dfp['stake'] >= 10.0].iterrows():
                    stake = float(bet['stake'])
                    combo = _canonical_combo(bet['combo'])
                    is_win = self._is_winning_bet(pool, combo, wc)

                    # PnL: HKJC dividend convention is per $10
                    if is_win:
                        div = self._lookup_dividend(
                            dividend_lookup, race_id, pool, combo, wc)
                        if div is None:
                            # Race won the bet but specific combo's
                            # dividend missing — skip the bet rather
                            # than charge stake. (Rare; possible on
                            # dead-heats with imperfect scrape.)
                            log.debug("R%s %s combo=%s won but no dividend "
                                      "row found; skipping settlement.",
                                      race_id, pool, combo)
                            continue
                        payout = (stake / 10.0) * div
                        profit = payout - stake
                        wins_per_pool[pool] += 1
                    else:
                        profit = -stake

                    self.bankroll += profit
                    bets_per_pool[pool] += 1
                    staked_per_pool[pool] += stake
                    profit_per_pool[pool] += profit

                    bet_ledger.append({
                        'race_id':     race_id,
                        'pool':        pool,
                        'combo':       combo,
                        'odds':        float(bet['odds']),
                        'p_model':     float(bet['p_model']),
                        'ev':          float(bet['ev']),
                        'stake':       stake,
                        'is_win':      bool(is_win),
                        'profit':      float(profit),
                        'odds_source': bet.get('odds_source', 'real'),
                    })

        # ---- summary ----
        self._log_summary(bets_per_pool, wins_per_pool,
                          staked_per_pool, profit_per_pool,
                          skipped_no_dividend,
                          bet_ledger=bet_ledger)

        # ---- per-pool distribution analysis ----
        # Surfaces concentration of bets across odds and probability bands,
        # so the operator can see whether 321 TRI bets are spread thinly
        # across many races or piled onto a handful of dislocated markets.
        # Same for whether PLA losses cluster on longshots vs favorites.
        if bet_ledger:
            self._log_distribution_summary(bet_ledger)

        # ---- ledger CSV export ----
        if self.ledger_csv_path and bet_ledger:
            try:
                ledger_df = pd.DataFrame(bet_ledger)
                ledger_df.to_csv(self.ledger_csv_path, index=False)
                log.info("Bet ledger written to %s (%d rows).",
                         self.ledger_csv_path, len(ledger_df))
            except Exception as e:
                log.warning("Failed to write ledger CSV: %s", e)

        return bet_ledger, self.bankroll

    # ------------------------------------------------------------------
    # Settlement primitives
    # ------------------------------------------------------------------

    def _is_winning_bet(self, pool: str, combo: str, wc: dict) -> bool:
        """Pool-aware winner check."""
        if pool not in wc:
            return False
        winners = wc[pool]
        if pool in ('WIN', 'QIN', 'TRI'):
            return combo == winners
        if pool == 'PLA':
            # combo is a single horse_no string; PLA winners is a set
            return combo in winners
        if pool == 'QPL':
            return combo in winners
        return False

    def _lookup_dividend(self, lookup: dict, race_id: str,
                         pool: str, combo: str, wc: dict) -> float | None:
        """Find the dividend for this winning bet.

        For QPL, race_dividends has one row per winning pair, all
        keyed on the same combo. WIN/QIN/TRI have one row per race.
        For PLA we look up the specific horse_no.
        """
        key = (str(race_id), pool, combo)
        return lookup.get(key)

    def _log_summary(self, bets_per_pool: dict, wins_per_pool: dict,
                     staked_per_pool: dict, profit_per_pool: dict,
                     skipped_no_dividend: dict | None = None,
                     bet_ledger: list[dict] | None = None):
        skipped_no_dividend = skipped_no_dividend or {}
        log.info("\n" + "=" * 70)
        log.info("   DRIFT-AWARE MULTI-POOL BACKTEST RESULTS")
        log.info("=" * 70)
        log.info(f"{'Pool':<6} {'Bets':>6} {'Wins':>6} {'WinRate':>8} "
                 f"{'Staked':>12} {'Profit':>12} {'ROI':>8}  {'Skip*':>6}")
        total_bets = total_wins = 0
        total_staked = total_profit = total_skipped = 0.0
        for p in self.pools:
            n = bets_per_pool.get(p, 0)
            w = wins_per_pool.get(p, 0)
            s = staked_per_pool.get(p, 0.0)
            pf = profit_per_pool.get(p, 0.0)
            sk = skipped_no_dividend.get(p, 0)
            wr = (w / n * 100) if n > 0 else 0.0
            roi = (pf / s * 100) if s > 0 else 0.0
            log.info(f"{p:<6} {n:>6} {w:>6} {wr:>7.2f}% "
                     f"${s:>11,.0f} ${pf:>+11,.0f} {roi:>+7.2f}% "
                     f"{sk:>6}")
            total_bets += n
            total_wins += w
            total_staked += s
            total_profit += pf
            total_skipped += sk
        log.info("-" * 70)
        total_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        log.info(f"{'TOTAL':<6} {total_bets:>6} {total_wins:>6} {'':>8} "
                 f"${total_staked:>11,.0f} ${total_profit:>+11,.0f} "
                 f"{total_roi:>+7.2f}% {int(total_skipped):>6}")
        if total_skipped > 0:
            log.info("\n* Skip = bets dropped because the pool had no "
                     "dividend data for that race (data quality issue, "
                     "not a policy decision). Run dividend_coverage "
                     "diagnostic if Skip > 0 on any pool.")

        # Secondary table: per-(pool, odds_source) breakdown. Only shown
        # if hybrid mode produced both sources, OR if synthetic mode
        # actually fired any exotic bets. In real-only mode every row is
        # 'real' and the breakdown adds no information.
        if bet_ledger:
            sources_seen = {b.get('odds_source', 'real') for b in bet_ledger}
            if len(sources_seen) > 1:
                log.info("\n" + "-" * 70)
                log.info("   BREAKDOWN BY ODDS SOURCE")
                log.info("-" * 70)
                log.info(f"{'Pool':<6} {'Source':<10} {'Bets':>6} {'Wins':>6} "
                         f"{'WinRate':>8} {'Staked':>12} {'Profit':>12} {'ROI':>8}")
                from collections import defaultdict
                acc: dict = defaultdict(lambda: {'n': 0, 'w': 0,
                                                  's': 0.0, 'pf': 0.0})
                for b in bet_ledger:
                    key = (b['pool'], b.get('odds_source', 'real'))
                    acc[key]['n']  += 1
                    acc[key]['w']  += int(b['is_win'])
                    acc[key]['s']  += float(b['stake'])
                    acc[key]['pf'] += float(b['profit'])
                for p in self.pools:
                    for src in ('real', 'synthetic'):
                        if (p, src) not in acc:
                            continue
                        a = acc[(p, src)]
                        wr = (a['w'] / a['n'] * 100) if a['n'] > 0 else 0.0
                        roi = (a['pf'] / a['s'] * 100) if a['s'] > 0 else 0.0
                        log.info(f"{p:<6} {src:<10} {a['n']:>6} {a['w']:>6} "
                                 f"{wr:>7.2f}% ${a['s']:>11,.0f} "
                                 f"${a['pf']:>+11,.0f} {roi:>+7.2f}%")

        log.info(f"\nEnding Bankroll: ${self.bankroll:,.2f}")
        log.info("=" * 70)

    # ------------------------------------------------------------------
    # Distribution diagnostics (called after _log_summary)
    # ------------------------------------------------------------------

    def _log_distribution_summary(self, bet_ledger: list[dict]) -> None:
        """Two diagnostics per pool:

        1. ODDS-BAND breakdown: bets bucketed by `odds`, showing whether
           a pool's losses cluster on shorts vs longs. Answers the
           question 'is the policy reading favorites correctly but
           bleeding on longshots?' or vice versa.

        2. RACE CONCENTRATION: how many races contributed how many
           bets to this pool. Answers 'are 321 TRI bets spread across
           80 races (~4 each) or piled onto 5 races with wild
           dislocations and silence everywhere else?'

        These are diagnostic only — they don't change any decisions.
        Their job is to make the operator's intuition catch up with
        what the policy is actually doing in the market.
        """
        if not bet_ledger:
            return

        # Pool-aware odds bands. Edges chosen to roughly equalise base
        # rates: WIN/PLA shortish-favorites are ≤3, mid 3-8, longs ≥8;
        # exotics span much wider so we extend the upper bands.
        bands_by_pool = {
            'WIN': [(1.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 25.0)],
            'PLA': [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 25.0)],
            'QIN': [(1.0, 10.0), (10.0, 30.0), (30.0, 80.0), (80.0, 300.0)],
            'QPL': [(1.0,  5.0), ( 5.0, 15.0), (15.0, 40.0), (40.0, 150.0)],
            'TRI': [(1.0, 30.0), (30.0,100.0), (100.0,300.0), (300.0,2000.0)],
        }

        from collections import defaultdict

        # Group bets by pool first so we can iterate in self.pools order
        per_pool: dict[str, list[dict]] = defaultdict(list)
        for b in bet_ledger:
            per_pool[b['pool']].append(b)

        log.info("\n" + "=" * 70)
        log.info("   PER-POOL DISTRIBUTION DIAGNOSTICS")
        log.info("=" * 70)

        for pool in self.pools:
            bets = per_pool.get(pool, [])
            if not bets:
                continue

            # ---- Odds-band breakdown ----
            bands = bands_by_pool.get(pool, [(1.0, 5.0), (5.0, 20.0),
                                              (20.0, 100.0), (100.0, 5000.0)])
            log.info("")
            log.info(f"{pool} — {len(bets)} bets — by ODDS band")
            log.info(f"  {'Range':<14} {'Bets':>5} {'Wins':>5} "
                     f"{'WinRate':>8} {'Staked':>10} {'Profit':>10} {'ROI':>8}")
            uncategorised: list[dict] = []
            for lo, hi in bands:
                rows = [b for b in bets if lo <= b['odds'] < hi]
                if not rows:
                    continue
                n = len(rows)
                w = sum(int(b['is_win']) for b in rows)
                s = sum(float(b['stake']) for b in rows)
                pf = sum(float(b['profit']) for b in rows)
                wr = (w / n * 100) if n else 0.0
                roi = (pf / s * 100) if s else 0.0
                log.info(f"  {f'{lo:.1f}-{hi:.0f}':<14} {n:>5} {w:>5} "
                         f"{wr:>7.2f}% ${s:>9,.0f} ${pf:>+9,.0f} "
                         f"{roi:>+7.2f}%")
            # Anything outside the bands (e.g. odds > top band)
            tracked = sum(len([b for b in bets
                               if lo <= b['odds'] < hi])
                          for lo, hi in bands)
            if tracked < len(bets):
                rows = [b for b in bets
                        if not any(lo <= b['odds'] < hi for lo, hi in bands)]
                n = len(rows)
                w = sum(int(b['is_win']) for b in rows)
                s = sum(float(b['stake']) for b in rows)
                pf = sum(float(b['profit']) for b in rows)
                wr = (w / n * 100) if n else 0.0
                roi = (pf / s * 100) if s else 0.0
                log.info(f"  {'(beyond bands)':<14} {n:>5} {w:>5} "
                         f"{wr:>7.2f}% ${s:>9,.0f} ${pf:>+9,.0f} "
                         f"{roi:>+7.2f}%")

            # ---- Race-concentration ----
            bets_per_race: dict[str, int] = defaultdict(int)
            for b in bets:
                bets_per_race[b['race_id']] += 1
            counts = sorted(bets_per_race.values())
            n_races = len(counts)
            total_bets = sum(counts)
            mean_bets = total_bets / n_races
            median_bets = counts[n_races // 2]
            max_bets = counts[-1]
            # Top-5% of races by bet count
            top_5pct_idx = max(0, int(n_races * 0.95))
            top_5pct_bets = sum(counts[top_5pct_idx:])
            top_5pct_share = (top_5pct_bets / total_bets * 100) if total_bets else 0.0
            log.info(f"  Race concentration: {n_races} races fired bets, "
                     f"mean={mean_bets:.1f} median={median_bets} max={max_bets}; "
                     f"top 5% of races held {top_5pct_share:.0f}% of bets.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start_date', default='2024-01-01')
    ap.add_argument('--end_date',   default='2026-01-01')
    ap.add_argument('--pools', default='WIN,PLA,QIN,QPL,TRI',
                    help='Comma-separated subset of WIN,PLA,QIN,QPL,TRI')
    ap.add_argument('--bankroll', type=float, default=100_000.0)
    ap.add_argument('--exotic_odds_source', default='real',
                    choices=['real', 'synthetic', 'hybrid'],
                    help="How to price exotic pools (PLA/QIN/QPL/TRI). "
                         "'real' (default) uses live_odds_history, skipping "
                         "races without coverage. 'synthetic' uses θ-symmetric "
                         "Harville projections from WIN-pool implieds (the "
                         "pre-Plan-C path; treat ROI as approximation only). "
                         "'hybrid' prefers real, falls back to synthetic.")
    ap.add_argument('--ledger_csv', default=None,
                    help="Path to write per-bet CSV ledger after the run "
                         "(columns: race_id, pool, combo, odds, p_model, "
                         "ev, stake, is_win, profit, odds_source). "
                         "If omitted, no CSV is written.")
    ap.add_argument('--theta_place', type=float, default=None,
                    help="Henery exponent for PLA public-side projection "
                         "(synthetic pub_odds). Fit against P_pub via "
                         "theta_place_diagnostic.py. Default: theta_2.")
    ap.add_argument('--theta_model_place', type=float, default=None,
                    help="Henery exponent for PLA model-side projection "
                         "(pp_model). Fit against P_ens via "
                         "theta_place_diagnostic --model_oof_csv "
                         "artifacts/wf_theta_input.csv. Lower than "
                         "theta_place corrects model longshot overconfidence. "
                         "Default: same as theta_place.")
    args = ap.parse_args()

    pools = tuple(p.strip().upper() for p in args.pools.split(','))
    bt = MultiPoolBacktester(
        db_url=DB_URL,
        starting_bankroll=args.bankroll,
        pools=pools,
        exotic_odds_source=args.exotic_odds_source,
        ledger_csv_path=args.ledger_csv,
        theta_place=args.theta_place,
        theta_model_place=args.theta_model_place,
    )
    bt.run_backtest(start_date=args.start_date, end_date=args.end_date)