"""
Multi-pool drift-aware backtester (Step 1 of the multi-pool roadmap).

Extends `XGBEnsembleBacktester` (which only settles WIN) to also settle
PLA / QIN / QPL / TRI. Sizing per pool reuses the live bot's existing
exotic-odds combinatoric logic (`p_quinella`, `p_quinella_place`, `p_trio`)
combined with the drift-aware `qualify_and_size` policy.
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
    DEFAULT_DRIFT_STATS, DriftStats, cap_simultaneous_stakes,
    fractional_kelly_stake, get_ev_hurdle, qualify_and_size,
    race_cap_for_pool, shrunk_ev,
)
from hkjc_engine.models.drift_forecaster import project_drift_to_exotic
from hkjc_engine.models.feature_factory import calculate_base_margin

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

POOL_NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    'WIN':  ('WIN', 'WINNER'),
    'PLA':  ('PLA', 'PLACE'),
    'QIN':  ('QIN', 'QUINELLA'),
    'QPL':  ('QPL', 'QUINELLA PLACE'),
    'TRI':  ('TRI', 'TRIO'),
}

def _normalise_pool(stored_pool: str) -> str | None:
    s = str(stored_pool).upper().strip()
    for canonical, variants in POOL_NAME_VARIANTS.items():
        if s in variants: return canonical
    return None

def _all_storage_names(pools: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in pools: out.extend(POOL_NAME_VARIANTS.get(p.upper(), (p.upper(),)))
    return out

def _canonical_combo(combination) -> str:
    s = str(combination).strip()
    if "," in s or "-" in s:
        parts = sorted(int(x) for x in s.replace(",", "-").split("-") if x.strip())
        return "-".join(str(x) for x in parts)
    return s

def _p_order_by_idx(p_arr: np.ndarray, i1: int, i2: int, i3: int | None, theta_2: float, theta_3: float) -> float:
    p1 = p_arr[i1]; sum_t2 = np.sum(p_arr ** theta_2) - (p1 ** theta_2)
    if sum_t2 <= 0: return 0.0
    p2 = p_arr[i2]; p_exact_2 = p1 * ((p2 ** theta_2) / sum_t2)
    if i3 is None: return float(p_exact_2)
    p3 = p_arr[i3]; sum_t3 = np.sum(p_arr ** theta_3) - (p1 ** theta_3) - (p2 ** theta_3)
    if sum_t3 <= 0: return 0.0
    return float(p_exact_2 * ((p3 ** theta_3) / sum_t3))

def p_quinella(p_arr, i, j, theta_2, theta_3): return _p_order_by_idx(p_arr, i, j, None, theta_2, theta_3) + _p_order_by_idx(p_arr, j, i, None, theta_2, theta_3)
def p_quinella_place(p_arr, i, j, theta_2, theta_3):
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j: continue
        total += _p_order_by_idx(p_arr, i, j, k, theta_2, theta_3) + _p_order_by_idx(p_arr, j, i, k, theta_2, theta_3) + _p_order_by_idx(p_arr, i, k, j, theta_2, theta_3) + _p_order_by_idx(p_arr, j, k, i, theta_2, theta_3) + _p_order_by_idx(p_arr, k, i, j, theta_2, theta_3) + _p_order_by_idx(p_arr, k, j, i, theta_2, theta_3)
    return total
def p_trio(p_arr, i, j, k, theta_2, theta_3): return sum(_p_order_by_idx(p_arr, *perm, theta_2, theta_3) for perm in itertools.permutations([i, j, k]))
def p_place(p_arr: np.ndarray, i: int, theta_2: float, theta_3: float) -> float:
    n = len(p_arr); p_1st = float(p_arr[i])
    p_2nd = sum(_p_order_by_idx(p_arr, j, i, None, theta_2, theta_3) for j in range(n) if j != i)
    p_3rd = 0.0
    for j in range(n):
        if j == i: continue
        for k in range(n):
            if k == i or k == j: continue
            p_3rd += _p_order_by_idx(p_arr, j, k, i, theta_2, theta_3)
    return p_1st + p_2nd + p_3rd

POOL_TAKEOUT = {'PLA': 0.175, 'QIN': 0.175, 'QPL': 0.175, 'TRI': 0.250}

def _public_pla_odds(p_public: np.ndarray, i: int, theta_2: float, theta_3: float) -> float:
    p_pub_place = p_place(p_public, i, theta_2, theta_3)
    if p_pub_place <= 0: return 0.0
    return (1.0 - POOL_TAKEOUT['PLA']) / p_pub_place

def _public_combo_odds(p_public: np.ndarray, combo_idx: tuple, calc_func, pool: str, theta_2: float, theta_3: float) -> float:
    p_pub = calc_func(p_public, *combo_idx, theta_2, theta_3)
    if p_pub <= 0: return 0.0
    return (1.0 - POOL_TAKEOUT[pool]) / p_pub

def _build_dividend_lookup(dividends_df: pd.DataFrame) -> dict:
    lookup: dict[tuple, float] = {}
    if dividends_df.empty: return lookup
    for _, row in dividends_df.iterrows():
        if pd.isna(row['pool_code']): continue
        key = (str(row['race_id']), str(row['pool_code']).upper(), _canonical_combo(row['combination']))
        try: lookup[key] = float(row['dividend'])
        except (TypeError, ValueError): continue
    return lookup

def _winning_combos(finish_df: pd.DataFrame, pla_combos_per_race: dict[str, set] | None = None) -> dict:
    out: dict[str, dict] = {}; pla_combos_per_race = pla_combos_per_race or {}
    for race_id, g in finish_df.groupby('race_id'):
        g = g.sort_values('finish_position')
        if len(g) < 2: continue
        g = g[g['finish_position'].notna() & (g['finish_position'] < 90)]
        if g.empty: continue
        top1 = int(g.iloc[0]['horse_no']) if len(g) >= 1 else None
        top2 = int(g.iloc[1]['horse_no']) if len(g) >= 2 else None
        top3 = int(g.iloc[2]['horse_no']) if len(g) >= 3 else None
        winner_no = str(top1)
        if race_id in pla_combos_per_race: placers_finish = pla_combos_per_race[race_id]
        else: placers_finish = {str(h) for h in (top1, top2, top3) if h is not None}
        out[race_id] = {'WIN': winner_no, 'PLA': placers_finish}
        if top1 is not None and top2 is not None:
            out[race_id]['QIN'] = '-'.join(str(x) for x in sorted([top1, top2]))
            placer_ints = sorted(int(h) for h in placers_finish)
            qpl_pairs = set()
            for combo in itertools.combinations(placer_ints, 2): qpl_pairs.add('-'.join(str(x) for x in sorted(combo)))
            out[race_id]['QPL'] = qpl_pairs
        if top1 is not None and top2 is not None and top3 is not None:
            out[race_id]['TRI'] = '-'.join(str(x) for x in sorted([top1, top2, top3]))
    return out


class MultiPoolBacktester(XGBEnsembleBacktester):
    POOLS_AVAILABLE: tuple[str, ...] = ('WIN', 'PLA', 'QIN', 'QPL', 'TRI')
    EXOTIC_ODDS_SOURCES: tuple[str, ...] = ('real', 'synthetic', 'hybrid')

    def __init__(self, *args, pools: Iterable[str] = ('WIN', 'PLA', 'QIN', 'QPL', 'TRI'),
                 top_n_horses_exotic: int = 8, exotic_odds_source: str = 'real',
                 ledger_csv_path: str | None = None, theta_place: float | None = None,
                 theta_model_place: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pools = tuple(p.upper() for p in pools)
        self.top_n_horses_exotic = top_n_horses_exotic
        self.exotic_odds_source = exotic_odds_source.lower()
        self.ledger_csv_path = ledger_csv_path
        self.theta_pub_place   = (theta_place if theta_place is not None else self.theta_2)
        self.theta_model_place = (theta_model_place if theta_model_place is not None else self.theta_pub_place)
        self.theta_place = self.theta_pub_place

    def _fetch_dividends(self, race_ids: list[str]) -> pd.DataFrame:
        if not race_ids: return pd.DataFrame(columns=['race_id', 'pool_code', 'combination', 'dividend'])
        q = text("SELECT race_id, pool_code, combination, dividend FROM race_dividends WHERE race_id = ANY(:race_ids) AND pool_code = ANY(:pools)")
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={'race_ids': race_ids, 'pools': list(self.pools)})
        if df.empty: return df
        df['dividend'] = df['dividend'].astype(float)
        return df

    def _fetch_real_exotic_odds(self, race_ids: list[str]) -> dict[tuple[str, str], dict[str, float]]:
        if not race_ids: return {}
        q = text("""
            WITH ranked AS (
                SELECT race_id, pool_type, combination, odds, phase, timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY race_id, pool_type, combination
                        ORDER BY CASE phase WHEN 'PRE_STOP_SELL' THEN 1 WHEN 'UNKNOWN' THEN 2 WHEN 'POST_STOP_SELL' THEN 3 WHEN 'FINAL' THEN 4 ELSE 5 END, timestamp DESC
                    ) AS rn
                FROM live_odds_history WHERE race_id = ANY(:race_ids) AND pool_type = ANY(:pools) AND odds > 1.0 AND odds < 999
            ) SELECT race_id, pool_type, combination, odds FROM ranked WHERE rn = 1
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(q, conn, params={'race_ids': race_ids, 'pools': list(self.pools)})
        if df.empty: return {}
        lookup: dict[tuple[str, str], dict[str, float]] = {}
        for _, row in df.iterrows():
            key = (str(row['race_id']), str(row['pool_type']))
            combo = _canonical_combo(row['combination'])
            try: odds = float(row['odds'])
            except (TypeError, ValueError): continue
            if odds <= 1.0: continue
            lookup.setdefault(key, {})[combo] = odds
        return lookup

    def _size_pla(self, p_arr_pla: np.ndarray, p_public: np.ndarray, horse_nos: list[str],
                  win_drift_df: pd.DataFrame, sim_pla_probs: dict[str, float], live_pla_odds: dict[str, float] | None = None) -> pd.DataFrame:
        rows = []; live_pla_odds = live_pla_odds or {}
        for i, hn in enumerate(horse_nos):
            pp_model = p_arr_pla[i] # Pull marginals directly from P_model_pla
            if pp_model <= 0: continue
            if hn in live_pla_odds: pub_odds = float(live_pla_odds[hn]); odds_source = 'real'
            elif self.exotic_odds_source == 'real': continue
            else: pub_odds = _public_pla_odds(p_public, i, self.theta_pub_place, self.theta_pub_place); odds_source = 'synthetic'
            if pub_odds <= 1.0: continue
            override = None
            if win_drift_df is not None and not win_drift_df.empty:
                from hkjc_engine.models.drift_forecaster import project_drift_to_exotic
                single = project_drift_to_exotic(win_drift_df, [int(hn)])
                override = {'PLA': DriftStats(**single)}
            from hkjc_engine.models.betting_policy import lookup_shrinkage
            eff_shrinkage = lookup_shrinkage(float(p_public[i]), self.shrinkage)
            stake, ev_eff = self._size_one(pp_model, pub_odds, pool='PLA', drift=override, shrinkage=eff_shrinkage)
            rows.append({'pool': 'PLA', 'combo': hn, 'odds': pub_odds, 'p_model': pp_model, 'ev': ev_eff, 'stake': stake, 'odds_source': odds_source})
        df = pd.DataFrame(rows)
        if not df.empty: df = self._apply_pool_cap(df, 'PLA')
        return df

    def _size_exotic(self, p_arr_exo: np.ndarray, p_public: np.ndarray, horse_nos: list[str],
                     pool: str, calc_func, comb_len: int, win_drift_df: pd.DataFrame,
                     sim_pool_probs: dict[str, float], live_combo_odds: dict[str, float] | None = None) -> pd.DataFrame:
        TOP_K_PER_RACE = {'QIN': 5, 'QPL': 5, 'TRI': 4}
        top_k = TOP_K_PER_RACE.get(pool.upper(), 5)

        n = len(p_arr_exo)
        top_indices = list(np.argsort(-p_arr_exo)[:min(self.top_n_horses_exotic, n)])
        rows = []; live_combo_odds = live_combo_odds or {}

        for combo_idx in itertools.combinations(top_indices, comb_len):
            h_nums = sorted(int(horse_nos[i]) for i in combo_idx)
            combo_key = '-'.join(str(x) for x in h_nums)
            
            p_model = sim_pool_probs.get(combo_key, 0.0)
            if p_model <= 0: continue

            if combo_key in live_combo_odds: pub_odds = float(live_combo_odds[combo_key]); odds_source = 'real'
            elif self.exotic_odds_source == 'real': continue
            else: pub_odds = _public_combo_odds(p_public, combo_idx, calc_func, pool, self.theta_2, self.theta_3); odds_source = 'synthetic'

            if pub_odds <= 1.0: continue

            # --- THE EXOTIC VETO (Syndicate Trap Detection) ---
            if pool in ['QIN', 'QPL']:
                fair_odds = 1.0 / p_model
                if pub_odds < fair_odds * 0.65: continue
                if pub_odds > fair_odds * 2.0: continue

            override = None
            if win_drift_df is not None and not win_drift_df.empty:
                from hkjc_engine.models.drift_forecaster import project_drift_to_exotic
                combo_drift = project_drift_to_exotic(win_drift_df, h_nums)
                override = {pool: DriftStats(**combo_drift)}

            from hkjc_engine.models.betting_policy import lookup_shrinkage
            min_p_pub = float(min(p_public[idx] for idx in combo_idx))
            
            # Geometric shrinkage implementation for N-dimension combination
            eff_shrinkage = lookup_shrinkage(min_p_pub, self.shrinkage) ** comb_len

            stake, ev_eff = self._size_one(p_model, pub_odds, pool=pool, drift=override, shrinkage=eff_shrinkage)
            rows.append({'pool': pool, 'combo': combo_key, 'odds': pub_odds, 'p_model': p_model, 'ev': ev_eff, 'stake': stake, 'odds_source': odds_source})

        df = pd.DataFrame(rows)
        if df.empty: return df

        qualifying = df[df['stake'] > 0]
        if len(qualifying) > top_k:
            keep_combos = (qualifying.nlargest(top_k, 'ev')['combo'].values)
            df = df[df['combo'].isin(keep_combos)]
        return self._apply_pool_cap(df, pool)

    def _size_one(self, p_raw: float, odds: float, pool: str, drift, shrinkage: float | None = None) -> tuple[float, float]:
        if pd.isna(odds) or odds <= 1.0 or p_raw <= 0: return 0.0, 0.0
        eff_shrinkage = shrinkage if shrinkage is not None else self.shrinkage
        if isinstance(eff_shrinkage, dict):
            from hkjc_engine.models.betting_policy import lookup_shrinkage
            eff_shrinkage = lookup_shrinkage(0.10, eff_shrinkage)
            
        ev_eff = shrunk_ev(p_raw, odds, eff_shrinkage, pool=pool, drift_override=drift)
        
        # DYNAMIC HURDLE SCALING
        pool_base_hurdle = 0.02
        pool_lambda_d = 4.0
        if pool == 'PLA':
            if odds < 3.0: pool_base_hurdle = 0.08
        elif pool in ['QIN', 'QPL']:
            pool_lambda_d = 7.0 
        elif pool == 'TRI':
            pool_lambda_d = 4.0
            
        hurdle = get_ev_hurdle(odds, base=pool_base_hurdle, longshot_buffer=0.015, longshot_threshold=15.0, pool=pool, lambda_d=pool_lambda_d, drift_override=drift)
        if ev_eff < hurdle: return 0.0, ev_eff
        stake = fractional_kelly_stake(p_raw, odds, self.bankroll, kelly_fraction=0.25, shrinkage=eff_shrinkage, per_bet_cap=0.02, min_stake_abs=10.0, pool=pool, drift_override=drift)
        return stake, ev_eff

    def _size_win_pool(self, df: pd.DataFrame, overrides: dict[int, DriftStats]) -> tuple[np.ndarray, np.ndarray]:
        from hkjc_engine.models.betting_policy import lookup_shrinkage, fractional_kelly_stake, get_ev_hurdle, race_cap_for_pool, qualify_and_size
        p_pub_raw = 1.0 / df['stop_sell_odds'].values
        p_pub = p_pub_raw / p_pub_raw.sum()
        shr_arr = lookup_shrinkage(p_pub, self.shrinkage)

        if not overrides:
            return qualify_and_size(p_raw=df['P_model_win'].values, odds=df['stop_sell_odds'].values, bankroll=self.bankroll, kelly_fraction=0.35, shrinkage=shr_arr, base_hurdle=0.005, longshot_buffer=0.005, longshot_threshold=25.0, per_bet_cap=0.05, race_cap=0.10, top_k=10, pool='WIN')

        n = len(df)
        stakes = np.zeros(n); evs = np.zeros(n)
        for i in range(n):
            ds = overrides.get(i, DEFAULT_DRIFT_STATS['WIN'])
            override_dict = {'WIN': ds}
            odds = float(df['stop_sell_odds'].iloc[i]); p = float(df['P_model_win'].iloc[i])
            shr_i = float(shr_arr[i]) if isinstance(shr_arr, np.ndarray) else float(shr_arr)
            ev = (min(shr_i * p, 1 - 1e-9) * odds * ds.median_R - 1.0)
            evs[i] = ev
            hurdle = get_ev_hurdle(odds, base=0.005, longshot_buffer=0.005, longshot_threshold=25.0, pool='WIN', drift_override=override_dict)
            if ev < hurdle or odds > 25.0: continue
            stakes[i] = fractional_kelly_stake(p_raw=p, odds=odds, bankroll=self.bankroll, kelly_fraction=0.35, shrinkage=shr_i, per_bet_cap=0.05, min_stake_abs=10.0, pool='WIN', drift_override=override_dict)

        keep_mask = stakes > 0
        if not keep_mask.any(): return np.array([], dtype=int), np.array([], dtype=float)
        keep_idx = np.where(keep_mask)[0]
        if len(keep_idx) > 10: keep_idx = keep_idx[np.argsort(-evs[keep_idx])[:10]]
        kept_stakes = stakes[keep_idx]
        kept_stakes = cap_simultaneous_stakes(kept_stakes, self.bankroll, race_cap=0.10, pool='WIN')
        final_keep = kept_stakes >= 10.0
        return keep_idx[final_keep], kept_stakes[final_keep]

    def _apply_pool_cap(self, df: pd.DataFrame, pool: str) -> pd.DataFrame:
        qual = df['stake'] > 0
        if not qual.any(): return df
        df = df.copy()
        df.loc[qual, 'stake'] = cap_simultaneous_stakes(df.loc[qual, 'stake'].values, self.bankroll, race_cap=0.04, pool=pool)
        df.loc[df['stake'] < 10.0, 'stake'] = 0.0
        return df

    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2026-01-01') -> tuple[list[dict], float]:
        log.info("Drift-aware MULTI-POOL backtest (%s)...", ', '.join(self.pools))
        self._debug_printed_pla = False 

        query = text("""
            WITH CareerCounts AS (
                SELECT e.race_id, e.horse_code, ROW_NUMBER() OVER (PARTITION BY e.horse_code ORDER BY r.race_date ASC, r.race_id ASC) AS career_run_number
                FROM race_entries e JOIN races r ON e.race_id = r.race_id
            )
            SELECT r.race_id, r.race_date, r.venue, r.distance, r.track_condition, r.rail_placement, r.race_class,
                   e.horse_code, e.horse_no, e.draw, e.actual_weight, e.win_odds, e.finish_position,
                   e.ema_early_z, e.ema_mid_z, e.ema_finish_z, e.pre_race_mu, e.pre_race_sigma, e.jockey, e.days_since_last_race,
                   e.is_class_drop, e.is_class_rise, CASE WHEN cc.career_run_number = 1 THEN 1 ELSE 0 END AS is_maiden
            FROM races r
            JOIN race_entries e ON r.race_id = e.race_id
            JOIN CareerCounts cc ON e.race_id = cc.race_id AND e.horse_code = cc.horse_code
            WHERE r.race_date >= :start_date AND r.race_date < :end_date AND e.win_odds IS NOT NULL AND e.finish_position IS NOT NULL
            ORDER BY r.race_date ASC, r.race_no ASC
        """)
        with self.engine.connect() as conn:
            raw = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        if raw.empty: return [], self.bankroll

        race_ids = raw['race_id'].astype(str).unique().tolist()
        dividends_df = self._fetch_dividends(race_ids)
        dividend_lookup = _build_dividend_lookup(dividends_df)

        exotic_pools = [p for p in self.pools if p != 'WIN']
        live_exotic_odds = self._fetch_real_exotic_odds(race_ids) if exotic_pools and self.exotic_odds_source != 'synthetic' else {}

        settleable_pool_keys: set[tuple[str, str]] = set()
        if not dividends_df.empty:
            for (rid, pool), _ in dividends_df.groupby(['race_id', 'pool_code']): settleable_pool_keys.add((str(rid), str(pool)))

        pla_combos_per_race: dict[str, set] = {}
        if not dividends_df.empty:
            pla_rows = dividends_df[dividends_df['pool_code'] == 'PLA']
            for race_id, g in pla_rows.groupby('race_id'): pla_combos_per_race[str(race_id)] = {_canonical_combo(c) for c in g['combination']}

        winning_combos = _winning_combos(raw[['race_id', 'horse_no', 'finish_position']].copy(), pla_combos_per_race=pla_combos_per_race)

        bet_ledger: list[dict] = []; bets_per_pool: dict[str, int] = {p: 0 for p in self.pools}; wins_per_pool: dict[str, int] = {p: 0 for p in self.pools}
        staked_per_pool: dict[str, float] = {p: 0.0 for p in self.pools}; profit_per_pool: dict[str, float] = {p: 0.0 for p in self.pools}; skipped_no_dividend: dict[str, int] = {p: 0 for p in self.pools}

        for race_id, race_df in raw.groupby('race_id'):
            df = self.factory.engineer_features(race_df.copy())
            df = self._attach_stop_sell(df)
            if df.empty or len(df) < 2: continue

            df['I_valid'] = (df['stop_sell_odds'].notna() & (df['stop_sell_odds'] != df['win_odds'])).astype(int)

            df['is_class_drop'] = df['is_class_drop'].astype(float).fillna(0.0)
            df['is_class_rise'] = df['is_class_rise'].astype(float).fillna(0.0)
            df['is_maiden']     = df['is_maiden'].astype(float)
            df['draw_x_early_pace']      = df['draw'] * df['relative_early_pace']
            df['straight_x_finish_pace'] = (df['straight_length'] / 360.0) * df['relative_finish_pace']
            df['class_drop_x_ts']        = df['is_class_drop'] * df['ts_advantage']
            df['class_rise_x_ts']        = df['is_class_rise'] * df['ts_advantage']

            # ---- Model A (WIN) ----
            dmat_a_win = xgb.DMatrix(df[self.FEATURES_A])
            df['base_margin_win'] = calculate_base_margin(df['stop_sell_odds'])
            dmat_a_win.set_base_margin(df['base_margin_win'])
            p_a_win_cal = self.calibrator_a_win.predict_proba(_softmax(self.model_a_win.predict(dmat_a_win)))[:, 1]

            # ---- Model A (PLA) ----
            dmat_a_pla = xgb.DMatrix(df[self.FEATURES_A])
            # Construct proper PLA base margin for backtester matching training
            places_paid = 3.0 if len(df) >= 7 else 2.0
            pla_odds = df.get('stop_sell_pla_odds', df['stop_sell_odds'] / 3.0)
            pi_pla = 1.0 / pla_odds
            p_fair_pla = (places_paid / pi_pla.sum()) * pi_pla
            p_clipped_pla = np.clip(p_fair_pla, 1e-5, 1 - 1e-5)
            df['base_margin_pla'] = np.log(p_clipped_pla / (1.0 - p_clipped_pla))
            
            dmat_a_pla.set_base_margin(df['base_margin_pla'])
            p_a_pla_cal = self.calibrator_a_pla.predict_proba(_softmax(self.model_a_pla.predict(dmat_a_pla)))[:, 1]

            # ---- Model B (CoxPH Shared) ----
            cox_features = ['relative_early_pace', 'relative_mid_pace', 'relative_finish_pace', 'weight_delta', 'draw', 'is_class_drop', 'is_class_rise', 'ts_advantage', 'is_maiden', 'track_width', 'straight_length']
            raw_b = self.model_b.predict_partial_hazard(df[cox_features])
            p_b_softmax = raw_b / raw_b.sum()
            
            bins = [-np.inf, -0.84, -0.25, 0.25, 0.84, np.inf]
            df['pace_archetype'] = pd.cut(df['relative_early_pace'], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
            p_b_cal = self.calibrator_b.predict_proba(p_b_softmax.values, strata=df['pace_archetype'].values)[:, 1]

            df['P_pub_raw'] = 1.0 / df['stop_sell_odds']
            p_pub = (df['P_pub_raw'] / df['P_pub_raw'].sum()).values
            p_mkt_pla = 1.0 / pla_odds.values
            p_mkt_pla /= p_mkt_pla.sum()

            # ---- Route to Stackers ----
            race_ids = df['race_id'].values
            
            P_win = np.column_stack([p_a_win_cal, p_b_cal, p_pub])
            df['P_model_win'] = self.stacker_win.predict(P_win, race_ids, I_valid=df['I_valid'].values)
            df['P_model'] = df['P_model_win'] # Assignment for any legacy inherited functions (e.g. diagnostics)
            
            P_pla = np.column_stack([p_a_pla_cal, p_b_cal, p_mkt_pla])
            df['P_model_pla'] = self.stacker_pla.predict(P_pla, race_ids, I_valid=df['I_valid'].values)
            
            df['P_model_exo'] = self.stacker_exo.predict(P_win, race_ids, I_valid=df['I_valid'].values)

            p_arr_win = df['P_model_win'].to_numpy(dtype=float)
            p_arr_pla = df['P_model_pla'].to_numpy(dtype=float)
            p_arr_exo = df['P_model_exo'].to_numpy(dtype=float)
            horse_nos = df['horse_no'].astype(str).tolist()
            pace_z_scores = df['relative_early_pace'].to_numpy(dtype=float)

            win_drift_df = self._drift_features_for_race(race_id) if self.drift_forecaster else pd.DataFrame()
            if not win_drift_df.empty and self.drift_forecaster: win_drift_df = self.drift_forecaster.predict_win(win_drift_df)

            if not hasattr(self, 'simulator'):
                from hkjc_engine.models.stern_simulator import CopulaGammaSimulator
                self.simulator = CopulaGammaSimulator(r_shape=2.5)
            
            sim_probs = self.simulator.simulate_exotics(p_target=p_arr_exo, horse_nos=horse_nos, pace_z_scores=pace_z_scores, n_paths=16384)

            pool_dfs: dict[str, pd.DataFrame] = {}
            if 'WIN' in self.pools:
                drift_overrides_per_idx = {}
                if not win_drift_df.empty:
                    preds = win_drift_df.set_index('horse_no')
                    for ridx, hn in enumerate(horse_nos):
                        if hn in preds.index:
                            p = preds.loc[hn]
                            from hkjc_engine.models.betting_policy import DriftStats
                            drift_overrides_per_idx[ridx] = DriftStats(mu_R=float(p['mu_R']), sigma_R=float(p['sigma_R']), median_R=float(p['median_R']))
                keep_idx, stakes = self._size_win_pool(df, drift_overrides_per_idx)
                rows = []
                for ridx, st in zip(keep_idx, stakes):
                    rows.append({'pool': 'WIN', 'combo': horse_nos[ridx], 'odds': float(df.iloc[ridx]['stop_sell_odds']), 'p_model': float(p_arr_win[ridx]), 'ev': float(p_arr_win[ridx] * df.iloc[ridx]['stop_sell_odds'] - 1.0), 'stake': float(st), 'odds_source': 'real'})
                pool_dfs['WIN'] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['pool','combo','odds','p_model','ev','stake','odds_source'])

            race_id_str = str(race_id)
            live_pla = live_exotic_odds.get((race_id_str, 'PLA'), {})
            live_qin = live_exotic_odds.get((race_id_str, 'QIN'), {})
            live_qpl = live_exotic_odds.get((race_id_str, 'QPL'), {})
            live_tri = live_exotic_odds.get((race_id_str, 'TRI'), {})

            if 'PLA' in self.pools: pool_dfs['PLA'] = self._size_pla(p_arr_pla, p_pub, horse_nos, win_drift_df, sim_probs.get('PLA', {}), live_pla)
            if 'QIN' in self.pools: pool_dfs['QIN'] = self._size_exotic(p_arr_exo, p_pub, horse_nos, 'QIN', p_quinella, 2, win_drift_df, sim_probs.get('QIN', {}), live_qin)
            if 'QPL' in self.pools: pool_dfs['QPL'] = self._size_exotic(p_arr_exo, p_pub, horse_nos, 'QPL', p_quinella_place, 2, win_drift_df, sim_probs.get('QPL', {}), live_qpl)
            if 'TRI' in self.pools: pool_dfs['TRI'] = self._size_exotic(p_arr_exo, p_pub, horse_nos, 'TRI', p_trio, 3, win_drift_df, sim_probs.get('TRI', {}), live_tri)

            total_stake = sum(d.loc[d['stake'] > 0, 'stake'].sum() for d in pool_dfs.values() if not d.empty)
            cap_dollars = self.bankroll * 0.06
            if total_stake > cap_dollars > 0:
                shrink = cap_dollars / total_stake
                for d in pool_dfs.values():
                    if d.empty: continue
                    d.loc[d['stake'] > 0, 'stake'] *= shrink
                    d.loc[d['stake'] < 10.0, 'stake'] = 0.0

            wc = winning_combos.get(race_id, {})
            for pool, dfp in pool_dfs.items():
                if dfp.empty: continue
                race_id_str = str(race_id)
                if (race_id_str, pool) not in settleable_pool_keys:
                    n_skipped = int((dfp['stake'] >= 10.0).sum())
                    if n_skipped > 0: skipped_no_dividend[pool] += n_skipped
                    continue

                for _, bet in dfp[dfp['stake'] >= 10.0].iterrows():
                    stake = float(bet['stake'])
                    combo = _canonical_combo(bet['combo'])
                    is_win = self._is_winning_bet(pool, combo, wc)
                    if is_win:
                        div = self._lookup_dividend(dividend_lookup, race_id, pool, combo, wc)
                        if div is None: continue
                        payout = (stake / 10.0) * div
                        profit = payout - stake
                        wins_per_pool[pool] += 1
                    else: profit = -stake

                    self.bankroll += profit
                    bets_per_pool[pool] += 1
                    staked_per_pool[pool] += stake
                    profit_per_pool[pool] += profit

                    bet_ledger.append({'race_id': race_id, 'pool': pool, 'combo': combo, 'odds': float(bet['odds']), 'p_model': float(bet['p_model']), 'ev': float(bet['ev']), 'stake': stake, 'is_win': bool(is_win), 'profit': float(profit), 'odds_source': bet.get('odds_source', 'real')})

        self._log_summary(bets_per_pool, wins_per_pool, staked_per_pool, profit_per_pool, skipped_no_dividend, bet_ledger=bet_ledger)
        if bet_ledger: self._log_distribution_summary(bet_ledger)
        if self.ledger_csv_path and bet_ledger:
            try: pd.DataFrame(bet_ledger).to_csv(self.ledger_csv_path, index=False)
            except Exception as e: pass

        return bet_ledger, self.bankroll

    def _is_winning_bet(self, pool: str, combo: str, wc: dict) -> bool:
        if pool not in wc: return False
        winners = wc[pool]
        if pool in ('WIN', 'QIN', 'TRI'): return combo == winners
        if pool in ('PLA', 'QPL'): return combo in winners
        return False

    def _lookup_dividend(self, lookup: dict, race_id: str, pool: str, combo: str, wc: dict) -> float | None:
        return lookup.get((str(race_id), pool, combo))

    def _log_summary(self, bets_per_pool: dict, wins_per_pool: dict, staked_per_pool: dict, profit_per_pool: dict, skipped_no_dividend: dict | None = None, bet_ledger: list[dict] | None = None):
        skipped_no_dividend = skipped_no_dividend or {}
        log.info("\n" + "=" * 70)
        log.info("   DRIFT-AWARE MULTI-POOL BACKTEST RESULTS")
        log.info("=" * 70)
        log.info(f"{'Pool':<6} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Staked':>12} {'Profit':>12} {'ROI':>8}  {'Skip*':>6}")
        total_bets = total_wins = 0
        total_staked = total_profit = total_skipped = 0.0
        for p in self.pools:
            n = bets_per_pool.get(p, 0); w = wins_per_pool.get(p, 0)
            s = staked_per_pool.get(p, 0.0); pf = profit_per_pool.get(p, 0.0)
            sk = skipped_no_dividend.get(p, 0)
            wr = (w / n * 100) if n > 0 else 0.0
            roi = (pf / s * 100) if s > 0 else 0.0
            log.info(f"{p:<6} {n:>6} {w:>6} {wr:>7.2f}% ${s:>11,.0f} ${pf:>+11,.0f} {roi:>+7.2f}% {sk:>6}")
            total_bets += n; total_wins += w; total_staked += s; total_profit += pf; total_skipped += sk
        log.info("-" * 70)
        total_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
        log.info(f"{'TOTAL':<6} {total_bets:>6} {total_wins:>6} {'':>8} ${total_staked:>11,.0f} ${total_profit:>+11,.0f} {total_roi:>+7.2f}% {int(total_skipped):>6}")
        if bet_ledger:
            sources_seen = {b.get('odds_source', 'real') for b in bet_ledger}
            if len(sources_seen) > 1:
                log.info("\n" + "-" * 70)
                log.info("   BREAKDOWN BY ODDS SOURCE")
                log.info("-" * 70)
                log.info(f"{'Pool':<6} {'Source':<10} {'Bets':>6} {'Wins':>6} {'WinRate':>8} {'Staked':>12} {'Profit':>12} {'ROI':>8}")
                from collections import defaultdict
                acc: dict = defaultdict(lambda: {'n': 0, 'w': 0, 's': 0.0, 'pf': 0.0})
                for b in bet_ledger:
                    key = (b['pool'], b.get('odds_source', 'real'))
                    acc[key]['n'] += 1; acc[key]['w'] += int(b['is_win'])
                    acc[key]['s'] += float(b['stake']); acc[key]['pf'] += float(b['profit'])
                for p in self.pools:
                    for src in ('real', 'synthetic'):
                        if (p, src) not in acc: continue
                        a = acc[(p, src)]
                        wr = (a['w'] / a['n'] * 100) if a['n'] > 0 else 0.0
                        roi = (a['pf'] / a['s'] * 100) if a['s'] > 0 else 0.0
                        log.info(f"{p:<6} {src:<10} {a['n']:>6} {a['w']:>6} {wr:>7.2f}% ${a['s']:>11,.0f} ${a['pf']:>+11,.0f} {roi:>+7.2f}%")
        log.info(f"\nEnding Bankroll: ${self.bankroll:,.2f}\n" + "=" * 70)

    def _log_distribution_summary(self, bet_ledger: list[dict]) -> None:
        if not bet_ledger: return
        bands_by_pool = {
            'WIN': [(1.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 25.0)],
            'PLA': [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 25.0)],
            'QIN': [(1.0, 10.0), (10.0, 30.0), (30.0, 80.0), (80.0, 300.0)],
            'QPL': [(1.0,  5.0), ( 5.0, 15.0), (15.0, 40.0), (40.0, 150.0)],
            'TRI': [(1.0, 30.0), (30.0,100.0), (100.0,300.0), (300.0,2000.0)],
        }
        from collections import defaultdict
        per_pool: dict[str, list[dict]] = defaultdict(list)
        for b in bet_ledger: per_pool[b['pool']].append(b)

        log.info("\n" + "=" * 70)
        log.info("   PER-POOL DISTRIBUTION DIAGNOSTICS")
        log.info("=" * 70)

        for pool in self.pools:
            bets = per_pool.get(pool, [])
            if not bets: continue
            bands = bands_by_pool.get(pool, [(1.0, 5.0), (5.0, 20.0), (20.0, 100.0), (100.0, 5000.0)])
            log.info(f"\n{pool} — {len(bets)} bets — by ODDS band")
            log.info(f"  {'Range':<14} {'Bets':>5} {'Wins':>5} {'WinRate':>8} {'Staked':>10} {'Profit':>10} {'ROI':>8}")
            for lo, hi in bands:
                rows = [b for b in bets if lo <= b['odds'] < hi]
                if not rows: continue
                n = len(rows); w = sum(int(b['is_win']) for b in rows); s = sum(float(b['stake']) for b in rows); pf = sum(float(b['profit']) for b in rows)
                log.info(f"  {f'{lo:.1f}-{hi:.0f}':<14} {n:>5} {w:>5} {(w/n*100) if n else 0:>7.2f}% ${s:>9,.0f} ${pf:>+9,.0f} {(pf/s*100) if s else 0:>+7.2f}%")
            
            bets_per_race: dict[str, int] = defaultdict(int)
            for b in bets: bets_per_race[b['race_id']] += 1
            counts = sorted(bets_per_race.values())
            n_races = len(counts); total_bets = sum(counts)
            top_5pct_idx = max(0, int(n_races * 0.95))
            log.info(f"  Race concentration: {n_races} races fired bets, mean={total_bets/n_races:.1f} median={counts[n_races//2]} max={counts[-1]}; top 5% of races held {(sum(counts[top_5pct_idx:])/total_bets*100) if total_bets else 0:.0f}% of bets.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start_date', default='2024-01-01')
    ap.add_argument('--end_date',   default='2026-01-01')
    ap.add_argument('--pools', default='WIN,PLA,QIN,QPL,TRI',
                    help='Comma-separated subset of WIN,PLA,QIN,QPL,TRI')
    ap.add_argument('--bankroll', type=float, default=100_000.0)
    ap.add_argument('--exotic_odds_source', default='real',
                    choices=['real', 'synthetic', 'hybrid'])
    ap.add_argument('--ledger_csv', default=None)
    ap.add_argument('--theta_place', type=float, default=None)
    ap.add_argument('--theta_model_place', type=float, default=None)
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