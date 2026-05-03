"""
stern_exotics_backtest.py
=========================

A standalone backtester to evaluate the Stern Gamma model against historical 
exotic odds. It inherits data-loading and sizing mechanics from 
`exotics_backtest.py`, but replaces the Henery probability projections with 
live Quasi-Monte Carlo simulations.

This script features "Dual-Simulation" to prevent survivorship bias: 
If a combination's true pre-jump odds are missing from the database, it 
simulates the Public's implied probabilities to generate synthetic closing 
odds, ensuring losing combinations are properly evaluated and gated.
"""

import argparse
import os
import logging
import pandas as pd
import numpy as np
import itertools

from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL
from hkjc_engine.diagnostics.exotics_backtest import (
    load_inputs, tear_sheet_by_pool, tear_sheet_by_decile, 
    POOL_HIT_FN, _size_single_exotic, _load_closing_from_csv, 
    _load_closing_from_dividends
)
from hkjc_engine.models.betting_policy import cap_simultaneous_stakes
from hkjc_engine.models.stern_simulator import SternGammaSimulator

# Re-use production constraints
MASTER_RACE_CAP = 0.06
POOL_CAP_EXO = 0.03
MIN_STAKE_ABS = 10.0
TOP_N_HORSES_EXO = 8

# Track takeout rates to construct synthetic odds for missing combinations
POOL_TAKEOUT = {
    'PLA': 0.175, 
    'QIN': 0.175, 
    'QPL': 0.175, 
    'TRI': 0.250
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _load_closing_from_live_history(race_ids: list[str]) -> pd.DataFrame:
    from sqlalchemy import create_engine, text
    from hkjc_engine.config import DB_URL
    import logging
    
    if not race_ids:
        return pd.DataFrame()

    rids_str = ", ".join(f"'{rid}'" for rid in race_ids)
    engine = create_engine(DB_URL)
    q = f"""
        WITH ranked AS (
            SELECT
                race_id, pool_type, combination, odds, phase, timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY race_id, pool_type, combination
                    ORDER BY
                        CASE phase
                            WHEN 'PRE_STOP_SELL'  THEN 1
                            WHEN 'UNKNOWN'        THEN 2
                            WHEN 'UKNOWN'         THEN 2
                            WHEN 'POST_STOP_SELL' THEN 3
                            WHEN 'FINAL'          THEN 4
                            ELSE 5
                        END ASC,
                        timestamp DESC
                ) AS rn
            FROM live_odds_history
            WHERE race_id IN ({rids_str})
              AND pool_type IN ('QIN', 'QPL', 'TRI')
              AND odds > 1.0 AND odds < 999
        )
        SELECT race_id, pool_type, combination, odds
        FROM ranked
        WHERE rn = 1
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)
        
    # FIX: Convert HKJC $10-based dividend odds to standard 1-based decimal odds
    df["odds"] = df["odds"] / 10.0
        
    log = logging.getLogger(__name__)
    log.info(f"Loaded {len(df)} live odds snapshots from live_odds_history.")
    return df


def build_stern_tickets_for_race(race, pool, model_stern_probs, public_stern_probs, bankroll, shrinkage):
    odds_dict = race.closing_exotic_odds.get(pool, {})
    
    n = len(race.p_model)
    top_n = min(TOP_N_HORSES_EXO, n)
    top_indices = np.argsort(-race.p_model)[:top_n]
    comb_len = 2 if pool in ('QIN', 'QPL') else 3

    rows = []
    for combo_idx in itertools.combinations(top_indices, comb_len):
        h_nums = sorted(race.horse_nos[i] for i in combo_idx)
        key = "-".join(str(x) for x in h_nums)
        
        if key in odds_dict:
            odds = odds_dict[key]
        else:
            p_pub_combo = public_stern_probs[pool].get(key, 0.0)
            # FIX: QMC Noise Floor. Require at least 5 simulation hits (out of 16384)
            # to trust the synthetic odds. Otherwise, it's just statistical noise.
            if p_pub_combo < 0.0012:
                continue
            odds = (1.0 - POOL_TAKEOUT[pool]) / p_pub_combo
            
        if odds <= 1.0:
            continue
            
        p = model_stern_probs[pool].get(key, 0.0)
        
        if p <= 0:
            continue
            
        p_adj = min(p * shrinkage, 1 - 1e-9)
        ev_adj = p_adj * odds - 1.0
        stake = _size_single_exotic(p, odds, bankroll, shrinkage)
        
        rows.append({
            "race_id": race.race_id, "pool": pool, "combo": key,
            "combo_horses": tuple(h_nums), "odds_close": odds,
            "p_model_raw": p, "p_model_shrunk": p_adj,
            "ev_pre_cap": ev_adj, "stake_pre_cap": stake
        })
        
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # APPLY TOP-K COMBINATORIAL CAP
    TOP_K_PER_RACE = {'QIN': 10, 'QPL': 10, 'TRI': 10}
    top_k = TOP_K_PER_RACE.get(pool, 5)
    
    qualifying = df[df["stake_pre_cap"] > 0]
    if len(qualifying) > top_k:
        keep_combos = qualifying.nlargest(top_k, 'ev_pre_cap')['combo'].values
        df = df[df['combo'].isin(keep_combos)].copy()
        
    qual_mask = df["stake_pre_cap"] > 0
    df["stake"] = df["stake_pre_cap"].copy()
    if qual_mask.any():
        capped = cap_simultaneous_stakes(
            df.loc[qual_mask, "stake_pre_cap"].values, bankroll, POOL_CAP_EXO,
        )
        df.loc[qual_mask, "stake"] = capped
        df.loc[df["stake"] < MIN_STAKE_ABS, "stake"] = 0.0
    return df


def replay_race_with_stern(race, sim, bankroll, shrinkage):
    """Dual-simulation per race to generate both model beliefs and synthetic public odds."""
    
    p_pub_raw = 1.0 / race.win_odds_close
    p_pub = p_pub_raw / np.sum(p_pub_raw)
    
    p_model_corrected = np.copy(race.p_model)
    
    # Identify longshots (e.g., implied probability < 5% or odds > 20)
    tail_mask = p_pub < 0.05
    
    # Blend: 80% Public, 20% Model on the extreme tail to kill phantom EV
    p_model_corrected[tail_mask] = (0.80 * p_pub[tail_mask]) + (0.20 * race.p_model[tail_mask])
    
    # Re-normalize
    p_model_corrected = p_model_corrected / np.sum(p_model_corrected)

    # 1. RUN THE SIMULATION FOR THE MODEL (Using Corrected Probs)
    model_stern_probs = sim.simulate_exotics(p_model_corrected, race.horse_nos, n_paths=16384)
    
    # 2. RUN THE SIMULATION FOR THE PUBLIC (To generate synthetic odds for losing combos)
    p_pub_raw = 1.0 / race.win_odds_close
    p_pub = p_pub_raw / np.sum(p_pub_raw) # Normalize
    public_stern_probs = sim.simulate_exotics(p_pub, race.horse_nos, n_paths=16384)
    
    # 3. SIZE THE TICKETS
    pool_dfs = {
        p: build_stern_tickets_for_race(race, p, model_stern_probs, public_stern_probs, bankroll, shrinkage)
        for p in ("QIN", "QPL", "TRI")
    }

    # 4. APPLY MASTER RACE CAP
    total_staked = sum(d.loc[d["stake"] > 0, "stake"].sum() for d in pool_dfs.values() if not d.empty)
    cap_dollars = bankroll * MASTER_RACE_CAP
    
    if total_staked > cap_dollars > 0:
        shrink = cap_dollars / total_staked
        for d in pool_dfs.values():
            if d.empty: continue
            d.loc[d["stake"] > 0, "stake"] *= shrink
            d.loc[(d["stake"] > 0) & (d["stake"] < MIN_STAKE_ABS), "stake"] = 0.0

    non_empty = [d for d in pool_dfs.values() if not d.empty]
    if not non_empty:
        return pd.DataFrame()
        
    out = pd.concat(non_empty, ignore_index=True)
    if out.empty:
        return out

    # 5. RESOLVE WINNERS AND PNL
    def _resolve(row):
        won = POOL_HIT_FN[row["pool"]](row["combo_horses"], race.result)
        stake = row["stake"]
        pnl = stake * (row["odds_close"] - 1.0) if won else (-stake if stake > 0 else 0.0)
        ev_dollars = stake * (row["p_model_shrunk"] * row["odds_close"] - 1.0)
        return pd.Series({"won": won, "pnl_realised": pnl, "pnl_expected": ev_dollars})
        
    out[["won", "pnl_realised", "pnl_expected"]] = out.apply(_resolve, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closing_source", choices=["csv", "dividends", "live"], default="csv")
    ap.add_argument("--closing_odds", default=None)
    ap.add_argument("--dividend_unit_base", type=float, default=10.0)
    ap.add_argument("--results", required=True)
    ap.add_argument("--p_model", required=True)
    ap.add_argument("--bankroll", type=float, default=100_000.0)
    ap.add_argument("--shrinkage", type=float, default=0.93)
    ap.add_argument("--out_dir", default="./stern_backtest_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p_model_df = pd.read_csv(args.p_model)
    race_ids = sorted(p_model_df["race_id"].unique().tolist())

    if args.closing_source == "csv":
        closing_df = _load_closing_from_csv(args.closing_odds)
    elif args.closing_source == "live":
        closing_df = _load_closing_from_live_history(race_ids)
    else:
        closing_df = _load_closing_from_dividends(race_ids, args.dividend_unit_base)

    races = load_inputs(closing_df, args.results, args.p_model)
    
    # --- NEW: FILTER OUT RACES WITHOUT LIVE ODDS ---
    if args.closing_source == "live":
        valid_live_races = set(closing_df["race_id"].unique())
        races = [r for r in races if r.race_id in valid_live_races]
        log.info(f"Filtered down to {len(races)} races that have actual live odds data.")
        
        if not races:
            log.error("No races left after filtering. Check database connection or date range.")
            return
    
    # Initialize the Simulator (r=2.5 is a solid starting baseline for HKJC)
    sim = SternGammaSimulator(r_shape=2.5)

    ledgers = []
    for i, race in enumerate(races, 1):
        if i % 10 == 0 or i == len(races):
            log.info(f"Simulating race {i}/{len(races)}...")
        led = replay_race_with_stern(race, sim, args.bankroll, args.shrinkage)
        if not led.empty:
            ledgers.append(led)

    if not ledgers:
        log.error("No tickets produced. Check your inputs or thresholds.")
        return

    full_ledger = pd.concat(ledgers, ignore_index=True)
    full_ledger.to_csv(os.path.join(args.out_dir, "stern_ticket_ledger.csv"), index=False)

    pool_sheet = tear_sheet_by_pool(full_ledger)
    pool_sheet.to_csv(os.path.join(args.out_dir, "stern_tear_sheet_by_pool.csv"))
    
    decile_sheet = tear_sheet_by_decile(full_ledger)
    decile_sheet.to_csv(os.path.join(args.out_dir, "stern_tear_sheet_by_decile.csv"))

    bets = full_ledger[full_ledger["stake"] > 0]
    print("\n" + "=" * 78)
    print(f"STERN GAMMA BACKTEST  |  bankroll=${args.bankroll:,.0f}  shrinkage={args.shrinkage}")
    print(f"closing_source={args.closing_source} | races={len(races)}  tickets_placed={len(bets)}  total_staked=${bets['stake'].sum():,.0f}")
    print(f"NET PnL: ${bets['pnl_realised'].sum():+,.0f}   (EV under model: ${bets['pnl_expected'].sum():+,.0f})")
    print("=" * 78)
    
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 220, "display.max_columns", None):
        print("\n--- Tear sheet by pool ---")
        print(pool_sheet)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()