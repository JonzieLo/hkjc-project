"""
exotics_backtester.py
=====================

Drop-in replacement for src/hkjc_engine/diagnostics/exotics_backtest.py.

Replays the production run_bot.py exotic-pool logic against historical
closing odds and matches every ticket to the actual race result to compute
realised PnL.

Key changes vs. previous version:

  1. Honest sample-size statistics: Clopper-Pearson 95% CIs on hit rate,
     expected wins, and a binomial p-value per pool. The previous
     `calibration_slip = 0.000` "model overstating win prob >30%" flag was
     mechanically firing on zero-win cases that were statistically
     indistinguishable from perfect calibration. That flag is now suppressed
     when E[wins] < 5 OR n_tickets < 30, and replaced with an explicit
     "INSUFFICIENT_SAMPLE" verdict.

  2. Closing-odds source is selectable. Previously we required a CSV in the
     `claude_exotics_diagnostic.csv` schema, which only exists for the
     ~9 days of live archival data. The replacement also reads from the
     `race_dividends` table for historical replay (--closing_source dividends),
     which gives access to thousands of historical races.

  3. Optional contextual theta. If a JSON file from theta_optimizer.py
     --stratify is provided via --theta_json, per-race theta is looked up
     by (distance, field_size, race_class) instead of using the global
     constant. Lets you A/B contextual vs. static theta on the same data.

  4. Output now includes an OOF dump consumable by theta_optimizer.py,
     closing the diagnostic loop: backtest -> OOF dump -> theta refit ->
     re-backtest with new theta.

Inputs:
    --closing_odds CSV : race_id, pool_type, combination, POST_STOP_SELL
                         (live archive format)  -- OR --
    --closing_source dividends : pull from race_dividends table directly
    --results      CSV : race_id, pos1_horse_no, pos2_horse_no, pos3_horse_no
    --p_model      CSV : race_id, horse_no, p_model, win_odds_close

Outputs (in --out_dir):
    ticket_ledger.csv       per-ticket detail
    tear_sheet_by_pool.csv  pool-level stats with sample-size honest verdict
    tear_sheet_by_decile.csv per-decile-by-pool breakdown
    oof_for_theta.csv       feed-forward into theta_optimizer.py

Usage examples:
    # Live-archive replay (your original 9-day dataset)
    python exotics_backtester.py \\
        --closing_odds claude_exotics_diagnostic.csv \\
        --results      results.csv \\
        --p_model      p_model_close.csv

    # Historical replay using race_dividends (1000+ races)
    python exotics_backtester.py \\
        --closing_source dividends \\
        --results      results.csv \\
        --p_model      p_model_close.csv \\
        --dividend_unit_base 10

    # With contextual theta from theta_optimizer.py --stratify
    python exotics_backtester.py \\
        --closing_source dividends \\
        --results results.csv --p_model p_model_close.csv \\
        --theta_json contextual_thetas.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hkjc_engine.models.betting_policy import (  # noqa: F401
    cap_simultaneous_stakes, get_ev_hurdle,
)


# --- Production constants (mirror run_bot.py) -------------------------------
THETA_2_DEFAULT  = 0.8824
THETA_3_DEFAULT  = 0.7760
KELLY_FRAC       = 0.25
BASE_HURDLE      = 0.02
LONGSHOT_BUF     = 0.015
LONGSHOT_D       = 15.0
PER_BET_CAP      = 0.02
POOL_CAP_EXO     = 0.03
MASTER_RACE_CAP  = 0.06
MIN_STAKE_ABS    = 10.0
TOP_N_HORSES_EXO = 8
SHRINKAGE_DEFAULT = 0.75

# Sample-size thresholds for triggering a calibration verdict
MIN_TICKETS_FOR_VERDICT       = 30
MIN_EXPECTED_WINS_FOR_VERDICT = 5.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Henery / Lo-Bacon-Shone probability primitives (theta now per-call)
# ---------------------------------------------------------------------------

def _p_order_by_idx(p_arr, i1, i2, i3=None,
                    theta_2=THETA_2_DEFAULT, theta_3=THETA_3_DEFAULT):
    p1 = p_arr[i1]; p2 = p_arr[i2]
    sum_t2 = np.sum(p_arr ** theta_2) - p1 ** theta_2
    if sum_t2 <= 0:
        return 0.0
    p_exact_2 = p1 * (p2 ** theta_2 / sum_t2)
    if i3 is None:
        return p_exact_2
    p3 = p_arr[i3]
    sum_t3 = np.sum(p_arr ** theta_3) - p1 ** theta_3 - p2 ** theta_3
    if sum_t3 <= 0:
        return 0.0
    return p_exact_2 * (p3 ** theta_3 / sum_t3)


def p_quinella(p_arr, i, j, theta_2, theta_3):
    return (_p_order_by_idx(p_arr, i, j, theta_2=theta_2, theta_3=theta_3)
            + _p_order_by_idx(p_arr, j, i, theta_2=theta_2, theta_3=theta_3))


def p_quinella_place(p_arr, i, j, theta_2, theta_3):
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        for a, b, c in (
            (i, j, k), (j, i, k), (i, k, j),
            (j, k, i), (k, i, j), (k, j, i),
        ):
            total += _p_order_by_idx(p_arr, a, b, c,
                                     theta_2=theta_2, theta_3=theta_3)
    return total


def p_trio(p_arr, i, j, k, theta_2, theta_3):
    total = 0.0
    for a, b, c in itertools.permutations([i, j, k]):
        total += _p_order_by_idx(p_arr, a, b, c,
                                 theta_2=theta_2, theta_3=theta_3)
    return total


# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------

def qin_hits(combo, result): return set(combo) == set(result[:2])
def qpl_hits(combo, result): return set(combo).issubset(set(result[:3]))
def tri_hits(combo, result): return tuple(combo) == tuple(result[:3])

POOL_HIT_FN   = {"QIN": qin_hits, "QPL": qpl_hits, "TRI": tri_hits}
POOL_PROB_FN  = {"QIN": p_quinella, "QPL": p_quinella_place, "TRI": p_trio}
POOL_COMB_LEN = {"QIN": 2, "QPL": 2, "TRI": 3}


# ---------------------------------------------------------------------------
# Sizing — replicates run_bot.py
# ---------------------------------------------------------------------------

def _size_single_exotic(p_raw, odds, bankroll, shrinkage):
    if pd.isna(odds) or odds <= 1.0 or p_raw <= 0:
        return 0.0
    p_adj = min(p_raw * shrinkage, 1 - 1e-9)
    ev_adj = p_adj * odds - 1.0
    hurdle = get_ev_hurdle(odds, BASE_HURDLE, LONGSHOT_BUF, LONGSHOT_D)
    if ev_adj < hurdle:
        return 0.0
    b = odds - 1.0
    f = min(KELLY_FRAC * (ev_adj / b), PER_BET_CAP)
    stake = f * bankroll
    return stake if stake >= MIN_STAKE_ABS else 0.0


# ---------------------------------------------------------------------------
# Per-race replay
# ---------------------------------------------------------------------------

@dataclass
class RaceReplay:
    race_id: str
    horse_nos: list
    p_model: np.ndarray
    win_odds_close: np.ndarray
    closing_exotic_odds: dict
    result: tuple
    distance: float | None = None
    field_size: int | None = None
    race_class: str | None = None


def _resolve_thetas(race: RaceReplay, theta_table: dict | None):
    """Returns (theta_2, theta_3) for this race using the contextual table if
    provided and the race has covariates, else the global defaults."""
    if (theta_table is None
            or race.distance is None
            or race.field_size is None
            or race.race_class is None):
        return THETA_2_DEFAULT, THETA_3_DEFAULT
    try:
        from hkjc_engine.models.theta_optimizer import get_thetas_for_race
        return get_thetas_for_race(race.distance, race.field_size,
                                   race.race_class, theta_table)
    except Exception:
        g = theta_table.get('_global', {})
        return (float(g.get('theta_2', THETA_2_DEFAULT)),
                float(g.get('theta_3', THETA_3_DEFAULT)))


def build_exotic_tickets_for_race(race, pool, bankroll, shrinkage,
                                  theta_2, theta_3):
    odds_dict = race.closing_exotic_odds.get(pool, {})
    if not odds_dict:
        return pd.DataFrame()
    n = len(race.p_model)
    top_n = min(TOP_N_HORSES_EXO, n)
    top_indices = np.argsort(-race.p_model)[:top_n]
    comb_len = POOL_COMB_LEN[pool]
    prob_fn  = POOL_PROB_FN[pool]

    rows = []
    for combo_idx in itertools.combinations(top_indices, comb_len):
        h_nums = sorted(race.horse_nos[i] for i in combo_idx)
        key = "-".join(str(x) for x in h_nums)
        if key not in odds_dict:
            continue
        odds = odds_dict[key]
        p = prob_fn(race.p_model, *combo_idx, theta_2=theta_2, theta_3=theta_3)
        if p <= 0:
            continue
        p_adj = min(p * shrinkage, 1 - 1e-9)
        ev_adj = p_adj * odds - 1.0
        stake = _size_single_exotic(p, odds, bankroll, shrinkage)
        rows.append({
            "race_id": race.race_id, "pool": pool, "combo": key,
            "combo_horses": tuple(h_nums), "odds_close": odds,
            "p_model_raw": p, "p_model_shrunk": p_adj,
            "ev_pre_cap": ev_adj, "stake_pre_cap": stake,
            "theta_2": theta_2, "theta_3": theta_3,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    qual_mask = df["stake_pre_cap"] > 0
    df["stake"] = df["stake_pre_cap"].copy()
    if qual_mask.any():
        capped = cap_simultaneous_stakes(
            df.loc[qual_mask, "stake_pre_cap"].values, bankroll, POOL_CAP_EXO,
        )
        df.loc[qual_mask, "stake"] = capped
        df.loc[df["stake"] < MIN_STAKE_ABS, "stake"] = 0.0
    return df


def replay_race(race, bankroll, shrinkage, theta_table=None):
    theta_2, theta_3 = _resolve_thetas(race, theta_table)
    pool_dfs = {p: build_exotic_tickets_for_race(race, p, bankroll, shrinkage,
                                                 theta_2, theta_3)
                for p in ("QIN", "QPL", "TRI")}

    total_staked = sum(d.loc[d["stake"] > 0, "stake"].sum()
                       for d in pool_dfs.values() if not d.empty)
    cap_dollars = bankroll * MASTER_RACE_CAP
    if total_staked > cap_dollars > 0:
        shrink = cap_dollars / total_staked
        for d in pool_dfs.values():
            if d.empty:
                continue
            d.loc[d["stake"] > 0, "stake"] *= shrink
            d.loc[(d["stake"] > 0) & (d["stake"] < MIN_STAKE_ABS), "stake"] = 0.0

    non_empty = [d for d in pool_dfs.values() if not d.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    if out.empty:
        return out

    def _resolve(row):
        won = POOL_HIT_FN[row["pool"]](row["combo_horses"], race.result)
        stake = row["stake"]
        pnl = stake * (row["odds_close"] - 1.0) if won else (-stake if stake > 0 else 0.0)
        ev_dollars = stake * (row["p_model_shrunk"] * row["odds_close"] - 1.0)
        return pd.Series({"won": won, "pnl_realised": pnl, "pnl_expected": ev_dollars})
    out[["won", "pnl_realised", "pnl_expected"]] = out.apply(_resolve, axis=1)
    return out


# ---------------------------------------------------------------------------
# Statistical helpers — replace the broken zero-win flag logic
# ---------------------------------------------------------------------------

def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact binomial CI via the beta distribution. Falls back to Wilson
    score interval if scipy is unavailable."""
    if n == 0:
        return (0.0, 1.0)
    try:
        from scipy.stats import beta
        low  = beta.ppf(alpha/2,     k,     n-k+1) if k > 0 else 0.0
        high = beta.ppf(1 - alpha/2, k+1,   n-k)   if k < n else 1.0
        return (float(low), float(high))
    except ImportError:
        z = 1.96
        p_hat = k / n
        denom = 1 + z*z/n
        center = (p_hat + z*z/(2*n)) / denom
        margin = z * math.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))


def _binomial_pvalue_two_sided(k: int, n: int, p: float) -> float:
    if n == 0 or p <= 0 or p >= 1:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p, alternative='two-sided').pvalue)
    except ImportError:
        mean = n * p
        sd = math.sqrt(n * p * (1 - p))
        if sd == 0:
            return 1.0
        z = (k - mean) / sd
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


# ---------------------------------------------------------------------------
# Tear sheets — sample-size honest version
# ---------------------------------------------------------------------------

def tear_sheet_by_pool(ledger):
    bets = ledger[ledger["stake"] > 0].copy()
    if bets.empty:
        return pd.DataFrame()

    rows = []
    for pool, g in bets.groupby("pool"):
        n = len(g); wins = int(g["won"].sum())
        staked = float(g["stake"].sum())
        ev_dollars = float(g["pnl_expected"].sum())
        rv_dollars = float(g["pnl_realised"].sum())
        p_shrunk_w = float((g["p_model_shrunk"] * g["stake"]).sum() / staked)
        hit_rate_w = float((g["won"] * g["stake"]).sum() / staked)
        expected_wins = float(g["p_model_shrunk"].sum())

        ci_lo, ci_hi = _clopper_pearson_ci(wins, n)
        avg_p = float(g["p_model_shrunk"].mean())
        pval = _binomial_pvalue_two_sided(wins, n, avg_p)

        if n < MIN_TICKETS_FOR_VERDICT or expected_wins < MIN_EXPECTED_WINS_FOR_VERDICT:
            verdict = "INSUFFICIENT_SAMPLE"
            cal_slip = float('nan')
        else:
            cal_slip = hit_rate_w / p_shrunk_w if p_shrunk_w > 0 else float('nan')
            if pval > 0.10:
                verdict = "consistent_with_calibrated"
            elif cal_slip < 0.7:
                verdict = "model_overstates_>30pct"
            elif cal_slip < 0.85:
                verdict = "model_overstates_15-30pct"
            elif cal_slip > 1.15:
                verdict = "model_understates_>15pct"
            else:
                verdict = "consistent_with_calibrated"

        rows.append({
            "pool": pool, "tickets": n, "wins": wins,
            "expected_wins": expected_wins,
            "hit_rate": wins / n,
            "hit_rate_ci_low": ci_lo, "hit_rate_ci_high": ci_hi,
            "binomial_pvalue_two_sided": pval,
            "staked": staked,
            "ev_dollars": ev_dollars, "rv_dollars": rv_dollars,
            "ev_pct_of_stake": ev_dollars / staked,
            "rv_pct_of_stake": rv_dollars / staked,
            "p_shrunk_stake_wtd": p_shrunk_w,
            "hit_rate_stake_wtd": hit_rate_w,
            "calibration_slip":  cal_slip,
            "verdict": verdict,
        })
    return pd.DataFrame(rows).set_index("pool")


def tear_sheet_by_decile(ledger, n_buckets=10):
    bets = ledger[ledger["stake"] > 0].copy()
    if bets.empty:
        return pd.DataFrame()
    bets["odds_decile"] = pd.qcut(
        bets["odds_close"], q=min(n_buckets, len(bets)),
        labels=False, duplicates="drop",
    )

    def _agg(g):
        n = len(g); wins = int(g["won"].sum())
        staked = float(g["stake"].sum())
        expected_wins = float(g["p_model_shrunk"].sum())
        ci_lo, ci_hi = _clopper_pearson_ci(wins, n)
        return pd.Series({
            "tickets": n, "wins": wins,
            "expected_wins": expected_wins,
            "hit_rate_ci_low": ci_lo, "hit_rate_ci_high": ci_hi,
            "odds_min": float(g["odds_close"].min()),
            "odds_max": float(g["odds_close"].max()),
            "staked": staked,
            "ev_dollars": float(g["pnl_expected"].sum()),
            "rv_dollars": float(g["pnl_realised"].sum()),
            "ev_pct": float(g["pnl_expected"].sum() / staked),
            "rv_pct": float(g["pnl_realised"].sum() / staked),
        })
    return (bets.groupby(["pool", "odds_decile"], group_keys=False)
                .apply(_agg, include_groups=False))


def write_oof_for_theta(ledger: pd.DataFrame, p_model_df: pd.DataFrame,
                        results_df: pd.DataFrame, out_path: str):
    """
    Produces an ensemble_oof_results.csv-format dump from p_model + results.
    This closes the loop: backtest -> OOF -> theta refit -> next backtest.
    """
    res_lookup = results_df.set_index("race_id")[
        ["pos1_horse_no", "pos2_horse_no", "pos3_horse_no"]
    ].to_dict("index")

    rows = []
    for race_id, sub in p_model_df.groupby("race_id"):
        if race_id not in res_lookup:
            continue
        r = res_lookup[race_id]
        for _, e in sub.iterrows():
            hno = int(e["horse_no"])
            if hno == r["pos1_horse_no"]: fp = 1
            elif hno == r["pos2_horse_no"]: fp = 2
            elif hno == r["pos3_horse_no"]: fp = 3
            else: fp = 99
            rows.append({
                "race_id": race_id,
                "horse_code": e.get("horse_code", f"H{hno}"),
                "horse_no":   hno,
                "finish_position": fp,
                "P_model": float(e["p_model"]),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Wrote OOF dump for theta refit -> %s", out_path)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _parse_combo_string(combo_str, pool):
    parts = [int(p) for p in str(combo_str).replace(",", "-").split("-") if p.strip()]
    if pool in ("QIN", "QPL") and len(parts) != 2:
        raise ValueError(f"Bad {pool} combo: {combo_str!r}")
    if pool == "TRI" and len(parts) != 3:
        raise ValueError(f"Bad TRI combo: {combo_str!r}")
    return tuple(sorted(parts))


def _load_closing_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"race_id", "pool_type", "combination", "POST_STOP_SELL"}
    if not required.issubset(df.columns):
        raise ValueError(f"closing_odds CSV must have columns {required}")
    return df.rename(columns={"POST_STOP_SELL": "odds"})[
        ["race_id", "pool_type", "combination", "odds"]
    ]


def _load_closing_from_dividends(race_ids, dividend_unit_base):
    """
    Pulls from race_dividends. HKJC dividends are typically quoted on a
    "$X dividend per $10 staked" basis -- decimal-odds = dividend / unit_base.
    Set --dividend_unit_base 10 for HK convention (the default), 1 if your
    scraper already converted to per-$1 odds.
    """
    from sqlalchemy import create_engine, text
    from hkjc_engine.config import DB_URL

    engine = create_engine(DB_URL)
    q = text("""
        SELECT race_id, pool, combination, dividend
        FROM race_dividends
        WHERE race_id IN :rids
          AND pool IN ('QIN', 'QUINELLA', 'QPL', 'QUINELLA PLACE',
                       'TRI', 'TRIO', 'TIERCE')
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"rids": tuple(race_ids)})
    if df.empty:
        log.warning("race_dividends returned 0 rows for %d race_ids", len(race_ids))
        return df

    pool_map = {
        "QIN": "QIN", "QUINELLA": "QIN",
        "QPL": "QPL", "QUINELLA PLACE": "QPL",
        "TRI": "TRI", "TRIO": "TRI", "TIERCE": "TRI",
    }
    df["pool_type"] = df["pool"].str.upper().map(pool_map)
    df = df.dropna(subset=["pool_type"])
    df["odds"] = df["dividend"].astype(float) / dividend_unit_base
    return df[["race_id", "pool_type", "combination", "odds"]]


def load_inputs(closing_df, results_csv, p_model_csv, race_meta=None):
    results = pd.read_csv(results_csv)
    p_model = pd.read_csv(p_model_csv)

    required_res = {"race_id", "pos1_horse_no", "pos2_horse_no", "pos3_horse_no"}
    if not required_res.issubset(results.columns):
        raise ValueError(f"results CSV must have columns {required_res}")
    required_p = {"race_id", "horse_no", "p_model", "win_odds_close"}
    if not required_p.issubset(p_model.columns):
        raise ValueError(f"p_model CSV must have columns {required_p}")

    races = []
    meta_lookup = (race_meta.set_index("race_id").to_dict("index")
                   if race_meta is not None else {})

    for race_id, sub in p_model.groupby("race_id"):
        sub = sub.sort_values("horse_no").reset_index(drop=True)
        try:
            r = results.loc[results["race_id"] == race_id].iloc[0]
        except IndexError:
            log.warning("No result for race %s — skipped", race_id)
            continue
        result = (int(r.pos1_horse_no), int(r.pos2_horse_no), int(r.pos3_horse_no))

        odds_dict = {"QIN": {}, "QPL": {}, "TRI": {}}
        for _, row in closing_df.loc[closing_df["race_id"] == race_id].iterrows():
            pool = row["pool_type"]
            if pool not in odds_dict:
                continue
            try:
                horses = _parse_combo_string(row["combination"], pool)
                key = "-".join(str(h) for h in horses)
                odds = float(row["odds"])
                if odds >= 999.0 or np.isnan(odds) or odds <= 1.0:
                    continue
                odds_dict[pool][key] = odds
            except (ValueError, TypeError) as e:
                log.debug("Skipping bad row in %s: %s (%s)",
                          race_id, row.to_dict(), e)

        meta = meta_lookup.get(race_id, {})
        n_runners = len(sub)
        races.append(RaceReplay(
            race_id=race_id,
            horse_nos=sub["horse_no"].astype(int).tolist(),
            p_model=sub["p_model"].to_numpy(dtype=float),
            win_odds_close=sub["win_odds_close"].to_numpy(dtype=float),
            closing_exotic_odds=odds_dict,
            result=result,
            distance=meta.get("distance"),
            field_size=int(meta.get("field_size", n_runners)),
            race_class=meta.get("race_class"),
        ))
    log.info("Loaded %d races", len(races))
    return races


def _load_race_meta(race_ids):
    """Pull (distance, race_class) from `races` table for contextual theta."""
    try:
        from sqlalchemy import create_engine, text
        from hkjc_engine.config import DB_URL
    except ImportError:
        return None
    engine = create_engine(DB_URL)
    q = text("""
        SELECT race_id, distance, race_class
        FROM races
        WHERE race_id IN :rids
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"rids": tuple(race_ids)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closing_odds", default=None,
                    help="Live-archive CSV (claude_exotics_diagnostic.csv format)")
    ap.add_argument("--closing_source", choices=["csv", "dividends"], default="csv",
                    help="csv = read --closing_odds; dividends = pull race_dividends from DB")
    ap.add_argument("--dividend_unit_base", type=float, default=10.0,
                    help="HKJC dividends are quoted per $10 stake; set 1 if already per-$1")
    ap.add_argument("--results", required=True)
    ap.add_argument("--p_model", required=True)
    ap.add_argument("--bankroll",  type=float, default=100_000.0)
    ap.add_argument("--shrinkage", type=float, default=SHRINKAGE_DEFAULT)
    ap.add_argument("--theta_json", default=None,
                    help="Optional contextual theta from theta_optimizer.py --stratify")
    ap.add_argument("--out_dir", default="./backtest_out")
    args = ap.parse_args()

    if args.closing_source == "csv" and not args.closing_odds:
        ap.error("--closing_source csv requires --closing_odds PATH")

    os.makedirs(args.out_dir, exist_ok=True)

    p_model_df = pd.read_csv(args.p_model)
    race_ids = sorted(p_model_df["race_id"].unique().tolist())

    if args.closing_source == "csv":
        closing_df = _load_closing_from_csv(args.closing_odds)
    else:
        closing_df = _load_closing_from_dividends(race_ids, args.dividend_unit_base)
        if closing_df.empty:
            log.error("No dividends found for the given p_model races. "
                      "Check that scraper_dividends.py has populated race_dividends.")
            return

    theta_table = None
    race_meta = None
    if args.theta_json:
        with open(args.theta_json) as f:
            theta_table = json.load(f)
        race_meta = _load_race_meta(race_ids)
        if race_meta is None:
            log.warning("Could not load race metadata; theta will fall back to global.")

    races = load_inputs(closing_df, args.results, args.p_model, race_meta=race_meta)
    if not races:
        log.error("No races loaded — exiting.")
        return

    ledgers = []
    for race in races:
        led = replay_race(race, args.bankroll, args.shrinkage, theta_table)
        if not led.empty:
            ledgers.append(led)
    if not ledgers:
        log.error("No tickets produced — check inputs.")
        return

    full_ledger = pd.concat(ledgers, ignore_index=True)
    full_ledger.to_csv(os.path.join(args.out_dir, "ticket_ledger.csv"), index=False)

    pool_sheet   = tear_sheet_by_pool(full_ledger)
    decile_sheet = tear_sheet_by_decile(full_ledger)
    pool_sheet.to_csv(os.path.join(args.out_dir, "tear_sheet_by_pool.csv"))
    decile_sheet.to_csv(os.path.join(args.out_dir, "tear_sheet_by_decile.csv"))

    write_oof_for_theta(
        full_ledger, p_model_df, pd.read_csv(args.results),
        os.path.join(args.out_dir, "oof_for_theta.csv"),
    )

    bets = full_ledger[full_ledger["stake"] > 0]
    print("\n" + "=" * 78)
    print(f"EXOTICS BACKTEST  |  bankroll=${args.bankroll:,.0f}  shrinkage={args.shrinkage}")
    print(f"closing_source={args.closing_source}  "
          + (f"theta=contextual({args.theta_json})"
             if theta_table else f"theta=static({THETA_2_DEFAULT}, {THETA_3_DEFAULT})"))
    print(f"races={len(races)}  tickets_placed={len(bets)}  "
          f"total_staked=${bets['stake'].sum():,.0f}")
    print(f"NET PnL: ${bets['pnl_realised'].sum():+,.0f}   "
          f"(EV under model: ${bets['pnl_expected'].sum():+,.0f})")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:,.4f}".format,
                           "display.max_columns", None, "display.width", 220):
        print("\n--- Tear sheet by pool ---")
        print(pool_sheet)
        print("\n--- Tear sheet by closing-odds decile ---")
        print(decile_sheet)

    print("\n--- Verdicts ---")
    for pool, row in pool_sheet.iterrows():
        n = int(row["tickets"]); wins = int(row["wins"])
        ew = row["expected_wins"]; pval = row["binomial_pvalue_two_sided"]
        ci_lo = row["hit_rate_ci_low"]; ci_hi = row["hit_rate_ci_high"]
        verdict = row["verdict"]
        print(f"  {pool}: n={n:<4d}  wins={wins:<3d}  E[wins]={ew:.2f}  "
              f"95% CI hit_rate=[{ci_lo:.3f}, {ci_hi:.3f}]  "
              f"binomial_p={pval:.3f}  -> {verdict}")
        if verdict == "INSUFFICIENT_SAMPLE":
            print(f"      ^ Need n>={MIN_TICKETS_FOR_VERDICT} AND "
                  f"E[wins]>={MIN_EXPECTED_WINS_FOR_VERDICT:.0f} "
                  "for a valid calibration claim.")

    print("\nNext step: feed oof_for_theta.csv into theta_optimizer.py to refit "
          "(global or stratified). Then re-run this backtester with the "
          "resulting --theta_json to A/B contextual vs. static theta.")


if __name__ == "__main__":
    main()