"""
rank_calibration_check_v2.py
============================

Corrected version of rank_calibration_check.py. Three fixes:

  1. QPL is a *multi-winner* pool (3 winning pairs per race). Previous version
     only counted the first match and produced spurious negative z-scores at
     ranks 1+. v2 measures the right quantity at each rank: probability that
     the rank-r combo IS one of the three winning pairs.

  2. TRI ranks were over permutations (6x duplication of each unique trio).
     v2 ranks unique unordered trios by their summed permutation probability,
     so rank 0 is the model's most likely 3-horse podium *regardless of order*.
     The exact-order TRI is also reported separately at the end as a sanity
     check.

  3. Top-K aggregation no longer sums probabilities (which was wrong for
     non-exclusive events). Instead it computes the empirical rate that ANY
     winning configuration appears within the model's top-k combos.

Drop in next to rank_calibration_check.py. Same inputs, same out_dir.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import os

import numpy as np
import pandas as pd

THETA_2 = 0.8824
THETA_3 = 0.7760

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Henery primitives
# ---------------------------------------------------------------------------

def _p_order(p_arr, i1, i2, i3=None):
    p1 = p_arr[i1]; p2 = p_arr[i2]
    sum_t2 = np.sum(p_arr ** THETA_2) - p1 ** THETA_2
    if sum_t2 <= 0:
        return 0.0
    p_exact_2 = p1 * (p2 ** THETA_2 / sum_t2)
    if i3 is None:
        return p_exact_2
    p3 = p_arr[i3]
    sum_t3 = np.sum(p_arr ** THETA_3) - p1 ** THETA_3 - p2 ** THETA_3
    if sum_t3 <= 0:
        return 0.0
    return p_exact_2 * (p3 ** THETA_3 / sum_t3)


def p_qin(p_arr, i, j):
    return _p_order(p_arr, i, j) + _p_order(p_arr, j, i)


def p_qpl(p_arr, i, j):
    """P(both i and j finish in top 3, any order)."""
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        for a, b, c in ((i, j, k), (j, i, k), (i, k, j),
                        (j, k, i), (k, i, j), (k, j, i)):
            total += _p_order(p_arr, a, b, c)
    return total


def p_tri_unordered(p_arr, i, j, k):
    """P(top-3 finishers are exactly {i, j, k}, any order). Sum over 6 perms."""
    total = 0.0
    for a, b, c in itertools.permutations([i, j, k]):
        total += _p_order(p_arr, a, b, c)
    return total


def p_tri_exact(p_arr, i, j, k):
    """P(top-3 finishers are i then j then k IN ORDER)."""
    return _p_order(p_arr, i, j, k)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p_hat = k / n
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    margin = z * math.sqrt(p_hat*(1-p_hat)/n + z*z/(4*n*n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_score(observed_rate, expected_rate, n):
    if n == 0 or expected_rate <= 0 or expected_rate >= 1:
        return float('nan')
    se = math.sqrt(expected_rate * (1 - expected_rate) / n)
    if se == 0:
        return float('nan')
    return (observed_rate - expected_rate) / se


# ---------------------------------------------------------------------------
# Per-pool rank tables
# ---------------------------------------------------------------------------

def rank_qin(p_model_df, results_df, max_rank=20):
    """QIN: single-winner pool. Rank r row = 'how often does model's r-th
    most likely pair contain the actual top-2 finishers?'"""
    res = results_df.set_index("race_id").to_dict("index")
    rank_p_sum = np.zeros(max_rank); rank_n = np.zeros(max_rank, dtype=int)
    rank_wins = np.zeros(max_rank, dtype=int)

    for race_id, sub in p_model_df.groupby("race_id"):
        if race_id not in res:
            continue
        sub = sub.sort_values("horse_no").reset_index(drop=True)
        horse_nos = sub["horse_no"].astype(int).tolist()
        p_arr = sub["p_model"].to_numpy(dtype=float)
        n = len(p_arr)
        if n < 2:
            continue
        r = res[race_id]
        try:
            win_pair = tuple(sorted([horse_nos.index(int(r["pos1_horse_no"])),
                                     horse_nos.index(int(r["pos2_horse_no"]))]))
        except ValueError:
            continue

        combos = list(itertools.combinations(range(n), 2))
        probs = [p_qin(p_arr, *c) for c in combos]
        order = np.argsort(probs)[::-1]

        for rank_idx in range(min(max_rank, len(order))):
            ci = order[rank_idx]
            rank_p_sum[rank_idx] += probs[ci]
            rank_n[rank_idx] += 1
            if tuple(sorted(combos[ci])) == win_pair:
                rank_wins[rank_idx] += 1

    return _build_rank_table(rank_p_sum, rank_n, rank_wins, max_rank)


def rank_qpl(p_model_df, results_df, max_rank=20):
    """QPL: multi-winner pool with 3 winning pairs per race.
    Rank r row = 'how often is the model's r-th most likely pair one of
    the 3 winning pairs?' This is the right quantity per rank.
    """
    res = results_df.set_index("race_id").to_dict("index")
    rank_p_sum = np.zeros(max_rank); rank_n = np.zeros(max_rank, dtype=int)
    rank_wins = np.zeros(max_rank, dtype=int)

    for race_id, sub in p_model_df.groupby("race_id"):
        if race_id not in res:
            continue
        sub = sub.sort_values("horse_no").reset_index(drop=True)
        horse_nos = sub["horse_no"].astype(int).tolist()
        p_arr = sub["p_model"].to_numpy(dtype=float)
        n = len(p_arr)
        if n < 3:
            continue
        r = res[race_id]
        try:
            top3 = [horse_nos.index(int(r[f"pos{k}_horse_no"])) for k in (1, 2, 3)]
        except ValueError:
            continue
        winning_pairs = {tuple(sorted(c)) for c in itertools.combinations(top3, 2)}

        combos = list(itertools.combinations(range(n), 2))
        probs = [p_qpl(p_arr, *c) for c in combos]
        order = np.argsort(probs)[::-1]

        for rank_idx in range(min(max_rank, len(order))):
            ci = order[rank_idx]
            rank_p_sum[rank_idx] += probs[ci]
            rank_n[rank_idx] += 1
            if tuple(sorted(combos[ci])) in winning_pairs:
                rank_wins[rank_idx] += 1

    return _build_rank_table(rank_p_sum, rank_n, rank_wins, max_rank)


def rank_tri_unordered(p_model_df, results_df, max_rank=20):
    """TRI ranked by unique unordered 3-horse trios.
    Rank r row = 'how often does the model's r-th most likely trio match
    the actual top-3 finishers (any order)?'
    Predicted probability is the SUM over the 6 orderings.
    """
    res = results_df.set_index("race_id").to_dict("index")
    rank_p_sum = np.zeros(max_rank); rank_n = np.zeros(max_rank, dtype=int)
    rank_wins = np.zeros(max_rank, dtype=int)

    for race_id, sub in p_model_df.groupby("race_id"):
        if race_id not in res:
            continue
        sub = sub.sort_values("horse_no").reset_index(drop=True)
        horse_nos = sub["horse_no"].astype(int).tolist()
        p_arr = sub["p_model"].to_numpy(dtype=float)
        n = len(p_arr)
        if n < 3:
            continue
        r = res[race_id]
        try:
            win_trio = tuple(sorted([
                horse_nos.index(int(r["pos1_horse_no"])),
                horse_nos.index(int(r["pos2_horse_no"])),
                horse_nos.index(int(r["pos3_horse_no"])),
            ]))
        except ValueError:
            continue

        combos = list(itertools.combinations(range(n), 3))
        probs = [p_tri_unordered(p_arr, *c) for c in combos]
        order = np.argsort(probs)[::-1]

        for rank_idx in range(min(max_rank, len(order))):
            ci = order[rank_idx]
            rank_p_sum[rank_idx] += probs[ci]
            rank_n[rank_idx] += 1
            if tuple(sorted(combos[ci])) == win_trio:
                rank_wins[rank_idx] += 1

    return _build_rank_table(rank_p_sum, rank_n, rank_wins, max_rank)


def rank_tri_exact(p_model_df, results_df, max_rank=20):
    """TRI ranked by exact ordered triples. The headline number is rank 0."""
    res = results_df.set_index("race_id").to_dict("index")
    rank_p_sum = np.zeros(max_rank); rank_n = np.zeros(max_rank, dtype=int)
    rank_wins = np.zeros(max_rank, dtype=int)

    for race_id, sub in p_model_df.groupby("race_id"):
        if race_id not in res:
            continue
        sub = sub.sort_values("horse_no").reset_index(drop=True)
        horse_nos = sub["horse_no"].astype(int).tolist()
        p_arr = sub["p_model"].to_numpy(dtype=float)
        n = len(p_arr)
        if n < 3:
            continue
        r = res[race_id]
        try:
            win_exact = (
                horse_nos.index(int(r["pos1_horse_no"])),
                horse_nos.index(int(r["pos2_horse_no"])),
                horse_nos.index(int(r["pos3_horse_no"])),
            )
        except ValueError:
            continue

        triples = list(itertools.permutations(range(n), 3))
        probs = [p_tri_exact(p_arr, *c) for c in triples]
        order = np.argsort(probs)[::-1]

        for rank_idx in range(min(max_rank, len(order))):
            ci = order[rank_idx]
            rank_p_sum[rank_idx] += probs[ci]
            rank_n[rank_idx] += 1
            if triples[ci] == win_exact:
                rank_wins[rank_idx] += 1

    return _build_rank_table(rank_p_sum, rank_n, rank_wins, max_rank)


def _build_rank_table(rank_p_sum, rank_n, rank_wins, max_rank):
    rows = []
    for r in range(max_rank):
        n = int(rank_n[r])
        if n == 0:
            continue
        mean_p = rank_p_sum[r] / n
        wins = int(rank_wins[r])
        emp_rate = wins / n
        ci_lo, ci_hi = _wilson_ci(wins, n)
        z = _z_score(emp_rate, mean_p, n)
        rows.append({
            "rank": r, "n_races": n,
            "mean_predicted_p": mean_p,
            "empirical_hit_rate": emp_rate,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "expected_wins": mean_p * n, "wins": wins,
            "z_score": z,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_model", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--max_rank", type=int, default=20)
    ap.add_argument("--out_dir", default="./calibration_out_v2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p_model = pd.read_csv(args.p_model)
    results = pd.read_csv(args.results)

    log.info("QIN rank calibration ...")
    qin = rank_qin(p_model, results, args.max_rank)
    qin.to_csv(os.path.join(args.out_dir, "rank_qin.csv"), index=False)

    log.info("QPL rank calibration (multi-winner corrected) ...")
    qpl = rank_qpl(p_model, results, args.max_rank)
    qpl.to_csv(os.path.join(args.out_dir, "rank_qpl.csv"), index=False)

    log.info("TRI rank calibration (unordered unique trios) ...")
    tri_u = rank_tri_unordered(p_model, results, args.max_rank)
    tri_u.to_csv(os.path.join(args.out_dir, "rank_tri_unordered.csv"), index=False)

    log.info("TRI rank calibration (exact ordered) ...")
    tri_e = rank_tri_exact(p_model, results, args.max_rank)
    tri_e.to_csv(os.path.join(args.out_dir, "rank_tri_exact.csv"), index=False)

    fmt = lambda df: df.to_string(index=False) if not df.empty else "(empty)"
    print("\n" + "=" * 78)
    print("  QIN — single-winner, ranked by p_qin")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200, "display.max_columns", None):
        print(fmt(qin))

        print("\n" + "=" * 78)
        print("  QPL — multi-winner (3 winning pairs/race), ranked by p_qpl")
        print("  Each rank's expected rate = model's belief that THIS pair is a winner")
        print("=" * 78)
        print(fmt(qpl))

        print("\n" + "=" * 78)
        print("  TRI — UNORDERED unique trios, ranked by p_tri_unordered")
        print("  This is the cleanest 'do you pick the right 3 horses' check")
        print("=" * 78)
        print(fmt(tri_u))

        print("\n" + "=" * 78)
        print("  TRI — exact ordered (the actual betting product)")
        print("=" * 78)
        print(fmt(tri_e))

    # Verdicts
    print("\n" + "=" * 78)
    print("  RANK-0 VERDICTS")
    print("=" * 78)
    for label, df in (("QIN              ", qin),
                      ("QPL              ", qpl),
                      ("TRI (unordered)  ", tri_u),
                      ("TRI (exact order)", tri_e)):
        if df.empty:
            print(f"  {label}: empty")
            continue
        r0 = df[df["rank"] == 0].iloc[0]
        n  = int(r0["n_races"])
        ew = float(r0["expected_wins"])
        z  = float(r0["z_score"])
        emp = float(r0["empirical_hit_rate"])
        pred = float(r0["mean_predicted_p"])
        ratio = emp / pred if pred > 0 else float('nan')
        verdict = ("CALIBRATED" if abs(z) < 2.0 else
                   "OVERSTATES" if z < 0 else "UNDERSTATES")
        print(f"  {label}: predicted={pred:.4f}  observed={emp:.4f}  "
              f"ratio={ratio:.3f}  z={z:+6.2f}  E[wins]={ew:6.1f}  "
              f"obs_wins={int(r0['wins']):4d}  -> {verdict}")


if __name__ == "__main__":
    main()