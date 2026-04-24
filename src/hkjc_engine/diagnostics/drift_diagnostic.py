from hkjc_engine.config import DB_URL
"""
Drift diagnostic — Option B (scheduled-race-time anchor).

Uses the scheduled race start time from the `races` table as a proxy for STOP_SELL time, since explicit stop_sell_time markers only exist for future races (Plan B). 
This is intentionally crude — we're not measuring drift to the second.

For each (race_id, pool_type, combination):
    1. Find the latest snapshot at or before scheduled jump → "T0_odds"
    2. Find the latest snapshot of all → "final_odds"
    3. Compute drift = final_odds / T0_odds
    4. For winners vs losers separately, summarize the drift distribution (adverse selection shows up as asymmetric drift by outcome)

Then for WIN-pool entries only:
    5. Backtest-style EV decay: if a model prob p were applied to both T0_odds and final_odds, how much edge evaporated? 
    (requires P_model in context — we use market-implied P as a proxy since we don't have historical model outputs for raw-odds-only runs)

Output: plots + printed summary
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from pathlib import Path

# DB_URL loaded from hkjc_engine.config
OUT_DIR = Path("drift_diagnostic_out")
OUT_DIR.mkdir(exist_ok=True)


def load_paired_snapshots(engine):
    """
    For each (race_id, pool_type, combination), extract:
      - t0_odds: last snapshot at/before scheduled jump
      - final_odds: last snapshot overall
      - race metadata (scheduled time, winner info)

    Returns a single wide DataFrame, one row per combination.
    """
    query = text("""
        WITH race_meta AS (
            -- Scheduled jump time per race. 
            -- Since we don't have the scheduled HH:MM stored, we use the earliest and latest snapshot timestamps to derive a window and treat the MAX timestamp as "settled" and the median-of-tight-cluster as T0.
            SELECT
                race_id,
                MIN(timestamp) AS first_snap,
                MAX(timestamp) AS last_snap
            FROM live_odds_history
            GROUP BY race_id
        ),
        -- T0 proxy: the snapshot taken closest to ~90 seconds before the last snapshot. This is a rough "pre-settlement" anchor that matches the typical 1-3 minute late-money window.
        t0_anchor AS (
            SELECT
                h.race_id,
                h.pool_type,
                h.combination,
                h.odds AS t0_odds,
                h.timestamp AS t0_timestamp
            FROM live_odds_history h
            JOIN race_meta m ON h.race_id = m.race_id
            JOIN LATERAL (
                SELECT timestamp
                FROM live_odds_history h2
                WHERE h2.race_id = h.race_id
                  AND h2.pool_type = h.pool_type
                  AND h2.combination = h.combination
                  AND h2.timestamp <= m.last_snap - INTERVAL '90 seconds'
                ORDER BY h2.timestamp DESC
                LIMIT 1
            ) latest_pre ON latest_pre.timestamp = h.timestamp
        ),
        final_anchor AS (
            SELECT DISTINCT ON (h.race_id, h.pool_type, h.combination)
                h.race_id,
                h.pool_type,
                h.combination,
                h.odds AS final_odds,
                h.timestamp AS final_timestamp
            FROM live_odds_history h
            ORDER BY h.race_id, h.pool_type, h.combination, h.timestamp DESC
        )
        SELECT
            t0.race_id,
            t0.pool_type,
            t0.combination,
            t0.t0_odds,
            t0.t0_timestamp,
            f.final_odds,
            f.final_timestamp,
            EXTRACT(EPOCH FROM (f.final_timestamp - t0.t0_timestamp)) AS gap_seconds
        FROM t0_anchor t0
        JOIN final_anchor f
          ON t0.race_id = f.race_id
         AND t0.pool_type = f.pool_type
         AND t0.combination = f.combination
        WHERE t0.t0_odds > 1.0 AND f.final_odds > 1.0
    """)
    return pd.read_sql(query, engine)


def attach_winners(df, engine):
    """
    Join WIN-pool snapshots against race_dividends to mark winners.
    For WIN pool, 'combination' is a single horse number.
    """
    wins = pd.read_sql(text("""
        SELECT race_id, combination AS winner_combo
        FROM race_dividends
        WHERE pool = 'WIN'
    """), engine)
    win_set = set(zip(wins['race_id'], wins['winner_combo'].astype(str)))

    df = df.copy()
    df['is_winner'] = df.apply(
        lambda r: 1 if (r['race_id'], str(r['combination'])) in win_set else 0,
        axis=1
    )
    return df


def summarize_drift(df, label):
    drift = df['final_odds'] / df['t0_odds']
    print(f"\n=== {label} (n={len(df)}) ===")
    print(f"  mean drift:   {drift.mean():.4f}")
    print(f"  median drift: {drift.median():.4f}")
    print(f"  p10 / p90:    {drift.quantile(0.10):.4f} / {drift.quantile(0.90):.4f}")
    print(f"  std:          {drift.std():.4f}")
    print(f"  fraction > 1 (odds drifted UP):   {(drift > 1).mean():.2%}")
    print(f"  fraction < 1 (odds drifted DOWN): {(drift < 1).mean():.2%}")
    return drift


def plot_drift_distribution(df_win, out_path):
    drift = df_win['final_odds'] / df_win['t0_odds']
    winners = drift[df_win['is_winner'] == 1]
    losers  = drift[df_win['is_winner'] == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.5, 1.5, 41)
    ax.hist(losers,  bins=bins, alpha=0.5, label=f'Losers (n={len(losers)})',  color='#cf4b4b')
    ax.hist(winners, bins=bins, alpha=0.7, label=f'Winners (n={len(winners)})', color='#2e7d32')
    ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
    ax.axvline(losers.median(),  color='#cf4b4b', linewidth=2, linestyle=':',
               label=f'Loser median = {losers.median():.3f}')
    ax.axvline(winners.median(), color='#2e7d32', linewidth=2, linestyle=':',
               label=f'Winner median = {winners.median():.3f}')
    ax.set_xlabel("final_odds / T0_odds")
    ax.set_ylabel("count")
    ax.set_title("Late-Money Drift Distribution — WIN Pool\n"
                 "If winners drift DOWN (ratio < 1) more than losers, "
                 "you have adverse selection.")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  → saved {out_path}")


def ev_decay_analysis(df_win):
    """
    Pretend you were using market-implied probs as your model (for diagnostic symmetry, since we don't have historical P_model).
    Quantify: of all combinations that were +EV at T0 (implied prob vs T0 odds > 1 after takeout), how many remained +EV at final odds?
    """
    # A horse with market-implied p = 1/T0_odds has exactly zero edge at T0 odds. So this won't show "model edge" — it shows how much the crowd's consensus drifted. Useful as a floor for the magnitude of effective drift.
    df = df_win.copy()
    df['assumed_model_p'] = (1.0 / df['t0_odds']) * 1.03
    df['ev_at_t0']    = df['assumed_model_p'] * df['t0_odds']    - 1.0
    df['ev_at_final'] = df['assumed_model_p'] * df['final_odds'] - 1.0

    pos_at_t0 = df[df['ev_at_t0'] > 0]
    still_pos = pos_at_t0[pos_at_t0['ev_at_final'] > 0]

    print(f"\n=== Simulated EV Decay (assumed constant +3% model edge) ===")
    print(f"  Bets +EV at T0:        {len(pos_at_t0)}")
    print(f"  Still +EV at final:    {len(still_pos)}  "
          f"({len(still_pos)/max(len(pos_at_t0),1):.1%})")
    if len(pos_at_t0) > 0:
        print(f"  Mean EV at T0:         {pos_at_t0['ev_at_t0'].mean():+.4f}")
        print(f"  Mean EV at final:      {pos_at_t0['ev_at_final'].mean():+.4f}")
        print(f"  Edge retained:         "
              f"{pos_at_t0['ev_at_final'].mean() / pos_at_t0['ev_at_t0'].mean():.2%}")

    if 'is_winner' in df.columns:
        print(f"\n  Split by outcome:")
        for label, sub in [('WINNERS', df[df['is_winner']==1]),
                           ('LOSERS',  df[df['is_winner']==0])]:
            d = sub['final_odds'] / sub['t0_odds']
            print(f"    {label:<8}: drift median = {d.median():.4f}  "
                  f"(mean = {d.mean():.4f}, n = {len(sub)})")


def main():
    engine = create_engine(DB_URL)

    print("Loading paired snapshots...")
    df = load_paired_snapshots(engine)
    print(f"  {len(df):,} (race × pool × combination) pairs loaded")
    print(f"  {df['race_id'].nunique()} distinct races")
    print(f"  pools: {sorted(df['pool_type'].unique())}")
    print(f"  mean gap from T0 → final: {df['gap_seconds'].mean():.0f}s  "
          f"(median {df['gap_seconds'].median():.0f}s)")

    for pool in sorted(df['pool_type'].unique()):
        sub = df[df['pool_type'] == pool]
        summarize_drift(sub, f"POOL = {pool}")

    df_win = df[df['pool_type'] == 'WIN'].copy()
    if not df_win.empty:
        df_win = attach_winners(df_win, engine)
        print(f"\nWIN pool: {df_win['is_winner'].sum()} winners, "
              f"{len(df_win) - df_win['is_winner'].sum()} losers matched")

        summarize_drift(df_win[df_win['is_winner'] == 1], "WINNERS ONLY")
        summarize_drift(df_win[df_win['is_winner'] == 0], "LOSERS ONLY")

        plot_drift_distribution(df_win, OUT_DIR / "drift_hist_win.png")
        ev_decay_analysis(df_win)

    df['abs_drift'] = (df['final_odds'] / df['t0_odds'] - 1).abs()
    print("\n=== Top 10 largest-drift combinations (sanity check) ===")
    top = df.nlargest(10, 'abs_drift')[
        ['race_id', 'pool_type', 'combination',
         't0_odds', 'final_odds', 'gap_seconds']
    ]
    print(top.to_string(index=False))

    print(f"\nDone. Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
