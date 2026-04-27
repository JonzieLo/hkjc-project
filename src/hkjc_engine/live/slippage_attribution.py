"""
slippage_attribution.py
=======================

Joins `placed_bets` to the nearest-in-time row in `snapshot_recommendations`
for the same (race_id, pool, combination), producing a per-ticket attribution
of where PnL came from:

    bet_ev_at_placement   = ev_pct from the recommendation snapshot closest
                             in time to placed_at
    closing_odds_drift     = live_odds_at_snapshot (closest snapshot before
                             placement) vs. snapshot AT placement time
    realised_vs_expected   = realised_pnl - (stake * ev_at_placement)

Three diagnostics fall out:

  1. Per-pool aggregate of stake-weighted EV vs. realised PnL. If they
     diverge, your bot's EV estimates are systematically biased -- a model
     issue.

  2. Per-ticket residual: realised - expected. The variance of this
     residual at the pool level is your noise floor; the mean is your
     systematic edge (or drag).

  3. Odds-at-placement vs. odds-at-close. If the bot logs the same
     recommendation at multiple snapshots and you bet partway through, the
     gap between the EV the bot was showing AT YOUR PLACEMENT TIME vs. the
     EV at close tells you slippage. If close-time EV is consistently lower
     than placement-time EV, you're getting picked off by late money.

Usage
-----
    # Build the view (idempotent: drops & recreates)
    python -m hkjc_engine.live.slippage_attribution --build-view

    # Generate report
    python -m hkjc_engine.live.slippage_attribution --report
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


VIEW_SQL = """
DROP VIEW IF EXISTS bet_attribution CASCADE;

CREATE VIEW bet_attribution AS
WITH ranked_snapshots AS (
    -- For each placed bet, find every snapshot row for the same
    -- (race, pool, combo) and compute time-distance to placement.
    SELECT
        b.ref_no,
        b.leg_index,
        b.race_id,
        b.pool,
        b.combination,
        b.placed_at,
        b.stake,
        b.is_banker,
        b.realised_dividend,
        b.realised_pnl,
        s.snapshot_id,
        s.snapshot_timestamp,
        s.live_odds_at_snapshot,
        s.fair_odds_model,
        s.ev_pct,
        s.p_model_shrunk,
        s.stake_recommended,
        EXTRACT(EPOCH FROM (b.placed_at - s.snapshot_timestamp)) AS secs_diff,
        ABS(EXTRACT(EPOCH FROM (b.placed_at - s.snapshot_timestamp))) AS abs_secs_diff,
        ROW_NUMBER() OVER (
            PARTITION BY b.ref_no, b.leg_index
            ORDER BY ABS(EXTRACT(EPOCH FROM (b.placed_at - s.snapshot_timestamp))) ASC
        ) AS rn
    FROM placed_bets b
    LEFT JOIN snapshot_recommendations s
        ON s.race_id = b.race_id
        AND s.pool = b.pool
        AND s.combination = b.combination
)
SELECT
    ref_no, leg_index, race_id, pool, combination,
    placed_at, stake, is_banker,
    realised_dividend, realised_pnl,
    -- Closest-in-time snapshot's recommendation
    snapshot_id            AS attrib_snapshot_id,
    snapshot_timestamp     AS attrib_snapshot_ts,
    secs_diff              AS attrib_secs_offset,
    live_odds_at_snapshot  AS attrib_odds_seen,
    ev_pct                 AS attrib_ev_pct,
    p_model_shrunk         AS attrib_p_shrunk,
    stake_recommended      AS attrib_stake_recommended,
    -- Expected PnL at placement (using closest snapshot's EV)
    stake * COALESCE(ev_pct, 0) AS expected_pnl,
    -- Residual (NULL if not yet settled)
    CASE WHEN realised_pnl IS NOT NULL
         THEN realised_pnl - (stake * COALESCE(ev_pct, 0))
         ELSE NULL
    END AS pnl_residual
FROM ranked_snapshots
WHERE rn = 1 OR rn IS NULL;
"""


CLOSING_VS_PLACEMENT_VIEW = """
DROP VIEW IF EXISTS slippage_per_ticket CASCADE;

CREATE VIEW slippage_per_ticket AS
WITH placement_snapshot AS (
    -- Snapshot CLOSEST in time to actual placement
    SELECT DISTINCT ON (b.ref_no, b.leg_index)
        b.ref_no, b.leg_index, b.race_id, b.pool, b.combination,
        b.placed_at, b.stake,
        s.live_odds_at_snapshot AS odds_at_placement,
        s.ev_pct AS ev_at_placement
    FROM placed_bets b
    LEFT JOIN snapshot_recommendations s
        ON s.race_id = b.race_id AND s.pool = b.pool
        AND s.combination = b.combination
    ORDER BY b.ref_no, b.leg_index,
             ABS(EXTRACT(EPOCH FROM (b.placed_at - s.snapshot_timestamp)))
),
closing_snapshot AS (
    -- LAST snapshot for each (race, pool, combo) -- treat as closing odds
    SELECT DISTINCT ON (s.race_id, s.pool, s.combination)
        s.race_id, s.pool, s.combination,
        s.live_odds_at_snapshot AS odds_at_close,
        s.ev_pct AS ev_at_close,
        s.snapshot_timestamp AS closing_snapshot_ts
    FROM snapshot_recommendations s
    ORDER BY s.race_id, s.pool, s.combination,
             s.snapshot_timestamp DESC
)
SELECT
    p.ref_no, p.leg_index, p.race_id, p.pool, p.combination,
    p.placed_at, p.stake,
    p.odds_at_placement,
    c.odds_at_close,
    p.odds_at_placement - c.odds_at_close AS odds_drift,
    p.ev_at_placement,
    c.ev_at_close,
    p.ev_at_placement - c.ev_at_close AS ev_drift,
    c.closing_snapshot_ts
FROM placement_snapshot p
LEFT JOIN closing_snapshot c
    ON c.race_id = p.race_id AND c.pool = p.pool
    AND c.combination = p.combination;
"""


def build_views(db_url: str):
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text(VIEW_SQL))
        conn.execute(text(CLOSING_VS_PLACEMENT_VIEW))
    log.info("Created views: bet_attribution, slippage_per_ticket")


def report(db_url: str):
    engine = create_engine(db_url, pool_pre_ping=True)

    pool_summary = pd.read_sql(text("""
        SELECT
            pool,
            COUNT(*) AS tickets,
            SUM(CASE WHEN realised_dividend IS NOT NULL THEN 1 ELSE 0 END) AS wins,
            SUM(stake) AS staked,
            SUM(expected_pnl) AS expected_pnl,
            SUM(realised_pnl) AS realised_pnl,
            AVG(attrib_secs_offset) AS avg_attrib_offset_secs,
            COUNT(*) FILTER (WHERE attrib_snapshot_id IS NULL) AS unattributed
        FROM bet_attribution
        WHERE realised_pnl IS NOT NULL
        GROUP BY pool
        ORDER BY pool
    """), engine)

    print("\n" + "=" * 78)
    print("  PER-POOL ATTRIBUTION (only settled tickets)")
    print("=" * 78)
    if pool_summary.empty:
        print("  No settled tickets yet.")
    else:
        with pd.option_context("display.float_format", "{:,.2f}".format,
                               "display.width", 200):
            print(pool_summary.to_string(index=False))

    slip = pd.read_sql(text("""
        SELECT
            pool,
            COUNT(*) AS tickets,
            AVG(odds_drift) AS mean_odds_drift,
            STDDEV_POP(odds_drift) AS std_odds_drift,
            AVG(ev_drift) AS mean_ev_drift,
            AVG(odds_at_placement) AS mean_odds_placement,
            AVG(odds_at_close) AS mean_odds_close
        FROM slippage_per_ticket
        WHERE odds_at_placement IS NOT NULL AND odds_at_close IS NOT NULL
        GROUP BY pool
        ORDER BY pool
    """), engine)

    print("\n" + "=" * 78)
    print("  SLIPPAGE: PLACEMENT-TIME ODDS vs CLOSING-TIME ODDS")
    print("  (positive odds_drift = you bet at higher odds than close")
    print("                       = you got a BETTER price than close")
    print("   negative odds_drift = you bet at lower odds than close")
    print("                       = late money pushed your combo's odds OUT")
    print("                         AFTER you bet, you got picked off)")
    print("=" * 78)
    if slip.empty:
        print("  No slippage data yet -- need both placement-time and "
              "post-stop-sell snapshots.")
    else:
        with pd.option_context("display.float_format", "{:,.4f}".format,
                               "display.width", 200):
            print(slip.to_string(index=False))


if __name__ == "__main__":
    from hkjc_engine.config import DB_URL
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-view", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.build_view:
        build_views(DB_URL)
    if args.report:
        report(DB_URL)
    if not (args.build_view or args.report):
        ap.print_help()