"""
Point-in-time STOP_SELL odds loader.

Purpose
-------
The training pipeline used to anchor `base_margin` on `race_entries.win_odds`, which is the FINAL settled dividend — i.e. it includes the late-money syndicate drop the live bot CANNOT see at execution time.
This module extracts the odds that WERE observable at STOP_SELL from `live_odds_history` so the trainer/backtester can use a point-in-time correct anchor.

Resolution rules (per (race_id, pool_type, combination))
--------------------------------------------------------
1. PRIMARY:   first POST_STOP_SELL row (smallest seconds_vs_stop_sell >= 0). This is the odds quote at the instant sales actually closed.
2. FALLBACK:  last PRE_STOP_SELL row. Used when the archiver missed the POST_STOP_SELL window.
3. FAIL:      caller falls back to FINAL odds adjusted by drift.

Pre-Plan-C history (no `phase` markers) is treated as PRIMARY-missing / FALLBACK-missing — caller decides whether to drop or impute.

Performance
-----------
For full-history training queries, prefer the `stop_sell_anchor` materialised view in sql/migrations/002_stop_sell_anchor.sql, which is indexedfaster than LATERAL JOIN below for >100k races.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

_Q_WIN_ANCHOR = text("""
    WITH ranked AS (
        SELECT
            race_id,
            combination AS horse_no,
            odds,
            phase,
            timestamp,
            seconds_vs_stop_sell,
            ROW_NUMBER() OVER (
                PARTITION BY race_id, combination
                ORDER BY
                    -- POST_STOP_SELL preferred, smallest positive offset
                    CASE
                        WHEN phase = 'POST_STOP_SELL'
                             AND seconds_vs_stop_sell >= 0 THEN 0
                        WHEN phase = 'PRE_STOP_SELL'        THEN 1
                        WHEN phase = 'POST_STOP_SELL'        THEN 2
                        ELSE 3
                    END ASC,
                    ABS(COALESCE(seconds_vs_stop_sell, 1e9)) ASC,
                    timestamp DESC
            ) AS rn
        FROM live_odds_history
        WHERE pool_type = 'WIN'
          AND odds > 1.0
          AND race_id = ANY(:race_ids)
    )
    SELECT race_id, horse_no, odds AS stop_sell_odds, phase, seconds_vs_stop_sell
    FROM ranked
    WHERE rn = 1
""")

_Q_POOL_ANCHOR = text("""
    WITH ranked AS (
        SELECT
            race_id, pool_type, combination, odds, phase, seconds_vs_stop_sell,
            ROW_NUMBER() OVER (
                PARTITION BY race_id, pool_type, combination
                ORDER BY
                    CASE
                        WHEN phase = 'POST_STOP_SELL'
                             AND seconds_vs_stop_sell >= 0 THEN 0
                        WHEN phase = 'PRE_STOP_SELL'        THEN 1
                        WHEN phase = 'POST_STOP_SELL'        THEN 2
                        ELSE 3
                    END ASC,
                    ABS(COALESCE(seconds_vs_stop_sell, 1e9)) ASC,
                    timestamp DESC
            ) AS rn
        FROM live_odds_history
        WHERE pool_type = ANY(:pools)
          AND odds > 1.0
          AND race_id = ANY(:race_ids)
    )
    SELECT race_id, pool_type, combination, odds AS stop_sell_odds, phase
    FROM ranked
    WHERE rn = 1
""")


_Q_FINAL_ANCHOR = text("""
    SELECT DISTINCT ON (race_id, pool_type, combination)
        race_id, pool_type, combination, odds AS final_odds
    FROM live_odds_history
    WHERE phase = 'FINAL'
      AND odds > 1.0
      AND race_id = ANY(:race_ids)
    ORDER BY race_id, pool_type, combination, timestamp DESC
""")



def fetch_win_anchor(engine, race_ids: Iterable[str]) -> pd.DataFrame:
    race_ids = list(race_ids)
    if not race_ids:
        return pd.DataFrame(columns=['race_id', 'horse_no', 'stop_sell_odds',
                                     'phase', 'seconds_vs_stop_sell'])
    with engine.connect() as conn:
        df = pd.read_sql(_Q_WIN_ANCHOR, conn, params={'race_ids': race_ids})
    df['stop_sell_odds'] = df['stop_sell_odds'].astype(float)
    return df


def fetch_pool_anchor(engine,
                      race_ids: Iterable[str],
                      pools: Iterable[str] = ('QIN', 'QPL', 'TRI', 'PLA')
                      ) -> pd.DataFrame:
    """Return DataFrame[race_id, pool_type, combination, stop_sell_odds, phase]."""
    race_ids = list(race_ids)
    pools = list(pools)
    if not race_ids:
        return pd.DataFrame(columns=['race_id', 'pool_type', 'combination',
                                     'stop_sell_odds', 'phase'])
    with engine.connect() as conn:
        df = pd.read_sql(_Q_POOL_ANCHOR, conn, params={'race_ids': race_ids, 'pools': pools})
    df['stop_sell_odds'] = df['stop_sell_odds'].astype(float)
    return df


def fetch_final_anchor(engine,
                       race_ids: Iterable[str]) -> pd.DataFrame:
    """Return DataFrame[race_id, pool_type, combination, final_odds]."""
    race_ids = list(race_ids)
    if not race_ids:
        return pd.DataFrame(columns=['race_id', 'pool_type', 'combination',
                                     'final_odds'])
    with engine.connect() as conn:
        df = pd.read_sql(_Q_FINAL_ANCHOR, conn, params={'race_ids': race_ids})
    df['final_odds'] = df['final_odds'].astype(float)
    return df


def attach_win_anchor(df: pd.DataFrame,
                      engine,
                      odds_col: str = 'win_odds',
                      out_col: str = 'stop_sell_odds',
                      fallback: str = 'final_with_drift_adj',
                      drift_ratio_default: float = 1.005,
                      ) -> pd.DataFrame:
    """Attach `stop_sell_odds` to a per-(race_id, horse_no) DataFrame.

    Parameters
    ----------
    df         must have columns ['race_id', 'horse_no', odds_col].
    engine     SQLAlchemy engine.
    odds_col   FINAL odds column already in `df` (e.g. 'win_odds').
    out_col    Name of the column to write.
    fallback   How to fill rows with no live snapshot:
               - 'final': copy odds_col verbatim (zero drift assumption,
                         OK only if you trust pre-Plan-C history)
               - 'final_with_drift_adj': divide by `drift_ratio_default`,
                         the empirical mean R_final/R_stop_sell ratio.
                         This is the recommended default — it is unbiased
                         in expectation rather than implicitly assuming
                         zero drift.
               - 'drop': leave NaN; downstream is expected to dropna.
    drift_ratio_default
               Used only when fallback='final_with_drift_adj'. 1.005 is
               the empirical mean for WIN; raise to 1.025 if anchoring
               an exotic.

    Notes
    -----
    `horse_no` column needs to be a string in `df` to match the `combination`
    field of live_odds_history. Caller is responsible for casting.
    """
    if df.empty:
        df = df.copy()
        df[out_col] = np.nan
        return df

    race_ids = df['race_id'].astype(str).unique().tolist()
    anchor = fetch_win_anchor(engine, race_ids)

    df = df.copy()
    df['horse_no'] = df['horse_no'].astype(str)
    df = df.merge(
        anchor[['race_id', 'horse_no', 'stop_sell_odds']],
        on=['race_id', 'horse_no'], how='left',
    )

    miss = df['stop_sell_odds'].isna()
    n_miss = int(miss.sum())
    n_total = len(df)
    if n_miss > 0:
        # Partial coverage is a real data-integrity signal worth surfacing (e.g. one horse mis-mapped to live_odds_history). 
        # All-missing or all-present is routine — suppress to DEBUG to avoid log spam in pre-Plan-C training windows where every race lacks live coverage.
        # Re-enable with:
        #   logging.getLogger('hkjc_engine.data.stop_sell_loader').setLevel(logging.DEBUG)
        is_partial = 0 < n_miss < n_total
        log_fn = log.info if is_partial else log.debug
        log_fn("STOP_SELL anchor: %d / %d entries missing live snapshot; "
               "applying fallback=%s.", n_miss, n_total, fallback)
        if fallback == 'final':
            df.loc[miss, 'stop_sell_odds'] = df.loc[miss, odds_col].astype(float)
        elif fallback == 'final_with_drift_adj':
            df.loc[miss, 'stop_sell_odds'] = (
                df.loc[miss, odds_col].astype(float) / drift_ratio_default
            )
        elif fallback == 'drop':
            pass  # leave NaN
        else:
            raise ValueError(f"Unknown fallback: {fallback}")

    if out_col != 'stop_sell_odds':
        df = df.rename(columns={'stop_sell_odds': out_col})
    return df


def coverage_report(engine, race_ids: Iterable[str]) -> dict:
    """Sanity-check how many race_ids actually have live_odds_history rows.

    Returns
    -------
    dict with keys:
      - n_races_total       (input size)
      - n_races_with_win    (any phase)
      - n_races_with_post   (POST_STOP_SELL specifically)
      - phase_breakdown     {phase: count} across WIN-pool combinations
    """
    race_ids = list(race_ids)
    if not race_ids:
        return {'n_races_total': 0}
    q = text("""
        SELECT phase, COUNT(*) AS n,
               COUNT(DISTINCT race_id) AS n_races
        FROM live_odds_history
        WHERE pool_type = 'WIN'
          AND race_id = ANY(:race_ids)
        GROUP BY phase
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={'race_ids': race_ids})
    n_post = int(df.loc[df['phase'] == 'POST_STOP_SELL', 'n_races'].sum() or 0)
    n_any = int(df['n_races'].sum() or 0)
    return {
        'n_races_total': len(set(race_ids)),
        'n_races_with_win': n_any,
        'n_races_with_post': n_post,
        'phase_breakdown': dict(zip(df['phase'], df['n'].astype(int))),
    }


if __name__ == "__main__":
    # CLI: python -m hkjc_engine.data.stop_sell_loader 20240101 20240131
    import sys
    from hkjc_engine.config import DB_URL

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(message)s')
    if len(sys.argv) < 3:
        print("usage: stop_sell_loader.py START_DATE END_DATE")
        raise SystemExit(1)

    start, end = sys.argv[1], sys.argv[2]
    eng = create_engine(DB_URL)
    with eng.connect() as conn:
        rids = pd.read_sql(text("""
            SELECT race_id FROM races
            WHERE race_date >= :s AND race_date < :e
        """), conn, params={'s': start, 'e': end})['race_id'].tolist()
    rep = coverage_report(eng, rids)
    print(f"Coverage report for {start} -> {end}:")
    for k, v in rep.items():
        print(f"  {k:<25} {v}")