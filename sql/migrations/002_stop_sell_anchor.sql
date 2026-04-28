-- Migration: 002_stop_sell_anchor.sql
-- ===================================================================
-- Purpose
-- -------
-- Materialised view of point-in-time STOP_SELL odds per (race_id,
-- pool_type, combination), used by the drift-aware refactor so the
-- training and backtesting pipelines can attach the anchor without
-- re-running the per-race CTE scan every time.
--
-- The window-function CTE used by stop_sell_loader.py is correct but
-- expensive on full-history backtests (N races x ~14 horses x ~30
-- snapshots each = ~50M rows scanned). This view collapses it to ~1M
-- rows that are indexed for direct (race_id, pool_type, combination)
-- lookup, and the trainer / backtester just LEFT JOIN against it.
--
-- Refresh policy
-- --------------
-- Refresh nightly after odds_archiver writes the day's FINAL snapshot:
--
--     REFRESH MATERIALIZED VIEW CONCURRENTLY stop_sell_anchor;
--
-- CONCURRENTLY needs a UNIQUE index, which we provide. Plan-C onwards
-- the refresh takes ~5-15 seconds; full historical rebuild ~30 minutes.
--
-- Safe to run multiple times (DROP IF EXISTS guards the view).
-- ===================================================================

-- ----- 1. Materialised view ------------------------------------------------

DROP MATERIALIZED VIEW IF EXISTS stop_sell_anchor;

CREATE MATERIALIZED VIEW stop_sell_anchor AS
WITH ranked AS (
    SELECT
        race_id,
        pool_type,
        combination,
        odds AS stop_sell_odds,
        phase,
        seconds_vs_stop_sell,
        timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY race_id, pool_type, combination
            ORDER BY
                -- Prefer first POST_STOP_SELL (the actual closing quote)
                CASE
                    WHEN phase = 'POST_STOP_SELL'
                         AND seconds_vs_stop_sell >= 0 THEN 0
                    -- Then last PRE_STOP_SELL (last observable)
                    WHEN phase = 'PRE_STOP_SELL'        THEN 1
                    -- Then any other POST_STOP_SELL row (e.g. negative offset)
                    WHEN phase = 'POST_STOP_SELL'        THEN 2
                    ELSE 3
                END ASC,
                ABS(COALESCE(seconds_vs_stop_sell, 1e9)) ASC,
                timestamp DESC
        ) AS rn
    FROM live_odds_history
    WHERE odds > 1.0
)
SELECT
    race_id, pool_type, combination, stop_sell_odds,
    phase, seconds_vs_stop_sell, timestamp
FROM ranked
WHERE rn = 1
WITH NO DATA;

-- Required for REFRESH ... CONCURRENTLY
CREATE UNIQUE INDEX ix_stop_sell_anchor_pk
    ON stop_sell_anchor (race_id, pool_type, combination);

-- Per-pool joins
CREATE INDEX ix_stop_sell_anchor_pool
    ON stop_sell_anchor (pool_type, race_id);

-- WIN-only join is the hot path; partial index is ~30% the size of the
-- per-pool index above and is hit by trainer_residual / ensemble.
CREATE INDEX ix_stop_sell_anchor_win
    ON stop_sell_anchor (race_id, combination)
    WHERE pool_type = 'WIN';

-- Initial population
REFRESH MATERIALIZED VIEW stop_sell_anchor;


-- ----- 2. Coverage diagnostic view ----------------------------------------
-- A small companion view so we can monitor live_odds_history coverage by
-- race_date without a full table scan.

CREATE OR REPLACE VIEW stop_sell_coverage AS
SELECT
    r.race_date::date AS race_date,
    COUNT(DISTINCT r.race_id) AS n_races_total,
    COUNT(DISTINCT a.race_id) FILTER (WHERE a.pool_type = 'WIN')
        AS n_races_with_win_anchor,
    COUNT(DISTINCT a.race_id) FILTER (
        WHERE a.pool_type = 'WIN' AND a.phase = 'POST_STOP_SELL'
    ) AS n_races_with_post_stop_sell
FROM races r
LEFT JOIN stop_sell_anchor a ON r.race_id = a.race_id
GROUP BY r.race_date
ORDER BY race_date DESC;
