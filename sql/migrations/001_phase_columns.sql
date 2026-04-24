-- Migration: add phase tracking to live_odds_history
-- Safe to run multiple times (idempotent). Existing rows get NULL/UNKNOWN defaults.

ALTER TABLE live_odds_history
    ADD COLUMN IF NOT EXISTS phase VARCHAR(32) DEFAULT 'UNKNOWN',
    ADD COLUMN IF NOT EXISTS seconds_vs_stop_sell DOUBLE PRECISION DEFAULT NULL;

-- Helpful index for diagnostic queries that filter on phase
CREATE INDEX IF NOT EXISTS idx_live_odds_history_phase
    ON live_odds_history (race_id, pool_type, phase);

-- Helpful index for finding the last pre-STOP_SELL or final snapshot per race
CREATE INDEX IF NOT EXISTS idx_live_odds_history_race_ts
    ON live_odds_history (race_id, pool_type, combination, timestamp DESC);
