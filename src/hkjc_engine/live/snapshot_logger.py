"""
snapshot_logger.py
==================

Postgres-backed logger for closing-snapshot recommendations.
Writes one row per (race, pool, combination) the bot evaluated at snapshot time, so we can later join against race_dividends to compute realised PnL on every recommendation -- whether or not it was actually placed.

Schema
------
DDL is defined in this module and idempotent (CREATE IF NOT EXISTS). First call to SnapshotLogger() ensures the table and indexes exist.

    snapshot_recommendations
    ------------------------
    snapshot_id            UUID    -- groups rows from one closing-snapshot capture
    race_id                TEXT    -- e.g. '20260427_ST_03'
    race_off_time          TIMESTAMPTZ
    snapshot_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now()
    seconds_to_off         REAL
    pool                   TEXT NOT NULL  -- WIN | PLA | QIN | QPL | TRI
    combination            TEXT NOT NULL  -- canonical sorted-dash form
    live_odds_at_snapshot  REAL NOT NULL
    fair_odds_model        REAL
    ev_pct                 REAL
    stake_recommended      REAL NOT NULL DEFAULT 0
    stake_placed           REAL    -- NULL until reconciled with what you actually bet
    p_model_raw            REAL
    p_model_shrunk         REAL
    theta_2_used           REAL
    theta_3_used           REAL
    shrinkage_used         REAL
    bankroll_at_snapshot   REAL
    config_hash            TEXT    -- hash of full config for traceability
    PRIMARY KEY (snapshot_id, pool, combination)

    Indexes:
      ON race_id              (for joining to dividends/results)
      ON (pool, ev_pct DESC)  (for slicing by pool with EV-ordering)
      ON snapshot_timestamp   (for time-range queries)

Usage
-----
    from hkjc_engine.live.snapshot_logger import SnapshotLogger
    from hkjc_engine.config import DB_URL

    logger = SnapshotLogger(DB_URL)  # creates table on first use
    logger.log_snapshot(
        race_id='20260427_ST_03',
        race_off_time=race_off_dt,
        bankroll=current_bankroll,
        recommendations=[
            {'pool': 'QIN', 'combination': '1-3', 'live_odds': 15.0,
             'p_raw': 0.085, 'p_shrunk': 0.072, 'stake': 860.0},
            ...
        ],
        config={'theta_2': 0.8824, 'theta_3': 0.7760, 'shrinkage': 0.85},
    )

    # After races settle:
    logger.reconcile()
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import create_engine, text


log = logging.getLogger(__name__)


DDL = """
CREATE TABLE IF NOT EXISTS snapshot_recommendations (
    snapshot_id            UUID         NOT NULL,
    race_id                TEXT         NOT NULL,
    race_off_time          TIMESTAMPTZ,
    snapshot_timestamp     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    seconds_to_off         REAL,
    pool                   TEXT         NOT NULL,
    combination            TEXT         NOT NULL,
    live_odds_at_snapshot  REAL         NOT NULL,
    fair_odds_model        REAL,
    ev_pct                 REAL,
    stake_recommended      REAL         NOT NULL DEFAULT 0,
    stake_placed           REAL,
    p_model_raw            REAL,
    p_model_shrunk         REAL,
    theta_2_used           REAL,
    theta_3_used           REAL,
    shrinkage_used         REAL,
    bankroll_at_snapshot   REAL,
    config_hash            TEXT,
    PRIMARY KEY (snapshot_id, pool, combination)
);

CREATE INDEX IF NOT EXISTS ix_snap_race
    ON snapshot_recommendations (race_id);
CREATE INDEX IF NOT EXISTS ix_snap_pool_ev
    ON snapshot_recommendations (pool, ev_pct DESC);
CREATE INDEX IF NOT EXISTS ix_snap_ts
    ON snapshot_recommendations (snapshot_timestamp);
"""


INSERT_SQL = text("""
    INSERT INTO snapshot_recommendations (
        snapshot_id, race_id, race_off_time, snapshot_timestamp,
        seconds_to_off, pool, combination, live_odds_at_snapshot,
        fair_odds_model, ev_pct, stake_recommended,
        p_model_raw, p_model_shrunk,
        theta_2_used, theta_3_used, shrinkage_used,
        bankroll_at_snapshot, config_hash
    ) VALUES (
        :snapshot_id, :race_id, :race_off_time, :snapshot_timestamp,
        :seconds_to_off, :pool, :combination, :live_odds_at_snapshot,
        :fair_odds_model, :ev_pct, :stake_recommended,
        :p_model_raw, :p_model_shrunk,
        :theta_2_used, :theta_3_used, :shrinkage_used,
        :bankroll_at_snapshot, :config_hash
    )
    ON CONFLICT (snapshot_id, pool, combination) DO NOTHING
""")


def _canonical_combo(combination) -> str:
    """Sort-and-dash-join, accepting comma- or dash-separated inputs."""
    s = str(combination).strip()
    if "," in s or "-" in s:
        parts = sorted(int(x) for x in s.replace(",", "-").split("-") if x.strip())
        return "-".join(str(x) for x in parts)
    return s


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


class SnapshotLogger:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, pool_pre_ping=True)
        self._ensure_schema()

    def _ensure_schema(self):
        with self.engine.begin() as conn:
            for stmt in DDL.strip().split(";"):
                if stmt.strip():
                    conn.execute(text(stmt))

    def log_snapshot(
        self,
        race_id: str,
        race_off_time: datetime,
        bankroll: float,
        recommendations: Iterable[dict],
        config: dict,
    ) -> str:
        """Append rows for one closing snapshot. Returns snapshot_id."""
        snapshot_id = str(uuid.uuid4())
        snapshot_ts = datetime.now(timezone.utc)
        secs_to_off = (
            (race_off_time - snapshot_ts).total_seconds()
            if race_off_time is not None else None
        )
        cfg_hash = _config_hash(config)

        rows = []
        for rec in recommendations:
            odds = float(rec["live_odds"])
            p_shrunk = float(rec["p_shrunk"])
            fair = (1.0 / p_shrunk) if p_shrunk > 0 else None
            ev = ((p_shrunk * odds - 1.0) if (odds > 1.0 and p_shrunk > 0)
                  else None)
            rows.append({
                "snapshot_id":           snapshot_id,
                "race_id":               race_id,
                "race_off_time":         race_off_time,
                "snapshot_timestamp":    snapshot_ts,
                "seconds_to_off":        secs_to_off,
                "pool":                  rec["pool"],
                "combination":           _canonical_combo(rec["combination"]),
                "live_odds_at_snapshot": odds,
                "fair_odds_model":       fair,
                "ev_pct":                ev,
                "stake_recommended":     float(rec.get("stake", 0.0)),
                "p_model_raw":           rec.get("p_raw"),
                "p_model_shrunk":        p_shrunk,
                "theta_2_used":          config.get("theta_2"),
                "theta_3_used":          config.get("theta_3"),
                "shrinkage_used":        config.get("shrinkage"),
                "bankroll_at_snapshot":  float(bankroll),
                "config_hash":           cfg_hash,
            })

        if not rows:
            return snapshot_id

        with self.engine.begin() as conn:
            conn.execute(INSERT_SQL, rows)
        log.info("Logged %d recommendations for race %s (snapshot=%s)",
                 len(rows), race_id, snapshot_id[:8])
        return snapshot_id

    def mark_placed(self, snapshot_id: str, pool: str, combination: str,
                    stake_placed: float):
        """Record what you actually staked (vs recommended). Lets you
        separate 'model edge' from 'execution drag' later."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE snapshot_recommendations
                SET    stake_placed = :stake
                WHERE  snapshot_id = :sid
                  AND  pool        = :pool
                  AND  combination = :combo
            """), {"stake": stake_placed, "sid": snapshot_id,
                   "pool": pool, "combo": _canonical_combo(combination)})

    def reconcile(self, dividend_unit_base: float = 10.0,
                  start_date: str | None = None) -> dict:
        """
        Compute realised PnL by joining snapshot_recommendations against
        race_dividends, returning summary stats per pool. The join happens
        in SQL because that's what the database is for.

        Returns a dict; query the table directly for row-level detail.
        """
        date_clause = ""
        params = {"unit_base": dividend_unit_base}
        if start_date:
            date_clause = "AND s.snapshot_timestamp >= :start"
            params["start"] = start_date

        q = text(f"""
            WITH races_with_divs AS (
                SELECT DISTINCT race_id FROM race_dividends
            ),
            normalized_divs AS (
                SELECT
                    race_id,
                    CASE upper(pool)
                        WHEN 'QUINELLA'       THEN 'QIN'
                        WHEN 'QUINELLA PLACE' THEN 'QPL'
                        WHEN 'TIERCE'         THEN 'TRI'
                        WHEN 'TRIO'           THEN 'TRI'
                        WHEN 'WINNER'         THEN 'WIN'
                        WHEN 'PLACE'          THEN 'PLA'
                        ELSE upper(pool)
                    END AS pool_norm,
                    array_to_string(
                        (SELECT array_agg(x::int ORDER BY x::int)
                         FROM unnest(string_to_array(
                             regexp_replace(combination, ',', '-', 'g'),
                             '-')) AS x WHERE x ~ '^[0-9]+$'),
                        '-'
                    ) AS combo_norm,
                    dividend / :unit_base AS realised_dividend
                FROM race_dividends
            )
            SELECT
                s.pool,
                COUNT(*)                                                 AS tickets,
                COUNT(*) FILTER (WHERE d.realised_dividend IS NOT NULL)  AS wins,
                SUM(s.stake_recommended)                                 AS rec_stake,
                SUM(CASE
                        WHEN d.realised_dividend IS NOT NULL
                            THEN s.stake_recommended * (d.realised_dividend - 1.0)
                        WHEN s.race_id IN (SELECT race_id FROM races_with_divs)
                            THEN -s.stake_recommended
                        ELSE 0
                    END)                                                 AS rec_pnl,
                SUM(COALESCE(s.stake_placed, 0))                         AS placed_stake,
                SUM(CASE
                        WHEN s.stake_placed IS NULL THEN 0
                        WHEN d.realised_dividend IS NOT NULL
                            THEN s.stake_placed * (d.realised_dividend - 1.0)
                        WHEN s.race_id IN (SELECT race_id FROM races_with_divs)
                            THEN -s.stake_placed
                        ELSE 0
                    END)                                                 AS placed_pnl
            FROM snapshot_recommendations s
            LEFT JOIN normalized_divs d
                   ON d.race_id    = s.race_id
                  AND d.pool_norm  = s.pool
                  AND d.combo_norm = s.combination
            WHERE s.stake_recommended > 0
              {date_clause}
            GROUP BY s.pool
            ORDER BY s.pool
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(q, params).mappings().all()

        if not rows:
            log.info("No staked recommendations to reconcile yet.")
            return {}

        result = {r["pool"]: dict(r) for r in rows}
        log.info("Reconciliation by pool:")
        for pool, r in result.items():
            rec_stake = r["rec_stake"] or 0.0
            roi_rec = (100 * (r["rec_pnl"] or 0) / rec_stake) if rec_stake else 0.0
            log.info("  %s  n=%-4d wins=%-4d staked=$%-9.0f pnl=$%+9.0f  ROI=%+.1f%%",
                     r["pool"], r["tickets"], r["wins"],
                     rec_stake, r["rec_pnl"] or 0, roi_rec)
        return result


# ---------------------------------------------------------------------------
# CLI for ad-hoc reconciliation runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from hkjc_engine.config import DB_URL

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date", default=None,
                    help="ISO date; only reconcile snapshots since this date")
    ap.add_argument("--dividend_unit_base", type=float, default=10.0)
    args = ap.parse_args()

    SnapshotLogger(DB_URL).reconcile(
        dividend_unit_base=args.dividend_unit_base,
        start_date=args.start_date,
    )