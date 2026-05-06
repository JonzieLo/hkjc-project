"""
prepare_backtest_inputs.py
==========================

Produces the two CSVs the exotics_backtester and rank_calibration_check consume:
    results.csv        race_id, pos1_horse_no, pos2_horse_no, pos3_horse_no
    p_model_close.csv  race_id, horse_no, p_model, win_odds_close
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

from hkjc_engine.config import DB_URL
from hkjc_engine.live.predictor import LiveRacePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date-censored predictor — eliminates lookahead bias in historical replay
# ---------------------------------------------------------------------------

class BacktestPredictor(LiveRacePredictor):
    """Identical to LiveRacePredictor except historical-state lookups are
    censored to races strictly before the target race date."""

    def _fetch_historical_states_at(self, horse_codes, cutoff_date):
        if not horse_codes:
            return {}
        query = text("""
            WITH RankedRaces AS (
                SELECT
                    e.horse_code,
                    e.ema_early_z, e.ema_mid_z, e.ema_finish_z,
                    e.pre_race_mu, e.pre_race_sigma,
                    r.race_class AS last_race_class,
                    r.race_date  AS last_race_date,
                    ROW_NUMBER() OVER(
                        PARTITION BY e.horse_code
                        ORDER BY r.race_date DESC, r.race_no DESC
                    ) AS rn
                FROM race_entries e
                JOIN races r ON e.race_id = r.race_id
                WHERE e.horse_code IN :h_codes
                  AND e.finish_position IS NOT NULL
                  AND r.race_date < :cutoff
            )
            SELECT * FROM RankedRaces WHERE rn = 1
        """)
        with self.factory.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "h_codes": tuple(horse_codes),
                "cutoff": cutoff_date,
            })
        return df.set_index('horse_code').to_dict('index')

    def predict_at_date(self, *, today_class, venue, distance,
                        rail_placement, entries_list, race_date):
        original = self._fetch_historical_states
        self._fetch_historical_states = (
            lambda codes: self._fetch_historical_states_at(codes, race_date)
        )
        try:
            return self.predict_live_race(
                today_class=today_class, venue=venue, distance=distance,
                rail_placement=rail_placement, entries_list=entries_list,
            )
        finally:
            self._fetch_historical_states = original


# ---------------------------------------------------------------------------
# Race-universe sourcing
# ---------------------------------------------------------------------------

def _race_ids_from_dividends(engine, start_date: str, end_date: str) -> list[str]:
    """Pull every race_id that has dividend data AND falls in the date window."""
    q = text("""
        SELECT DISTINCT d.race_id
        FROM race_dividends d
        JOIN races r ON r.race_id = d.race_id
        WHERE r.race_date >= :start AND r.race_date <= :end
          AND d.pool IN ('QIN', 'QUINELLA', 'QPL', 'QUINELLA PLACE',
                         'TRI', 'TRIO', 'TIERCE')
        ORDER BY d.race_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start": start_date, "end": end_date})
    return df["race_id"].tolist()


def _race_ids_from_csv(closing_csv: str) -> list[str]:
    """Backward-compat path."""
    df = pd.read_csv(closing_csv)
    return sorted(df["race_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Main prep routine
# ---------------------------------------------------------------------------

def prepare_inputs(out_dir: str,
                   start_date: str | None = None,
                   end_date:   str | None = None,
                   closing_csv: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    engine = create_engine(DB_URL)

    if closing_csv:
        race_ids = _race_ids_from_csv(closing_csv)
        log.info("Loaded %d race_ids from %s (CSV mode)", len(race_ids), closing_csv)
    else:
        if not (start_date and end_date):
            today = date.today()
            two_years_ago = today.replace(year=today.year - 2)
            start_date = start_date or two_years_ago.isoformat()
            end_date   = end_date   or (today - timedelta(days=1)).isoformat()
        race_ids = _race_ids_from_dividends(engine, start_date, end_date)
        log.info("Loaded %d race_ids from race_dividends in [%s, %s]",
                 len(race_ids), start_date, end_date)

    if not race_ids:
        log.error("No race_ids to process — exiting.")
        return

    # -- Race-level metadata
    races_q = text("""
        SELECT race_id, race_date, venue, race_class, distance, rail_placement
        FROM races
        WHERE race_id IN :rids
    """)
    with engine.connect() as conn:
        races_meta = pd.read_sql(
            races_q, conn, params={"rids": tuple(race_ids)}
        ).set_index("race_id")
    log.info("Pulled metadata for %d races", len(races_meta))

    # -- Per-runner data
    entries_q = text("""
        SELECT race_id, horse_code, horse_no, jockey, draw,
               actual_weight, win_odds, finish_position
        FROM race_entries
        WHERE race_id IN :rids
    """)
    with engine.connect() as conn:
        entries = pd.read_sql(entries_q, conn, params={"rids": tuple(race_ids)})
    log.info("Pulled %d entry rows", len(entries))

    # ----- results.csv ----------------------------------------------------
    top3 = entries[entries["finish_position"].isin([1, 2, 3])].copy()
    results = (
        top3.pivot_table(
            index="race_id", columns="finish_position",
            values="horse_no", aggfunc="first",
        )
        .rename(columns={1: "pos1_horse_no",
                         2: "pos2_horse_no",
                         3: "pos3_horse_no"})
        .dropna(subset=["pos1_horse_no", "pos2_horse_no", "pos3_horse_no"])
        .astype(int)
        .reset_index()
    )
    results_path = os.path.join(out_dir, "results.csv")
    results.to_csv(results_path, index=False)
    log.info("Wrote %d results to %s", len(results), results_path)

    # ----- p_model_close.csv ---------------------------------------------
    predictor = BacktestPredictor(DB_URL)
    p_rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    t0 = time.time()
    n = len(race_ids)

    for i, race_id in enumerate(race_ids, start=1):
        if i % 50 == 0 or i == n:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (n - i) / max(rate, 1e-9)
            log.info("  [%d/%d]  rate=%.1f races/sec  ETA=%.0fs  "
                     "kept=%d  skipped=%d",
                     i, n, rate, eta, len(p_rows), len(skipped))

        if race_id not in races_meta.index:
            skipped.append((race_id, "no metadata in races table"))
            continue
        meta = races_meta.loc[race_id]
        race_entries_df = entries[entries["race_id"] == race_id]

        entries_list = []
        for _, r in race_entries_df.iterrows():
            wo = r["win_odds"]
            if pd.isna(wo) or float(wo) <= 1.0:
                continue
            entries_list.append({
                "horse_no":      str(int(r["horse_no"])),
                "horse_code":    r["horse_code"],
                "jockey":        r["jockey"] or "UNKNOWN",
                "draw":          int(r["draw"]) if pd.notna(r["draw"]) else 0,
                "actual_weight": float(r["actual_weight"])
                                 if pd.notna(r["actual_weight"]) else 120.0,
                "live_odds":     float(wo),
                "live_pla_odds": 0.0,
            })
        if len(entries_list) < 2:
            skipped.append((race_id, f"only {len(entries_list)} valid runners"))
            continue

        try:
            preds = predictor.predict_at_date(
                today_class=meta["race_class"],
                venue=meta["venue"],
                distance=int(meta["distance"]),
                rail_placement=meta["rail_placement"],
                entries_list=entries_list,
                race_date=meta["race_date"],
            )
        except Exception as e:
            log.exception("Predictor failed for %s: %s", race_id, e)
            skipped.append((race_id, f"predictor error: {e}"))
            continue

        if preds.empty:
            skipped.append((race_id, "predictor returned empty frame"))
            continue

        for _, row in preds.iterrows():
            p_rows.append({
                "race_id": race_id,
                "horse_no": int(row["horse_no"]),
                "p_model": float(row["P_model_exo"]), # <--- Multi-Agent Exotics Stacker Used Here
                "win_odds_close": float(row["live_odds"]),
            })

    p_model_path = os.path.join(out_dir, "p_model_close.csv")
    pd.DataFrame(p_rows).to_csv(p_model_path, index=False)
    log.info("Wrote %d predictions to %s", len(p_rows), p_model_path)

    if skipped:
        log.warning("Skipped %d races. First 20 reasons:", len(skipped))
        for rid, reason in skipped[:20]:
            log.warning("  %s -> %s", rid, reason)
        if len(skipped) > 20:
            log.warning("  ... and %d more", len(skipped) - 20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",     default="./backtest_inputs")
    ap.add_argument("--start_date",  default=None,
                    help="ISO date (default: 2 years before yesterday)")
    ap.add_argument("--end_date",    default=None,
                    help="ISO date (default: yesterday)")
    ap.add_argument("--closing_odds", default=None,
                    help="Backward-compat: drive race universe off this CSV "
                         "instead of race_dividends date window")
    args = ap.parse_args()
    prepare_inputs(
        out_dir=args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        closing_csv=args.closing_odds,
    )