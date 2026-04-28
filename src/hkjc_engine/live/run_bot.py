"""
HKJC live bot — drift-aware refactor.

What changed from the original
------------------------------
1. Sizing per pool now uses `betting_policy.qualify_and_size` with the
   pool name passed in, so the drift-variance hurdle, pool-conditional
   Kelly fraction and pool-conditional race cap are applied
   automatically. WIN stays roughly the same; PLA/QIN/QPL/TRI tighten
   substantially.
2. If a trained `DriftForecaster` artifact is present, we build per-race
   WIN-pool drift features once and pass per-horse `(mu_R, sigma_R,
   median_R)` overrides into the policy. Exotic combos receive their
   drift stats from Harville projection of the WIN per-horse drifts.
3. `live_config.json` is now the single source of truth for shrinkage,
   thetas AND drift_stats — populated by walk_forward.py per training
   window.
4. Master cross-pool cap is unchanged (`MASTER_RACE_CAP`), but per-pool
   caps are now derived from `race_cap_for_pool`, not hard-coded.

The Redis schema, Discord rendering, snapshot-logger contract and main
poll loop are unchanged so the operational interface is identical.
"""
from __future__ import annotations

import datetime
import itertools
import json
import logging
import os
import time
from typing import Mapping

import numpy as np
import pandas as pd
import redis  # noqa: F401  (transitive via redis_client; kept for clarity)
from discord_webhook import DiscordWebhook

from hkjc_engine.config import (
    DB_URL,
    LIVE_BANKROLL,
    LIVE_VENUE,
    WEBHOOK_URL,
    artifact,
    redis_client,
)
from hkjc_engine.live.predictor import LiveRacePredictor
from hkjc_engine.live.snapshot_logger import SnapshotLogger
from hkjc_engine.models.betting_policy import (
    DEFAULT_DRIFT_STATS,
    DriftStats,
    cap_simultaneous_stakes,
    fractional_kelly_stake,
    get_ev_hurdle,
    qualify_and_size,
    race_cap_for_pool,
    shrunk_ev,
)
from hkjc_engine.models.drift_forecaster import (
    DriftForecaster,
    build_win_features,
    project_drift_to_exotic,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VENUE             = LIVE_VENUE
BANKROLL          = LIVE_BANKROLL
KELLY_FRAC        = 0.25
BASE_HURDLE       = 0.02
LONGSHOT_BUF      = 0.015
LONGSHOT_D        = 15.0
PER_BET_CAP       = 0.02
RACE_CAP_BASE     = 0.04          # base WIN-pool race cap; exotic caps derived from this
MASTER_RACE_CAP   = 0.06
MIN_STAKE_ABS     = 10.0
TOP_N_HORSES_EXO  = 8
MAX_ODDS_WIN      = 25.0

THETA_2 = 0.8824
THETA_3 = 0.7760

CONFIG_PATH       = artifact('live_config.json')
SHRINKAGE_DEFAULT = 0.75


# ---------------------------------------------------------------------------
# Live config loader (shrinkage + theta + drift stats + drift forecaster)
# ---------------------------------------------------------------------------

def _load_live_config(path: str = CONFIG_PATH) -> dict:
    """Load shrinkage / theta / drift_stats. Returns dict with safe defaults."""
    cfg: dict = {
        'shrinkage': SHRINKAGE_DEFAULT,
        'theta_2': THETA_2,
        'theta_3': THETA_3,
        'drift_stats': None,
    }
    if not os.path.exists(path):
        logging.info("No %s; using defaults.", path)
        return cfg
    try:
        with open(path) as f:
            data = json.load(f)
        if 'shrinkage' in data:
            cfg['shrinkage'] = float(min(max(data['shrinkage'], 0.6), 0.95))
        if 'theta_2' in data: cfg['theta_2'] = float(data['theta_2'])
        if 'theta_3' in data: cfg['theta_3'] = float(data['theta_3'])
        if 'drift_stats' in data and isinstance(data['drift_stats'], dict):
            ds = {}
            for pool, stats in data['drift_stats'].items():
                try:
                    ds[pool.upper()] = DriftStats(
                        mu_R=float(stats['mu_R']),
                        sigma_R=float(stats['sigma_R']),
                        median_R=float(stats.get('median_R', stats['mu_R'])),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if ds:
                cfg['drift_stats'] = ds
    except Exception as e:
        logging.warning("Failed to parse %s: %s", path, e)
    return cfg


def _load_drift_forecaster() -> DriftForecaster | None:
    """Load the trained drift forecaster from artifacts dir; None if missing."""
    path = artifact('drift_forecaster.pkl')
    if not os.path.exists(path):
        logging.info("No drift forecaster at %s — using static defaults.", path)
        return None
    try:
        return DriftForecaster(model_path=path).load()
    except Exception as e:
        logging.warning("Drift forecaster load failed (%s); using static defaults.", e)
        return None


_LIVE_CFG = _load_live_config()
SHRINKAGE: float = _LIVE_CFG['shrinkage']
DRIFT_OVERRIDE: dict | None = _LIVE_CFG['drift_stats']
DRIFT_FORECASTER: DriftForecaster | None = _load_drift_forecaster()

logging.info("LIVE BOT CONFIG: shrinkage=%.4f | theta=[%.4f, %.4f] | "
             "drift_override_keys=%s | drift_forecaster=%s",
             SHRINKAGE, _LIVE_CFG['theta_2'], _LIVE_CFG['theta_3'],
             list((DRIFT_OVERRIDE or {}).keys()),
             'YES' if DRIFT_FORECASTER else 'NO')

r_cache = redis_client()


# ---------------------------------------------------------------------------
# Redis state helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_dynamic_metadata(venue, race_no):
    data = r_cache.get(f"live_race_metadata:{venue}:{race_no}")
    return json.loads(data) if data else None


def get_race_status(venue, race_no):
    return r_cache.get(f"race_status:{venue}:{race_no}")


def get_active_race(venue):
    active = []
    now = datetime.datetime.now()
    for i in range(1, 12):
        meta = get_dynamic_metadata(venue, i)
        if meta and 'time' in meta and ':' in meta['time']:
            try:
                start = datetime.datetime.strptime(meta['time'], "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day)
                if start - datetime.timedelta(minutes=30) <= now <= start + datetime.timedelta(minutes=3):
                    active.append(i)
            except ValueError:
                continue
    return active


def parse_live_race_data(venue, race_no):
    win_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:WIN")
    pla_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:PLA")
    runners_raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:runners")
    if not win_raw or not runners_raw:
        return None

    try:
        def _parse_singles(raw, kind):
            d = {}
            data = json.loads(raw)
            pools = data.get('data', {}).get('raceMeetings', [])[0].get('pmPools', [])
            for p in pools:
                if p.get('oddsType') == kind:
                    for node in p.get('oddsNodes', []):
                        combo = str(node.get('combString')).lstrip('0')
                        val = node.get('oddsValue')
                        if val:
                            try: d[combo] = float(val)
                            except ValueError: pass
            return d

        win_dict = _parse_singles(win_raw, 'WIN')
        pla_dict = _parse_singles(pla_raw, 'PLA') if pla_raw else {}

        runners_data = json.loads(runners_raw)
        races = runners_data.get('data', {}).get('raceMeetings', [])[0].get('races', [])
        target = next((r for r in races
                       if str(r.get('no')).lstrip('0') == str(race_no)), None)
        if not target:
            return None

        entries = []
        for r in target.get('runners', []):
            hn = str(r.get('no')).lstrip('0')
            if hn not in win_dict:
                continue
            entries.append({
                'horse_no':      hn,
                'horse_code':    r.get('horse', {}).get('code', 'UNKNOWN'),
                'jockey':        r.get('jockey', {}).get('name', 'UNKNOWN'),
                'draw':          int(r.get('barrierDrawNumber') or 0),
                'actual_weight': float(r.get('handicapWeight') or 120),
                'live_odds':     win_dict[hn],
                'live_pla_odds': pla_dict.get(hn, 0.0),
            })
        return entries
    except Exception as e:
        logging.error("Parse Error: %s", e)
        return None


def get_live_exotic_odds(venue, race_no, pool_type="QIN"):
    odds_dict = {}
    raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:{pool_type}")
    if not raw:
        return odds_dict
    try:
        data = json.loads(raw)
        meetings = data.get('data', {}).get('raceMeetings', [])
        if not meetings:
            return odds_dict
        target = None
        for p in meetings[0].get('pmPools', []):
            if p.get('oddsType') == pool_type:
                races = p.get('leg', {}).get('races', []) or p.get('races', [])
                if int(race_no) in races:
                    target = p
                    break
        if not target:
            return odds_dict
        for node in target.get('oddsNodes', []):
            val = node.get('value') or node.get('oddsValue')
            combo = node.get('combination') or node.get('combString')
            if val is not None and combo:
                parts = str(combo).replace(',', '-').split('-')
                clean = "-".join(str(int(p)) for p in parts if str(p).strip())
                try: odds_dict[clean] = float(val)
                except ValueError: pass
    except Exception as e:
        logging.error("Failed to parse %s API odds: %s", pool_type, e)
    return odds_dict


# ---------------------------------------------------------------------------
# Henery-discounted Harville projections (unchanged math)
# ---------------------------------------------------------------------------

def _p_order_by_idx(p_arr, i1, i2, i3=None):
    p1 = p_arr[i1]
    sum_t2 = np.sum(p_arr ** THETA_2) - (p1 ** THETA_2)
    if sum_t2 <= 0:
        return 0.0
    p2 = p_arr[i2]
    p_exact_2 = p1 * ((p2 ** THETA_2) / sum_t2)
    if i3 is None:
        return p_exact_2
    p3 = p_arr[i3]
    sum_t3 = np.sum(p_arr ** THETA_3) - (p1 ** THETA_3) - (p2 ** THETA_3)
    if sum_t3 <= 0:
        return 0.0
    return p_exact_2 * ((p3 ** THETA_3) / sum_t3)


def p_quinella(p_arr, i, j):
    return _p_order_by_idx(p_arr, i, j) + _p_order_by_idx(p_arr, j, i)


def p_quinella_place(p_arr, i, j):
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j: continue
        total += _p_order_by_idx(p_arr, i, j, k)
        total += _p_order_by_idx(p_arr, j, i, k)
        total += _p_order_by_idx(p_arr, i, k, j)
        total += _p_order_by_idx(p_arr, j, k, i)
        total += _p_order_by_idx(p_arr, k, i, j)
        total += _p_order_by_idx(p_arr, k, j, i)
    return total


def p_trio(p_arr, i, j, k):
    return sum(_p_order_by_idx(p_arr, *perm)
               for perm in itertools.permutations([i, j, k]))


def p_place(p_arr, i):
    n = len(p_arr); p_1st = p_arr[i]
    p_2nd = sum(_p_order_by_idx(p_arr, j, i) for j in range(n) if j != i)
    p_3rd = 0.0
    for j in range(n):
        if j == i: continue
        for k in range(n):
            if k == i or k == j: continue
            p_3rd += _p_order_by_idx(p_arr, j, k, i)
    return p_1st + p_2nd + p_3rd


# ---------------------------------------------------------------------------
# Drift-aware sizing helpers
# ---------------------------------------------------------------------------

def _resolve_pool_drift(pool: str, override: Mapping[str, DriftStats] | None
                        ) -> Mapping[str, DriftStats] | None:
    """Pass-through: caller-level override takes precedence over defaults."""
    return override


def _size_drift_aware(p_raw: float, odds: float, pool: str,
                      drift: Mapping[str, DriftStats] | None) -> tuple[float, float]:
    """Return (stake, ev_eff) for a single bet using drift-aware policy."""
    if pd.isna(odds) or odds <= 1.0 or p_raw <= 0:
        return 0.0, 0.0
    ev_eff = shrunk_ev(p_raw, odds, SHRINKAGE, pool=pool, drift_override=drift)
    hurdle = get_ev_hurdle(odds, BASE_HURDLE, LONGSHOT_BUF, LONGSHOT_D,
                           pool=pool, drift_override=drift)
    if ev_eff < hurdle:
        return 0.0, ev_eff
    stake = fractional_kelly_stake(
        p_raw, odds, BANKROLL,
        kelly_fraction=KELLY_FRAC, shrinkage=SHRINKAGE,
        per_bet_cap=PER_BET_CAP, min_stake_abs=MIN_STAKE_ABS,
        pool=pool, drift_override=drift,
    )
    return stake, ev_eff


def build_exotic_table(p_arr: np.ndarray,
                       horse_nos: list[str],
                       live_odds_dict: dict,
                       calc_func,
                       comb_len: int,
                       pool: str,
                       win_drift_df: pd.DataFrame | None) -> pd.DataFrame:
    """Top-N-by-P_model combinatoric search with combo-level drift projection.

    For each combination, builds a per-combo `DriftStats` via
    `project_drift_to_exotic` from the WIN forecaster output; falls back
    to DEFAULT_DRIFT_STATS[pool] when the forecaster is unavailable.
    """
    if not live_odds_dict:
        return pd.DataFrame()
    n = len(p_arr)
    top_indices = np.argsort(-p_arr)[:min(TOP_N_HORSES_EXO, n)]

    rows: list[dict] = []
    for combo_idx in itertools.combinations(top_indices, comb_len):
        h_nums = sorted(int(horse_nos[i]) for i in combo_idx)
        key = "-".join(str(x) for x in h_nums)
        if key not in live_odds_dict:
            continue
        odds = live_odds_dict[key]
        p_model = calc_func(p_arr, *combo_idx)
        if p_model <= 0:
            continue

        # Build per-combo override
        if win_drift_df is not None and not win_drift_df.empty:
            combo_drift = project_drift_to_exotic(win_drift_df, h_nums)
            override = {pool: DriftStats(**combo_drift)}
        else:
            override = None      # uses DEFAULT_DRIFT_STATS[pool]

        stake, ev_eff = _size_drift_aware(p_model, odds, pool=pool,
                                          drift=override)
        rows.append({
            'combo':   key,
            'live':    odds,
            'fair':    round(1.0 / p_model, 1),
            'p_model': p_model,
            'ev':      ev_eff,
            'stake':   stake,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Pool-conditional race-level cap
    qual = df['stake'] > 0
    if qual.any():
        df.loc[qual, 'stake'] = cap_simultaneous_stakes(
            df.loc[qual, 'stake'].values, BANKROLL,
            race_cap=RACE_CAP_BASE, pool=pool, drift_override=DRIFT_OVERRIDE,
        )
        df.loc[df['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
    return df.sort_values('ev', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Discord rendering (unchanged from original)
# ---------------------------------------------------------------------------

def handle_discord_pool(venue, race_no, pool_name, is_empty, message_content):
    if not WEBHOOK_URL:
        if not is_empty:
            print(f"\n--- [{venue} R{race_no} {pool_name}] (no Discord) ---")
            print(message_content)
        return
    msg_key = f"discord_msg_id:{venue}:{race_no}:{pool_name}"
    existing = r_cache.get(msg_key)
    if is_empty and not existing:
        return
    try:
        if existing:
            wh = DiscordWebhook(url=WEBHOOK_URL, id=existing, content=message_content)
            resp = wh.edit()
            if resp.status_code == 404:
                wh = DiscordWebhook(url=WEBHOOK_URL, content=message_content)
                resp = wh.execute()
                if resp.status_code in (200, 204):
                    r_cache.set(msg_key, resp.json()['id'])
        else:
            wh = DiscordWebhook(url=WEBHOOK_URL, content=message_content)
            resp = wh.execute()
            if resp.status_code in (200, 204):
                r_cache.set(msg_key, resp.json()['id'])
    except Exception as e:
        logging.error("Webhook failed for %s: %s", pool_name, e)


def send_to_discord(venue, race_no, meta, df_win, df_pla, df_qin, df_qpl, df_tri,
                    is_closing=False):
    time_str = meta.get('time', 'Unknown') if isinstance(meta, dict) else 'Unknown'
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    tag = " [CLOSING SNAPSHOT]" if is_closing else ""
    header = f"**{'='*10} {venue} R{race_no} @ {time_str}{tag} {'='*10}**"

    if all(d.empty for d in [df_win, df_pla, df_qin, df_qpl, df_tri]):
        return

    def _fmt_stake(x): return f"${int(x)}" if x > 0 else "-"

    def _render_singles(df, limit=None):
        if df.empty: return "(no runners)\n"
        show = df.copy() if limit is None else df.head(limit).copy()
        show['Odds_str']  = show['odds'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'horse': 'No.', 'code': 'Code',
                'Odds_str': 'Odds', 'Fair_str': 'FAIR',
                'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols)].rename(columns=cols).to_string(index=False) + "\n"

    def _render_exotic(df, label, limit=20):
        if df.empty: return "(no live odds yet)\n"
        show = df.head(limit).copy()
        show['Live_str']  = show['live'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'combo': label, 'Live_str': 'Live', 'Fair_str': 'Fair',
                'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols)].rename(columns=cols).to_string(index=False) + "\n"

    for label, df_, limit in [("WIN", df_win, None), ("PLA", df_pla, None)]:
        msg = f"{header if label == 'WIN' else f'**{venue} R{race_no} - {label} ODDS:**'}"
        if label == 'WIN':
            msg += f"\n*Last Updated: {now_str}*\n**WIN ODDS:**"
        msg += "\n```\n" + _render_singles(df_, limit) + "```"
        handle_discord_pool(venue, race_no, label, df_.empty, msg)

    for label, df_, limit in [("QIN", df_qin, 15), ("QPL", df_qpl, 15), ("TRI", df_tri, 20)]:
        msg = f"**{venue} R{race_no} - {label}:**\n```\n"
        msg += _render_exotic(df_, label, limit) + "```"
        handle_discord_pool(venue, race_no, label, df_.empty, msg)


# ---------------------------------------------------------------------------
# Main per-race prediction loop
# ---------------------------------------------------------------------------

def _build_win_drift_for_race(predictor, race_id_for_drift: str | None
                              ) -> pd.DataFrame:
    """Build per-horse WIN drift forecasts. Empty df if forecaster missing
    or feature build fails.

    NOTE: builds features from `live_odds_history` for the given race_id.
    For active LIVE races whose data has not yet been archived to
    Postgres, the DB query will return empty; we fall back to static
    DEFAULT_DRIFT_STATS in that case (still drift-aware, just not
    per-horse-conditional).
    """
    if DRIFT_FORECASTER is None or not race_id_for_drift:
        return pd.DataFrame()
    try:
        feats = build_win_features(predictor.factory.engine, [race_id_for_drift])
        if feats.empty:
            return pd.DataFrame()
        return DRIFT_FORECASTER.predict_win(feats)
    except Exception as e:
        logging.warning("Drift forecast failed for %s: %s",
                        race_id_for_drift, e)
        return pd.DataFrame()


def run_prediction_for_race(predictor, venue, race_no, snap_logger,
                            is_closing=False):
    start = time.perf_counter()
    print(f"\nFetching Live Data for {venue} Race {race_no}...")
    meta = get_dynamic_metadata(venue, race_no)
    if not meta:
        print(f"Skipping R{race_no}: No metadata in Redis.")
        return

    race_class = meta.get('class', 'Unknown')
    distance   = meta.get('distance', 1200)
    rail       = meta.get('rail', 'A')
    race_time  = meta.get('time', 'Unknown')
    print(f"Processing {venue} R{race_no}: {race_class} | {distance}M | "
          f"Rail: {rail} | Time: {race_time}")

    t0 = time.perf_counter()
    entries = parse_live_race_data(venue, race_no)
    if not entries:
        print(f"R{race_no}: no entries parsed yet; skipping.")
        return
    live_qin = get_live_exotic_odds(venue, race_no, "QIN")
    live_qpl = get_live_exotic_odds(venue, race_no, "QPL")
    live_tri = get_live_exotic_odds(venue, race_no, "TRI")
    t1 = time.perf_counter()

    try:
        results = predictor.predict_live_race(
            today_class=race_class, venue=venue, distance=distance,
            rail_placement=rail, entries_list=entries,
        )
    except Exception as e:
        logging.exception("R%s: predictor failed: %s", race_no, e)
        return
    if results.empty or len(results) < 2:
        print(f"R{race_no}: predictor returned <2 rows; skipping.")
        return
    t2 = time.perf_counter()
    print(f"R{race_no} Profiling -> Scrape: {(t1-t0)*1000:.1f}ms | "
          f"Inference: {(t2-t1)*1000:.1f}ms")

    p_arr     = results['P_model'].to_numpy(dtype=float)
    horse_nos = results['horse_no'].astype(str).tolist()
    codes     = results['horse_code'].tolist()
    live_odds = results['live_odds'].astype(float).to_numpy()

    # Build drift forecast once per race (used by all pools).
    # The race_id is only resolvable from snapshot_recommendations after
    # closing snapshot; for live we approximate it as today + venue + R{n}.
    today = datetime.datetime.now().strftime('%Y%m%d')
    race_id_for_drift = f"{today}_{venue}_{race_no:02d}"
    win_drift_df = _build_win_drift_for_race(predictor, race_id_for_drift)

    # ----- WIN pool: drift-aware sizing with per-horse override -----
    df_win = _size_win_pool(p_arr, horse_nos, codes, live_odds, win_drift_df)

    # ----- PLA pool: per-horse sizing, pool-level cap -----
    df_pla = _size_pla_pool(p_arr, horse_nos, codes, results, win_drift_df)

    # ----- Exotic pools -----
    df_qin = build_exotic_table(p_arr, horse_nos, live_qin, p_quinella, 2,
                                pool='QIN', win_drift_df=win_drift_df)
    df_qpl = build_exotic_table(p_arr, horse_nos, live_qpl, p_quinella_place, 2,
                                pool='QPL', win_drift_df=win_drift_df)
    df_tri = build_exotic_table(p_arr, horse_nos, live_tri, p_trio, 3,
                                pool='TRI', win_drift_df=win_drift_df)

    # ----- Master cross-pool cap -----
    all_dfs = [df_win, df_pla, df_qin, df_qpl, df_tri]
    total = sum(d.loc[d['stake'] > 0, 'stake'].sum()
                for d in all_dfs if not d.empty)
    cap_dollars = BANKROLL * MASTER_RACE_CAP
    if total > cap_dollars > 0:
        shrink = cap_dollars / total
        for d in all_dfs:
            if d.empty: continue
            d.loc[d['stake'] > 0, 'stake'] *= shrink
            d.loc[(d['stake'] > 0) & (d['stake'] < MIN_STAKE_ABS), 'stake'] = 0.0

    # ----- Persist snapshot -----
    _log_snapshot(snap_logger, venue, race_no, race_time,
                  df_win, df_pla, df_qin, df_qpl, df_tri)

    # ----- Console + Discord -----
    def _n(d): return 0 if d.empty else int((d['stake'] > 0).sum())
    def _s(d): return 0.0 if d.empty else float(d.loc[d['stake'] > 0, 'stake'].sum())
    end = time.perf_counter()
    print(f"R{race_no}: WIN={_n(df_win)} PLA={_n(df_pla)} "
          f"QIN={_n(df_qin)} QPL={_n(df_qpl)} TRI={_n(df_tri)} | "
          f"total_stake=${sum(_s(d) for d in all_dfs):,.0f} | "
          f"Latency: {(end-start)*1000:.1f}ms")

    try:
        send_to_discord(venue, race_no, meta,
                        df_win, df_pla, df_qin, df_qpl, df_tri,
                        is_closing=is_closing)
    except Exception as e:
        logging.error("Discord dispatch failed for R%s: %s", race_no, e)


def _size_win_pool(p_arr: np.ndarray, horse_nos, codes, live_odds,
                   win_drift_df: pd.DataFrame) -> pd.DataFrame:
    """WIN-pool sizing using qualify_and_size with per-row overrides.

    When `win_drift_df` is non-empty, builds per-row DriftStats for each
    horse and runs the size-and-cap pipeline manually so the per-horse
    drift is respected. Otherwise, uses the static WIN pool stats.
    """
    rows: list[dict] = []
    if win_drift_df is None or win_drift_df.empty:
        # Vectorised path with static defaults
        keep_idx, stakes = qualify_and_size(
            p_raw=p_arr, odds=live_odds, bankroll=BANKROLL,
            kelly_fraction=KELLY_FRAC, shrinkage=SHRINKAGE,
            base_hurdle=BASE_HURDLE,
            longshot_buffer=LONGSHOT_BUF, longshot_threshold=LONGSHOT_D,
            per_bet_cap=PER_BET_CAP, race_cap=RACE_CAP_BASE,
            min_stake_abs=MIN_STAKE_ABS, top_k=10, max_odds=MAX_ODDS_WIN,
            pool='WIN', drift_override=DRIFT_OVERRIDE,
        )
        keep_set = set(int(i) for i in keep_idx)
        stake_lookup = dict(zip([int(i) for i in keep_idx], stakes))
        for i, (hn, hc, odds) in enumerate(zip(horse_nos, codes, live_odds)):
            p = float(p_arr[i])
            ev_eff = shrunk_ev(p, float(odds), SHRINKAGE,
                               pool='WIN', drift_override=DRIFT_OVERRIDE)
            stake = float(stake_lookup.get(i, 0.0))
            rows.append({
                'horse': hn, 'code': hc, 'odds': float(odds),
                'p_model': p, 'fair': 1.0 / p if p > 0 else float('inf'),
                'ev': ev_eff, 'stake': stake,
            })
    else:
        # Per-row path with forecaster-derived DriftStats
        drift_lookup = win_drift_df.set_index(
            win_drift_df['horse_no'].astype(str))
        for i, (hn, hc, odds) in enumerate(zip(horse_nos, codes, live_odds)):
            p = float(p_arr[i])
            if str(hn) in drift_lookup.index:
                row = drift_lookup.loc[str(hn)]
                override = {'WIN': DriftStats(
                    mu_R=float(row['mu_R']),
                    sigma_R=float(row['sigma_R']),
                    median_R=float(row['median_R']),
                )}
            else:
                override = DRIFT_OVERRIDE
            stake, ev_eff = _size_drift_aware(p, float(odds),
                                              pool='WIN', drift=override)
            if odds > MAX_ODDS_WIN:
                stake = 0.0
            rows.append({
                'horse': hn, 'code': hc, 'odds': float(odds),
                'p_model': p, 'fair': 1.0 / p if p > 0 else float('inf'),
                'ev': ev_eff, 'stake': stake,
            })

        # Race-level cap (pool-conditional)
        df_tmp = pd.DataFrame(rows)
        qmask = df_tmp['stake'] > 0
        if qmask.any():
            df_tmp.loc[qmask, 'stake'] = cap_simultaneous_stakes(
                df_tmp.loc[qmask, 'stake'].values, BANKROLL,
                race_cap=RACE_CAP_BASE, pool='WIN',
                drift_override=DRIFT_OVERRIDE,
            )
            df_tmp.loc[df_tmp['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
        return df_tmp.sort_values('ev', ascending=False).reset_index(drop=True)

    df_win = pd.DataFrame(rows).sort_values('ev', ascending=False).reset_index(drop=True)
    return df_win


def _size_pla_pool(p_arr: np.ndarray, horse_nos, codes,
                   results: pd.DataFrame, win_drift_df: pd.DataFrame) -> pd.DataFrame:
    """Place-pool sizing using p_place + pool='PLA' drift stats."""
    pla_rows = []
    for i, (hn, hc) in enumerate(zip(horse_nos, codes)):
        p_odds = float(results['live_pla_odds'].iloc[i])
        if pd.isna(p_odds) or p_odds < 1.0:
            continue
        pp = p_place(p_arr, i)

        # If WIN drift forecaster fired, project the single-horse drift
        # onto PLA via the same Harville projection (single horse = no
        # combination, so it's just the WIN drift for that horse).
        if win_drift_df is not None and not win_drift_df.empty:
            single = project_drift_to_exotic(win_drift_df, [int(hn)])
            override = {'PLA': DriftStats(**single)}
        else:
            override = DRIFT_OVERRIDE

        stake, ev_eff = _size_drift_aware(pp, p_odds, pool='PLA', drift=override)
        pla_rows.append({
            'horse': hn, 'code': hc, 'odds': p_odds,
            'p_model': pp, 'fair': 1.0 / pp if pp > 0 else float('inf'),
            'ev': ev_eff, 'stake': stake,
        })

    df_pla = pd.DataFrame(pla_rows)
    if df_pla.empty:
        return df_pla
    qmask = df_pla['stake'] > 0
    if qmask.any():
        df_pla.loc[qmask, 'stake'] = cap_simultaneous_stakes(
            df_pla.loc[qmask, 'stake'].values, BANKROLL,
            race_cap=RACE_CAP_BASE, pool='PLA',
            drift_override=DRIFT_OVERRIDE,
        )
        df_pla.loc[df_pla['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
    return df_pla.sort_values('ev', ascending=False).reset_index(drop=True)


def _log_snapshot(snap_logger, venue, race_no, race_time,
                  df_win, df_pla, df_qin, df_qpl, df_tri):
    try:
        recs = []
        for d, pool, combo_col in [(df_win, "WIN", "horse"),
                                   (df_pla, "PLA", "horse"),
                                   (df_qin, "QIN", "combo"),
                                   (df_qpl, "QPL", "combo"),
                                   (df_tri, "TRI", "combo")]:
            if d.empty: continue
            for _, row in d.iterrows():
                recs.append({
                    "pool":        pool,
                    "combination": str(row[combo_col]),
                    "live_odds":   float(row["odds"]) if pool in ("WIN", "PLA")
                                   else float(row["live"]),
                    "p_raw":       float(row["p_model"]),
                    "p_shrunk":    min(float(row["p_model"]) * SHRINKAGE, 1 - 1e-9),
                    "stake":       float(row["stake"]),
                })
        if not recs:
            return
        race_id = f"{datetime.datetime.now().strftime('%Y%m%d')}_{venue}_{race_no:02d}"
        try:
            race_off_dt = datetime.datetime.strptime(
                race_time, "%H:%M",
            ).replace(year=datetime.datetime.now().year,
                      month=datetime.datetime.now().month,
                      day=datetime.datetime.now().day,
                      tzinfo=datetime.timezone.utc)
        except Exception:
            race_off_dt = datetime.datetime.now(datetime.timezone.utc)

        # Persist drift stats in the config blob for offline analysis
        cfg = {"theta_2": _LIVE_CFG['theta_2'],
               "theta_3": _LIVE_CFG['theta_3'],
               "shrinkage": SHRINKAGE,
               "drift_forecaster_used": DRIFT_FORECASTER is not None}
        if DRIFT_OVERRIDE:
            cfg['drift_stats'] = {
                p: {'mu_R': s.mu_R, 'sigma_R': s.sigma_R, 'median_R': s.median_R}
                for p, s in DRIFT_OVERRIDE.items()
            }

        snap_logger.log_snapshot(
            race_id=race_id, race_off_time=race_off_dt,
            bankroll=BANKROLL, recommendations=recs, config=cfg,
        )
    except Exception as e:
        logging.error("Snapshot log failed for R%s: %s", race_no, e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = LiveRacePredictor(DB_URL)
    snap_logger = SnapshotLogger(DB_URL)
    closed_processed: set[int] = set()

    print(f"--- LIVE BOT START | venue={VENUE} | shrinkage={SHRINKAGE:.3f} | "
          f"theta=[{_LIVE_CFG['theta_2']}, {_LIVE_CFG['theta_3']}] | "
          f"kelly=alpha{KELLY_FRAC} | drift_forecaster="
          f"{'ON' if DRIFT_FORECASTER else 'OFF'} ---")

    while True:
        try:
            active = get_active_race(VENUE)

            # Closing snapshots (once per race when STOP_SELL hits)
            for r_no in range(1, 12):
                if r_no in closed_processed:
                    continue
                if get_race_status(VENUE, r_no) == 'CLOSED':
                    print(f"R{r_no} STOP_SELL — sending final closing snapshot.")
                    try:
                        run_prediction_for_race(predictor, VENUE, r_no,
                                                snap_logger, is_closing=True)
                    except Exception as e:
                        logging.exception("Closing snapshot R%s failed: %s",
                                          r_no, e)
                    closed_processed.add(r_no)

            if not active:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"No races in active window. Polling Redis...")
                time.sleep(15)
                continue

            for r_no in active:
                if r_no in closed_processed:
                    continue
                if get_race_status(VENUE, r_no) == 'CLOSED':
                    continue
                try:
                    run_prediction_for_race(predictor, VENUE, r_no, snap_logger)
                except Exception as e:
                    logging.exception("Race %s failed: %s", r_no, e)

            # Cadence: tighten near jump
            meta_first = get_dynamic_metadata(VENUE, active[0])
            tts = 999
            if meta_first and 'time' in meta_first:
                try:
                    now = datetime.datetime.now()
                    start_t = datetime.datetime.strptime(
                        meta_first['time'], "%H:%M",
                    ).replace(year=now.year, month=now.month, day=now.day)
                    tts = (start_t - now).total_seconds()
                except Exception:
                    pass
            time.sleep(3 if -180 < tts <= 120 else 15)

        except KeyboardInterrupt:
            print("\nShutting down live bot cleanly.")
            break
        except Exception as e:
            logging.exception("Main loop error: %s", e)
            time.sleep(10)
