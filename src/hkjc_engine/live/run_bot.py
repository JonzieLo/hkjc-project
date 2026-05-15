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
import redis
from discord_webhook import DiscordWebhook

from hkjc_engine.config import (
    DB_URL, LIVE_BANKROLL, LIVE_VENUE, WEBHOOK_URL, artifact, redis_client,
)
from hkjc_engine.live.predictor import LiveRacePredictor
from hkjc_engine.live.snapshot_logger import SnapshotLogger
from hkjc_engine.models.betting_policy import (
    DEFAULT_DRIFT_STATS, DriftStats, cap_simultaneous_stakes,
    fractional_kelly_stake, get_ev_hurdle, qualify_and_size,
    race_cap_for_pool, shrunk_ev,
)
from hkjc_engine.models.drift_forecaster import (
    DriftForecaster, build_win_features, project_drift_to_exotic,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

VENUE             = LIVE_VENUE
BANKROLL          = LIVE_BANKROLL
KELLY_FRAC        = 0.25
BASE_HURDLE       = 0.02
LONGSHOT_BUF      = 0.015
LONGSHOT_D        = 15.0
PER_BET_CAP       = 0.02
RACE_CAP_BASE     = 0.04
MASTER_RACE_CAP   = 0.06
MIN_STAKE_ABS     = 10.0
TOP_N_HORSES_EXO  = 8
MAX_ODDS_WIN      = 25.0

THETA_2 = 0.8824
THETA_3 = 0.7760

CONFIG_PATH       = artifact('live_config.json')
SHRINKAGE_DEFAULT = 0.75

def _load_live_config(path: str = CONFIG_PATH) -> dict:
    cfg: dict = {'shrinkage': SHRINKAGE_DEFAULT, 'theta_2': THETA_2, 'theta_3': THETA_3, 'drift_stats': None}
    if not os.path.exists(path): return cfg
    try:
        with open(path) as f: data = json.load(f)
        if 'shrinkage_stratified' in data: cfg['shrinkage'] = data['shrinkage_stratified']
        elif 'shrinkage' in data: cfg['shrinkage'] = float(min(max(data['shrinkage'], 0.6), 0.95))
        if 'theta_2' in data: cfg['theta_2'] = float(data['theta_2'])
        if 'theta_3' in data: cfg['theta_3'] = float(data['theta_3'])
        if 'drift_stats' in data and isinstance(data['drift_stats'], dict):
            ds = {}
            for pool, stats in data['drift_stats'].items():
                try: ds[pool.upper()] = DriftStats(mu_R=float(stats['mu_R']), sigma_R=float(stats['sigma_R']), median_R=float(stats.get('median_R', stats['mu_R'])))
                except (KeyError, TypeError, ValueError): continue
            if ds: cfg['drift_stats'] = ds
    except Exception as e:
        pass
    return cfg

def _load_drift_forecaster() -> DriftForecaster | None:
    path = artifact('drift_forecaster.pkl')
    if not os.path.exists(path): return None
    try: return DriftForecaster(model_path=path).load()
    except Exception as e: return None

_LIVE_CFG = _load_live_config()
SHRINKAGE: float = _LIVE_CFG['shrinkage']
DRIFT_OVERRIDE: dict | None = _LIVE_CFG['drift_stats']
DRIFT_FORECASTER: DriftForecaster | None = _load_drift_forecaster()

r_cache = redis_client()

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
                start = datetime.datetime.strptime(meta['time'], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
                if start - datetime.timedelta(minutes=30) <= now <= start + datetime.timedelta(minutes=3): active.append(i)
            except ValueError: continue
    return active

def parse_live_race_data(venue, race_no):
    win_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:WIN")
    pla_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:PLA")
    runners_raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:runners")
    if not win_raw or not runners_raw: return None
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
        target = next((r for r in races if str(r.get('no')).lstrip('0') == str(race_no)), None)
        if not target: return None
        entries = []
        for r in target.get('runners', []):
            hn = str(r.get('no')).lstrip('0')
            if hn not in win_dict: continue
            entries.append({
                'horse_no': hn, 'horse_code': r.get('horse', {}).get('code', 'UNKNOWN'),
                'jockey': r.get('jockey', {}).get('name', 'UNKNOWN'), 'draw': int(r.get('barrierDrawNumber') or 0),
                'actual_weight': float(r.get('handicapWeight') or 120), 'live_odds': win_dict[hn], 'live_pla_odds': pla_dict.get(hn, 0.0),
            })
        return entries
    except Exception as e:
        return None

def get_live_exotic_odds(venue, race_no, pool_type="QIN"):
    odds_dict = {}
    raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:{pool_type}")
    if not raw: return odds_dict
    try:
        data = json.loads(raw)
        meetings = data.get('data', {}).get('raceMeetings', [])
        if not meetings: return odds_dict
        target = None
        for p in meetings[0].get('pmPools', []):
            if p.get('oddsType') == pool_type:
                races = p.get('leg', {}).get('races', []) or p.get('races', [])
                if int(race_no) in races:
                    target = p; break
        if not target: return odds_dict
        for node in target.get('oddsNodes', []):
            val = node.get('value') or node.get('oddsValue')
            combo = node.get('combination') or node.get('combString')
            if val is not None and combo:
                parts = str(combo).replace(',', '-').split('-')
                clean = "-".join(str(int(p)) for p in parts if str(p).strip())
                try: odds_dict[clean] = float(val)
                except ValueError: pass
    except Exception as e:
        pass
    return odds_dict

def _p_order_by_idx(p_arr, i1, i2, i3=None):
    p1 = p_arr[i1]; sum_t2 = np.sum(p_arr ** THETA_2) - (p1 ** THETA_2)
    if sum_t2 <= 0: return 0.0
    p2 = p_arr[i2]; p_exact_2 = p1 * ((p2 ** THETA_2) / sum_t2)
    if i3 is None: return p_exact_2
    p3 = p_arr[i3]; sum_t3 = np.sum(p_arr ** THETA_3) - (p1 ** THETA_3) - (p2 ** THETA_3)
    if sum_t3 <= 0: return 0.0
    return p_exact_2 * ((p3 ** THETA_3) / sum_t3)

def p_quinella(p_arr, i, j): return _p_order_by_idx(p_arr, i, j) + _p_order_by_idx(p_arr, j, i)
def p_quinella_place(p_arr, i, j):
    n = len(p_arr); total = 0.0
    for k in range(n):
        if k == i or k == j: continue
        total += _p_order_by_idx(p_arr, i, j, k) + _p_order_by_idx(p_arr, j, i, k) + _p_order_by_idx(p_arr, i, k, j) + _p_order_by_idx(p_arr, j, k, i) + _p_order_by_idx(p_arr, k, i, j) + _p_order_by_idx(p_arr, k, j, i)
    return total
def p_trio(p_arr, i, j, k): return sum(_p_order_by_idx(p_arr, *perm) for perm in itertools.permutations([i, j, k]))
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

def _resolve_pool_drift(pool: str, override: Mapping[str, DriftStats] | None) -> Mapping[str, DriftStats] | None:
    return override

def _size_drift_aware(p_raw: float, odds: float, pool: str, drift: Mapping[str, DriftStats] | None, eff_shrinkage: float) -> tuple[float, float]:
    if pd.isna(odds) or odds <= 1.0 or p_raw <= 0: return 0.0, 0.0
    ev_eff = shrunk_ev(p_raw, odds, eff_shrinkage, pool=pool, drift_override=drift)
    
    pool_base_hurdle = 0.02
    pool_lambda_d = 4.0
    if pool == 'PLA':
        if odds < 3.0: pool_base_hurdle = 0.08
    elif pool in ['QIN', 'QPL']:
        pool_lambda_d = 7.0 
    elif pool == 'TRI':
        pool_lambda_d = 4.0
        
    hurdle = get_ev_hurdle(odds, pool_base_hurdle, LONGSHOT_BUF, LONGSHOT_D, pool=pool, lambda_d=pool_lambda_d, drift_override=drift)
    if ev_eff < hurdle: return 0.0, ev_eff
    stake = fractional_kelly_stake(p_raw, odds, BANKROLL, kelly_fraction=KELLY_FRAC, shrinkage=eff_shrinkage, per_bet_cap=PER_BET_CAP, min_stake_abs=MIN_STAKE_ABS, pool=pool, drift_override=drift)
    return stake, ev_eff

def build_exotic_table(p_arr_exo: np.ndarray, horse_nos: list[str], live_odds_dict: dict, sim_pool_probs: dict, comb_len: int, pool: str, win_drift_df: pd.DataFrame | None, p_pub: np.ndarray) -> pd.DataFrame:
    if not live_odds_dict: return pd.DataFrame()
    n = len(p_arr_exo)
    top_indices = np.argsort(-p_arr_exo)[:min(TOP_N_HORSES_EXO, n)]
    from hkjc_engine.models.betting_policy import lookup_shrinkage

    rows: list[dict] = []
    for combo_idx in itertools.combinations(top_indices, comb_len):
        h_nums = sorted(int(horse_nos[i]) for i in combo_idx)
        key = "-".join(str(x) for x in h_nums)
        if key not in live_odds_dict: continue
        odds = float(live_odds_dict[key])
        p_model = sim_pool_probs.get(key, 0.0)
        if p_model <= 0: continue

        # --- THE EXOTIC VETO (Syndicate Trap Detection) ---
        if pool in ['QIN', 'QPL']:
            fair_odds = 1.0 / p_model
            if odds < fair_odds * 0.65: stake = 0
            if odds > fair_odds * 2.0: stake = 0

        if win_drift_df is not None and not win_drift_df.empty:
            combo_drift = project_drift_to_exotic(win_drift_df, h_nums)
            override = {pool: DriftStats(**combo_drift)}
        else: override = None

        min_p_pub = float(min(p_pub[idx] for idx in combo_idx))
        eff_shr = lookup_shrinkage(min_p_pub, SHRINKAGE) ** comb_len

        stake, ev_eff = _size_drift_aware(p_model, odds, pool=pool, drift=override, eff_shrinkage=eff_shr)

        if pool in ['QIN', 'QPL']:
            fair_odds = 1.0 / p_model
            # if odds < fair_odds * 0.65 or odds > fair_odds * 2.0:
            #     stake = 0.0

        rows.append({'combo': key, 'live': odds, 'fair': round(1.0 / p_model, 1), 'p_model': p_model, 'ev': ev_eff, 'stake': stake, 'eff_shrinkage': eff_shr})

    df = pd.DataFrame(rows)
    if df.empty: return df
    qual = df['stake'] > 0
    if qual.any():
        df.loc[qual, 'stake'] = cap_simultaneous_stakes(df.loc[qual, 'stake'].values, BANKROLL, race_cap=RACE_CAP_BASE, pool=pool, drift_override=DRIFT_OVERRIDE)
        df.loc[df['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
    return df.sort_values('ev', ascending=False).reset_index(drop=True)

def handle_discord_pool(venue, race_no, pool_name, is_empty, message_content):
    if not WEBHOOK_URL:
        if not is_empty:
            print(f"\n--- [{venue} R{race_no} {pool_name}] (no Discord) ---")
            print(message_content)
        return
        
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    msg_key = f"discord_msg_id:{today_str}:{venue}:{race_no}:{pool_name}"
    existing = r_cache.get(msg_key)
    if is_empty and not existing: return
    
    try:
        if existing:
            wh = DiscordWebhook(url=WEBHOOK_URL.strip('"\''), id=existing, content=message_content)
            resp = wh.edit()
            if resp.status_code == 404:
                wh = DiscordWebhook(url=WEBHOOK_URL.strip('"\''), content=message_content)
                resp = wh.execute()
                if resp.status_code in (200, 204): 
                    r_cache.set(msg_key, resp.json()['id'])
                else: 
                    print(f"Discord Error (New after 404): {resp.status_code} - {resp.text}")
            elif resp.status_code not in (200, 204):
                print(f"Discord Error (Edit): {resp.status_code} - {resp.text}")
        else:
            wh = DiscordWebhook(url=WEBHOOK_URL.strip('"\''), content=message_content)
            resp = wh.execute()
            if resp.status_code in (200, 204): 
                r_cache.set(msg_key, resp.json()['id'])
            else: 
                print(f"Discord Error (New): {resp.status_code} - {resp.text}")
    except Exception as e: 
        print(f"Discord Webhook Execution Error: {e}")

def send_to_discord(venue, race_no, meta, df_win, df_pla, df_qin, df_qpl, df_tri, is_closing=False):
    time_str = meta.get('time', 'Unknown') if isinstance(meta, dict) else 'Unknown'
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    tag = " [CLOSING SNAPSHOT]" if is_closing else ""
    header = f"**{'='*10} {venue} R{race_no} @ {time_str}{tag} {'='*10}**"

    if all(d.empty for d in [df_win, df_pla, df_qin, df_qpl, df_tri]): return

    def _fmt_stake(x): return f"${int(x)}" if x > 0 else "-"

    def _render_singles(df, limit=None):
        if df.empty: return "(no runners)\n"
        show = df.copy().sort_values('horse', ascending=True) if limit is None else df.head(limit).copy()
        show['Odds_str']  = show['odds'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'horse': 'No.', 'code': 'Code', 'Odds_str': 'Odds', 'Fair_str': 'FAIR', 'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols)].rename(columns=cols).to_string(index=False) + "\n"

    def _render_exotic(df, label, limit=20):
        if df.empty: return "(no live odds yet)\n"
        bets = df[df['stake'] > 0]
        non_bets = df[df['stake'] <= 0].sort_values('fair', ascending=True)
        pad_size = max(0, limit - len(bets))
        show = pd.concat([bets, non_bets.head(pad_size)])
        show = show.sort_values('fair', ascending=True).copy()
        show['Live_str']  = show['live'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'combo': label, 'Live_str': 'Live', 'Fair_str': 'Fair', 'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols)].rename(columns=cols).to_string(index=False) + "\n"

    for label, df_, limit in [("WIN", df_win, None), ("PLA", df_pla, None)]:
        msg = f"{header if label == 'WIN' else f'**{venue} R{race_no} - {label} ODDS:**'}"
        if label == 'WIN': msg += f"\n*Last Updated: {now_str}*\n**WIN ODDS:**"
        msg += "\n```\n" + _render_singles(df_, limit) + "```"
        handle_discord_pool(venue, race_no, label, df_.empty, msg)

    for label, df_, limit in [("QIN", df_qin, 15), ("QPL", df_qpl, 15), ("TRI", df_tri, 20)]:
        msg = f"**{venue} R{race_no} - {label}:**\n```\n"
        msg += _render_exotic(df_, label, limit) + "```"
        handle_discord_pool(venue, race_no, label, df_.empty, msg)

def _build_win_drift_for_race(predictor, race_id_for_drift: str | None) -> pd.DataFrame:
    if DRIFT_FORECASTER is None or not race_id_for_drift: return pd.DataFrame()
    try:
        feats = build_win_features(predictor.factory.engine, [race_id_for_drift])
        if feats.empty: return pd.DataFrame()
        return DRIFT_FORECASTER.predict_win(feats)
    except Exception as e: return pd.DataFrame()

def run_prediction_for_race(predictor, venue, race_no, snap_logger, is_closing=False):
    start = time.perf_counter()
    print(f"\nFetching Live Data for {venue} Race {race_no}...")
    meta = get_dynamic_metadata(venue, race_no)
    if not meta: return
    race_class = meta.get('class', 'Unknown')
    distance   = meta.get('distance', 1200)
    rail       = meta.get('rail', 'A')
    race_time  = meta.get('time', 'Unknown')
    print(f"Processing {venue} R{race_no}: {race_class} | {distance}M | Rail: {rail} | Time: {race_time}")

    t0 = time.perf_counter()
    entries = parse_live_race_data(venue, race_no)
    if not entries: return
    live_qin = get_live_exotic_odds(venue, race_no, "QIN")
    live_qpl = get_live_exotic_odds(venue, race_no, "QPL")
    live_tri = get_live_exotic_odds(venue, race_no, "TRI")
    t1 = time.perf_counter()

    try:
        results = predictor.predict_live_race(today_class=race_class, venue=venue, distance=distance, rail_placement=rail, entries_list=entries)
    except Exception as e: return
    if results.empty or len(results) < 2: return
    
    t2 = time.perf_counter()
    print(f"R{race_no} Profiling -> Scrape: {(t1-t0)*1000:.1f}ms | Inference: {(t2-t1)*1000:.1f}ms")

    p_arr_win = results['P_model_win'].to_numpy(dtype=float)
    p_arr_pla = results['P_model_pla'].to_numpy(dtype=float)
    p_arr_exo = results['P_model_exo'].to_numpy(dtype=float)
    horse_nos = results['horse_no'].astype(str).tolist()
    codes     = results['horse_code'].tolist()
    live_odds = results['live_odds'].astype(float).to_numpy()

    p_pub_raw = 1.0 / np.maximum(live_odds, 1.0)
    p_pub = p_pub_raw / np.sum(p_pub_raw)

    pace_z_scores = results['relative_early_pace'].to_numpy(dtype=float) if 'relative_early_pace' in results.columns else np.zeros(len(p_arr_win))
    today = datetime.datetime.now().strftime('%Y%m%d')
    race_id_for_drift = f"{today}_{venue}_{race_no:02d}"
    win_drift_df = _build_win_drift_for_race(predictor, race_id_for_drift)

    from hkjc_engine.models.stern_simulator import CopulaGammaSimulator
    simulator = CopulaGammaSimulator(r_shape=2.5)
    sim_probs = simulator.simulate_exotics(p_target=p_arr_exo, horse_nos=horse_nos, pace_z_scores=pace_z_scores, n_paths=16384)

    df_win = _size_win_pool(p_arr_win, horse_nos, codes, live_odds, win_drift_df, p_pub)
    df_pla = _size_pla_pool(p_arr_pla, horse_nos, codes, results, win_drift_df, sim_probs.get('PLA', {}), p_pub)
    df_qin = build_exotic_table(p_arr_exo, horse_nos, live_qin, sim_probs.get('QIN', {}), 2, 'QIN', win_drift_df, p_pub)
    df_qpl = build_exotic_table(p_arr_exo, horse_nos, live_qpl, sim_probs.get('QPL', {}), 2, 'QPL', win_drift_df, p_pub)
    df_tri = build_exotic_table(p_arr_exo, horse_nos, live_tri, sim_probs.get('TRI', {}), 3, 'TRI', win_drift_df, p_pub)

    all_dfs = [df_win, df_pla, df_qin, df_qpl, df_tri]
    total = sum(d.loc[d['stake'] > 0, 'stake'].sum() for d in all_dfs if not d.empty)
    cap_dollars = BANKROLL * MASTER_RACE_CAP
    if total > cap_dollars > 0:
        shrink = cap_dollars / total
        for d in all_dfs:
            if d.empty: continue
            d.loc[d['stake'] > 0, 'stake'] *= shrink
            d.loc[(d['stake'] > 0) & (d['stake'] < MIN_STAKE_ABS), 'stake'] = 0.0

    _log_snapshot(snap_logger, venue, race_no, race_time, df_win, df_pla, df_qin, df_qpl, df_tri)

    def _n(d): return 0 if d.empty else int((d['stake'] > 0).sum())
    def _s(d): return 0.0 if d.empty else float(d.loc[d['stake'] > 0, 'stake'].sum())
    end = time.perf_counter()
    print(f"R{race_no}: WIN={_n(df_win)} PLA={_n(df_pla)} QIN={_n(df_qin)} QPL={_n(df_qpl)} TRI={_n(df_tri)} | total_stake=${sum(_s(d) for d in all_dfs):,.0f} | Latency: {(end-start)*1000:.1f}ms")
    try: send_to_discord(venue, race_no, meta, df_win, df_pla, df_qin, df_qpl, df_tri, is_closing=is_closing)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error formatting Discord Message: {e}")

def _size_win_pool(p_arr_win: np.ndarray, horse_nos, codes, live_odds, win_drift_df: pd.DataFrame, p_pub: np.ndarray) -> pd.DataFrame:
    from hkjc_engine.models.betting_policy import lookup_shrinkage
    shr_arr = lookup_shrinkage(p_pub, SHRINKAGE)
    rows: list[dict] = []
    
    if win_drift_df is None or win_drift_df.empty:
        keep_idx, stakes = qualify_and_size(
            p_raw=p_arr_win, odds=live_odds, bankroll=BANKROLL, kelly_fraction=KELLY_FRAC, shrinkage=shr_arr,
            base_hurdle=BASE_HURDLE, longshot_buffer=LONGSHOT_BUF, longshot_threshold=LONGSHOT_D,
            per_bet_cap=PER_BET_CAP, race_cap=RACE_CAP_BASE, min_stake_abs=MIN_STAKE_ABS, top_k=10, max_odds=MAX_ODDS_WIN, pool='WIN', drift_override=DRIFT_OVERRIDE,
        )
        stake_lookup = dict(zip([int(i) for i in keep_idx], stakes))
        for i, (hn, hc, odds) in enumerate(zip(horse_nos, codes, live_odds)):
            p = float(p_arr_win[i])
            eff_shr = float(shr_arr[i])
            ev_eff = shrunk_ev(p, float(odds), eff_shr, pool='WIN', drift_override=DRIFT_OVERRIDE)
            stake = float(stake_lookup.get(i, 0.0))
            rows.append({'horse': hn, 'code': hc, 'odds': float(odds), 'p_model': p, 'fair': 1.0 / p if p > 0 else float('inf'), 'ev': ev_eff, 'stake': stake, 'eff_shrinkage': eff_shr})
    else:
        drift_lookup = win_drift_df.set_index(win_drift_df['horse_no'].astype(str))
        for i, (hn, hc, odds) in enumerate(zip(horse_nos, codes, live_odds)):
            p = float(p_arr_win[i])
            eff_shr = float(shr_arr[i])
            if str(hn) in drift_lookup.index:
                row = drift_lookup.loc[str(hn)]
                override = {'WIN': DriftStats(mu_R=float(row['mu_R']), sigma_R=float(row['sigma_R']), median_R=float(row['median_R']))}
            else: override = DRIFT_OVERRIDE
            stake, ev_eff = _size_drift_aware(p, float(odds), pool='WIN', drift=override, eff_shrinkage=eff_shr)
            if odds > MAX_ODDS_WIN: stake = 0.0
            rows.append({'horse': hn, 'code': hc, 'odds': float(odds), 'p_model': p, 'fair': 1.0 / p if p > 0 else float('inf'), 'ev': ev_eff, 'stake': stake, 'eff_shrinkage': eff_shr})

        df_tmp = pd.DataFrame(rows)
        qmask = df_tmp['stake'] > 0
        if qmask.any():
            df_tmp.loc[qmask, 'stake'] = cap_simultaneous_stakes(df_tmp.loc[qmask, 'stake'].values, BANKROLL, race_cap=RACE_CAP_BASE, pool='WIN', drift_override=DRIFT_OVERRIDE)
            df_tmp.loc[df_tmp['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
        return df_tmp.sort_values('ev', ascending=False).reset_index(drop=True)

    df_win = pd.DataFrame(rows).sort_values('ev', ascending=False).reset_index(drop=True)
    return df_win

def _size_pla_pool(p_arr_pla: np.ndarray, horse_nos, codes, results: pd.DataFrame, win_drift_df: pd.DataFrame, sim_pla_probs: dict, p_pub: np.ndarray) -> pd.DataFrame:
    from hkjc_engine.models.betting_policy import lookup_shrinkage
    shr_arr = lookup_shrinkage(p_pub, SHRINKAGE)
    pla_rows = []
    
    for i, (hn, hc) in enumerate(zip(horse_nos, codes)):
        p_odds = float(results['live_pla_odds'].iloc[i])
        if pd.isna(p_odds) or p_odds < 1.0: continue
        
        # We blend the PLA stacker output with Copula PLA probabilities for stability if Copula goes haywire
        pp_model = sim_pla_probs.get(str(hn), 0.0)
        
        if win_drift_df is not None and not win_drift_df.empty:
            single = project_drift_to_exotic(win_drift_df, [int(hn)])
            override = {'PLA': DriftStats(**single)}
        else: override = DRIFT_OVERRIDE

        eff_shr = float(shr_arr[i])
        stake, ev_eff = _size_drift_aware(pp_model, p_odds, pool='PLA', drift=override, eff_shrinkage=eff_shr)
        
        pla_rows.append({
            'horse': hn, 'code': hc, 'odds': p_odds, 'p_model': pp_model, 'fair': 1.0 / pp_model if pp_model > 0 else float('inf'),
            'ev': ev_eff, 'stake': stake, 'eff_shrinkage': eff_shr,
        })

    df_pla = pd.DataFrame(pla_rows)
    if df_pla.empty: return df_pla
    qmask = df_pla['stake'] > 0
    if qmask.any():
        df_pla.loc[qmask, 'stake'] = cap_simultaneous_stakes(df_pla.loc[qmask, 'stake'].values, BANKROLL, race_cap=RACE_CAP_BASE, pool='PLA', drift_override=DRIFT_OVERRIDE)
        df_pla.loc[df_pla['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
    return df_pla.sort_values('ev', ascending=False).reset_index(drop=True)

def _log_snapshot(snap_logger, venue, race_no, race_time, df_win, df_pla, df_qin, df_qpl, df_tri):
    try:
        recs = []
        for d, pool, combo_col in [(df_win, "WIN", "horse"), (df_pla, "PLA", "horse"), (df_qin, "QIN", "combo"), (df_qpl, "QPL", "combo"), (df_tri, "TRI", "combo")]:
            if d.empty: continue
            for _, row in d.iterrows():
                recs.append({
                    "pool": pool, "combination": str(row[combo_col]), "live_odds": float(row["odds"]) if pool in ("WIN", "PLA") else float(row["live"]),
                    "p_raw": float(row["p_model"]), "p_shrunk": min(float(row["p_model"]) * float(row.get("eff_shrinkage", SHRINKAGE)), 1 - 1e-9), "stake": float(row["stake"]),
                })
        if not recs: return
        race_id = f"{datetime.datetime.now().strftime('%Y%m%d')}_{venue}_{race_no:02d}"
        try: race_off_dt = datetime.datetime.strptime(race_time, "%H:%M").replace(year=datetime.datetime.now().year, month=datetime.datetime.now().month, day=datetime.datetime.now().day, tzinfo=datetime.timezone.utc)
        except Exception: race_off_dt = datetime.datetime.now(datetime.timezone.utc)
        cfg = {"theta_2": _LIVE_CFG['theta_2'], "theta_3": _LIVE_CFG['theta_3'], "shrinkage": SHRINKAGE, "drift_forecaster_used": DRIFT_FORECASTER is not None}
        if DRIFT_OVERRIDE: cfg['drift_stats'] = {p: {'mu_R': s.mu_R, 'sigma_R': s.sigma_R, 'median_R': s.median_R} for p, s in DRIFT_OVERRIDE.items()}
        snap_logger.log_snapshot(race_id=race_id, race_off_time=race_off_dt, bankroll=BANKROLL, recommendations=recs, config=cfg)
    except Exception as e: pass

if __name__ == "__main__":
    predictor = LiveRacePredictor(DB_URL)
    snap_logger = SnapshotLogger(DB_URL)
    closed_processed: set[int] = set()

    print(f"--- LIVE BOT START | venue={VENUE} | shrinkage={SHRINKAGE:.3f} | theta=[{_LIVE_CFG['theta_2']}, {_LIVE_CFG['theta_3']}] | kelly=alpha{KELLY_FRAC} | drift_forecaster={'ON' if DRIFT_FORECASTER else 'OFF'} ---")

    while True:
        try:
            active = get_active_race(VENUE)
            for r_no in range(1, 12):
                if r_no in closed_processed: continue
                if get_race_status(VENUE, r_no) == 'CLOSED':
                    try: run_prediction_for_race(predictor, VENUE, r_no, snap_logger, is_closing=True)
                    except Exception as e: pass
                    closed_processed.add(r_no)
            if not active:
                time.sleep(15)
                continue
            for r_no in active:
                if r_no in closed_processed: continue
                if get_race_status(VENUE, r_no) == 'CLOSED': continue
                try: run_prediction_for_race(predictor, VENUE, r_no, snap_logger)
                except Exception as e: pass
            meta_first = get_dynamic_metadata(VENUE, active[0])
            tts = 999
            if meta_first and 'time' in meta_first:
                try:
                    now = datetime.datetime.now()
                    start_t = datetime.datetime.strptime(meta_first['time'], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
                    tts = (start_t - now).total_seconds()
                except Exception: pass
            time.sleep(3 if -180 < tts <= 120 else 15)
        except KeyboardInterrupt:
            break
        except Exception as e:
            time.sleep(10)