import redis
import json
import os
import pandas as pd
import numpy as np
import logging
import datetime
import itertools
import time
from discord_webhook import DiscordWebhook
from hkjc_engine.config import (
    redis_client,
    DB_URL,
    WEBHOOK_URL,
    LIVE_VENUE,
    LIVE_BANKROLL,
)
from hkjc_engine.live.snapshot_logger import SnapshotLogger
from hkjc_engine.live.predictor import LiveRacePredictor
from hkjc_engine.models.betting_policy import (
    get_ev_hurdle,
    fractional_kelly_stake,
    cap_simultaneous_stakes,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Configuration
VENUE       = LIVE_VENUE
BANKROLL          = LIVE_BANKROLL
KELLY_FRAC        = 0.25
BASE_HURDLE       = 0.02
LONGSHOT_BUF      = 0.015
LONGSHOT_D        = 15.0
PER_BET_CAP       = 0.02
RACE_CAP_WIN      = 0.04       # Smoczyński-Tomkins cap for all WIN bets in a race
POOL_CAP_EXO      = 0.03       # per-pool cap for QIN / QPL / TRI / PLA
MASTER_RACE_CAP   = 0.06       # cross-pool master cap
MIN_STAKE_ABS     = 10.0
TOP_N_HORSES_EXO  = 8          # restrict exotic combinatorics to top-N by P_model
MAX_ODDS_WIN      = 25.0

# --- MLE-calibrated Benter discount exponents ---
THETA_2 = 0.8824
THETA_3 = 0.7760

# --- Dynamic shrinkage loaded from live_config.json ---
CONFIG_PATH = 'live_config.json'
SHRINKAGE_DEFAULT = 0.75

def load_shrinkage(path=CONFIG_PATH, default=SHRINKAGE_DEFAULT):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                cfg = json.load(f)
            v = float(cfg.get('shrinkage', default))
            v = min(max(v, 0.60), 0.95)
            logging.info(f"Loaded SHRINKAGE = {v:.4f} from {path}")
            return v
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
    logging.info(f"Using default SHRINKAGE = {default}")
    return default

SHRINKAGE = load_shrinkage()

r_cache = redis_client()


# Redis state helpers
def get_dynamic_metadata(venue, race_no):
    data = r_cache.get(f"live_race_metadata:{venue}:{race_no}")
    return json.loads(data) if data else None


def get_race_status(venue, race_no):
    return r_cache.get(f"race_status:{venue}:{race_no}")


def get_active_race(venue):
    """
    Preserves the original bot's multi-race scan (races 1-11) — any racewhose scheduled jump is within [-3min, +30min] is considered active.
    """
    active_races = []
    now = datetime.datetime.now()
    for i in range(1, 12):
        meta = get_dynamic_metadata(venue, i)
        if meta and 'time' in meta and ":" in meta['time']:
            try:
                time_str = meta['time']
                start_time = datetime.datetime.strptime(time_str, "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
                window_start = start_time - datetime.timedelta(minutes=30)
                window_end   = start_time + datetime.timedelta(minutes=3)
                if window_start <= now <= window_end:
                    active_races.append(i)
            except ValueError:
                continue
    return active_races


def seconds_to_jump(venue, race_no):
    meta = get_dynamic_metadata(venue, race_no)
    if not meta or 'time' not in meta or ':' not in meta['time']:
        return 9999
    try:
        now = datetime.datetime.now()
        start = datetime.datetime.strptime(meta['time'], "%H:%M").replace(
            year=now.year, month=now.month, day=now.day)
        return (start - now).total_seconds()
    except ValueError:
        return 9999


def parse_live_race_data(venue, race_no):
    win_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:WIN")
    pla_raw     = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:PLA")
    runners_raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:runners")

    if not win_raw or not runners_raw:
        return None

    try:
        win_odds_dict = {}
        pla_odds_dict = {}

        win_data  = json.loads(win_raw)
        win_pools = win_data.get('data', {}).get('raceMeetings', [])[0].get('pmPools', [])
        for p in win_pools:
            if p.get('oddsType') == 'WIN':
                for node in p.get('oddsNodes', []):
                    combo = str(node.get('combString')).lstrip('0')
                    val   = node.get('oddsValue')
                    if val:
                        try:
                            win_odds_dict[combo] = float(val)
                        except ValueError:
                            pass

        if pla_raw:
            pla_data  = json.loads(pla_raw)
            pla_pools = pla_data.get('data', {}).get('raceMeetings', [])[0].get('pmPools', [])
            for p in pla_pools:
                if p.get('oddsType') == 'PLA':
                    for node in p.get('oddsNodes', []):
                        combo = str(node.get('combString')).lstrip('0')
                        val   = node.get('oddsValue')
                        if val:
                            try:
                                pla_odds_dict[combo] = float(val)
                            except ValueError:
                                pass

        # Map to Runners
        runners_data = json.loads(runners_raw)
        entries_list = []
        races  = runners_data.get('data', {}).get('raceMeetings', [])[0].get('races', [])
        target = next(
            (r for r in races if str(r.get('no')).lstrip('0') == str(race_no)),
            None,
        )
        if target:
            for r in target.get('runners', []):
                horse_no = str(r.get('no')).lstrip('0')
                if horse_no not in win_odds_dict:
                    continue
                entries_list.append({
                    'horse_no':      horse_no,
                    'horse_code':    r.get('horse', {}).get('code', 'UNKNOWN'),
                    'jockey':        r.get('jockey', {}).get('name', 'UNKNOWN'),
                    'draw':          int(r.get('barrierDrawNumber') or 0),
                    'actual_weight': float(r.get('handicapWeight') or 120),
                    'live_odds':     win_odds_dict[horse_no],
                    'live_pla_odds': pla_odds_dict.get(horse_no, 0.0),
                })
        return entries_list

    except Exception as e:
        logging.error(f"Parse Error: {e}")
        return None


def get_live_exotic_odds(venue, race_no, pool_type="QIN"):
    odds_dict = {}
    odds_raw = r_cache.get(f"live_odds_raw:{venue}:{race_no}:odds:{pool_type}")
    if not odds_raw:
        return odds_dict

    try:
        data = json.loads(odds_raw)
        meetings = data.get('data', {}).get('raceMeetings', [])
        if not meetings:
            return odds_dict

        pools = meetings[0].get('pmPools', [])
        target_pool = None
        for p in pools:
            if p.get('oddsType') == pool_type:
                races = p.get('leg', {}).get('races', [])
                if not races:
                    races = p.get('races', [])
                if int(race_no) in races:
                    target_pool = p
                    break

        if target_pool:
            for node in target_pool.get('oddsNodes', []):
                val   = node.get('value')       or node.get('oddsValue')
                combo = node.get('combination') or node.get('combString')
                if val is not None and combo:
                    parts = str(combo).replace(',', '-').split('-')
                    clean = "-".join(str(int(p)) for p in parts if str(p).strip())
                    try:
                        odds_dict[clean] = float(val)
                    except ValueError:
                        pass

    except Exception as e:
        logging.error(f"Failed to parse {pool_type} API odds: {e}")

    return odds_dict


def _p_order_by_idx(p_arr, i1, i2, i3=None):
    """P(i1 1st, i2 2nd [, i3 3rd]) via Henery/Lo-Bacon-Shone discounts."""
    p1 = p_arr[i1]; p2 = p_arr[i2]
    sum_t2 = np.sum(p_arr ** THETA_2) - (p1 ** THETA_2)
    if sum_t2 <= 0:
        return 0.0
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
    n = len(p_arr)
    total = 0.0
    for k in range(n):
        if k == i or k == j:
            continue
        total += _p_order_by_idx(p_arr, i, j, k)
        total += _p_order_by_idx(p_arr, j, i, k)
        total += _p_order_by_idx(p_arr, i, k, j)
        total += _p_order_by_idx(p_arr, j, k, i)
        total += _p_order_by_idx(p_arr, k, i, j)
        total += _p_order_by_idx(p_arr, k, j, i)
    return total


def p_trio(p_arr, i, j, k):
    total = 0.0
    for a, b, c in itertools.permutations([i, j, k]):
        total += _p_order_by_idx(p_arr, a, b, c)
    return total


def p_place(p_arr, i):
    n = len(p_arr)
    p_1st = p_arr[i]
    p_2nd = 0.0
    p_3rd = 0.0
    for j in range(n):
        if j == i:
            continue
        p_2nd += _p_order_by_idx(p_arr, j, i)
        for k in range(n):
            if k == i or k == j:
                continue
            p_3rd += _p_order_by_idx(p_arr, j, k, i)
    return p_1st + p_2nd + p_3rd


def _size_single_bet(p_raw, odds, bankroll):
    """Shrunk EV → hurdle check → fractional Kelly."""
    if pd.isna(odds) or odds <= 1.0 or p_raw <= 0:
        return 0.0
    p_adj = min(p_raw * SHRINKAGE, 1 - 1e-9)
    ev_adj = p_adj * odds - 1.0
    hurdle = get_ev_hurdle(odds, BASE_HURDLE, LONGSHOT_BUF, LONGSHOT_D)
    if ev_adj < hurdle:
        return 0.0
    return fractional_kelly_stake(
        p_raw, odds, bankroll,
        kelly_fraction=KELLY_FRAC,
        shrinkage=SHRINKAGE,
        per_bet_cap=PER_BET_CAP,
        min_stake_abs=MIN_STAKE_ABS,
    )


def build_exotic_table(p_arr, horse_nos, live_odds_dict, calc_func, comb_len):
    """
    Top-N-by-P_model combinatoric search.
    Returns ALL combinations that have a live-odds quote (regardless of EV), with a 'stake' column that is >0 only for combos that pass the fractional-Kelly hurdle + per-pool Smoczyński-Tomkins cap.
    """
    if not live_odds_dict:
        return pd.DataFrame()
    n = len(p_arr)
    top_n = min(TOP_N_HORSES_EXO, n)
    top_indices = np.argsort(-p_arr)[:top_n]

    rows = []
    for combo_idx in itertools.combinations(top_indices, comb_len):
        h_nums = sorted([int(horse_nos[i]) for i in combo_idx])
        key = "-".join(str(x) for x in h_nums)
        if key not in live_odds_dict:
            continue
        odds = live_odds_dict[key]
        p_model = calc_func(p_arr, *combo_idx)
        if p_model <= 0:
            continue
        p_adj   = min(p_model * SHRINKAGE, 1 - 1e-9)
        ev_adj  = p_adj * odds - 1.0
        # Raw Kelly: only actually staked if hurdle is met
        stake = _size_single_bet(p_model, odds, BANKROLL)
        rows.append({
            'combo':   key,
            'live':    odds,
            'fair':    round(1.0 / p_model, 1),
            'p_model': p_model,
            'ev':      ev_adj,
            'stake':   stake,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Per-pool cap only to qualifying stakes
    qual_mask = df['stake'] > 0
    if qual_mask.any():
        qual_stakes = df.loc[qual_mask, 'stake'].values
        capped = cap_simultaneous_stakes(qual_stakes, BANKROLL, POOL_CAP_EXO)
        df.loc[qual_mask, 'stake'] = capped
        df.loc[df['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0

    return df.sort_values('ev', ascending=False).reset_index(drop=True)


def handle_discord_pool(venue, race_no, pool_name, is_empty, message_content):
    # If no webhook is configured, log to stdout and return.
    if not WEBHOOK_URL:
        if not is_empty:
            print(f"\n--- [{venue} R{race_no} {pool_name}] (no Discord configured) ---")
            print(message_content)
        return

    msg_key = f"discord_msg_id:{venue}:{race_no}:{pool_name}"
    existing_msg_id = r_cache.get(msg_key)
    if is_empty and not existing_msg_id:
        return
    try:
        if existing_msg_id:
            webhook = DiscordWebhook(url=WEBHOOK_URL, id=existing_msg_id, content=message_content)
            response = webhook.edit()
            if response.status_code == 404:
                webhook = DiscordWebhook(url=WEBHOOK_URL, content=message_content)
                response = webhook.execute()
                if response.status_code in (200, 204):
                    r_cache.set(msg_key, response.json()['id'])
        else:
            webhook = DiscordWebhook(url=WEBHOOK_URL, content=message_content)
            response = webhook.execute()
            if response.status_code in (200, 204):
                r_cache.set(msg_key, response.json()['id'])
    except Exception as e:
        logging.error(f"Webhook execution failed for {pool_name}: {e}")


def send_to_discord(venue, race_no, meta, df_win, df_pla, df_qin, df_qpl, df_tri, is_closing=False):
    """
    v1-style render: show ALL runners/combos in each pool table sorted by EV. Stake column reads $N where Kelly qualified, '-' elsewhere.
    """
    time_str = meta.get('time', 'Unknown') if isinstance(meta, dict) else 'Unknown'
    now_str  = datetime.datetime.now().strftime('%H:%M:%S')
    tag      = " [CLOSING SNAPSHOT]" if is_closing else ""
    header   = f"**{'='*10} {venue} R{race_no} @ {time_str}{tag} {'='*10}**"

    all_empty = all(d.empty for d in [df_win, df_pla, df_qin, df_qpl, df_tri])
    if all_empty:
        logging.info(f"R{race_no}: no data in any pool yet.")
        return

    def _fmt_stake(x):
        return f"${int(x)}" if x > 0 else "-"

    def _render_singles(df, limit=None):
        """Renders WIN / PLA tables with No. Code Odds FAIR EV Stake columns."""
        if df.empty:
            return "(no runners)\n"
        show = df.copy()
        if limit is not None:
            show = show.head(limit)
        show['Odds_str']  = show['odds'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}" if x != float('inf') else "∞")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'horse': 'No.', 'code': 'Code',
                'Odds_str': 'Odds', 'Fair_str': 'FAIR',
                'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols.keys())].rename(columns=cols).to_string(index=False) + "\n"

    def _render_exotic(df, label, limit=20):
        """Renders QIN / QPL / TRI tables with Combo Live Fair EV Stake columns."""
        if df.empty:
            return "(no live odds yet)\n"
        show = df.head(limit).copy()
        show['Live_str']  = show['live'].apply(lambda x: f"{x:.1f}")
        show['Fair_str']  = show['fair'].apply(lambda x: f"{x:.1f}")
        show['EV_str']    = show['ev'].apply(lambda x: f"{x:+.1%}")
        show['Stake_str'] = show['stake'].apply(_fmt_stake)
        cols = {'combo': label,
                'Live_str': 'Live', 'Fair_str': 'Fair',
                'EV_str': 'EV', 'Stake_str': 'Stake'}
        return show[list(cols.keys())].rename(columns=cols).to_string(index=False) + "\n"

    # WIN — always send
    win_msg = f"{header}\n*Last Updated: {now_str}*\n**WIN ODDS:**\n```\n"
    win_msg += _render_singles(df_win)
    win_msg += "```"
    handle_discord_pool(venue, race_no, "WIN", df_win.empty, win_msg)

    # PLA — always send
    pla_msg = f"**{venue} R{race_no} - PLACE ODDS:**\n```\n"
    pla_msg += _render_singles(df_pla)
    pla_msg += "```"
    handle_discord_pool(venue, race_no, "PLA", df_pla.empty, pla_msg)

    # QIN — top 15 combos by EV
    qin_msg = f"**{venue} R{race_no} - QIN:**\n```\n"
    qin_msg += _render_exotic(df_qin, 'QIN', limit=15)
    qin_msg += "```"
    handle_discord_pool(venue, race_no, "QIN", df_qin.empty, qin_msg)

    # QPL — top 15 combos by EV
    qpl_msg = f"**{venue} R{race_no} - QPL:**\n```\n"
    qpl_msg += _render_exotic(df_qpl, 'QPL', limit=15)
    qpl_msg += "```"
    handle_discord_pool(venue, race_no, "QPL", df_qpl.empty, qpl_msg)

    # TRI — top 20 combos by EV
    tri_msg = f"**{venue} R{race_no} - TRI:**\n```\n"
    tri_msg += _render_exotic(df_tri, 'TRI', limit=20)
    tri_msg += "```"
    handle_discord_pool(venue, race_no, "TRI", df_tri.empty, tri_msg)


def run_prediction_for_race(predictor, venue, race_no, snap_logger, is_closing=False):
    print(f"\nFetching Live Data for {venue} Race {race_no}...")
    meta = get_dynamic_metadata(venue, race_no)
    if not meta:
        print(f"Skipping R{race_no}: No metadata in Redis.")
        return

    race_class = meta.get('class', 'Unknown')
    distance   = meta.get('distance', 1200)
    rail       = meta.get('rail', 'A')
    race_time  = meta.get('time', 'Unknown')
    print(f"Processing {venue} R{race_no}: {race_class} | {distance}M | Rail: {rail} | Time: {race_time}")

    entries = parse_live_race_data(venue, race_no)
    if not entries:
        print(f"R{race_no}: no entries parsed yet; skipping.")
        return

    live_qin = get_live_exotic_odds(venue, race_no, "QIN")
    live_qpl = get_live_exotic_odds(venue, race_no, "QPL")
    live_tri = get_live_exotic_odds(venue, race_no, "TRI")

    try:
        results = predictor.predict_live_race(
            today_class=race_class, venue=venue, distance=distance,
            rail_placement=rail, entries_list=entries,
        )
    except Exception as e:
        logging.exception(f"R{race_no}: predictor failed: {e}")
        return

    if results.empty or len(results) < 2:
        print(f"R{race_no}: predictor returned <2 rows; skipping.")
        return

    p_arr     = results['P_model'].to_numpy(dtype=float)
    horse_nos = results['horse_no'].astype(str).tolist()
    codes     = results['horse_code'].astype(str).tolist()

    # ----Win----
    win_rows = []
    for i, (p, d, hn, hc) in enumerate(zip(
            p_arr,
            results['live_odds'].to_numpy(dtype=float),
            horse_nos,
            codes)):
        if pd.isna(d) or d < 1.0:
            continue
        p_adj  = min(p * SHRINKAGE, 1 - 1e-9)
        ev_adj = p_adj * d - 1.0
        if d <= MAX_ODDS_WIN:
            stake = _size_single_bet(p, d, BANKROLL)
        else:
            stake = 0.0
        win_rows.append({
            'horse': hn, 'code': hc, 'odds': d,
            'p_model': p, 'fair': 1.0 / p if p > 0 else float('inf'),
            'ev': ev_adj, 'stake': stake,
        })
    df_win = pd.DataFrame(win_rows)
    if not df_win.empty:
        qmask = df_win['stake'] > 0
        if qmask.any():
            capped = cap_simultaneous_stakes(df_win.loc[qmask, 'stake'].values,
                                             BANKROLL, RACE_CAP_WIN)
            df_win.loc[qmask, 'stake'] = capped
            df_win.loc[df_win['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
        df_win = df_win.sort_values('ev', ascending=False).reset_index(drop=True)

    # ----Place----
    pla_rows = []
    for i, (hn, hc) in enumerate(zip(horse_nos, codes)):
        p_odds = float(results['live_pla_odds'].iloc[i])
        if pd.isna(p_odds) or p_odds < 1.0:
            continue
        pp     = p_place(p_arr, i)
        p_adj  = min(pp * SHRINKAGE, 1 - 1e-9)
        ev_adj = p_adj * p_odds - 1.0
        stake  = _size_single_bet(pp, p_odds, BANKROLL)
        pla_rows.append({
            'horse': hn, 'code': hc, 'odds': p_odds,
            'p_model': pp, 'fair': 1.0 / pp if pp > 0 else float('inf'),
            'ev': ev_adj, 'stake': stake,
        })
    df_pla = pd.DataFrame(pla_rows)
    if not df_pla.empty:
        qmask = df_pla['stake'] > 0
        if qmask.any():
            capped = cap_simultaneous_stakes(df_pla.loc[qmask, 'stake'].values,
                                             BANKROLL, POOL_CAP_EXO)
            df_pla.loc[qmask, 'stake'] = capped
            df_pla.loc[df_pla['stake'] < MIN_STAKE_ABS, 'stake'] = 0.0
        df_pla = df_pla.sort_values('ev', ascending=False).reset_index(drop=True)


    df_qin = build_exotic_table(p_arr, horse_nos, live_qin, p_quinella,       2)
    df_qpl = build_exotic_table(p_arr, horse_nos, live_qpl, p_quinella_place, 2)
    df_tri = build_exotic_table(p_arr, horse_nos, live_tri, p_trio,           3)

    # ---- Master cross-pool cap (only staked rows) ----
    all_dfs = [df_win, df_pla, df_qin, df_qpl, df_tri]
    total = sum(d.loc[d['stake'] > 0, 'stake'].sum()
                for d in all_dfs if not d.empty)
    cap   = BANKROLL * MASTER_RACE_CAP
    if total > cap > 0:
        shrink = cap / total
        for d in all_dfs:
            if not d.empty:
                d.loc[d['stake'] > 0, 'stake'] *= shrink
                # Zero out any rows shrunk below min
                d.loc[(d['stake'] > 0) & (d['stake'] < MIN_STAKE_ABS), 'stake'] = 0.0
    # ---- Persist recommendations to snapshot_recommendations ----
    try:
        recs = []
        for d, pool, combo_col in [(df_win, "WIN", "horse"),
                                    (df_pla, "PLA", "horse"),
                                    (df_qin, "QIN", "combo"),
                                    (df_qpl, "QPL", "combo"),
                                    (df_tri, "TRI", "combo")]:
            if d.empty:
                continue
            for _, row in d.iterrows():
                recs.append({
                    "pool": pool,
                    "combination": str(row[combo_col]),
                    "live_odds": float(row["odds"]),
                    "p_raw": float(row["p_model"]),
                    "p_shrunk": min(float(row["p_model"]) * SHRINKAGE, 1 - 1e-9),
                    "stake": float(row["stake"]),
                })
        if recs:
            race_id = f"{datetime.datetime.now().strftime('%Y%m%d')}_{venue}_{race_no:02d}"
            try:
                race_off_dt = datetime.datetime.strptime(
                    race_time, "%H:%M"
                ).replace(year=datetime.datetime.now().year,
                          month=datetime.datetime.now().month,
                          day=datetime.datetime.now().day,
                          tzinfo=datetime.timezone.utc)
            except Exception:
                race_off_dt = datetime.datetime.now(datetime.timezone.utc)

            snap_logger.log_snapshot(
                race_id=race_id,
                race_off_time=race_off_dt,
                bankroll=BANKROLL,
                recommendations=recs,
                config={"theta_2": THETA_2, "theta_3": THETA_3,
                        "shrinkage": SHRINKAGE},
            )
    except Exception as e:
        logging.error(f"Snapshot log failed for R{race_no}: {e}")

    # Console summary
    def _n_bets(d):
        return 0 if d.empty else int((d['stake'] > 0).sum())
    def _staked(d):
        return 0.0 if d.empty else float(d.loc[d['stake'] > 0, 'stake'].sum())
    print(f"R{race_no}: "
          f"WIN={_n_bets(df_win)} PLA={_n_bets(df_pla)} "
          f"QIN={_n_bets(df_qin)} QPL={_n_bets(df_qpl)} TRI={_n_bets(df_tri)} "
          f"| total_stake=${sum(_staked(d) for d in all_dfs):,.0f}")

    # Discord
    try:
        send_to_discord(venue, race_no, meta,
                        df_win, df_pla, df_qin, df_qpl, df_tri,
                        is_closing=is_closing)
    except Exception as e:
        logging.error(f"Discord dispatch failed for R{race_no}: {e}")

if __name__ == "__main__":
    predictor = LiveRacePredictor(DB_URL)
    snap_logger = SnapshotLogger(DB_URL)
    closed_processed = set()

    print(f"--- LIVE BOT START | venue={VENUE} | shrinkage={SHRINKAGE:.3f} "
          f"| θ=[{THETA_2}, {THETA_3}] | kelly=α{KELLY_FRAC} ---")

    while True:
        try:
            active = get_active_race(VENUE)

            # Always send a closing snapshot once per race when STOP_SELL hits
            for r_no in range(1, 12):
                if r_no in closed_processed:
                    continue
                if get_race_status(VENUE, r_no) == 'CLOSED':
                    print(f"R{r_no} STOP_SELL — sending final closing snapshot.")
                    try:
                        run_prediction_for_race(predictor, VENUE, r_no, snap_logger, is_closing=True)
                    except Exception as e:
                        logging.exception(f"Closing snapshot R{r_no} failed: {e}")
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
                    logging.exception(f"Race {r_no} failed: {e}")

            # Dynamic polling cadence
            meta_first = get_dynamic_metadata(VENUE, active[0])
            tts = 999
            if meta_first and 'time' in meta_first:
                try:
                    now = datetime.datetime.now()
                    start = datetime.datetime.strptime(
                        meta_first['time'], "%H:%M"
                    ).replace(year=now.year, month=now.month, day=now.day)
                    tts = (start - now).total_seconds()
                except Exception:
                    pass

            if -180 < tts <= 120:
                time.sleep(3)
            else:
                time.sleep(15)

        except KeyboardInterrupt:
            print("\nShutting down live bot cleanly.")
            break
        except Exception as e:
            logging.exception(f"Main loop error: {e}")
            time.sleep(10)
