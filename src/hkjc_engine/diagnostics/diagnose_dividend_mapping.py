"""
Diagnose why the historical exotics backtest is showing impossible 100% hit
rates. Hypothesis: the horse numbers in race_dividends.combination do not
match race_entries.horse_no for the same race.

Strategy:
  - For each race, look up the actual top-3 finishers (by horse_no from
    race_entries).
  - Look up the WIN dividend (= the winning horse) in race_dividends.
  - Check whether the WIN combination string equals str(actual winner).
  - If the WIN combos consistently disagree with race_entries top-1, we have
    a numbering mismatch and need to re-key dividends before the backtest.

Run:
    python -m hkjc_engine.diagnostics.diagnose_dividend_mapping
"""
from sqlalchemy import create_engine, text
import pandas as pd

from hkjc_engine.config import DB_URL


def main():
    engine = create_engine(DB_URL)

    # 100 random recent races, with their actual winners (horse_no of pos1)
    q_winners = text("""
        SELECT e.race_id, e.horse_no AS actual_winner_no, e.horse_code
        FROM race_entries e
        JOIN races r ON e.race_id = r.race_id
        WHERE e.finish_position = 1
          AND r.race_date >= '2024-01-01'
        ORDER BY r.race_date DESC
        LIMIT 100
    """)
    with engine.connect() as conn:
        winners = pd.read_sql(q_winners, conn)

    # Pull the WIN dividends for those races
    rids = tuple(winners["race_id"].tolist())
    q_win_div = text("""
        SELECT race_id, pool, combination, dividend
        FROM race_dividends
        WHERE race_id IN :rids
          AND pool IN ('WIN', 'WINNER')
    """)
    with engine.connect() as conn:
        wins = pd.read_sql(q_win_div, conn, params={"rids": rids})

    # Compare: does combination (a horse number string) match actual_winner_no?
    merged = winners.merge(wins, on="race_id", how="inner")
    merged["combo_clean"] = (
        merged["combination"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    )
    merged["actual_str"] = merged["actual_winner_no"].astype(int).astype(str)
    merged["match"] = merged["combo_clean"] == merged["actual_str"]

    n = len(merged)
    n_match = int(merged["match"].sum())
    print(f"\nWIN-dividend vs race_entries.pos1 alignment check on {n} races")
    print(f"  matched: {n_match}/{n}  ({n_match/max(n,1):.1%})")
    print()
    print("First 15 mismatches (if any):")
    print(merged[~merged["match"]][
        ["race_id", "actual_winner_no", "horse_code", "combination", "dividend"]
    ].head(15).to_string(index=False))

    if n_match / max(n, 1) < 0.95:
        print("\n>>> CONFIRMED: dividend horse numbers do NOT align with "
              "race_entries.horse_no.")
        print(">>> The backtester's combo lookup is wrong. Need to re-key.")
    elif n_match / max(n, 1) >= 0.95:
        print("\n>>> WIN dividends align with race_entries pos1. The bug is "
              "elsewhere (likely in QIN/QPL/TRI parsing or hit logic).")

    # Bonus: look at a single race in detail so we can see the data shape
    print("\n--- Single-race detail ---")
    sample_rid = winners.iloc[0]["race_id"]
    print(f"race_id = {sample_rid}")
    q_entries = text("""
        SELECT horse_no, horse_code, finish_position, win_odds
        FROM race_entries WHERE race_id = :rid
        ORDER BY finish_position NULLS LAST, horse_no
    """)
    q_divs = text("""
        SELECT pool, combination, dividend
        FROM race_dividends WHERE race_id = :rid
        ORDER BY pool, combination
    """)
    with engine.connect() as conn:
        e_df = pd.read_sql(q_entries, conn, params={"rid": sample_rid})
        d_df = pd.read_sql(q_divs, conn, params={"rid": sample_rid})
    print("\nrace_entries:")
    print(e_df.to_string(index=False))
    print("\nrace_dividends:")
    print(d_df.to_string(index=False))


if __name__ == "__main__":
    main()