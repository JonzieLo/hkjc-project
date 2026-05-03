# import pandas as pd
# from sqlalchemy import create_engine, text
# from hkjc_engine.config import DB_URL, artifact

# def generate_claude_csv():
#     print("Connecting to PostgreSQL...")
#     engine = create_engine(DB_URL)
    
#     # We only want the exotics, and we ONLY want the execution phase and the settlement phase.
#     # This reduces millions of rows down to a highly dense, LLM-friendly dataset.
#     query = """
#         SELECT 
#             race_id, 
#             pool_type, 
#             combination, 
#             odds, 
#             phase, 
#             timestamp
#         FROM live_odds_history 
#         WHERE pool_type IN ('QIN', 'QPL', 'TRI')
#           AND phase IN ('FINAL', 'POST_STOP_SELL')
#           AND seconds_vs_stop_sell <80
#         ORDER BY race_id, pool_type, combination, phase DESC;
#     """
    
#     print("Executing query (filtering for STOP_SELL and FINAL phases)...")
#     with engine.connect() as conn:
#         df = pd.read_sql(text(query), conn)
    
#     if df.empty:
#         print("WARNING: No data found. Make sure your archiver has successfully captured STOP_SELL and FINAL phases.")
#         return

#     # 2. Updated Pivot Table
#     print("Pivoting data for LLM ingestion...")
#     df_pivot = df.pivot_table(
#         index=['race_id', 'pool_type', 'combination'], 
#         columns='phase', 
#         values='odds', 
#         aggfunc='first'
#     ).reset_index()

#     # Save it using your config's artifact manager
#     output_path = artifact("claude_exotics_diagnostic.csv")
#     df_pivot.to_csv(output_path, index=False)
    
#     print(f"\n✅ SUCCESS: Exported {len(df_pivot)} exotic combinations.")
#     print(f"📄 Saved to: {output_path}")
#     print("Attach this CSV to Master Prompt #2 for Claude!")

# if __name__ == "__main__":
#     generate_claude_csv()


import redis
import json

def inspect_redis():
    print("🔍 Connecting to local Redis...")
    try:
        # Connect to default local Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
    except Exception as e:
        return

    venue = "ST"
    race = "1"

    current_race = r.get(f"current_scraping_race:{venue}")
    status = r.get(f"race_status:{venue}:{race}")
    
    print("\n--- STATE KEYS ---")
    print(f"Current Scraping Race : {current_race}")
    print(f"Race Status           : {status}")
    print(f"keys: {r.keys('*')}")
    win_key = f"live_odds_raw:{venue}:{race}:odds:WIN"
    win_data_str = r.get(win_key)
    
    print(f"\n--- WIN POOL PAYLOAD ({win_key}) ---")
    if not win_data_str:
        print(" NO DATA FOUND! The scraper is not writing to this key.")
        return
        
    print(" WIN data found! Parsing JSON...")
    win_data = json.loads(win_data_str)

    try:
        pools = win_data.get('pmPools', [])
        if not pools:
            print("⚠️ Key 'pmPools' missing from JSON payload.")
            print(json.dumps(win_data, indent=2))
            return
            
        odds_nodes = pools[0].get('oddsNodes', [])
        print(f"Found {len(odds_nodes)} runners in the JSON payload.")
        
        print("\nFirst 3 runners parsed:")
        for node in odds_nodes[:3]:
            horse_no = node.get('comb')
            odds = node.get('odds')
            print(f"Horse #{horse_no} -> Odds: {odds}")
            
    except Exception as e:
        print(json.dumps(win_data, indent=2)[:1000])

if __name__ == "__main__":
    inspect_redis()


