import subprocess
import time
import sys

RACES_TO_MONITOR = [1, 2, 3, 4, 5, 6, 7, 8, 9]
VENUE = "HV"


###### PARALLEL ######
## Deprecated - Running all races would result in (# of races) * (3 webpages {WIN/QIN/TRI}) * 5 concurrent requests ~= 12 requests/sec which could ramp up to 25 requests/sec, instantly getting flagged by HKJC servers.

# processes = []
# print(f" Launching Scraper for {len(RACES_TO_MONITOR)} races...")
# for r_no in RACES_TO_MONITOR:
#     # This launches a new background process for each race
#     # It's like opening 8 browser tabs, but in the terminal
#     p = subprocess.Popen([sys.executable, "live_scraper.py", VENUE, str(r_no)])
#     processes.append(p)
#     print(f"✅ Started Monitor for Race {r_no}")
#     time.sleep(2) # Space them out slightly to avoid WAF triggering

# print("\n--- ALL MONITORS ACTIVE ---")
# print("Redis is now being populated with live data for the entire card.")
# print("Press Ctrl+C to stop all monitors.")

# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("\nShutting down all monitors...")
#     for p in processes:
#         p.terminate()
#     print("Cleaned up. Good luck with the punting!")


###### SEQUENTIAL ######
print(f"Launching Sequential Syndicate Scraper for {len(RACES_TO_MONITOR)} races...")

try:
    for r_no in RACES_TO_MONITOR:
        print(f"\n{'='*50}")
        print(f" Starting Monitor for {VENUE} Race {r_no}")
        print(f"{'='*50}")

        p = subprocess.Popen([sys.executable, "live_scraper.py", VENUE, str(r_no)])
        p.wait() 
        
        print(f" Race {r_no} scraper exited. Waiting 5 seconds before launching next race...")
        time.sleep(5) 

    print("\n--- ALL RACES COMPLETED FOR THE DAY ---")
    print("Cleaned up. Good luck with the punting!")

except KeyboardInterrupt:
    print("\nShutting down orchestrator...")
    if 'p' in locals() and p.poll() is None:
        p.terminate()
    print("Exited.")