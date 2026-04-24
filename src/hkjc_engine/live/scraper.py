import asyncio
import json
import redis
from hkjc_engine.config import redis_client
import datetime
import sys
import random
import re
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

r_cache = redis_client()
today = datetime.datetime.now().strftime("%Y-%m-%d")
# today = '2026-04-06'

# How long (seconds) to keep polling the API after STOP_SELL is first seen. HKJC's tote continues to update dividends for 1-3 minutes after sell stops as late money settles into the final pools.
POST_STOP_SELL_POLL_SECONDS = 180


class HKJCLiveScraper:
    def __init__(self, venue, race_no):
        self.venue = venue
        self.race_no = race_no
        self.url = f"https://bet.hkjc.com/en/racing/wp/{today}/{self.venue}/{self.race_no}"
        self.current_pool_type = "WIN"

        # State for the two-phase shutdown:
        #   - stop_sell_seen_at: timestamp we first saw STOP_SELL from the API
        #   - post_stop_sell_mode: once True, the interceptor must NOT overwrite race_status back to STOP_SELL after we write CLOSED, and must never clobber CLOSED with any in-flight API payload.
        self.stop_sell_seen_at = None
        self.post_stop_sell_mode = False

    async def get_time_to_race(self):
        data = r_cache.get(f"live_race_metadata:{self.venue}:{self.race_no}")
        if data:
            meta = json.loads(data)
            if meta and 'time' in meta and ":" in meta['time']:
                try:
                    now = datetime.datetime.now()
                    time_str = meta['time']
                    start_time = datetime.datetime.strptime(time_str, "%H:%M").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    return (start_time - now).total_seconds()
                except ValueError:
                    pass
        return 9999

    async def intercept_odds_api(self, response):
        try:
            if "info.cld.hkjc.com/graphql" in response.url and response.status == 200:
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    data = await response.json()
                    meetings = data.get('data', {}).get('raceMeetings', [])
                    if not meetings: return

                    first_meeting = meetings[0]

                    if 'races' in first_meeting:
                        races = first_meeting.get('races', [])
                        if races and 'runners' in races[0]:
                            r_cache.setex(f"live_odds_raw:{self.venue}:{self.race_no}:runners", 9999, json.dumps(data))
                    if 'pmPools' in first_meeting:
                        pools = first_meeting.get('pmPools', [])
                        saved_pools = []
                        for p in pools:
                            o_type = p.get('oddsType')
                            races = p.get('leg', {}).get('races', [])
                            if not races: races = p.get('races', [])
                            status = str(p.get('status') or p.get('poolStatus') or p.get('sellStatus', '')).upper()

                            # --- STATE MACHINE: STOP_SELL detection + late-money polling ---
                            # On FIRST STOP_SELL sighting:
                            #   - write STOP_SELL to Redis so the live bot fires
                            #   - start the 180s post-sell polling window
                            # During post-sell polling:
                            #   - do NOT propagate status from the API to Redis (the main loop owns the STOP_SELL → CLOSED transition)
                            # Before STOP_SELL (normal sell window):
                            #   - propagate status as before
                            if status in ('STOP_SELL', 'STOPSELL'):
                                if not self.post_stop_sell_mode:
                                    self.stop_sell_seen_at = datetime.datetime.now()
                                    self.post_stop_sell_mode = True
                                    r_cache.setex(
                                        f"race_status:{self.venue}:{self.race_no}",
                                        600, 'STOP_SELL'
                                    )
                                    # Plan C: persist exact STOP_SELL wall-clock so the archiver can stamp every snapshot with its offset from this anchor. Enables exact drift diagnostics.
                                    r_cache.setex(
                                        f"stop_sell_time:{self.venue}:{self.race_no}",
                                        86400,
                                        self.stop_sell_seen_at.isoformat()
                                    )
                                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                                          f"API DETECTED 'STOP_SELL'. Live bot can fire. "
                                          f"Entering {POST_STOP_SELL_POLL_SECONDS}s late-money capture window.")
                            elif status and not self.post_stop_sell_mode:
                                # Normal pre-STOP_SELL status propagation
                                r_cache.setex(
                                    f"race_status:{self.venue}:{self.race_no}",
                                    300, status
                                )
                            # else: post_stop_sell_mode AND status != STOP_SELL  → ignore so CLOSED written by scrape_loop sticks
                            if o_type and int(self.race_no) in races:
                                isolated_payload = {
                                    "data": {
                                        "raceMeetings": [{
                                            "pmPools": [p]
                                        }]
                                    }
                                }
                                r_cache.setex(f"live_odds_raw:{self.venue}:{self.race_no}:odds:{o_type}", 10000, json.dumps(isolated_payload))
                                saved_pools.append(o_type)

                        if saved_pools:
                            time_str = datetime.datetime.now().strftime('%H:%M:%S')
                            tag = " [POST-SELL]" if self.post_stop_sell_mode else ""
                            print(f"[{time_str}] API Caught{tag}: {', '.join(set(saved_pools))} for R{self.race_no}")
        except Exception as e:
            pass

    async def extract_metadata(self, page):
        try:
            await page.wait_for_selector(".meeting-info-content-text", timeout=10000)
            info_text = await page.locator(".meeting-info-content-text").inner_text()
            dist_match = re.search(r'(\d+)\s*[mM]', info_text)
            distance = int(dist_match.group(1)) if dist_match else 1200

            class_match = re.search(r'CLASS\s+(\d)', info_text, re.I)
            race_class = f"Class {class_match.group(1)}" if class_match else "Unknown"

            rail = "A"
            if "ALL WEATHER" in info_text.upper() or "AWT" in info_text.upper():
                rail = "AWT"
            else:
                rail_match = re.search(r'"(.+)"\s+Course', info_text)
                if rail_match:
                    rail = rail_match.group(1)

            parts = [p.strip() for p in info_text.split(',')]
            race_time = "Unknown"
            if len(parts) > 2:
                race_time = parts[2]
            if ":" not in race_time:
                time_match = re.search(r'(\d{1,2}:\d{2})', info_text)
                if time_match:
                    race_time = time_match.group(1)

            metadata = {
                "venue": self.venue,
                "race_no": self.race_no,
                "class": race_class,
                "distance": distance,
                "rail": rail,
                "time": race_time,
                "scraped_at": datetime.datetime.now().strftime('%H:%M:%S')
            }

            r_cache.set(f"live_race_metadata:{self.venue}:{self.race_no}", json.dumps(metadata))
            print(f"[{metadata['scraped_at']}] Metadata Captured: {race_class} | {distance}M | Rail: {rail}")

            return metadata
        except Exception as e:
            print(f"Metadata Scrape Error: {e}")
            return None

    async def scrape_loop(self):
        r_cache.set(f"current_scraping_race:{self.venue}", self.race_no)
        async with Stealth().use_async(async_playwright()) as p:

            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )

            page = await context.new_page()
            page.on("response", self.intercept_odds_api)

            pool_urls = [
                (f"{self.url}", "WIN"),
                (f"https://bet.hkjc.com/en/racing/wpq/{today}/{self.venue}/{self.race_no}", "QIN"),
                (f"https://bet.hkjc.com/en/racing/tri/{today}/{self.venue}/{self.race_no}", "TRI")
            ]

            try:
                print(f"Establishing secure connection to {self.url}...")
                await page.goto(self.url, wait_until="domcontentloaded", timeout=60000)
                await self.extract_metadata(page)
                print("Connection established. Commencing passive API interception...")

                url_index = 0
                while True:
                    if self.post_stop_sell_mode and self.stop_sell_seen_at is not None:
                        elapsed = (datetime.datetime.now() - self.stop_sell_seen_at).total_seconds()
                        if elapsed >= POST_STOP_SELL_POLL_SECONDS:
                            r_cache.setex(
                                f"race_status:{self.venue}:{self.race_no}",
                                600, 'CLOSED'
                            )
                            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                                  f"Late-money window complete ({elapsed:.0f}s elapsed). "
                                  f"Wrote CLOSED → archiver can now snapshot. Shutting down browser.")
                            break

                    seconds_to_jump = await self.get_time_to_race()
                    if seconds_to_jump < -600 and not self.post_stop_sell_mode:
                        print(f"10 minutes past jump and no STOP_SELL detected. Forcing timeout.")
                        r_cache.setex(
                            f"race_status:{self.venue}:{self.race_no}",
                            600, 'CLOSED'
                        )
                        break

                    current_pool_url, pool_type = pool_urls[url_index % len(pool_urls)]
                    try:
                        await page.goto(current_pool_url, wait_until="domcontentloaded", timeout=30000)
                        if pool_type == "TRI":
                            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🖱️ Expanding Trio matrix...")
                            try:
                                await page.wait_for_selector("label[for='rdBankerALL']", state="visible", timeout=10000)
                                await asyncio.sleep(2)
                                await page.locator("label[for='rdBankerALL']").click(force=True, timeout=5000)
                                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🏆 Trio Matrix expanded. Interceptor taking over.")
                            except Exception as e:
                                print(f"Could not click Trio 'All' button: {e}")

                    except Exception as e:
                        print(f"Navigation timed out: {e}")

                    url_index += 1

                    # ---- Polling cadence ----
                    if self.post_stop_sell_mode:
                        delay = 5 + random.uniform(0, 1)
                    elif seconds_to_jump <= 120 and seconds_to_jump > -600:
                        delay = 5 + random.uniform(0, 2)
                    elif seconds_to_jump <= 300:
                        delay = 12 + random.uniform(1, 4)
                    else:
                        delay = 60 + random.uniform(2, 8)

                    await asyncio.sleep(delay)

            except Exception as e:
                print(f"Fatal Scrape Error: {e}")
            finally:
                await browser.close()


if __name__ == "__main__":
    venue = sys.argv[1].upper() if len(sys.argv) > 1 else 'ST'
    race_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"--- STARTING SCRAPER FOR {venue} RACE {race_no} ---")
    scraper = HKJCLiveScraper(venue=venue, race_no=race_no)
    asyncio.run(scraper.scrape_loop())
