"""
account_statement_ingester.py
=============================

Parses HKJC's "Account Records" text export and writes structured rows to
a `placed_bets` table. Designed to be idempotent: re-running on the same
file (or overlapping date ranges) does NOT create duplicates -- the
ref_no field is a stable HKJC primary key.

Handles:
    - WIN, PLACE single bets
    - QUINELLA, QUINELLA-PLACE pairs (simple)
    - QUINELLA-PLACE bankers ("X Banker with A + B + C" -> 3 pair rows)
    - TRIO triples (simple)
    - TRIO bankers ("X + Y Banker with A + B" -> 2 triple rows)
    - DEPOSIT / WITHDRAWAL lines (recorded in account_transactions for
      bankroll tracking)

Writes two tables:
    placed_bets             -- one row per actual combination bet
    account_transactions    -- deposits, withdrawals, balance markers

Schema is defined in this module and idempotent.

Usage
-----
    python -m hkjc_engine.live.account_statement_ingester \\
        --statement_file ./statements/acctstmt_2026-04-26.txt
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from sqlalchemy import create_engine, text


log = logging.getLogger(__name__)


PLACED_BETS_DDL = """
CREATE TABLE IF NOT EXISTS placed_bets (
    ref_no             TEXT        NOT NULL,
    leg_index          INTEGER     NOT NULL DEFAULT 0,
    placed_at          TIMESTAMPTZ NOT NULL,
    venue              TEXT,
    race_no            INTEGER,
    race_id            TEXT,
    pool               TEXT        NOT NULL,
    combination        TEXT        NOT NULL,
    horse_names        TEXT,
    stake              REAL        NOT NULL,
    is_banker          BOOLEAN     NOT NULL DEFAULT FALSE,
    banker_horses      TEXT,
    realised_dividend  REAL,
    realised_pnl       REAL,
    PRIMARY KEY (ref_no, leg_index)
);
CREATE INDEX IF NOT EXISTS ix_placed_race    ON placed_bets (race_id);
CREATE INDEX IF NOT EXISTS ix_placed_at      ON placed_bets (placed_at);
CREATE INDEX IF NOT EXISTS ix_placed_pool_combo
    ON placed_bets (race_id, pool, combination);
"""

ACCOUNT_TX_DDL = """
CREATE TABLE IF NOT EXISTS account_transactions (
    ref_no       TEXT        PRIMARY KEY,
    tx_time      TIMESTAMPTZ NOT NULL,
    tx_type      TEXT        NOT NULL,
    amount       REAL        NOT NULL,
    balance_at   REAL,
    raw_text     TEXT
);
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedBlock:
    ref_no:    str
    timestamp: datetime
    raw_lines: list[str]


# Each block in the statement is delimited by lines of asterisks. The first
# non-asterisk line is the ref_no; the second is "DD/MM/YYYY HH:MM" plus venue.
# Some blocks are deposits/withdrawals (no race info).
BLOCK_DELIM = re.compile(r"^\*{20,}\s*$")
DATE_TIME_RE = re.compile(
    r"^(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})\s*(.*)?$"
)
RACE_RE = re.compile(r"^Race\s+(\d+)\s*$", re.IGNORECASE)
HORSE_LINE_RE = re.compile(r"^\s*(\d+)\s+([A-Z][A-Z0-9 '\-]+?)\s*\+?\s*$")
BANKER_RE = re.compile(r"Banker\s+with", re.IGNORECASE)


def _split_into_blocks(text: str) -> Iterator[list[str]]:
    """Yield each block as a list of lines (excluding delimiter lines)."""
    current: list[str] = []
    for line in text.splitlines():
        if BLOCK_DELIM.match(line):
            if current:
                yield current
                current = []
        else:
            current.append(line.rstrip())
    if current:
        yield current


def _parse_block_header(lines: list[str]):
    """
    First content line is ref_no (digits only). Second is DD/MM/YYYY HH:MM.
    Third is venue (e.g. 'Sha Tin', 'Happy Valley'). Fourth is day-of-week.
    Returns (ref_no, timestamp, venue_or_none, body_starting_index) or None
    if this isn't a transaction block.
    """
    # Skip leading empty lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None

    ref_match = re.match(r"^\s*(\d+)\s*$", lines[i])
    if not ref_match:
        return None
    ref_no = ref_match.group(1)
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None

    dt = DATE_TIME_RE.match(lines[i])
    if not dt:
        return None
    dd, mm, yyyy, hh, mn, tail = dt.groups()
    ts = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mn))
    i += 1

    # Look ahead for venue line. Skip blank lines.
    venue = (tail or "").strip() or None
    if not venue:
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            candidate = lines[j].strip()
            # Venue line is short text like 'Sha Tin' or 'Happy Valley'
            # (not a number, not 'Race N', not a $ amount)
            if (candidate
                    and not candidate.startswith("$")
                    and not candidate[0].isdigit()
                    and not RACE_RE.match(candidate)
                    and "DEPOSIT" not in candidate.upper()
                    and "WITHDRAWAL" not in candidate.upper()
                    and len(candidate) < 50):
                venue = candidate

    return (ref_no, ts, venue, i)


def _normalize_combo(numbers: list[int]) -> str:
    return "-".join(str(n) for n in sorted(set(numbers)))


def _extract_horse_lines(body: list[str]) -> list[tuple[int, str]]:
    """Pull (number, name) tuples from horse lines like '11 MASSIVE REWARD'.
    Tolerates trailing '+' continuation markers."""
    horses = []
    for line in body:
        m = HORSE_LINE_RE.match(line)
        if m:
            horses.append((int(m.group(1)), m.group(2).strip()))
    return horses


def _classify_pool(body: list[str]) -> str | None:
    """Find the pool type. The statement uses these strings:
        'Win', 'Place', 'Quinella', 'Quinella - Place', 'Trio'
    Returns canonical short form: WIN | PLA | QIN | QPL | TRI."""
    text = " ".join(line.strip() for line in body[:6]).upper()
    if "QUINELLA - PLACE" in text or "QUINELLA-PLACE" in text:
        return "QPL"
    if "QUINELLA" in text:
        return "QIN"
    if "TRIO" in text:
        return "TRI"
    if "PLACE" in text and "QUINELLA" not in text:
        return "PLA"
    if re.search(r"\bWIN\b", text):
        return "WIN"
    return None


def _has_banker(body: list[str]) -> bool:
    return any(BANKER_RE.search(line) for line in body)


def _split_banker_legs(horses: list[tuple[int, str]],
                       body: list[str]) -> tuple[list[int], list[int]]:
    """
    For 'X (+ Y) Banker with A + B + C', returns (banker_numbers, leg_numbers).
    The Banker line appears AFTER the banker horses and BEFORE the legs.
    """
    banker_idx = next((i for i, line in enumerate(body)
                       if BANKER_RE.search(line)), None)
    if banker_idx is None:
        return ([n for n, _ in horses], [])

    bankers, legs = [], []
    for i, line in enumerate(body):
        m = HORSE_LINE_RE.match(line)
        if not m:
            continue
        num = int(m.group(1))
        if i < banker_idx:
            bankers.append(num)
        else:
            legs.append(num)
    return (bankers, legs)


def _race_id_from_venue(venue_text: str | None, ts: datetime,
                        race_no: int) -> str | None:
    """Build race_id of form YYYYMMDD_VENUE_RR. Venue parsing is best-effort
    -- HKJC writes 'Sha Tin' / 'Happy Valley' on the second line of each
    transaction block, with day-of-week tag below."""
    if venue_text is None or race_no is None:
        return None
    v = venue_text.upper()
    if "SHA TIN" in v or v.strip().startswith("ST"):
        venue_code = "ST"
    elif "HAPPY VALLEY" in v or v.strip().startswith("HV"):
        venue_code = "HV"
    else:
        return None
    return f"{ts.strftime('%Y%m%d')}_{venue_code}_{race_no:02d}"


def _parse_race_no(body: list[str]) -> int | None:
    for line in body:
        m = RACE_RE.match(line.strip())
        if m:
            return int(m.group(1))
    return None


def _parse_stake(body: list[str]) -> float | None:
    """Stake is the FIRST $-amount line in the body. (For settled bets a
    second $-amount line follows it -- that's the credit, parsed separately.)
    """
    for line in body:
        m = re.match(r"^\s*\$\s*([\d,]+(?:\.\d+)?)\s*$", line)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _parse_credit(body: list[str]) -> float | None:
    """Parse the WINNING PAYOUT, if present.

    HKJC's account-statement block format puts dollar amounts in this order:
        line 1 ($N):    per-leg stake (debit column)
        line 2 ($N.NN): total stake billed (= per-leg × num_legs)
        line 3 ($N.NN): WINNING PAYOUT  -- only present if the bet won

    A bet that lost has only TWO dollar lines. A bet that won has THREE.
    Returns the third $-amount if it exists, else None (not a winner).
    """
    dollar_amounts = []
    for line in body:
        m = re.match(r"^\s*\$\s*([\d,]+(?:\.\d+)?)\s*$", line)
        if m:
            dollar_amounts.append(float(m.group(1).replace(",", "")))
    if len(dollar_amounts) >= 3:
        return dollar_amounts[2]
    return None


# ---------------------------------------------------------------------------
# Block dispatch
# ---------------------------------------------------------------------------

def _is_deposit_or_withdrawal(body: list[str]) -> str | None:
    """Returns 'DEPOSIT' or 'WITHDRAWAL' if this block is a money-movement
    record, else None."""
    text = " ".join(body).upper()
    if "DEPOSIT" in text:
        return "DEPOSIT"
    if "WITHDRAWAL" in text:
        return "WITHDRAWAL"
    return None


def parse_statement(raw_text: str):
    """
    Iterates parsed records from the statement. Yields dicts with kind in
    {'bet', 'transaction'}. Bet records may expand into multiple combination
    legs (banker bets) before insertion -- that expansion happens here.
    """
    for block_lines in _split_into_blocks(raw_text):
        header = _parse_block_header(block_lines)
        if header is None:
            continue
        ref_no, ts, venue, body_start = header
        body = block_lines[body_start:]

        # Money movement block?
        tx_kind = _is_deposit_or_withdrawal(body)
        if tx_kind is not None:
            amount = _parse_stake(body) or _parse_credit(body)
            yield {
                "kind": "transaction",
                "ref_no": ref_no,
                "tx_time": ts,
                "tx_type": tx_kind,
                "amount": amount or 0.0,
                "raw_text": "\n".join(body),
            }
            continue

        # Bet block
        pool = _classify_pool(body)
        if pool is None:
            log.debug("Skipping unparseable block (no pool): ref %s", ref_no)
            continue
        race_no = _parse_race_no(body)
        race_id = _race_id_from_venue(venue, ts, race_no) if race_no else None
        stake = _parse_stake(body) or 0.0
        credit = _parse_credit(body)

        horses = _extract_horse_lines(body)
        if not horses:
            log.warning("Block ref %s has no parseable horse lines", ref_no)
            continue
        horse_name_lookup = {n: name for n, name in horses}
        is_banker = _has_banker(body)

        # Build the combination list this single ticket expands into
        combos: list[list[int]] = []
        banker_repr: str | None = None

        if is_banker:
            bankers, legs = _split_banker_legs(horses, body)
            banker_repr = "-".join(str(b) for b in sorted(bankers))
            if pool == "QPL" or pool == "QIN":
                # QPL/QIN banker: each leg paired with the banker(s)
                if len(bankers) == 1:
                    for leg in legs:
                        combos.append([bankers[0], leg])
                else:
                    for leg in legs:
                        combos.append(sorted(bankers + [leg]))
            elif pool == "TRI":
                # Trio banker: e.g. "1 + 11 Banker with 3 + 9" -> {1,3,11},{1,9,11}
                if len(bankers) == 2:
                    for leg in legs:
                        combos.append(sorted(bankers + [leg]))
                elif len(bankers) == 1:
                    from itertools import combinations as _comb
                    for pair in _comb(legs, 2):
                        combos.append(sorted([bankers[0]] + list(pair)))
                else:
                    log.warning("Unhandled trio banker shape ref %s", ref_no)
        else:
            # No "Banker with" keyword. Two shapes are possible:
            #   - Simple ticket: n horses == pool combo length (e.g. QPL with 2 horses)
            #   - Box/perm ticket: n horses > pool combo length, expanded to
            #     ALL combinations of size = pool combo length. HKJC bills
            #     stake * num_combinations for these. No "Banker" keyword
            #     because there's no anchor horse.
            from itertools import combinations as _comb
            nums = sorted(n for n, _ in horses)
            pool_combo_len = {"QIN": 2, "QPL": 2, "TRI": 3,
                              "WIN": 1, "PLA": 1}.get(pool, len(nums))
            if len(nums) == pool_combo_len:
                # Straight ticket
                combos.append(nums)
            elif len(nums) > pool_combo_len and pool in ("QIN", "QPL", "TRI"):
                # Box/perm ticket: expand to all combinations
                for c in _comb(nums, pool_combo_len):
                    combos.append(list(c))
                log.debug("Expanded %s box ticket ref %s with %d horses "
                          "into %d combinations",
                          pool, ref_no, len(nums), len(combos))
            else:
                # Unknown shape (e.g., n < combo_len, or WIN/PLA with multiple
                # horses on one ticket — shouldn't happen but be defensive)
                log.warning("Ref %s pool=%s has %d horses; expected %d. "
                            "Recording as-is.", ref_no, pool,
                            len(nums), pool_combo_len)
                combos.append(nums)

        # Stake handling: HKJC's account statement quotes the per-leg stake
        # (not the total). For a ticket that expands to N combinations, each
        # leg gets the FULL `stake` value -- the customer was billed
        # stake * N total. Reconciliation only credits the leg whose combo
        # matches the winning dividend row.
        per_leg_stake = stake

        for leg_idx, combo_nums in enumerate(combos):
            yield {
                "kind": "bet",
                "ref_no": ref_no,
                "leg_index": leg_idx,
                "placed_at": ts,
                "venue": venue,
                "race_no": race_no,
                "race_id": race_id,
                "pool": pool,
                "combination": _normalize_combo(combo_nums),
                "horse_names": ", ".join(
                    f"{n} {horse_name_lookup.get(n, '?')}" for n in combo_nums),
                "stake": per_leg_stake,
                "is_banker": is_banker,
                "banker_horses": banker_repr,
                "credit_total": credit if leg_idx == 0 else None,
            }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def ensure_schema(engine):
    """Create tables and indexes. Each statement runs in its own transaction
    so a failure in one (e.g. an index already exists with wrong definition)
    doesn't roll back the others."""
    statements = []
    for ddl_block in (PLACED_BETS_DDL, ACCOUNT_TX_DDL):
        for stmt in ddl_block.split(";"):
            if stmt.strip():
                statements.append(stmt.strip())

    log.info("Ensuring schema (%d DDL statements)...", len(statements))
    for i, stmt in enumerate(statements, start=1):
        first_line = stmt.split("\n", 1)[0][:80]
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            log.debug("  [%d/%d] OK: %s", i, len(statements), first_line)
        except Exception as e:
            log.error("  [%d/%d] FAILED: %s -- %s", i, len(statements),
                      first_line, e)
            raise
    log.info("Schema ready.")


INSERT_BET = text("""
    INSERT INTO placed_bets (
        ref_no, leg_index, placed_at, venue, race_no, race_id,
        pool, combination, horse_names, stake, is_banker, banker_horses
    ) VALUES (
        :ref_no, :leg_index, :placed_at, :venue, :race_no, :race_id,
        :pool, :combination, :horse_names, :stake, :is_banker, :banker_horses
    )
    ON CONFLICT (ref_no, leg_index) DO UPDATE SET
        stake = EXCLUDED.stake,
        race_id = EXCLUDED.race_id,
        combination = EXCLUDED.combination
""")

INSERT_TX = text("""
    INSERT INTO account_transactions (ref_no, tx_time, tx_type, amount, raw_text)
    VALUES (:ref_no, :tx_time, :tx_type, :amount, :raw_text)
    ON CONFLICT (ref_no) DO NOTHING
""")


def ingest_file(statement_path: str, db_url: str) -> dict:
    log.info("Connecting to database...")
    engine = create_engine(db_url, pool_pre_ping=True)
    ensure_schema(engine)

    log.info("Reading statement file: %s", statement_path)
    with open(statement_path, encoding="utf-8") as f:
        raw = f.read()
    log.info("Read %d bytes (%d lines)", len(raw), raw.count("\n"))

    n_blocks = sum(1 for _ in _split_into_blocks(raw))
    log.info("Found %d blocks to parse", n_blocks)

    bet_rows = []; tx_rows = []; n_skipped = 0
    for rec in parse_statement(raw):
        if rec["kind"] == "bet":
            bet_rows.append({k: rec[k] for k in (
                "ref_no", "leg_index", "placed_at", "venue", "race_no",
                "race_id", "pool", "combination", "horse_names", "stake",
                "is_banker", "banker_horses")})
        elif rec["kind"] == "transaction":
            tx_rows.append({k: rec[k] for k in (
                "ref_no", "tx_time", "tx_type", "amount", "raw_text")})
        else:
            n_skipped += 1

    log.info("Parsed %d bet legs, %d transactions, %d skipped/unparseable",
             len(bet_rows), len(tx_rows), n_skipped)

    if not bet_rows and not tx_rows:
        log.warning("No records parsed from %s. Run with --debug to see "
                    "block-by-block parsing detail.", statement_path)
        return {"bets": 0, "transactions": 0}

    log.info("Inserting %d bet rows + %d transaction rows...",
             len(bet_rows), len(tx_rows))
    with engine.begin() as conn:
        if bet_rows:
            conn.execute(INSERT_BET, bet_rows)
        if tx_rows:
            conn.execute(INSERT_TX, tx_rows)
    log.info("Insert complete.")

    # Quick post-insert verification
    with engine.connect() as conn:
        n_bets_in_db = conn.execute(text(
            "SELECT COUNT(*) FROM placed_bets WHERE placed_at::date = "
            "(SELECT MIN(placed_at::date) FROM placed_bets WHERE ref_no = ANY(:refs))"
        ), {"refs": [r["ref_no"] for r in bet_rows[:5]] if bet_rows else [""]}).scalar()
        log.info("Verified: %d rows in placed_bets for that date", n_bets_in_db or 0)

    return {"bets": len(bet_rows), "transactions": len(tx_rows)}


def reconcile_with_dividends(db_url: str, dividend_unit_base: float = 10.0) -> int:
    """
    Updates placed_bets.realised_dividend and realised_pnl by joining to
    race_dividends. Returns count of rows updated.
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = text("""
        WITH normalized_divs AS (
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
        UPDATE placed_bets b SET
            realised_dividend = d.realised_dividend,
            realised_pnl = b.stake * (d.realised_dividend - 1.0)
        FROM normalized_divs d
        WHERE b.race_id     = d.race_id
          AND b.pool        = d.pool_norm
          AND b.combination = d.combo_norm
    """)
    settled_losers_sql = text("""
        UPDATE placed_bets b SET
            realised_dividend = NULL,
            realised_pnl = -b.stake
        WHERE b.realised_dividend IS NULL
          AND b.realised_pnl IS NULL
          AND EXISTS (SELECT 1 FROM race_dividends d
                      WHERE d.race_id = b.race_id)
    """)
    with engine.begin() as conn:
        winners = conn.execute(sql, {"unit_base": dividend_unit_base})
        losers = conn.execute(settled_losers_sql)
    log.info("Reconciled: %d winners updated, %d losers marked",
             winners.rowcount, losers.rowcount)
    return winners.rowcount + losers.rowcount


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from hkjc_engine.config import DB_URL

    ap = argparse.ArgumentParser()
    ap.add_argument("--statement_file", required=True,
                    help="Path to HKJC account-records text export")
    ap.add_argument("--reconcile", action="store_true",
                    help="After ingest, join against race_dividends")
    ap.add_argument("--dividend_unit_base", type=float, default=10.0)
    ap.add_argument("--debug", action="store_true",
                    help="Verbose parsing output (shows every block)")
    ap.add_argument("--verify-only", action="store_true",
                    help="Parse the file but don't write to DB. For "
                         "diagnosing why nothing parses.")
    args = ap.parse_args()

    # Set up logging BEFORE any other code runs so all messages are captured
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        force=True,  # override any prior config
    )

    if not os.path.exists(args.statement_file):
        log.error("File not found: %s", args.statement_file)
        raise SystemExit(1)

    if args.verify_only:
        log.info("VERIFY-ONLY MODE: parsing %s without writing to DB",
                 args.statement_file)
        with open(args.statement_file, encoding="utf-8") as f:
            raw = f.read()
        log.info("Read %d bytes", len(raw))
        n_blocks = sum(1 for _ in _split_into_blocks(raw))
        log.info("Found %d blocks", n_blocks)
        bets = []; txs = []
        for rec in parse_statement(raw):
            if rec["kind"] == "bet":
                bets.append(rec)
            else:
                txs.append(rec)
        log.info("Parsed: %d bet legs, %d transactions", len(bets), len(txs))
        if bets:
            log.info("First bet leg: %s", bets[0])
        if txs:
            log.info("First transaction: %s", txs[0])
        raise SystemExit(0)

    summary = ingest_file(args.statement_file, DB_URL)
    log.info("DONE: %d bet legs, %d transactions ingested.",
             summary["bets"], summary["transactions"])

    if args.reconcile:
        n = reconcile_with_dividends(DB_URL, args.dividend_unit_base)
        log.info("Reconciliation updated %d bet rows.", n)