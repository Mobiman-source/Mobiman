#!/usr/bin/env python3
"""
amf_suci_sync.py — tail /var/log/open5gs/amf.log and write SUCI last-two-digits
into the ue_ids.suci_suffix column of /tmp/ue_monitor.db.

Pairing logic (same as ue_map.py):
  AMF emits a SUCI line and a RAN_UE_NGAP_ID/AMF_UE_NGAP_ID line within a short
  time window for the same UE registration.  We pair them by timestamp proximity
  (AMF_PAIR_WINDOW_S).  Once paired we update ue_ids WHERE ran_ue_id = <ran_id>
  (and additionally backfill by amf_ue_id once that is available).

Run this script alongside the CU (it is a separate process, safe to kill/restart).
"""

import re
import os
import time
import sqlite3
import argparse
from datetime import datetime

AMF_LOG          = "/var/log/open5gs/amf.log"
DB_PATH          = "/tmp/ue_monitor.db"
AMF_PAIR_WINDOW  = 1.0   # seconds: max gap between SUCI line and ID line (log timestamps)
POLL_INTERVAL    = 0.05  # seconds between readline() retries
UPDATE_RETRY_WINDOW = 300.0  # seconds to keep unmatched DB updates alive
BACKFILL_RETRY_WINDOW = 300.0  # seconds to keep retrying a blocked backfill

re_suci    = re.compile(r"\[suci-[^\]]+\]")
re_ran_amf = re.compile(r"RAN_UE_NGAP_ID\[(\d+)\]\s+AMF_UE_NGAP_ID\[(\d+)\]")
# AMF log timestamp: "03/24 11:26:00.100:"
re_ts      = re.compile(r"^(\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d+):")

def parse_log_ts(line: str) -> float | None:
    """Parse the log timestamp from a line and return as float seconds since epoch (approx)."""
    m = re_ts.match(line)
    if not m:
        return None
    try:
        # Use current year since log doesn't include it
        year = datetime.now().year
        dt = datetime.strptime(f"{year}/{m.group(1)}", "%Y/%m/%d %H:%M:%S.%f")
        return dt.timestamp()
    except ValueError:
        return None


def suci_last2(suci_str: str) -> str:
    """Return last two digits of a SUCI string.
    'suci-0-001-01-0-0-0-0000000020' -> '20'
    """
    digits = [c for c in suci_str if c.isdigit()]
    if len(digits) >= 2:
        return ''.join(digits[-2:])
    return ''.join(digits)


def ensure_column(conn: sqlite3.Connection) -> None:
    """Add suci_suffix column if it does not exist yet (idempotent)."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(ue_ids)")]
    if "suci_suffix" not in cols:
        conn.execute("ALTER TABLE ue_ids ADD COLUMN suci_suffix TEXT")
        conn.commit()
        print("[amf_suci_sync] Added suci_suffix column to ue_ids")


def update_db(conn: sqlite3.Connection, ran_ue_id: int, amf_ue_id: int, suffix: str) -> dict[str, int]:
    """Update ue_ids/ue_records rows matching ran_ue_id first, then amf_ue_id.
    Returns per-table update counts."""
    ids_rows = 0
    rec_rows = 0

    cur = conn.execute(
        "UPDATE ue_ids SET suci_suffix = ?"
        " WHERE ran_ue_id = ? AND (suci_suffix IS NULL OR suci_suffix != ?)",
        (suffix, ran_ue_id, suffix)
    )
    ids_rows += max(cur.rowcount, 0)

    cur = conn.execute(
        "UPDATE ue_records SET suci_suffix = ?"
        " WHERE ran_ue_id = ? AND (suci_suffix IS NULL OR suci_suffix != ?)",
        (suffix, ran_ue_id, suffix)
    )
    rec_rows += max(cur.rowcount, 0)

    if ids_rows == 0:
        # Fall back to amf_ue_id match (row may have been inserted with different ran_ue_id
        # due to re-registration, or ran_ue_id not yet populated)
        cur = conn.execute(
            "UPDATE ue_ids SET suci_suffix = ?"
            " WHERE amf_ue_id = ? AND (suci_suffix IS NULL OR suci_suffix != ?)",
            (suffix, amf_ue_id, suffix)
        )
        ids_rows += max(cur.rowcount, 0)

    cur = conn.execute(
        "UPDATE ue_records SET suci_suffix = ?"
        " WHERE amf_ue_id = ? AND (suci_suffix IS NULL OR suci_suffix != ?)",
        (suffix, amf_ue_id, suffix)
    )
    rec_rows += max(cur.rowcount, 0)

    conn.commit()
    return {"ue_ids": ids_rows, "ue_records": rec_rows}


def open_db(db_path: str) -> sqlite3.Connection | None:
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        ensure_column(conn)
        return conn
    except sqlite3.Error as e:
        print(f"[amf_suci_sync] DB open error: {e}")
        return None


BACKFILL_INTERVAL = 1.0   # seconds between amf_ue_id → suci backfill passes


def backfill_suci(conn: sqlite3.Connection) -> bool:
    """Propagate suci_suffix to any ue_ids/ue_records rows that share an amf_ue_id
    with a row that already has suci_suffix filled in.
    Each amf_ue_id can only ever correspond to one SUCI (assigned at first registration),
    so subsequent rows (created during HO / re-establishment) can safely inherit it."""
    try:
        cur = conn.execute(
            "UPDATE ue_ids"
            " SET suci_suffix = ("
            "   SELECT h.suci_suffix FROM ue_ids h"
            "   WHERE h.amf_ue_id = ue_ids.amf_ue_id"
            "     AND h.suci_suffix IS NOT NULL"
            "   LIMIT 1"
            " )"
            " WHERE suci_suffix IS NULL AND amf_ue_id IS NOT NULL"
            "   AND EXISTS ("
            "     SELECT 1 FROM ue_ids h"
            "     WHERE h.amf_ue_id = ue_ids.amf_ue_id"
            "       AND h.suci_suffix IS NOT NULL"
            "   )"
        )
        ids_rows = cur.rowcount
        cur = conn.execute(
            "UPDATE ue_records"
            " SET suci_suffix = ("
            "   SELECT h.suci_suffix FROM ue_ids h"
            "   WHERE h.amf_ue_id = ue_records.amf_ue_id"
            "     AND h.suci_suffix IS NOT NULL"
            "   LIMIT 1"
            " )"
            " WHERE suci_suffix IS NULL AND amf_ue_id IS NOT NULL"
            "   AND EXISTS ("
            "     SELECT 1 FROM ue_ids h"
            "     WHERE h.amf_ue_id = ue_records.amf_ue_id"
            "       AND h.suci_suffix IS NOT NULL"
            "   )"
        )
        rec_rows = cur.rowcount
        conn.commit()
        if ids_rows or rec_rows:
            print(f"[amf_suci_sync] backfill: ue_ids={ids_rows} ue_records={rec_rows} rows updated")
        return True
    except sqlite3.Error as e:
        print(f"[amf_suci_sync] backfill error: {e}")
        return False


def tail_amf(amf_log: str, db_path: str, from_start: bool) -> None:
    conn: sqlite3.Connection | None = None

    pending_sucis: list[dict] = []   # [{"suffix": str, "ts": float}]
    pending_ids:   list[dict] = []   # [{"ran": int, "amf": int, "ts": float}]
    pending_updates: list[dict] = [] # [{"suffix": str, "ran": int, "amf": int, "first_ts": float, "last_try": float}]
    last_ts: list[float] = [time.time()]  # mutable cell for use in nested function
    last_backfill: list[float] = [0.0]
    last_log_activity: list[float] = [0.0]
    backfill_pending: list[bool] = [False]
    backfill_first_pending: list[float] = [0.0]

    def prune():
        cutoff = last_ts[0] - AMF_PAIR_WINDOW
        pending_sucis[:] = [x for x in pending_sucis if x["ts"] >= cutoff]
        pending_ids[:]   = [x for x in pending_ids   if x["ts"] >= cutoff]
        pending_updates[:] = [
            x for x in pending_updates
            if (time.time() - x["first_ts"]) <= UPDATE_RETRY_WINDOW
        ]

    def try_update_pending(force: bool = False):
        nonlocal conn
        if not pending_updates:
            return

        # Lazy-open DB (CU may not have started yet when this script launches)
        if conn is None:
            conn = open_db(db_path)
        if conn is None:
            return

        now = time.time()
        remaining: list[dict] = []
        for item in pending_updates:
            if not force and now - item["last_try"] < POLL_INTERVAL:
                remaining.append(item)
                continue

            item["last_try"] = now
            rows = update_db(conn, item["ran"], item["amf"], item["suffix"])
            total_rows = rows["ue_ids"] + rows["ue_records"]
            if total_rows:
                print(
                    "[amf_suci_sync]   -> applied pending mapping "
                    f"(ue_ids={rows['ue_ids']} ue_records={rows['ue_records']})"
                )
            else:
                remaining.append(item)
        pending_updates[:] = remaining

    def try_pair():
        while pending_sucis and pending_ids:
            best_gap: float | None = None
            best_pair: tuple[int, int] | None = None

            for si, s in enumerate(pending_sucis):
                for ri, r in enumerate(pending_ids):
                    gap = abs(s["ts"] - r["ts"])
                    if gap > AMF_PAIR_WINDOW:
                        continue
                    if (
                        best_gap is None or gap < best_gap or
                        (gap == best_gap and s["ts"] < pending_sucis[best_pair[0]]["ts"])
                    ):
                        best_gap = gap
                        best_pair = (si, ri)

            if best_pair is None:
                break

            si, ri = best_pair
            s = pending_sucis.pop(si)
            r = pending_ids.pop(ri)
            suffix = s["suffix"]
            ran    = r["ran"]
            amf    = r["amf"]

            print(
                f"[amf_suci_sync] PAIR  suci_suffix={suffix!r}  ran_ue_id={ran}  "
                f"amf_ue_id={amf}  gap={best_gap:.3f}s"
            )

            pending_updates.append({
                "suffix": suffix,
                "ran": ran,
                "amf": amf,
                "first_ts": time.time(),
                "last_try": 0.0,
            })

        try_update_pending()

    print(f"[amf_suci_sync] Opening {amf_log} ...")
    with open(amf_log, "r", errors="replace") as f:
        if not from_start:
            f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                prune()
                now = time.time()
                if conn is None:
                    conn = open_db(db_path)
                if conn is not None:
                    try_update_pending(force=True)
                should_retry_blocked_backfill = (
                    backfill_pending[0] and
                    (backfill_first_pending[0] == 0.0 or now - backfill_first_pending[0] <= BACKFILL_RETRY_WINDOW)
                )
                if (
                    conn is not None and
                    now - last_backfill[0] >= BACKFILL_INTERVAL and
                    (last_log_activity[0] > last_backfill[0] or should_retry_blocked_backfill)
                ):
                    ok = backfill_suci(conn)
                    last_backfill[0] = now
                    if ok:
                        backfill_pending[0] = False
                        backfill_first_pending[0] = 0.0
                    else:
                        if not backfill_pending[0]:
                            backfill_first_pending[0] = now
                        backfill_pending[0] = True
                continue

            if "[amf" not in line.lower():
                continue

            ts = parse_log_ts(line)
            if ts is None:
                ts = time.time()  # fallback for lines without parseable timestamp
            last_ts[0] = ts
            last_log_activity[0] = time.time()

            m_suci = re_suci.search(line)
            if m_suci and "warning" not in line.lower():
                raw  = m_suci.group(0)[1:-1]   # strip [ ]
                suf  = suci_last2(raw)
                pending_sucis.append({"suffix": suf, "ts": ts})

            m_ids = re_ran_amf.search(line)
            if m_ids and "warning" not in line.lower():
                pending_ids.append({
                    "ran": int(m_ids.group(1)),
                    "amf": int(m_ids.group(2)),
                    "ts": ts,
                })

            prune()
            try_pair()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync SUCI suffix from AMF log into ue_monitor DB")
    parser.add_argument("--amf-log", default=AMF_LOG)
    parser.add_argument("--db",      default=DB_PATH)
    parser.add_argument("--from-start", action="store_true",
                        help="Parse the log from the beginning (backfill historical entries)")
    args = parser.parse_args()

    tail_amf(args.amf_log, args.db, args.from_start)


if __name__ == "__main__":
    main()
