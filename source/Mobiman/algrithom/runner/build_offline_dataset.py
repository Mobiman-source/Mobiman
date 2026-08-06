"""
Build offline (s, a, r, s') transitions from ran.db.

Action inference rule:
- For each UE trajectory (grouped by SUCI), sort by (ts, id).
- If next_du_id != curr_du_id, action = handover to next_du_id.
- Else action = stay on current DU.

Reward rule:
- r = thp_dl(suci, t_next) - thp_dl(suci, t) - handover_penalty * is_handover

Output:
- Flat CSV with s_*, a_*, r_*, sp_* columns.
"""

import argparse
import bisect
import csv
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def _to_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _to_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class TimeSeriesIndex:
    """Lookup latest value at or before a timestamp."""

    def __init__(self):
        self.ts: List[float] = []
        self.values: List[float] = []

    def append(self, ts: float, value: float):
        self.ts.append(float(ts))
        self.values.append(float(value))

    def lookup(self, ts: float, default: float = 0.0) -> float:
        if not self.ts:
            return default
        pos = bisect.bisect_right(self.ts, float(ts)) - 1
        if pos < 0:
            return default
        return self.values[pos]


class CellSeriesIndex:
    """Lookup latest DU-level cell metrics at or before a timestamp."""

    def __init__(self):
        self.ts: List[float] = []
        self.values: List[Tuple[float, float, float]] = []

    def append(self, ts: float, thp_dl: float, prb_dl: float, prb_ul: float):
        self.ts.append(float(ts))
        self.values.append((float(thp_dl), float(prb_dl), float(prb_ul)))

    def lookup(self, ts: float) -> Tuple[float, float, float]:
        if not self.ts:
            return 0.0, 0.0, 0.0
        pos = bisect.bisect_right(self.ts, float(ts)) - 1
        if pos < 0:
            return 0.0, 0.0, 0.0
        return self.values[pos]


def load_ue_throughput(conn: sqlite3.Connection) -> Dict[str, TimeSeriesIndex]:
    idx: Dict[str, TimeSeriesIndex] = {}
    cur = conn.cursor()
    cur.execute(
        """
        SELECT suci, ts, thp_dl
        FROM du_ue_kpm
        WHERE suci IS NOT NULL AND ts IS NOT NULL AND thp_dl IS NOT NULL
        ORDER BY suci ASC, ts ASC, id ASC
        """
    )
    for suci, ts, thp_dl in cur.fetchall():
        s = str(suci)
        if s not in idx:
            idx[s] = TimeSeriesIndex()
        idx[s].append(_to_float(ts), _to_float(thp_dl))
    return idx


def load_du_cell_metrics(conn: sqlite3.Connection) -> Dict[int, CellSeriesIndex]:
    idx: Dict[int, CellSeriesIndex] = {}
    cur = conn.cursor()
    cur.execute(
        """
        SELECT du_id, ts, thp_dl, prb_use_perc_dl, prb_use_perc_ul
        FROM du_cell_kpm
        WHERE du_id IS NOT NULL AND ts IS NOT NULL
        ORDER BY CAST(du_id AS INTEGER) ASC, ts ASC, id ASC
        """
    )
    for du_id_raw, ts, thp_dl, prb_dl, prb_ul in cur.fetchall():
        du_id = _to_int(du_id_raw)
        if du_id is None:
            continue
        if du_id not in idx:
            idx[du_id] = CellSeriesIndex()
        idx[du_id].append(
            _to_float(ts),
            _to_float(thp_dl),
            _to_float(prb_dl),
            _to_float(prb_ul),
        )
    return idx


def iter_cu_rows_by_suci(
    conn: sqlite3.Connection,
    suci_allowlist: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple]]:
    allow = set(suci_allowlist) if suci_allowlist else None
    rows_by_suci: Dict[str, List[Tuple]] = defaultdict(list)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id, ts, suci, du_id,
            serv_rsrp, serv_rsrq, serv_sinr,
            neigh1_pci, neigh1_rsrp, neigh1_rsrq,
            neigh2_pci, neigh2_rsrp, neigh2_rsrq
        FROM cu_ue_meas
        WHERE suci IS NOT NULL AND du_id IS NOT NULL AND ts IS NOT NULL
        ORDER BY suci ASC, ts ASC, id ASC
        """
    )
    for row in cur.fetchall():
        suci = str(row[2])
        if allow is not None and suci not in allow:
            continue
        rows_by_suci[suci].append(row)
    return rows_by_suci


def _state_features_from_cu_row(
    row: Tuple,
    cell_idx: Dict[int, CellSeriesIndex],
) -> Dict[str, float]:
    (
        _id,
        ts,
        _suci,
        du_id_raw,
        serv_rsrp,
        serv_rsrq,
        serv_sinr,
        neigh1_pci,
        neigh1_rsrp,
        neigh1_rsrq,
        neigh2_pci,
        neigh2_rsrp,
        neigh2_rsrq,
    ) = row
    du_id = _to_int(du_id_raw)
    if du_id is None:
        du_id = -1

    c_thp, c_prb_dl, c_prb_ul = (0.0, 0.0, 0.0)
    if du_id in cell_idx:
        c_thp, c_prb_dl, c_prb_ul = cell_idx[du_id].lookup(_to_float(ts))

    return {
        "ts": _to_float(ts),
        "du_id": float(du_id),
        "serv_rsrp": _to_float(serv_rsrp),
        "serv_rsrq": _to_float(serv_rsrq),
        "serv_sinr": _to_float(serv_sinr),
        "neigh1_pci": _to_float(neigh1_pci, default=-1.0),
        "neigh1_rsrp": _to_float(neigh1_rsrp),
        "neigh1_rsrq": _to_float(neigh1_rsrq),
        "neigh2_pci": _to_float(neigh2_pci, default=-1.0),
        "neigh2_rsrp": _to_float(neigh2_rsrp),
        "neigh2_rsrq": _to_float(neigh2_rsrq),
        "cell_thp_dl": c_thp,
        "cell_prb_use_perc_dl": c_prb_dl,
        "cell_prb_use_perc_ul": c_prb_ul,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline transitions from ran.db")
    parser.add_argument(
        "--db",
        type=str,
        default="/path/to/your/oran-sc-ric/data/ran.db",
        help="Path to SQLite DB (ran.db)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/path/to/your/oran-sc-ric/data/offline_transitions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--handover-penalty",
        type=float,
        default=0.0,
        help="Penalty term subtracted from reward when handover=1",
    )
    parser.add_argument(
        "--suci-list",
        type=str,
        default="",
        help="Optional comma-separated SUCI allowlist",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=0,
        help="Optional cap on generated transitions (0 = no cap)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    allowlist = [s.strip() for s in args.suci_list.split(",") if s.strip()] or None

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA busy_timeout=5000")

    print("[offline-ds] Loading du_ue_kpm throughput index ...")
    ue_thp_idx = load_ue_throughput(conn)
    print(f"[offline-ds] Throughput series loaded: {len(ue_thp_idx)} SUCIs")

    print("[offline-ds] Loading du_cell_kpm index ...")
    cell_idx = load_du_cell_metrics(conn)
    print(f"[offline-ds] DU cell series loaded: {len(cell_idx)} DUs")

    print("[offline-ds] Loading cu_ue_meas trajectories ...")
    rows_by_suci = iter_cu_rows_by_suci(conn, suci_allowlist=allowlist)
    print(f"[offline-ds] CU trajectories loaded: {len(rows_by_suci)} SUCIs")

    conn.close()

    columns = [
        "suci",
        "t",
        "t_next",
        "dt",
        "s_du_id",
        "s_serv_rsrp",
        "s_serv_rsrq",
        "s_serv_sinr",
        "s_neigh1_pci",
        "s_neigh1_rsrp",
        "s_neigh1_rsrq",
        "s_neigh2_pci",
        "s_neigh2_rsrp",
        "s_neigh2_rsrq",
        "s_cell_thp_dl",
        "s_cell_prb_use_perc_dl",
        "s_cell_prb_use_perc_ul",
        "a_handover",
        "a_target_du",
        "a_source_du",
        "r",
        "r_thp_delta",
        "done",
        "sp_du_id",
        "sp_serv_rsrp",
        "sp_serv_rsrq",
        "sp_serv_sinr",
        "sp_neigh1_pci",
        "sp_neigh1_rsrp",
        "sp_neigh1_rsrq",
        "sp_neigh2_pci",
        "sp_neigh2_rsrp",
        "sp_neigh2_rsrq",
        "sp_cell_thp_dl",
        "sp_cell_prb_use_perc_dl",
        "sp_cell_prb_use_perc_ul",
    ]

    n_trans = 0
    n_ho = 0
    n_suci_used = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for suci, traj in rows_by_suci.items():
            if len(traj) < 2:
                continue
            n_suci_used += 1

            thp_series = ue_thp_idx.get(suci, TimeSeriesIndex())

            for i in range(len(traj) - 1):
                curr = traj[i]
                nxt = traj[i + 1]

                s = _state_features_from_cu_row(curr, cell_idx)
                sp = _state_features_from_cu_row(nxt, cell_idx)

                src_du = int(s["du_id"])
                tgt_du = int(sp["du_id"])
                is_ho = 1 if src_du != tgt_du else 0
                n_ho += is_ho

                thp_t = thp_series.lookup(s["ts"], default=0.0)
                thp_tp1 = thp_series.lookup(sp["ts"], default=thp_t)
                r_delta = thp_tp1 - thp_t
                reward = r_delta - float(args.handover_penalty) * is_ho

                done = 1 if i == len(traj) - 2 else 0

                row = {
                    "suci": suci,
                    "t": s["ts"],
                    "t_next": sp["ts"],
                    "dt": sp["ts"] - s["ts"],
                    "s_du_id": src_du,
                    "s_serv_rsrp": s["serv_rsrp"],
                    "s_serv_rsrq": s["serv_rsrq"],
                    "s_serv_sinr": s["serv_sinr"],
                    "s_neigh1_pci": s["neigh1_pci"],
                    "s_neigh1_rsrp": s["neigh1_rsrp"],
                    "s_neigh1_rsrq": s["neigh1_rsrq"],
                    "s_neigh2_pci": s["neigh2_pci"],
                    "s_neigh2_rsrp": s["neigh2_rsrp"],
                    "s_neigh2_rsrq": s["neigh2_rsrq"],
                    "s_cell_thp_dl": s["cell_thp_dl"],
                    "s_cell_prb_use_perc_dl": s["cell_prb_use_perc_dl"],
                    "s_cell_prb_use_perc_ul": s["cell_prb_use_perc_ul"],
                    "a_handover": is_ho,
                    "a_target_du": tgt_du,
                    "a_source_du": src_du,
                    "r": reward,
                    "r_thp_delta": r_delta,
                    "done": done,
                    "sp_du_id": tgt_du,
                    "sp_serv_rsrp": sp["serv_rsrp"],
                    "sp_serv_rsrq": sp["serv_rsrq"],
                    "sp_serv_sinr": sp["serv_sinr"],
                    "sp_neigh1_pci": sp["neigh1_pci"],
                    "sp_neigh1_rsrp": sp["neigh1_rsrp"],
                    "sp_neigh1_rsrq": sp["neigh1_rsrq"],
                    "sp_neigh2_pci": sp["neigh2_pci"],
                    "sp_neigh2_rsrp": sp["neigh2_rsrp"],
                    "sp_neigh2_rsrq": sp["neigh2_rsrq"],
                    "sp_cell_thp_dl": sp["cell_thp_dl"],
                    "sp_cell_prb_use_perc_dl": sp["cell_prb_use_perc_dl"],
                    "sp_cell_prb_use_perc_ul": sp["cell_prb_use_perc_ul"],
                }
                writer.writerow(row)
                n_trans += 1

                if args.max_transitions > 0 and n_trans >= args.max_transitions:
                    break

            if args.max_transitions > 0 and n_trans >= args.max_transitions:
                break

    ho_ratio = (float(n_ho) / float(n_trans)) if n_trans > 0 else 0.0
    print("[offline-ds] Done.")
    print(f"[offline-ds] Output: {args.out}")
    print(f"[offline-ds] SUCIs used: {n_suci_used}")
    print(f"[offline-ds] Transitions: {n_trans}")
    print(f"[offline-ds] Handover transitions: {n_ho} ({ho_ratio:.2%})")


if __name__ == "__main__":
    main()
