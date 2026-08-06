#!/usr/bin/env python3

import argparse
import signal
from lib.xAppBase import xAppBase
import multiprocessing as mp
import threading
import queue
import os
import socket
import json
import sqlite3
import time
import datetime
import math

# ─── 5-RU constants ──────────────────────────────────────────────────────────

NUM_CELLS = 5  # DU 1 … 5

# DU ID (1-5) → NR Cell ID used in control_handover()
DU_TO_NR_CELL: dict[int, int] = {
    1: int("0x2131", 16),
    2: int("0x2132", 16),
    3: int("0x2133", 16),
    4: int("0x2134", 16),
    5: int("0x2135", 16),
}

ACTION_IDX_TO_DU: dict[int, int] = {i: i + 1 for i in range(NUM_CELLS)}
DU_TO_ACTION_IDX: dict[int, int] = {v: k for k, v in ACTION_IDX_TO_DU.items()}

HYSTERESIS_DB: float = 3.0

HO_COOLDOWN_SEC: float = 3.0

def parse_nr_cell_id(value):
    if isinstance(value, int):
        return value
    try:
        if value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"NR Cell ID must be an integer or hex string, got '{value}'"
        )


class MyHOXapp(xAppBase):
    """
    Responsibilities:
      1. Receive RL HO commands via UDP (port 6006) and execute them.
      2. Run rule-based HO loop: A3 condition filtered by topology.
      3. Periodically publish UE state + action masks to RL agent via UDP (port 6007).
    """

    def __init__(self, config, http_server_port, rmr_port,
                 ho_e2_node_id, ho_plmn, ho_gnb_cu_ue_f1ap_id):
        super().__init__(config, http_server_port, rmr_port)
        self.ho_e2_node_id = ho_e2_node_id
        self.ho_plmn = ho_plmn
        self.ho_gnb_cu_ue_f1ap_id = ho_gnb_cu_ue_f1ap_id
        self.db_path = os.environ.get("RAN_DB", "/path/to/your/data/ran.db")

        self._last_rule_ho: dict[str, float] = {}
        self._ho_trigger_count: dict[str, int] = {}
        self._last_processed_meas_id: dict[str, int] = {}
        self._rule_meas_max_age_sec: float = 2.0

    # ── start ────────────────────────────────────────────────────────────────

    @xAppBase.start_function
    def start(self, udp_ip="0.0.0.0", udp_port=6006,
              state_pub_ip="127.0.0.1", state_pub_port=6007,
              state_pub_interval=1.0):
        if not self._start_udp_listener(udp_ip=udp_ip, udp_port=udp_port):
            print("[HOXapp] UDP listener failed, exiting")
            return
        self._start_rule_based_ho()
        self._start_state_publisher(
            dest_ip=state_pub_ip,
            dest_port=state_pub_port,
            interval=state_pub_interval,
        )
        print("[HOXapp] Ready — rule-based HO active, RL commands accepted")
        while True:
            time.sleep(1)

    def _start_udp_listener(self, udp_ip="0.0.0.0", udp_port=6006) -> bool:
        # Pre-bind check
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((udp_ip, udp_port))
            s.close()
        except Exception as e:
            print(f"[HOXapp][UDP] Cannot bind {udp_ip}:{udp_port}: {e}")
            return False

        def _loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((udp_ip, udp_port))
            print(f"[HOXapp][UDP] Listening on {udp_ip}:{udp_port}")
            while True:
                try:
                    data, addr = sock.recvfrom(8192)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") != "HO_COMMAND":
                        continue
                    cmds = msg.get("commands", [])
                    print(f"[HOXapp][UDP] {len(cmds)} HO cmd(s) from {addr}")
                    for cmd in cmds:
                        self._execute_ho(cmd)
                except Exception as e:
                    print("[HOXapp][UDP] Error:", e)

        threading.Thread(target=_loop, daemon=True).start()
        return True

    def _execute_ho(self, ho_cmd: dict):
        amf_ue_ngap_id = ho_cmd["amf_ue_ngap_id"]
        raw_target = ho_cmd["target_nr_cell_id"]
        target_nr_cell_id = parse_nr_cell_id(raw_target)
        # If caller passed DU ID (1-5) instead of NR Cell hex, map it
        if target_nr_cell_id in DU_TO_NR_CELL:
            target_nr_cell_id = DU_TO_NR_CELL[target_nr_cell_id]
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{ts} [HOXapp] HO -> node={self.ho_e2_node_id} "
              f"amf={amf_ue_ngap_id} target={hex(target_nr_cell_id)}")
        self.e2sm_rc.control_handover(
            self.ho_e2_node_id,
            amf_ue_ngap_id,
            self.ho_gnb_cu_ue_f1ap_id,
            self.ho_plmn,
            target_nr_cell_id,
        )

    # ── Rule-based HO loop ───────────────────────────────────────────────────

    def _start_rule_based_ho(self, interval_sec: float = 0.5):
        def _loop():
            while True:
                try:
                    for suci in self._get_active_sucis():
                        should_ho, target_du, meas_id = self._ho_decision(suci)
                        if meas_id is not None:
                            self._last_processed_meas_id[suci] = meas_id
                        if not should_ho or target_du is None:
                            self._ho_trigger_count[suci] = 0
                            continue
                        # Require 2 consecutive triggers to avoid transient spikes
                        self._ho_trigger_count[suci] = (
                            self._ho_trigger_count.get(suci, 0) + 1
                        )
                        if self._ho_trigger_count[suci] < 2:
                            continue
                        self._ho_trigger_count[suci] = 0

                        amf_id = self._get_amf_for_suci(suci)
                        if amf_id is None:
                            continue
                        nr_cell_id = DU_TO_NR_CELL.get(int(target_du))
                        if nr_cell_id is None:
                            continue

                        now = time.time()
                        if now - self._last_rule_ho.get(suci, 0.0) < HO_COOLDOWN_SEC:
                            continue
                        self._last_rule_ho[suci] = now

                        print(
                            f"[HOXapp][RULE] SUCI={suci} AMF={amf_id} "
                            f"target_du={target_du} cell={hex(nr_cell_id)}"
                        )
                        self.e2sm_rc.control_handover(
                            self.ho_e2_node_id, amf_id,
                            self.ho_gnb_cu_ue_f1ap_id,
                            self.ho_plmn, nr_cell_id,
                        )
                except Exception as e:
                    print("[HOXapp][RULE] Error:", e)
                time.sleep(interval_sec)

        threading.Thread(target=_loop, daemon=True).start()

    # ── A3-based HO decision ─────────────────────────────────────────────────

    def _ho_decision(self, ue_suci: str):
        """
        Return (should_ho, target_du_id, meas_id).

        HO is triggered when any reported neighbor satisfies the A3 condition:
          neighbor_rsrp >= serving_rsrp + HYSTERESIS_DB

        No fixed topology constraint — candidates come directly from the
        neigh1/neigh2 fields in cu_ue_meas. The best-RSRP neighbor wins.
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, ts, du_id, serv_rsrp,
                       neigh1_pci, neigh1_rsrp,
                       neigh2_pci, neigh2_rsrp
                FROM cu_ue_meas
                WHERE suci = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (ue_suci,),
            )
            row = cur.fetchone()
            if not row:
                return False, None, None

            meas_id, _ts, serv_du_id, serv_rsrp, n1_pci, n1_rsrp, n2_pci, n2_rsrp = row

            # Skip already-processed measurements
            if meas_id <= self._last_processed_meas_id.get(ue_suci, 0):
                return False, None, None

            # Staleness check: compare against the freshest CU record in DB
            cur.execute("SELECT MAX(ts) FROM cu_ue_meas")
            max_ts_row = cur.fetchone()
            latest_ts = float(max_ts_row[0]) if (max_ts_row and max_ts_row[0] is not None) else float(_ts)
            if latest_ts - float(_ts) > self._rule_meas_max_age_sec:
                return False, None, meas_id

            if serv_rsrp is None:
                return False, None, meas_id

            serv_rsrp_f = float(serv_rsrp)

            # Collect neighbor candidates that satisfy A3 (no topology filter)
            def _safe_int(v):
                try:
                    return int(v)
                except Exception:
                    return None

            candidates = []
            for pci_raw, rsrp_raw in [(n1_pci, n1_rsrp), (n2_pci, n2_rsrp)]:
                if pci_raw is None or rsrp_raw is None:
                    continue
                pci = _safe_int(pci_raw)
                if pci is None:
                    continue
                rsrp = float(rsrp_raw)
                if rsrp >= serv_rsrp_f + HYSTERESIS_DB:
                    candidates.append((pci, rsrp))

            if not candidates:
                return False, None, meas_id

            # Pick the neighbor with the best RSRP
            best_du, _ = max(candidates, key=lambda x: x[1])
            return True, best_du, meas_id

        finally:
            conn.close()

    # ── State + action-mask publisher ────────────────────────────────────────

    def _start_state_publisher(self, dest_ip: str, dest_port: int, interval: float):
        """
        Periodically compute per-UE state + action masks and send to RL agent.

        Packet format (JSON):
        {
          "type": "STATE_UPDATE",
          "ts": <float>,
          "ues": [
            {
              "suci": "...",
              "amf": <int|null>,
              "serv_du": <int>,
              "serv_rsrp": <float>,
              "neigh": [{"du": <int>, "rsrp": <float>}, ...],
              "action_mask": [stay_ok, ho_ok, t1, t2, t3, t4, t5]  // length 7
            },
            ...
          ]
        }
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def _loop():
            while True:
                try:
                    payload = self._build_state_payload()
                    data = json.dumps(payload).encode("utf-8")
                    sock.sendto(data, (dest_ip, dest_port))
                except Exception as e:
                    print("[HOXapp][PUB] Error:", e)
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()
        print(f"[HOXapp][PUB] State publisher -> {dest_ip}:{dest_port} every {interval}s")

    def _build_state_payload(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT suci FROM cu_ue_meas
                WHERE suci IS NOT NULL ORDER BY ts DESC LIMIT 50
                """
            )
            sucis = [r[0] for r in cur.fetchall()]

            ue_list = []
            for suci in sucis:
                cur.execute(
                    """
                    SELECT serv_rsrp, neigh1_pci, neigh1_rsrp,
                           neigh2_pci, neigh2_rsrp, du_id, amf
                    FROM cu_ue_meas
                    WHERE suci = ?
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    (suci,),
                )
                row = cur.fetchone()
                if not row:
                    continue
                serv_rsrp, n1_pci, n1_rsrp, n2_pci, n2_rsrp, du_id, amf = row

                def _si(v):
                    try:
                        return int(v)
                    except Exception:
                        return None

                def _sf(v):
                    try:
                        return float(v)
                    except Exception:
                        return None

                serv_du = _si(du_id)
                serv_rsrp_f = _sf(serv_rsrp)

                # Build neighbor list from measurement data
                neighbors = []
                for pci_raw, rsrp_raw in [(n1_pci, n1_rsrp), (n2_pci, n2_rsrp)]:
                    pci = _si(pci_raw)
                    rsrp = _sf(rsrp_raw)
                    if pci is not None and rsrp is not None:
                        neighbors.append({"du": pci, "rsrp": rsrp})

                # Build action mask: [stay_ok, ho_ok, du1, du2, du3, du4, du5]
                # A3 condition only — no topology filter
                target_mask = [0] * NUM_CELLS  # index i -> DU(i+1)
                ho_ok = False
                if serv_rsrp_f is not None:
                    for nb in neighbors:
                        if nb["rsrp"] >= serv_rsrp_f + HYSTERESIS_DB:
                            idx = DU_TO_ACTION_IDX.get(nb["du"])
                            if idx is not None:
                                target_mask[idx] = 1
                                ho_ok = True

                # If no valid HO target: allow stay action only; keep target mask all-zero
                # (RL agent should sample stay in this case)
                action_mask = [1, 1 if ho_ok else 0] + target_mask

                ue_list.append({
                    "suci": suci,
                    "amf": _si(amf),
                    "serv_du": serv_du,
                    "serv_rsrp": serv_rsrp_f,
                    "neigh": neighbors,
                    "action_mask": action_mask,
                })

            return {
                "type": "STATE_UPDATE",
                "ts": time.time(),
                "ues": ue_list,
            }
        finally:
            conn.close()

    # ── DB helpers ───────────────────────────────────────────────────────────

    def _get_active_sucis(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT suci FROM cu_ue_meas
                WHERE suci IS NOT NULL ORDER BY ts DESC LIMIT 50
                """
            )
            return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    def _get_amf_for_suci(self, ue_suci: str):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT amf FROM cu_ue_meas WHERE suci = ? ORDER BY ts DESC LIMIT 1",
                (ue_suci,),
            )
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  MyXapp — KPM + CU UDP data collection (5 DUs)
# ═══════════════════════════════════════════════════════════════════════════════

class MyXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        super().__init__(config, http_server_port, rmr_port)
        self.db_path = "/path/to/your/data/ran.db"
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError as e:
            print(f"[DB] Warning: {e}, falling back to /tmp/ran.db")
            self.db_path = "/tmp/ran.db"

        self._init_sqlite()
        self.start_time = time.time()
        self._udp_print_cnt = 0
        self._kpm_print_cnt = 0
        self._PRINT_EVERY_UDP = 50
        self._PRINT_EVERY_KPM = 20
        self.suci_cache: dict = {}
        self.db_queue: queue.Queue = queue.Queue()
        self._start_db_writer()

    # ── SQLite init ──────────────────────────────────────────────────────────

    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            print(f"[DB] Failed to init: {e}")
            return
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS cu_ue_meas ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, suci TEXT, "
            "ran INTEGER, amf INTEGER, ue_id INTEGER, du_id INTEGER, du_ue INTEGER, "
            "serv_rsrp REAL, serv_rsrq REAL, serv_sinr REAL, "
            "neigh1_pci INTEGER, neigh1_rsrp REAL, neigh1_rsrq REAL, neigh1_sinr REAL, "
            "neigh2_pci INTEGER, neigh2_rsrp REAL, neigh2_rsrq REAL, neigh2_sinr REAL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS du_ue_kpm ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, du_id TEXT, ue_id INTEGER, "
            "suci TEXT, thp_ul REAL, thp_dl REAL, rlc_delay_dl REAL, rlc_pkt_drop REAL, "
            "vol_dl REAL, vol_ul REAL, air_if_delay_ul REAL, rlc_delay_ul REAL, "
            "cqi REAL, rsrp REAL, rsrq REAL, prb_used_dl REAL, prb_used_ul REAL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS du_cell_kpm ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, du_id TEXT, "
            "prb_used_dl REAL, prb_used_ul REAL, prb_use_perc_dl REAL, prb_use_perc_ul REAL, "
            "thp_dl REAL, thp_ul REAL)"
        )
        cur.execute("DELETE FROM cu_ue_meas")
        cur.execute("DELETE FROM du_ue_kpm")
        cur.execute("DELETE FROM du_cell_kpm")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cu_ts    ON cu_ue_meas(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_du_ue_ts ON du_ue_kpm(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_du_cl_ts ON du_cell_kpm(ts)")
        conn.commit()
        conn.close()

    # ── Async DB writer ──────────────────────────────────────────────────────

    def _start_db_writer(self):
        SQL_MAP = {
            "du_cell_kpm": (
                "INSERT INTO du_cell_kpm "
                "(ts, du_id, prb_used_dl, prb_used_ul, prb_use_perc_dl, prb_use_perc_ul, thp_dl, thp_ul) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            "du_ue_kpm": (
                "INSERT INTO du_ue_kpm "
                "(ts, du_id, ue_id, suci, thp_ul, thp_dl, rlc_delay_dl, rlc_pkt_drop, "
                "vol_dl, vol_ul, air_if_delay_ul, rlc_delay_ul, cqi, rsrp, rsrq, prb_used_dl, prb_used_ul) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            "cu_ue_meas": (
                "INSERT INTO cu_ue_meas VALUES "
                "(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
        }

        def _writer():
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            batch = []
            last_commit = time.time()
            last_ckpt = time.time()
            while True:
                try:
                    item = self.db_queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    pass
                if batch and (len(batch) >= 200 or time.time() - last_commit > 1.0):
                    for table, values in batch:
                        try:
                            cur.execute(SQL_MAP[table], values)
                        except sqlite3.Error as e:
                            print(f"[DB] Insert error: {e}")
                    conn.commit()
                    batch.clear()
                    last_commit = time.time()
                    if time.time() - last_ckpt > 5.0:
                        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        last_ckpt = time.time()

        threading.Thread(target=_writer, daemon=True).start()

    # ── KPM callback ────────────────────────────────────────────────────────

    def make_kpm_callback(self, kpm_report_style, ue_id=None):
        def _cb(e2_agent_id, subscription_id, indication_hdr, indication_msg):
            self.my_subscription_callback(
                e2_agent_id, subscription_id,
                indication_hdr, indication_msg,
                kpm_report_style, ue_id,
            )
        return _cb

    def my_subscription_callback(self, e2_agent_id, subscription_id,
                                  indication_hdr, indication_msg,
                                  kpm_report_style, ue_id):
        self._kpm_print_cnt += 1
        do_print = (self._kpm_print_cnt % self._PRINT_EVERY_KPM == 0)
        if do_print:
            print(f"\n[KPM] From {e2_agent_id} | Sub {subscription_id} | Style {kpm_report_style}")

        indication_hdr = self.e2sm_kpm.extract_hdr_info(indication_hdr)
        meas_data = self.e2sm_kpm.extract_meas_data(indication_msg)
        timestamp = indication_hdr.get("colletStartTime")

        if kpm_report_style in (1, 2):
            if kpm_report_style == 1:
                self._store_du_cell_kpm(e2_agent_id, timestamp, meas_data["measData"])
        else:
            if do_print:
                print("UE count:", len(meas_data.get("ueMeasData", {})))
            for current_ue_id, ue_meas_data in meas_data["ueMeasData"].items():
                self._store_du_kpm(e2_agent_id, current_ue_id, e2_agent_id,
                                   timestamp, ue_meas_data["measData"])

    # ── UDP listener (CU measurements) ──────────────────────────────────────

    def start_udp_listener(self, udp_ip="0.0.0.0", udp_port=5005):
        def _loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((udp_ip, udp_port))
            print(f"[KPM][UDP] Listening on {udp_ip}:{udp_port}")
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    self._udp_print_cnt += 1
                    if self._udp_print_cnt % self._PRINT_EVERY_UDP == 0:
                        print(f"[UDP] From {addr}")
                    self._store_cu_udp(msg)
                except Exception as e:
                    print("[UDP] Error:", e)

        threading.Thread(target=_loop, daemon=True).start()

    def _store_cu_udp(self, msg: dict):
        du_id_n = self._normalize_du_id(msg.get("du_id"))
        du_ue_n = self._normalize_ue_id(msg.get("cu_ue_f1ap_id"))
        suci_n = self._normalize_text(msg.get("suci"))
        if du_id_n is not None and du_ue_n is not None and suci_n is not None:
            self.suci_cache[(du_id_n, du_ue_n)] = suci_n

        ts = time.time() - self.start_time
        values = (
            ts,
            msg.get("suci"), msg.get("ran_ue_id"), msg.get("amf_ue_id"),
            msg.get("ue_index"), msg.get("du_id"), msg.get("cu_ue_f1ap_id"),
            msg.get("serv_rsrp"), msg.get("serv_rsrq"), msg.get("serv_sinr"),
            msg.get("neigh1_pci"), msg.get("neigh1_rsrp"),
            msg.get("neigh1_rsrq"), msg.get("neigh1_sinr"),
            msg.get("neigh2_pci"), msg.get("neigh2_rsrp"),
            msg.get("neigh2_rsrq"), msg.get("neigh2_sinr"),
        )
        self.db_queue.put(("cu_ue_meas", values))

    # ── KPM storage ──────────────────────────────────────────────────────────

    def _store_du_cell_kpm(self, du_id, ts, metrics: dict):
        ts_val = time.time() - self.start_time
        du_id = self._normalize_du_id(du_id)
        values = (
            ts_val, du_id,
            self._nf(metrics.get("RRU.PrbUsedDl")),
            self._nf(metrics.get("RRU.PrbUsedUl")),
            self._nf(metrics.get("RRU.PrbUsedDl")),
            self._nf(metrics.get("RRU.PrbUsedUl")),
            self._nf(metrics.get("DRB.UEThpDl")),
            self._nf(metrics.get("DRB.UEThpUl")),
        )
        self.db_queue.put(("du_cell_kpm", values))

    def _store_du_kpm(self, du_id, ue_id, cell_id, ts, metrics: dict):
        ts_val = time.time() - self.start_time
        du_id = self._normalize_du_id(du_id)
        ue_id = self._normalize_ue_id(ue_id)
        suci = self.suci_cache.get((du_id, ue_id))
        values = (
            ts_val, du_id, ue_id, suci,
            self._nf(metrics.get("DRB.UEThpUl")),
            self._nf(metrics.get("DRB.UEThpDl")),
            self._nf(metrics.get("DRB.RlcSduDelayDl")),
            self._nf(metrics.get("DRB.RlcPacketDropRateDl")),
            self._nf(metrics.get("DRB.RlcSduTransmittedVolumeDL")),
            self._nf(metrics.get("DRB.RlcSduTransmittedVolumeUL")),
            self._nf(metrics.get("DRB.AirIfDelayUl")),
            self._nf(metrics.get("DRB.RlcDelayUl")),
            0,  # CQI (not in Style-4 report)
            0,  # RSRP
            0,  # RSRQ
            self._nf(metrics.get("RRU.PrbUsedDl")),
            self._nf(metrics.get("RRU.PrbUsedUl")),
        )
        self.db_queue.put(("du_ue_kpm", values))

    # ── Normalization helpers ────────────────────────────────────────────────

    def _normalize_ue_id(self, ue_id):
        if ue_id is None:
            return None
        if isinstance(ue_id, dict) and "ueId" in ue_id:
            return int(ue_id["ueId"])
        try:
            return int(ue_id)
        except Exception:
            return None

    def _normalize_text(self, v):
        return str(v) if v is not None else None

    def _normalize_du_id(self, v):
        if v is None:
            return None
        digits = [c for c in str(v) if c.isdigit()]
        return digits[-1] if digits else None

    def _unwrap_metric(self, v):
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    def _nf(self, v) -> float:
        if v is None:
            return 0.0
        v = self._unwrap_metric(v)
        try:
            result = float(v)
            return 0.0 if math.isnan(result) else result
        except Exception:
            return 0.0

    # ── KPM subscription ────────────────────────────────────────────────────

    @xAppBase.start_function
    def start(self, e2_node_id, kpm_report_style, ue_ids, metric_names):
        report_period = 10
        granul_period = 10
        if kpm_report_style == 1:
            print(f"[KPM] Style 1 — {e2_node_id}")
            cb = self.make_kpm_callback(1)
            self.e2sm_kpm.subscribe_report_service_style_1(
                e2_node_id, report_period, metric_names, granul_period, cb
            )
        elif kpm_report_style == 4:
            print(f"[KPM] Style 4 — {e2_node_id}")
            matchingUeConds = [{
                "testCondInfo": {
                    "testType": ("ul-rSRP", "true"),
                    "testExpr": "lessthan",
                    "testValue": ("valueInt", 1000),
                }
            }]
            cb = self.make_kpm_callback(4)
            self.e2sm_kpm.subscribe_report_service_style_4(
                e2_node_id, report_period, matchingUeConds, metric_names, granul_period, cb
            )
        else:
            print(f"[KPM] Style {kpm_report_style} not implemented")


# ═══════════════════════════════════════════════════════════════════════════════
#  HO process entry point
# ═══════════════════════════════════════════════════════════════════════════════

def start_ho_process(config, http_port, rmr_port, rc_ran_func_id,
                     ho_e2_node_id, ho_plmn, ho_gnb_cu_ue_f1ap_id,
                     state_pub_ip, state_pub_port, state_pub_interval):
    ho_xapp = MyHOXapp(
        config, http_port, rmr_port,
        ho_e2_node_id, ho_plmn, ho_gnb_cu_ue_f1ap_id,
    )
    ho_xapp.e2sm_rc.set_ran_func_id(rc_ran_func_id)
    ho_xapp.start(
        udp_ip="0.0.0.0", udp_port=6006,
        state_pub_ip=state_pub_ip,
        state_pub_port=state_pub_port,
        state_pub_interval=state_pub_interval,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Margo 5-RU KPM + HO xApp")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--http_server_port", type=int, default=8092)
    parser.add_argument("--rmr_port", type=int, default=4562)
    # E2 node IDs for all 5 DUs
    parser.add_argument("--e2_node_id_1", type=str, default="gnbd_001_001_00000213_1")
    parser.add_argument("--e2_node_id_2", type=str, default="gnbd_001_001_00000213_2")
    parser.add_argument("--e2_node_id_3", type=str, default="gnbd_001_001_00000213_3")
    parser.add_argument("--e2_node_id_4", type=str, default="gnbd_001_001_00000213_4")
    parser.add_argument("--e2_node_id_5", type=str, default="gnbd_001_001_00000213_5")
    # RAN function IDs
    parser.add_argument("--ran_func_id", type=int, default=2, help="KPM RAN function ID")
    parser.add_argument("--rc_ran_func_id", type=int, default=3, help="RC RAN function ID")
    # KPM config
    parser.add_argument("--kpm_report_style", type=int, default=4)
    parser.add_argument("--ue_ids", type=str, default="0")
    parser.add_argument(
        "--metrics", type=str,
        default=(
            "DRB.UEThpUl,DRB.UEThpDl,DRB.RlcSduDelayDl,DRB.RlcPacketDropRateDl,"
            "DRB.RlcSduTransmittedVolumeDL,DRB.RlcSduTransmittedVolumeUL,"
            "DRB.AirIfDelayUl,DRB.RlcDelayUl,RRU.PrbUsedDl,RRU.PrbUsedUl"
        ),
    )
    # HO xApp params
    parser.add_argument("--ho_e2_node_id", type=str, default="gnb_001_001_00000213")
    parser.add_argument("--ho_plmn", type=str, default="00101")
    parser.add_argument("--ho_gnb_cu_ue_f1ap_id", type=int, default=1)
    # State publisher (to RL agent)
    parser.add_argument("--state_pub_ip", type=str, default="127.0.0.1",
                        help="IP to send state+mask updates (RL agent listener)")
    parser.add_argument("--state_pub_port", type=int, default=6007,
                        help="UDP port for state+mask updates")
    parser.add_argument("--state_pub_interval", type=float, default=1.0,
                        help="Seconds between state+mask broadcasts")
    # Feature flags
    parser.add_argument("--enable_rule_ho", action="store_true", default=True,
                        help="Enable rule-based HO (A3 condition)")
    parser.add_argument("--enable_rl_ho", action="store_true", default=True,
                        help="Enable RL HO command receiver (UDP 6006)")

    args = parser.parse_args()
    node_ids = [
        args.e2_node_id_1,
        args.e2_node_id_2,
        args.e2_node_id_3,
        args.e2_node_id_4,
        args.e2_node_id_5,
    ]
    metrics = args.metrics.split(",")
    ue_ids = list(map(int, args.ue_ids.split(",")))

    # ── Launch HO xApp process ───────────────────────────────────────────────
    print("[Main] Launching HO process…")
    ho_proc = mp.Process(
        target=start_ho_process,
        args=(
            args.config,
            args.http_server_port - 2,
            args.rmr_port - 2,
            args.rc_ran_func_id,
            args.ho_e2_node_id,
            args.ho_plmn,
            args.ho_gnb_cu_ue_f1ap_id,
            args.state_pub_ip,
            args.state_pub_port,
            args.state_pub_interval,
        ),
        daemon=False,
    )
    ho_proc.start()

    # ── Launch KPM xApp ──────────────────────────────────────────────────────
    print(f"[Main] Initializing KPM xApp (KPM func {args.ran_func_id})…")
    myXapp = MyXapp(args.config, args.http_server_port, args.rmr_port)
    myXapp.e2sm_kpm.set_ran_func_id(args.ran_func_id)
    myXapp.start_udp_listener(udp_ip="0.0.0.0", udp_port=5005)

    signal.signal(signal.SIGQUIT, myXapp.signal_handler)
    signal.signal(signal.SIGTERM, myXapp.signal_handler)
    signal.signal(signal.SIGINT, myXapp.signal_handler)

    cell_metrics = [
        "RRU.PrbUsedDl", "RRU.PrbUsedUl",
        "RRU.PrbTotDl",  "RRU.PrbTotUl",
        "DRB.UEThpDl",   "DRB.UEThpUl",
    ]

    # Style-1 (cell-level) subscriptions for all 5 DUs
    for nid in node_ids:
        t = threading.Thread(target=myXapp.start, args=(nid, 1, [], cell_metrics))
        t.daemon = True
        t.start()

    # Style-4 (UE-level) subscriptions for all 5 DUs
    for nid in node_ids:
        t = threading.Thread(
            target=myXapp.start,
            args=(nid, args.kpm_report_style, ue_ids, metrics),
        )
        t.daemon = True
        t.start()

    print("[Main] All 5 DUs active. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
