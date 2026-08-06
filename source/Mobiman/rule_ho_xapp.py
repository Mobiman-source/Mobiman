
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

def parse_nr_cell_id(value):
    if isinstance(value, int):
        return value
    try:
        if value.startswith('0x'):
            return int(value, 16)
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"NR Cell ID must be an integer or hex string (e.g. '0x19B1'), got '{value}'")

class MyHOXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port, ho_e2_node_id, ho_plmn, ho_gnb_cu_ue_f1ap_id):
        super(MyHOXapp, self).__init__(config, http_server_port, rmr_port)
        self.ho_e2_node_id = ho_e2_node_id
        self.ho_plmn = ho_plmn
        self.ho_gnb_cu_ue_f1ap_id = ho_gnb_cu_ue_f1ap_id
        self.db_path = os.environ.get("RAN_DB", "/path/to/your/data/ran.db")
        self._du_to_nr_cell = {
            1: int("0x2131", 16),
            2: int("0x2132", 16),
            3: int("0x2133", 16),
            4: int("0x2134", 16),
            5: int("0x2135", 16),
        }
        self._last_rule_ho = {}
        self._ho_trigger_count = {}  # suci -> consecutive trigger count

    def start_udp_listener(self, udp_ip="0.0.0.0", udp_port=6006):
        def udp_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind((udp_ip, udp_port))
                print(f"[HOXapp][UDP] Listening for RL actions on {udp_ip}:{udp_port}")
            except Exception as e:
                print(f"[HOXapp][UDP] Failed to bind port {udp_port}: {e}")
                return

            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") != "HO_COMMAND":
                        continue
                    cmds = msg.get("commands", [])
                    print(f"[HOXapp][UDP] Received {len(cmds)} HO commands from {addr}: {cmds}")
                    for ho_cmd in cmds:
                        self._execute_ho(ho_cmd)

                except Exception as e:
                    print("[HOXapp][UDP] Error:", e)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((udp_ip, udp_port))
        sock.close()

        t = threading.Thread(target=udp_loop, daemon=True)
        t.start()
        return True

    @xAppBase.start_function
    def start(self, udp_ip="0.0.0.0", udp_port=6006):
        if not self.start_udp_listener(udp_ip=udp_ip, udp_port=udp_port):
            print("[HOXapp] UDP listener not running, exiting")
            return
        self._start_rule_based_ho()
        print("[HOXapp] Ready to receive RL actions")
        while True:
            time.sleep(1)

    # RSRP(dBm) = -140 + value
    #
    # SINR(dB) = -23 + 0.5×value

    def _execute_ho(self, ho_cmd):

        amf_ue_ngap_id = ho_cmd["amf_ue_ngap_id"]
        target_nr_cell_id = parse_nr_cell_id(ho_cmd["target_nr_cell_id"])
        if target_nr_cell_id in self._du_to_nr_cell:
            target_nr_cell_id = self._du_to_nr_cell[target_nr_cell_id]

        current_time = datetime.datetime.now()
        print(f"{current_time.strftime('%H:%M:%S')} Sending HO command -> Node: {self.ho_e2_node_id}, TargetID: {target_nr_cell_id}")

        self.e2sm_rc.control_handover(
            self.ho_e2_node_id,
            amf_ue_ngap_id,
            self.ho_gnb_cu_ue_f1ap_id,
            self.ho_plmn,
            target_nr_cell_id,
        )
    # rule-based part, DU 2: static suci 11, DU 1: static suci 03, DU 5: static suci 09
    def _start_rule_based_ho(self, interval_sec: float = 0.5):   #evey 0.5s check once
        def rule_loop():
            while True:
                try:
                    for suci in self._get_active_sucis():
                        should_ho, target_du = self.rule_based_ho_decision(suci)
                        if should_ho and target_du is not None:
                            self._ho_trigger_count[suci] = self._ho_trigger_count.get(suci, 0) + 1
                        else:
                            self._ho_trigger_count[suci] = 0
                            continue
                        if self._ho_trigger_count.get(suci, 0) < 2:
                            continue
                        self._ho_trigger_count[suci] = 0
                        amf_ue_ngap_id = self._get_amf_for_suci(suci)
                        if amf_ue_ngap_id is None:
                            continue
                        target_nr_cell_id = self._du_to_nr_cell.get(int(target_du))
                        if target_nr_cell_id is None:
                            continue
                        last_ts = self._last_rule_ho.get(suci, 0.0)
                        now_ts = time.time()
                        if now_ts - last_ts < 1.0:
                            continue
                        self._last_rule_ho[suci] = now_ts
                        print(
                            f"[HOXapp][RULE] SUCI={suci} AMF={amf_ue_ngap_id} "
                            f"target_du={target_du} target_nr_cell={hex(target_nr_cell_id)}"
                        )
                        self.e2sm_rc.control_handover(
                            self.ho_e2_node_id,
                            amf_ue_ngap_id,
                            self.ho_gnb_cu_ue_f1ap_id,
                            self.ho_plmn,
                            target_nr_cell_id,
                        )
                except Exception as e:
                    print("[HOXapp][RULE] Error:", e)
                time.sleep(interval_sec)

        t = threading.Thread(target=rule_loop, daemon=True)
        t.start()

    def _get_active_sucis(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT DISTINCT suci
                FROM cu_ue_meas
                WHERE suci IS NOT NULL
                ORDER BY ts DESC
                LIMIT 50
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
                """
                SELECT amf
                FROM cu_ue_meas
                WHERE suci = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (ue_suci,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return row[0]
        finally:
            conn.close()

    def rule_based_ho_decision(
        self,
        ue_suci: str,
        hysteresis_db: float = 10.0,
        max_prb_perc: float = 200,
        fallback_serv_rsrp_db: float = 45,
    ):
        """Return (should_ho, target_du_id) based on simple RSRP + load rules."""
        # Per-serving-cell RSRP thresholds for fallback HO trigger.
        rsrp_threshold_by_serv_du = {
            4: 50,
            1: 40,
            # 2: 49,  # RU2 disabled
        }
        # Per-serving-cell handover policy: RU1 <-> RU4 only.
        allowed_targets_by_serv_du = {
            1: {4},
            4: {1},
        }
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT ts, du_id, serv_rsrp,
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
                return False, None

            _ts, serv_du_id, serv_rsrp, n1_pci, n1_rsrp, n2_pci, n2_rsrp = row
            if serv_rsrp is None:
                return False, None
            try:
                serv_du_id_int = int(serv_du_id)
            except Exception:
                return False, None

            def _safe_int(v):
                try:
                    return int(v)
                except Exception:
                    return None

            candidates = []
            if n1_pci is not None and n1_rsrp is not None:
                n1_pci_int = _safe_int(n1_pci)
                if n1_pci_int is not None:
                    candidates.append((n1_pci_int, float(n1_rsrp)))
            if n2_pci is not None and n2_rsrp is not None:
                n2_pci_int = _safe_int(n2_pci)
                if n2_pci_int is not None:
                    candidates.append((n2_pci_int, float(n2_rsrp)))

            allowed_targets = allowed_targets_by_serv_du.get(serv_du_id_int)
            if allowed_targets is not None:
                candidates = [c for c in candidates if c[0] in allowed_targets]

            if not candidates:
                # Fallback: neighbor RSRP missing due to data gaps; use service RSRP threshold.
                threshold = rsrp_threshold_by_serv_du.get(serv_du_id_int, fallback_serv_rsrp_db)
                if float(serv_rsrp) < threshold:
                    # def _du_load_ok(du_id: int) -> bool:
                    #     cur.execute(
                    #         """
                    #         SELECT prb_use_perc_dl, prb_use_perc_ul
                    #         FROM du_cell_kpm
                    #         WHERE du_id = ?
                    #         ORDER BY ts DESC
                    #         LIMIT 1
                    #         """,
                    #         (str(du_id),),
                    #     )
                    #     du_load = cur.fetchone()
                    #     if not du_load:
                    #         return True
                    #     prb_dl, prb_ul = du_load
                    #     if prb_dl is not None and prb_dl > max_prb_perc:
                    #         return False
                    #     if prb_ul is not None and prb_ul > max_prb_perc:
                    #         return False
                    #     return True

                    fallback_targets = []
                    # if serv_du_id_int == 5:  # RU5 disabled
                    #     fallback_targets = [1, 2]
                    if serv_du_id_int == 1:
                        # RU1 only allowed to HO to RU4
                        # Query last 3 measurements to check historical neighbor RSRP
                        cur.execute(
                            """
                            SELECT serv_rsrp, neigh1_pci, neigh1_rsrp, neigh2_pci, neigh2_rsrp
                            FROM cu_ue_meas
                            WHERE suci = ?
                            ORDER BY ts DESC
                            LIMIT 3
                            """,
                            (ue_suci,),
                        )
                        hist_rows = cur.fetchall()
                        neighbor_win_count = {}
                        seen_neighbor_pcis = set()
                        for r_serv, r_n1_pci, r_n1_rsrp, r_n2_pci, r_n2_rsrp in hist_rows:
                            if r_serv is None:
                                continue
                            for pci, rsrp in [(r_n1_pci, r_n1_rsrp), (r_n2_pci, r_n2_rsrp)]:
                                if pci is not None:
                                    pci_int = _safe_int(pci)
                                    if pci_int is not None and pci_int in allowed_targets_by_serv_du.get(1, set()):
                                        seen_neighbor_pcis.add(pci_int)
                                        if rsrp is not None and float(rsrp) > float(r_serv):
                                            neighbor_win_count[pci_int] = neighbor_win_count.get(pci_int, 0) + 1
                        # Prefer a neighbor that consistently beat serving RSRP 3 times
                        for pci, count in neighbor_win_count.items():
                            if count >= 3:
                                print(f"[HOXapp][RULE] DU1 SUCI={ue_suci} consistent neighbor pci={pci} wins={count}, HO to it")
                                return True, pci
                        # No consistent winner: fallback to RU4 only
                        all_du1_targets = [4]
                        for tgt in all_du1_targets:
                            if tgt not in seen_neighbor_pcis:
                                print(f"[HOXapp][RULE] DU1 SUCI={ue_suci} no consistent neighbor, HO to tgt={tgt} (no neighbor info)")
                                return True, tgt
                        # All targets seen as neighbors but none consistently won: pick first
                        return True, all_du1_targets[0]
                    # elif serv_du_id_int == 2:  # RU2 disabled
                    #     fallback_targets = [1]
                    # elif serv_du_id_int == 3:  # RU3 disabled
                    #     fallback_targets = [2]
                    elif serv_du_id_int == 4:
                        fallback_targets = [1]  # RU4 only HO to RU1

                    if allowed_targets is not None:
                        fallback_targets = [t for t in fallback_targets if t in allowed_targets]

                    for tgt in fallback_targets:
                        # if _du_load_ok(tgt):
                        return True, tgt
                return False, None

            best_du, best_rsrp = max(candidates, key=lambda x: x[1])
            if best_rsrp - float(serv_rsrp) < hysteresis_db:
                return False, None

            # cur.execute(
            #     """
            #     SELECT prb_use_perc_dl, prb_use_perc_ul
            #     FROM du_cell_kpm
            #     WHERE du_id = ?
            #     ORDER BY ts DESC
            #     LIMIT 1
            #     """,
            #     (str(best_du),),
            # )
            # du_load = cur.fetchone()
            # if du_load:
            #     prb_dl, prb_ul = du_load
            #     if prb_dl is not None and prb_dl > max_prb_perc:
            #         return False, None
            #     if prb_ul is not None and prb_ul > max_prb_perc:
            #         return False, None

            return True, best_du
        finally:
            conn.close()

class MyXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        super(MyXapp, self).__init__(config, http_server_port, rmr_port)
        self.db_path = "/path/to/your/data/ran.db"
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError as e:
            print(f"[DB] Warning: cannot create {os.path.dirname(self.db_path)} ({e}), fallback to /tmp/ran.db")
            self.db_path = "/tmp/ran.db"
            
        self._init_sqlite()
        self.start_time = time.time()

        self._udp_print_cnt = 0
        self._kpm_print_cnt = 0
        self._PRINT_EVERY_UDP = 50     
        self._PRINT_EVERY_KPM = 20     

        self.suci_cache = {}
        self.db_queue = queue.Queue()
        self._start_db_writer()

    def _start_db_writer(self):
        def writer_loop():
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
            except sqlite3.Error as e:
                print(f"[DB] Failed to open {self.db_path}: {e}")
                return
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")

            last_commit = time.time()
            batch = []
            last_checkpoint = time.time()

            # SQL definitions 
            SQL_MAP = {
                "du_cell_kpm": "INSERT INTO du_cell_kpm (ts, du_id, prb_used_dl, prb_used_ul, prb_use_perc_dl, prb_use_perc_ul, thp_dl, thp_ul) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                "du_ue_kpm": "INSERT INTO du_ue_kpm (ts, du_id, ue_id, suci, thp_ul, thp_dl, rlc_delay_dl, rlc_pkt_drop, vol_dl, vol_ul, air_if_delay_ul, rlc_delay_ul, cqi, rsrp, rsrq, prb_used_dl, prb_used_ul) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                "cu_ue_meas": "INSERT INTO cu_ue_meas VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            }

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

                    if time.time() - last_checkpoint > 5.0:
                        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                        last_checkpoint = time.time()

        t = threading.Thread(target=writer_loop, daemon=True)
        t.start()

    def make_kpm_callback(self, kpm_report_style, ue_id=None):
        def _cb(e2_agent_id, subscription_id, indication_hdr, indication_msg):
            self.my_subscription_callback(e2_agent_id, subscription_id, indication_hdr, indication_msg, kpm_report_style, ue_id)
        return _cb

    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            print(f"[DB] Failed to init {self.db_path}: {e}")
            return
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        
        # Tables creation
        cur.execute("CREATE TABLE IF NOT EXISTS cu_ue_meas (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, suci TEXT, ran INTEGER, amf INTEGER, ue_id INTEGER, du_id INTEGER, du_ue INTEGER, serv_rsrp REAL, serv_rsrq REAL, serv_sinr REAL, neigh1_pci INTEGER, neigh1_rsrp REAL, neigh1_rsrq REAL, neigh1_sinr REAL, neigh2_pci INTEGER, neigh2_rsrp REAL, neigh2_rsrq REAL, neigh2_sinr REAL)")
        cur.execute("CREATE TABLE IF NOT EXISTS du_ue_kpm (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, du_id TEXT, ue_id INTEGER, suci TEXT, thp_ul REAL, thp_dl REAL, rlc_delay_dl REAL, rlc_pkt_drop REAL, vol_dl REAL, vol_ul REAL, air_if_delay_ul REAL, rlc_delay_ul REAL, cqi REAL, rsrp REAL, rsrq REAL, prb_used_dl REAL, prb_used_ul REAL)")
        cur.execute("CREATE TABLE IF NOT EXISTS du_cell_kpm (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, du_id TEXT, prb_used_dl REAL, prb_used_ul REAL, prb_use_perc_dl REAL, prb_use_perc_ul REAL, thp_dl REAL, thp_ul REAL)")

        # Clear existing data to start fresh on each run
        cur.execute("DELETE FROM cu_ue_meas")
        cur.execute("DELETE FROM du_ue_kpm")
        cur.execute("DELETE FROM du_cell_kpm")
        
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cu_ts ON cu_ue_meas(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_du_ue_ts ON du_ue_kpm(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_du_cell_ts ON du_cell_kpm(ts)")
        
        conn.commit()
        conn.close()

    def _store_du_cell_kpm(self, du_id, ts, metrics):
        ts_val = self._normalize_float(ts) or (time.time() - self.start_time)
        du_id = self._normalize_du_id(du_id)
        prb_used_dl = self._normalize_float(metrics.get("RRU.PrbUsedDl"))
        prb_used_ul = self._normalize_float(metrics.get("RRU.PrbUsedUl"))
        values = (
            ts_val, du_id,
            prb_used_dl,
            prb_used_ul,
            self._normalize_float(metrics.get("RRU.PrbUsedDl")),
            self._normalize_float(metrics.get("RRU.PrbUsedUl")),
            self._normalize_float(metrics.get("DRB.UEThpDl")),
            self._normalize_float(metrics.get("DRB.UEThpUl")),
        )
        self.db_queue.put(("du_cell_kpm", values))

    def _lookup_suci_from_cu(self, du_id, du_ue):
        du_id_n = self._normalize_du_id(du_id)
        du_ue_n = self._normalize_ue_id(du_ue)
        return self.suci_cache.get((du_id_n, du_ue_n))

    def start_udp_listener(self, udp_ip="0.0.0.0", udp_port=5005):
        def udp_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((udp_ip, udp_port))
            print(f"[KPM][UDP] Listening on {udp_ip}:{udp_port}")
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    self._udp_print_cnt += 1
                    if self._udp_print_cnt % self._PRINT_EVERY_UDP == 0:
                        # print(f"[UDP] From {addr}: {msg}")
                        print(f"[UDP] From {addr}")
                    self._store_cu_udp(msg)
                except Exception as e:
                    print("[UDP] Error:", e)
        t = threading.Thread(target=udp_loop, daemon=True)
        t.start()

    def _store_cu_udp(self, msg):
        du_id_n = self._normalize_du_id(msg.get("du"))
        du_ue_n = self._normalize_ue_id(msg.get("du_ue"))
        suci_n = self._normalize_text(msg.get("suci"))
        if du_id_n and du_ue_n and suci_n:
            self.suci_cache[(du_id_n, du_ue_n)] = suci_n

        ts = time.time() - self.start_time
        values = (
            ts, msg.get("suci"), msg.get("ran"), msg.get("amf"), msg.get("ue"),
            msg.get("du"), msg.get("du_ue"), msg.get("serv_rsrp"), msg.get("serv_rsrq"), msg.get("serv_sinr"), 
            msg.get("neigh1_pci"), msg.get("neigh1_rsrp"), msg.get("neigh1_rsrq"), msg.get("neigh1_sinr"), 
            msg.get("neigh2_pci"), msg.get("neigh2_rsrp"), msg.get("neigh2_rsrq"), msg.get("neigh2_sinr"),
        )
        self.db_queue.put(("cu_ue_meas", values))

    def my_subscription_callback(self, e2_agent_id, subscription_id, indication_hdr, indication_msg, kpm_report_style, ue_id):
        self._kpm_print_cnt += 1
        do_print = (self._kpm_print_cnt % self._PRINT_EVERY_KPM == 0)

        if do_print:
            print(f"\n[KPM] From {e2_agent_id} | Sub {subscription_id} | Style {kpm_report_style}")

        indication_hdr = self.e2sm_kpm.extract_hdr_info(indication_hdr)
        meas_data = self.e2sm_kpm.extract_meas_data(indication_msg)
        timestamp = indication_hdr.get("colletStartTime", None)

        if kpm_report_style in [1, 2]:
            metrics_dict = meas_data["measData"]
            if kpm_report_style == 1:
                self._store_du_cell_kpm(e2_agent_id, timestamp, metrics_dict)
        else:
            if do_print:
                print("UE count:", len(meas_data.get("ueMeasData", {})))
            for current_ue_id, ue_meas_data in meas_data["ueMeasData"].items():
                metrics_dict = ue_meas_data["measData"]
                self._store_du_kpm(e2_agent_id, current_ue_id, e2_agent_id, timestamp, metrics_dict)

    def _normalize_ue_id(self, ue_id):
        if ue_id is None: return None
        if isinstance(ue_id, dict) and "ueId" in ue_id: return int(ue_id["ueId"])
        if isinstance(ue_id, str) and ue_id.isdigit(): return int(ue_id)
        try: return int(ue_id)
        except: return None

    def _normalize_text(self, v):
        return str(v) if v is not None else None

    def _normalize_du_id(self, v):
        if v is None: return None
        s = str(v)
        digits = [c for c in s if c.isdigit()]
        if not digits: return None
        return digits[-1]  # only keep the last digit to match our 5 DU setup

    def _unwrap_metric(self, v):
        if isinstance(v, (list, tuple)): return v[0] if len(v) > 0 else None
        return v

    def _normalize_float(self, v):
        if v is None: return 0.0  # None → 0.0
        v = self._unwrap_metric(v)
        try:
            result = float(v)
            if math.isnan(result):
                return 0.0  # NaN → 0.0
            return result
        except:
            return 0.0  # transfer false → 0.0

    def _store_du_kpm(self, du_id, ue_id, cell_id, ts, metrics):
        ts_val = self._normalize_float(ts) or (time.time() - self.start_time)
        du_id = self._normalize_du_id(du_id)
        ue_id = self._normalize_ue_id(ue_id)
        suci = self._lookup_suci_from_cu(du_id, ue_id)

        values = (
            ts_val, du_id, ue_id, suci,
            self._normalize_float(metrics.get("DRB.UEThpUl")),
            self._normalize_float(metrics.get("DRB.UEThpDl")),
            self._normalize_float(metrics.get("DRB.RlcSduDelayDl")),
            self._normalize_float(metrics.get("DRB.RlcPacketDropRateDl")),
            self._normalize_float(metrics.get("DRB.RlcSduTransmittedVolumeDL")),
            self._normalize_float(metrics.get("DRB.RlcSduTransmittedVolumeUL")),
            self._normalize_float(metrics.get("DRB.AirIfDelayUl")),
            self._normalize_float(metrics.get("DRB.RlcDelayUl")),
            self._normalize_float(metrics.get("CQI")),
            self._normalize_float(metrics.get("RSRP")),
            self._normalize_float(metrics.get("RSRQ")),
            self._normalize_float(metrics.get("RRU.PrbUsedDl")),
            self._normalize_float(metrics.get("RRU.PrbUsedUl")),
        )
        self.db_queue.put(("du_ue_kpm", values))

    @xAppBase.start_function
    def start(self, e2_node_id, kpm_report_style, ue_ids, metric_names):
        report_period = 480          # 120, 240, 480, 640, 1024, 2048
        granul_period = 480
        # NOTE: Reduced logic for brevity, matches user's intent
        if kpm_report_style == 1:
            print(f"[KPM] Subscribing to {e2_node_id}, Style 1")
            cb = self.make_kpm_callback(1)
            self.e2sm_kpm.subscribe_report_service_style_1(e2_node_id, report_period, metric_names, granul_period, cb)
        elif kpm_report_style == 4:
            print(f"[KPM] Subscribing to {e2_node_id}, Style 4")
            matchingUeConds = [{'testCondInfo': {'testType': ('ul-rSRP', 'true'), 'testExpr': 'lessthan', 'testValue': ('valueInt', 1000)}}]
            cb = self.make_kpm_callback(4)
            self.e2sm_kpm.subscribe_report_service_style_4(e2_node_id, report_period, matchingUeConds, metric_names, granul_period, cb)
        else:
            print(f"[KPM] Style {kpm_report_style} logic omitted in cleanup for brevity, add back if needed.")


def start_ho_process(config, http_port, rmr_port, rc_ran_func_id, ho_e2_node_id, ho_plmn, ho_gnb_cu_ue_f1ap_id):
    """
    Process entry point for the Handover xApp.
    It initializes its own RMR context via MyHOXapp class.
    """
    ho_xapp = MyHOXapp(
        config,
        http_port,
        rmr_port,
        ho_e2_node_id,
        ho_plmn,
        ho_gnb_cu_ue_f1ap_id,
    )
    ho_xapp.e2sm_rc.set_ran_func_id(rc_ran_func_id)
    ho_xapp.start(udp_ip="0.0.0.0", udp_port=6006)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrated KPM and HO xApp')
    parser.add_argument("--config", type=str, default='', help="xApp config file path")
    parser.add_argument("--http_server_port", type=int, default=8092, help="HTTP server listen port")
    parser.add_argument("--rmr_port", type=int, default=4562, help="RMR port")
    # Node IDs
    parser.add_argument("--e2_node_id_1", type=str, default='gnbd_001_001_00000213_1')
    parser.add_argument("--e2_node_id_2", type=str, default='gnbd_001_001_00000213_2')
    parser.add_argument("--e2_node_id_3", type=str, default='gnbd_001_001_00000213_3')
    parser.add_argument("--e2_node_id_4", type=str, default='gnbd_001_001_00000213_4')
    parser.add_argument("--e2_node_id_5", type=str, default='gnbd_001_001_00000213_5')
    # RAN Function IDs
    parser.add_argument("--ran_func_id", type=int, default=2, help="KPM RAN function ID (Default 2)")
    parser.add_argument("--rc_ran_func_id", type=int, default=3, help="RC RAN function ID (Default 3)")
    # Configs
    parser.add_argument("--kpm_report_style", type=int, default=4)
    parser.add_argument("--ue_ids", type=str, default='0')
    parser.add_argument("--metrics", type=str, default="DRB.UEThpUl,DRB.UEThpDl,DRB.RlcSduDelayDl,DRB.RlcPacketDropRateDl,DRB.RlcSduTransmittedVolumeDL,DRB.RlcSduTransmittedVolumeUL,DRB.AirIfDelayUl,DRB.RlcDelayUl,CQI,RSRQ,RSRQ,RRU.PrbUsedDl,RRU.PrbUsedUl")
    # HO Params
    parser.add_argument("--ho_e2_node_id", type=str, default='gnb_001_001_00000213')
    parser.add_argument("--ho_plmn", type=str, default='00101')
    parser.add_argument("--ho_gnb_cu_ue_f1ap_id", type=int, default=1)

    args = parser.parse_args()
    id1 = args.e2_node_id_1
    id2 = args.e2_node_id_2
    id3 = args.e2_node_id_3
    id4 = args.e2_node_id_4
    id5 = args.e2_node_id_5
    
    metrics = args.metrics.split(",")
    ue_ids = list(map(int, args.ue_ids.split(",")))
    
    # ************************************************************
    # Start HO xApp as a separate process
    print("[Main] Launching HO Process...")
    ho_proc = mp.Process(
        target=start_ho_process,
        args=(
            args.config,
            args.http_server_port - 2,   # Offset port
            args.rmr_port - 2,           # Offset port
            args.rc_ran_func_id,          # Pass RC ID (3)
            args.ho_e2_node_id,
            args.ho_plmn,
            args.ho_gnb_cu_ue_f1ap_id,
        ),
        daemon=False
    )
    ho_proc.start()
    
    # ************************************************************

    print(f"[Main] Initializing KPM xApp (Func ID: {args.ran_func_id})...")
    myXapp = MyXapp(args.config, args.http_server_port, args.rmr_port)
    myXapp.e2sm_kpm.set_ran_func_id(args.ran_func_id)

    myXapp.start_udp_listener(udp_ip="0.0.0.0", udp_port=5005)

    # Signal handling
    signal.signal(signal.SIGQUIT, myXapp.signal_handler)
    signal.signal(signal.SIGTERM, myXapp.signal_handler)
    signal.signal(signal.SIGINT, myXapp.signal_handler)

    print(f"[Main] Starting Cell Metrics (Style 1) for {id1}...")
    cell_metrics = [
        "RRU.PrbUsedDl",
        "RRU.PrbUsedUl",
        "RRU.PrbTotDl",
        "RRU.PrbTotUl",
        "DRB.UEThpDl",
        "DRB.UEThpUl",
    ]

    c1 = threading.Thread(target=myXapp.start, args=(id1, 1, [], cell_metrics))
    c1.start()
    # c2 = threading.Thread(target=myXapp.start, args=(id2, 1, [], cell_metrics))
    # c2.start()
    c4 = threading.Thread(target=myXapp.start, args=(id4, 1, [], cell_metrics))
    c4.start()
    # c5 = threading.Thread(target=myXapp.start, args=(id5, 1, [], cell_metrics))
    # c5.start()
    # c3 = threading.Thread(target=myXapp.start, args=(id3, 1, [], cell_metrics))
    # c3.start()
    # time.sleep(1)

    print(f"[Main] Starting Subscription Worker for")
    t1 = threading.Thread(target=myXapp.start,args=(id1, args.kpm_report_style, ue_ids, metrics))
    t1.start()
    # t2 = threading.Thread(target=myXapp.start,args=(id2, args.kpm_report_style, ue_ids, metrics))
    # t2.start()
    t4 = threading.Thread(target=myXapp.start,args=(id4, args.kpm_report_style, ue_ids, metrics))
    t4.start()
    # t5 = threading.Thread(target=myXapp.start,args=(id5, args.kpm_report_style, ue_ids, metrics))
    # t5.start()
    # t3 = threading.Thread(target=myXapp.start,args=(id3, args.kpm_report_style, ue_ids, metrics))
    # t3.start()
    # time.sleep(1)



    print("[Main] All systems running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
