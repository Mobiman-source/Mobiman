#!/usr/bin/env python3
import re
import csv
import time
import os
from threading import Thread
import json
import socket

# RSRP(dBm) = -140 + value

# SINR(dB) = -23 + 0.5×value

AMF_LOG = "/var/log/open5gs/amf.log"
CU_LOG = "/tmp/cu.log"

CU_LOOKBACK_LINES = 300

UDP_IP = "35.9.28.119"     # change if xApp runs elsewhere
UDP_PORT = 5005

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ue_map = {}

# --- SUCI normalization helper ---
def normalize_suci_last2(suci):
    """
    Keep only the last two digits of a SUCI string.
    Example: 'suci-0-001-01-0-0-0-0000000011' -> '11'
    """
    if suci is None:
        return None
    s = str(suci)
    digits = [c for c in s if c.isdigit()]
    if len(digits) >= 2:
        return ''.join(digits[-2:])
    elif len(digits) == 1:
        return digits[-1]
    return None


def init_csv():
    if not os.path.exists("ue_id_map.csv"):
        with open("ue_id_map.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "suci", "ran_ue_ngap_id", "amf_ue_ngap_id",
                "ue", "du", "du_ue",
                "serv_rsrp", "serv_rsrq", "serv_sinr",
                "neigh1_pci", "neigh1_rsrp", "neigh1_rsrq", "neigh1_sinr",
                "neigh2_pci", "neigh2_rsrp", "neigh2_rsrq", "neigh2_sinr"
            ])

def is_ue_state_ready(info):
    required = ["ue", "du", "du_ue"]
    for k in required:
        if not info.get(k):
            return False
    return True

def update_csv_and_print(suci):
    info = ue_map[suci]
    
    # Do NOT send or print if UE–DU binding is not ready
    if not is_ue_state_ready(info):
        return
    
    if not hasattr(update_csv_and_print, 'last_state'):
        update_csv_and_print.last_state = {}
    
    current_state = {
        "ran_ue_ngap_id": info.get("ran_ue_ngap_id", ""),
        "amf_ue_ngap_id": info.get("amf_ue_ngap_id", ""),
        "ue": info.get("ue", ""),
        "du": info.get("du", ""),
        "du_ue": info.get("du_ue", ""),
        "serv_rsrp": info.get("serv_rsrp", ""),
        "serv_rsrq": info.get("serv_rsrq", ""),
        "serv_sinr": info.get("serv_sinr", ""),
        "neigh1_pci": info.get("neigh1_pci", ""),
        "neigh1_rsrp": info.get("neigh1_rsrp", ""),
        "neigh1_rsrq": info.get("neigh1_rsrq", ""),
        "neigh1_sinr": info.get("neigh1_sinr", ""),
        "neigh2_pci": info.get("neigh2_pci", ""),
        "neigh2_rsrp": info.get("neigh2_rsrp", ""),
        "neigh2_rsrq": info.get("neigh2_rsrq", ""),
        "neigh2_sinr": info.get("neigh2_sinr", "")
    }
    
    if suci in update_csv_and_print.last_state:
        if update_csv_and_print.last_state[suci] == current_state:
            return
    
    update_csv_and_print.last_state[suci] = current_state.copy()

    send_udp = info.get("serv_rsrp") not in (None, "")
    if send_udp:
        payload = {
            "timestamp": time.time(),
            # "suci": suci,
            "suci": normalize_suci_last2(suci),
            "ran": info.get("ran_ue_ngap_id"),
            "amf": info.get("amf_ue_ngap_id"),
            "ue": info.get("ue"),
            "du": info.get("du"),
            "du_ue": info.get("du_ue"),

            "serv_rsrp": info.get("serv_rsrp"),
            "serv_rsrq": info.get("serv_rsrq"),
            "serv_sinr": info.get("serv_sinr"),

            "neigh1_pci": info.get("neigh1_pci"),
            "neigh1_rsrp": info.get("neigh1_rsrp"),
            "neigh1_rsrrq": info.get("neigh1_rsrq"),
            "neigh1_sinr": info.get("neigh1_sinr"),

            "neigh2_pci": info.get("neigh2_pci"),
            "neigh2_rsrp": info.get("neigh2_rsrp"),
            "neigh2_rsrq": info.get("neigh2_rsrq"),
            "neigh2_sinr": info.get("neigh2_sinr")
        }

        try:
            udp_sock.sendto(
                json.dumps(payload).encode("utf-8"),
                (UDP_IP, UDP_PORT)
            )
            print(f"[UDP-SEND] UE {payload['ue']} → DU {payload['du']}")
        except Exception as e:
            print("[UDP-SEND] Error:", e)

    print("\n+-------------------------------------------------------------------------------------------------------------+")
    print("| {:>3} | {:>3} | {:>3} | {:>3} | {:>2} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} |".format(
        "suci", "ran", "amf", "ue", "du", "du_ue", "s_rsrp", "s_rsrq", "s_sinr", 
        "n1_pci", "n1_rsrp", "n1_rsrq", "n1_sinr",
        "n2_pci", "n2_rsrp", "n2_rsrq", "n2_sinr"))
    print("+-------------------------------------------------------------------------------------------------------------+")
    for s, d in ue_map.items():
        print("| {:>3} | {:>3} | {:>3} | {:>3} | {:>2} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} | {:>3} |".format(
            # s,
            normalize_suci_last2(s),
            d.get("ran_ue_ngap_id", ""),
            d.get("amf_ue_ngap_id", ""),
            d.get("ue", ""),
            d.get("du", ""),
            d.get("du_ue", ""),
            d.get("serv_rsrp", ""),
            d.get("serv_rsrq", ""),
            d.get("serv_sinr", ""),
            d.get("neigh1_pci", ""),
            d.get("neigh1_rsrp", ""),
            d.get("neigh1_rsrq", ""),
            d.get("neigh1_sinr", ""),
            d.get("neigh2_pci", ""),
            d.get("neigh2_rsrp", ""),
            d.get("neigh2_rsrq", ""),
            d.get("neigh2_sinr", "")
        ))
    print("+-------------------------------------------------------------------------------------------------------------+\n")

re_suci = re.compile(r"\[suci-[^\]]+\]")

re_ran_amf = re.compile(r"RAN_UE_NGAP_ID\[(\d+)\]\s+AMF_UE_NGAP_ID\[(\d+)\]")

re_ngap = re.compile(r"\[NGAP\s+\].*?ue=(\d+)\s+ran_ue=(\d+)\s+amf_ue=(\d+)")

re_cu = re.compile(
    r"du=(\d+).*?ue=(\d+).*?cu_ue=(\d+).*?du_ue=(\d+)"
)

# re_handover = re.compile(r"ue=(\d+)[:\s].*handover.*from source_du=\d+ to target_du=(\d+)")  #this one is for CU "ho 1 4601 2"

re_handover = re.compile(r"handover.*ue=(\d+).*from source_du=\d+ to target_du=(\d+)")   #this is for xapp 
# docker compose xapp --amf_ue_ngap_id 618 --target_nr_cell_id 0x2131
re_handover_intra = re.compile(r"handover.*ue=(\d+).*on du=(\d+)")

#both: (?:ue=(\d+).*?handover.*?from source_du=\d+\s+to target_du=(\d+))|(?:handover.*?ue=(\d+).*?from source_du=\d+\s+to target_du=(\d+))

re_new_cu_ue = re.compile(r"ue=(\d+)[:\s].*du_index=(\d+): Created new CU-CP UE")
re_rrc_reest = re.compile(r'ue=(\d+).*"RRC Reestablishment Procedure".*old_ue=(\d+)\s+finalized')
re_meas_ue = re.compile(r'\[RRC\s+\].*ue=(\d+).*Containerized measurementReport')

# --- Semantic measurement regexes ---
re_serv_block = re.compile(r'"measResultServingCell"\s*:\s*\{')
re_neigh_block = re.compile(r'"measResultNeighCells"\s*:\s*\{')

re_pci = re.compile(r'"physCellId"\s*:\s*(\d+)')
re_rsrp = re.compile(r'"rsrp"\s*:\s*(\d+)')
re_rsrq = re.compile(r'"rsrq"\s*:\s*(\d+)')
re_sinr = re.compile(r'"sinr"\s*:\s*(\d+)')

def search_cu_backward_by_ue(ue_target):
    try:
        with open(CU_LOG, "r") as f:
            lines = f.readlines()
    except:
        return None

    for line in reversed(lines[-CU_LOOKBACK_LINES:]):
        m = re_cu.search(line)
        if m:
            du, ue, cu_ue, du_ue = m.groups()

            if ue == ue_target:
                return {
                    "du": du,
                    "du_ue": du_ue
                }

    return None


# =======================
# AMF log monitor
# =======================

def tail_amf_log():
    with open(AMF_LOG, "r") as f:
        f.seek(0, os.SEEK_END)
        last_ids = None  

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue

            m_ids = re_ran_amf.search(line)
            if m_ids:
                last_ids = {
                    "ran": m_ids.group(1),
                    "amf": m_ids.group(2)
                }

            m_suci = re_suci.search(line)
            if m_suci and last_ids:
                suci = m_suci.group(0)[1:-1]  # strip []

                if suci not in ue_map:
                    ue_map[suci] = {}

                ue_map[suci]["ran_ue_ngap_id"] = last_ids["ran"]
                ue_map[suci]["amf_ue_ngap_id"] = last_ids["amf"]
                # AMF stage only initializes identity; do NOT send UDP here

            
# =======================
# CU log monitor
# =======================

def find_suci_by_ran_amf(ran, amf):
    for suci, info in ue_map.items():
        if info.get("ran_ue_ngap_id") == ran and info.get("amf_ue_ngap_id") == amf:
            return suci
    return None


def tail_cu_log():
    with open(CU_LOG, "r") as f:
        f.seek(0, os.SEEK_END)
        active_ue_id = None
        last_release = None
        pending_handover = None
        # measurement state variables for semantic parsing
        serv_found = False
        neigh_found = 0
        serv_vals = {'rsrp': '0', 'rsrq': '0', 'sinr': '0'}
        serv_pci = None
        neigh_vals = []
        # --- Add state flags ---
        serv_done = False
        finalized = False
        in_neigh_block = False
        neigh_block_seen = False
        neigh_metrics_done = False
        neigh_brace_depth = 0

        def finalize_measurement(ue_id, clear_neighbors=False):
            for suci, info in ue_map.items():
                if info.get("ue") == ue_id:
                    if clear_neighbors:
                        info['neigh1_pci'] = ''
                        info['neigh1_rsrp'] = ''
                        info['neigh1_rsrq'] = ''
                        info['neigh1_sinr'] = ''
                        info['neigh2_pci'] = ''
                        info['neigh2_rsrp'] = ''
                        info['neigh2_rsrq'] = ''
                        info['neigh2_sinr'] = ''
                    update_csv_and_print(suci)
                    break

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue

            m_handover = re_handover.search(line) or re_handover_intra.search(line)
            if m_handover:
                print(f"[HANDOVER DETECTED] {line.strip()}")
                old_ue_val = m_handover.group(1)
                target_du = m_handover.group(2)
                print(f"[DEBUG] Handover detected for UE {old_ue_val} to DU {target_du}")

                for suci, info in ue_map.items():
                    if info.get("ue") == old_ue_val:
                        pending_handover = {
                            "suci": suci,
                            "old_ue": old_ue_val,
                            "target_du": target_du,
                        }
                        info["du"] = target_du
                        update_csv_and_print(suci)
                        break
                continue 

            m_new_cu_ue = re_new_cu_ue.search(line)
            if m_new_cu_ue and pending_handover:
                new_ue_val, du_index = m_new_cu_ue.groups()
                if du_index == pending_handover["target_du"]:
                    suci = pending_handover.get("suci")
                    if suci in ue_map:
                        ue_map[suci]["ue"] = new_ue_val
                        cu_info = search_cu_backward_by_ue(new_ue_val)
                        if cu_info:
                            ue_map[suci]["du_ue"] = cu_info["du_ue"]

                        print(f"[SUCCESS] Handover complete: {pending_handover['old_ue']} -> {new_ue_val}")
                        update_csv_and_print(suci)
                        pending_handover = None
                continue

            m_rrc_reest = re_rrc_reest.search(line)
            if m_rrc_reest:
                new_ue_val, old_ue_val = m_rrc_reest.groups()
                for suci, info in ue_map.items():
                    if info.get("ue") == old_ue_val:
                        info["ue"] = new_ue_val
                        print(f"[RRC-REEST] UE {old_ue_val} -> {new_ue_val}")
                        update_csv_and_print(suci)
                        break
                continue

            m_ngap = re_ngap.search(line)
            if m_ngap:
                ue_id, ran_id, amf_id = m_ngap.groups()
                is_release = "UEContextReleaseCommand" in line
                is_initial = "InitialContextSetupRequest" in line

                if is_release:
                    last_release = (ue_id, ran_id, amf_id)

                if is_initial and last_release:
                    rel_ue_id, rel_ran_id, rel_amf_id = last_release
                    for suci, info in ue_map.items():
                        if info.get("ue") == rel_ue_id or (
                            info.get("ran_ue_ngap_id") == rel_ran_id
                            and info.get("amf_ue_ngap_id") == rel_amf_id
                        ):
                            info["ue"] = ue_id
                            info["ran_ue_ngap_id"] = ran_id
                            info["amf_ue_ngap_id"] = amf_id
                            last_release = None
                            break

                # Prefer updating by UE id (CU log is authoritative for latest RAN/AMF IDs)
                updated = False
                for suci, info in ue_map.items():
                    if info.get("ue") == ue_id:
                        info["ran_ue_ngap_id"] = ran_id
                        info["amf_ue_ngap_id"] = amf_id
                        updated = True
                        break

                if not updated:
                    suci = find_suci_by_ran_amf(ran_id, amf_id)
                    if suci:
                        ue_map[suci]["ue"] = ue_id
                        # Refresh RAN/AMF IDs to match CU log
                        ue_map[suci]["ran_ue_ngap_id"] = ran_id
                        ue_map[suci]["amf_ue_ngap_id"] = amf_id
                        # UE identity known, but DU binding not ready yet
                continue

            # --- Measurement semantic block ---
            m_meas_start = re_meas_ue.search(line)
            if m_meas_start:
                if active_ue_id and serv_done and not finalized:
                    if neigh_metrics_done:
                        # deferred finalize for 1-neighbor case
                        finalize_measurement(active_ue_id)
                    elif not neigh_block_seen:
                        finalize_measurement(active_ue_id, clear_neighbors=True)

                active_ue_id = m_meas_start.group(1)

                # reset measurement state
                serv_found = False
                neigh_found = 0

                serv_vals = {'rsrp': '0', 'rsrq': '0', 'sinr': '0'}
                serv_pci = None
                neigh_vals = []
                neigh_metrics_done = False
                # reset state flags
                serv_done = False
                finalized = False
                in_neigh_block = False
                neigh_block_seen = False
                neigh_brace_depth = 0

                continue

            # ---- Serving cell ----
            if active_ue_id:
                # 1) detect entering serving block
                if not serv_found and re_serv_block.search(line):
                    serv_found = True
                    continue

                # 2) parse serving metrics after block start
                if serv_found and not serv_done:
                    m_pci = re_pci.search(line)
                    if m_pci and serv_pci is None:
                        serv_pci = m_pci.group(1)

                    m = re_rsrp.search(line)
                    if m:
                        serv_vals['rsrp'] = m.group(1)

                    m = re_rsrq.search(line)
                    if m:
                        serv_vals['rsrq'] = m.group(1)

                    m = re_sinr.search(line)
                    if m:
                        serv_vals['sinr'] = m.group(1)

                    # 3) once all three are available, write and stop serving parsing
                    if all(serv_vals[k] != '0' for k in ('rsrp', 'rsrq', 'sinr')):
                        for suci, info in ue_map.items():
                            if info.get("ue") == active_ue_id:
                                info['serv_rsrp'] = serv_vals['rsrp']
                                info['serv_rsrq'] = serv_vals['rsrq']
                                info['serv_sinr'] = serv_vals['sinr']
                                break
                        serv_done = True
                        # stop serving parsing to avoid overwriting with neighbor values
                        serv_found = False

            # ---- Neighbor cell parsing ----
            if active_ue_id and serv_done:
                if re_neigh_block.search(line):
                    in_neigh_block = True
                    neigh_block_seen = True
                    neigh_brace_depth = line.count('{') - line.count('}')
                    continue

                if in_neigh_block:
                    m_pci = re_pci.search(line)
                    if m_pci:
                        pci_val = m_pci.group(1)

                        # skip serving cell itself
                        if pci_val == serv_pci:
                            continue

                        if neigh_found < 2:
                            neigh_vals.append({'pci': pci_val, 'rsrp': '0', 'rsrq': '0', 'sinr': '0'})
                            neigh_found += 1
                            neigh_metrics_done = False  # reset for new neighbor
                            continue

                    if neigh_vals:
                        m = re_rsrp.search(line)
                        if m:
                            neigh_vals[-1]['rsrp'] = m.group(1)
                        m = re_rsrq.search(line)
                        if m:
                            neigh_vals[-1]['rsrq'] = m.group(1)
                        m = re_sinr.search(line)
                        if m:
                            neigh_vals[-1]['sinr'] = m.group(1)

                        # mark neighbor metrics done once all three are seen
                        if all(neigh_vals[-1][k] != '0' for k in ('rsrp', 'rsrq', 'sinr')):
                            neigh_metrics_done = True

                    neigh_brace_depth += line.count('{') - line.count('}')
                    if neigh_brace_depth <= 0:
                        in_neigh_block = False

            # ---- Finalize measurement ----
            # Finalize once neighbor block ends: write 0/1/2 neighbors exactly as parsed.
            if active_ue_id and serv_done and neigh_block_seen and not in_neigh_block and not finalized:
                for suci, info in ue_map.items():
                    if info.get("ue") == active_ue_id:
                        # Neighbor 1
                        n1 = neigh_vals[0] if len(neigh_vals) > 0 else {}
                        info['neigh1_pci']  = n1.get('pci', '')
                        info['neigh1_rsrp'] = n1.get('rsrp', '')
                        info['neigh1_rsrq'] = n1.get('rsrq', '')
                        info['neigh1_sinr'] = n1.get('sinr', '')

                        # Neighbor 2
                        n2 = neigh_vals[1] if len(neigh_vals) > 1 else {}
                        info['neigh2_pci']  = n2.get('pci', '')
                        info['neigh2_rsrp'] = n2.get('rsrp', '')
                        info['neigh2_rsrq'] = n2.get('rsrq', '')
                        info['neigh2_sinr'] = n2.get('sinr', '')

                        update_csv_and_print(suci)
                        break

                in_neigh_block = False
                finalized = True
                active_ue_id = None
                neigh_brace_depth = 0
                # do not continue; allow other parsing below as well

            m_cu = re_cu.search(line)
            if m_cu:
                du, ue_id, cu_ue, du_ue = m_cu.groups()
                for suci, info in ue_map.items():
                    if info.get("ue") == ue_id:
                        info["du"] = du
                        info["du_ue"] = du_ue
                        update_csv_and_print(suci)
                        break


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    print("Monitoring logs ...")
    print(f"Sending CU measurements via UDP → {UDP_IP}:{UDP_PORT}")

    t1 = Thread(target=tail_amf_log, daemon=True)
    t2 = Thread(target=tail_cu_log, daemon=True)
    t1.start()
    t2.start()

    while True:
        time.sleep(1)
