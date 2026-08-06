import time
import sqlite3
import numpy as np
import gym
import json
import socket
from gym import spaces
from typing import Dict, List, Optional

from tgnmappo.tgn import TGNStreaming
from tgnmappo.tgn.utils import DU_OFFSET


class TGNUEEnv(gym.Env):
    """
    Multi-agent environment for TGN-MAPPO handover control.

    Supports two modes:
      - "online":  connects to live RAN, sends handover commands via UDP,
                   waits for new data, reward reflects real-time throughput.
      - "offline": replays historical data from the database only,
                   no UDP commands, no sleep, TGN processes DB window-by-window,
                   reward = throughput delta between steps.

    Each agent = one UE (identified by SUCI).

    Observations:
      - Actor obs:  TGN node embedding (per UE), shape (embedding_dim,)
      - Critic obs:  TGN graph embedding (shared), shape (embedding_dim,)

    Actions (MultiDiscrete([2, num_cells])):
      - dim 0: handover (0=stay, 1=handover)
      - dim 1: target cell (0..num_cells-1)

    Action masking (per-UE):
      - Based on Event A3: neighbor RSRP >= serving RSRP + offset
      - e.g. serving cell=1 -> can handover to 2,4 only
             serving cell=2 -> can handover to 1 only
    """

    def __init__(
        self,
        db_path: str,
        num_ue: int,
        num_cells: int = 3,
        embedding_dim: int = 32,
        control_period: float = 0.1,
        a3_offset: float = 3.0,
        udp_ip: str = "127.0.0.1",
        udp_port: int = 6006,
        target_cell_map: Optional[Dict[int, str]] = None,
        tgn: Optional[TGNStreaming] = None,
        tgn_checkpoint: Optional[str] = None,
        num_du_nodes: int = 16,
        time_window_s: float = 1.0,
        mode: str = "online",  # "online" or "offline"
    ):
        super().__init__()
        assert mode in ("online", "offline"), f"mode must be 'online' or 'offline', got '{mode}'"

        self.db_path = db_path
        self.num_ue = num_ue
        self.embedding_dim = embedding_dim
        self.num_cells = num_cells
        self.a3_offset = float(a3_offset)
        self.control_period = control_period
        self.udp_ip = udp_ip
        self.udp_port = int(udp_port)
        self.mode = mode
        # Action index (0,1,2) -> NR cell hex string for DU 2, 3, 5
        # _decode_actions returns tgt_idx = argmax+1 (1-based), so map keys are 1,2,3
        self.target_cell_map = target_cell_map or {
            1: "0x2132",  # action index 0 -> DU 2
            2: "0x2135",  # action index 1 -> DU 5
        }
        # DU ID -> action index (for action mask)
        self._du_id_to_action_idx = {2: 0, 5: 1}

        # ---- TGN (use external or create internal) ----
        if tgn is not None:
            self.tgn = tgn
            self._owns_tgn = False
        else:
            self.tgn = TGNStreaming(
                db_path=db_path,
                num_du_nodes=num_du_nodes,
                neighbor_size=10,
                time_window_s=time_window_s,
                checkpoint_path=tgn_checkpoint,
                train_mode=False,
            )
            self._owns_tgn = True

        # ---- DB ----
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if mode == "online" else None

        # ---- Spaces ----
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32)
            for _ in range(num_ue)
        ]
        # Critic input = concat(graph_emb, ue_node_emb), so 2x embedding_dim
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(embedding_dim * 2,), dtype=np.float32)
            for _ in range(num_ue)
        ]
        self.action_space = [
            spaces.MultiDiscrete([2, self.num_cells])
            for _ in range(num_ue)
        ]

        self.last_obs = None
        self._suci_list: List[str] = []
        self.graph_emb_last = np.zeros(self.embedding_dim, dtype=np.float32)
        self._available_actions = None

        # Offline mode: track throughput for delta reward
        self._prev_throughput = 0.0
        # Offline mode: count consecutive empty steps for episode boundary
        self._empty_steps = 0
        self._max_empty_steps = 10  # reset after this many empty TGN steps

    # ------------------------------------------------------------------
    # UE / DB helpers
    # ------------------------------------------------------------------

    def _refresh_suci_list(self) -> List[str]:
        cur = self.conn.cursor()
        # ORDER BY suci for stable agent-index assignment across resets
        cur.execute(
            "SELECT DISTINCT suci FROM cu_ue_meas WHERE suci IS NOT NULL ORDER BY suci ASC LIMIT ?",
            (self.num_ue,),
        )
        rows = cur.fetchall()
        suci_list = [r[0] for r in rows if r and r[0] is not None]
        if len(suci_list) < self.num_ue:
            suci_list += [""] * (self.num_ue - len(suci_list))
        self._suci_list = suci_list[: self.num_ue]
        return self._suci_list

    def _get_latest_ue_meas(self, suci: str) -> Optional[Dict[str, float]]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT serv_rsrp, neigh1_pci, neigh1_rsrp, neigh2_pci, neigh2_rsrp, du_id
               FROM cu_ue_meas WHERE suci = ? ORDER BY id DESC LIMIT 1""",
            (suci,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        serv_rsrp, n1_pci, n1_rsrp, n2_pci, n2_rsrp, du_id = row

        def _safe_int(v):
            if v is None or v == '':
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        def _safe_float(v):
            if v is None or v == '':
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        return {
            "serv_rsrp": _safe_float(serv_rsrp),
            "neigh1_pci": _safe_int(n1_pci),
            "neigh1_rsrp": _safe_float(n1_rsrp),
            "neigh2_pci": _safe_int(n2_pci),
            "neigh2_rsrp": _safe_float(n2_rsrp),
            "serv_du": _safe_int(du_id),
        }

    def _get_amf_id_by_suci(self, suci: str) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT amf FROM cu_ue_meas
               WHERE suci = ? AND amf IS NOT NULL ORDER BY id DESC LIMIT 1""",
            (suci,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return int(row[0])
        except Exception:
            return None

    def _get_ue_embedding(self, suci: str) -> np.ndarray:
        if not suci:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        emb = self.tgn.get_ue_embedding(suci)
        if emb is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return emb.numpy().astype(np.float32)

    def _get_total_throughput(self) -> float:
        try:
            cur = self.conn.cursor()
            placeholders = ",".join("?" for _ in self._suci_list if _)
            suci_list = [s for s in self._suci_list if s]
            if not suci_list:
                return 0.0
            cur.execute(
                f"""SELECT SUM(thp_dl) + SUM(thp_ul) FROM du_ue_kpm
                    WHERE suci IN ({placeholders})""",
                suci_list,
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                return 0.0
            return float(row[0])
        except Exception:
            return 0.0

    def _get_ue_latency(self, suci: str, latency_req_ms: float = 50.0) -> float:
        """Reward based on latency SLA violation.

        reward = -max(0, (dl + ul - latency_req) / latency_req)
        Within SLA -> 0; exceeding SLA -> negative proportional to violation.
        """
        if not suci:
            return 0.0
        try:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT rlc_delay_dl, rlc_delay_ul
                   FROM du_ue_kpm WHERE suci = ?
                   ORDER BY id DESC LIMIT 1""",
                (suci,),
            )
            row = cur.fetchone()
            if row is None:
                return 0.0
            dl = float(row[0]) if row[0] is not None else 0.0
            ul = float(row[1]) if row[1] is not None else 0.0
            violation = (dl + ul) - latency_req_ms
            return -(violation / latency_req_ms) if violation > 0 else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _build_action_mask(self) -> np.ndarray:
        """
        Per-UE mask: [handover_mask(2), target_cell_mask(num_cells)].

        Each UE has different available target cells:
          - service cell=1, can HO to 2,4 -> target_mask=[0,1,0,1]
          - service cell=2, can HO to 1   -> target_mask=[1,0,0,0]
        If no neighbor satisfies A3, handover action is masked out.
        """
        total_dim = 2 + self.num_cells
        masks = np.ones((self.num_ue, total_dim), dtype=np.float32)
        for i, suci in enumerate(self._suci_list):
            if not suci:
                continue
            meas = self._get_latest_ue_meas(suci)
            if meas is None or meas["serv_rsrp"] is None:
                continue

            serv_rsrp = meas["serv_rsrp"]
            target_mask = np.zeros(self.num_cells, dtype=np.float32)

            def _try_add(pci, rsrp):
                if pci is None or rsrp is None:
                    return False
                action_idx = self._du_id_to_action_idx.get(int(pci))
                if action_idx is not None and rsrp >= serv_rsrp + self.a3_offset:
                    target_mask[action_idx] = 1.0
                    return True
                return False

            _try_add(meas["neigh1_pci"], meas["neigh1_rsrp"])
            _try_add(meas["neigh2_pci"], meas["neigh2_rsrp"])

            ho_allowed = bool(target_mask.sum() > 0)
            handover_mask = np.array([1.0, 1.0 if ho_allowed else 0.0], dtype=np.float32)

            if not ho_allowed:
                serv_du = meas.get("serv_du")
                action_idx = self._du_id_to_action_idx.get(serv_du) if serv_du is not None else None
                if action_idx is not None:
                    target_mask[action_idx] = 1.0
                else:
                    target_mask[:] = 1.0

            masks[i, :] = np.concatenate([handover_mask, target_mask], axis=0)

        return masks

    def get_available_actions(self) -> np.ndarray:
        if not self._suci_list:
            self._refresh_suci_list()
        self._available_actions = self._build_action_mask()
        return np.expand_dims(self._available_actions, axis=0)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self):
        self._refresh_suci_list()

        if self.mode == "offline":
            # Offline: reset TGN cursor to beginning of data
            self.tgn.state.last_cu_id = 0
            self.tgn.state.last_du_ue_id = 0
            self.tgn.state.last_du_cell_id = 0
            self.tgn.state.last_timestamp = 0.0
            self._empty_steps = 0

        updated, graph_emb, has_updates = self.tgn.step()
        if hasattr(graph_emb, "numpy"):
            self.graph_emb_last = graph_emb.numpy().astype(np.float32)
        self.get_available_actions()

        obs = []
        for suci in self._suci_list:
            obs.append(self._get_ue_embedding(suci))

        self.last_obs = np.stack(obs, axis=0)
        self._prev_throughput = self._get_total_throughput()
        return np.expand_dims(self.last_obs, axis=0)

    def step(self, actions):
        decoded_actions = self._decode_actions(actions)

        # === Online mode: send handover commands + wait ===
        if self.mode == "online":
            ho_count = sum(1 for a in decoded_actions if a["handover"])
            print(f"[RL Action] {len(decoded_actions)} UEs, HO decisions: {ho_count} handover / {len(decoded_actions)-ho_count} stay")
            for i, act in enumerate(decoded_actions):
                suci = self._suci_list[i] if i < len(self._suci_list) else None
                print(f"  UE[{i}] suci={suci} -> handover={act['handover']} target={act['target_cell']}")
                if not act["handover"]:
                    continue
                if not suci:
                    continue
                amf_id = self._get_amf_id_by_suci(suci)
                if amf_id is None:
                    continue
                target = act["target_cell"]
                if target is None:
                    continue
                target = self.target_cell_map.get(int(target), target)
                self.send_action(amf_id, target)

            time.sleep(self.control_period)

        # === Both modes: advance TGN (read next time window from DB) ===
        # Online: if DB has no new data, wait until it does (avoid training on stale data)
        if self.mode == "online":
            while True:
                updated, graph_emb, has_updates = self.tgn.step()
                if has_updates:
                    break
                time.sleep(0.5)
        else:
            updated, graph_emb, has_updates = self.tgn.step()

        if hasattr(graph_emb, "numpy"):
            self.graph_emb_last = graph_emb.numpy().astype(np.float32)
        self.get_available_actions()

        # Build observations
        obs = []
        for suci in self._suci_list:
            obs.append(self._get_ue_embedding(suci))
        obs = np.stack(obs, axis=0)

        # === Reward: per-UE negative latency (dl + ul) ===
        # Each agent gets its own reward; lower latency -> higher reward.
        per_ue_rewards = np.array(
            [self._get_ue_latency(suci) for suci in self._suci_list],
            dtype=np.float32,
        ).reshape(self.num_ue, 1)
        rewards = per_ue_rewards

        # === Done ===
        dones = np.zeros(self.num_ue, dtype=bool)

        if self.mode == "offline":
            # In offline mode, episode ends when DB data is exhausted
            if not has_updates:
                self._empty_steps += 1
            else:
                self._empty_steps = 0
            if self._empty_steps >= self._max_empty_steps:
                dones[:] = True  # Signal episode end
            # Note: runner will call reset() which re-winds TGN cursor

        infos = {
            i: {"reward": float(per_ue_rewards[i, 0]), **decoded_actions[i]}
            for i in range(self.num_ue)
        }

        self.last_obs = obs
        obs = np.expand_dims(obs, axis=0)
        rewards = np.expand_dims(rewards, axis=0)
        dones = np.expand_dims(dones, axis=0)
        return obs, rewards, dones, infos

    def _decode_actions(self, actions):
        arr = np.asarray(actions)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[0] > 0:
            arr = arr[0]
        ho_logits = arr[:, :2]
        tgt_logits = arr[:, 2:2 + self.num_cells]
        ho_flag = np.argmax(ho_logits, axis=-1)
        tgt_idx = np.argmax(tgt_logits, axis=-1) + 1
        decoded = []
        for i in range(self.num_ue):
            decoded.append({
                "handover": bool(ho_flag[i] == 1),
                "target_cell": int(tgt_idx[i]) if ho_flag[i] == 1 else None,
            })
        return decoded

    def send_action(self, amf_ue_ngap_id, target_nr_cell_id):
        if self._udp_sock is None:
            return
        print(f"  [UDP] Sending HO: amf_id={amf_ue_ngap_id} -> target={target_nr_cell_id} to {self.udp_ip}:{self.udp_port}")
        msg = {
            "type": "HO_COMMAND",
            "commands": [{
                "amf_ue_ngap_id": int(amf_ue_ngap_id),
                "target_nr_cell_id": target_nr_cell_id,
            }],
        }
        try:
            self._udp_sock.sendto(
                json.dumps(msg).encode("utf-8"),
                (self.udp_ip, self.udp_port),
            )
        except Exception:
            pass

    def get_graph_emb(self) -> np.ndarray:
        return self.graph_emb_last

    def close(self):
        if self._owns_tgn:
            self.tgn.close()
        self.conn.close()
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
