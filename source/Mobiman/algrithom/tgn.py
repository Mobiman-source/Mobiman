import os
import time
import sqlite3
from typing import Dict, List, Optional, Tuple
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tgnmappo.tgn.normalization import NormalizationConfig
from tgnmappo.tgn.utils import DU_OFFSET, SUCIMapper, StreamState, to_float
from tgnmappo.tgn.message_builder import (
    MSG_DIM,
    msg_from_cu_row,
    msg_from_du_ue_row,
    msg_from_du_cell_row,
    msg_from_neighbor_cell,
)
from tgnmappo.tgn.models import GraphAttentionEmbedding, LinkPredictor

class TGNStreaming:
    def __init__(
        self,
        db_path: str,
        num_du_nodes: int = 16,
        device: Optional[torch.device] = None,
        neighbor_size: int = 10,
        time_window_s: float = 1.0,  # Read data in time windows instead of fixed batch size
        dump_every_s: float = 0.0,
        checkpoint_path: Optional[str] = None,
        train_mode: bool = False,  # Enable training mode
        learning_rate: float = 0.0001,
        online_mode: bool = False,  # If True, train on all data (no time split)
    ):
        self.db_path = db_path
        self.device = device if device is not None else torch.device("cpu")
        self.time_window_s = float(time_window_s)
        self.dump_every_s = float(dump_every_s)
        self._last_dump_t = 0.0
        self.checkpoint_path = checkpoint_path
        self.train_mode = train_mode
        self._online_mode = online_mode

        # Normalization configuration
        self.norm_config = NormalizationConfig()
        requested_num_du_nodes = int(num_du_nodes)
        self.raw_du_ids = self._discover_raw_du_ids()
        if len(self.raw_du_ids) > requested_num_du_nodes:
            raise ValueError(
                f"Database contains {len(self.raw_du_ids)} distinct DU ids {self.raw_du_ids}, "
                f"but num_du_nodes={requested_num_du_nodes}."
            )
        self.du_id_to_slot = {du_id: idx for idx, du_id in enumerate(self.raw_du_ids)}
        self.slot_to_du_id = {idx: du_id for du_id, idx in self.du_id_to_slot.items()}
        self.num_du_nodes = len(self.raw_du_ids)
        self.num_nodes = DU_OFFSET + self.num_du_nodes + 1

        # SUCI mapper to avoid hash collisions
        self.suci_mapper = SUCIMapper()

        memory_dim = 16
        time_dim = 32
        embedding_dim = 32
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        self.memory = TGNMemory(
            self.num_nodes,
            MSG_DIM,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(MSG_DIM, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(self.device)

        if train_mode:
            self.memory.train()
        else:
            self.memory.eval()

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=MSG_DIM,
            time_enc=self.memory.time_enc,
        ).to(self.device)
        self.type_emb = torch.nn.Embedding(2, memory_dim).to(self.device)

        self.neighbor_loader = LastNeighborLoader(self.num_nodes, size=neighbor_size, device=self.device)
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=self.device)
        self.state = StreamState()
        # Keep a dedicated cursor for auxiliary TGN training batches so rollout
        # streaming state (self.state) is not consumed by train_step().
        self.train_state = StreamState()
        self._edge_t: List[torch.Tensor] = []
        self._edge_msg: List[torch.Tensor] = []

        self.dump_path = os.environ.get("EMBEDDING_OUT", "").strip() or None

        self.conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=5000")
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass

        # Offline training:
        self.train_test_split_ratio = 0.8
        self.train_time_max = None  # Will be computed on first use
        self.test_time_min = None
        self.test_time_max = None
        if train_mode:
            if online_mode:
                self.train_time_max = float('inf')
                self.test_time_min = float('inf')
                self.test_time_max = float('inf')
                print("[TGN] Online mode: training on all data (no time split)")
            else:
                self._compute_time_split()

        # Training 
        if train_mode:
            self.link_pred = LinkPredictor(in_channels=embedding_dim).to(self.device)
            self.optimizer = torch.optim.Adam(
                set(self.memory.parameters()) | set(self.gnn.parameters())
                | set(self.type_emb.parameters())
                | set(self.link_pred.parameters()),
                lr=learning_rate
            )
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.link_pred = None
            self.optimizer = None
            self.criterion = None

        # Load checkpoint if exists
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def reset(self):
        """Reset all state (use with caution - loses all learned embeddings)."""
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self.state = StreamState()
        self.train_state = StreamState()
        self.suci_mapper = SUCIMapper()
        self._edge_t = []
        self._edge_msg = []
        self._ensure_memory_store_device()

    def _ensure_memory_store_device(self):
        """Force TGNMemory message stores onto the configured device."""
        def _needs_move(obj):
            if torch.is_tensor(obj):
                return obj.device != self.device
            if isinstance(obj, (list, tuple)):
                return any(_needs_move(x) for x in obj)
            if isinstance(obj, dict):
                return any(_needs_move(v) for v in obj.values())
            return False

        def _move(obj):
            if torch.is_tensor(obj):
                return obj.to(self.device)
            if isinstance(obj, list):
                return [_move(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_move(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            return obj

        for attr in ("msg_s_store", "msg_d_store"):
            if hasattr(self.memory, attr):
                value = getattr(self.memory, attr)
                if _needs_move(value):
                    setattr(self.memory, attr, _move(value))

    def _discover_raw_du_ids(self) -> List[int]:
        conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        try:
            cur = conn.cursor()
            raw_ids = set()
            for table in ("cu_ue_meas", "du_ue_kpm", "du_cell_kpm"):
                cur.execute(f"SELECT DISTINCT du_id FROM {table} WHERE du_id IS NOT NULL")
                for (du_id,) in cur.fetchall():
                    try:
                        raw_ids.add(int(du_id))
                    except (TypeError, ValueError):
                        continue
            return sorted(raw_ids)
        finally:
            conn.close()

    def _map_du_id(self, du_id: int) -> Optional[int]:
        try:
            return self.du_id_to_slot.get(int(du_id))
        except (TypeError, ValueError):
            return None

    def _slot_to_node_id(self, du_slot: int) -> int:
        return DU_OFFSET + int(du_slot)

    def _valid_du_id(self, du_id: int) -> bool:
        return self._map_du_id(du_id) is not None

    def _compute_time_split(self):
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT MIN(min_ts), MAX(max_ts) FROM (
                    SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM cu_ue_meas WHERE ts IS NOT NULL
                    UNION ALL
                    SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM du_ue_kpm WHERE ts IS NOT NULL
                    UNION ALL
                    SELECT MIN(ts) as min_ts, MAX(ts) as max_ts FROM du_cell_kpm WHERE ts IS NOT NULL
                )
            """)
            row = cur.fetchone()
            if row and row[0] is not None and row[1] is not None:
                time_min = float(row[0])
                time_max = float(row[1])
                time_range = time_max - time_min

                # 80% for training, 20% for testing
                self.train_time_max = time_min + time_range * self.train_test_split_ratio
                self.test_time_min = self.train_time_max
                self.test_time_max = time_max

                print(f"[TGN] Data time range: {time_min:.2f} - {time_max:.2f}")
                print(f"[TGN] Train time range: {time_min:.2f} - {self.train_time_max:.2f}")
                print(f"[TGN] Test time range: {self.test_time_min:.2f} - {self.test_time_max:.2f}")
            else:
                print("[TGN] Warning: Could not compute time split, database may be empty")
                self.train_time_max = float('inf')
                self.test_time_min = float('inf')
                self.test_time_max = float('inf')
        except Exception as e:
            print(f"[TGN] Error computing time split: {e}")
            self.train_time_max = float('inf')
            self.test_time_min = float('inf')
            self.test_time_max = float('inf')

    def _fetch_cu_rows(self, time_window_end: float) -> List[Tuple]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, ts, suci, du_id,
                   serv_rsrp, serv_rsrq, serv_sinr,
                   neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                   neigh2_pci, neigh2_rsrp, neigh2_rsrq
            FROM cu_ue_meas
            WHERE id > ? AND ts <= ? AND suci IS NOT NULL AND du_id IS NOT NULL
            ORDER BY ts ASC, id ASC
            """,
            (self.state.last_cu_id, time_window_end),
        )
        return cur.fetchall()

    def _fetch_du_ue_rows(self, time_window_end: float) -> List[Tuple]:
        """Fetch DU UE rows within time window."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, ts, du_id, ue_id, suci,
                   thp_ul, thp_dl,
                   rlc_delay_dl, rlc_delay_ul,
                   rlc_pkt_drop,
                   prb_used_dl, prb_used_ul,
                   vol_dl, vol_ul,
                   air_if_delay_ul,
                   cqi, rsrp, rsrq
            FROM du_ue_kpm
            WHERE id > ? AND ts <= ? AND du_id IS NOT NULL AND suci IS NOT NULL
            ORDER BY ts ASC, id ASC
            """,
            (self.state.last_du_ue_id, time_window_end),
        )
        return cur.fetchall()

    def _fetch_du_cell_rows(self, time_window_end: float) -> List[Tuple]:
        """Fetch DU Cell rows within time window."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, ts, du_id,
                   prb_used_dl, prb_used_ul,
                   prb_use_perc_dl, prb_use_perc_ul,
                   thp_dl
            FROM du_cell_kpm
            WHERE id > ? AND ts <= ? AND du_id IS NOT NULL
            ORDER BY ts ASC, id ASC
            """,
            (self.state.last_du_cell_id, time_window_end),
        )
        return cur.fetchall()

    def _get_current_max_timestamp(self) -> float:
        """Get the maximum timestamp across all tables."""
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT MAX(max_ts) FROM (
                    SELECT MAX(ts) as max_ts FROM cu_ue_meas
                    UNION ALL
                    SELECT MAX(ts) as max_ts FROM du_ue_kpm
                    UNION ALL
                    SELECT MAX(ts) as max_ts FROM du_cell_kpm
                )
                """
            )
            row = cur.fetchone()
            return float(row[0]) if row and row[0] is not None else self.state.last_timestamp
        except Exception:
            return self.state.last_timestamp

    def _rows_to_events(self, cu_rows, du_ue_rows, du_cell_rows):
        src: List[int] = []
        dst: List[int] = []
        ts: List[float] = []
        msg: List[List[float]] = []

        updated_max: Dict[str, int] = {}

        # CU: UE <-> DU (bidirectional edges for better information flow)
        for (
            rid, t, suci, du_id,
            serv_rsrp, serv_rsrq, serv_sinr,
            neigh1_pci, neigh1_rsrp, neigh1_rsrq,
            neigh2_pci, neigh2_rsrp, neigh2_rsrq,
        ) in cu_rows:
            try:
                ue_node = self.suci_mapper.get_node_id(str(suci))
            except ValueError:
                continue  

            du_slot = self._map_du_id(du_id)
            if du_slot is None:
                continue
            du_node = self._slot_to_node_id(du_slot)
            msg_vec = msg_from_cu_row(
                serv_rsrp, serv_rsrq, serv_sinr,
                neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                neigh2_pci, neigh2_rsrp, neigh2_rsrq,
                self.norm_config,
            )
            timestamp = to_float(t)

            # 1. Serving cell: UE <-> DU (bidirectional, strong connection)
            # UE -> DU
            src.append(int(ue_node))
            dst.append(int(du_node))
            ts.append(timestamp)
            msg.append(msg_vec)

            # DU -> UE (bidirectional)
            src.append(int(du_node))
            dst.append(int(ue_node))
            ts.append(timestamp)
            msg.append(msg_vec)

            # 2. Neighbor cells: UE -> neighbor DU (unidirectional, potential handover)
            # Neighbor 1
            if neigh1_pci is not None and neigh1_rsrp is not None:
                try:
                    neigh1_du_slot = self._map_du_id(int(to_float(neigh1_pci)))
                    if neigh1_du_slot is not None:
                        neigh1_du_node = self._slot_to_node_id(neigh1_du_slot)
                        neigh1_msg = msg_from_neighbor_cell(
                            neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                            self.norm_config,
                        )
                        src.append(int(ue_node))
                        dst.append(int(neigh1_du_node))
                        ts.append(timestamp)
                        msg.append(neigh1_msg)
                except (ValueError, TypeError):
                    pass  

            # Neighbor 2
            if neigh2_pci is not None and neigh2_rsrp is not None:
                try:
                    neigh2_du_slot = self._map_du_id(int(to_float(neigh2_pci)))
                    if neigh2_du_slot is not None:
                        neigh2_du_node = self._slot_to_node_id(neigh2_du_slot)
                        neigh2_msg = msg_from_neighbor_cell(
                            neigh2_pci, neigh2_rsrp, neigh2_rsrq,
                            self.norm_config,
                        )
                        src.append(int(ue_node))
                        dst.append(int(neigh2_du_node))
                        ts.append(timestamp)
                        msg.append(neigh2_msg)
                except (ValueError, TypeError):
                    pass 
            updated_max["cu"] = int(rid)

        # DU UE KPM: UE -> DU (encode UE<->DU performance on the serving DU)
        for (
            rid, t, du_id, ue_id, suci,
            thp_ul, thp_dl,
            rlc_delay_dl, rlc_delay_ul,
            rlc_pkt_drop,
            prb_used_dl, prb_used_ul,
            vol_dl, vol_ul,
            air_if_delay_ul,
            cqi, rsrp, rsrq,
        ) in du_ue_rows:
            if not self._valid_du_id(du_id):
                continue
            try:
                ue_node = self.suci_mapper.get_node_id(str(suci))
            except ValueError:
                continue  
            du_slot = self._map_du_id(du_id)
            if du_slot is None:
                continue
            du_node = self._slot_to_node_id(du_slot)
            msg_vec = msg_from_du_ue_row(
                thp_ul, thp_dl,
                rlc_delay_dl, rlc_delay_ul,
                rlc_pkt_drop,
                prb_used_dl, prb_used_ul,
                vol_dl, vol_ul,
                air_if_delay_ul,
                cqi, rsrp, rsrq,
                self.norm_config,
            )
            timestamp = to_float(t)
            
            # UE -> DU (DU's measurement of this UE on this DU)
            src.append(int(ue_node))
            dst.append(int(du_node))
            ts.append(timestamp)
            msg.append(msg_vec)

            updated_max["du_ue"] = int(rid)

        # DU CELL KPM: DU -> DU (self-loop)
        for (
            rid, t, du_id,
            prb_used_dl, prb_used_ul,
            prb_use_perc_dl, prb_use_perc_ul,
            thp_dl,
        ) in du_cell_rows:
            if not self._valid_du_id(du_id):
                continue
            du_slot = self._map_du_id(du_id)
            if du_slot is None:
                continue
            du_node = self._slot_to_node_id(du_slot)
            src.append(int(du_node))
            dst.append(int(du_node))
            ts.append(to_float(t))
            msg.append(
                msg_from_du_cell_row(
                    prb_used_dl, prb_used_ul,
                    prb_use_perc_dl, prb_use_perc_ul,
                    thp_dl,
                    self.norm_config,
                )
            )
            updated_max["du_cell"] = int(rid)

        if not src:
            return None, updated_max

        # Sort by timestamp to maintain temporal order
        events = sorted(zip(src, dst, ts, msg), key=lambda x: x[2])
        src, dst, ts, msg = zip(*events) if events else ([], [], [], [])
        
        src_t = torch.tensor(src, dtype=torch.long, device=self.device)
        dst_t = torch.tensor(dst, dtype=torch.long, device=self.device)
        # Use long dtype for timestamps as required by TGNMemory
        t_t = torch.tensor([int(t * 1000) for t in ts], dtype=torch.long, device=self.device)  # Convert to milliseconds
        msg_t = torch.tensor(msg, dtype=torch.float, device=self.device)
        
        return (src_t, dst_t, t_t, msg_t), updated_max

    def _insert_edges(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, msg: torch.Tensor):
        """Insert edges into neighbor loader and store their timestamps/messages."""
        for i in range(t.size(0)):
            self._edge_t.append(t[i].detach())
            self._edge_msg.append(msg[i].detach())
        self.neighbor_loader.insert(src, dst)

    def _build_graph_inputs(
        self,
        n_id: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build edge_index, edge timestamps, and edge messages including history."""
        n_id, edge_index_hist, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        edge_t_hist = None
        edge_msg_hist = None
        if e_id is not None and e_id.numel() > 0:
            e_id_list = e_id.tolist()
            edge_t_hist = torch.stack([self._edge_t[i] for i in e_id_list], dim=0).to(self.device)
            edge_msg_hist = torch.stack([self._edge_msg[i] for i in e_id_list], dim=0).to(self.device)

        src_local = self.assoc[src]
        dst_local = self.assoc[dst]
        edge_index_cur = torch.stack([src_local, dst_local], dim=0)

        if edge_t_hist is None:
            edge_index = edge_index_cur
            edge_t = t
            edge_msg = msg
        else:
            edge_index = torch.cat([edge_index_hist, edge_index_cur], dim=1)
            edge_t = torch.cat([edge_t_hist, t], dim=0)
            edge_msg = torch.cat([edge_msg_hist, msg], dim=0)

        return n_id, edge_index, edge_t, edge_msg


    @torch.no_grad()
    def step(self) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, bool]:
        """ updated: Dict mapping node_id to updated embedding (CPU)
            graph_emb: Graph-level embedding (CPU)
            has_updates: Whether any updates occurred"""
        self.gnn.eval()
        self._ensure_memory_store_device()

        # 1) Determine time window
        current_max_ts = self._get_current_max_timestamp()
        if current_max_ts <= self.state.last_timestamp:
            # No new data
            return {}, torch.zeros(self.memory.memory_dim), False
        
        time_window_end = min(
            self.state.last_timestamp + self.time_window_s,
            current_max_ts
        )

        # 2) Fetch new rows within time window
        cu_rows = self._fetch_cu_rows(time_window_end)
        du_ue_rows = self._fetch_du_ue_rows(time_window_end)
        du_cell_rows = self._fetch_du_cell_rows(time_window_end)

        # 3) Build events
        result = self._rows_to_events(cu_rows, du_ue_rows, du_cell_rows)
        if result[0] is None:
            # No new events in this window
            self.state.last_timestamp = time_window_end
            return {}, self._compute_stable_graph_embedding(), False

        (src, dst, t, msg), updated_max = result

        # 4) Get neighbor subgraph for involved nodes
        n_id = torch.unique(torch.cat([src, dst], dim=0))
        n_id, edge_index, edge_t, edge_msg = self._build_graph_inputs(n_id, src, dst, t, msg)

        # 5) Get current memory state BEFORE update
        z_memory, last_update = self.memory(n_id)

        # 6) update memory before computing GNN embeddings
        self.memory.update_state(src, dst, t, msg)
        self._insert_edges(src, dst, t, msg)

        # 7) Get UPDATED memory state
        z_updated, last_update_new = self.memory(n_id)
        z_updated = self._add_type_embedding(n_id, z_updated)

        # 8) Run GNN to get enriched embeddings
        # Map global to local indices
        src_local = self.assoc[src]
        dst_local = self.assoc[dst]
        z_gnn = self.gnn(
            z_updated,
            last_update_new,
            edge_index,
            edge_t.float(),  # GNN expects float timestamps
            edge_msg,
        )

        # 9) Detach memory to prevent backprop issues
        self.memory.detach()

        # 10) Build output dictionary with GNN embeddings (not just memory!)
        updated = {int(n.item()): z_gnn[i].detach().cpu() for i, n in enumerate(n_id)}

        # 11) Compute stable graph embedding (all DUs, not just updated ones)
        graph_emb = self._compute_stable_graph_embedding()

        # 12) Update cursors
        if "cu" in updated_max:
            self.state.last_cu_id = max(self.state.last_cu_id, updated_max["cu"])
        if "du_ue" in updated_max:
            self.state.last_du_ue_id = max(self.state.last_du_ue_id, updated_max["du_ue"])
        if "du_cell" in updated_max:
            self.state.last_du_cell_id = max(self.state.last_du_cell_id, updated_max["du_cell"])
        
        self.state.last_timestamp = time_window_end

        # 13) Maybe dump embeddings
        self._maybe_dump(updated, graph_emb)

        return updated, graph_emb, True

    def _compute_stable_graph_embedding(self) -> torch.Tensor:
        """Compute graph-level embedding as mean of all DU embeddings."""
        # Get all DU node IDs
        all_du_ids = torch.arange(
            DU_OFFSET, 
            DU_OFFSET + self.num_du_nodes,
            dtype=torch.long,
            device=self.device
        )
        
        # Get their current embeddings
        try:
            z_all_du, _ = self.memory(all_du_ids)
            z_all_du = self._add_type_embedding(all_du_ids, z_all_du)
            graph_emb = z_all_du.mean(dim=0).detach().cpu()
        except Exception as e:
            print(f"[TGN] Error computing graph embedding: {e}")
            graph_emb = torch.zeros(self.memory.memory_dim)
        
        return graph_emb

    def _compute_stable_graph_embedding_trainable(self) -> torch.Tensor:
        """Trainable graph-level embedding on current device (no detach)."""
        all_du_ids = torch.arange(
            DU_OFFSET,
            DU_OFFSET + self.num_du_nodes,
            dtype=torch.long,
            device=self.device
        )
        z_all_du, _ = self.memory(all_du_ids)
        z_all_du = self._add_type_embedding(all_du_ids, z_all_du)
        return z_all_du.mean(dim=0)

    def _add_type_embedding(self, n_id: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Add explicit node-type embedding (UE vs DU) to memory states."""
        type_ids = (n_id >= DU_OFFSET).long()
        return z + self.type_emb(type_ids)

    def get_node_embedding(self, node_id: int) -> torch.Tensor:
        """Get current embedding for a specific node."""
        try:
            node_tensor = torch.tensor([node_id], dtype=torch.long, device=self.device)
            z, _ = self.memory(node_tensor)
            z = self._add_type_embedding(node_tensor, z)
            return z[0].detach().cpu()
        except Exception:
            return torch.zeros(self.memory.memory_dim)

    def get_node_embedding_trainable(self, node_id: int) -> torch.Tensor:
        """Get node embedding on current device without detach for end-to-end training."""
        node_tensor = torch.tensor([node_id], dtype=torch.long, device=self.device)
        z, _ = self.memory(node_tensor)
        z = self._add_type_embedding(node_tensor, z)
        return z[0]
    
    def get_ue_embedding(self, suci: str) -> Optional[torch.Tensor]:
        node_id = self.suci_mapper.suci_to_id.get(suci)
        if node_id is None:
            return None
        return self.get_node_embedding(node_id)

    def get_ue_embedding_trainable(self, suci: str) -> Optional[torch.Tensor]:
        node_id = self.suci_mapper.suci_to_id.get(suci)
        if node_id is None:
            return None
        return self.get_node_embedding_trainable(node_id)
    
    def get_du_embedding(self, du_id: int) -> torch.Tensor:
        du_slot = self._map_du_id(du_id)
        if du_slot is None:
            return torch.zeros(self.memory.memory_dim)
        node_id = self._slot_to_node_id(du_slot)
        return self.get_node_embedding(node_id)

    def step_trainable(self, detach_memory: bool = False) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, bool]:
        """
        Trainable variant of step() for end-to-end RL.
        Returns device tensors without detach so task losses can backprop to TGN.
        """
        self.gnn.train()
        self.memory.train()
        self._ensure_memory_store_device()

        current_max_ts = self._get_current_max_timestamp()
        if current_max_ts <= self.state.last_timestamp:
            return {}, torch.zeros(self.memory.memory_dim, device=self.device), False

        time_window_end = min(
            self.state.last_timestamp + self.time_window_s,
            current_max_ts
        )

        cu_rows = self._fetch_cu_rows(time_window_end)
        du_ue_rows = self._fetch_du_ue_rows(time_window_end)
        du_cell_rows = self._fetch_du_cell_rows(time_window_end)

        result = self._rows_to_events(cu_rows, du_ue_rows, du_cell_rows)
        if result[0] is None:
            self.state.last_timestamp = time_window_end
            return {}, self._compute_stable_graph_embedding_trainable(), False

        (src, dst, t, msg), updated_max = result

        n_id = torch.unique(torch.cat([src, dst], dim=0))
        n_id, edge_index, edge_t, edge_msg = self._build_graph_inputs(n_id, src, dst, t, msg)

        self.memory(n_id)  # initialize/update internal caches as in step()
        self.memory.update_state(src, dst, t, msg)
        self._insert_edges(src, dst, t, msg)

        z_updated, last_update_new = self.memory(n_id)
        z_updated = self._add_type_embedding(n_id, z_updated)
        z_gnn = self.gnn(
            z_updated,
            last_update_new,
            edge_index,
            edge_t.float(),
            edge_msg,
        )

        if detach_memory:
            self.memory.detach()

        updated = {int(n.item()): z_gnn[i] for i, n in enumerate(n_id)}
        graph_emb = self._compute_stable_graph_embedding_trainable()

        if "cu" in updated_max:
            self.state.last_cu_id = max(self.state.last_cu_id, updated_max["cu"])
        if "du_ue" in updated_max:
            self.state.last_du_ue_id = max(self.state.last_du_ue_id, updated_max["du_ue"])
        if "du_cell" in updated_max:
            self.state.last_du_cell_id = max(self.state.last_du_cell_id, updated_max["du_cell"])
        self.state.last_timestamp = time_window_end

        return updated, graph_emb, True

    def _maybe_dump(self, updated: Dict[int, torch.Tensor], graph_emb: torch.Tensor):
        """Periodically dump embeddings to disk."""
        if not self.dump_path:
            return
        if self.dump_every_s <= 0:
            return
        now = time.time()
        if now - self._last_dump_t < self.dump_every_s:
            return
        self._last_dump_t = now
        
        # Get all DU embeddings
        all_du_embs = {}
        for du_slot in range(self.num_du_nodes):
            node_id = self._slot_to_node_id(du_slot)
            all_du_embs[self.slot_to_du_id[du_slot]] = self.get_node_embedding(node_id)
        
        payload = {
            "ts": now,
            "num_nodes": int(self.num_nodes),
            "updated_embeddings": updated,
            "all_du_embeddings": all_du_embs,
            "graph_embedding": graph_emb,
            "cursor": {
                "cu": int(self.state.last_cu_id),
                "du_ue": int(self.state.last_du_ue_id),
                "du_cell": int(self.state.last_du_cell_id),
                "timestamp": float(self.state.last_timestamp),
            },
            "suci_mapping": self.suci_mapper.suci_to_id,
        }
        try:
            torch.save(payload, self.dump_path)
        except Exception as e:
            print(f"[TGN] Error dumping embeddings: {e}")

    # --- Training methods ---

    def _build_training_batch(self, batch_size: int = 200) -> Optional[Tuple]:
        """ Build a training batch from database with positive and negative samples."""
        if not self.train_mode:
            raise RuntimeError("Training batch construction requires train_mode=True")
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT id, ts, suci, du_id,
                       serv_rsrp, serv_rsrq, serv_sinr,
                       neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                       neigh2_pci, neigh2_rsrp, neigh2_rsrq
                FROM cu_ue_meas
                WHERE id > ? AND suci IS NOT NULL AND du_id IS NOT NULL
                  AND ts IS NOT NULL AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (self.train_state.last_cu_id, self.train_time_max, max(1, batch_size // 2)),
            )
            cu_rows = cur.fetchall()
            
            cur.execute(
                """
                SELECT id, ts, du_id, ue_id, suci,
                       thp_ul, thp_dl,
                       rlc_delay_dl, rlc_delay_ul,
                       rlc_pkt_drop,
                       prb_used_dl, prb_used_ul,
                       vol_dl, vol_ul,
                       air_if_delay_ul,
                       cqi, rsrp, rsrq
                FROM du_ue_kpm
                WHERE id > ? AND ts IS NOT NULL AND du_id IS NOT NULL AND suci IS NOT NULL
                  AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (self.train_state.last_du_ue_id, self.train_time_max, max(1, batch_size // 4)),
            )
            du_ue_rows = cur.fetchall()
            # Fetch DU cell KPM rows (DU -> DU self-loop, training time range)
            cur.execute(
                """
                SELECT id, ts, du_id,
                       prb_used_dl, prb_used_ul,
                       prb_use_perc_dl, prb_use_perc_ul,
                       thp_dl
                FROM du_cell_kpm
                WHERE id > ? AND ts IS NOT NULL AND du_id IS NOT NULL
                  AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (self.train_state.last_du_cell_id, self.train_time_max, max(1, batch_size // 4)),
            )
            du_cell_rows = cur.fetchall()

            if not cu_rows and not du_ue_rows and not du_cell_rows:
                return None

            src_list, dst_list, ts_list, msg_list, label_list = [], [], [], [], []

            # Process positive samples
            for (rid, t, suci, du_id,
                 serv_rsrp, serv_rsrq, serv_sinr,
                 neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                 neigh2_pci, neigh2_rsrp, neigh2_rsrq) in cu_rows:
                try:
                    ue_node = self.suci_mapper.get_node_id(str(suci))
                except ValueError:
                    continue
                if not self._valid_du_id(du_id):
                    continue

                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_cu_row(
                    serv_rsrp, serv_rsrq, serv_sinr,
                    neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                    neigh2_pci, neigh2_rsrp, neigh2_rsrq,
                    self.norm_config,
                )

                # Positive sample: UE -> DU (actual connection)
                src_list.append(ue_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample: UE -> random other DU
                neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                if neg_du_slot == du_slot and self.num_du_nodes > 1:
                    neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                neg_du_node = self._slot_to_node_id(neg_du_slot)

                src_list.append(ue_node)
                dst_list.append(neg_du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(0.0)

            # DU UE KPM samples: UE -> DU (positive) + UE -> random DU (negative)
            for (
                rid, t, du_id, _ue_id, suci,
                thp_ul, thp_dl,
                rlc_delay_dl, rlc_delay_ul,
                rlc_pkt_drop,
                prb_used_dl, prb_used_ul,
                vol_dl, vol_ul,
                air_if_delay_ul,
                cqi, rsrp, rsrq,
            ) in du_ue_rows:
                if not self._valid_du_id(du_id):
                    continue
                try:
                    ue_node = self.suci_mapper.get_node_id(str(suci))
                except ValueError:
                    continue

                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_du_ue_row(
                    thp_ul, thp_dl,
                    rlc_delay_dl, rlc_delay_ul,
                    rlc_pkt_drop,
                    prb_used_dl, prb_used_ul,
                    vol_dl, vol_ul,
                    air_if_delay_ul,
                    cqi, rsrp, rsrq,
                    self.norm_config,
                )

                # Positive sample: UE -> DU (actual connection)
                src_list.append(ue_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample: UE -> random other DU
                if self.num_du_nodes > 1:
                    neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                    if neg_du_slot == du_slot:
                        neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                    neg_du_node = self._slot_to_node_id(neg_du_slot)
                    src_list.append(ue_node)
                    dst_list.append(neg_du_node)
                    ts_list.append(to_float(t))
                    msg_list.append(msg_vec)
                    label_list.append(0.0)

            # DU cell KPM: DU -> DU self-loop (positive) + DU -> random DU (negative)
            for (
                rid, t, du_id,
                prb_used_dl, prb_used_ul,
                prb_use_perc_dl, prb_use_perc_ul,
                thp_dl,
            ) in du_cell_rows:
                if not self._valid_du_id(du_id):
                    continue
                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_du_cell_row(
                    prb_used_dl, prb_used_ul,
                    prb_use_perc_dl, prb_use_perc_ul,
                    thp_dl,
                    self.norm_config,
                )

                # Positive sample: DU -> DU (self-loop)
                src_list.append(du_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample: DU -> random other DU
                if self.num_du_nodes > 1:
                    neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                    if neg_du_slot == du_slot:
                        neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                    neg_du_node = self._slot_to_node_id(neg_du_slot)
                    src_list.append(du_node)
                    dst_list.append(neg_du_node)
                    ts_list.append(to_float(t))
                    msg_list.append(msg_vec)
                    label_list.append(0.0)

            if not src_list:
                return None

            # Convert to tensors
            src_t = torch.tensor(src_list, dtype=torch.long, device=self.device)
            dst_t = torch.tensor(dst_list, dtype=torch.long, device=self.device)
            t_t = torch.tensor([int(t * 1000) for t in ts_list], dtype=torch.long, device=self.device)
            msg_t = torch.tensor(msg_list, dtype=torch.float, device=self.device)
            label_t = torch.tensor(label_list, dtype=torch.float, device=self.device)

            # Update cursor
            if cu_rows:
                self.train_state.last_cu_id = max(self.train_state.last_cu_id, cu_rows[-1][0])
            if du_ue_rows:
                self.train_state.last_du_ue_id = max(self.train_state.last_du_ue_id, du_ue_rows[-1][0])
            if du_cell_rows:
                self.train_state.last_du_cell_id = max(self.train_state.last_du_cell_id, du_cell_rows[-1][0])

            return src_t, dst_t, t_t, msg_t, label_t

        except Exception as e:
            print(f"[TGN] Error building training batch: {e}")
            return None

    def train_step(self) -> Optional[float]:
        """
        Returns:
            loss value or None if no data available
        """
        if not self.train_mode:
            raise RuntimeError("Training requires train_mode=True")

        self.memory.train()
        self.gnn.train()
        self.link_pred.train()
        self._ensure_memory_store_device()

        # Build training batch
        batch_data = self._build_training_batch()
        if batch_data is None:
            return None

        src, dst, t, msg, labels = batch_data

        self.optimizer.zero_grad()

        # Get neighbor subgraph
        n_id = torch.unique(torch.cat([src, dst], dim=0))
        n_id_full, edge_index, edge_t, edge_msg = self._build_graph_inputs(n_id, src, dst, t, msg)

        # Get memory and run GNN
        z, last_update = self.memory(n_id_full)
        z = self._add_type_embedding(n_id_full, z)

        # Build edge index from current batch (not neighbor_loader)
        src_local = self.assoc[src]
        dst_local = self.assoc[dst]
        z = self.gnn(z, last_update, edge_index, edge_t.float(), edge_msg)

        # Compute predictions
        src_emb = z[self.assoc[src]]
        dst_emb = z[self.assoc[dst]]
        pred = self.link_pred(src_emb, dst_emb).squeeze()

        # Compute loss
        loss = self.criterion(pred, labels)

        # Update memory with ground-truth interactions (only positive samples)
        pos_mask = labels == 1.0
        if pos_mask.sum() > 0:
            self.memory.update_state(
                src[pos_mask],
                dst[pos_mask],
                t[pos_mask],
                msg[pos_mask]
            )
            self._insert_edges(src[pos_mask], dst[pos_mask], t[pos_mask], msg[pos_mask])

        # Backprop
        loss.backward()
        self.optimizer.step()
        self.memory.detach()

        return float(loss)

    def _build_test_batch(self, batch_size: int = 200, test_cursor_state: Optional[Dict] = None) -> Optional[Tuple]:
        """
        Build a test batch from database with positive and negative samples.
        Uses test time range (the last 20% of data).

        Args:
            batch_size: Size of the batch
            test_cursor_state: Dict with test cursors {'cu_id', 'du_ue_id', 'du_cell_id'}

        Returns:
            (src, dst, t, msg, labels, updated_cursors) or None if no data available
        """
        if not self.train_mode:
            raise RuntimeError("Test batch construction requires train_mode=True")

        # Use separate cursors for test data
        if test_cursor_state is None:
            test_cu_id = 0
            test_du_ue_id = 0
            test_du_cell_id = 0
        else:
            test_cu_id = test_cursor_state.get('cu_id', 0)
            test_du_ue_id = test_cursor_state.get('du_ue_id', 0)
            test_du_cell_id = test_cursor_state.get('du_cell_id', 0)

        try:
            cur = self.conn.cursor()

            # Fetch test samples from cu_ue_meas (test time range only)
            cur.execute(
                """
                SELECT id, ts, suci, du_id,
                       serv_rsrp, serv_rsrq, serv_sinr,
                       neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                       neigh2_pci, neigh2_rsrp, neigh2_rsrq
                FROM cu_ue_meas
                WHERE id > ? AND suci IS NOT NULL AND du_id IS NOT NULL
                  AND ts IS NOT NULL AND ts >= ? AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (test_cu_id, self.test_time_min, self.test_time_max, max(1, batch_size // 2)),
            )
            cu_rows = cur.fetchall()

            # Fetch test samples from du_ue_kpm
            cur.execute(
                """
                SELECT id, ts, du_id, ue_id, suci,
                       thp_ul, thp_dl,
                       rlc_delay_dl, rlc_delay_ul,
                       rlc_pkt_drop,
                       prb_used_dl, prb_used_ul,
                       vol_dl, vol_ul,
                       air_if_delay_ul,
                       cqi, rsrp, rsrq
                FROM du_ue_kpm
                WHERE id > ? AND ts IS NOT NULL AND du_id IS NOT NULL AND suci IS NOT NULL
                  AND ts >= ? AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (test_du_ue_id, self.test_time_min, self.test_time_max, max(1, batch_size // 4)),
            )
            du_ue_rows = cur.fetchall()

            # Fetch test samples from du_cell_kpm
            cur.execute(
                """
                SELECT id, ts, du_id,
                       prb_used_dl, prb_used_ul,
                       prb_use_perc_dl, prb_use_perc_ul,
                       thp_dl
                FROM du_cell_kpm
                WHERE id > ? AND ts IS NOT NULL AND du_id IS NOT NULL
                  AND ts >= ? AND ts <= ?
                ORDER BY ts ASC, id ASC
                LIMIT ?
                """,
                (test_du_cell_id, self.test_time_min, self.test_time_max, max(1, batch_size // 4)),
            )
            du_cell_rows = cur.fetchall()

            if not cu_rows and not du_ue_rows and not du_cell_rows:
                return None

            src_list, dst_list, ts_list, msg_list, label_list = [], [], [], [], []

            # Process positive samples (same logic as training)
            for (rid, t, suci, du_id,
                 serv_rsrp, serv_rsrq, serv_sinr,
                 neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                 neigh2_pci, neigh2_rsrp, neigh2_rsrq) in cu_rows:
                try:
                    ue_node = self.suci_mapper.get_node_id(str(suci))
                except ValueError:
                    continue
                if not self._valid_du_id(du_id):
                    continue

                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_cu_row(
                    serv_rsrp, serv_rsrq, serv_sinr,
                    neigh1_pci, neigh1_rsrp, neigh1_rsrq,
                    neigh2_pci, neigh2_rsrp, neigh2_rsrq,
                    self.norm_config,
                )

                # Positive sample
                src_list.append(ue_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample
                neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                if neg_du_slot == du_slot and self.num_du_nodes > 1:
                    neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                neg_du_node = self._slot_to_node_id(neg_du_slot)

                src_list.append(ue_node)
                dst_list.append(neg_du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(0.0)

                test_cu_id = max(test_cu_id, int(rid))

            # Process du_ue_kpm samples
            for (rid, t, du_id, _ue_id, suci,
                 thp_ul, thp_dl, rlc_delay_dl, rlc_delay_ul,
                 rlc_pkt_drop, prb_used_dl, prb_used_ul,
                 vol_dl, vol_ul, air_if_delay_ul, cqi, rsrp, rsrq) in du_ue_rows:
                if not self._valid_du_id(du_id):
                    continue
                try:
                    ue_node = self.suci_mapper.get_node_id(str(suci))
                except ValueError:
                    continue

                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_du_ue_row(
                    thp_ul, thp_dl, rlc_delay_dl, rlc_delay_ul,
                    rlc_pkt_drop, prb_used_dl, prb_used_ul,
                    vol_dl, vol_ul, air_if_delay_ul, cqi, rsrp, rsrq,
                    self.norm_config,
                )

                # Positive sample
                src_list.append(ue_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample
                if self.num_du_nodes > 1:
                    neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                    if neg_du_slot == du_slot:
                        neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                    neg_du_node = self._slot_to_node_id(neg_du_slot)
                    src_list.append(ue_node)
                    dst_list.append(neg_du_node)
                    ts_list.append(to_float(t))
                    msg_list.append(msg_vec)
                    label_list.append(0.0)

                test_du_ue_id = max(test_du_ue_id, int(rid))

            # Process du_cell_kpm samples
            for (rid, t, du_id, prb_used_dl, prb_used_ul,
                 prb_use_perc_dl, prb_use_perc_ul, thp_dl) in du_cell_rows:
                if not self._valid_du_id(du_id):
                    continue
                du_slot = self._map_du_id(du_id)
                if du_slot is None:
                    continue
                du_node = self._slot_to_node_id(du_slot)
                msg_vec = msg_from_du_cell_row(
                    prb_used_dl, prb_used_ul,
                    prb_use_perc_dl, prb_use_perc_ul,
                    thp_dl,
                    self.norm_config,
                )

                # Positive sample
                src_list.append(du_node)
                dst_list.append(du_node)
                ts_list.append(to_float(t))
                msg_list.append(msg_vec)
                label_list.append(1.0)

                # Negative sample
                if self.num_du_nodes > 1:
                    neg_du_slot = torch.randint(0, self.num_du_nodes, (1,)).item()
                    if neg_du_slot == du_slot:
                        neg_du_slot = (neg_du_slot + 1) % self.num_du_nodes
                    neg_du_node = self._slot_to_node_id(neg_du_slot)
                    src_list.append(du_node)
                    dst_list.append(neg_du_node)
                    ts_list.append(to_float(t))
                    msg_list.append(msg_vec)
                    label_list.append(0.0)

                test_du_cell_id = max(test_du_cell_id, int(rid))

            if not src_list:
                return None

            # Convert to tensors
            src_t = torch.tensor(src_list, dtype=torch.long, device=self.device)
            dst_t = torch.tensor(dst_list, dtype=torch.long, device=self.device)
            t_t = torch.tensor([int(t * 1000) for t in ts_list], dtype=torch.long, device=self.device)
            msg_t = torch.tensor(msg_list, dtype=torch.float, device=self.device)
            label_t = torch.tensor(label_list, dtype=torch.float, device=self.device)

            # Return updated cursors
            updated_cursors = {
                'cu_id': test_cu_id,
                'du_ue_id': test_du_ue_id,
                'du_cell_id': test_du_cell_id
            }

            return src_t, dst_t, t_t, msg_t, label_t, updated_cursors

        except Exception as e:
            print(f"[TGN] Error building test batch: {e}")
            return None

    @torch.no_grad()
    def evaluate(self, num_batches: int = 10) -> Tuple[float, float]:
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()
        self._ensure_memory_store_device()
        all_preds, all_labels = [], []
        test_cursor_state = {'cu_id': 0, 'du_ue_id': 0, 'du_cell_id': 0}
        for _ in range(num_batches):
            batch_result = self._build_test_batch(batch_size=100, test_cursor_state=test_cursor_state)
            if batch_result is None:
                break
            src, dst, t, msg, labels, test_cursor_state = batch_result
            # Get neighbor subgraph
            n_id = torch.unique(torch.cat([src, dst], dim=0))
            n_id_full, edge_index, edge_t, edge_msg = self._build_graph_inputs(n_id, src, dst, t, msg)
            # Get memory and run GNN
            z, last_update = self.memory(n_id_full)
            z = self._add_type_embedding(n_id_full, z)

            # Build edge index from current batch (not neighbor_loader)
            src_local = self.assoc[src]
            dst_local = self.assoc[dst]
            z = self.gnn(z, last_update, edge_index, edge_t.float(), edge_msg)

            src_emb = z[self.assoc[src]]
            dst_emb = z[self.assoc[dst]]
            pred = self.link_pred(src_emb, dst_emb).squeeze().sigmoid()

            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

            # Note: We do NOT update memory during evaluation to avoid leaking test data？？？

        if not all_preds:
            return 0, 0

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        try:
            ap = average_precision_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            ap, auc = 0, 0

        return float(ap), float(auc)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save complete state for recovery."""
        save_path = path or self.checkpoint_path
        if not save_path:
            return
        
        try:
            checkpoint = {
                "memory_state": self.memory.state_dict(),
                "gnn_state": self.gnn.state_dict(),
                "type_emb_state": self.type_emb.state_dict(),
                "neighbor_loader_state": {
                    "last_neighbor_id": self.neighbor_loader.__dict__.get("last_neighbor_id", {}),
                },
                "stream_state": {
                    "last_cu_id": self.state.last_cu_id,
                    "last_du_ue_id": self.state.last_du_ue_id,
                    "last_du_cell_id": self.state.last_du_cell_id,
                    "last_timestamp": self.state.last_timestamp,
                },
                "train_state": {
                    "last_cu_id": self.train_state.last_cu_id,
                    "last_du_ue_id": self.train_state.last_du_ue_id,
                    "last_du_cell_id": self.train_state.last_du_cell_id,
                    "last_timestamp": self.train_state.last_timestamp,
                },
                "suci_mapper": {
                    "suci_to_id": self.suci_mapper.suci_to_id,
                    "id_to_suci": self.suci_mapper.id_to_suci,
                    "next_id": self.suci_mapper.next_id,
                },
            }
            if self.train_mode and self.link_pred is not None:
                checkpoint["link_pred_state"] = self.link_pred.state_dict()
                checkpoint["optimizer_state"] = self.optimizer.state_dict()

            torch.save(checkpoint, save_path)
            print(f"[TGN] Checkpoint saved to {save_path}")
        except Exception as e:
            print(f"[TGN] Error saving checkpoint: {e}")

    def load_checkpoint(self, path: Optional[str] = None):
        """Load complete state from checkpoint."""
        load_path = path or self.checkpoint_path
        if not load_path or not os.path.exists(load_path):
            return
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            
            self.memory.load_state_dict(checkpoint["memory_state"])
            self.gnn.load_state_dict(checkpoint["gnn_state"])
            if "type_emb_state" in checkpoint:
                self.type_emb.load_state_dict(checkpoint["type_emb_state"])

            # Restore stream state
            self.state.last_cu_id = checkpoint["stream_state"]["last_cu_id"]
            self.state.last_du_ue_id = checkpoint["stream_state"]["last_du_ue_id"]
            self.state.last_du_cell_id = checkpoint["stream_state"]["last_du_cell_id"]
            self.state.last_timestamp = checkpoint["stream_state"]["last_timestamp"]
            if "train_state" in checkpoint:
                self.train_state.last_cu_id = checkpoint["train_state"].get("last_cu_id", 0)
                self.train_state.last_du_ue_id = checkpoint["train_state"].get("last_du_ue_id", 0)
                self.train_state.last_du_cell_id = checkpoint["train_state"].get("last_du_cell_id", 0)
                self.train_state.last_timestamp = checkpoint["train_state"].get("last_timestamp", 0.0)

            # Restore SUCI mapper
            self.suci_mapper.suci_to_id = checkpoint["suci_mapper"]["suci_to_id"]
            self.suci_mapper.id_to_suci = checkpoint["suci_mapper"]["id_to_suci"]
            self.suci_mapper.next_id = checkpoint["suci_mapper"]["next_id"]

            # Load training components if available
            if self.train_mode and "link_pred_state" in checkpoint:
                if self.link_pred is not None:
                    try:
                        self.link_pred.load_state_dict(checkpoint["link_pred_state"])
                        print("[TGN] Loaded link_pred state from checkpoint")
                    except Exception as e:
                        print(f"[TGN] Warning: Could not load link_pred state: {e}")
                if self.optimizer is not None and "optimizer_state" in checkpoint:
                    print("[TGN] Note: Optimizer state exists in checkpoint but not loaded (starting fresh)")

            print(f"[TGN] Checkpoint loaded from {load_path}")
        except Exception as e:
            print(f"[TGN] Error loading checkpoint: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="Mode: train or inference")
    parser.add_argument("--db", type=str, default="/path/to/your/oran-sc-ric/data/ran.db",
                        help="Database path")
    parser.add_argument("--num_du", type=int, default=3, help="Number of DU nodes")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--checkpoint", type=str, default="/tmp/tgn_checkpoint.pt",
                        help="Checkpoint path")
    parser.add_argument(
        "--load-weights-only",
        action="store_true",
        default=False,
        help="Load model weights from checkpoint but reset runtime memory, cursors, and SUCI mapping",
    )
    args = parser.parse_args()
    DB = os.environ.get("RAN_DB", args.db)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode == "train":
        print(f"[TGN] Training mode on device: {device}")
        tgn = TGNStreaming(
            db_path=DB,
            num_du_nodes=args.num_du,
            device=device,
            neighbor_size=10,
            checkpoint_path=args.checkpoint,
            train_mode=True,
            learning_rate=0.0001,
        )
        if args.load_weights_only:
            tgn.reset()
            print("[TGN] Reset runtime state after checkpoint load; keeping only learned weights")
        print("[TGN] Starting offline training (train/test split: 80/20 by time)...")
        try:
            for epoch in range(1, args.epochs + 1):
                tgn.state.last_cu_id = 0
                tgn.state.last_du_ue_id = 0
                tgn.state.last_du_cell_id = 0
                tgn.train_state.last_cu_id = 0
                tgn.train_state.last_du_ue_id = 0
                tgn.train_state.last_du_cell_id = 0
                epoch_losses = []
                batch_idx = 0
                while True:
                    loss = tgn.train_step()
                    if loss is None:
                        break
                    epoch_losses.append(loss)
                    batch_idx += 1
                    if batch_idx % 50 == 0:
                        recent_loss = sum(epoch_losses[-50:]) / min(50, len(epoch_losses[-50:]))
                        print(f"  Epoch {epoch:02d}, Batch {batch_idx}, Recent loss: {recent_loss:.4f}")
                if epoch_losses:
                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    print(f"Epoch {epoch:02d} completed: {len(epoch_losses)} batches, Avg loss: {avg_loss:.4f}")

                    if epoch % 5 == 0:
                        print(f"  Evaluating on test set...")
                        ap, auc = tgn.evaluate()
                        print(f"  Test AP: {ap:.4f}, Test AUC: {auc:.4f}")
                    if epoch % 10 == 0:
                        tgn.save_checkpoint()
                else:
                    print(f"Epoch {epoch:02d}, No training data available")
        except KeyboardInterrupt:
            print("\n[TGN] Training interrupted by user")
        finally:
            tgn.save_checkpoint()
            tgn.close()
            print("[TGN] Training stopped and checkpoint saved")

    else:  # inference mode
        print(f"[TGN] Inference mode on device: {device}")
        tgn = TGNStreaming(
            db_path=DB,
            num_du_nodes=args.num_du,
            device=device,
            neighbor_size=10,
            time_window_s=0.5,     
            dump_every_s=2.0,      
            checkpoint_path=args.checkpoint,
            train_mode=False,
        )
        if args.load_weights_only:
            tgn.reset()
            print("[TGN] Reset runtime state after checkpoint load; keeping only learned weights")
        print("[TGN] Streaming started. Reading new rows and updating embeddings...")
        try:
            step_count = 0
            while True:
                updated, graph_emb, has_updates = tgn.step()
                if has_updates:
                    step_count += 1
                    print(f"[TGN] Step {step_count}: updated_nodes={len(updated)}, "
                          f"graph_emb_norm={graph_emb.norm().item():.4f}")
                    if step_count % 100 == 0:
                        tgn.save_checkpoint()
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[TGN] Interrupted by user")
        finally:
            tgn.save_checkpoint()
            tgn.close()
            print("[TGN] Stopped and checkpoint saved")
