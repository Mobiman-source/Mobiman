from dataclasses import dataclass
from typing import Dict, Optional

DU_OFFSET = 100000


class SUCIMapper:
    """Maps SUCI (subscriber identifiers) to unique node IDs."""

    def __init__(self):
        self.suci_to_id: Dict[str, int] = {}
        self.id_to_suci: Dict[int, str] = {}
        self.next_id = 0

    def get_node_id(self, suci: str) -> int:
        """Get or create a unique node ID for a SUCI."""
        if suci not in self.suci_to_id:
            if self.next_id >= DU_OFFSET:
                raise ValueError(f"UE node ID space exhausted (>={DU_OFFSET})")
            self.suci_to_id[suci] = self.next_id
            self.id_to_suci[self.next_id] = suci
            self.next_id += 1
        return self.suci_to_id[suci]

    def get_suci(self, node_id: int) -> Optional[str]:
        """Get SUCI from node ID."""
        return self.id_to_suci.get(node_id)


def suci_to_node_id(suci: Optional[str], mapper: SUCIMapper) -> Optional[int]:
    """Public helper to map SUCI -> UE node-id."""
    if suci is None:
        return None
    return mapper.get_node_id(suci)


def to_float(x, default: float = 0.0) -> float:
    """Convert value to float with fallback default."""
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class StreamState:
    """Track streaming state for incremental data processing."""
    last_cu_id: int = 0
    last_du_ue_id: int = 0
    last_du_cell_id: int = 0
    last_timestamp: float = 0.0
