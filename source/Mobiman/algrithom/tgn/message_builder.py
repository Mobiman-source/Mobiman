from typing import List, Optional

from .normalization import NormalizationConfig, normalize
from .utils import to_float

# --- Message schema -------
# Separate message dimensions for different event types to avoid sparse vectors

CU_MSG_DIM = 9  # serving(3) + neigh1(3) + neigh2(3)
DU_UE_MSG_DIM = 13  # thp_ul, thp_dl, rlc_delay_dl, rlc_delay_ul, rlc_pkt_drop, prb_used_dl, prb_used_ul, vol_dl, vol_ul, air_if_delay_ul, cqi, rsrp, rsrq
DU_CELL_MSG_DIM = 5  # prb_used_dl, prb_used_ul, prb_use_perc_dl, prb_use_perc_ul, thp_dl
MSG_DIM = 3 + max(CU_MSG_DIM, DU_UE_MSG_DIM, DU_CELL_MSG_DIM)


def msg_from_cu_row(
    serv_rsrp, serv_rsrq, serv_sinr,
    neigh1_pci, neigh1_rsrp, neigh1_rsrq,
    neigh2_pci, neigh2_rsrp, neigh2_rsrq,
    norm_config: Optional[NormalizationConfig] = None,
) -> List[float]:
    """Create message vector from CU UE measurement."""
    m = [0.0] * MSG_DIM
    # Event type one-hot: [1, 0, 0] for CU
    m[0] = 1.0

    if norm_config is not None:
        # Serving cell (normalized)
        m[3] = normalize(to_float(serv_rsrp), norm_config.rsrp_min, norm_config.rsrp_max)
        m[4] = normalize(to_float(serv_rsrq), norm_config.rsrq_min, norm_config.rsrq_max)
        m[5] = normalize(to_float(serv_sinr), norm_config.sinr_min, norm_config.sinr_max)

        # Neighbor 1
        m[6] = normalize(to_float(neigh1_pci), norm_config.pci_min, norm_config.pci_max)
        m[7] = normalize(to_float(neigh1_rsrp), norm_config.rsrp_min, norm_config.rsrp_max)
        m[8] = normalize(to_float(neigh1_rsrq), norm_config.rsrq_min, norm_config.rsrq_max)

        # Neighbor 2
        m[9] = normalize(to_float(neigh2_pci), norm_config.pci_min, norm_config.pci_max)
        m[10] = normalize(to_float(neigh2_rsrp), norm_config.rsrp_min, norm_config.rsrp_max)
        m[11] = normalize(to_float(neigh2_rsrq), norm_config.rsrq_min, norm_config.rsrq_max)
    else:
        # No normalization (legacy behavior)
        m[3] = to_float(serv_rsrp)
        m[4] = to_float(serv_rsrq)
        m[5] = to_float(serv_sinr)

        # Neighbor 1
        m[6] = to_float(neigh1_pci)
        m[7] = to_float(neigh1_rsrp)
        m[8] = to_float(neigh1_rsrq)

        # Neighbor 2
        m[9] = to_float(neigh2_pci)
        m[10] = to_float(neigh2_rsrp)
        m[11] = to_float(neigh2_rsrq)

    return m


def msg_from_du_ue_row(
    thp_ul, thp_dl,
    rlc_delay_dl, rlc_delay_ul,
    rlc_pkt_drop,
    prb_used_dl, prb_used_ul,
    vol_dl, vol_ul,
    air_if_delay_ul,
    cqi, rsrp, rsrq,
    norm_config: Optional[NormalizationConfig] = None,
) -> List[float]:
    """Create message vector from DU UE KPM."""
    m = [0.0] * MSG_DIM
    # Event type one-hot: [0, 1, 0] for DU UE
    m[1] = 1.0

    if norm_config is not None:
        m[3] = normalize(to_float(thp_ul), norm_config.thp_min, norm_config.thp_max)
        m[4] = normalize(to_float(thp_dl), norm_config.thp_min, norm_config.thp_max)
        m[5] = normalize(to_float(rlc_delay_dl), norm_config.rlc_delay_min, norm_config.rlc_delay_max)
        m[6] = normalize(to_float(rlc_delay_ul), norm_config.rlc_delay_min, norm_config.rlc_delay_max)
        m[7] = normalize(to_float(rlc_pkt_drop), norm_config.rlc_pkt_drop_min, norm_config.rlc_pkt_drop_max)
        m[8] = normalize(to_float(prb_used_dl), norm_config.prb_used_min, norm_config.prb_used_max)
        m[9] = normalize(to_float(prb_used_ul), norm_config.prb_used_min, norm_config.prb_used_max)
        m[10] = normalize(to_float(vol_dl), norm_config.vol_min, norm_config.vol_max)
        m[11] = normalize(to_float(vol_ul), norm_config.vol_min, norm_config.vol_max)
        m[12] = normalize(to_float(air_if_delay_ul), norm_config.air_if_delay_min, norm_config.air_if_delay_max)
        m[13] = normalize(to_float(cqi), norm_config.cqi_min, norm_config.cqi_max)
        m[14] = normalize(to_float(rsrp), norm_config.rsrp_min, norm_config.rsrp_max)
        m[15] = normalize(to_float(rsrq), norm_config.rsrq_min, norm_config.rsrq_max)
    else:
        # No normalization (legacy behavior)
        m[3] = to_float(thp_ul)
        m[4] = to_float(thp_dl)
        m[5] = to_float(rlc_delay_dl)
        m[6] = to_float(rlc_delay_ul)
        m[7] = to_float(rlc_pkt_drop)
        m[8] = to_float(prb_used_dl)
        m[9] = to_float(prb_used_ul)
        m[10] = to_float(vol_dl)
        m[11] = to_float(vol_ul)
        m[12] = to_float(air_if_delay_ul)
        m[13] = to_float(cqi)
        m[14] = to_float(rsrp)
        m[15] = to_float(rsrq)

    return m


def msg_from_du_cell_row(
    prb_used_dl, prb_used_ul,
    prb_use_perc_dl, prb_use_perc_ul,
    thp_dl,
    norm_config: Optional[NormalizationConfig] = None,
) -> List[float]:
    """Create message vector from DU Cell KPM."""
    m = [0.0] * MSG_DIM
    # Event type one-hot: [0, 0, 1] for DU Cell
    m[2] = 1.0

    if norm_config is not None:
        m[3] = normalize(to_float(prb_used_dl), norm_config.prb_used_min, norm_config.prb_used_max)
        m[4] = normalize(to_float(prb_used_ul), norm_config.prb_used_min, norm_config.prb_used_max)
        m[5] = normalize(to_float(prb_use_perc_dl), norm_config.prb_use_perc_min, norm_config.prb_use_perc_max)
        m[6] = normalize(to_float(prb_use_perc_ul), norm_config.prb_use_perc_min, norm_config.prb_use_perc_max)
        m[7] = normalize(to_float(thp_dl), norm_config.thp_min, norm_config.thp_max)
    else:
        # No normalization (legacy behavior)
        m[3] = to_float(prb_used_dl)
        m[4] = to_float(prb_used_ul)
        m[5] = to_float(prb_use_perc_dl)
        m[6] = to_float(prb_use_perc_ul)
        m[7] = to_float(thp_dl)

    return m


def msg_from_neighbor_cell(
    neigh_pci, neigh_rsrp, neigh_rsrq,
    norm_config: Optional[NormalizationConfig] = None,
) -> List[float]:
    """Create message vector for neighbor cell (potential handover target)."""
    m = [0.0] * MSG_DIM
    # Event type one-hot: [1, 0, 0] for neighbor cell (similar to CU, but only neighbor info)
    m[0] = 1.0

    if norm_config is not None:
        # Neighbor cell signal quality (stored in neighbor positions)
        m[6] = normalize(to_float(neigh_pci), norm_config.pci_min, norm_config.pci_max)
        m[7] = normalize(to_float(neigh_rsrp), norm_config.rsrp_min, norm_config.rsrp_max)
        m[8] = normalize(to_float(neigh_rsrq), norm_config.rsrq_min, norm_config.rsrq_max)
    else:
        # No normalization (legacy behavior)
        m[6] = to_float(neigh_pci)
        m[7] = to_float(neigh_rsrp)
        m[8] = to_float(neigh_rsrq)

    return m
