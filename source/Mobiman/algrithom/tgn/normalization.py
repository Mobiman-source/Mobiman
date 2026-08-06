from dataclasses import dataclass


@dataclass
class NormalizationConfig:
    """Define min/max ranges for each feature type based on 5G standards."""
    # CU measurements (in dBm/dB)
    rsrp_min: float = -140.0
    rsrp_max: float = -40.0
    rsrq_min: float = -20.0
    rsrq_max: float = -3.0
    sinr_min: float = -10.0
    sinr_max: float = 30.0
    pci_min: float = 0.0
    pci_max: float = 1007.0  # Max PCI value in 5G

    # DU UE KPM (throughput in Mbps, delay in ms)
    thp_min: float = 0.0
    thp_max: float = 1000.0  # 1 Gbps
    rlc_delay_min: float = 0.0
    rlc_delay_max: float = 100.0  # 100ms
    rlc_pkt_drop_min: float = 0.0
    rlc_pkt_drop_max: float = 100.0  # percentage
    prb_used_min: float = 0.0
    prb_used_max: float = 273.0  # Max PRBs in 5G NR (100MHz BW)
    vol_min: float = 0.0
    vol_max: float = 1_000_000.0  # Bytes/KB/MB scale varies; tune to your data
    air_if_delay_min: float = 0.0
    air_if_delay_max: float = 100.0  # ms
    cqi_min: float = 0.0
    cqi_max: float = 15.0  # CQI range

    # DU Cell KPM
    prb_use_perc_min: float = 0.0
    prb_use_perc_max: float = 100.0  # percentage


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1] based on min and max bounds."""
    if max_val <= min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))  # Clip to [0, 1]
