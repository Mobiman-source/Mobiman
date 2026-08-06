# Mobiman

**Mobiman: Mobility Management for Delay-Critical Edge AI Offloading in 5G Open RAN**

Mobiman is a 5G Open RAN (O-RAN) system that optimizes user mobility management for delay-critical edge AI offloading. It builds a temporal graph model of the RAN across users and cells to drive near real-time handover decisions, and pairs it with a multi-agent reinforcement learning (MARL) policy — with rule-based action masking and proactive resource preparation — to make handovers safe, stable, and efficient. Mobiman is evaluated on a multi-cell indoor 5G O-RAN testbed with diverse VR workloads.

This repository contains the project webpage and the reference implementation used in the testbed.

## Repository Structure

```
.
├── index.html            # Project webpage (paper landing page)
├── assets/               # Figures and demo videos used on the webpage
├── static/                # CSS/JS for the webpage
└── source/
    ├── Mobiman/            # Near-RT RIC xApps + the TGN-MAPPO training/inference pipeline
    ├── gnb/                # gNB-side monitoring scripts and F1AP/CU-CP scaffolding
    ├── core/               # Example 5G core subscriber database
    └── data_example/       # Example collected RAN/traffic data
```

### `source/Mobiman/`

Near-RT RIC xApps and the RL pipeline behind Mobiman's handover decisions.

```
Mobiman/
├── Mobiman_xapp.py     # Main handover xApp: streams RAN telemetry into the shared DB and issues handover commands
├── rule_ho_xapp.py     # Rule/threshold-based handover xApp, used as a baseline
├── monitor.py          # Real-time Dash dashboard (reads data/ran.db, serves http://localhost:8050)
├── ue_map.py           # Pairs UE identifiers (SUCI / RAN-UE-ID / AMF-UE-ID) from AMF and CU logs
└── algrithom/           # TGN + MAPPO training and inference code
    ├── tgn.py, tgn/       # Temporal Graph Network (TGN) streaming embeddings
    ├── mappo/             # Multi-agent PPO policy (actor-critic, action masking, utils)
    ├── runner/            # train_sc.py / sc_runner.py — training entry point
    ├── env.py             # TGNUEEnv — Gym environment wrapping the TGN embeddings and RAN state
    ├── config.py          # CLI argument parser (get_config)
    └── tran_sc.sh          # Example training launch script
```

Run training:

```bash
cd source/Mobiman/algrithom
./tran_sc.sh
```

Run inference:

```bash
python source/Mobiman/algrithom/tgn.py --mode inference --db /path/to/your/ran.db --num_du 3
```

See `python runner/train_sc.py --help` / `python tgn.py --help` for the full list of options.

### `source/gnb/`

Monitoring scripts that run alongside the gNB/CU, plus scaffolding for the F1AP CU-CP procedures and UE monitor library.

```
gnb/
├── docs/
│   ├── amf_suci_sync.py    # Tails the AMF log and syncs SUCI suffixes into the UE monitor DB
│   ├── iperf_monitor.py    # Generates UL/DL iperf traffic and measures loaded RTT
│   └── ping_monitor.py     # Records per-ping RTT and cumulative packet loss
└── lib/
    ├── f1ap/cu_cp/procedures/   # F1AP CU-CP procedure implementations
    └── ue_monitor/               # UE monitor library
```

### `source/core/` and `source/data_example/`

Example 5G core subscriber configuration and sample RAN/traffic data collected from the testbed, provided for reference.

## Requirements

```
torch
torch-geometric
gym
numpy
scikit-learn
dash
plotly
```

Optional, for training: `tensorboardX`, `wandb`, `setproctitle`.

## Citation

Citation details will be added upon publication.
